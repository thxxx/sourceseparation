import torch
import torch.nn as nn
from model.predictmodel import ContinuousTransformer
from config import config
from condition.time_condition_model import NumberConditioner
from model.predictmodel import ContinuousTransformer
from condition.timestep import FourierFeatures
from einops import rearrange
import random

class DiffusionTransformer(nn.Module):
    """
    role : 
    1) make condition embeddings to usable context for model - concat, dimension change
    2) add timestep scalar value to global condition as embedding
    
    """
    def __init__(
            self, 
            d_model, 
            depth, 
            num_heads, 
            latent_channels, 
            config, 
            device, 
            global_cond_type='prepend', 
            input_concat_dim=0, 
            use_skip=False, 
            ada_cond_dim=None,
            context_dim=None
        ):
        super().__init__()
        self.global_cond_type = global_cond_type
        dim_in = latent_channels
        dim_per_head = d_model // num_heads

        default_config = config['model']
        default_config['dim'] = d_model
        default_config['depth'] = depth
        default_config['dim_heads'] = dim_per_head
        default_config['dim_in'] = latent_channels
        default_config['dim_out'] = latent_channels
        default_config['use_skip'] = use_skip
        default_config['ada_cond_dim'] = ada_cond_dim
        default_config['use_skip_norm'] = False
        default_config['attn_kwargs'] = {
            "qk_norm": None
        }
        self.model = ContinuousTransformer(**default_config)

        self.ada_cond_dim = ada_cond_dim
        self.latent_channels = latent_channels
        
        self.timing_start_conditioner = NumberConditioner(**config['timing_config'])
        self.timing_total_conditioner = NumberConditioner(**config['timing_config'])

        self.cross_attention_inputs_keys = config['cross_attn_cond_keys']
        self.global_cond_keys = config['global_cond_keys']

        self.cond_token_dim = config['cond_token_dim']
        self.global_cond_dim = config['global_cond_dim']
        
        # 차원을 맞추기 위한 용도로 정의되는 layer인 것 같긴하지만, input-output dim이 같더라도 매번 통과한다. 한번 더 태우는거인듯
        if self.cond_token_dim > 0:
            cond_embed_dim = self.cond_token_dim
            self.to_cond_embed = nn.Sequential( # 1
                nn.Linear(self.cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            ).to(device)
        else:
            cond_embed_dim = 0
        
        if self.global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = d_model # 어짜피 같다.
            self.to_global_embed = nn.Sequential( # 2
                nn.Linear(self.global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            ).to(device)

        if context_dim is not None:
            self.to_context_dim = nn.Sequential(
                nn.Linear(context_dim, latent_channels, bias=False),
                nn.SiLU(),
                nn.Linear(latent_channels, latent_channels, bias=False)
            ).to(device)
            nn.init.zeros_(self.to_context_dim[-1].weight)
            self.channel_wise_proj = nn.Sequential(
                nn.Linear(latent_channels*2, latent_channels*4, bias=False),
                nn.SiLU(),
                nn.Linear(latent_channels*4, latent_channels, bias=False)
            ).to(device)
        
        timestep_features_dim = 256
        self.fourier = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential( # 3
            nn.Linear(timestep_features_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
            
        self.preprocess_conv = nn.Conv1d(latent_channels, latent_channels, 1, bias=False) # 4
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(latent_channels, latent_channels, 1, bias=False) # 5
        nn.init.zeros_(self.postprocess_conv.weight)

    def get_context(self, input_ids, attention_mask, seconds_start, seconds_total):
        model_dtype = torch.float32
        
        start_emb, start_mask = self.timing_start_conditioner(seconds_start, device=input_ids.device) # start_mask is just 1
        total_emb, total_mask = self.timing_total_conditioner(seconds_total, device=input_ids.device)

        conditioning_tensors = {
            'prompt': (input_ids.type(model_dtype), attention_mask), # (batch, seq(max_len), channels)
            'seconds_start': (start_emb.type(model_dtype), start_mask), # (batch, seq=1, channels)
            'seconds_total': (total_emb.type(model_dtype), total_mask) # (batch, seq=1, channels)
        }

        # 기존 stable audio의 코드 : conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
        if len(self.cross_attention_inputs_keys) > 0:
            cross_attention_inputs = []
            cross_attention_masks = []

            for key in self.cross_attention_inputs_keys:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]
                
                cross_attention_inputs.append(cross_attn_in.to(torch.float32))
                cross_attention_masks.append(cross_attn_mask.to(torch.float32))
            # print("cross_attention_inputs : ", cross_attention_inputs)
            cross_attention_inputs = torch.cat([x.to(torch.float32) for x in cross_attention_inputs], dim=1)
            # cross_attention_inputs = torch.cat(cross_attention_inputs, dim=1, d) # dim=1 is sequence. So, total sequence length is text seq 128 + 1 + 1
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_keys) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_keys:
                global_cond_input = conditioning_tensors[key][0] # 0 is only emb, not mask
                if len(global_cond_input.shape) == 3:
                    global_cond_input = global_cond_input.squeeze(1) # remove sequence(which has value 1)
                global_conds.append(global_cond_input)
            # Concatenate over the channel dimension 얘는 또 채널 dimension에서 하네. 그래서 768의 두배, 1536
            global_cond = torch.cat(global_conds, dim=-1)
        
        return [cross_attention_inputs, cross_attention_masks.to(bool), global_cond]

    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attention_inputs=None,
        cross_attention_masks=None,
        global_cond=None,
        prefix_cond=None,
        prepend_inputs=None,
        prepend_mask=None,
    ):
        if self.ada_cond_dim:
            ada_cond_embed = (cross_attention_inputs.sum(dim=1, keepdim=True)/cross_attention_masks.sum(dim=1, keepdim=True).unsqueeze(dim=1)).squeeze()
        else:
            ada_cond_embed = None
        
        if self.to_cond_embed:
            cross_attention_inputs = self.to_cond_embed(cross_attention_inputs) # bs, 128+1+1, 768
        if self.to_global_embed:
            global_cond = self.to_global_embed(global_cond)

        time_feat = self.fourier(t[:, None])
        timestep_embed = self.to_timestep_embed(time_feat).squeeze()
        
        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        if global_cond is not None:
            global_cond = global_cond + timestep_embed # 더해버리네. timing 정보에다가
        else:
            global_cond = timestep_embed

        if self.global_cond_type == 'prepend':
            # Prepend inputs are just the global embed, and the mask is all ones
            if len(global_cond.shape) == 2:
                prepend_inputs = global_cond.unsqueeze(1) # bs, seq_len(=1), channels
            prepend_mask = torch.ones((x.shape[0], prepend_inputs.shape[1]), device=x.device, dtype=torch.bool)
        
        extra_args = {}
        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_cond

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b t c -> b c t")

        output = self.model(
            x, 
            mask=None, # None이다.
            context=cross_attention_inputs, 
            context_mask=cross_attention_masks, # 텍스트 길이만큼 1, 그다음 쭉 0, 마지막 두개 1, 1
            prepend_embeds=prepend_inputs, 
            prepend_mask=prepend_mask, # 그냥 bs, 1
            ada_cond_embed=ada_cond_embed
        )
        
        prepend_length = prepend_inputs.shape[1]
        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]
        output = self.postprocess_conv(output) + output

        return output

    def forward(
            self, x, t, mask, input_ids, attention_mask, seconds_start, seconds_total, prefix_cond=None, 
            cfg_dropout_prob=0.1, 
            cfg_scale=None, prepend_inputs=None, scale_phi=0.75, 
            audio_context=None):
        cross_attention_inputs, cross_attention_masks, global_cond = self.get_context(input_ids, attention_mask, seconds_start, seconds_total)

        if audio_context is not None:
            if random.random() > cfg_dropout_prob:
                BS, CH, SL = audio_context.shape
            else:
                audio_context = torch.zeros_like(audio_context)

            audio_context = rearrange(audio_context, "b d n -> b n d")
            audio_context = self.to_context_dim(audio_context)
            audio_context = rearrange(audio_context, "b n d -> b d n")
            
            x = torch.concat((audio_context, x), dim=1) # channel-wise
            x = rearrange(x, "b d n -> b n d")
            x = self.channel_wise_proj(x) # latent_chn * 2 -> latent_chn
            x = rearrange(x, "b n d -> b d n")
        
        if cfg_dropout_prob > 0:
            
            if cross_attention_inputs is not None:
                null_embed = torch.zeros_like(cross_attention_inputs, device=cross_attention_inputs.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attention_inputs.shape[0], 1, 1), cfg_dropout_prob, device=cross_attention_inputs.device)).to(torch.bool)
                cross_attention_inputs = torch.where(dropout_mask, null_embed, cross_attention_inputs)
            
            if prepend_inputs is not None:
                null_embed = torch.zeros_like(prepend_inputs, device=prepend_inputs.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_inputs.shape[0], 1, 1), cfg_dropout_prob, device=prepend_inputs.device)).to(torch.bool)
                prepend_inputs = torch.where(dropout_mask, null_embed, prepend_inputs)
        
        if cfg_dropout_prob == 0.0 and cfg_scale is not None:
            # if audio_context is not None:
            #     audio_context = rearrange(audio_context, "b d n -> b n d")
            #     audio_context = self.to_context_dim(audio_context)
            #     audio_context = rearrange(audio_context, "b n d -> b d n")
                
            #     x_stacked = torch.concat((audio_context, x), dim=1) # channel-wise
            #     x_stacked = rearrange(x_stacked, "b d n -> b n d")
            #     x_stacked = self.channel_wise_proj(x_stacked) # latent_chn * 2 -> latent_chn
            #     x_stacked = rearrange(x_stacked, "b n d -> b d n")
            
            # batch_inputs = torch.cat([x_stacked, x], dim=0)
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)
            
            if global_cond is not None:
                batch_global_cond = torch.cat([global_cond, global_cond], dim=0)
            else:
                batch_global_cond = None
            
            if prefix_cond is not None:
                batch_prefix_cond = torch.cat([prefix_cond, prefix_cond], dim=0)
            else:
                batch_prefix_cond = None

            batch_cond = None
            batch_cond_masks = None

            # Handle CFG for cross-attention conditioning
            if cross_attention_inputs is not None:
                null_embed = torch.zeros_like(cross_attention_inputs, device=cross_attention_inputs.device)
                batch_cond = torch.cat([cross_attention_inputs, null_embed], dim=0)

                if cross_attention_masks is not None:
                    batch_cond_masks = torch.cat([cross_attention_masks, cross_attention_masks], dim=0)
               
            batch_prepend_inputs = None
            batch_prepend_inputs_mask = None

            if prepend_inputs is not None:
                null_embed = torch.zeros_like(prepend_inputs, device=prepend_inputs.device)
                batch_prepend_inputs = torch.cat([prepend_inputs, null_embed], dim=0)

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            batch_output = self._forward(
                x=batch_inputs,
                t=batch_timestep,
                mask=batch_masks,
                cross_attention_inputs=batch_cond, 
                cross_attention_masks=batch_cond_masks,
                global_cond=batch_global_cond,
                prefix_cond=batch_prefix_cond,
                prepend_inputs=batch_prepend_inputs, 
                prepend_mask=batch_prepend_inputs_mask,
            )
            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale
            
            # # CFG Rescale
            # if scale_phi != 0.0:
            #     cond_out_std = cond_output.std(dim=1, keepdim=True)
            #     out_cfg_std = cfg_output.std(dim=1, keepdim=True)
            #     output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            # else:
            #     output = cfg_output

            output = cfg_output
        else:
            output = self._forward(
                x, 
                t,
                mask=None,
                cross_attention_inputs=cross_attention_inputs, 
                cross_attention_masks=cross_attention_masks,
                global_cond=global_cond,
                prefix_cond=prefix_cond,
                prepend_inputs=prepend_inputs, 
                prepend_mask=None,
            )
        
        return output