from torch import nn
import torch
from .TransformerBlock import TransformerBlock
from .utils import checkpoint
from .positional_encoding import RotaryEmbedding

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=True,
        cond_embed_dim=None,
        ada_cond_dim=None,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_skip=False,
        use_skip_norm=False,
        **kwargs
        ):

        super().__init__()

        self.d_model = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.use_skip = use_skip

        self.project_in = nn.Linear(dim_in, self.d_model, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(self.d_model, dim_out, bias=False) if dim_out is not None else nn.Identity()

        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb is not None else None
        
        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    self.d_model,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_embed_dim,
                    ada_cond_dim = ada_cond_dim,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    long_skip=use_skip and i == depth-1,
                    long_skip_norm=use_skip_norm,
                    **kwargs
                )
            )
        
        self.ada_cond_dim = ada_cond_dim
        if ada_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ada_cond_dim, dim * 6, bias=False)
            )
            nn.init.zeros_(self.to_scale_shift_gate[1].weight)
        
    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        context = None,
        context_mask = None,
        ada_cond_embed = None,
        **kwargs
    ):
        batch, seq, device = x.shape[0], x.shape[1], x.device
        
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        
        x = self.project_in(x)

        if prepend_embeds is not None:
            bs, prepend_length, prepend_dim = prepend_embeds.shape
            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'
            x = torch.cat((prepend_embeds, x), dim = -2) # 여기서 드디어 prepend.
            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)
                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers 
        rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1]) if self.rotary_pos_emb else None

        if ada_cond_embed is not None and self.to_scale_shift_gate:
            ada_vals = self.to_scale_shift_gate(ada_cond_embed).unsqueeze(1).chunk(6, dim=-1)
        else:
            ada_vals = None

        # print(f"\nTransformer inner : {x}\n")
        # print(f"\n rotary_pos_emb inner : {rotary_pos_emb}\n")
        # print(f"\n global_cond : {global_cond}\n")
        # print(f"\n context : {context}\n")
        # print(f"\n context_mask : {context_mask}\n")
        
        # Iterate over the transformer layers
        for idx, layer in enumerate(self.layers):
            # x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs) # 을 해도 되는데, 아래가 효율적임.
            # context_mask를 넣으면 안된다.
            x = checkpoint(layer, x, context=context, context_mask=None, rotary_pos_emb=rotary_pos_emb)

        # print(f"\nTransformer outer : {x}, {x.shape}\n")
        x = self.project_out(x)

        return x
