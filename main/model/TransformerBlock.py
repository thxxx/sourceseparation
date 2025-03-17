import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import LayerNorm
from .Attn import Attention
from .FeedForaward import FeedForward

class LayerScale(nn.Module):
    def __init__(self, dim, init_val = 1e-2):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            ada_cond_dim = None,
            zero_init_branch_outputs = True,
            remove_norms = False,
            long_skip=False,
            long_skip_norm=False,
            layer_scale=False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {} # 근데 3개 모두 들어오는 값 없다. 확작성을 위해 둔 것 뿐. kwargs를 넣은건 대부분 혹시 모를 상황때문이라고 생각하자
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.long_skip = long_skip
        self.ada_cond_dim = ada_cond_dim

        self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )
        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context=dim_context,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
        
        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
        if long_skip:
            self.skip_norm = nn.LayerNorm(2 * dim) if long_skip_norm else nn.Identity()
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None
        
        # self.rope = RotaryEmbedding(max(dim_heads // 2, 32)) if add_rope else None

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        skip = None,
        ada_vals=None
    ):
        B, C, T = x.shape
        # if self.skip_linear is not None and skip is not None:
        #     cat = torch.cat([x, skip], dim=-1) # in channel dimension
        #     cat = self.skip_norm(cat)
        #     x = self.skip_linear(cat)
        
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])
        
        # below is utilize global_cond with adaLN
        if self.ada_cond_dim and self.ada_cond_dim > 0 and ada_vals is not None:
            # print(ada_vals)
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = ada_vals

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual

            # cross-attention
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual
        else:
            # x에 이미 concat되어서 정보가 prepend 되어있기 때문에 추가 연산이 없을 뿐, timestep, duration 정보가 적용되어 있다.
            x = x + self.self_attn(self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb)
            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
            x = x + self.ff(self.ff_norm(x))

        return x