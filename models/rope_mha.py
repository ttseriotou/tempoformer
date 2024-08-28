#sourced from llama and removed norms and ffn, blog: https://github.com/pytorch/pytorch/issues/97899
#adaptation of key_padding_mask: https://github.com/pytorch/pytorch/blob/778006918c31c3fa0ca3794575a65c1f854f861b/torch/nn/functional.py#L4297
#core code for scaled_dot_product_attention : https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

#sources to understand RoPE:
#primary source: https://github.com/lucidrains/PaLM-rlhf-pytorch/blob/main/palm_rlhf_pytorch/palm.py
#cross-reference source: https://nn.labml.ai/transformers/rope/index.html

from typing import Tuple
import torch.nn as nn
import torch
import math

# Taken from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert (freqs_cis.shape[-2], freqs_cis.shape[-1]) == (x.shape[1], x.shape[-1])
    if (freqs_cis.ndim == 3):
        shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, device: str, dt: list[float]= None, theta: float = 10000.0):
    freqs = (1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim) 
    )).to(device)
    if dt == None:
        dt = torch.arange(end, device=device) 
        freqs = torch.outer(dt, freqs).float()
    else:
        freqs = torch.einsum('bk,j->bkj', dt, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64   

    return freqs_cis

def scaled_dot_product_attention(query, key, value, key_padding_mask=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, device='cpu') -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    bsz, num_heads, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == S

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    if key_padding_mask is not None:
        attn_weight = attn_weight.view(bsz, num_heads, L, S)
        attn_weight = attn_weight.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
    
    
class MHARoPE(nn.Module):
    """Transformer encoder block using F.scaled_dot_product_attention().

    This block has the following changes from a typical transformer encoder:

        - Rotary embeddings are applied to the key/query matrices.
        - Keys arising from padding are masked during attention.
    """

    def __init__(self,
        embed_dim, 
        num_heads, 
        dropout,
        max_seq_len):

        super().__init__()

        self.embed_dim = embed_dim
        self.drop_p = dropout
        self.drop_ffn = 0.1
        self.n_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.max_seq_len = max_seq_len

        # Attention
        self.q = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False,
        )
        self.k = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False,
        )
        self.v = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
        )

        self.resid_dropout = nn.Dropout(self.drop_ffn)

        # FF Layer
        self.ff_dropout = nn.Dropout(self.drop_ffn)
        self.ff_linear_1 = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim * 4,
        )
        self.ff_linear_2 = nn.Linear(
            in_features=embed_dim * 4,
            out_features=embed_dim,
        )
        self.ff_activation = nn.GELU()

        # Pre layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor, dt: torch.Tensor
    ):
        self.device = x.device
        x = self._att_block(x, key_padding_mask, dt)
        #x = x + self._ff_block(self.norm2(x))
        return x

    def _att_block(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor, dt: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)
        freqs_cis = precompute_freqs_cis(self.d_head, seq_len, device=self.device, dt=dt)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = self.drop_p
        else:
            att_dropout = 0.0

        # Using beta torch functionality (subject to change)
        att = scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
            dropout_p=att_dropout,
            is_causal=False,
            device =self.device
        )

        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

    def _ff_block(self, x: torch.Tensor):
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))

        return self.ff_dropout(x)