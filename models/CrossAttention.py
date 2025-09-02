import torch
from einops import rearrange
from torch import nn


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CTAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        x12 = torch.chunk(x, chunks=2, dim=0)
        x1 = x12[0]
        x2 = x12[1]

        q = self.to_q(x2)
        k = self.to_k(x1)
        v = self.to_v(x1)
        qkv = []
        qkv.append(q)
        qkv.append(k)
        qkv.append(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out

class CT_Transformer(nn.Module):
    def __init__(self, h_dim, depth, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                LayerNormalize(h_dim, CTAttention(h_dim, heads=heads, dropout=dropout))
            )

    def forward(self, h_tokens):

        for h_attend_lg in self.layers:
            h_tokens = h_attend_lg(h_tokens)

        return h_tokens


class FusionEncoder(nn.Module):
    def __init__(self, depth, h_dim, ct_attn_heads, ct_attn_depth, dropout=0.1, patchsize=13):
        super().__init__()
        self.pojo = nn.Conv1d(16, patchsize**2, kernel_size=1, stride=1)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                CT_Transformer(h_dim=h_dim, depth=ct_attn_depth, heads=ct_attn_heads, dropout=dropout)
            )

    def forward(self, h_tokens, l_tokens):
        l_tokens = self.pojo(l_tokens)
        h_tokens = torch.concat((h_tokens, l_tokens), dim=0)

        for cross_attend in self.layers:
            h_tokens = cross_attend(h_tokens)

        return h_tokens
