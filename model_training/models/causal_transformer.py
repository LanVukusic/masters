import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norm self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-norm feed-forward
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_act = nn.GELU()
        self.ffn_inner_dropout = nn.Dropout(dropout)
        self.ffn_linear2 = nn.Linear(d_ff, d_model)
        self.ffn_out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        residual = x
        x = self.norm1(x)
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v}

        if T > 1:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn = self.out_proj(attn)
        attn = self.attn_dropout(attn)
        x = residual + attn

        residual = x
        x = self.norm2(x)
        x = self.ffn_linear1(x)
        x = self.ffn_act(x)
        x = self.ffn_inner_dropout(x)
        x = self.ffn_linear2(x)
        x = self.ffn_out_dropout(x)
        x = residual + x

        return x, new_cache


class CausalTransformer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CausalTransformerLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, kv_caches: list[dict] | None = None):
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, kv_cache=cache)
            new_caches.append(new_cache)
        return x, new_caches
