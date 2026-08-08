"""Hybrid Mamba-Transformer blocks with per-token routing for V-JEPA."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class RotaryEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for (depth, height, width) grid."""

    def __init__(self, head_dim, grid_size=(4, 16, 16)):
        super().__init__()
        self.grid_size = grid_size
        gd, gh, gw = grid_size
        n_patches = gd * gh * gw

        # Split head_dim into 3 parts for d/h/w
        dim_d = head_dim // 3
        dim_h = head_dim // 3
        dim_w = head_dim - dim_d - dim_h

        # Precompute position indices
        pos = torch.arange(n_patches)
        d_pos = pos // (gh * gw)
        h_pos = (pos % (gh * gw)) // gw
        w_pos = pos % gw

        # Compute freqs for each dimension
        freqs_d = self._compute_freqs(d_pos.float(), dim_d)
        freqs_h = self._compute_freqs(h_pos.float(), dim_h)
        freqs_w = self._compute_freqs(w_pos.float(), dim_w)

        # Concat: (n_patches, head_dim)
        cos_embed = torch.cat([freqs_d[0], freqs_h[0], freqs_w[0]], dim=-1)
        sin_embed = torch.cat([freqs_d[1], freqs_h[1], freqs_w[1]], dim=-1)

        self.register_buffer("cos_embed", cos_embed)
        self.register_buffer("sin_embed", sin_embed)

    @staticmethod
    def _compute_freqs(positions, dim):
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cos(angles).repeat(1, 2)[:, :dim], torch.sin(angles).repeat(1, 2)[:, :dim]

    def forward(self, x):
        """Apply RoPE. x: (B, n_heads, L, head_dim)"""
        L = x.shape[2]
        cos = self.cos_embed[:L].unsqueeze(0).unsqueeze(0)
        sin = self.sin_embed[:L].unsqueeze(0).unsqueeze(0)

        # Rotate pairs
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rot = x_rot.reshape(x.shape)
        return x * cos + x_rot * sin


class LightweightGQA(nn.Module):
    """Lightweight Group-Query Attention with 3D RoPE."""

    def __init__(self, dim, n_heads=8, n_kv_heads=2, grid_size=(4, 16, 16), use_rope=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_group_size = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryEmbedding3D(self.head_dim, grid_size) if use_rope else None

    def forward(self, x):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, -1)

        return self.out_proj(attn)


def compute_balance_loss(probs):
    """MoE-style load-balancing loss."""
    N = probs.shape[-1]
    assignments = probs.argmax(dim=-1)
    f = torch.stack([(assignments == i).float().mean() for i in range(N)])
    P = probs.mean(dim=(0, 1))
    return N * (f * P).sum()


class HybridBlock(nn.Module):
    """Hybrid Mamba-Transformer block with per-token routing."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2,
                 n_attn_heads=8, n_kv_heads=2, grid_size=(4, 16, 16),
                 use_rope=True, router_mode="token"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.router_mode = router_mode

        # Mamba branch
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # Transformer branch
        self.gqa = LightweightGQA(dim, n_attn_heads, n_kv_heads, grid_size, use_rope)

        # Router (not needed for the fixed 50/50 baseline)
        if router_mode == "fixed":
            self.router = None
        else:
            self.router = nn.Linear(dim, 2, bias=False)

        self._balance_loss = None

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Both branches process full sequence
        mamba_out = self.mamba(x)
        trans_out = self.gqa(x)

        if self.router_mode == "fixed":
            # Non-learned 50/50 average — control to test whether routing helps
            out = 0.5 * mamba_out + 0.5 * trans_out
            self._balance_loss = None
        elif self.router_mode == "layer":
            # Per-layer: pool all tokens, produce single gate for entire layer
            pooled = x.mean(dim=1)  # (B, dim)
            router_logits = self.router(pooled)  # (B, 2)
            probs = F.softmax(router_logits, dim=-1)  # (B, 2)
            w_mamba = probs[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            w_trans = probs[:, 1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            out = w_mamba * mamba_out + w_trans * trans_out
            self._balance_loss = self._layer_balance_loss(probs)
        else:
            # Per-token routing
            router_logits = self.router(x)
            probs = F.softmax(router_logits, dim=-1)  # (B, L, 2)
            top_val, top_idx = probs.max(dim=-1)
            mamba_mask = (top_idx == 0).unsqueeze(-1).float()
            trans_mask = (top_idx == 1).unsqueeze(-1).float()
            weights = top_val.unsqueeze(-1)
            out = mamba_mask * weights * mamba_out + trans_mask * weights * trans_out
            self._balance_loss = compute_balance_loss(probs)

        return residual + out

    @staticmethod
    def _layer_balance_loss(probs):
        # probs: (B, 2) — encourage ~50/50 split across batch
        mean_probs = probs.mean(dim=0)  # (2,)
        return 2.0 * (mean_probs * mean_probs).sum()
