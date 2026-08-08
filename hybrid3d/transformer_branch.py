"""Transformer branch: coarse patches, global attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stems import PatchEmbed3D


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBranch(nn.Module):

    def __init__(self, in_chans, dim, patch_t, depth_t, heads, mlp_ratio, drop, pool, n_tokens_t):
        super().__init__()
        self.embed = PatchEmbed3D(in_chans, dim, patch_t)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens_t, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, drop) for _ in range(depth_t)
        ])
        self.norm = nn.LayerNorm(dim)
        self.pool = pool

    def encode_tokens(self, tokens):
        """Encode pre-embedded tokens (for Phase 2 — may be a subset)."""
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, vol):
        tokens, grid = self.embed(vol)
        tokens = tokens + self.pos_embed
        tokens = self.encode_tokens(tokens)
        return tokens.mean(dim=1)
