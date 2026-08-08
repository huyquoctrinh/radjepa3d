"""Mamba branch: fine patches, bidirectional SSM with Morton scan."""

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba

from .stems import PatchEmbed3D
from .serialize import scan_order


class BiMamba(nn.Module):

    def __init__(self, dim, bidirectional=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.bidirectional = bidirectional
        self.fwd = Mamba(d_model=dim)
        self.bwd = Mamba(d_model=dim) if bidirectional else None

    def forward(self, x):
        y = self.norm(x)
        out = x + self.fwd(y)
        if self.bwd is not None:
            out = out + self.bwd(y.flip(1)).flip(1)
        return out


class MambaBranch(nn.Module):

    def __init__(self, in_chans, dim, patch_m, depth_m, drop, n_tokens_m,
                 bidirectional=True, use_morton=True):
        super().__init__()
        self.embed = PatchEmbed3D(in_chans, dim, patch_m)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens_m, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.use_morton = use_morton
        self.blocks = nn.ModuleList([
            BiMamba(dim, bidirectional) for _ in range(depth_m)
        ])
        self.norm = nn.LayerNorm(dim)

        self._cached_order = None
        self._cached_grid = None

    def _get_order(self, grid, device):
        if self._cached_grid != grid:
            self._cached_order = scan_order(*grid, use_morton=self.use_morton, device=device)
            self._cached_grid = grid
        return self._cached_order.to(device)

    def encode_seq(self, seq):
        """Encode a pre-serialized sequence (for Phase 2)."""
        for block in self.blocks:
            seq = block(seq)
        return self.norm(seq)

    def forward(self, vol):
        tokens, grid = self.embed(vol)
        tokens = tokens + self.pos_embed

        order = self._get_order(grid, tokens.device)
        tokens = tokens[:, order]

        tokens = self.encode_seq(tokens)
        return tokens.mean(dim=1)
