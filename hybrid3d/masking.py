"""Shared volume-space masking for hybrid SSL.

One mask sampled in canonical space, projected to each branch's grid.
Transformer: DROP masked tokens. Mamba: REPLACE with mask_token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stems import PatchEmbed3D
from .serialize import scan_order


def region_mask(canon, mask_ratio, device):
    """Sample a block mask in canonical volume space.

    Returns BoolTensor (cd, ch, cw) where True = VISIBLE.
    """
    cd, ch, cw = canon
    keep = torch.rand(cd, ch, cw, device=device) > mask_ratio
    if keep.sum() == 0:
        keep[0, 0, 0] = True
    return keep


def project_to_grid(region_keep, grid):
    """Project canonical keep-mask to a branch's patch grid via nearest interp.

    region_keep: (cd, ch, cw) bool
    grid: (gd, gh, gw)
    Returns: (N,) bool where N = gd*gh*gw
    """
    gd, gh, gw = grid
    mask_5d = region_keep.float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(mask_5d, size=(gd, gh, gw), mode="nearest")
    return upsampled.squeeze(0).squeeze(0).flatten().bool()


class HybridMaskedInput(nn.Module):
    """Prepare masked inputs for both branches from a shared volume mask."""

    def __init__(self, cfg):
        super().__init__()
        self.canon = cfg.mask_canon
        self.mask_ratio = cfg.mask_ratio
        self.mask_mode = getattr(cfg, "mask_mode", "shared")
        self.use_morton = getattr(cfg, "use_morton", True)

        self.embed_t = PatchEmbed3D(cfg.in_chans, cfg.dim, cfg.patch_t)
        self.embed_m = PatchEmbed3D(cfg.in_chans, cfg.dim, cfg.patch_m)

        self.pos_t = nn.Parameter(torch.zeros(1, cfg.n_tokens_t, cfg.dim))
        self.pos_m = nn.Parameter(torch.zeros(1, cfg.n_tokens_m, cfg.dim))
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_m, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, vol):
        device = vol.device
        region_keep = region_mask(self.canon, self.mask_ratio, device)
        # Independent mode: the Mamba branch gets its own separately sampled mask
        region_keep_m = (region_mask(self.canon, self.mask_ratio, device)
                         if self.mask_mode == "independent" else region_keep)

        # Transformer: embed, add pos, DROP masked
        tok_t, grid_t = self.embed_t(vol)
        tok_t = tok_t + self.pos_t
        keep_t = project_to_grid(region_keep, grid_t)
        vis_t = tok_t[:, keep_t]

        # Mamba: embed, add pos, REPLACE masked with mask_token, serialize
        tok_m, grid_m = self.embed_m(vol)
        tok_m = tok_m + self.pos_m
        keep_m = project_to_grid(region_keep_m, grid_m)

        mask_expanded = self.mask_token.expand(tok_m.shape[0], tok_m.shape[1], -1)
        pos_expanded = self.pos_m.expand_as(tok_m)
        tok_m = torch.where(
            keep_m.unsqueeze(0).unsqueeze(-1).expand_as(tok_m),
            tok_m,
            mask_expanded + pos_expanded,
        )

        order = scan_order(*grid_m, use_morton=self.use_morton, device=device)
        seq_m = tok_m[:, order]
        keep_m_s = keep_m[order]

        return {
            "region_keep": region_keep,
            "vis_t": vis_t,
            "keep_t": keep_t,
            "seq_m": seq_m,
            "keep_m_s": keep_m_s,
            "order": order,
            "grid_t": grid_t,
            "grid_m": grid_m,
        }
