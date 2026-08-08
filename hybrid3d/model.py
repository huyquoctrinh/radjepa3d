"""Phase 1: HybridLateFusion encoder — two branches, late concat fusion."""

import torch
import torch.nn as nn

from .config import Hybrid3DConfig
from .transformer_branch import TransformerBranch
from .mamba_branch import MambaBranch
from .fusion import build_fusion


class HybridLateFusion(nn.Module):

    def __init__(self, cfg: Hybrid3DConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = TransformerBranch(
            cfg.in_chans, cfg.dim, cfg.patch_t, cfg.depth_t,
            cfg.heads, cfg.mlp_ratio, cfg.drop, cfg.pool, cfg.n_tokens_t,
        )
        self.mamba = MambaBranch(
            cfg.in_chans, cfg.dim, cfg.patch_m, cfg.depth_m,
            cfg.drop, cfg.n_tokens_m,
            bidirectional=cfg.mamba_bidirectional, use_morton=cfg.use_morton,
        )
        self.head = build_fusion(cfg.fusion, cfg.dim, cfg.num_classes,
                                 cfg.mlp_ratio, cfg.drop)

    def forward(self, vol, return_features=False):
        t_feat = self.transformer(vol)
        m_feat = self.mamba(vol)
        logits = self.head(t_feat, m_feat)

        if return_features:
            return logits, (t_feat, m_feat)
        return logits
