"""HybridJEPA: per-branch masked JEPA pretraining with shared volume mask."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Hybrid3DConfig
from ..transformer_branch import TransformerBranch
from ..mamba_branch import MambaBranch
from ..masking import HybridMaskedInput, project_to_grid
from ..stems import PatchEmbed3D
from ..serialize import morton_order
from .target_encoder import EMATarget
from .predictor import TransformerPredictor, MambaPredictor


class HybridJEPA(nn.Module):

    def __init__(self, cfg: Hybrid3DConfig):
        super().__init__()
        self.cfg = cfg

        # Shared masker
        self.masker = HybridMaskedInput(cfg)

        # Context encoders (trained)
        self.transformer = TransformerBranch(
            cfg.in_chans, cfg.dim, cfg.patch_t, cfg.depth_t,
            cfg.heads, cfg.mlp_ratio, cfg.drop, cfg.pool, cfg.n_tokens_t,
        )
        self.mamba = MambaBranch(
            cfg.in_chans, cfg.dim, cfg.patch_m, cfg.depth_m,
            cfg.drop, cfg.n_tokens_m,
            bidirectional=cfg.mamba_bidirectional, use_morton=cfg.use_morton,
        )

        # EMA target encoders (frozen, one per branch)
        self.target_t = EMATarget(self.transformer, cfg.ema_decay)
        self.target_m = EMATarget(self.mamba, cfg.ema_decay)

        # Predictors
        self.pred_t = TransformerPredictor(
            cfg.dim, cfg.pred_depth_t, cfg.heads, cfg.n_tokens_t,
        )
        self.pred_m = MambaPredictor(cfg.dim, cfg.pred_depth_m)

        # Target branch embeddings (for full-volume target computation)
        self.target_embed_t = PatchEmbed3D(cfg.in_chans, cfg.dim, cfg.patch_t)
        self.target_pos_t = nn.Parameter(torch.zeros(1, cfg.n_tokens_t, cfg.dim))
        self.target_embed_m = PatchEmbed3D(cfg.in_chans, cfg.dim, cfg.patch_m)
        self.target_pos_m = nn.Parameter(torch.zeros(1, cfg.n_tokens_m, cfg.dim))
        nn.init.trunc_normal_(self.target_pos_t, std=0.02)
        nn.init.trunc_normal_(self.target_pos_m, std=0.02)

    def forward(self, vol):
        batch = self.masker(vol)

        # Context: encode visible/masked inputs
        enc_t = self.transformer.encode_tokens(batch["vis_t"])
        enc_m = self.mamba.encode_seq(batch["seq_m"])

        # Predict masked positions
        pred_t = self.pred_t(enc_t, batch["keep_t"], self.cfg.n_tokens_t)
        masked_m_outputs = enc_m[:, ~batch["keep_m_s"]]
        pred_m = self.pred_m(masked_m_outputs)

        # Targets (no grad): encode full volume per branch
        with torch.no_grad():
            # Transformer target
            tgt_tok_t, _ = self.target_embed_t(vol)
            tgt_tok_t = tgt_tok_t + self.target_pos_t
            tgt_full_t = self.target_t.encoder.encode_tokens(tgt_tok_t)
            masked_idx_t = (~batch["keep_t"]).nonzero(as_tuple=True)[0]
            tgt_t = tgt_full_t[:, masked_idx_t].detach()

            # Mamba target
            tgt_tok_m, grid_m = self.target_embed_m(vol)
            tgt_tok_m = tgt_tok_m + self.target_pos_m
            order = batch["order"]
            tgt_tok_m = tgt_tok_m[:, order]
            tgt_full_m = self.target_m.encoder.encode_seq(tgt_tok_m)
            tgt_m = tgt_full_m[:, ~batch["keep_m_s"]].detach()

        # Per-branch JEPA loss
        loss_t = F.smooth_l1_loss(pred_t, tgt_t)
        loss_m = F.smooth_l1_loss(pred_m, tgt_m)

        return loss_t + loss_m

    @torch.no_grad()
    def ema_step(self):
        self.target_t.update(self.transformer)
        self.target_m.update(self.mamba)

    def load_into(self, encoder):
        """Transfer pretrained branch weights into a HybridLateFusion model."""
        encoder.transformer.load_state_dict(self.transformer.state_dict())
        encoder.mamba.load_state_dict(self.mamba.state_dict())
