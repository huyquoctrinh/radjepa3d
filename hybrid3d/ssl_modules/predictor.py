"""Per-branch predictors for JEPA masked prediction."""

import torch
import torch.nn as nn

from ..mamba_branch import BiMamba


class TransformerPredictor(nn.Module):
    """Predict masked transformer token representations.

    Takes visible encoder outputs + mask token placeholders at masked positions,
    runs lightweight transformer blocks, extracts predictions at masked positions.
    """

    def __init__(self, dim, depth, heads=6, n_tokens=None):
        super().__init__()
        self.dim = dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens or 4096, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        from ..transformer_branch import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, vis_tokens, keep_mask, n_total):
        """
        vis_tokens: (B, n_vis, dim) — encoded visible tokens
        keep_mask: (N_total,) bool — True=visible
        n_total: int — total number of tokens in this branch
        """
        B = vis_tokens.shape[0]

        full_seq = self.mask_token.expand(B, n_total, -1).clone()
        vis_idx = keep_mask.nonzero(as_tuple=True)[0]
        full_seq[:, vis_idx] = vis_tokens

        full_seq = full_seq + self.pos_embed[:, :n_total]

        for block in self.blocks:
            full_seq = block(full_seq)
        full_seq = self.norm(full_seq)

        masked_idx = (~keep_mask).nonzero(as_tuple=True)[0]
        predictions = full_seq[:, masked_idx]
        return self.proj(predictions)


class MambaPredictor(nn.Module):
    """Predict masked Mamba token representations.

    The Mamba context encoder already processed the full sequence (with mask tokens).
    This predictor refines the masked-position outputs.
    """

    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([BiMamba(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, masked_outputs):
        """
        masked_outputs: (B, n_masked, dim) — encoder outputs at masked positions
        """
        x = masked_outputs
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x))
