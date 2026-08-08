"""V-JEPA model with Mamba backbone for 3D CT volumes."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

from vjepa_config import VJEPAConfig
from vjepa_hybrid import HybridBlock
from hybrid3d.serialize import morton_order


class PatchEmbed3D(nn.Module):

    def __init__(self, in_channels=1, dim=384, patch_size=(8, 16, 16)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b d gd gh gw -> b (gd gh gw) d")
        return x


class MambaBlock(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, bidirectional=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.bidirectional = bidirectional
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        y = self.norm(x)
        out = self.mamba(y)
        if self.bidirectional:
            out = out + self.mamba_bwd(y.flip(1)).flip(1)
        return x + out


class MambaEncoder(nn.Module):

    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.dim = config.dim
        self.use_morton = getattr(config, "use_morton_scan", False)
        self.patch_embed = PatchEmbed3D(
            in_channels=1, dim=config.dim, patch_size=config.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_patches, config.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.use_morton:
            gd, gh, gw = config.grid_size
            self.register_buffer("_morton_order", morton_order(gd, gh, gw))

        if getattr(config, "use_hybrid", False):
            self.blocks = nn.ModuleList([
                HybridBlock(
                    config.dim, config.d_state, config.d_conv, config.expand,
                    config.n_attn_heads, config.n_kv_heads, config.grid_size,
                    config.use_rope, getattr(config, "router_mode", "token"),
                )
                for _ in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                MambaBlock(config.dim, config.d_state, config.d_conv, config.expand,
                           bidirectional=getattr(config, "use_bidirectional", False))
                for _ in range(config.n_layers)
            ])
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, x, indices=None, return_hidden=False):
        """
        x: (B, 1, D, H, W)
        indices: (B, n_tokens) or None (all patches)
        return_hidden: if True, also return mean-pooled hidden states per layer
        Returns: tokens (B, n_tokens, dim), [optional] hiddens list of (B, dim)
        """
        tokens = self.patch_embed(x) + self.pos_embed

        if self.use_morton:
            tokens = tokens[:, self._morton_order]

        if indices is not None:
            idx = indices.unsqueeze(-1).expand(-1, -1, self.dim)
            tokens = torch.gather(tokens, 1, idx)

        hiddens = []
        for block in self.blocks:
            tokens = block(tokens)
            if return_hidden:
                hiddens.append(tokens.mean(dim=1))  # (B, dim)

        out = self.norm(tokens)
        if return_hidden:
            return out, hiddens
        return out


class VJEPAPredictor(nn.Module):

    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.dim = config.dim
        self.pred_dim = config.predictor_dim
        self.n_patches = config.n_patches

        self.input_proj = nn.Linear(config.dim, config.predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_patches, config.predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if getattr(config, "use_hybrid", False):
            pred_heads = max(2, config.n_attn_heads // 2)
            pred_kv = max(1, config.n_kv_heads // 2)
            self.blocks = nn.ModuleList([
                HybridBlock(
                    config.predictor_dim, config.d_state, config.d_conv, config.expand,
                    pred_heads, pred_kv, config.grid_size,
                    config.use_rope, getattr(config, "router_mode", "token"),
                )
                for _ in range(config.predictor_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                MambaBlock(config.predictor_dim, config.d_state, config.d_conv, config.expand)
                for _ in range(config.predictor_layers)
            ])
        self.norm = nn.LayerNorm(config.predictor_dim)
        self.output_proj = nn.Linear(config.predictor_dim, config.dim)

    def forward(self, context_tokens, visible_indices, masked_indices):
        B = context_tokens.shape[0]
        n_mask = masked_indices.shape[1]

        context = self.input_proj(context_tokens)

        full_seq = self.mask_token.expand(B, self.n_patches, -1).to(dtype=context.dtype).clone()

        vis_idx = visible_indices.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        full_seq.scatter_(1, vis_idx, context)

        full_seq = full_seq + self.pos_embed

        for block in self.blocks:
            full_seq = block(full_seq)
        full_seq = self.norm(full_seq)

        mask_idx = masked_indices.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        predictions = torch.gather(full_seq, 1, mask_idx)

        return self.output_proj(predictions)


def soft_orthogonality_loss(model):
    """SO regularizer: ||W^T W - I||_F^2 for Linear/Conv weight matrices.

    Skips Mamba internal weights (conv1d, SSM projections) — their shapes
    are too non-square for orthogonality to be meaningful.
    """
    loss = 0.0
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, Mamba):
            continue
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            W = m.weight.view(m.weight.shape[0], -1)
            if W.shape[0] <= W.shape[1]:
                gram = W @ W.T
            else:
                gram = W.T @ W
            I = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            loss = loss + (gram - I).pow(2).sum()
            count += 1
    return loss / max(count, 1)


def orthogonal_alignment_loss(student_hiddens, teacher_hiddens):
    """Align mean-pooled hidden states between student (masked) and teacher (full).

    For each layer, compute cross-correlation matrix between normalized student
    and teacher features, and push it toward identity — the on-diagonal elements
    should be 1 (alignment) and off-diagonal should be 0 (decorrelation).
    """
    loss = 0.0
    n_layers = min(len(student_hiddens), len(teacher_hiddens))
    for i in range(n_layers):
        s = student_hiddens[i]  # (B, dim)
        t = teacher_hiddens[i]  # (B, dim)

        # Normalize along batch dimension
        s = (s - s.mean(0)) / (s.std(0) + 1e-6)
        t = (t - t.mean(0)) / (t.std(0) + 1e-6)

        B = s.shape[0]
        # Cross-correlation: (dim, dim)
        cc = (s.T @ t) / B

        I = torch.eye(cc.shape[0], device=cc.device, dtype=cc.dtype)
        # On-diagonal: push toward 1 (alignment)
        on_diag = (cc.diag() - 1).pow(2).sum()
        # Off-diagonal: push toward 0 (decorrelation)
        off_diag = cc.pow(2).sum() - cc.diag().pow(2).sum()

        loss = loss + on_diag + 0.005 * off_diag

    return loss / max(n_layers, 1)


class VJEPA3D(nn.Module):

    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        self.context_encoder = MambaEncoder(config)
        self.predictor = VJEPAPredictor(config)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, volumes, visible_indices, masked_indices):
        """
        Returns: dict with 'loss', 'jepa_loss', 'ortho_align_loss', 'ortho_reg_loss'
        """
        # Context encoder with hidden states
        context_out, ctx_hiddens = self.context_encoder(
            volumes, indices=visible_indices, return_hidden=True,
        )

        # Predictor
        predictions = self.predictor(context_out, visible_indices, masked_indices)

        # Target encoder with hidden states (no grad)
        with torch.no_grad():
            target_out, tgt_hiddens = self.target_encoder(
                volumes, indices=None, return_hidden=True,
            )
            target_masked = torch.gather(
                target_out, 1,
                masked_indices.unsqueeze(-1).expand(-1, -1, target_out.shape[-1]),
            )

        # Main JEPA loss: L1 in representation space
        jepa_loss = F.l1_loss(predictions, target_masked)

        # Orthogonal alignment loss: align per-layer spatial features
        ortho_align = orthogonal_alignment_loss(ctx_hiddens, tgt_hiddens)

        # Soft orthogonality regularizer on encoder weights
        ortho_reg = soft_orthogonality_loss(self.context_encoder)

        # Balance loss from hybrid routing
        balance_loss = torch.tensor(0.0, device=jepa_loss.device)
        if getattr(self.config, "use_hybrid", False):
            for block in self.context_encoder.blocks:
                if hasattr(block, "_balance_loss") and block._balance_loss is not None:
                    balance_loss = balance_loss + block._balance_loss
            for block in self.predictor.blocks:
                if hasattr(block, "_balance_loss") and block._balance_loss is not None:
                    balance_loss = balance_loss + block._balance_loss
            n_blocks = self.config.n_layers + self.config.predictor_layers
            balance_loss = balance_loss / max(n_blocks, 1)

        gamma_align = self.config.ortho_align_weight
        gamma_reg = self.config.ortho_reg_weight
        gamma_bal = getattr(self.config, "balance_loss_weight", 0.01)
        loss = jepa_loss + gamma_align * ortho_align + gamma_reg * ortho_reg + gamma_bal * balance_loss

        return {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "ortho_align_loss": ortho_align,
            "ortho_reg_loss": ortho_reg,
            "balance_loss": balance_loss,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        for p_t, p_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1.0 - momentum)
