"""Fusion heads for combining the two branch vectors (downstream classifier).

Variants (cfg.fusion):
    concat     — LayerNorm(concat) -> MLP  (baseline)
    sum        — LayerNorm(t + m) -> MLP
    gated      — learned scalar gate g: g*t + (1-g)*m -> MLP
    cross_attn — one cross-attention block (t attends to m and vice versa) -> MLP
"""

import torch
import torch.nn as nn


class _MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(in_dim * mlp_ratio)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class ConcatFusion(nn.Module):
    def __init__(self, dim, num_classes, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.head = _MLPHead(2 * dim, num_classes, mlp_ratio, drop)

    def forward(self, t_feat, m_feat):
        return self.head(torch.cat([t_feat, m_feat], dim=-1))


class SumFusion(nn.Module):
    def __init__(self, dim, num_classes, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.head = _MLPHead(dim, num_classes, mlp_ratio, drop)

    def forward(self, t_feat, m_feat):
        return self.head(t_feat + m_feat)


class GatedFusion(nn.Module):
    def __init__(self, dim, num_classes, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.head = _MLPHead(dim, num_classes, mlp_ratio, drop)

    def forward(self, t_feat, m_feat):
        g = self.gate(torch.cat([t_feat, m_feat], dim=-1))
        return self.head(g * t_feat + (1.0 - g) * m_feat)


class CrossAttnFusion(nn.Module):
    """Symmetric single-token cross-attention between the two branch vectors."""

    def __init__(self, dim, num_classes, mlp_ratio=4.0, drop=0.0, heads=4):
        super().__init__()
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.attn_m = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = _MLPHead(2 * dim, num_classes, mlp_ratio, drop)

    def forward(self, t_feat, m_feat):
        t = t_feat.unsqueeze(1)   # (B, 1, dim)
        m = m_feat.unsqueeze(1)
        t2 = t + self.attn_t(t, m, m, need_weights=False)[0]
        m2 = m + self.attn_m(m, t, t, need_weights=False)[0]
        fused = torch.cat([self.norm(t2).squeeze(1), self.norm(m2).squeeze(1)], dim=-1)
        return self.head(fused)


_FUSIONS = {
    "concat": ConcatFusion,
    "sum": SumFusion,
    "gated": GatedFusion,
    "cross_attn": CrossAttnFusion,
}


def build_fusion(fusion, dim, num_classes, mlp_ratio=4.0, drop=0.0):
    if fusion not in _FUSIONS:
        raise ValueError(f"Unknown fusion '{fusion}', choose from {list(_FUSIONS)}")
    return _FUSIONS[fusion](dim, num_classes, mlp_ratio, drop)


# Backwards-compatible alias (old name used by model.py)
class LateFusionHead(ConcatFusion):
    pass
