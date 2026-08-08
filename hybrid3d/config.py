"""Configuration for Hybrid 3D Late Fusion Encoder."""

from dataclasses import dataclass, field


@dataclass
class Hybrid3DConfig:
    in_chans: int = 1
    img_size: tuple = (32, 256, 256)
    dim: int = 384
    num_classes: int = 2

    # Transformer branch (coarse patches)
    patch_t: tuple = (8, 8, 8)
    depth_t: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0
    pool: str = "mean"

    # Mamba branch (fine patches)
    patch_m: tuple = (4, 4, 4)
    depth_m: int = 8

    # Masking (Phase 2)
    mask_canon: tuple = (4, 4, 4)
    mask_ratio: float = 0.6
    ema_decay: float = 0.996
    pred_depth_t: int = 2
    pred_depth_m: int = 2
    ssl_objective: str = "jepa"

    # Ablation knobs
    mask_mode: str = "shared"        # "shared" (one canonical mask) or "independent"
    mamba_bidirectional: bool = True  # BiMamba forward+backward vs forward-only
    use_morton: bool = True           # Morton scan vs raster order in Mamba branch
    fusion: str = "concat"            # concat | sum | gated | cross_attn (downstream)

    def __post_init__(self):
        D, H, W = self.img_size
        for name, patch in [("patch_t", self.patch_t), ("patch_m", self.patch_m)]:
            pd, ph, pw = patch
            assert D % pd == 0, f"{name}[0]={pd} does not divide D={D}"
            assert H % ph == 0, f"{name}[1]={ph} does not divide H={H}"
            assert W % pw == 0, f"{name}[2]={pw} does not divide W={W}"

    @property
    def n_tokens_t(self):
        D, H, W = self.img_size
        pd, ph, pw = self.patch_t
        return (D // pd) * (H // ph) * (W // pw)

    @property
    def n_tokens_m(self):
        D, H, W = self.img_size
        pd, ph, pw = self.patch_m
        return (D // pd) * (H // ph) * (W // pw)

    @property
    def grid_t(self):
        D, H, W = self.img_size
        pd, ph, pw = self.patch_t
        return (D // pd, H // ph, W // pw)

    @property
    def grid_m(self):
        D, H, W = self.img_size
        pd, ph, pw = self.patch_m
        return (D // pd, H // ph, W // pw)
