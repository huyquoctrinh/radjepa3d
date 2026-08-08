"""Configuration for V-JEPA with Mamba backbone on 3D CT volumes."""

from dataclasses import dataclass, field


@dataclass
class VJEPAConfig:
    # Data
    data_json: str = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/M3D_Cap.json"
    ct_case_dir: str = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/ct_case"
    ct_quizze_dir: str = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/ct_quizze"
    npy_dir: str = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"
    volume_size: tuple = (32, 256, 256)
    patch_size: tuple = (8, 16, 16)
    num_workers: int = 8
    prefetch_factor: int = 2

    # Model
    dim: int = 384
    n_layers: int = 12
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Scan order
    use_morton_scan: bool = True  # apply Morton Z-order to tokens before Mamba
    use_bidirectional: bool = False  # bidirectional Mamba (forward + backward scan)

    # Hybrid block
    use_hybrid: bool = False
    n_attn_heads: int = 8
    n_kv_heads: int = 2
    router_top_k: int = 1
    router_mode: str = "layer"  # "token" or "layer"
    balance_loss_weight: float = 0.01
    use_rope: bool = True

    # Predictor
    predictor_dim: int = 192
    predictor_layers: int = 4

    # Masking
    mask_ratio: float = 0.75
    mask_strategy: str = "random"

    # Orthogonal loss
    ortho_align_weight: float = 0.1
    ortho_reg_weight: float = 1e-6

    # EMA
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.9999

    # Training
    batch_size: int = 16
    total_epochs: int = 300
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 40
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"

    # Checkpointing
    checkpoint_dir: str = "/data2/gdp2988/3d_cot/checkpoints/vjepa_mamba_morton_ortho"
    save_every_epochs: int = 20
    log_every_steps: int = 50
    resume_from: str = ""

    @property
    def n_patches(self) -> int:
        d, h, w = self.volume_size
        pd, ph, pw = self.patch_size
        return (d // pd) * (h // ph) * (w // pw)

    @property
    def grid_size(self) -> tuple:
        d, h, w = self.volume_size
        pd, ph, pw = self.patch_size
        return (d // pd, h // ph, w // pw)

    @property
    def n_visible(self) -> int:
        return int(self.n_patches * (1 - self.mask_ratio))

    @property
    def n_masked(self) -> int:
        return self.n_patches - self.n_visible
