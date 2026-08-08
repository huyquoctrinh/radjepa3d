"""Configuration for 3D segmentation on M3D-Seg."""

from dataclasses import dataclass


@dataclass
class SegConfig:
    # Vision encoder (pretrained V-JEPA Mamba)
    vision_checkpoint: str = "/data2/gdp2988/3d_cot/checkpoints/vjepa_ortho/checkpoint_best.pt"
    vision_dim: int = 384
    vision_n_layers: int = 12
    vision_d_state: int = 16
    vision_d_conv: int = 4
    vision_expand: int = 2
    volume_size: tuple = (32, 256, 256)
    patch_size: tuple = (8, 16, 16)

    # Decoder
    decoder_dims: tuple = (256, 128, 64)
    num_classes: int = 25  # max classes across all sub-datasets

    # Data
    data_dir: str = "/data2/gdp2988/3d_cot/dataset/m3d_seg/M3D_Seg"
    # Sub-datasets to use (0000-0024)
    dataset_codes: tuple = tuple(f"{i:04d}" for i in range(25))

    # Training
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    lr_encoder: float = 1e-5
    weight_decay: float = 0.01
    total_epochs: int = 50
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "/data2/gdp2988/3d_cot/checkpoints/seg_multiclass"
    save_every_epochs: int = 5
    log_every_steps: int = 50
