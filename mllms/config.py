"""Configuration for the 3D Medical MLLM."""

from dataclasses import dataclass


@dataclass
class MLLMConfig:
    # Vision encoder (from V-JEPA pretrained)
    vision_checkpoint: str = "/data2/gdp2988/3d_cot/checkpoints/vjepa_hybrid_layer/checkpoint_best.pt"
    use_hybrid: bool = True
    vision_use_morton: bool = False  # match the SSL encoder's scan order
    vision_pool_tokens: int = 1024   # 1024 patch tokens -> N (1024 = no pooling)
    vision_dim: int = 384
    vision_n_patches: int = 1024
    vision_n_layers: int = 12
    vision_d_state: int = 16
    vision_d_conv: int = 4
    vision_expand: int = 2
    volume_size: tuple = (32, 256, 256)
    patch_size: tuple = (8, 16, 16)

    # LLM
    llm_name: str = "Qwen/Qwen3-4B"
    llm_dim: int = 2560
    llm_max_length: int = 512

    # Projector
    projector_type: str = "mlp"  # "linear" or "mlp"
    projector_hidden: int = 1024

    # Contrastive loss (CLIP-style, stage 1 auxiliary)
    contrastive_lambda: float = 0.1
    contrastive_temperature: float = 0.07

    # LoRA (stage 2)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj")

    # Data
    data_json: str = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/M3D_Cap.json"
    npy_dir: str = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"
    vqa_train_csv: str = "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_train.csv"
    vqa_val_csv: str = "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_val.csv"
    caption_oversample: int = 1  # repeat caption data N times on train (balance VQA:Cap)
    vqa_max_train: int = 0        # cap VQA train samples (0=all) to balance vs captions

    # Training
    batch_size_stage1: int = 2
    batch_size_stage2: int = 2
    num_workers: int = 4
    lr_projector: float = 1e-3
    lr_vision: float = 1e-5
    lr_lora: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    total_epochs_stage1: int = 1
    total_epochs_stage2: int = 1
    grad_clip: float = 1.0
    grad_accum_steps: int = 4

    # Checkpointing
    checkpoint_dir: str = "/data2/gdp2988/3d_cot/checkpoints/mllm_hybrid_layer"
    save_every_epochs: int = 1
    log_every_steps: int = 50

    # Special tokens
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"
