"""Example: loading Rad-JEPA 3D encoder and MLLM for inference.

Usage:
    cd src/
    CUDA_VISIBLE_DEVICES=0 python example.py --mode encoder
    CUDA_VISIBLE_DEVICES=0 python example.py --mode mllm
    CUDA_VISIBLE_DEVICES=0 python example.py --mode both
"""

import argparse
import os

import numpy as np
import torch

import sys as _sys, os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_SRC, _os.path.join(_SRC, "utils")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from load_checkpoint import load_encoder, encoder_extract_fn, load_mllm

ENCODER_CKPT = "/data2/gdp2988/3d_cot/checkpoints/vjepa_hybrid_layer/checkpoint_best.pt"
MLLM_CKPT = "/data2/gdp2988/3d_cot/checkpoints/mllm_hybrid_layer/stage2_best.pt"
SAMPLE_VOLUME = "/data2/gdp2988/3d_cot/extracted_M3D_Cap/ct_case/000001/Abdomen___Pelvis.npy"


def demo_encoder(ckpt_path, volume_path=None):
    """Load a hybrid Rad-JEPA encoder and extract embeddings."""

    print("=" * 60)
    print("Loading Rad-JEPA 3D Encoder")
    print("=" * 60)

    # ---- Load encoder ----
    # router_mode is the only field not inferable from the checkpoint.
    # Use 'layer' for vjepa_hybrid_layer checkpoints,
    # 'token' for radjepa_merged_hybrid_token checkpoints.
    enc, cfg, meta = load_encoder(ckpt_path, router_mode="layer")

    print(f"\n  Model info:")
    print(f"    dim        = {cfg.dim}")
    print(f"    layers     = {cfg.n_layers}")
    print(f"    patches    = {cfg.n_patches} (grid {cfg.grid_size})")
    print(f"    patch_size = {cfg.patch_size}")
    print(f"    hybrid     = {cfg.use_hybrid}")
    print(f"    morton     = {cfg.use_morton_scan}")
    if cfg.use_hybrid:
        print(f"    router     = {cfg.router_mode}")
        print(f"    kv_heads   = {cfg.n_kv_heads}")
        print(f"    rope       = {cfg.use_rope}")
    print(f"    epoch      = {meta['epoch']}")

    # ---- Single volume inference ----
    if volume_path and os.path.exists(volume_path):
        vol = np.load(volume_path).astype(np.float32)
        x = torch.from_numpy(vol).unsqueeze(0).cuda()    # (1, 1, 32, 256, 256)
        print(f"\n  Input volume: {volume_path}")
    else:
        x = torch.randn(1, 1, 32, 256, 256, device="cuda")
        print(f"\n  Input volume: random tensor (no volume file found)")

    print(f"    shape = {tuple(x.shape)}")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens = enc(x, indices=None)             # (1, 1024, 384)
        embedding = tokens.mean(dim=1)            # (1, 384)

    print(f"\n  Output:")
    print(f"    tokens    = {tuple(tokens.shape)}")
    print(f"    embedding = {tuple(embedding.shape)}")
    print(f"    norm      = {embedding.norm().item():.4f}")

    # ---- Batch extraction (for kNN benchmarks) ----
    extract = encoder_extract_fn(enc, batch_size=32)
    print(f"\n  Batch extractor ready: extract(paths) -> (N, {cfg.dim})")

    del enc
    torch.cuda.empty_cache()
    print("\n  Done.\n")


def demo_mllm(mllm_ckpt_path, encoder_ckpt_path, volume_path=None):
    """Load a hybrid MLLM and generate a response."""

    print("=" * 60)
    print("Loading Rad-JEPA 3D MLLM")
    print("=" * 60)

    if not os.path.exists(mllm_ckpt_path):
        print(f"\n  [SKIP] MLLM checkpoint not found: {mllm_ckpt_path}")
        print("  To train an MLLM, run:")
        print("    torchrun --nproc_per_node=4 training/train_mllm.py --stage 1")
        print("    torchrun --nproc_per_node=4 training/train_mllm.py --stage 2 --resume <stage1.pt>")
        return

    # ---- Load full MLLM ----
    model = load_mllm(
        mllm_ckpt_path,
        config_overrides={
            "use_hybrid": True,
            "vision_use_morton": False,
            "vision_checkpoint": encoder_ckpt_path,
        },
    )

    # ---- Prepare volume ----
    if volume_path and os.path.exists(volume_path):
        vol = np.load(volume_path).astype(np.float32)
        x = torch.from_numpy(vol).unsqueeze(0).cuda()
        print(f"\n  Input volume: {volume_path}")
    else:
        x = torch.randn(1, 1, 32, 256, 256, device="cuda")
        print(f"\n  Input volume: random tensor")

    # ---- Generate ----
    prompts = [
        "What imaging plane is this CT scan acquired in?",
        "Describe the findings in this 3D CT scan.",
    ]

    for prompt in prompts:
        print(f"\n  Prompt: {prompt}")
        answer = model.generate(x, prompt, max_new_tokens=128)
        print(f"  Answer: {answer}")

    del model
    torch.cuda.empty_cache()
    print("\n  Done.\n")


def main():
    p = argparse.ArgumentParser(description="Rad-JEPA 3D loading examples")
    p.add_argument("--mode", default="encoder", choices=["encoder", "mllm", "both"])
    p.add_argument("--encoder-ckpt", default=ENCODER_CKPT)
    p.add_argument("--mllm-ckpt", default=MLLM_CKPT)
    p.add_argument("--volume", default=SAMPLE_VOLUME)
    args = p.parse_args()

    if args.mode in ("encoder", "both"):
        demo_encoder(args.encoder_ckpt, args.volume)

    if args.mode in ("mllm", "both"):
        demo_mllm(args.mllm_ckpt, args.encoder_ckpt, args.volume)


if __name__ == "__main__":
    main()
