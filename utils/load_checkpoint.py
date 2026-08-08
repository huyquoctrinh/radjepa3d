"""Unified checkpoint loading for RadJepa encoders and Med3D MLLM.

Consolidates the encoder-loading logic that was duplicated across
bench_ablations_clean, encoders, ipmn_classify, inspect_eval, mllms/model,
seg/model, and the visualize_* scripts.

Config is inferred from the checkpoint's own tensors rather than from
directory names or hardcoded defaults -- directory names have already proven
unreliable (e.g. vjepa_hybrid_layer is raster-scan despite the config
defaulting to Morton).

Usage — standalone encoder (JEPA, frozen-feature eval, etc.):

    from load_checkpoint import load_encoder
    enc, cfg, meta = load_encoder(
        "checkpoints/vjepa_hybrid_layer/checkpoint_best.pt",
        router_mode="layer",      # only field not inferable from weights
    )
    tokens = enc(volume, indices=None)          # (B, n_patches, dim)
    embedding = tokens.mean(dim=1)              # (B, dim) global pool

Usage — full MLLM (vision encoder + projector + LLM with LoRA):

    from load_checkpoint import load_mllm
    model = load_mllm(
        "checkpoints/mllm/stage2_best.pt",
        config_overrides={"use_hybrid": True},
    )
    answer = model.generate(volume, "Describe the findings in this CT scan.")
"""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import os

import torch


# ---------------------------------------------------------------------------
# Encoder config inference
# ---------------------------------------------------------------------------

def infer_encoder_config(state_dict, router_mode="layer"):
    """Derive encoder architecture from a target_encoder/context_encoder state dict.

    router_mode is the only field not recoverable from the weights: both
    'token' and 'layer' produce identical state dicts and differ solely in
    forward(). Pass the correct mode if you know it; defaults to 'layer'.
    """
    from vjepa_config import VJEPAConfig

    cfg = VJEPAConfig()

    pw = state_dict["patch_embed.proj.weight"]      # (dim, 1, pd, ph, pw)
    cfg.dim = pw.shape[0]
    cfg.patch_size = tuple(pw.shape[2:])
    cfg.use_morton_scan = "_morton_order" in state_dict
    cfg.use_hybrid = any("gqa" in k for k in state_dict)
    cfg.use_bidirectional = any("mamba_bwd" in k for k in state_dict)
    cfg.n_layers = len(
        {k.split(".")[1] for k in state_dict if k.startswith("blocks.")})

    if cfg.use_hybrid:
        cfg.use_rope = any("gqa.rope" in k for k in state_dict)
        has_router = any(".router." in k for k in state_dict)
        cfg.router_mode = "fixed" if not has_router else router_mode
        k_weight = state_dict["blocks.0.gqa.k_proj.weight"]
        head_dim = cfg.dim // cfg.n_attn_heads
        cfg.n_kv_heads = k_weight.shape[0] // head_dim

    if "pos_embed" in state_dict:
        assert state_dict["pos_embed"].shape[1] == cfg.n_patches, (
            f"pos_embed {state_dict['pos_embed'].shape[1]} != "
            f"n_patches {cfg.n_patches} for patch {cfg.patch_size}")

    return cfg


def describe_config(cfg):
    """Human-readable one-line description of an encoder config."""
    bits = [f"patch{'x'.join(map(str, cfg.patch_size))}"]
    bits.append("morton" if cfg.use_morton_scan else "raster")
    if cfg.use_bidirectional:
        bits.append("bidir")
    if cfg.use_hybrid:
        bits.append(f"router={cfg.router_mode}")
        bits.append(f"kv{cfg.n_kv_heads}")
        bits.append("rope" if cfg.use_rope else "norope")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def _find_encoder_key(ckpt):
    """Auto-detect which key holds the encoder state dict."""
    for key in ("target_encoder", "context_encoder"):
        if key in ckpt:
            return key
    return None


def load_encoder(ckpt_path, device="cuda", router_mode="layer",
                 encoder_key=None, batch_size=32):
    """Load a RadJepa MambaEncoder from any checkpoint.

    Handles JEPA checkpoints (target_encoder / context_encoder keys) and
    standalone encoder state dicts.

    Args:
        ckpt_path: path to .pt checkpoint.
        device: target device.
        router_mode: 'layer', 'token', or 'fixed'. Only matters for hybrid
            models with a learned router. Cannot be inferred from weights.
        encoder_key: override which key to load. Auto-detected if None.
        batch_size: stored in meta for downstream extract() helpers.

    Returns:
        (encoder, config, meta) where encoder is a MambaEncoder on `device`
        in eval mode, config is VJEPAConfig, and meta is a dict.
    """
    from vjepa_model import MambaEncoder

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if encoder_key is None:
        encoder_key = _find_encoder_key(ckpt)
    sd = ckpt[encoder_key] if encoder_key else ckpt

    cfg = infer_encoder_config(sd, router_mode=router_mode)
    enc = MambaEncoder(cfg).to(device).eval()

    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for {ckpt_path} (key={encoder_key}): "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")

    meta = {
        "epoch": ckpt.get("epoch"),
        "encoder_key": encoder_key,
        "best_loss": ckpt.get("best_loss"),
        "global_step": ckpt.get("global_step"),
        "batch_size": batch_size,
    }

    print(f"  loaded {os.path.basename(ckpt_path)} "
          f"(key={encoder_key}, epoch={meta['epoch']}) :: "
          f"{describe_config(cfg)} dim={cfg.dim}", flush=True)

    return enc, cfg, meta


def encoder_extract_fn(enc, device="cuda", batch_size=32):
    """Build an extract(paths) -> (N, D) closure from a loaded encoder."""
    from encoders import batched_extract

    def forward(batch):
        return enc(batch, indices=None).mean(dim=1)

    def extract(paths):
        return batched_extract(forward, paths, batch_size, device,
                               label="radjepa")

    return extract


# ---------------------------------------------------------------------------
# VJEPAConfig from MLLMConfig / SegConfig
# ---------------------------------------------------------------------------

def build_vjepa_config_from(config):
    """Build a VJEPAConfig from an MLLMConfig or SegConfig.

    Avoids duplicating the field-by-field copy that was in mllms/model.py
    and seg/model.py.
    """
    from vjepa_config import VJEPAConfig

    cfg = VJEPAConfig()
    cfg.dim = config.vision_dim
    cfg.n_layers = config.vision_n_layers
    cfg.d_state = config.vision_d_state
    cfg.d_conv = config.vision_d_conv
    cfg.expand = config.vision_expand
    cfg.volume_size = config.volume_size
    cfg.patch_size = config.patch_size
    cfg.use_hybrid = getattr(config, "use_hybrid", False)
    cfg.use_morton_scan = getattr(config, "vision_use_morton", False)
    return cfg


# ---------------------------------------------------------------------------
# MLLM loading
# ---------------------------------------------------------------------------

def load_mllm(ckpt_path, vision_ckpt_path=None, config_overrides=None,
              device="cuda"):
    """Load a full Med3DMLLM from a training checkpoint.

    Args:
        ckpt_path: path to the MLLM training checkpoint (stage1 or stage2).
        vision_ckpt_path: pretrained vision encoder checkpoint. Uses the
            MLLMConfig default if None.
        config_overrides: dict of MLLMConfig field overrides, e.g.
            {"use_hybrid": True, "llm_name": "Qwen/Qwen3-4B"}.
        device: target device.

    Returns:
        Med3DMLLM in eval mode on device.
    """
    from mllms.config import MLLMConfig
    from mllms.model import Med3DMLLM

    config = MLLMConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)
    if vision_ckpt_path:
        config.vision_checkpoint = vision_ckpt_path

    model = Med3DMLLM(config)

    if os.path.exists(config.vision_checkpoint):
        model.load_vision_checkpoint(config.vision_checkpoint)

    model.apply_lora()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.vision_encoder.load_state_dict(ckpt["vision_encoder"])
    model.projector.load_state_dict(ckpt["projector"])
    if "lora" in ckpt:
        model.llm.load_state_dict(ckpt["lora"], strict=False)

    print(f"  loaded MLLM from {os.path.basename(ckpt_path)} "
          f"(epoch={ckpt.get('epoch', '?')})", flush=True)

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 72)
    print("Example: Loading hybrid RadJepa encoder (JEPA checkpoint)")
    print("=" * 72)
    print("""
    from load_checkpoint import load_encoder, encoder_extract_fn

    # --- Hybrid JEPA encoder (Mamba + GQA attention, per-layer routing) ---
    enc, cfg, meta = load_encoder(
        "checkpoints/vjepa_hybrid_layer/checkpoint_best.pt",
        router_mode="layer",          # 'token', 'layer', or 'fixed'
    )

    # Encode a single volume
    volume = torch.randn(1, 1, 32, 256, 256).cuda()
    tokens = enc(volume, indices=None)        # (1, 1024, 384)
    embedding = tokens.mean(dim=1)            # (1, 384)

    # Or build a batch-extraction closure for kNN benchmarks
    extract = encoder_extract_fn(enc, batch_size=32)
    embeddings = extract(list_of_npy_paths)   # (N, 384)
    """)

    print("=" * 72)
    print("Example: Loading hybrid MLLM (vision + projector + LLM)")
    print("=" * 72)
    print("""
    from load_checkpoint import load_mllm

    # --- Hybrid MLLM with Qwen3-4B ---
    model = load_mllm(
        "checkpoints/mllm_hybrid_layer/stage2_best.pt",
        config_overrides={
            "use_hybrid": True,
            "vision_use_morton": False,
            "vision_checkpoint": "checkpoints/vjepa_hybrid_layer/checkpoint_best.pt",
        },
    )

    volume = torch.randn(1, 1, 32, 256, 256).cuda()
    answer = model.generate(volume, "What organ is shown in this CT scan?")
    print(answer)

    # --- With a different LLM backbone ---
    model = load_mllm(
        "checkpoints/mllm_qwen25_ortho/stage2_best.pt",
        config_overrides={
            "llm_name": "Qwen/Qwen2.5-3B-Instruct",
            "llm_dim": 2048,
            "use_hybrid": False,
            "vision_checkpoint": "checkpoints/vjepa_ortho/checkpoint_best.pt",
        },
    )
    """)

    print("=" * 72)
    print("Example: Loading a merged (M3D+INSPECT) hybrid checkpoint")
    print("=" * 72)
    print("""
    from load_checkpoint import load_encoder

    # Merged checkpoints use token routing
    enc, cfg, meta = load_encoder(
        "checkpoints/radjepa_merged_hybrid_token/checkpoint_epoch0039.pt",
        router_mode="token",
    )
    print(f"  dim={cfg.dim}, layers={cfg.n_layers}, hybrid={cfg.use_hybrid}")
    print(f"  patches={cfg.n_patches}, grid={cfg.grid_size}")
    print(f"  morton={cfg.use_morton_scan}, router={cfg.router_mode}")
    """)

    print("=" * 72)
    print("Live demo: loading available checkpoints")
    print("=" * 72)

    DEMO_CKPTS = [
        ("Hybrid JEPA (layer router)",
         "/data2/gdp2988/3d_cot/checkpoints/vjepa_hybrid_layer/checkpoint_best.pt",
         "layer"),
        ("Merged hybrid (token router)",
         "/data2/gdp2988/3d_cot/checkpoints/radjepa_merged_hybrid_token/checkpoint_epoch0039.pt",
         "token"),
        ("Pure Mamba + Ortho",
         "/data2/gdp2988/3d_cot/checkpoints/vjepa_ortho/checkpoint_best.pt",
         "layer"),
    ]

    for label, path, rmode in DEMO_CKPTS:
        if not os.path.exists(path):
            print(f"\n[SKIP] {label}: {path} not found")
            continue
        print(f"\n--- {label} ---")
        enc, cfg, meta = load_encoder(path, router_mode=rmode)
        print(f"  config: {describe_config(cfg)}")
        print(f"  dim={cfg.dim}, patches={cfg.n_patches}, "
              f"grid={cfg.grid_size}, epoch={meta['epoch']}")

        vol = torch.randn(1, 1, 32, 256, 256, device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = enc(vol, indices=None)
        print(f"  output: {tokens.shape} -> embedding {tokens.mean(1).shape}")
        del enc
        torch.cuda.empty_cache()
