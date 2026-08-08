"""Frozen 3D CT encoders for the M3D kNN benchmark.

Each entry in REGISTRY builds a callable `extract(paths) -> (N, D) float32`.
Encoders are constructed lazily so that running one does not require the others'
weights or dependencies to be present.

All extractors share `batched_extract`, which loads volumes on the fly and
*drops* samples it cannot read rather than substituting a zero volume. Because
the eval harness pairs embeddings with labels positionally, a silent zero-volume
substitution would attach a real label to a blank scan; loader failures are
raised instead.
"""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import os

import numpy as np
import torch
import torch.nn.functional as F

# M3D npy volumes are (1, 32, 256, 256) float16, already min-max normalized to
# [0,1] per volume by extract_npy.py. No HU information survives, so encoders
# that expect HU windowing get the best-effort [0,1] input; this is a real
# handicap for them and is noted in the results.
NATIVE_SHAPE = (1, 32, 256, 256)


def load_volume(path):
    vol = np.load(path).astype(np.float32)
    if vol.ndim == 3:
        vol = vol[None]
    return torch.from_numpy(vol)


@torch.no_grad()
def batched_extract(forward, paths, batch_size, device, prep=None,
                    label="", log_every=25):
    """Run `forward` over `paths` in batches, returning (N, D) float32.

    `prep` maps a loaded (C,D,H,W) tensor to the encoder's expected layout.
    """
    out = []
    n_batches = (len(paths) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i + batch_size]
        vols = []
        for p in batch_paths:
            v = load_volume(p)
            vols.append(prep(v) if prep else v)
        batch = torch.stack(vols).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feats = forward(batch)
        out.append(feats.float().cpu())
        if bi % log_every == 0 or bi == n_batches - 1:
            print(f"    {label} {min(i + batch_size, len(paths))}/{len(paths)}",
                  flush=True)
    return torch.cat(out, dim=0).numpy()


# --------------------------------------------------------------------------- #
# RadJepa (this work) — V-JEPA Mamba variants
# --------------------------------------------------------------------------- #

RADJEPA_VARIANTS = {
    # The raster-scan family evaluated in checkpoints/knn_comparison.log, whose
    # config comes from visualize_encoder_comparison.py (use_morton_scan=False).
    # These are NOT the same models as the vjepa_mamba_morton* runs below, so
    # the two sets of logged numbers are not comparable to each other.
    "radjepa-raster-ep279": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa/checkpoint_epoch0279.pt",
        morton=False, hybrid=False),
    "radjepa-raster-ortho": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa_ortho/checkpoint_best.pt",
        morton=False, hybrid=False),

    # Epoch-matched pair for the orthogonality comparison. The logged numbers
    # compared ep279 against ep230, confounding the effect with training
    # length. Caveat: checkpoints/vjepa/ has no train.log, so there is no
    # record that it was actually trained without the ortho terms -- the only
    # evidence is the directory name.
    "radjepa-raster-ep239": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa/checkpoint_epoch0239.pt",
        morton=False, hybrid=False),
    "radjepa-raster-ortho-ep239": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa_ortho/checkpoint_epoch0239.pt",
        morton=False, hybrid=False),

    "radjepa-mamba": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa_mamba_morton/checkpoint_best.pt",
        morton=True, hybrid=False),
    "radjepa-mamba-ortho": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa_mamba_morton_ortho/checkpoint_best.pt",
        morton=True, hybrid=False),
    # Trained with raster scan, not Morton: the checkpoint has no
    # _morton_order buffer, and does have blocks.*.gqa / blocks.*.router.
    "radjepa-hybrid": dict(
        ckpt="/data2/gdp2988/3d_cot/checkpoints/vjepa_hybrid_layer/checkpoint_best.pt",
        morton=False, hybrid=True, router_mode="layer"),
}


def build_radjepa(spec, device="cuda", batch_size=32):
    from load_checkpoint import load_encoder, encoder_extract_fn

    router_mode = spec.get("router_mode", "layer")
    enc, cfg, meta = load_encoder(
        spec["ckpt"], device=device, router_mode=router_mode,
        batch_size=batch_size)

    return encoder_extract_fn(enc, device=device, batch_size=batch_size)


# --------------------------------------------------------------------------- #
# Merlin (StanfordMIMI) — supervised I3-ResNet CT encoder, 2048-d
# --------------------------------------------------------------------------- #

def build_merlin(merlin_repo, device="cuda", batch_size=4):
    import sys
    if merlin_repo not in sys.path:
        sys.path.insert(0, merlin_repo)
    from merlin import Merlin

    model = Merlin(ImageEmbedding=True).to(device).eval()

    def prep(v):                                  # (1,32,256,256)
        t = v.permute(0, 2, 3, 1).unsqueeze(0)    # (1,1,256,256,32) N,C,H,W,D
        t = F.interpolate(t, size=(224, 224, 160), mode="trilinear",
                          align_corners=False)
        return t.squeeze(0)                       # (1,224,224,160)

    def forward(batch):
        return model(batch).squeeze(0)            # (1,B,2048) -> (B,2048)

    def extract(paths):
        return batched_extract(forward, paths, batch_size, device, prep=prep,
                               label="merlin")

    return extract


# --------------------------------------------------------------------------- #
# External pretrained 3D CT baselines
# --------------------------------------------------------------------------- #
#
# IMPORTANT CAVEAT, applies to every encoder below and to Merlin.
#
# All of these were pretrained on CT in Hounsfield units and expect a specific
# HU window mapped to [0,1] (Swin UNETR: [-175,250]; VoCo chest: [-1000,500];
# Models Genesis: [-1000,1000]). M3D-Cap ships JPEG slices, so extract_npy.py
# can only min-max normalize each volume to [0,1] — HU is not recoverable from
# the source data, not merely absent from our pipeline.
#
# Every baseline therefore receives a [0,1] volume that does not match its
# pretraining intensity distribution, and none of their windowing steps can be
# applied. RadJepa was pretrained on exactly this [0,1] representation, so it
# has a home-field advantage that is a property of the benchmark, not of the
# method. This must be stated whenever these numbers are reported.

WEIGHTS_DIR = "/data2/gdp2988/3d_cot/third_party/weights"
MODELS_GENESIS_REPO = "/data2/gdp2988/3d_cot/third_party/ModelsGenesis"

BASELINE_VARIANTS = {
    # Tang et al. CVPR'22 self-supervised Swin UNETR, ~5050 CT volumes.
    "swinunetr-ssl": dict(
        name="swinunetr-ssl", kind="swin",
        ckpt=os.path.join(WEIGHTS_DIR, "model_swinvit.pt"),
        feature_size=48, use_v2=False, roi=(96, 96, 96),
        pool="bottleneck", batch_size=8,
    ),
    # VoCo-B, self-supervised on 160k volumes (TPAMI'25). Same MONAI backbone
    # as above but use_v2=True, and its contrastive head pooled 5 taps.
    "voco-b": dict(
        name="voco-b", kind="swin",
        ckpt=os.path.join(WEIGHTS_DIR, "VoCo_B_SSL_head.pt"),
        feature_size=48, use_v2=True, roi=(96, 96, 96),
        pool="voco5tap", batch_size=8,
    ),
    # Models Genesis (MICCAI'19) UNet3D, self-supervised on LUNA16 chest CT.
    "models-genesis": dict(
        name="models-genesis", kind="genesis",
        ckpt=os.path.join(WEIGHTS_DIR, "Genesis_Chest_CT.pt"),
        roi=(32, 64, 64), batch_size=16,
    ),
}


def _resize_to(v, roi):
    """(C,D,H,W) -> (C,*roi) by trilinear resampling."""
    t = F.interpolate(v.unsqueeze(0), size=roi, mode="trilinear",
                      align_corners=False)
    return t.squeeze(0)


def build_swin_module(spec, device="cuda"):
    """Return the loaded SwinUNETR nn.Module and its embedding dim.

    _build_swin wraps this into an extract() closure; fine-tuning needs the
    module itself so peft can inject adapters into it.
    """
    enc, dim, _ = _swin_core(spec, device)
    return enc, dim


def _swin_core(spec, device):
    """Load SwinUNETR + weights; returns (module, dim, forward_fn)."""
    from monai.networks.nets import SwinUNETR

    # monai 1.5.2 removed img_size from SwinUNETR's signature.
    model = SwinUNETR(
        in_channels=1, out_channels=14,
        feature_size=spec["feature_size"], use_v2=spec["use_v2"],
    )

    sd = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # MONAI's model_swinvit.pt is a bare swinViT dict with the old timm MLP
    # names and the SSL pretext heads attached. VoCo's is already keyed to
    # MONAI attribute names. Normalize both to full-model keys.
    fixed = {}
    # 'norm' is the SSL pretext head's final norm; MONAI's SwinTransformer
    # normalizes inside proj_out and has no top-level .norm. Per-block norms
    # live under layersN.* so they are unaffected by this top-level check.
    pretext = ("contrastive_head", "rotation_head", "convTrans3d", "norm")
    for k, v in sd.items():
        k = k[len("module."):] if k.startswith("module.") else k
        top = k.split(".")[0]
        if top in pretext:
            continue
        if top not in ("swinViT", "encoder1", "encoder2", "encoder3",
                       "encoder4", "encoder10", "decoder1", "decoder2",
                       "decoder3", "decoder4", "decoder5", "out"):
            # Bare swinViT dict (MONAI SSL): 'norm.*' here is the swinViT
            # norm, and everything else is a swinViT submodule.
            k = "swinViT." + k
        # timm -> MONAI MLPBlock naming. Without this, strict=False silently
        # drops all 16 MLP tensors (94/126 loaded instead of 126/126).
        k = k.replace("mlp.fc1", "mlp.linear1").replace("mlp.fc2", "mlp.linear2")
        fixed[k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if unexpected:
        raise RuntimeError(f"{spec['name']}: unexpected keys {unexpected[:8]}")
    n_loaded = len(fixed)
    swin_missing = [k for k in missing if k.startswith("swinViT.")]
    if swin_missing:
        raise RuntimeError(
            f"{spec['name']}: {len(swin_missing)} swinViT tensors did not load, "
            f"e.g. {swin_missing[:5]}")
    print(f"  loaded {n_loaded} tensors into Swin UNETR "
          f"(feature_size={spec['feature_size']}, use_v2={spec['use_v2']}); "
          f"{len(missing)} decoder tensors left random (unused)", flush=True)

    model = model.to(device).eval()
    fs = spec["feature_size"]

    if spec["pool"] == "voco5tap":
        # Reproduces VoCoHead.forward_encs: GAP over 5 taps concatenated,
        # in_dim = 24*feature_size = 1152 for f48. This is the representation
        # the contrastive objective actually shaped.
        def forward(batch):
            h = model.swinViT(batch, model.normalize)
            encs = [model.encoder1(batch), model.encoder2(h[0]),
                    model.encoder3(h[1]), model.encoder4(h[2]),
                    model.encoder10(h[4])]
            return torch.cat(
                [F.adaptive_avg_pool3d(e, 1).flatten(1) for e in encs], dim=1)
        dim = 24 * fs
    else:
        def forward(batch):
            h = model.swinViT(batch, model.normalize)
            return F.adaptive_avg_pool3d(h[4], 1).flatten(1)
        dim = 16 * fs

    print(f"  embedding dim = {dim}", flush=True)
    return model, dim, forward


def _build_swin(spec, device, batch_size):
    model, dim, forward = _swin_core(spec, device)
    roi = spec["roi"]

    def extract(paths):
        return batched_extract(forward, paths, batch_size, device,
                               prep=lambda v: _resize_to(v, roi),
                               label=spec["name"])

    return extract


def _build_genesis(spec, device, batch_size):
    """Models Genesis UNet3D encoder; embedding is the 512-ch bottleneck."""
    import sys
    pkg = os.path.join(MODELS_GENESIS_REPO, "pytorch")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    import unet3d

    model = unet3d.UNet3D()
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    print(f"  loaded {len(sd)} tensors into UNet3D "
          f"(epoch={ckpt.get('epoch', '?')}); embedding dim = 512", flush=True)
    model = model.to(device).eval()

    # Run the encoder path only; the decoder is irrelevant to the embedding.
    def forward(batch):
        out64, skip64 = model.down_tr64(batch)
        out128, skip128 = model.down_tr128(out64)
        out256, skip256 = model.down_tr256(out128)
        out512, _ = model.down_tr512(out256)
        return F.adaptive_avg_pool3d(out512, 1).flatten(1)

    roi = spec["roi"]

    def extract(paths):
        return batched_extract(forward, paths, batch_size, device,
                               prep=lambda v: _resize_to(v, roi),
                               label=spec["name"])

    return extract


def build_baseline(spec, device="cuda", batch_size=8):
    kind = spec["kind"]
    if kind == "swin":
        return _build_swin(spec, device, batch_size)
    if kind == "genesis":
        return _build_genesis(spec, device, batch_size)
    raise ValueError(f"unknown baseline kind: {kind}")
