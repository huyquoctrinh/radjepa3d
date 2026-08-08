"""Compare encoder saliency across Pure Mamba / Mamba+Ortho / Hybrid.

Fixes the previous figure, which showed raw 16x16 feature-norm maps standalone:
  - overlays the map on the CT so anatomy correspondence is visible
  - percentile (2-98) normalisation so one outlier patch can't wash the map out
  - smooth upsampling to full CT resolution instead of blocky 16x16
  - shared colour scale + a colourbar

Usage:
    CUDA_VISIBLE_DEVICES=3 python viz_encoder.py
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom

torch.backends.cudnn.enabled = False

from load_checkpoint import load_encoder as _load_encoder

MODELS = [
    ("Pure Mamba",     "/data2/gdp2988/3d_cot/checkpoints/vjepa/checkpoint_epoch0279.pt", "layer"),
    ("Mamba + Ortho",  "/data2/gdp2988/3d_cot/checkpoints/vjepa_ortho/checkpoint_best.pt", "layer"),
    ("Hybrid (Ours)",  "/data2/gdp2988/3d_cot/checkpoints/vjepa_hybrid_layer/checkpoint_best.pt", "layer"),
]
NPY_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"


def load_encoder(path, router_mode="layer"):
    enc, cfg, meta = _load_encoder(path, router_mode=router_mode)
    return enc, cfg


@torch.no_grad()
def saliency(enc, cfg, vol):
    """Per-patch feature norm -> (gd, gh, gw) saliency volume."""
    x = torch.from_numpy(vol).unsqueeze(0).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tok = enc(x)                      # (1, n_patches, dim)
    f = tok[0].float()
    f = f - f.mean(0, keepdim=True)       # centre so norm reflects deviation, not bias
    s = f.norm(dim=-1).cpu().numpy()
    return s.reshape(*cfg.grid_size)


def to_image(sal2d, out_hw, smooth=1.2, lo=2, hi=98):
    """Upsample a coarse saliency slice smoothly and normalise robustly to [0,1]."""
    zy, zx = out_hw[0] / sal2d.shape[0], out_hw[1] / sal2d.shape[1]
    up = zoom(sal2d, (zy, zx), order=3)           # bicubic, not blocky nearest
    up = gaussian_filter(up, sigma=smooth * max(zy, zx) / 8.0)
    a, b = np.percentile(up, lo), np.percentile(up, hi)
    return np.clip((up - a) / max(b - a, 1e-8), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--cmap", type=str, default="turbo")
    ap.add_argument("--out", type=str,
                    default="/data2/gdp2988/3d_cot/checkpoints/encoder_comparison.png")
    args = ap.parse_args()

    vols = sorted(glob.glob(os.path.join(NPY_DIR, "*", "*", "*.npy")))[:200]
    picks = [vols[i] for i in np.linspace(0, len(vols) - 1, args.n_cases).astype(int)]

    encoders = []
    for name, path, rmode in MODELS:
        print(f"  loading {name}")
        encoders.append((name,) + load_encoder(path, rmode))

    nrow, ncol = len(picks), len(MODELS) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.6 * nrow), dpi=200)
    if nrow == 1:
        axes = axes[None, :]

    for r, vp in enumerate(picks):
        vol = np.load(vp).astype(np.float32)          # (1, D, H, W)
        D, H, W = vol.shape[1:]
        ct = vol[0]

        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # depth slice: middle of the volume, and the matching feature-grid depth
        d_ct = D // 2
        axes[r, 0].imshow(ct[d_ct], cmap="gray")
        if r == 0:
            axes[r, 0].set_title("CT Scan", fontsize=11, fontweight="bold")

        # body mask: ignore air/background so saliency is only shown on tissue
        body = gaussian_filter(ct[d_ct], 3) > 0.08
        for c, (name, enc, cfg) in enumerate(encoders, start=1):
            sal = saliency(enc, cfg, vol)
            gd = sal.shape[0]
            d_f = min(int(d_ct / D * gd), gd - 1)
            heat = to_image(sal[d_f], (H, W))
            heat = np.where(body, heat, np.nan)          # blank out background
            # transparency scales with saliency: anatomy stays visible where low
            a = np.nan_to_num(heat) * args.alpha * body
            axes[r, c].imshow(ct[d_ct], cmap="gray")                       # anatomy underneath
            im = axes[r, c].imshow(heat, cmap=args.cmap, alpha=a,
                                   vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(name, fontsize=11,
                                     fontweight="bold" if "Ours" in name else "normal")

    fig.subplots_adjust(wspace=0.04, hspace=0.04, right=0.9)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("normalised feature saliency", fontsize=9)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["low", "", "high"])

    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
