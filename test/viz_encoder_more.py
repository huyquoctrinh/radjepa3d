"""More cases + PCA + a quantitative test of WHAT the saliency tracks.

Question: do the encoder features highlight organs/lesions, or just edges and
the body outline? We test by correlating the saliency map against:
  - |grad(CT)|  (edges / boundaries)
  - CT intensity (tissue density)
  - distance from the body boundary (outline-vs-interior)
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from scipy.ndimage import gaussian_filter, zoom, sobel, distance_transform_edt

torch.backends.cudnn.enabled = False
from load_checkpoint import load_encoder as _load_encoder

MODELS = [
    ("Pure Mamba",    "/data2/gdp2988/3d_cot/checkpoints/vjepa/checkpoint_epoch0279.pt", "layer"),
    ("Mamba + Ortho", "/data2/gdp2988/3d_cot/checkpoints/vjepa_ortho/checkpoint_best.pt", "layer"),
    ("Hybrid (Ours)", "/data2/gdp2988/3d_cot/checkpoints/vjepa_hybrid_layer/checkpoint_best.pt", "layer"),
]
NPY_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"


def load_encoder(path, router_mode="layer"):
    enc, cfg, meta = _load_encoder(path, router_mode=router_mode)
    return enc, cfg


@torch.no_grad()
def feats(enc, cfg, vol):
    x = torch.from_numpy(vol).unsqueeze(0).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        t = enc(x)
    return t[0].float().cpu().numpy()          # (n_patches, dim)


def sal_from(f, grid):
    fc = f - f.mean(0, keepdims=True)
    return np.linalg.norm(fc, axis=-1).reshape(*grid)


def pca_rgb(f, grid):
    fc = f - f.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(fc, full_matrices=False)
    p = fc @ Vt[:3].T
    p = (p - p.min(0)) / (np.ptp(p, axis=0) + 1e-8)
    return p.reshape(*grid, 3)


def up(m, hw, sm=1.2, lo=2, hi=98):
    z = (hw[0] / m.shape[0], hw[1] / m.shape[1])
    u = zoom(m, z, order=3)
    u = gaussian_filter(u, sm * max(z) / 8.0)
    a, b = np.percentile(u, lo), np.percentile(u, hi)
    return np.clip((u - a) / max(b - a, 1e-8), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=6)
    ap.add_argument("--out", default="/data2/gdp2988/3d_cot/checkpoints/encoder_comparison_more.png")
    args = ap.parse_args()

    vols = sorted(glob.glob(os.path.join(NPY_DIR, "*", "*", "*.npy")))
    picks = [vols[i] for i in np.linspace(0, len(vols) - 1, args.n_cases).astype(int)]
    encs = [(n,) + load_encoder(p, rmode) for n, p, rmode in MODELS]

    ncol = 2 + len(MODELS)      # CT | PCA(hybrid) | 3 saliencies
    fig, ax = plt.subplots(args.n_cases, ncol, figsize=(2.4 * ncol, 2.5 * args.n_cases), dpi=190)
    stats = {n: {"edge": [], "tissue": [], "dist": []} for n, _, _ in encs}

    for r, vp in enumerate(picks):
        vol = np.load(vp).astype(np.float32); D, H, W = vol.shape[1:]
        ct = vol[0]; d = D // 2; sl = ct[d]
        body = gaussian_filter(sl, 3) > 0.08
        edge = np.hypot(sobel(gaussian_filter(sl, 2), 0), sobel(gaussian_filter(sl, 2), 1))
        dist = distance_transform_edt(body)          # depth inside the body

        for c in range(ncol):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s in ax[r, c].spines.values(): s.set_visible(False)
        ax[r, 0].imshow(sl, cmap="gray")
        if r == 0: ax[r, 0].set_title("CT Scan", fontsize=10, fontweight="bold")

        for c, (name, enc, cfg) in enumerate(encs, start=2):
            f = feats(enc, cfg, vol); g = cfg.grid_size
            df = min(int(d / D * g[0]), g[0] - 1)
            s2 = up(sal_from(f, g)[df], (H, W))
            ax[r, c].imshow(sl, cmap="gray")
            im = ax[r, c].imshow(np.where(body, s2, np.nan), cmap="jet",
                                 alpha=np.nan_to_num(s2) * 0.85 * body, vmin=0, vmax=1)
            if r == 0:
                ax[r, c].set_title(name, fontsize=10,
                                   fontweight="bold" if "Ours" in name else "normal")
            m = body & np.isfinite(s2)
            sv = s2[m]
            for k, ref in (("edge", edge[m]), ("tissue", sl[m]), ("dist", dist[m])):
                if sv.std() > 1e-6 and ref.std() > 1e-6:
                    stats[name][k].append(float(np.corrcoef(sv, ref)[0, 1]))
            if name == "Hybrid (Ours)":
                ax[r, 1].imshow(zoom(pca_rgb(f, g)[df], (H / g[1], W / g[2], 1), order=1))
                if r == 0: ax[r, 1].set_title("PCA (Hybrid)", fontsize=10)

    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(args.out, bbox_inches="tight"); print("saved", args.out)

    print("\n=== What does the saliency actually track? (mean Pearson r over cases) ===")
    print(f"{'model':<16}{'|grad CT| (edges)':>19}{'CT intensity':>15}{'depth inside body':>20}")
    for n in stats:
        e, t, d_ = (np.mean(stats[n][k]) if stats[n][k] else float('nan')
                    for k in ("edge", "tissue", "dist"))
        print(f"{n:<16}{e:>19.3f}{t:>15.3f}{d_:>20.3f}")
    print("\n(high |edge| r => tracks boundaries; high depth r => interior/organs;")
    print(" negative depth r => concentrated at the body outline)")


if __name__ == "__main__":
    main()
