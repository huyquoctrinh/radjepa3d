"""Benchmark Design B (hybrid3d late-fusion) SSL ablations on kNN.

Loads pretrained transformer+mamba branches from each ablation checkpoint,
extracts concatenated branch features, runs kNN (k=20) on Plane/Phase/Organ.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_hybrid_b.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_hybrid_b.py --only baseline,mask_independent
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import glob
import numpy as np
import torch

torch.backends.cudnn.enabled = False

from hybrid3d.config import Hybrid3DConfig
from hybrid3d.transformer_branch import TransformerBranch
from hybrid3d.mamba_branch import MambaBranch
from vjepa_benchmark import CSV_FILES, load_labels, img_path_to_npy, run_knn

CKPT_ROOT = "/data2/gdp2988/3d_cot/checkpoints/ablations_hybrid_b"
NPY_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"

# Branch-architecture spec per ablation (must match run_hybrid_b_ablations.sh).
_B = dict(patch_t=(8, 16, 16), patch_m=(8, 8, 8), bidir=True, morton=True)
ABLATION_CFG = {
    "baseline":            dict(_B),
    "mask_independent":    dict(_B),
    "patch_symmetric":     {**_B, "patch_t": (8, 8, 8), "patch_m": (8, 8, 8)},
    "mamba_unidirectional": {**_B, "bidir": False},
    "no_morton":           {**_B, "morton": False},
    "pred_depth_4":        dict(_B),
}


def build_branches(spec):
    cfg = Hybrid3DConfig(
        img_size=(32, 256, 256),
        patch_t=spec["patch_t"], patch_m=spec["patch_m"],
        mamba_bidirectional=spec["bidir"], use_morton=spec["morton"],
    )
    t = TransformerBranch(cfg.in_chans, cfg.dim, cfg.patch_t, cfg.depth_t,
                          cfg.heads, cfg.mlp_ratio, cfg.drop, cfg.pool, cfg.n_tokens_t)
    m = MambaBranch(cfg.in_chans, cfg.dim, cfg.patch_m, cfg.depth_m,
                    cfg.drop, cfg.n_tokens_m,
                    bidirectional=cfg.mamba_bidirectional, use_morton=cfg.use_morton)
    return t, m


def latest_ckpt(name):
    best = os.path.join(CKPT_ROOT, name, "best.pt")
    if os.path.exists(best):
        return best
    ckpts = sorted(glob.glob(os.path.join(CKPT_ROOT, name, "epoch*.pt")))
    return ckpts[-1] if ckpts else None


@torch.no_grad()
def extract_concat(t, m, npy_paths, batch_size=32, device="cuda"):
    embeds = []
    for i in range(0, len(npy_paths), batch_size):
        vols = []
        for p in npy_paths[i:i + batch_size]:
            try:
                vols.append(torch.from_numpy(np.load(p).astype(np.float32)))
            except Exception:
                vols.append(torch.zeros(1, 32, 256, 256))
        batch = torch.stack(vols).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tf = t(batch).float()   # (B, dim)
            mf = m(batch).float()   # (B, dim)
        embeds.append(torch.cat([tf, mf], dim=-1).cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  Extracted {min(i + batch_size, len(npy_paths))}/{len(npy_paths)}",
                  flush=True)
    return torch.cat(embeds, dim=0).numpy()


def benchmark_model(name, spec, max_train=5000):
    ckpt_path = latest_ckpt(name)
    if ckpt_path is None:
        print(f"[SKIP] {name}: no checkpoint", flush=True)
        return None

    t, m = build_branches(spec)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    t.load_state_dict(ckpt["transformer"]); m.load_state_dict(ckpt["mamba"])
    t = t.cuda().eval(); m = m.cuda().eval()
    print(f"[{name}] loaded (epoch {ckpt.get('epoch', '?')}) from {os.path.basename(ckpt_path)}",
          flush=True)

    results = {}
    for qtype, task in [("1", "Plane"), ("2", "Phase"), ("3", "Organ")]:
        train_labels = load_labels(CSV_FILES["train"], qtype)
        test_labels = load_labels(CSV_FILES["val"], qtype)

        tr_paths, tr_labs = [], []
        for img, lab in train_labels.items():
            npy = img_path_to_npy(img, NPY_DIR)
            if os.path.exists(npy):
                tr_paths.append(npy); tr_labs.append(lab)
        te_paths, te_labs = [], []
        for img, lab in test_labels.items():
            npy = img_path_to_npy(img, NPY_DIR)
            if os.path.exists(npy):
                te_paths.append(npy); te_labs.append(lab)

        common = sorted(set(tr_labs) & set(te_labs))
        tr_paths = [p for p, l in zip(tr_paths, tr_labs) if l in common][:max_train]
        tr_labs = [l for l in tr_labs if l in common][:max_train]
        te_paths = [p for p, l in zip(te_paths, te_labs) if l in common]
        te_labs = [l for l in te_labs if l in common]

        tr_emb = extract_concat(t, m, tr_paths)
        te_emb = extract_concat(t, m, te_paths)
        acc, _, _ = run_knn(tr_emb, tr_labs, te_emb, te_labs)
        results[task] = acc
        print(f"  {task}: kNN={acc:.4f}", flush=True)

    del t, m
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    names = list(ABLATION_CFG.keys())
    if args.only:
        names = [n.strip() for n in args.only.split(",")]

    all_results = {}
    for name in names:
        res = benchmark_model(name, ABLATION_CFG[name])
        if res:
            all_results[name] = res

    print()
    print("=" * 70)
    print("Design B (late-fusion) Ablation kNN Results (k=20)")
    print("=" * 70)
    print(f'{"Ablation":<22} {"Plane":>10} {"Phase":>10} {"Organ":>10} {"Mean":>10}')
    print("-" * 64)
    base = all_results.get("baseline")
    for name in names:
        if name not in all_results:
            continue
        r = all_results[name]
        mean = (r["Plane"] + r["Phase"] + r["Organ"]) / 3
        tag = ""
        if base and name != "baseline":
            bmean = (base["Plane"] + base["Phase"] + base["Organ"]) / 3
            tag = f"  ({mean - bmean:+.3f})"
        print(f'{name:<22} {100*r["Plane"]:>9.1f}% {100*r["Phase"]:>9.1f}% '
              f'{100*r["Organ"]:>9.1f}% {100*mean:>9.1f}%{tag}')


if __name__ == "__main__":
    main()
