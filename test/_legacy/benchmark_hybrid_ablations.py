"""Benchmark Design A hybrid ablation checkpoints on kNN (Plane/Phase/Organ).

Reconstructs each hybrid encoder from its config spec and runs the standard
kNN (k=20) benchmark. Only encoder-architecture fields matter for loading
target_encoder: router_mode, n_kv_heads, use_rope, patch, morton.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_hybrid_ablations.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_hybrid_ablations.py --only baseline,kv8
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import glob
import torch

torch.backends.cudnn.enabled = False

from vjepa_config import VJEPAConfig
from vjepa_model import MambaEncoder
from vjepa_benchmark import (
    CSV_FILES, load_labels, img_path_to_npy,
    extract_embeddings, run_knn,
)

CKPT_ROOT = "/data2/gdp2988/3d_cot/checkpoints/ablations_hybrid"

# Encoder-arch fields per ablation (must match run_hybrid_ablations.sh).
# router/kv/rope/morton/patch define the encoder; balance & predictor do not.
_B = dict(morton=False, router="layer", kv=2, rope=True, patch=(8, 16, 16))
ABLATION_CFG = {
    "baseline":      dict(_B),
    "router_token":  {**_B, "router": "token"},
    "router_fixed":  {**_B, "router": "fixed"},
    "balance_0":     dict(_B),
    "balance_0.1":   dict(_B),
    "morton":        {**_B, "morton": True},
    "no_rope":       {**_B, "rope": False},
    "kv1":           {**_B, "kv": 1},
    "kv8":           {**_B, "kv": 8},
    "pred_layers_2": dict(_B),
    "pred_layers_6": dict(_B),
}


class Wrapper:
    def __init__(self, enc):
        self.target_encoder = enc

    def eval(self):
        self.target_encoder.eval()
        return self


def build_config(spec):
    cfg = VJEPAConfig()
    cfg.use_hybrid = True
    cfg.use_morton_scan = spec["morton"]
    cfg.router_mode = spec["router"]
    cfg.n_kv_heads = spec["kv"]
    cfg.use_rope = spec["rope"]
    cfg.patch_size = spec["patch"]
    return cfg


def latest_ckpt(name):
    ckpts = sorted(glob.glob(os.path.join(CKPT_ROOT, name, "checkpoint_epoch*.pt")))
    best = os.path.join(CKPT_ROOT, name, "checkpoint_best.pt")
    if os.path.exists(best):
        return best
    return ckpts[-1] if ckpts else None


def benchmark_model(name, spec, max_train=5000):
    ckpt_path = latest_ckpt(name)
    if ckpt_path is None:
        print(f"[SKIP] {name}: no checkpoint", flush=True)
        return None

    cfg = build_config(spec)
    encoder = MambaEncoder(cfg).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = encoder.load_state_dict(ckpt["target_encoder"], strict=False)
    if missing or unexpected:
        print(f"  [{name}] load: missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
    model = Wrapper(encoder)
    print(f"[{name}] loaded (epoch {ckpt.get('epoch', '?')}) from {os.path.basename(ckpt_path)}",
          flush=True)

    results = {}
    for qtype, task in [("1", "Plane"), ("2", "Phase"), ("3", "Organ")]:
        train_labels = load_labels(CSV_FILES["train"], qtype)
        test_labels = load_labels(CSV_FILES["val"], qtype)

        tr_paths, tr_labs = [], []
        for img, lab in train_labels.items():
            npy = img_path_to_npy(img, cfg.npy_dir)
            if os.path.exists(npy):
                tr_paths.append(npy); tr_labs.append(lab)
        te_paths, te_labs = [], []
        for img, lab in test_labels.items():
            npy = img_path_to_npy(img, cfg.npy_dir)
            if os.path.exists(npy):
                te_paths.append(npy); te_labs.append(lab)

        common = sorted(set(tr_labs) & set(te_labs))
        tr_paths = [p for p, l in zip(tr_paths, tr_labs) if l in common][:max_train]
        tr_labs = [l for l in tr_labs if l in common][:max_train]
        te_paths = [p for p, l in zip(te_paths, te_labs) if l in common]
        te_labs = [l for l in te_labs if l in common]

        tr_emb = extract_embeddings(model, tr_paths, device="cuda")
        te_emb = extract_embeddings(model, te_paths, device="cuda")
        acc, _, _ = run_knn(tr_emb, tr_labs, te_emb, te_labs)
        results[task] = acc
        print(f"  {task}: kNN={acc:.4f}", flush=True)

    del encoder
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
    print("Hybrid Ablation kNN Results (k=20)")
    print("=" * 70)
    print(f'{"Ablation":<16} {"Plane":>10} {"Phase":>10} {"Organ":>10} {"Mean":>10}')
    print("-" * 58)
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
        print(f'{name:<16} {100*r["Plane"]:>9.1f}% {100*r["Phase"]:>9.1f}% '
              f'{100*r["Organ"]:>9.1f}% {100*mean:>9.1f}%{tag}')


if __name__ == "__main__":
    main()
