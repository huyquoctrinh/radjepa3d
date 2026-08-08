"""Benchmark all ablation checkpoints on kNN + linear probe.

Reconstructs each model's config from its checkpoint dir name and runs the
standard kNN (k=20) benchmark on Plane / Phase / Organ tasks.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_ablations.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_ablations.py --only baseline,mask_block
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import torch

torch.backends.cudnn.enabled = False

from vjepa_config import VJEPAConfig
from vjepa_model import MambaEncoder
from vjepa_benchmark import (
    CSV_FILES, load_labels, img_path_to_npy,
    extract_embeddings, run_knn,
)

CKPT_ROOT = "/data2/gdp2988/3d_cot/checkpoints/ablations"

# Config overrides per ablation (must match run_ablations.sh)
ABLATION_CFG = {
    "baseline":        dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(8, 16, 16), pdim=192, play=4),
    "mask_ratio_50":   dict(morton=False, bidir=False, mask=0.50, strat="random", patch=(8, 16, 16), pdim=192, play=4),
    "mask_ratio_60":   dict(morton=False, bidir=False, mask=0.60, strat="random", patch=(8, 16, 16), pdim=192, play=4),
    "mask_ratio_85":   dict(morton=False, bidir=False, mask=0.85, strat="random", patch=(8, 16, 16), pdim=192, play=4),
    "mask_block":      dict(morton=False, bidir=False, mask=0.75, strat="block",  patch=(8, 16, 16), pdim=192, play=4),
    "bidirectional":   dict(morton=False, bidir=True,  mask=0.75, strat="random", patch=(8, 16, 16), pdim=192, play=4),
    "patch_8_8_8":     dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(8, 8, 8),   pdim=192, play=4),
    "patch_4_16_16":   dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(4, 16, 16), pdim=192, play=4),
    "patch_16_16_16":  dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(16, 16, 16), pdim=192, play=4),
    "pred_dim_96":     dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(8, 16, 16), pdim=96,  play=4),
    "pred_dim_384":    dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(8, 16, 16), pdim=384, play=4),
    "pred_layers_2":   dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(8, 16, 16), pdim=192, play=2),
    "pred_layers_6":   dict(morton=False, bidir=False, mask=0.75, strat="random", patch=(8, 16, 16), pdim=192, play=6),
}


class Wrapper:
    def __init__(self, enc):
        self.target_encoder = enc

    def eval(self):
        self.target_encoder.eval()
        return self


def build_config(spec):
    cfg = VJEPAConfig()
    cfg.use_hybrid = False
    cfg.use_morton_scan = spec["morton"]
    cfg.use_bidirectional = spec["bidir"]
    cfg.mask_ratio = spec["mask"]
    cfg.mask_strategy = spec["strat"]
    cfg.patch_size = spec["patch"]
    cfg.predictor_dim = spec["pdim"]
    cfg.predictor_layers = spec["play"]
    return cfg


def benchmark_model(name, spec, max_train=5000):
    ckpt_path = os.path.join(CKPT_ROOT, name, "checkpoint_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {name}: no checkpoint", flush=True)
        return None

    cfg = build_config(spec)
    encoder = MambaEncoder(cfg).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["target_encoder"])
    model = Wrapper(encoder)
    epoch = ckpt.get("epoch", "?")
    print(f"[{name}] loaded (epoch {epoch})", flush=True)

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
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated ablation names")
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
    print("Ablation kNN Results (k=20)")
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
            delta = mean - bmean
            tag = f"  ({delta:+.3f})"
        print(f'{name:<22} {100*r["Plane"]:>9.1f}% {100*r["Phase"]:>9.1f}% '
              f'{100*r["Organ"]:>9.1f}% {100*mean:>9.1f}%{tag}')


if __name__ == "__main__":
    main()
