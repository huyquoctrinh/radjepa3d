"""Re-run every ablation under one consistent protocol.

The three existing ablation logs cannot be compared, to each other or to the
main benchmark, for four independent reasons:

  1. Epoch mismatch. Each suite reported `checkpoint_best.pt`, which lands
     wherever validation loss bottomed -- ep55 to ep93 within a single table.
     In ablations_hybrid the baseline (ep55) is the *least* trained model, and
     every delta is measured against it. Here every model in a suite is loaded
     at the same epoch instead.
  2. Protocol. Those runs used the leaky case-overlapping split, no label
     case-normalization, and uniform kNN weights.
  3. Metric. They reported accuracy with no majority baseline, and a "Mean"
     column averaging tasks whose chance levels are 91/35/15 -- roughly 90% a
     restatement of Plane.
  4. Architecture. Suite A/B are MambaEncoder; Suite C is the hybrid3d
     late-fusion pair. Cross-suite deltas were never meaningful.

Encoder config is inferred from the checkpoint's own tensors rather than from
the directory name, because directory names have already proven unreliable
(vjepa_hybrid_layer is raster-scan despite the surrounding config defaulting to
Morton). Only router_mode layer-vs-token is not inferable -- both produce
identical state dicts and differ solely in forward() -- so it is the one field
taken from the ablation name.

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_ablations_clean.py --suite A
    CUDA_VISIBLE_DEVICES=0 python bench_ablations_clean.py --suite A,B --epoch 99
"""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import os
import traceback

import torch

torch.backends.cudnn.enabled = False

import m3d_eval as M
import encoders as E
from load_checkpoint import (
    infer_encoder_config as infer_config,
    describe_config as describe,
    load_encoder,
)

CKPT = "/data2/gdp2988/3d_cot/checkpoints"

SUITES = {
    "A": dict(root=f"{CKPT}/ablations", epoch=99, hybrid=False,
              label="Design A -- Mamba"),
    "B": dict(root=f"{CKPT}/ablations_hybrid", epoch=79, hybrid=True,
              label="Design A -- Hybrid (Mamba+GQA)"),
}

# router_mode is the only field not recoverable from the weights.
ROUTER_MODE = {"router_token": "token", "router_fixed": "fixed"}


def build(ckpt_path, name, batch_size, device="cuda"):
    router_mode = ROUTER_MODE.get(name, "layer")
    enc, cfg, meta = load_encoder(
        ckpt_path, device=device, router_mode=router_mode,
        batch_size=batch_size)

    def forward(batch):
        return enc(batch, indices=None).mean(dim=1)

    def extract(paths):
        return E.batched_extract(forward, paths, batch_size, device, label=name)

    return extract, cfg, meta.get("epoch", "?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="A,B")
    p.add_argument("--epoch", type=int, default=None,
                   help="Override the matched epoch for every suite")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=f"{CKPT}/ablations_clean.json")
    args = p.parse_args()

    tasks = M.build_all_tasks()
    print("Protocol: case-disjoint, val split, k=20 cosine, distance-weighted\n")
    for t in tasks:
        print(t.describe())
    print()

    import json
    results = {}
    for key in [s.strip() for s in args.suite.split(",")]:
        suite = SUITES[key]
        epoch = args.epoch if args.epoch is not None else suite["epoch"]
        root = suite["root"]
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
        print("=" * 74)
        print(f"Suite {key}: {suite['label']}  (all models at epoch {epoch})")
        print("=" * 74, flush=True)

        for name in names:
            ckpt_path = os.path.join(root, name, f"checkpoint_epoch{epoch:04d}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] {name}: no checkpoint at epoch {epoch}", flush=True)
                continue
            tag = f"{key}/{name}"
            try:
                extract, cfg, ep = build(ckpt_path, name, args.batch_size)
            except Exception:
                print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
                continue
            print(f"[{tag}] epoch={ep} :: {describe(cfg)}", flush=True)

            per_task = {}
            for t in tasks:
                cache_key = f"abl-{key}-{name}-ep{epoch}"
                tr = M.cached_embeddings(cache_key, t.name, "train",
                                         t.train_paths, extract)
                te = M.cached_embeddings(cache_key, t.name, "test",
                                         t.test_paths, extract)
                knn = M.run_knn(tr, t.train_labels, te, t.test_labels)
                knn.pop("preds", None)
                per_task[t.name] = {"knn": knn, "dim": int(tr.shape[1])}
                print(f"   {t.name}: acc={100*knn['acc']:.2f}% "
                      f"bal={100*knn['balanced_acc']:.2f}%", flush=True)
            results[tag] = {"config": describe(cfg), "epoch": epoch,
                            **per_task}
            del extract
            torch.cuda.empty_cache()

        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
