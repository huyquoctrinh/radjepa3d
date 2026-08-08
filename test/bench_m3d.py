"""Run the M3D frozen-feature kNN benchmark across encoders.

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_m3d.py --encoders radjepa-mamba-ortho,merlin
    CUDA_VISIBLE_DEVICES=0 python bench_m3d.py --encoders all
    python bench_m3d.py --list

Embeddings are cached under checkpoints/embeddings/, so re-running with a
different --k or --knn-weights is nearly free.
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import traceback

import torch

torch.backends.cudnn.enabled = False

import m3d_eval as M
import encoders as E

MERLIN_REPO = os.environ.get(
    "MERLIN_REPO", "/data2/gdp2988/3d_cot/third_party/Merlin")


def encoder_builders(args):
    """name -> zero-arg builder returning extract(paths) -> (N, D)."""
    b = {}
    for name, spec in E.RADJEPA_VARIANTS.items():
        b[name] = lambda s=spec: E.build_radjepa(
            s, batch_size=args.batch_size)
    b["merlin"] = lambda: E.build_merlin(MERLIN_REPO, batch_size=4)
    for name, spec in E.BASELINE_VARIANTS.items():
        b[name] = lambda s=spec: E.build_baseline(s, batch_size=s.get(
            "batch_size", args.batch_size))
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", default="all")
    p.add_argument("--list", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--knn-weights", default="distance",
                   choices=["distance", "uniform"])
    p.add_argument("--max-train", type=int, default=5000)
    p.add_argument("--min-test-count", type=int, default=10)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--no-case-disjoint", action="store_true",
                   help="Reproduce the old leaky protocol for comparison")
    p.add_argument("--out", default=None, help="Write the results table here")
    args = p.parse_args()

    builders = encoder_builders(args)

    if args.list:
        print("Available encoders:")
        for n in builders:
            print(f"  {n}")
        return

    names = (list(builders) if args.encoders == "all"
             else [n.strip() for n in args.encoders.split(",")])
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise SystemExit(f"unknown encoder(s): {unknown}\n"
                        f"available: {list(builders)}")

    tasks = M.build_all_tasks(
        split=args.split,
        case_disjoint=not args.no_case_disjoint,
        min_test_count=args.min_test_count,
        max_train=args.max_train,
    )
    protocol = "leaky (case-overlapping)" if args.no_case_disjoint else "case-disjoint"
    print(f"Protocol: {protocol}, split={args.split}, "
          f"k={args.k}, weights={args.knn_weights}\n")
    for t in tasks:
        print(t.describe())
    print()

    all_results = {}
    for name in names:
        print("=" * 70)
        print(f"Encoder: {name}")
        print("=" * 70, flush=True)
        try:
            extract = builders[name]()
        except Exception:
            print(f"  [SKIP] {name}: failed to build\n{traceback.format_exc()}",
                  flush=True)
            continue

        per_task = {}
        for t in tasks:
            try:
                tr = M.cached_embeddings(name, t.name, "train", t.train_paths,
                                         extract)
                te = M.cached_embeddings(name, t.name, "test", t.test_paths,
                                         extract)
                res = M.evaluate(t, tr, te, k=args.k, weights=args.knn_weights)
                per_task[t.name] = res
                print(f"  {t.name}: acc={100*res['acc']:.2f}% "
                      f"(majority {100*res['majority']:.2f}%, "
                      f"{res['gain_over_majority']*100:+.2f}pt)  "
                      f"bal_acc={100*res['balanced_acc']:.2f}%  "
                      f"macroF1={100*res['macro_f1']:.2f}%", flush=True)
            except Exception:
                print(f"  [FAIL] {t.name}\n{traceback.format_exc()}", flush=True)

        if per_task:
            all_results[name] = per_task
        del extract
        torch.cuda.empty_cache()

    table = M.format_results(all_results, tasks)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"Protocol: {protocol}, split={args.split}, k={args.k}, "
                    f"weights={args.knn_weights}\n")
            for t in tasks:
                f.write(t.describe() + "\n")
            f.write(table + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
