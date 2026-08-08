"""Merge sharded open-VQA jsonls and recompute the full metrics table."""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob, json, re
from mllms.benchmark import compute_metrics_batch, bertscore

NAMES = {"1": "Plane", "2": "Phase", "3": "Organ", "4": "Abnormality", "5": "Location"}


def merge(ckpt_dir, label):
    rows = []
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "benchmark_open_shard*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    if not rows:
        print(f"[{label}] no shard files"); return
    # shards are disjoint (samples[shard_id::num_shards]) — no dedup needed
    by = {}
    for r in rows:
        by.setdefault(r["qtype"], []).append(r)

    print("=" * 78)
    print(f"{label}  (merged {len(rows)} samples from shards)")
    print("=" * 78)
    print(f"{'Type':<13}{'N':>6}{'TopExact':>10}{'TopMatch':>10}{'BLEU-1':>9}{'ROUGE-1':>9}{'ROUGE-L':>9}")
    print("-" * 78)
    all_p, all_g = [], []
    te = tm = tn = 0
    for qt in sorted(by):
        rs = by[qt]
        preds = [r["pred"] for r in rs]; gts = [r["gt"] for r in rs]
        all_p += preds; all_g += gts
        m = compute_metrics_batch(preds, gts)
        e = sum(r["exact"] for r in rs); c = sum(r.get("contains", False) for r in rs)
        n = len(rs); te += e; tm += c; tn += n
        print(f"{NAMES.get(qt,qt):<13}{n:>6}{100*e/n:>9.1f}%{100*c/n:>9.1f}%"
              f"{m['bleu1']:>9.3f}{m['rouge1']:>9.3f}{m['rougeL']:>9.3f}")
    ma = compute_metrics_batch(all_p, all_g)
    print("-" * 78)
    print(f"{'Overall':<13}{tn:>6}{100*te/tn:>9.1f}%{100*tm/tn:>9.1f}%"
          f"{ma['bleu1']:>9.3f}{ma['rouge1']:>9.3f}{ma['rougeL']:>9.3f}")
    # BERTScore on top answers
    tops = []
    for r in rows:
        line = r["pred"].strip().splitlines()[0] if r["pred"].strip() else ""
        tops.append(re.sub(r"^[a-dA-D][.\):]\s*", "", line))
    try:
        _, _, bf = bertscore(tops, [r["gt"] for r in rows])
        print(f"BERTScore-F1 (top answer): {bf:.4f}")
    except Exception as e:
        print(f"BERTScore failed: {e}")
    print()


if __name__ == "__main__":
    merge("/data2/gdp2988/3d_cot/checkpoints/mllm", "Qwen3-4B baseline (fixed prompt)")
    merge("/data2/gdp2988/3d_cot/checkpoints/mllm_qwen25_ortho", "Qwen2.5-3B Rad-JEPA (fixed prompt)")
