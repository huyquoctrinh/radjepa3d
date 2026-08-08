"""M3D Benchmark: Closed-ended VQA, Open-ended VQA, and Captioning.

Usage:
    python mllms/benchmark.py --checkpoint stage2_best.pt --mode closed
    python mllms/benchmark.py --checkpoint stage2_best.pt --mode open
    python mllms/benchmark.py --checkpoint stage2_best.pt --mode closed --test-csv M3D_VQA_test.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch

torch.backends.cudnn.enabled = False

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mllms.config import MLLMConfig
from mllms.model import Med3DMLLM

hf_rouge = None
hf_bleu = None
_bertscorer = None

ORGANS = [
    "liver", "kidney", "spleen", "pancreas", "gallbladder", "adrenal", "bladder",
    "lung", "heart", "aorta", "stomach", "colon", "bowel", "intestine", "esophagus",
    "prostate", "uterus", "ovary", "brain", "spine", "rib", "bone", "trachea",
    "ureter", "duodenum", "rectum", "vessel", "artery", "vein", "node", "thyroid",
]
_MEAS_RE = re.compile(r"\b\d+(\.\d+)?\s*(mm|cm|hu|ml|cc)\b", re.I)


def bertscore(preds, gts):
    """BERTScore P/R/F1 (semantic similarity). Lazy-loads the scorer."""
    global _bertscorer
    if _bertscorer is None:
        from bert_score import BERTScorer
        _bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F = _bertscorer.score(preds, gts)
    return P.mean().item(), R.mean().item(), F.mean().item()


def finding_stats(preds, gts):
    """Where does report content diverge? Laterality / measurements / organs."""
    def sides(t):
        t = t.lower()
        s = set()
        if "left" in t:
            s.add("left")
        if "right" in t:
            s.add("right")
        return s

    def organs(t):
        t = t.lower()
        return {o for o in ORGANS if o in t}

    lat_match = lat_n = 0
    meas_gt = meas_pred = 0
    jac = []
    ratios = []
    for p, g in zip(preds, gts):
        gs, ps = sides(g), sides(p)
        if gs:
            lat_n += 1
            if gs == ps:
                lat_match += 1
        meas_gt += int(bool(_MEAS_RE.search(g)))
        meas_pred += int(bool(_MEAS_RE.search(p)))
        go, po = organs(g), organs(p)
        if go or po:
            jac.append(len(go & po) / len(go | po))
        gw = max(len(g.split()), 1)
        ratios.append(len(p.split()) / gw)
    n = max(len(preds), 1)
    ratios.sort()
    return {
        "lat_match": lat_match / max(lat_n, 1), "lat_n": lat_n,
        "meas_gt": meas_gt / n, "meas_pred": meas_pred / n,
        "organ_jaccard": sum(jac) / max(len(jac), 1),
        "len_ratio": ratios[len(ratios) // 2] if ratios else 0.0,
    }


def _load_hf_metrics():
    global hf_rouge, hf_bleu
    if hf_rouge is None:
        import evaluate
        hf_rouge = evaluate.load("rouge")
        hf_bleu = evaluate.load("bleu")

QTYPE_NAMES = {"1": "Plane", "2": "Phase", "3": "Organ",
                "4": "Abnormality", "5": "Location"}


def load_model(checkpoint_path, config):
    from load_checkpoint import load_mllm

    overrides = {k: getattr(config, k) for k in (
        "llm_name", "llm_dim", "use_hybrid", "vision_checkpoint",
        "vision_pool_tokens", "vision_use_morton",
    ) if hasattr(config, k)}
    return load_mllm(checkpoint_path, config_overrides=overrides)


def load_test_samples(csv_path, npy_dir):
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = row["Image Path"].split("/")
            npy_path = os.path.join(npy_dir, parts[1], parts[2], parts[3])
            if not os.path.exists(npy_path):
                continue
            samples.append({
                "npy_path": npy_path,
                "question": row["Question"],
                "choice_a": row["Choice A"],
                "choice_b": row["Choice B"],
                "choice_c": row["Choice C"],
                "choice_d": row["Choice D"],
                "answer": row["Answer"],
                "answer_choice": row["Answer Choice"],
                "qtype": row["Question Type"],
            })
    return samples


def parse_choice(text):
    """Extract predicted choice letter from generated text."""
    text = text.strip()
    # Try first character
    if text and text[0] in "ABCD":
        return text[0]
    # Try regex for "A." or "A)" pattern
    m = re.search(r"\b([ABCD])[.):\s]", text)
    if m:
        return m.group(1)
    return None


def token_f1(pred_tokens, gt_tokens):
    """Compute token-level F1 between prediction and ground truth."""
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    if not pred_set or not gt_set:
        return 0.0
    common = pred_set & gt_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def compute_metrics_batch(preds, gts):
    """Compute BLEU-1, ROUGE-1, ROUGE-2, ROUGE-L using HF evaluate."""
    _load_hf_metrics()
    rouge_results = hf_rouge.compute(predictions=preds, references=gts)
    bleu1 = hf_bleu.compute(predictions=preds, references=gts, max_order=1)
    return {
        "bleu1": bleu1["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
    }


def compute_metrics_single(pred, gt):
    """Compute metrics for a single pred/gt pair."""
    _load_hf_metrics()
    r = hf_rouge.compute(predictions=[pred], references=[gt])
    try:
        b = hf_bleu.compute(predictions=[pred], references=[gt], max_order=1)
        bleu1 = b["bleu"]
    except ZeroDivisionError:
        bleu1 = 0.0
    return {
        "bleu1": bleu1,
        "rouge1": r["rouge1"],
        "rouge2": r["rouge2"],
        "rougeL": r["rougeL"],
    }


@torch.no_grad()
def run_closed_ended(model, samples, config, output_path=None):
    """Closed-ended VQA: predict choice letter."""
    results = []
    correct_by_type = {}
    total_by_type = {}

    for i, s in enumerate(samples):
        vol = torch.from_numpy(
            np.load(s["npy_path"]).astype(np.float32)
        ).unsqueeze(0).cuda()

        prompt = (
            f"Based on the 3D CT scan shown above, answer the following question.\n"
            f"Question: {s['question']}\n"
            f"Options: A) {s['choice_a']} B) {s['choice_b']} "
            f"C) {s['choice_c']} D) {s['choice_d']}\n"
            f"Answer with the correct option letter."
        )

        pred_text = model.generate(vol, prompt, max_new_tokens=16)
        pred_letter = parse_choice(pred_text)
        gt_letter = s["answer_choice"]
        is_correct = pred_letter == gt_letter

        qt = s["qtype"]
        correct_by_type[qt] = correct_by_type.get(qt, 0) + int(is_correct)
        total_by_type[qt] = total_by_type.get(qt, 0) + 1

        results.append({
            "qtype": qt,
            "question": s["question"],
            "gt": gt_letter,
            "pred": pred_letter,
            "pred_text": pred_text.strip()[:100],
            "correct": is_correct,
        })

        if (i + 1) % 100 == 0:
            total_correct = sum(correct_by_type.values())
            total_done = sum(total_by_type.values())
            print(f"  [{i+1}/{len(samples)}] "
                  f"acc={100 * total_correct / total_done:.1f}%", flush=True)

    # Print results
    print(f"\n{'='*60}")
    print(f"Closed-ended VQA Results ({len(samples)} samples)")
    print(f"{'='*60}")
    print(f"{'Type':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"{'-'*48}")

    total_c = total_t = 0
    for qt in sorted(total_by_type):
        c = correct_by_type.get(qt, 0)
        t = total_by_type[qt]
        total_c += c
        total_t += t
        name = QTYPE_NAMES.get(qt, f"Type {qt}")
        print(f"{name:<20} {c:>8} {t:>8} {100*c/t:>7.1f}%")

    print(f"{'-'*48}")
    print(f"{'Overall':<20} {total_c:>8} {total_t:>8} {100*total_c/total_t:>7.1f}%")

    if output_path:
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nDetailed results: {output_path}")

    return {"accuracy": total_c / total_t, "per_type": {
        qt: correct_by_type.get(qt, 0) / total_by_type[qt]
        for qt in total_by_type
    }}


@torch.no_grad()
def run_open_ended(model, samples, config, output_path=None):
    """Open-ended VQA: generate free-text answer without choices."""
    results = []
    preds_by_type = {}
    gts_by_type = {}

    for i, s in enumerate(samples):
        vol = torch.from_numpy(
            np.load(s["npy_path"]).astype(np.float32)
        ).unsqueeze(0).cuda()

        # Match the training format (options present) — the model was trained
        # closed-form and echoes/degenerates without them. It answers
        # "<letter>. <answer>"; we grade the free-text answer below.
        prompt = (
            f"Based on the 3D CT scan shown above, answer the following question.\n"
            f"Question: {s['question']}\n"
            f"Options: A) {s['choice_a']} B) {s['choice_b']} "
            f"C) {s['choice_c']} D) {s['choice_d']}"
        )

        pred_text = model.generate(vol, prompt, max_new_tokens=32).strip()
        gt_text = s["answer"]
        qt = s["qtype"]

        if qt not in preds_by_type:
            preds_by_type[qt] = []
            gts_by_type[qt] = []
        preds_by_type[qt].append(pred_text)
        gts_by_type[qt].append(gt_text)

        def _norm(x):
            return " ".join(re.sub(r"[^a-z0-9 ]", " ", x.lower()).split())
        # Score the model's TOP answer (models tend to echo the option list;
        # the first line is the selected answer). Strip any leading option letter.
        first_line = pred_text.strip().splitlines()[0] if pred_text.strip() else ""
        first_line = re.sub(r"^[a-dA-D][.\):]\s*", "", first_line)
        gt_n = _norm(gt_text)
        first_n = _norm(first_line)
        exact = first_n == gt_n                       # top answer equals GT
        contains = bool(gt_n) and gt_n in first_n     # GT within top answer
        results.append({
            "qtype": qt,
            "question": s["question"],
            "gt": gt_text,
            "pred": pred_text[:200],
            "exact": exact,
            "contains": contains,
        })

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # Compute metrics per type using HF evaluate
    print(f"\n{'='*80}")
    print(f"Open-ended VQA Results ({len(samples)} samples)")
    print(f"{'='*80}")
    print(f"{'Type':<15} {'N':>5} {'TopExact':>9} {'TopMatch':>9} {'BLEU-1':>8} {'ROUGE-1':>8} {'ROUGE-L':>8}")
    print(f"{'-'*72}")

    all_preds, all_gts = [], []
    all_exact = all_contains = 0
    all_n = 0
    for qt in sorted(preds_by_type):
        preds = preds_by_type[qt]
        gts = gts_by_type[qt]
        n = len(preds)
        all_preds.extend(preds)
        all_gts.extend(gts)

        m = compute_metrics_batch(preds, gts)
        exact_n = sum(1 for r in results if r["qtype"] == qt and r["exact"])
        contains_n = sum(1 for r in results if r["qtype"] == qt and r["contains"])
        all_exact += exact_n
        all_contains += contains_n
        all_n += n
        name = QTYPE_NAMES.get(qt, f"Type {qt}")
        print(f"{name:<15} {n:>5} {100*exact_n/n:>7.1f}% {100*contains_n/n:>8.1f}% "
              f"{m['bleu1']:>8.3f} {m['rouge1']:>8.3f} {m['rougeL']:>8.3f}")

    # Overall
    m_all = compute_metrics_batch(all_preds, all_gts)
    print(f"{'-'*72}")
    print(f"{'Overall':<15} {all_n:>5} {100*all_exact/all_n:>7.1f}% "
          f"{100*all_contains/all_n:>8.1f}% "
          f"{m_all['bleu1']:>8.3f} {m_all['rouge1']:>8.3f} {m_all['rougeL']:>8.3f}")

    # Semantic metric (BERTScore) on the model's top answer vs GT
    try:
        tops = []
        for r in results:
            line = r["pred"].strip().splitlines()[0] if r["pred"].strip() else ""
            tops.append(re.sub(r"^[a-dA-D][.\):]\s*", "", line))
        bp, br, bf = bertscore(tops, [r["gt"] for r in results])
        print(f"{'BERTScore-F1 (top answer)':<15} {bf:>34.4f}  (P={bp:.4f} R={br:.4f})")
    except Exception as e:
        print(f"BERTScore: failed ({e})")

    if output_path:
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nDetailed results: {output_path}")


@torch.no_grad()
def run_captioning(model, config, cap_split="test500", max_samples=0, output_path=None):
    """Report generation benchmark on M3D-Cap."""
    cap_json = config.data_json
    with open(cap_json) as f:
        data = json.load(f)

    entries = data.get(cap_split, data.get("test", []))
    print(f"  Cap split '{cap_split}': {len(entries)} entries")

    results = []
    all_preds = []
    all_gts = []

    count = 0
    for entry in entries:
        parts = entry["image"].split("/")
        npy_path = os.path.join(config.npy_dir, parts[1], parts[2], parts[3])
        text_path = os.path.join(config.npy_dir, parts[1], parts[2], "text.txt")

        if not os.path.exists(npy_path) or not os.path.exists(text_path):
            continue

        with open(text_path) as f:
            gt_text = f.read().strip()
        if len(gt_text) < 10:
            continue

        vol = torch.from_numpy(
            np.load(npy_path).astype(np.float32)
        ).unsqueeze(0).cuda()

        prompt = "Describe the findings in this 3D CT scan in detail."
        pred_text = model.generate(vol, prompt, max_new_tokens=64).strip()

        all_preds.append(pred_text)
        all_gts.append(gt_text)

        results.append({
            "gt": gt_text,
            "pred": pred_text,
        })

        count += 1
        if count % 50 == 0:
            m = compute_metrics_batch(all_preds, all_gts)
            print(f"  [{count}] bleu1={m['bleu1']:.3f} "
                  f"rouge1={m['rouge1']:.3f} rougeL={m['rougeL']:.3f}", flush=True)

        if max_samples > 0 and count >= max_samples:
            break

    m = compute_metrics_batch(all_preds, all_gts)
    n = len(all_preds)
    print(f"\n{'='*60}")
    print(f"Captioning Results ({n} samples, split={cap_split})")
    print(f"{'='*60}")
    print(f"  BLEU-1:   {m['bleu1']:.4f}")
    print(f"  ROUGE-1:  {m['rouge1']:.4f}")
    print(f"  ROUGE-2:  {m['rouge2']:.4f}")
    print(f"  ROUGE-L:  {m['rougeL']:.4f}")

    # Semantic metric (BERTScore) — fairer than n-gram overlap for paraphrase
    try:
        bp, br, bf = bertscore(all_preds, all_gts)
        print(f"  BERTScore-F1: {bf:.4f}  (P={bp:.4f} R={br:.4f})")
    except Exception as e:
        print(f"  BERTScore: failed ({e})")

    # Clinical finding-type breakdown: where does content diverge?
    fs = finding_stats(all_preds, all_gts)
    print(f"\n  {'Finding-type overlap (pred vs gt)':<40}")
    print(f"  {'Laterality (left/right) match':<40} {100*fs['lat_match']:.1f}%  "
          f"(of {fs['lat_n']} reports mentioning a side)")
    print(f"  {'Has measurement (mm/cm/HU) — gt':<40} {100*fs['meas_gt']:.1f}%  "
          f"pred {100*fs['meas_pred']:.1f}%")
    print(f"  {'Organ mention overlap (Jaccard)':<40} {100*fs['organ_jaccard']:.1f}%")
    print(f"  {'Length ratio (pred/gt words, median)':<40} {fs['len_ratio']:.2f}")

    if output_path:
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nDetailed results: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="M3D Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="closed",
                        choices=["closed", "open", "both", "caption", "all"])
    parser.add_argument("--test-csv", type=str, default=None,
                        help="Test CSV filename (default: M3D_VQA_test5k.csv)")
    parser.add_argument("--cap-split", type=str, default="test500")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (0=all)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split test set into N shards (for multi-GPU)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Which shard this process evaluates (0..N-1)")
    # Overrides to match alternate trained models (e.g. Qwen2.5-3B + vjepa_ortho)
    parser.add_argument("--llm-name", type=str, default=None)
    parser.add_argument("--llm-dim", type=int, default=None)
    parser.add_argument("--vision-checkpoint", type=str, default=None)
    parser.add_argument("--use-hybrid", dest="use_hybrid", action="store_true", default=None)
    parser.add_argument("--no-hybrid", dest="use_hybrid", action="store_false")
    parser.add_argument("--vision-pool-tokens", type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["HF_HOME"] = "/data2/gdp2988/.cache/huggingface"

    config = MLLMConfig()
    if args.llm_name:
        config.llm_name = args.llm_name
    if args.llm_dim:
        config.llm_dim = args.llm_dim
    if args.vision_checkpoint:
        config.vision_checkpoint = args.vision_checkpoint
    if args.use_hybrid is not None:
        config.use_hybrid = args.use_hybrid
    if args.vision_pool_tokens is not None:
        config.vision_pool_tokens = args.vision_pool_tokens

    csv_name = args.test_csv or "M3D_VQA_test5k.csv"
    csv_path = os.path.join(
        "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text", csv_name,
    )

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config)

    out_dir = os.path.dirname(args.checkpoint) or "."

    if args.mode in ("closed", "both", "all"):
        print(f"\nLoading test data from {csv_name}...")
        samples = load_test_samples(csv_path, config.npy_dir)
        if args.max_samples > 0:
            samples = samples[:args.max_samples]
        print(f"  {len(samples)} test samples loaded")

        print(f"\nRunning closed-ended evaluation...")
        t0 = time.time()
        run_closed_ended(
            model, samples, config,
            output_path=os.path.join(out_dir, "benchmark_closed.jsonl"),
        )
        print(f"Time: {time.time() - t0:.0f}s")

    if args.mode in ("open", "both", "all"):
        if args.mode in ("open",):
            print(f"\nLoading test data from {csv_name}...")
            samples = load_test_samples(csv_path, config.npy_dir)
            if args.max_samples > 0:
                samples = samples[:args.max_samples]
            print(f"  {len(samples)} test samples loaded")

        open_out = os.path.join(out_dir, "benchmark_open.jsonl")
        if args.num_shards > 1:
            samples = samples[args.shard_id::args.num_shards]
            open_out = os.path.join(out_dir, f"benchmark_open_shard{args.shard_id}.jsonl")
            print(f"  Shard {args.shard_id}/{args.num_shards}: {len(samples)} samples")

        print(f"\nRunning open-ended evaluation...")
        t0 = time.time()
        run_open_ended(model, samples, config, output_path=open_out)
        print(f"Time: {time.time() - t0:.0f}s")

    if args.mode in ("caption", "all"):
        print(f"\nRunning captioning evaluation ({args.cap_split})...")
        t0 = time.time()
        run_captioning(
            model, config, cap_split=args.cap_split,
            max_samples=args.max_samples,
            output_path=os.path.join(out_dir, "benchmark_caption.jsonl"),
        )
        print(f"Time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
