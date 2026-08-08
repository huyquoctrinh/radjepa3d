"""M3D Multi-Turn Chain-of-Thought Dataset Generator.

Usage:
    python m3d_cot_gen.py audit
    python m3d_cot_gen.py generate --gpu-ids 6,7 --resume
    python m3d_cot_gen.py generate --gpu-ids 0,1 --shard 0/2 --resume
    python m3d_cot_gen.py generate --gpu-ids 6,7 --shard 1/2 --resume
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import time
from pathlib import Path

from config import OUT_DIR, QTYPE_LABELS
from data_loader import load_csv, audit, group_by_image
from prompts import SYSTEM_PROMPT, make_prompt
from generator import load_model, generate_conversation, parse_json_response
from qc import run_qc, print_qc_report, format_with_think_tags


def make_id(image_path: str) -> str:
    parts = image_path.split("/")
    for p in parts:
        if p.isdigit() or (len(p) == 6 and p.isdigit()):
            return f"m3d_{p}"
    return f"m3d_{abs(hash(image_path)) % 10**8:08d}"


def run_audit():
    print("Stage 0: Auditing dataset...")
    rows = load_csv()
    audit(rows)


def run_generate(limit: int, gpu_ids: str, resume: bool,
                 shard_idx: int, num_shards: int):
    if num_shards > 1:
        out_path = OUT_DIR / f"conversations_shard{shard_idx}.jsonl"
    else:
        out_path = OUT_DIR / "conversations_v1.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if resume and out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_ids)} conversations already done.")

    print("Stage 1: Loading and grouping data...")
    rows = load_csv()
    groups = group_by_image(rows)
    print(f"  {len(groups)} unique images found.")

    # Shard: each worker takes every Nth sample
    if num_shards > 1:
        groups = [g for i, g in enumerate(groups) if i % num_shards == shard_idx]
        print(f"  Shard {shard_idx}/{num_shards}: {len(groups)} images assigned.")

    remaining = [g for g in groups if make_id(g["image_path"]) not in done_ids]
    if limit > 0:
        remaining = remaining[:limit]
    groups = remaining
    total = len(groups)
    print(f"  Processing {total} conversations...")

    print("Stage 2-3: Loading model and generating...")
    load_model(gpu_ids=gpu_ids)

    results = []
    failed = 0
    t0 = time.time()

    with open(out_path, "a") as fout:
        for i, group in enumerate(groups):
            sample_id = make_id(group["image_path"])

            t_start = time.time()
            print(f"\n[{i+1}/{total}] {sample_id} "
                  f"({len(group['turns'])} turns)...", end=" ", flush=True)

            prompt = make_prompt(group)

            try:
                raw = generate_conversation(SYSTEM_PROMPT, prompt)
                conv = parse_json_response(raw)
            except Exception as e:
                print(f"ERROR: {e}")
                raw = ""
                conv = None

            if conv is None:
                print("PARSE FAIL")
                failed += 1
                record = {
                    "id": sample_id,
                    "image_path": group["image_path"],
                    "findings": group["findings"],
                    "impression": group["impression"],
                    "conversations": None,
                    "raw_response": raw,
                    "qc": {"parse_ok": False, "n_turns": 0,
                           "flags": ["json_parse_failed"]},
                }
            else:
                qc = run_qc(conv, group["findings"], group["impression"])
                record = {
                    "id": sample_id,
                    "image_path": group["image_path"],
                    "findings": group["findings"],
                    "impression": group["impression"],
                    "conversations": format_with_think_tags(conv),
                    "qc": qc,
                }
                flag_str = f" flags={qc['flags']}" if qc["flags"] else ""
                elapsed = time.time() - t_start
                print(f"OK ({qc['n_turns']} turns, {elapsed:.1f}s){flag_str}")

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(record)
            done_ids.add(sample_id)

            # Periodic progress
            done_total = i + 1
            if done_total % 100 == 0:
                elapsed_total = time.time() - t0
                rate = done_total / elapsed_total
                eta = (total - done_total) / rate if rate > 0 else 0
                print(f"\n  >> Progress: {done_total}/{total} "
                      f"({100*done_total/total:.1f}%) "
                      f"rate={rate*3600:.0f}/h "
                      f"ETA={eta/3600:.1f}h")

    elapsed_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Generation complete: {len(results)} conversations in {elapsed_total:.0f}s")
    print(f"  Failed to parse: {failed}")
    print(f"  Throughput: {len(results)/elapsed_total*3600:.0f} conv/h")
    print(f"  Output: {out_path}")

    print_qc_report(results)


def main():
    parser = argparse.ArgumentParser(description="M3D CoT Dataset Generator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("audit", help="Stage 0: Audit the dataset")

    gen = sub.add_parser("generate", help="Stages 1-4: Generate conversations")
    gen.add_argument("--limit", type=int, default=0,
                     help="Max conversations to generate (0 = all)")
    gen.add_argument("--gpu-ids", type=str, default="6,7",
                     help="Comma-separated GPU IDs (default: 6,7)")
    gen.add_argument("--shard", type=str, default=None,
                     help="Shard index/total, e.g. 0/2 or 1/2")
    gen.add_argument("--resume", action="store_true",
                     help="Resume from existing output file")

    args = parser.parse_args()

    if args.command == "audit":
        run_audit()
    elif args.command == "generate":
        shard_idx, num_shards = 0, 1
        if args.shard:
            parts = args.shard.split("/")
            shard_idx, num_shards = int(parts[0]), int(parts[1])
        run_generate(args.limit, args.gpu_ids, args.resume,
                     shard_idx, num_shards)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
