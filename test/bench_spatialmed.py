"""Benchmark on SpatialMed dataset — spatial reasoning QA with 3D volumes.

Usage:
    python bench_spatialmed.py --checkpoint stage2_best.pt --gpu-id 0
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import re
import time

import numpy as np
import torch

torch.backends.cudnn.enabled = False

# Import peft (via model) before monai to avoid torchvision circular import
from mllms.config import MLLMConfig
from mllms.benchmark import load_model, parse_choice
from mllms.spatialmed_taxonomy import classify_metric

import nibabel as nib
import monai.transforms as mtf

BASE_DIR = "/data2/gdp2988/3d_cot/dataset/spatial_med"
JSONL_PATH = os.path.join(BASE_DIR, "spatial_med_v1.jsonl")

VOLUME_MAP = {
    "abdoment": lambda sid: os.path.join(BASE_DIR, "total_segmentator", sid, "ct.nii.gz"),
    "amos": lambda sid: os.path.join(BASE_DIR, "amos/amos22/imagesTr", f"{sid}.nii"),
    "brats": lambda sid: os.path.join(BASE_DIR, "brats/training_data", sid, f"{sid}_t1ce.nii.gz"),
    "kits": lambda sid: os.path.join(BASE_DIR, "kits/kits23/dataset", sid, "imaging.nii.gz"),
    "liver": lambda sid: os.path.join(BASE_DIR, "liver_decalthon/Task03_Liver/imagesTr", f"{sid}.nii.gz"),
    "pancreas": lambda sid: os.path.join(BASE_DIR, "pancreas_decalthon/Task07_Pancreas/imagesTr", f"{sid}.nii.gz"),
    "vessel": lambda sid: os.path.join(BASE_DIR, "vessel_decalthon/Task08_HepaticVessel/imagesTr", f"{sid}.nii.gz"),
    "prostate": lambda sid: os.path.join(BASE_DIR, "prostate_decalthon/Task05_Prostate/imagesTr", f"{sid}.nii.gz"),
}

transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear"),
])


def load_nifti_volume(path):
    """Load NIfTI, normalize, resize to (1, 32, 256, 256)."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # Handle multi-channel (take first channel if 4D)
    if data.ndim == 4:
        data = data[:, :, :, 0]

    # Transpose to (D, H, W) if needed — NIfTI is typically (H, W, D)
    if data.shape[2] > data.shape[0]:
        data = data.transpose(2, 0, 1)

    # Add channel dim: (1, D, H, W)
    data = data[np.newaxis, ...]

    # Normalize
    data = data - data.min()
    data = data / np.clip(data.max(), a_min=1e-8, a_max=None)

    # Resize
    data = transform(data)
    return torch.as_tensor(np.array(data), dtype=torch.float32)


def get_volume_path(stype, sid):
    mapper = VOLUME_MAP.get(stype)
    if mapper is None:
        return None
    path = mapper(sid)
    if not os.path.exists(path):
        # Try with .nii.gz if .nii doesn't exist
        if not path.endswith(".gz"):
            alt = path + ".gz"
            if os.path.exists(alt):
                return alt
        return None
    return path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--llm-name", type=str, default=None)
    parser.add_argument("--llm-dim", type=int, default=None)
    parser.add_argument("--vision-checkpoint", type=str, default=None)
    parser.add_argument("--use-hybrid", dest="use_hybrid", action="store_true", default=None)
    parser.add_argument("--no-hybrid", dest="use_hybrid", action="store_false")
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
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config)

    with open(JSONL_PATH) as f:
        samples = [json.loads(l) for l in f]
    print(f"Loaded {len(samples)} samples")

    # Cache loaded volumes
    vol_cache = {}
    correct_by_type = {}
    total_by_type = {}
    skipped_by_type = {}
    correct_by_tax = {}
    total_by_tax = {}
    correct_by_metric = {}   # canonical taxonomy (spatialmed_taxonomy.classify_metric)
    total_by_metric = {}
    per_sample = []          # saved predictions for offline re-analysis

    t0 = time.time()
    count = 0

    for i, s in enumerate(samples):
        if args.max_samples > 0 and count >= args.max_samples:
            break

        stype = s["type"]
        sid = s["id"]
        gt = s["gt_answers"].strip()

        cache_key = (stype, sid)
        if cache_key not in vol_cache:
            vpath = get_volume_path(stype, sid)
            if vpath is None:
                vol_cache[cache_key] = None
            else:
                try:
                    vol_cache[cache_key] = load_nifti_volume(vpath)
                except Exception as e:
                    vol_cache[cache_key] = None

        vol_tensor = vol_cache[cache_key]
        if vol_tensor is None:
            skipped_by_type[stype] = skipped_by_type.get(stype, 0) + 1
            continue

        vol = vol_tensor.unsqueeze(0).cuda()

        question = s["questions"].strip()
        prompt = (
            f"Based on the 3D medical scan shown above, answer the following question.\n"
            f"{question}\n"
            f"Answer with the correct option letter."
        )

        pred_text = model.generate(vol, prompt, max_new_tokens=16)
        pred_letter = parse_choice(pred_text)

        is_correct = pred_letter == gt

        # Taxonomy classification
        q_lower = question.lower()
        if 'volume' in q_lower and 'ratio' in q_lower:
            tax = 'Volume Ratio'
        elif 'volume' in q_lower and any(w in q_lower for w in ['difference','percentage','larger','smaller','significantly','greater','compare']):
            tax = 'Volume Comparison'
        elif 'volume' in q_lower:
            tax = 'Volume Other'
        elif any(w in q_lower for w in ['bounding box','depth of','width of','height of']):
            tax = 'Size/BBox'
        elif any(w in q_lower for w in ['superior','inferior','anterior','posterior','located','location','position','above','below']):
            tax = 'Spatial Position'
        elif any(w in q_lower for w in ['distance','far','close','nearest','proximity']):
            tax = 'Distance'
        elif any(w in q_lower for w in ['fraction','occupi','overlap','proportion']):
            tax = 'Spatial Overlap'
        elif any(w in q_lower for w in ['change','growth','increase','decrease']):
            tax = 'Temporal Change'
        else:
            tax = 'Other'

        correct_by_type[stype] = correct_by_type.get(stype, 0) + int(is_correct)
        total_by_type[stype] = total_by_type.get(stype, 0) + 1

        if tax not in correct_by_tax:
            correct_by_tax[tax] = 0
            total_by_tax[tax] = 0
        correct_by_tax[tax] += int(is_correct)
        total_by_tax[tax] += 1

        # Canonical SpatialMed taxonomy metric
        metric = classify_metric(question)
        correct_by_metric[metric] = correct_by_metric.get(metric, 0) + int(is_correct)
        total_by_metric[metric] = total_by_metric.get(metric, 0) + 1

        per_sample.append({
            "id": sid, "type": stype, "question": question,
            "gt": gt, "pred": pred_letter, "pred_text": pred_text,
            "correct": bool(is_correct), "tax": tax, "metric": metric,
        })

        count += 1

        if count % 200 == 0:
            total_c = sum(correct_by_type.values())
            total_t = sum(total_by_type.values())
            print(f"  [{count}] acc={100*total_c/total_t:.1f}%", flush=True)

    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"SpatialMed Benchmark ({count} samples, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"{'Type':<15} {'Correct':>8} {'Total':>8} {'Acc':>8} {'Skipped':>8}")
    print(f"{'-'*50}")

    total_c = total_t = 0
    for stype in sorted(total_by_type):
        c = correct_by_type.get(stype, 0)
        t = total_by_type[stype]
        sk = skipped_by_type.get(stype, 0)
        total_c += c
        total_t += t
        print(f"{stype:<15} {c:>8} {t:>8} {100*c/t:>7.1f}% {sk:>8}")

    print(f"{'-'*50}")
    total_sk = sum(skipped_by_type.values())
    print(f"{'Overall':<15} {total_c:>8} {total_t:>8} {100*total_c/total_t:>7.1f}% {total_sk:>8}")

    # Taxonomy breakdown
    print(f"\n{'='*60}")
    print(f"By Question Taxonomy")
    print(f"{'='*60}")
    print(f"{'Category':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"{'-'*48}")
    tax_c = tax_t = 0
    for tax in sorted(total_by_tax, key=lambda x: -total_by_tax[x]):
        c = correct_by_tax[tax]
        t = total_by_tax[tax]
        tax_c += c
        tax_t += t
        print(f"{tax:<20} {c:>8} {t:>8} {100*c/t:>7.1f}%")
    print(f"{'-'*48}")
    print(f"{'Overall':<20} {tax_c:>8} {tax_t:>8} {100*tax_c/tax_t:>7.1f}%")

    # Canonical SpatialMed taxonomy (classify_metric)
    print(f"\n{'='*60}")
    print(f"Canonical Taxonomy (classify_metric)")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"{'-'*40}")
    order = ["DIR", "DIST", "EXT", "VOL", "COMP", "Unknown"]
    m_c = m_t = 0
    for m in order:
        if m not in total_by_metric:
            continue
        c = correct_by_metric.get(m, 0)
        t = total_by_metric[m]
        m_c += c
        m_t += t
        print(f"{m:<12} {c:>8} {t:>8} {100*c/t:>7.2f}%")
    print(f"{'-'*40}")
    print(f"{'Overall':<12} {m_c:>8} {m_t:>8} {100*m_c/m_t:>7.2f}%")

    # Save per-sample predictions for offline re-analysis
    out_dir = os.path.dirname(args.checkpoint) or "."
    out_path = os.path.join(out_dir, "benchmark_spatialmed.jsonl")
    with open(out_path, "w") as f:
        for r in per_sample:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-sample results saved to {out_path}")


if __name__ == "__main__":
    main()
