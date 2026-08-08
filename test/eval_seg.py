"""Evaluate multi-class segmentation per sub-dataset with per-class Dice and IoU.

For each sub-dataset, reports mean foreground-class Dice/IoU (macro over the
classes present in that dataset's test split), then overall averages.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_seg.py
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

torch.backends.cudnn.enabled = False

from seg.config import SegConfig
from seg.model import SegModel
from seg.data import M3DSegSubDataset, collate_fn, num_classes_map
from torch.utils.data import DataLoader


@torch.no_grad()
def per_class_scores(pred, target, num_classes):
    """Return dict {class_idx: (dice, iou)} over foreground classes present."""
    out = {}
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        dice = (2.0 * inter / denom).item()
        iou = (inter / union).item() if union > 0 else 0.0
        out[c] = (dice, iou)
    return out


def main():
    config = SegConfig()
    ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ncmap = ckpt.get("num_classes_by_code") or num_classes_map(config)

    model = SegModel(config, ncmap).cuda().eval()
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"best_dice={ckpt.get('best_dice', float('nan')):.4f})")

    results = []
    tot_dice = tot_iou = 0.0
    tot_n = 0

    for code in config.dataset_codes:
        if code not in ncmap:
            continue
        ds = M3DSegSubDataset(config.data_dir, code, split="test",
                              volume_size=config.volume_size)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_fn)
        C = ncmap[code]

        # accumulate per-class dice/iou across the sub-dataset
        cls_dice = {c: [] for c in range(1, C)}
        cls_iou = {c: [] for c in range(1, C)}
        with torch.no_grad():
            for volumes, labels, _ in loader:
                volumes = volumes.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(volumes, code)
                pred = out["logits"].argmax(dim=1)
                for b in range(volumes.shape[0]):
                    sc = per_class_scores(pred[b], labels[b], C)
                    for cls, (d, i) in sc.items():
                        cls_dice[cls].append(d)
                        cls_iou[cls].append(i)

        present = [c for c in range(1, C) if cls_dice[c]]
        if not present:
            continue
        mean_dice = sum(sum(cls_dice[c]) / len(cls_dice[c]) for c in present) / len(present)
        mean_iou = sum(sum(cls_iou[c]) / len(cls_iou[c]) for c in present) / len(present)
        results.append((code, len(ds), len(present), mean_dice, mean_iou))
        tot_dice += mean_dice * len(ds)
        tot_iou += mean_iou * len(ds)
        tot_n += len(ds)
        print(f"  {code}: n={len(ds)}, fg_classes={len(present)}, "
              f"Dice={mean_dice:.4f}, IoU={mean_iou:.4f}", flush=True)

    print()
    print("=" * 78)
    print(f"{'Code':<6} {'N':>5} {'FGcls':>6} {'Dice':>10} {'IoU':>10}")
    print("-" * 78)
    for code, n, k, dice, iou in results:
        print(f"{code:<6} {n:>5} {k:>6} {dice:>9.4f}  {iou:>9.4f}")
    print("-" * 78)
    if tot_n:
        macro_d = sum(r[3] for r in results) / len(results)
        macro_i = sum(r[4] for r in results) / len(results)
        print(f"{'Wtd':<6} {tot_n:>5} {'':>6} {tot_dice/tot_n:>9.4f}  {tot_iou/tot_n:>9.4f}")
        print(f"{'Macro':<6} {'':>5} {'':>6} {macro_d:>9.4f}  {macro_i:>9.4f}")


if __name__ == "__main__":
    main()
