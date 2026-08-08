"""Training for 3D multi-class semantic segmentation on M3D-Seg (multi-GPU).

Per-dataset heads: every step all ranks draw from ONE sub-dataset (via
DistSameDatasetBatchSampler), so a single taxonomy-specific head applies on all
ranks. Eval runs on rank 0 over the full test set for exact coverage.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,5 torchrun --nproc_per_node=4 training/train_seg.py
    CUDA_VISIBLE_DEVICES=0 python training/train_seg.py --gpu-id 0   # single-GPU
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import math
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

from seg.config import SegConfig
from seg.model import SegModel
from seg.data import (
    build_seg_dataset, collate_fn, num_classes_map,
    SameDatasetBatchSampler, DistSameDatasetBatchSampler,
)


def setup_distributed():
    if "RANK" in os.environ:
        # Long timeout: rank-0 eval over the full test set keeps other ranks
        # idle at the post-eval barrier, which must not trip the NCCL watchdog.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    torch.cuda.set_device(0)
    return 0, 1, False


@torch.no_grad()
def batch_dice(pred, target, num_classes):
    """Mean foreground Dice for one batch. pred/target: (B, D, H, W) int."""
    dices = []
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        inter = (p & t).sum()
        dices.append((2.0 * inter / denom).item())
    return sum(dices) / len(dices) if dices else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.gpu_id and "RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    local_rank, world_size, distributed = setup_distributed()
    is_main = local_rank == 0

    config = SegConfig()
    if args.epochs:
        config.total_epochs = args.epochs

    if is_main:
        print("3D Multi-class Segmentation Training (M3D-Seg)")
        print(f"  Distributed: {distributed}, World size: {world_size}")

    ncmap = num_classes_map(config)
    if is_main:
        print(f"  Sub-datasets: {len(ncmap)}, classes range "
              f"{min(ncmap.values())}-{max(ncmap.values())}")

    model = SegModel(config, ncmap)
    if os.path.exists(config.vision_checkpoint):
        model.load_pretrained_encoder(config.vision_checkpoint)
    model = model.cuda()

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_unwrapped = model.module if distributed else model

    if is_main:
        n_enc = sum(p.numel() for p in model_unwrapped.encoder.parameters()) / 1e6
        n_dec = sum(p.numel() for p in model_unwrapped.decoder.parameters()) / 1e6
        n_head = sum(p.numel() for p in model_unwrapped.heads.parameters()) / 1e6
        print(f"  Encoder: {n_enc:.1f}M, Decoder: {n_dec:.1f}M, Heads: {n_head:.1f}M")
        print("Loading datasets...")

    train_dataset = build_seg_dataset(config, split="train")
    test_dataset = build_seg_dataset(config, split="test") if is_main else None

    if distributed:
        train_sampler = DistSameDatasetBatchSampler(
            train_dataset, config.batch_size, world_size, local_rank,
            shuffle=True,
        )
    else:
        train_sampler = SameDatasetBatchSampler(
            train_dataset, config.batch_size, shuffle=True, drop_last=True,
        )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    test_loader = None
    if is_main:
        test_sampler = SameDatasetBatchSampler(
            test_dataset, config.batch_size, shuffle=False, drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_sampler=test_sampler,
            num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn,
        )

    param_groups = [
        {"params": model_unwrapped.encoder.parameters(), "lr": config.lr_encoder},
        {"params": model_unwrapped.decoder.parameters(), "lr": config.lr},
        {"params": model_unwrapped.heads.parameters(), "lr": config.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_sampler)
    total_steps = steps_per_epoch * config.total_epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if is_main:
        print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}\n")

    best_dice = 0.0
    global_step = 0

    for epoch in range(config.total_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = epoch_dice = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_idx, (volumes, labels, code) in enumerate(train_loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            volumes = volumes.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(volumes, code, labels)
                loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                dice = batch_dice(out["logits"].argmax(dim=1), labels, out["num_classes"])
            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_steps += 1

            if is_main and global_step % config.log_every_steps == 0:
                print(f"  [E{epoch} S{global_step}] loss={loss.item():.4f} "
                      f"ce={out['ce_loss'].item():.4f} dice_l={out['dice_loss'].item():.4f} "
                      f"dice={dice:.4f} avg_dice={epoch_dice/epoch_steps:.4f} ({code})",
                      flush=True)

        # Eval on rank 0 over the full test set
        if distributed:
            dist.barrier()
        avg_eval_dice = 0.0
        if is_main:
            model_unwrapped.eval()
            eval_dice = 0.0
            eval_n = 0
            with torch.no_grad():
                for volumes, labels, code in test_loader:
                    volumes = volumes.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model_unwrapped(volumes, code)
                    eval_dice += batch_dice(out["logits"].argmax(dim=1), labels,
                                            out["num_classes"])
                    eval_n += 1
            avg_eval_dice = eval_dice / max(eval_n, 1)

            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_train_dice = epoch_dice / max(epoch_steps, 1)
            elapsed = time.time() - t0
            print(f"Epoch {epoch}: loss={avg_loss:.4f} train_dice={avg_train_dice:.4f} "
                  f"val_dice={avg_eval_dice:.4f} time={elapsed:.0f}s", flush=True)

            if avg_eval_dice > best_dice:
                best_dice = avg_eval_dice
                torch.save({"model": model_unwrapped.state_dict(), "epoch": epoch,
                            "best_dice": best_dice, "num_classes_by_code": ncmap},
                           os.path.join(config.checkpoint_dir, "best.pt"))
                print(f"  New best val_dice={best_dice:.4f}")
            if (epoch + 1) % config.save_every_epochs == 0:
                torch.save({"model": model_unwrapped.state_dict(), "epoch": epoch,
                            "num_classes_by_code": ncmap},
                           os.path.join(config.checkpoint_dir, f"epoch{epoch:04d}.pt"))
        if distributed:
            dist.barrier()

    if is_main:
        print(f"Training complete. Best val_dice={best_dice:.4f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
