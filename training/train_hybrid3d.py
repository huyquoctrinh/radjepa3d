"""Pretraining loop for HybridJEPA (scan_and_patching).

Usage:
    python training/train_hybrid3d.py --gpu-ids 2,3
    torchrun --nproc_per_node=2 training/train_hybrid3d.py
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import math
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.enabled = False

from hybrid3d.config import Hybrid3DConfig
from hybrid3d.ssl_modules.pretrain_model import HybridJEPA


class NpyDataset(torch.utils.data.Dataset):
    """Simple dataset loading pre-extracted .npy volumes."""

    def __init__(self, npy_dir, json_path, split="train"):
        import json
        with open(json_path) as f:
            data = json.load(f)
        entries = data[split]

        self.samples = []
        for entry in entries:
            parts = entry["image"].split("/")
            npy = os.path.join(npy_dir, parts[1], parts[2], parts[3])
            if os.path.exists(npy):
                self.samples.append(npy)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        vol = np.load(self.samples[idx]).astype(np.float32)
        return torch.from_numpy(vol)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    local_rank = 0
    torch.cuda.set_device(local_rank)
    return local_rank, 1, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/data2/gdp2988/3d_cot/checkpoints/scan_and_patch")
    # Ablation knobs
    parser.add_argument("--patch-t", type=str, default="8,16,16")
    parser.add_argument("--patch-m", type=str, default="8,8,8")
    parser.add_argument("--mask-mode", type=str, default="shared",
                        choices=["shared", "independent"])
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--no-bidirectional", dest="bidirectional",
                        action="store_false", default=True)
    parser.add_argument("--no-morton", dest="morton",
                        action="store_false", default=True)
    parser.add_argument("--pred-depth", type=int, default=None,
                        help="Set both pred_depth_t and pred_depth_m")
    args = parser.parse_args()

    if args.gpu_ids and "RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    local_rank, world_size, distributed = setup_distributed()
    is_main = local_rank == 0

    cfg = Hybrid3DConfig(
        img_size=(32, 256, 256),
        patch_t=tuple(int(x) for x in args.patch_t.split(",")),
        patch_m=tuple(int(x) for x in args.patch_m.split(",")),
        mask_mode=args.mask_mode,
        mamba_bidirectional=args.bidirectional,
        use_morton=args.morton,
    )
    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio
    if args.pred_depth is not None:
        cfg.pred_depth_t = args.pred_depth
        cfg.pred_depth_m = args.pred_depth

    if is_main:
        print("HybridJEPA Pretraining (scan_and_patching)")
        print(f"  Distributed: {distributed}, World: {world_size}")
        print(f"  T tokens: {cfg.n_tokens_t}, M tokens: {cfg.n_tokens_m}")
        print(f"  Mask ratio: {cfg.mask_ratio}")

    model = HybridJEPA(cfg).cuda()
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Trainable params: {n_params:.1f}M")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_unwrapped = model.module if distributed else model

    dataset = NpyDataset(
        "/data2/gdp2988/3d_cot/extracted_M3D_Cap",
        "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/M3D_Cap.json",
    )
    if is_main:
        print(f"  Dataset: {len(dataset)} samples")

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=4, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model_unwrapped.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if is_main:
        print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
        print()

    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_idx, volumes in enumerate(loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break

            volumes = volumes.cuda(non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(volumes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_unwrapped.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            model_unwrapped.ema_step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % 50 == 0 and is_main:
                avg = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  [E{epoch} S{global_step}] loss={loss.item():.4f} "
                      f"avg={avg:.4f} lr={lr:.6f}", flush=True)

        elapsed = time.time() - t0
        if is_main and epoch_steps > 0:
            avg = epoch_loss / epoch_steps
            print(f"Epoch {epoch}: avg_loss={avg:.4f} time={elapsed:.0f}s", flush=True)

            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "transformer": model_unwrapped.transformer.state_dict(),
                    "mamba": model_unwrapped.mamba.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                }, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  New best: loss={best_loss:.4f}")

        if (epoch + 1) % 20 == 0 and is_main:
            torch.save({
                "transformer": model_unwrapped.transformer.state_dict(),
                "mamba": model_unwrapped.mamba.state_dict(),
                "epoch": epoch,
            }, os.path.join(args.checkpoint_dir, f"epoch{epoch:04d}.pt"))

    if is_main:
        print(f"Training complete. Best loss={best_loss:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
