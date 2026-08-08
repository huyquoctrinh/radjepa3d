"""Training loop for V-JEPA with Mamba backbone on 3D CT volumes."""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import math
import time

import torch
torch.backends.cudnn.enabled = False
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vjepa_config import VJEPAConfig
from vjepa_data import M3DCTDataset, InspectSSLDataset, MergedSSLDataset
from vjepa_model import VJEPA3D
from vjepa_masking import build_masker
from vjepa_utils import (
    cosine_ema_schedule,
    cosine_lr_lambda,
    save_checkpoint,
    load_checkpoint,
)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    else:
        local_rank = 0
        torch.cuda.set_device(local_rank)
        return local_rank, 1, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-strategy", type=str, default=None)
    parser.add_argument("--patch-size", type=str, default=None,
                        help="Comma-separated patch dims, e.g. 8,8,8")
    parser.add_argument("--predictor-dim", type=int, default=None)
    parser.add_argument("--predictor-layers", type=int, default=None)
    parser.add_argument("--bidirectional", action="store_true", default=None)
    parser.add_argument("--no-bidirectional", dest="bidirectional",
                        action="store_false")
    parser.add_argument("--morton", action="store_true", default=None)
    parser.add_argument("--no-morton", dest="morton", action="store_false")
    parser.add_argument("--use-hybrid", action="store_true", default=None)
    parser.add_argument("--router-mode", type=str, default=None,
                        help="Hybrid router: token, layer, or fixed (50/50)")
    parser.add_argument("--balance-loss-weight", type=float, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None,
                        help="GQA key/value heads (use n_attn_heads for full MHA)")
    parser.add_argument("--rope", action="store_true", default=None)
    parser.add_argument("--no-rope", dest="rope", action="store_false")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="Limit steps per epoch for debugging")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs (single-GPU mode only)")
    parser.add_argument("--dataset", type=str, default="m3d",
                        choices=["m3d", "inspect", "merged"])
    parser.add_argument("--inspect-repeat", type=int, default=1,
                        help="Oversample INSPECT in --dataset merged")
    parser.add_argument("--npy-dir", type=str, default=None,
                        help="Volume directory (required for --dataset inspect)")
    parser.add_argument("--hu-window", type=float, nargs=2, default=[-1000.0, 1000.0],
                        metavar=("LO", "HI"),
                        help="HU range mapped to [0,1] (inspect only)")
    args = parser.parse_args()

    config = VJEPAConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.total_epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.dim:
        config.dim = args.dim
    if args.n_layers:
        config.n_layers = args.n_layers
    if args.mask_ratio:
        config.mask_ratio = args.mask_ratio
    if args.mask_strategy:
        config.mask_strategy = args.mask_strategy
    if args.patch_size:
        config.patch_size = tuple(int(x) for x in args.patch_size.split(","))
    if args.predictor_dim:
        config.predictor_dim = args.predictor_dim
    if args.predictor_layers:
        config.predictor_layers = args.predictor_layers
    if args.bidirectional is not None:
        config.use_bidirectional = args.bidirectional
    if args.morton is not None:
        config.use_morton_scan = args.morton
    if args.use_hybrid is not None:
        config.use_hybrid = args.use_hybrid
    if args.router_mode:
        config.router_mode = args.router_mode
    if args.balance_loss_weight is not None:
        config.balance_loss_weight = args.balance_loss_weight
    if args.n_kv_heads is not None:
        config.n_kv_heads = args.n_kv_heads
    if args.rope is not None:
        config.use_rope = args.rope
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.resume:
        config.resume_from = args.resume

    if args.gpu_ids and "RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    local_rank, world_size, distributed = setup_distributed()
    is_main = local_rank == 0

    if is_main:
        print(f"V-JEPA 3D Training")
        print(f"  Distributed: {distributed}, World size: {world_size}")
        print(f"  Patches: {config.n_patches}, Grid: {config.grid_size}")
        print(f"  Visible: {config.n_visible}, Masked: {config.n_masked}")
        print(f"  Model: dim={config.dim}, layers={config.n_layers}")
        print(f"  Predictor: dim={config.predictor_dim}, layers={config.predictor_layers}")

    # Dataset
    if is_main:
        print("Loading dataset...")
    if args.dataset == "inspect":
        if not args.npy_dir:
            raise SystemExit("--dataset inspect requires --npy-dir")
        dataset = InspectSSLDataset(args.npy_dir, hu_window=tuple(args.hu_window))
        if is_main:
            print(f"  INSPECT CTPA: {args.npy_dir}")
            print(f"  HU window: {args.hu_window} -> [0,1]")
    elif args.dataset == "merged":
        if not args.npy_dir:
            raise SystemExit("--dataset merged requires --npy-dir (INSPECT)")
        dataset = MergedSSLDataset(M3DCTDataset(config, split="train"),
                                   args.npy_dir, args.inspect_repeat)
        if is_main:
            c = dataset.source_counts()
            print(f"  Merged M3D + INSPECT: {c['m3d']} + {c['inspect']} "
                  f"= {len(dataset)} volumes")
            print(f"  INSPECT oversampled x{args.inspect_repeat}; "
                  f"per-volume min-max applied to both sources")
    else:
        dataset = M3DCTDataset(config, split="train")
    if is_main:
        print(f"  {len(dataset)} training samples")

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )

    # Model
    model = VJEPA3D(config).cuda()
    if is_main:
        n_params = sum(p.numel() for p in model.context_encoder.parameters()) / 1e6
        n_pred = sum(p.numel() for p in model.predictor.parameters()) / 1e6
        print(f"  Encoder params: {n_params:.1f}M, Predictor params: {n_pred:.1f}M")

    if distributed:
        model._ddp_params_and_buffers_to_ignore = [
            f"target_encoder.{n}"
            for n, _ in model.target_encoder.named_parameters()
        ] + [
            f"target_encoder.{n}"
            for n, _ in model.target_encoder.named_buffers()
        ]
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    broadcast_buffers=False)
    model_unwrapped = model.module if distributed else model

    # Optimizer — only context encoder + predictor
    param_groups = [
        {"params": model_unwrapped.context_encoder.parameters()},
        {"params": model_unwrapped.predictor.parameters()},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=config.lr, weight_decay=config.weight_decay,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * config.total_epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    min_lr_ratio = config.min_lr / config.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr_lambda(step, total_steps, warmup_steps, min_lr_ratio),
    )

    # Masker
    masker = build_masker(config)

    # Resume
    start_epoch = 0
    global_step = 0
    if config.resume_from:
        start_epoch, global_step = load_checkpoint(
            config.resume_from, model_unwrapped, optimizer, scheduler,
        )
        start_epoch += 1

    if is_main:
        print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
        print(f"  Starting from epoch {start_epoch}")
        print()

    # Training
    best_loss = float("inf")
    for epoch in range(start_epoch, config.total_epochs):
        if distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for batch_idx, volumes in enumerate(loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break

            volumes = volumes.cuda(non_blocking=True)
            B = volumes.shape[0]

            visible_idx, masked_idx = masker(B, device=volumes.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(volumes, visible_idx, masked_idx)
                if isinstance(out, dict):
                    loss = out["loss"]
                else:
                    loss = out

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_unwrapped.context_encoder.parameters())
                + list(model_unwrapped.predictor.parameters()),
                config.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            momentum = cosine_ema_schedule(
                global_step, total_steps,
                config.ema_momentum_start, config.ema_momentum_end,
            )
            model_unwrapped.update_target_encoder(momentum)

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % config.log_every_steps == 0 and is_main:
                lr = scheduler.get_last_lr()[0]
                avg = epoch_loss / epoch_steps
                extra = ""
                if isinstance(out, dict):
                    extra = (f" jepa={out['jepa_loss'].item():.4f}"
                             f" align={out['ortho_align_loss'].item():.4f}"
                             f" reg={out['ortho_reg_loss'].item():.4f}")
                    if "balance_loss" in out:
                        extra += f" bal={out['balance_loss'].item():.4f}"
                print(f"  [E{epoch} S{global_step}] "
                      f"loss={loss.item():.4f} avg={avg:.4f} "
                      f"lr={lr:.6f} ema={momentum:.6f}{extra}")

        epoch_time = time.time() - t_epoch
        if is_main and epoch_steps > 0:
            avg = epoch_loss / epoch_steps
            print(f"Epoch {epoch}: avg_loss={avg:.4f} time={epoch_time:.0f}s")

        # Save periodic checkpoint
        if (epoch + 1) % config.save_every_epochs == 0 and is_main:
            save_checkpoint(model_unwrapped, optimizer, scheduler,
                            epoch, global_step, config)

        # Save best checkpoint
        if is_main and epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pt")
                torch.save({
                    "context_encoder": model_unwrapped.context_encoder.state_dict(),
                    "target_encoder": model_unwrapped.target_encoder.state_dict(),
                    "predictor": model_unwrapped.predictor.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                }, best_path)
                print(f"  New best: loss={best_loss:.4f} (epoch {epoch})")

    if is_main:
        save_checkpoint(model_unwrapped, optimizer, scheduler,
                        config.total_epochs - 1, global_step, config)
        print("Training complete.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
