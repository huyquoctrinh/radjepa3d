"""Two-stage training for 3D Medical MLLM on M3D-VQA.

Stage 1: Freeze vision encoder + LLM, train projector only (5 epochs)
Stage 2: Unfreeze vision encoder, add LoRA to LLM, train all (10 epochs)

Usage:
    torchrun --nproc_per_node=4 training/train_mllm.py --stage 1
    torchrun --nproc_per_node=4 training/train_mllm.py --stage 2 --resume checkpoints/mllm/stage1_final.pt
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import math
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.enabled = False

from mllms.config import MLLMConfig
from mllms.model import Med3DMLLM
from mllms.data import M3DVQADataset, build_merged_dataset, collate_fn


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    local_rank = 0
    torch.cuda.set_device(local_rank)
    return local_rank, 1, False


def save_checkpoint(model_unwrapped, optimizer, epoch, global_step, path):
    state = {
        "vision_encoder": model_unwrapped.vision_encoder.state_dict(),
        "projector": model_unwrapped.projector.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
    }
    if hasattr(model_unwrapped.llm, "peft_config"):
        state["lora"] = {
            k: v for k, v in model_unwrapped.llm.state_dict().items()
            if "lora" in k
        }
    torch.save(state, path)
    print(f"Saved: {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.vision_encoder.load_state_dict(ckpt["vision_encoder"])
    model.projector.load_state_dict(ckpt["projector"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "lora" in ckpt and hasattr(model.llm, "peft_config"):
        model.llm.load_state_dict(ckpt["lora"], strict=False)
    print(f"Loaded: {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


@torch.no_grad()
def validate(model, val_loader, config, device, distributed=False, stage=1):
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for volumes, input_ids, attention_mask, labels in val_loader:
        volumes = volumes.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(volumes, input_ids, attention_mask, labels, stage=stage)

        n_tokens = (labels != -100).sum().item()
        total_loss += out["lm_loss"].item() * n_tokens
        total_tokens += n_tokens

    # Gather across processes
    if distributed:
        stats = torch.tensor([total_loss, total_tokens], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = stats[0].item(), stats[1].item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--limit-batches", type=int, default=0)
    # Overrides so alternate MLLM versions run without editing the shared config
    parser.add_argument("--llm-name", type=str, default=None)
    parser.add_argument("--llm-dim", type=int, default=None)
    parser.add_argument("--vision-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--use-hybrid", dest="use_hybrid", action="store_true", default=None)
    parser.add_argument("--no-hybrid", dest="use_hybrid", action="store_false")
    parser.add_argument("--caption-oversample", type=int, default=None)
    parser.add_argument("--vqa-max-train", type=int, default=None)
    parser.add_argument("--vision-pool-tokens", type=int, default=None)
    parser.add_argument("--resume-full", action="store_true",
                        help="true same-stage resume: restore optimizer, LR position and data position")
    args = parser.parse_args()

    if args.gpu_ids and "RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    os.environ["HF_HOME"] = "/data2/gdp2988/.cache/huggingface"

    local_rank, world_size, distributed = setup_distributed()
    is_main = local_rank == 0

    config = MLLMConfig()
    if args.llm_name:
        config.llm_name = args.llm_name
    if args.llm_dim:
        config.llm_dim = args.llm_dim
    if args.vision_checkpoint:
        config.vision_checkpoint = args.vision_checkpoint
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.use_hybrid is not None:
        config.use_hybrid = args.use_hybrid
    if args.caption_oversample is not None:
        config.caption_oversample = args.caption_oversample
    if args.vqa_max_train is not None:
        config.vqa_max_train = args.vqa_max_train
    if args.vision_pool_tokens is not None:
        config.vision_pool_tokens = args.vision_pool_tokens

    if is_main:
        print(f"3D Medical MLLM Training — Stage {args.stage}")
        print(f"  Distributed: {distributed}, World size: {world_size}")

    # Build model
    if is_main:
        print("Building model...")
    model = Med3DMLLM(config)

    if os.path.exists(config.vision_checkpoint):
        model.load_vision_checkpoint(config.vision_checkpoint)

    if args.stage == 1:
        model.setup_stage1()
        total_epochs = config.total_epochs_stage1
        batch_size = config.batch_size_stage1
        if is_main:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Stage 1: projector only, {n_train / 1e6:.1f}M trainable params")
    else:
        model.setup_stage2()
        total_epochs = config.total_epochs_stage2
        batch_size = config.batch_size_stage2
        if is_main:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Stage 2: vision + projector + LoRA, {n_train / 1e6:.1f}M trainable params")

    model = model.cuda()

    if distributed:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model.llm.gradient_checkpointing_enable()
    model_unwrapped = model.module if distributed else model

    # Datasets
    if is_main:
        print("Loading datasets...")
    train_dataset = build_merged_dataset(config, model_unwrapped.tokenizer, split="train")
    val_dataset = M3DVQADataset(config, model_unwrapped.tokenizer, split="val")
    if is_main:
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    collate = partial(collate_fn, tokenizer=model_unwrapped.tokenizer,
                      max_length=config.llm_max_length)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    # Optimizer
    param_groups = []
    if args.stage == 1:
        param_groups.append({
            "params": model_unwrapped.projector.parameters(),
            "lr": config.lr_projector,
        })
    else:
        param_groups.append({
            "params": model_unwrapped.vision_encoder.parameters(),
            "lr": config.lr_vision,
        })
        param_groups.append({
            "params": model_unwrapped.projector.parameters(),
            "lr": config.lr_projector,
        })
        param_groups.append({
            "params": [p for p in model_unwrapped.llm.parameters() if p.requires_grad],
            "lr": config.lr_lora,
        })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume. Default: weights only (correct when switching stage1 -> stage2).
    # --resume-full: true continuation of the SAME stage (optimizer + LR + position).
    start_epoch = 0
    global_step = 0
    skip_batches = 0
    if args.resume:
        ep, gs = load_checkpoint(
            args.resume, model_unwrapped,
            optimizer=optimizer if args.resume_full else None)
        if args.resume_full:
            start_epoch, global_step = ep, gs
            for _ in range(global_step):          # fast-forward the LR schedule
                scheduler.step()
            skip_batches = (global_step % steps_per_epoch) * config.grad_accum_steps
            if is_main:
                print(f"  RESUME-FULL: epoch {start_epoch}, global_step {global_step}, "
                      f"skipping {skip_batches} batches, lr={scheduler.get_last_lr()[0]:.2e}")
        else:
            start_epoch, global_step = 0, 0

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if is_main:
        print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
        print(f"  Grad accum: {config.grad_accum_steps}, Effective batch: "
              f"{batch_size * world_size * config.grad_accum_steps}")
        print(f"  Starting from epoch {start_epoch}")
        print()

    # Training
    best_ppl = float("inf")
    for epoch in range(start_epoch, total_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()
        optimizer.zero_grad()

        for batch_idx, (volumes, input_ids, attention_mask, labels) in enumerate(train_loader):
            if skip_batches and epoch == start_epoch and batch_idx < skip_batches:
                continue          # already trained on these before the checkpoint
            if args.limit_batches and batch_idx >= args.limit_batches:
                break

            volumes = volumes.cuda(non_blocking=True)
            input_ids = input_ids.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(volumes, input_ids, attention_mask, labels,
                            stage=args.stage)
                loss = out["loss"] / config.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model_unwrapped.get_trainable_params(),
                    config.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += out["loss"].item()
                epoch_steps += 1

                if global_step % config.log_every_steps == 0 and is_main:
                    avg = epoch_loss / epoch_steps
                    lr_cur = scheduler.get_last_lr()[0]
                    extra = ""
                    if args.stage == 1:
                        extra = (f" lm={out['lm_loss'].item():.4f}"
                                 f" clip={out['contrastive_loss'].item():.4f}")
                    print(f"  [E{epoch} S{global_step}] "
                          f"loss={out['loss'].item():.4f} avg={avg:.4f} "
                          f"lr={lr_cur:.6f}{extra}", flush=True)

                if global_step % 5000 == 0 and is_main:
                    save_checkpoint(
                        model_unwrapped, optimizer, epoch, global_step,
                        os.path.join(config.checkpoint_dir,
                                     f"stage{args.stage}_step{global_step}.pt"),
                    )

        # End of epoch
        epoch_time = time.time() - t_epoch
        train_avg = epoch_loss / max(epoch_steps, 1)

        # Validation
        t_val = time.time()
        val_loss, val_ppl = validate(
            model, val_loader, config, device=torch.device("cuda"),
            distributed=distributed, stage=args.stage,
        )
        val_time = time.time() - t_val

        if is_main:
            print(f"Epoch {epoch}: train_loss={train_avg:.4f} "
                  f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
                  f"time={epoch_time:.0f}s val_time={val_time:.0f}s", flush=True)

            # Save best by perplexity
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                save_checkpoint(
                    model_unwrapped, optimizer, epoch, global_step,
                    os.path.join(config.checkpoint_dir, f"stage{args.stage}_best.pt"),
                )
                print(f"  New best val_ppl={best_ppl:.2f}")

        # Periodic save
        if (epoch + 1) % config.save_every_epochs == 0 and is_main:
            save_checkpoint(
                model_unwrapped, optimizer, epoch, global_step,
                os.path.join(config.checkpoint_dir,
                             f"stage{args.stage}_epoch{epoch:04d}.pt"),
            )

    # Final save
    if is_main:
        save_checkpoint(
            model_unwrapped, optimizer, total_epochs - 1, global_step,
            os.path.join(config.checkpoint_dir, f"stage{args.stage}_final.pt"),
        )
        print(f"Training complete. Best val_ppl={best_ppl:.2f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
