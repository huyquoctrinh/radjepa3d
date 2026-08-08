"""Utilities: EMA schedule, checkpointing, logging."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import math
import os
import torch


def cosine_ema_schedule(step, total_steps, start=0.996, end=0.9999):
    progress = step / max(total_steps, 1)
    return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2


def cosine_lr_lambda(step, total_steps, warmup_steps, min_lr_ratio):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch:04d}.pt")
    torch.save({
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)
    print(f"Saved checkpoint: {path}")
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.predictor.load_state_dict(ckpt["predictor"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Loaded checkpoint: {path} (epoch {ckpt['epoch']}, step {ckpt['global_step']})")
    return ckpt["epoch"], ckpt["global_step"]
