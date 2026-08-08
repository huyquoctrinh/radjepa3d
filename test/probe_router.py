"""Measure what the hybrid router actually does, per layer.

The routing ablation shows `fixed` (a non-learned 50/50 average, zero router
parameters) is statistically indistinguishable from the learned `layer` router.
There are two very different explanations, and they imply different figures:

  (a) the gates sit near 0.5/0.5, so `layer` *is* `fixed` in effect, or
  (b) the gates are decisive but the choice does not matter.

This dumps per-layer gate statistics for both learned modes so the answer is
measured rather than assumed. For `token` mode it also returns the per-token
assignment map, reshaped to the (gd, gh, gw) patch grid, so routing decisions
can be rendered spatially over the volume.
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

import m3d_eval as M
from load_checkpoint import load_encoder
from bench_ablations_clean import ROUTER_MODE

CKPT = "/data2/gdp2988/3d_cot/checkpoints/ablations_hybrid"


@torch.no_grad()
def collect(name, epoch, paths, device="cuda", batch_size=8):
    path = f"{CKPT}/{name}/checkpoint_epoch{epoch:04d}.pt"
    router_mode = ROUTER_MODE.get(name, "layer")
    enc, cfg, meta = load_encoder(
        path, device=device, router_mode=router_mode)

    mode = cfg.router_mode
    if mode == "fixed":
        return None

    # Capture the router probabilities each block computes, without touching
    # the model: re-run the block's own router on its normed input.
    gates = [[] for _ in enc.blocks]

    hooks = []

    def make_hook(i, block):
        def hook(mod, inp, out):
            x = block.norm(inp[0])
            if mode == "layer":
                p = F.softmax(block.router(x.mean(dim=1)), dim=-1)   # (B,2)
                gates[i].append(p.float().cpu())
            else:
                p = F.softmax(block.router(x), dim=-1)               # (B,L,2)
                gates[i].append(p.float().cpu())
        return hook

    for i, b in enumerate(enc.blocks):
        hooks.append(b.register_forward_hook(make_hook(i, b)))

    for i in range(0, len(paths), batch_size):
        vols = [torch.from_numpy(np.load(p).astype(np.float32))
                for p in paths[i:i + batch_size]]
        batch = torch.stack(vols).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc(batch, indices=None)

    for h in hooks:
        h.remove()

    out = {"mode": mode, "grid": list(cfg.grid_size), "layers": []}
    for i, g in enumerate(gates):
        G = torch.cat(g, dim=0)                     # (N,2) or (N,L,2)
        p_mamba = G[..., 0]
        entry = {
            "layer": i,
            "mean_p_mamba": float(p_mamba.mean()),
            "std_p_mamba": float(p_mamba.std()),
            "min_p_mamba": float(p_mamba.min()),
            "max_p_mamba": float(p_mamba.max()),
        }
        if mode == "token":
            # Fraction of tokens routed to Mamba, and the spatial map.
            hard = (G.argmax(-1) == 0).float()       # (N,L)
            entry["frac_tokens_to_mamba"] = float(hard.mean())
            entry["token_map"] = hard.mean(0).reshape(cfg.grid_size).tolist()
            # How decisive is the top-1 choice?
            entry["mean_top1_prob"] = float(G.max(-1).values.mean())
        out["layers"].append(entry)
    del enc
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--epoch", type=int, default=79)
    p.add_argument("--out", default="/data2/gdp2988/3d_cot/checkpoints/router_stats.json")
    args = p.parse_args()

    task = M.build_task("1", "Plane", max_train=args.n)
    paths = task.train_paths[:args.n]
    print(f"probing on {len(paths)} volumes\n")

    res = {}
    for name in ["baseline", "router_token", "balance_0", "balance_0.1"]:
        r = collect(name, args.epoch, paths)
        if r is None:
            continue
        res[name] = r
        pm = [l["mean_p_mamba"] for l in r["layers"]]
        print(f"[{name}] mode={r['mode']}")
        print("  P(mamba) per layer: " + " ".join(f"{v:.3f}" for v in pm))
        print(f"  across layers: mean={np.mean(pm):.3f} "
              f"min={min(pm):.3f} max={max(pm):.3f}")
        if r["mode"] == "token":
            fr = [l["frac_tokens_to_mamba"] for l in r["layers"]]
            t1 = [l["mean_top1_prob"] for l in r["layers"]]
            print("  frac tokens->mamba: " + " ".join(f"{v:.3f}" for v in fr))
            print("  mean top-1 prob:    " + " ".join(f"{v:.3f}" for v in t1))
        print()

    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
