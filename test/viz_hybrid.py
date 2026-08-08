"""Visualize hybrid Mamba-Transformer vision encoder.

Three visualization types:
1. GQA Attention maps — real attention weights from Transformer branch
2. Router maps — which tokens chose Mamba vs Transformer per layer
3. Feature maps — PCA/norm/similarity (like DINO)

Usage:
    python viz_hybrid.py --checkpoint vjepa_hybrid/checkpoint_epoch0019.pt --num-samples 5
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

torch.backends.cudnn.enabled = False

from vjepa_config import VJEPAConfig
from vjepa_model import MambaEncoder
from vjepa_hybrid import HybridBlock
from mllms.config import MLLMConfig


def load_encoder(checkpoint_path):
    config = VJEPAConfig()
    encoder = MambaEncoder(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "target_encoder" if "target_encoder" in ckpt else "context_encoder"
    encoder.load_state_dict(ckpt[key])
    return encoder.cuda().eval(), config


@torch.no_grad()
def extract_all(encoder, volume):
    """Extract per-layer features, attention maps, and router decisions."""
    tokens = encoder.patch_embed(volume) + encoder.pos_embed

    layer_features = []
    layer_attentions = []
    layer_router_probs = []

    for block in encoder.blocks:
        if isinstance(block, HybridBlock):
            # Get router probs
            normed = block.norm(tokens)
            router_logits = block.router(normed)
            probs = F.softmax(router_logits, dim=-1)  # (B, L, 2)
            layer_router_probs.append(probs[0].cpu())

            # Get attention weights from GQA
            x = normed
            B, L, _ = x.shape
            gqa = block.gqa
            q = gqa.q_proj(x).view(B, L, gqa.n_heads, gqa.head_dim).transpose(1, 2)
            k = gqa.k_proj(x).view(B, L, gqa.n_kv_heads, gqa.head_dim).transpose(1, 2)

            if gqa.rope is not None:
                q = gqa.rope(q)
                k = gqa.rope(k)

            k = k.repeat_interleave(gqa.kv_group_size, dim=1)
            attn_weights = (q @ k.transpose(-2, -1)) / (gqa.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)  # (B, n_heads, L, L)
            layer_attentions.append(attn_weights[0].float().cpu())

        # Forward through block
        tokens = block(tokens)
        layer_features.append(tokens[0].float().cpu())

    final = encoder.norm(tokens)
    layer_features.append(final[0].float().cpu())

    return layer_features, layer_attentions, layer_router_probs


def visualize_attention_maps(attentions, grid_size, save_path, layer_indices=None):
    """Visualize GQA attention maps for selected layers."""
    gd, gh, gw = grid_size
    n_layers = len(attentions)
    if layer_indices is None:
        layer_indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]

    n_cols = len(layer_indices)
    # Show attention from center patch and mean across heads
    center_idx = (gd // 2) * gh * gw + (gh // 2) * gw + gw // 2

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, li in enumerate(layer_indices):
        attn = attentions[li]  # (n_heads, L, L)

        # Mean attention across all heads for center patch
        mean_attn = attn.mean(dim=0)[center_idx]  # (L,)
        attn_map = mean_attn.reshape(gd, gh, gw)

        # Show mid-depth slice
        mid_d = gd // 2
        axes[0, col].imshow(attn_map[mid_d].numpy(), cmap="hot", interpolation="bilinear")
        axes[0, col].set_title(f"Layer {li}\nCenter→All", fontsize=9)
        axes[0, col].axis("off")

        # Mean attention across all queries (global attention pattern)
        global_attn = attn.mean(dim=(0, 1))  # (L,) — avg attention received per token
        global_map = global_attn.reshape(gd, gh, gw)
        axes[1, col].imshow(global_map[mid_d].numpy(), cmap="viridis", interpolation="bilinear")
        axes[1, col].set_title(f"Global Attn\nReceived", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("GQA Attention Maps (mid-depth slice)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Center Patch →", fontsize=10)
    axes[1, 0].set_ylabel("Avg Received", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_router_maps(router_probs, grid_size, save_path):
    """Visualize which tokens route to Mamba vs Transformer per layer."""
    gd, gh, gw = grid_size
    n_layers = len(router_probs)
    layer_indices = list(range(min(n_layers, 6)))
    if n_layers > 6:
        step = n_layers // 6
        layer_indices = [i * step for i in range(6)]

    n_cols = len(layer_indices)
    mid_d = gd // 2

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, li in enumerate(layer_indices):
        probs = router_probs[li]  # (L, 2)

        # Transformer probability
        trans_prob = probs[:, 1].reshape(gd, gh, gw)
        axes[0, col].imshow(trans_prob[mid_d].numpy(), cmap="RdYlBu_r",
                           vmin=0, vmax=1, interpolation="bilinear")
        axes[0, col].set_title(f"Layer {li}", fontsize=9)
        axes[0, col].axis("off")

        # Hard routing decision
        choice = probs.argmax(dim=-1).reshape(gd, gh, gw)
        axes[1, col].imshow(choice[mid_d].numpy(), cmap="coolwarm",
                           vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Trans. prob", fontsize=9)
    axes[1, 0].set_ylabel("Route\n(blue=Mamba\nred=Trans)", fontsize=8)
    fig.suptitle("Router Decisions Per Layer (mid-depth)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_features(volume_np, features, grid_size, save_path):
    """PCA + norm + similarity feature maps (same as DINO-style viz)."""
    gd, gh, gw = grid_size
    feat = features[-1]  # Last layer

    # PCA
    n, dim = feat.shape
    pca = PCA(n_components=3)
    projected = pca.fit_transform(feat.numpy())
    for i in range(3):
        mi, ma = projected[:, i].min(), projected[:, i].max()
        if ma > mi:
            projected[:, i] = (projected[:, i] - mi) / (ma - mi)
    pca_map = projected.reshape(gd, gh, gw, 3)

    # Norm
    norms = torch.norm(feat, dim=-1)
    norms = ((norms - norms.min()) / (norms.max() - norms.min() + 1e-8)).numpy()
    norm_map = norms.reshape(gd, gh, gw)

    # Similarity to mean
    mean_feat = feat.mean(dim=0, keepdim=True)
    sim = F.cosine_similarity(feat, mean_feat, dim=-1)
    sim = ((sim - sim.min()) / (sim.max() - sim.min() + 1e-8)).numpy()
    sim_map = sim.reshape(gd, gh, gw)

    vol = volume_np[0]
    D = vol.shape[0]
    n_slices = min(4, gd)
    slice_indices = [int(i * gd / n_slices) for i in range(n_slices)]
    vol_indices = [int(i * D / n_slices) for i in range(n_slices)]

    fig, axes = plt.subplots(4, n_slices, figsize=(4 * n_slices, 16))

    for col, (si, vi) in enumerate(zip(slice_indices, vol_indices)):
        axes[0, col].imshow(vol[vi], cmap="gray")
        axes[0, col].set_title(f"CT Slice {vi}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(pca_map[si], interpolation="bilinear")
        axes[1, col].set_title(f"PCA (d={si})", fontsize=9)
        axes[1, col].axis("off")

        axes[2, col].imshow(norm_map[si], cmap="hot", interpolation="bilinear")
        axes[2, col].set_title(f"Norm (d={si})", fontsize=9)
        axes[2, col].axis("off")

        axes[3, col].imshow(sim_map[si], cmap="viridis", interpolation="bilinear")
        axes[3, col].set_title(f"Similarity (d={si})", fontsize=9)
        axes[3, col].axis("off")

    labels = ["CT Scan", "PCA Features", "Feature Norm", "Cosine Similarity"]
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")

    fig.suptitle("Feature Maps (Hybrid Encoder)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_combined(volume_np, attentions, router_probs, features, grid_size, save_path):
    """Combined overview: CT + attention + router + features in one image."""
    gd, gh, gw = grid_size
    mid_d = gd // 2
    vol = volume_np[0]
    D = vol.shape[0]
    mid_vol = D // 2

    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 5, figure=fig, wspace=0.15)

    # CT
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(vol[mid_vol], cmap="gray")
    ax.set_title("CT Scan", fontsize=10, fontweight="bold")
    ax.axis("off")

    # Attention (last layer, center patch)
    if attentions:
        ax = fig.add_subplot(gs[0, 1])
        attn = attentions[-1].mean(dim=0)
        center = (gd // 2) * gh * gw + (gh // 2) * gw + gw // 2
        attn_map = attn[center].reshape(gd, gh, gw)[mid_d]
        ax.imshow(attn_map.numpy(), cmap="hot", interpolation="bilinear")
        ax.set_title("GQA Attention", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Router (last layer)
    if router_probs:
        ax = fig.add_subplot(gs[0, 2])
        trans_prob = router_probs[-1][:, 1].reshape(gd, gh, gw)[mid_d]
        ax.imshow(trans_prob.numpy(), cmap="RdYlBu_r", vmin=0, vmax=1,
                 interpolation="bilinear")
        ax.set_title("Router\n(red=Transformer)", fontsize=10, fontweight="bold")
        ax.axis("off")

    # PCA features
    feat = features[-1]
    pca = PCA(n_components=3)
    proj = pca.fit_transform(feat.numpy())
    for i in range(3):
        mi, ma = proj[:, i].min(), proj[:, i].max()
        if ma > mi:
            proj[:, i] = (proj[:, i] - mi) / (ma - mi)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(proj.reshape(gd, gh, gw, 3)[mid_d], interpolation="bilinear")
    ax.set_title("PCA Features", fontsize=10, fontweight="bold")
    ax.axis("off")

    # Feature norm
    norms = torch.norm(feat, dim=-1)
    norms = ((norms - norms.min()) / (norms.max() - norms.min() + 1e-8)).numpy()
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(norms.reshape(gd, gh, gw)[mid_d], cmap="hot", interpolation="bilinear")
    ax.set_title("Feature Norm", fontsize=10, fontweight="bold")
    ax.axis("off")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["HF_HOME"] = "/data2/gdp2988/.cache/huggingface"

    config = MLLMConfig()
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "viz")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading hybrid vision encoder...")
    encoder, vjepa_cfg = load_encoder(args.checkpoint)
    grid_size = vjepa_cfg.grid_size
    n_hybrid = sum(1 for b in encoder.blocks if isinstance(b, HybridBlock))
    print(f"  Grid: {grid_size}, Patches: {vjepa_cfg.n_patches}, Hybrid blocks: {n_hybrid}")

    with open(config.data_json) as f:
        data = json.load(f)

    entries = data["test100"]
    count = 0
    for entry in entries:
        if count >= args.num_samples:
            break
        parts = entry["image"].split("/")
        npy_path = os.path.join(config.npy_dir, parts[1], parts[2], parts[3])
        if not os.path.exists(npy_path):
            continue

        print(f"\nSample {count}: {entry['image']}")
        vol_np = np.load(npy_path).astype(np.float32)
        vol = torch.from_numpy(vol_np).unsqueeze(0).cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            features, attentions, router_probs = extract_all(encoder, vol)

        # All visualizations
        visualize_combined(
            vol_np, attentions, router_probs, features, grid_size,
            os.path.join(out_dir, f"sample_{count}_combined.png"),
        )
        print(f"  Saved combined")

        visualize_attention_maps(
            attentions, grid_size,
            os.path.join(out_dir, f"sample_{count}_attention.png"),
        )
        print(f"  Saved attention maps")

        visualize_router_maps(
            router_probs, grid_size,
            os.path.join(out_dir, f"sample_{count}_router.png"),
        )
        print(f"  Saved router maps")

        visualize_features(
            vol_np, features, grid_size,
            os.path.join(out_dir, f"sample_{count}_features.png"),
        )
        print(f"  Saved feature maps")

        count += 1

    print(f"\nDone! Visualizations in {out_dir}")


if __name__ == "__main__":
    main()
