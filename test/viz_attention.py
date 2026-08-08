"""Visualize Mamba vision encoder feature maps (DINO-style).

Since Mamba has no explicit attention, we visualize:
1. PCA of patch features -> RGB color map (like DINO)
2. Feature norm per patch -> intensity map (salient regions)
3. Cosine similarity of each patch to the global mean -> attention-like map

Usage:
    python viz_attention.py --checkpoint stage2_step15000.pt --num-samples 5
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

torch.backends.cudnn.enabled = False

from vjepa_config import VJEPAConfig
from vjepa_model import MambaEncoder
from mllms.config import MLLMConfig


def load_vision_encoder(checkpoint_path):
    config = MLLMConfig()
    vjepa_cfg = VJEPAConfig()
    vjepa_cfg.dim = config.vision_dim
    vjepa_cfg.n_layers = config.vision_n_layers
    vjepa_cfg.d_state = config.vision_d_state
    vjepa_cfg.d_conv = config.vision_d_conv
    vjepa_cfg.expand = config.vision_expand
    vjepa_cfg.volume_size = config.volume_size
    vjepa_cfg.patch_size = config.patch_size

    encoder = MambaEncoder(vjepa_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "target_encoder" if "target_encoder" in ckpt else "vision_encoder"
    encoder.load_state_dict(ckpt[key])
    return encoder.cuda().eval(), vjepa_cfg


@torch.no_grad()
def extract_layer_features(encoder, volume):
    """Extract per-layer features from the encoder.

    Returns list of (n_patches, dim) tensors, one per layer.
    """
    tokens = encoder.patch_embed(volume) + encoder.pos_embed
    layer_features = []
    for block in encoder.blocks:
        tokens = block(tokens)
        layer_features.append(tokens[0].float().cpu())
    final = encoder.norm(tokens)
    layer_features.append(final[0].float().cpu())
    return layer_features


def features_to_grid(features, grid_size):
    """Reshape (n_patches, dim) -> (gd, gh, gw, dim)."""
    gd, gh, gw = grid_size
    return features.reshape(gd, gh, gw, -1)


def compute_pca_map(features, n_components=3):
    """PCA on patch features -> RGB map per depth slice."""
    n, dim = features.shape
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(features.numpy())
    # Normalize each component to [0, 1]
    for i in range(n_components):
        mi, ma = projected[:, i].min(), projected[:, i].max()
        if ma > mi:
            projected[:, i] = (projected[:, i] - mi) / (ma - mi)
    return projected


def compute_norm_map(features):
    """L2 norm per patch -> saliency."""
    norms = torch.norm(features, dim=-1)
    norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    return norms.numpy()


def compute_similarity_map(features):
    """Cosine similarity of each patch to global mean."""
    mean_feat = features.mean(dim=0, keepdim=True)
    sim = torch.nn.functional.cosine_similarity(features, mean_feat, dim=-1)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim.numpy()


def visualize_sample(volume_np, layer_features, grid_size, save_path, sample_idx=0):
    """Create DINO-style visualization for one CT volume."""
    gd, gh, gw = grid_size
    # Use last layer features
    features = layer_features[-1]  # (n_patches, dim)

    pca_map = compute_pca_map(features).reshape(gd, gh, gw, 3)
    norm_map = compute_norm_map(features).reshape(gd, gh, gw)
    sim_map = compute_similarity_map(features).reshape(gd, gh, gw)

    # Pick depth slices to show
    n_slices = min(4, gd)
    slice_indices = [int(i * gd / n_slices) for i in range(n_slices)]

    # Volume slices (original CT)
    vol = volume_np[0]  # (D, H, W)
    D, H, W = vol.shape
    vol_slice_indices = [int(i * D / n_slices) for i in range(n_slices)]

    fig = plt.figure(figsize=(4 * n_slices, 16))
    gs = GridSpec(4, n_slices, figure=fig, hspace=0.3, wspace=0.1)

    for col, (si, vi) in enumerate(zip(slice_indices, vol_slice_indices)):
        # Row 0: Original CT slice
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(vol[vi], cmap="gray", aspect="equal")
        ax.set_title(f"CT Slice {vi}", fontsize=10)
        ax.axis("off")

        # Row 1: PCA feature map
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(pca_map[si], aspect="equal", interpolation="bilinear")
        ax.set_title(f"PCA (d={si})", fontsize=10)
        ax.axis("off")

        # Row 2: Feature norm (saliency)
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(norm_map[si], cmap="hot", aspect="equal", interpolation="bilinear")
        ax.set_title(f"Norm (d={si})", fontsize=10)
        ax.axis("off")

        # Row 3: Similarity to mean
        ax = fig.add_subplot(gs[3, col])
        ax.imshow(sim_map[si], cmap="viridis", aspect="equal", interpolation="bilinear")
        ax.set_title(f"Similarity (d={si})", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Sample {sample_idx}: Vision Encoder Feature Maps", fontsize=14, y=0.98)
    row_labels = ["CT Scan", "PCA Features", "Feature Norm", "Cosine Similarity"]
    for row, label in enumerate(row_labels):
        fig.text(0.01, 0.88 - row * 0.24, label, fontsize=11, fontweight="bold",
                 rotation=90, va="center")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_layer_evolution(layer_features, grid_size, save_path):
    """Show how PCA features evolve across layers."""
    gd, gh, gw = grid_size
    n_layers = len(layer_features)
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    mid_d = gd // 2
    fig, axes = plt.subplots(2, len(layer_indices), figsize=(3 * len(layer_indices), 6))

    for col, li in enumerate(layer_indices):
        feat = layer_features[li]
        pca = compute_pca_map(feat).reshape(gd, gh, gw, 3)
        norm = compute_norm_map(feat).reshape(gd, gh, gw)

        axes[0, col].imshow(pca[mid_d], aspect="equal", interpolation="bilinear")
        axes[0, col].set_title(f"Layer {li}" if li < n_layers - 1 else "Final", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(norm[mid_d], cmap="hot", aspect="equal", interpolation="bilinear")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("PCA", fontsize=10)
    axes[1, 0].set_ylabel("Norm", fontsize=10)
    fig.suptitle("Feature Evolution Across Layers (mid slice)", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


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

    print("Loading vision encoder...")
    encoder, vjepa_cfg = load_vision_encoder(args.checkpoint)
    grid_size = vjepa_cfg.grid_size
    print(f"  Grid: {grid_size}, Patches: {vjepa_cfg.n_patches}")

    # Load samples
    import json
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
            layer_features = extract_layer_features(encoder, vol)

        visualize_sample(
            vol_np, layer_features, grid_size,
            os.path.join(out_dir, f"sample_{count}_features.png"), count,
        )
        visualize_layer_evolution(
            layer_features, grid_size,
            os.path.join(out_dir, f"sample_{count}_layers.png"),
        )
        count += 1

    print(f"\nDone! Visualizations in {out_dir}")


if __name__ == "__main__":
    main()
