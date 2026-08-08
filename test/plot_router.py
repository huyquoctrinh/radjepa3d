"""Figures for the routing ablation.

The routing result is a negative one for the `layer` router and a modest
positive one for `token`, and the reason is measurable: the layer router
converges to a near-uniform blend (which is what `fixed` hardcodes), while the
token router makes confident, spatially varying choices. These panels show that
rather than asserting it.
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATS = "/data2/gdp2988/3d_cot/checkpoints/router_stats.json"
OUT = "/data2/gdp2988/3d_cot/checkpoints"

LABEL = {
    "baseline": "layer router",
    "router_token": "token router",
    "balance_0": r"layer, $\gamma_{\mathrm{bal}}{=}0$",
    "balance_0.1": r"layer, $\gamma_{\mathrm{bal}}{=}0.1$",
}
COLOR = {"baseline": "#c0392b", "router_token": "#2471a3",
         "balance_0": "#e59866", "balance_0.1": "#7d3c98"}


def fig_gates(d):
    """Panel A: gate value per layer, against the fixed-50/50 reference."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0.5, color="0.35", ls="--", lw=1.2, zorder=1)
    # Anchor the annotation outside the data range so it cannot collide.
    ax.annotate("fixed\n50/50", xy=(11.6, 0.5), color="0.35", fontsize=7.5,
                ha="left", va="center", annotation_clip=False,
                linespacing=1.3)
    ax.axhspan(0.45, 0.55, color="0.88", alpha=0.6, zorder=0)

    for k in ["baseline", "balance_0", "balance_0.1", "router_token"]:
        if k not in d:
            continue
        y = [l["mean_p_mamba"] for l in d[k]["layers"]]
        ax.plot(range(len(y)), y, "-o", ms=3.5, lw=1.6,
                color=COLOR[k], label=LABEL[k],
                zorder=3 if k == "router_token" else 2)

    ax.set_xlabel("encoder layer")
    ax.set_ylabel(r"$P(\mathrm{Mamba})$")
    ax.set_ylim(0.30, 0.78)
    ax.set_xlim(-0.4, 11.45)   # leaves the annotation outside the axes
    ax.set_xticks(range(12))
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    # The honest claim: a layer gate is a *blend* weight, never a selection --
    # even its most extreme value (0.74) mixes both branches -- which is why
    # the learned layer router matches a hardcoded 50/50 average.
    ax.set_title("Layer gates blend rather than select; "
                 r"$\gamma_{\mathrm{bal}}{=}0.1$ pins them to 50/50",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/router_gates.pdf")
    fig.savefig(f"{OUT}/router_gates.png", dpi=200)
    print(f"wrote {OUT}/router_gates.{{pdf,png}}")


def fig_token(d):
    """Panel B: token-router decisiveness and its in-plane specialization."""
    t = d["router_token"]
    layers = t["layers"]
    n_show = [0, 3, 6, 11]

    fig = plt.figure(figsize=(9.6, 2.5))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.5, 0.18, 1, 1, 1, 1],
                          wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    conf = [l["mean_top1_prob"] for l in layers]
    frac = [l["frac_tokens_to_mamba"] for l in layers]
    ax.plot(range(12), conf, "-o", ms=3.5, lw=1.6, color="#2471a3",
            label="top-1 confidence")
    ax.plot(range(12), frac, "-s", ms=3.5, lw=1.6, color="#7f8c8d",
            label="frac. to Mamba")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylim(0.25, 1.0)
    ax.set_ylabel("probability")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("token router is decisive", fontsize=9)

    # In-plane maps: depth contributes almost nothing (range < 0.09), so
    # averaging over the 4 depth slabs loses no structure and is far more
    # readable than a 4x12 grid of slabs.
    for j, li in enumerate(n_show):
        axm = fig.add_subplot(gs[0, j + 2])
        m = np.array(layers[li]["token_map"]).mean(axis=0)   # (16,16)
        im = axm.imshow(m, cmap="coolwarm", vmin=0, vmax=1)
        axm.set_title(f"layer {li}", fontsize=9)
        axm.set_xticks([]); axm.set_yticks([])
        if j == len(n_show) - 1:
            cb = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.06)
            cb.set_label(r"$P(\mathrm{Mamba})$", fontsize=7)
            cb.ax.tick_params(labelsize=6)

    # The maps track the circular CT field of view: the router separates
    # patient tissue from background air, and swaps which branch takes which
    # as depth increases.
    fig.suptitle("Per-token routing splits foreground from background, "
                 "and inverts with depth", fontsize=10, y=1.04)
    fig.savefig(f"{OUT}/router_token_maps.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/router_token_maps.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}/router_token_maps.{{pdf,png}}")


if __name__ == "__main__":
    with open(STATS) as f:
        d = json.load(f)
    fig_gates(d)
    fig_token(d)
