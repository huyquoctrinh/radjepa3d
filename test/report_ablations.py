"""Render the epoch-matched ablation results as one consistent table.

Reports balanced accuracy against chance, plus the difference from each suite's
own baseline and whether that difference clears the noise floor. Accuracy deltas
of a point or two on these test sets are not distinguishable from zero, and the
original ablation logs reported them to three decimals as if they were.

Usage:
    python report_ablations.py --json checkpoints/ablations_clean_A.json \
                                      checkpoints/ablations_clean_B.json
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import math

import m3d_eval as M

# Which ablation belongs to which study, and how to display it. Anything not
# listed falls through to an "other" group.
GROUPS = [
    ("Masking ratio",   ["mask_ratio_50", "mask_ratio_60", "mask_ratio_85"]),
    ("Masking strategy", ["mask_block"]),
    ("Patch / block size", ["patch_8_8_8", "patch_4_16_16", "patch_16_16_16"]),
    ("Predictor",       ["pred_dim_96", "pred_dim_384",
                         "pred_layers_2", "pred_layers_6"]),
    ("Scan direction",  ["bidirectional"]),
    ("Routing mode",    ["router_token", "router_fixed"]),
    ("Router balance loss", ["balance_0", "balance_0.1"]),
    ("Attention heads (KV)", ["kv1", "kv8"]),
    ("Positional encoding", ["no_rope"]),
    ("Scan order",      ["morton"]),
]

PRETTY = {
    "mask_ratio_50": r"mask ratio $0.50$",
    "mask_ratio_60": r"mask ratio $0.60$",
    "mask_ratio_85": r"mask ratio $0.85$",
    "mask_block": "block masking",
    "patch_8_8_8": r"patch $8{\times}8{\times}8$ (4096 tok.)",
    "patch_4_16_16": r"patch $4{\times}16{\times}16$ (2048 tok.)",
    "patch_16_16_16": r"patch $16{\times}16{\times}16$ (512 tok.)",
    "pred_dim_96": r"predictor dim $96$",
    "pred_dim_384": r"predictor dim $384$",
    "pred_layers_2": r"predictor depth $2$",
    "pred_layers_6": r"predictor depth $6$",
    "bidirectional": "bidirectional Mamba",
    "router_token": "per-token routing",
    "router_fixed": r"fixed $50/50$ (no router)",
    "balance_0": r"balance weight $0$",
    "balance_0.1": r"balance weight $0.1$",
    "kv1": r"$1$ KV head",
    "kv8": r"$8$ KV heads",
    "no_rope": "no RoPE",
    "morton": "Morton scan",
    "baseline": "baseline",
}


def noise_floor(p, n):
    """95% min. detectable difference between two independent proportions."""
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    return 1.96 * se * math.sqrt(2)


def load(paths):
    rows = {}
    for p in paths:
        with open(p) as f:
            rows.update(json.load(f))
    return rows


def normalized_score(entry, tasks, metric="balanced_acc"):
    """Mean headroom captured above chance, averaged over tasks.

    A plain mean over tasks is dominated by Plane, whose chance level is 50%
    against Organ's 2.9%. Normalizing each task to (score - chance) / (100 -
    chance) puts them on a common 0-1 scale so "best overall" is not just
    "best on Plane".
    """
    vals = []
    for t in tasks:
        chance = 100.0 / len(t.classes)
        v = 100 * entry[t.name]["knn"][metric]
        vals.append((v - chance) / (100.0 - chance))
    return sum(vals) / len(vals)


def pick_anchor(members, tasks, metric="balanced_acc"):
    """Highest normalized score in the suite."""
    return max(members, key=lambda k: normalized_score(members[k], tasks, metric))


def suite_of(tag):
    return tag.split("/", 1)[0]


def name_of(tag):
    return tag.split("/", 1)[1]


def render(rows, tasks, metric="balanced_acc", anchor="baseline"):
    out = []
    suites = sorted({suite_of(t) for t in rows})

    for s in suites:
        members = {name_of(k): v for k, v in rows.items() if suite_of(k) == s}
        if "baseline" not in members:
            continue
        anchor_key = (pick_anchor(members, tasks, metric) if anchor == "best"
                      else "baseline")
        base = members[anchor_key]
        epoch = base["epoch"]
        out.append("")
        out.append("=" * 88)
        out.append(f"Suite {s} — all models at epoch {epoch}, "
                   f"case-disjoint protocol, k-NN k=20 cosine")
        out.append(f"anchor: {anchor_key}"
                   + ("  (best normalized score in suite)" if anchor == "best"
                      else "  (as-trained baseline)"))
        out.append("=" * 88)

        hdr = f"{'Ablation':<30}"
        for t in tasks:
            hdr += f"{t.name:>17}"
        out.append(hdr)
        sub = f"{'':<30}" + "".join(f"{'bal.acc  Δ':>17}" for _ in tasks)
        out.append(sub)
        out.append("-" * len(hdr))

        ref = f"{PRETTY.get(anchor_key, anchor_key) + '  (anchor)':<30}"
        for t in tasks:
            ref += f"{100 * base[t.name]['knn'][metric]:>11.2f}{'':>6}"
        out.append(ref)

        # Noise floor per task, computed at the baseline's rate.
        nf = {}
        for t in tasks:
            n = len(t.test_labels)
            nf[t.name] = noise_floor(base[t.name]["knn"]["acc"], n)

        seen = {anchor_key}
        for gname, keys in GROUPS:
            present = [k for k in keys if k in members and k != anchor_key]
            if not present:
                continue
            out.append(f"  {gname}:")
            for k in present:
                seen.add(k)
                line = f"    {PRETTY.get(k, k):<26}"
                for t in tasks:
                    v = members[k][t.name]["knn"][metric]
                    b = base[t.name]["knn"][metric]
                    d = 100 * (v - b)
                    sig = "*" if abs(d) > 100 * nf[t.name] else " "
                    line += f"{100 * v:>11.2f}{d:>+5.1f}{sig}"
                out.append(line)

        leftover = [k for k in members if k not in seen]
        if leftover:
            out.append("  Other:")
            for k in sorted(leftover):
                line = f"    {PRETTY.get(k, k):<26}"
                for t in tasks:
                    v = members[k][t.name]["knn"][metric]
                    d = 100 * (v - base[t.name]["knn"][metric])
                    sig = "*" if abs(d) > 100 * nf[t.name] else " "
                    line += f"{100 * v:>11.2f}{d:>+5.1f}{sig}"
                out.append(line)

        out.append("")
        out.append("  * exceeds the 95% noise floor: " + ", ".join(
            f"{t.name} ±{100 * nf[t.name]:.1f}pt" for t in tasks))

    out.append("")
    out.append("Reference levels — " + ", ".join(
        f"{t.name}: chance {100 / len(t.classes):.1f}%, "
        f"majority-class predictor {100 / len(t.classes):.1f}% balanced"
        for t in tasks))
    return "\n".join(out)


def render_latex(rows, tasks, metric="balanced_acc"):
    L = []
    suites = sorted({suite_of(t) for t in rows})
    n = len(tasks)

    L.append(r"\begin{table}[t]")
    L.append(r"\centering\small")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    L.append(r"\begin{tabular}{l" + "r" * n + "}")
    L.append(r"\toprule")
    L.append("Ablation & " + " & ".join(t.name for t in tasks) + r" \\")
    L.append(r"\midrule")
    L.append(r"\textit{Chance} & " + " & ".join(
        f"{100 / len(t.classes):.1f}" for t in tasks) + r" \\")

    for s in suites:
        members = {name_of(k): v for k, v in rows.items() if suite_of(k) == s}
        base = members.get("baseline")
        if not base:
            continue
        nf = {t.name: noise_floor(base[t.name]["knn"]["acc"],
                                  len(t.test_labels)) for t in tasks}
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{{1 + n}}}{{l}}{{\emph{{Suite {s} "
                 rf"(epoch {base['epoch']})}}}} \\")
        L.append("baseline & " + " & ".join(
            f"{100 * base[t.name]['knn'][metric]:.1f}" for t in tasks)
            + r" \\")
        seen = {"baseline"}
        for gname, keys in GROUPS:
            present = [k for k in keys if k in members]
            if not present:
                continue
            L.append(rf"\quad\emph{{{gname}}} & " + " & ".join([""] * n)
                     + r" \\")
            for k in present:
                seen.add(k)
                cells = []
                for t in tasks:
                    v = 100 * members[k][t.name]["knn"][metric]
                    d = v - 100 * base[t.name]["knn"][metric]
                    s_ = rf"$^{{*}}$" if abs(d) > 100 * nf[t.name] else ""
                    cells.append(rf"{v:.1f}{s_}")
                L.append(rf"\quad\quad {PRETTY.get(k, k)} & "
                         + " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\caption{Ablation study, frozen-feature balanced accuracy (\%) "
             r"under the case-disjoint protocol with $k$-NN ($k{=}20$, cosine). "
             r"All models within a suite are evaluated at the same epoch, so "
             r"differences are not confounded by training length. "
             r"$^{*}$ marks a difference from the suite baseline exceeding the "
             r"95\% noise floor for that task; unmarked rows are statistically "
             r"indistinguishable from the baseline.}")
    L.append(r"\label{tab:ablations}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", required=True)
    p.add_argument("--latex", default=None)
    p.add_argument("--anchor", default="baseline", choices=["baseline","best"])
    args = p.parse_args()

    tasks = M.build_all_tasks()
    rows = load(args.json)
    text = render(rows, tasks, anchor=args.anchor)
    print(text)
    if args.latex:
        tex = render_latex(rows, tasks)
        with open(args.latex, "w") as f:
            f.write("% Requires \\usepackage{booktabs}\n" + tex + "\n")
        print(f"\nWrote {args.latex}")
        print(tex)


if __name__ == "__main__":
    main()
