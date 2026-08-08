"""Consolidate cached M3D embeddings into the final comparison table.

Reads whatever embeddings exist under checkpoints/embeddings/ and reports, per
encoder and task, both readouts against their reference levels:

  kNN accuracy      vs the majority-class baseline
  balanced accuracy vs chance (1 / n_classes)

Both readouts are needed. kNN measures whether labels are recoverable from
cosine neighbourhoods; the linear probe measures whether the information is in
the representation at all. On these tasks the two disagree substantially, and
reporting only kNN conflates "the encoder learned nothing" with "the encoder's
features are not metrically organized".

Usage:
    python report_m3d.py                    # every encoder with cached embeddings
    python report_m3d.py --no-probe         # kNN only (fast)
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json

import m3d_eval as M

# Display names and grouping for the LaTeX table.
DISPLAY = {
    "swinunetr-ssl":        ("Swin UNETR (SSL)",      "baseline"),
    "voco-b":               ("VoCo-B",                "baseline"),
    "models-genesis":       ("Models Genesis",        "baseline"),
    "merlin":               ("Merlin",                "baseline"),
    "radjepa-raster-ep279": ("RadJepa-Mamba",         "ours"),
    "radjepa-raster-ortho": ("RadJepa-Mamba + Ortho", "ours"),
    "radjepa-hybrid":       ("RadJepa-Hybrid",        "ours"),
    "radjepa-mamba":        ("RadJepa-Mamba (Morton)",         "morton"),
    "radjepa-mamba-ortho":  ("RadJepa-Mamba + Ortho (Morton)", "morton"),
}
GROUP_ORDER = ["baseline", "ours", "morton"]
GROUP_LABEL = {
    "baseline": "Pretrained baselines",
    "ours": "RadJepa (raster scan)",
    "morton": "RadJepa (Morton scan)",
}


def _not_cached(*a, **k):
    raise RuntimeError("not cached")


def collect(encoders, tasks, probe=True):
    rows = {}
    for enc in encoders:
        per_task = {}
        for t in tasks:
            try:
                tr = M.cached_embeddings(enc, t.name, "train", t.train_paths,
                                         _not_cached, verbose=False)
                te = M.cached_embeddings(enc, t.name, "test", t.test_paths,
                                         _not_cached, verbose=False)
            except Exception:
                continue
            knn = M.run_knn(tr, t.train_labels, te, t.test_labels)
            knn.pop("preds", None)          # ndarray, not JSON-serializable
            entry = {"knn": knn, "dim": int(tr.shape[1])}
            if probe:
                entry["lp"] = M.run_linear_probe(
                    tr, t.train_labels, te, t.test_labels)
            per_task[t.name] = entry
        if per_task:
            rows[enc] = per_task
    return rows


def render(rows, tasks, probe=True):
    out = []
    w = 24

    out.append("=" * 92)
    out.append("M3D frozen-feature benchmark — case-disjoint protocol, "
               "val split, k=20 cosine")
    out.append("=" * 92)
    out.append("")
    out.append("Reference levels per task:")
    for t in tasks:
        out.append(f"  {t.name:<7} {len(t.classes):>3} classes | "
                   f"majority {100 * t.majority_baseline:>6.2f}% | "
                   f"chance {100 / len(t.classes):>6.2f}% | "
                   f"train {len(t.train_labels)} / test {len(t.test_labels)}")
    out.append("")

    # kNN accuracy vs majority
    out.append("-" * 92)
    out.append("kNN accuracy (delta vs majority-class baseline)")
    out.append("-" * 92)
    hdr = f"{'Encoder':<{w}}{'dim':>6}"
    for t in tasks:
        hdr += f"{t.name:>20}"
    out.append(hdr)
    for enc, pt in rows.items():
        dim = next(iter(pt.values()))["dim"]
        line = f"{enc:<{w}}{dim:>6}"
        for t in tasks:
            e = pt.get(t.name)
            if not e:
                line += f"{'--':>20}"
                continue
            a = e["knn"]["acc"]
            d = a - t.majority_baseline
            line += f"{100 * a:>13.2f}%{d * 100:>+6.1f}"
        out.append(line)

    # Balanced accuracy, both readouts, vs chance
    out.append("")
    out.append("-" * 92)
    out.append("Balanced accuracy — kNN vs linear probe (chance shown above)")
    out.append("-" * 92)
    hdr = f"{'Encoder':<{w}}"
    for t in tasks:
        hdr += f"{t.name + ' kNN':>14}{t.name + ' LP':>14}"
    out.append(hdr)
    for enc, pt in rows.items():
        line = f"{enc:<{w}}"
        for t in tasks:
            e = pt.get(t.name)
            if not e:
                line += f"{'--':>14}{'--':>14}"
                continue
            line += f"{100 * e['knn']['balanced_acc']:>13.2f}%"
            if probe and "lp" in e:
                line += f"{100 * e['lp']['balanced_acc']:>13.2f}%"
            else:
                line += f"{'--':>14}"
        out.append(line)

    out.append("")
    out.append("Caveat: M3D-Cap ships JPEG slices, so HU is unrecoverable and every")
    out.append("volume is min-max scaled to [0,1]. Swin UNETR, VoCo, Models Genesis and")
    out.append("Merlin all expect specific HU windows and therefore run out of")
    out.append("distribution; RadJepa was pretrained on exactly this [0,1] input.")
    return "\n".join(out)


def _fmt(v, best, bold_best=True):
    """Format a percentage, bolding it when it is the best in its column."""
    s = f"{100 * v:.1f}"
    return rf"\textbf{{{s}}}" if bold_best and v == best else s


def render_latex(rows, tasks, metric="balanced_acc", label="tab:m3d_knn_lp"):
    """Emit a booktabs table with kNN and linear-probe columns per task."""
    metric_name = {"balanced_acc": "balanced accuracy",
                   "acc": "accuracy"}[metric]

    # Best value per (task, readout) column, for bolding.
    best = {}
    for t in tasks:
        for readout in ("knn", "lp"):
            vals = [pt[t.name][readout][metric]
                    for pt in rows.values()
                    if t.name in pt and readout in pt[t.name]]
            best[(t.name, readout)] = max(vals) if vals else None

    n = len(tasks)
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    L.append(r"\begin{tabular}{l" + "cc" * n + "}")
    L.append(r"\toprule")

    head = r"\multirow{2}{*}{Encoder}"
    for t in tasks:
        head += rf" & \multicolumn{{2}}{{c}}{{{t.name}}}"
    L.append(head + r" \\")
    cmids = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                     for i in range(n))
    L.append(cmids)
    L.append("" + " & ".join([""] + ["$k$-NN", "LP"] * n) + r" \\")
    L.append(r"\midrule")

    # Reference row: what a trivial majority-class predictor achieves under
    # this metric. Its accuracy is the class prior, but its balanced accuracy
    # is 1/n_classes, since it gets recall 1 on one class and 0 on the rest.
    ref = ("Majority-class predictor" if metric == "acc"
           else "Majority-class predictor / chance")
    for t in tasks:
        v = (t.majority_baseline if metric == "acc" else 1.0 / len(t.classes))
        ref += rf" & \multicolumn{{2}}{{c}}{{{100 * v:.1f}}}"
    L.append(rf"\textit{{{ref}}} \\")

    for group in GROUP_ORDER:
        members = [e for e in rows if DISPLAY.get(e, (None, None))[1] == group]
        if not members:
            continue
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{{1 + 2 * n}}}{{l}}{{\emph{{{GROUP_LABEL[group]}}}}} \\")
        for enc in members:
            name = DISPLAY[enc][0]
            row = name
            for t in tasks:
                e = rows[enc].get(t.name)
                for readout in ("knn", "lp"):
                    if not e or readout not in e:
                        row += " & --"
                    else:
                        row += " & " + _fmt(e[readout][metric],
                                            best[(t.name, readout)])
            L.append(row + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(rf"\caption{{Frozen-feature {metric_name} (\%) on M3D VQA "
             r"classification under a case-disjoint split, comparing $k$-NN "
             r"($k{=}20$, cosine) against a linear probe (LP). Best per column "
             r"in bold. M3D-Cap ships JPEG slices, so Hounsfield units are "
             r"unrecoverable and all volumes are min--max scaled to $[0,1]$; "
             r"the pretrained baselines therefore cannot apply their native HU "
             r"windowing, whereas RadJepa was pretrained on this "
             r"representation.}")
    L.append(rf"\label{{{label}}}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None,
                   help="Write raw metrics here for later re-rendering")
    p.add_argument("--from-json", default=None,
                   help="Render from cached metrics instead of recomputing")
    p.add_argument("--latex", default=None, help="Write LaTeX tables here")
    args = p.parse_args()

    tasks = M.build_all_tasks()

    if args.from_json:
        with open(args.from_json) as f:
            rows = json.load(f)
    else:
        known = (list(__import__("encoders").RADJEPA_VARIANTS)
                 + ["merlin"]
                 + list(__import__("encoders").BASELINE_VARIANTS))
        rows = collect(known, tasks, probe=not args.no_probe)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"Wrote {args.json}")

    text = render(rows, tasks, probe=not args.no_probe)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nWrote {args.out}")

    if args.latex:
        tex = "\n\n".join([
            render_latex(rows, tasks, "balanced_acc", "tab:m3d_balacc"),
            render_latex(rows, tasks, "acc", "tab:m3d_acc"),
        ])
        with open(args.latex, "w") as f:
            f.write("% Requires \\usepackage{booktabs,multirow}\n" + tex + "\n")
        print(f"\nWrote {args.latex}")
        print(tex)


if __name__ == "__main__":
    main()
