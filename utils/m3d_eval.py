"""Shared frozen-feature evaluation harness for M3D VQA classification tasks.

Replaces the ad-hoc protocol in vjepa_benchmark / benchmark_ablations /
benchmark_merlin. Differences from the old harness, all of which changed the
numbers:

  1. Case-disjoint train set. The M3D VQA train/val splits share case IDs for
     77-86% of val images (multiple series per patient), so a val volume's
     nearest neighbour was routinely a sibling series of the same case. Train
     images whose case appears in test are dropped; the test set is untouched.
  2. Labels are lowercased. 'Arterial phase' and 'Arterial Phase' were separate
     classes, so ~4.5% of the Phase test set sat in a bin kNN had to match by
     exact casing.
  3. min_count is applied to *train* counts, not per-split. Previously val's
     small size decided which classes survived, which silently deleted the
     Sagittal plane class (5 val samples).
  4. Majority-class baseline, balanced accuracy and macro-F1 are reported
     alongside accuracy. Plane's majority baseline is 91.3% and Organ's is
     15.1%, so raw accuracy alone is not interpretable on these tasks.
  5. Embeddings are cached to disk per (encoder, task, split), so protocol
     changes do not require re-extracting every encoder.

Loading a volume is allowed to fail, but a failure drops the sample rather than
substituting a zero volume with a real label attached.
"""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import csv
import hashlib
import os
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier

CSV_FILES = {
    "train": "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_train.csv",
    "val": "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_val.csv",
    "test": "/data2/gdp2988/3d_cot/dataset/m3d_vqa_text/M3D_VQA_test.csv",
}

NPY_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"
EMBED_CACHE_DIR = "/data2/gdp2988/3d_cot/checkpoints/embeddings"

TASKS = [("1", "Plane"), ("2", "Phase"), ("3", "Organ")]


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #

def _case_of(img_path):
    """'M3D_Cap_npy/ct_case/007789/Axial.npy' -> ('ct_case', '007789')."""
    parts = img_path.split("/")
    if len(parts) < 4:
        raise ValueError(f"unexpected image path layout: {img_path!r}")
    return parts[1], parts[2]


def img_path_to_npy(img_path, npy_dir=NPY_DIR):
    parts = img_path.split("/")
    if len(parts) < 4:
        raise ValueError(f"unexpected image path layout: {img_path!r}")
    return os.path.join(npy_dir, parts[1], parts[2], parts[3])


def _read_split(csv_path, qtype, normalize_labels=True):
    """image_path -> label, for one question type. One row per image in M3D VQA."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["Question Type"] != qtype:
                continue
            label = row["Answer"].strip()
            if normalize_labels:
                label = label.lower()
            out[row["Image Path"]] = label
    return out


class Task:
    """One M3D classification task with a case-disjoint train/test split."""

    def __init__(self, qtype, name, train_paths, train_labels,
                 test_paths, test_labels, classes, stats):
        self.qtype = qtype
        self.name = name
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.test_paths = test_paths
        self.test_labels = test_labels
        self.classes = classes
        self.stats = stats

    @property
    def majority_baseline(self):
        c = Counter(self.test_labels)
        return max(c.values()) / len(self.test_labels)

    def describe(self):
        s = self.stats
        return (
            f"[{self.name}] train={len(self.train_paths)} test={len(self.test_paths)} "
            f"classes={len(self.classes)} majority={100 * self.majority_baseline:.2f}%\n"
            f"    dropped: {s['dropped_case_overlap']} train imgs sharing a test case, "
            f"{s['dropped_missing_npy']} missing npy, "
            f"{s['dropped_rare']} rare-class, {s['dropped_unseen']} test-only-class"
        )


def build_task(qtype, name, npy_dir=NPY_DIR, split="val", case_disjoint=True,
               normalize_labels=True, min_train_count=10, min_test_count=10,
               max_train=5000, seed=42):
    """min_test_count=10 reproduces the old harness's label space exactly
    (Plane 2 / Phase 10 / Organ 34 classes), so the only differences from the
    previously reported numbers are case-disjointness and label lowercasing.
    Use min_test_count=5 to additionally retain the Sagittal plane class and a
    longer organ tail (Plane 3 / Phase 13 / Organ 66).
    """
    """Build a Task, reporting exactly what was filtered out at each step."""
    train_map = _read_split(CSV_FILES["train"], qtype, normalize_labels)
    test_map = _read_split(CSV_FILES[split], qtype, normalize_labels)

    stats = Counter()

    # 1. Drop train images whose case also appears in test. This is what makes
    #    the kNN readout non-trivial; without it the neighbour is often the same
    #    patient's other series.
    if case_disjoint:
        test_cases = {_case_of(p) for p in test_map}
        before = len(train_map)
        train_map = {p: l for p, l in train_map.items()
                     if _case_of(p) not in test_cases}
        stats["dropped_case_overlap"] = before - len(train_map)

    # 2. Keep only samples whose volume actually exists on disk.
    def existing(mapping):
        keep, missing = {}, 0
        for p, l in mapping.items():
            if os.path.exists(img_path_to_npy(p, npy_dir)):
                keep[p] = l
            else:
                missing += 1
        return keep, missing

    train_map, miss_tr = existing(train_map)
    test_map, miss_te = existing(test_map)
    stats["dropped_missing_npy"] = miss_tr + miss_te

    # 3. Rare-class filter. A class needs enough train samples to be retrievable
    #    (min_train_count) and enough test samples for its per-class score to
    #    mean anything (min_test_count). The old harness applied a single
    #    min_count to each split independently, so the 2k-row val set decided the
    #    label space and deleted classes that are plentiful in train.
    train_counts = Counter(train_map.values())
    test_counts = Counter(test_map.values())
    frequent = {l for l, c in train_counts.items() if c >= min_train_count}
    testable = {l for l, c in test_counts.items() if c >= min_test_count}
    stats["dropped_rare"] = sum(c for l, c in train_counts.items()
                                if l not in frequent)
    classes = sorted(frequent & testable)
    classset = set(classes)

    stats["dropped_unseen"] = sum(1 for l in test_map.values()
                                  if l not in classset)

    train_items = [(p, l) for p, l in train_map.items() if l in classset]
    test_items = [(p, l) for p, l in test_map.items() if l in classset]

    # 4. Subsample train. Stratified so rare classes survive the cut, and
    #    seeded so every encoder sees the identical train set.
    if max_train and len(train_items) > max_train:
        train_items = _stratified_subsample(train_items, max_train, seed)

    train_items.sort(key=lambda x: x[0])
    test_items.sort(key=lambda x: x[0])

    return Task(
        qtype, name,
        [img_path_to_npy(p, npy_dir) for p, _ in train_items],
        [l for _, l in train_items],
        [img_path_to_npy(p, npy_dir) for p, _ in test_items],
        [l for _, l in test_items],
        classes, stats,
    )


def _stratified_subsample(items, n_total, seed):
    """Sample n_total items keeping class proportions, with >=1 per class."""
    rng = np.random.RandomState(seed)
    by_class = {}
    for p, l in items:
        by_class.setdefault(l, []).append((p, l))

    n_classes = len(by_class)
    out = []
    # Guarantee at least one sample per class, then fill proportionally.
    for l, group in sorted(by_class.items()):
        quota = max(1, int(round(n_total * len(group) / len(items))))
        quota = min(quota, len(group))
        idx = rng.permutation(len(group))[:quota]
        out.extend(group[i] for i in idx)

    if len(out) > n_total:
        # Trim from the largest classes so the >=1-per-class guarantee holds.
        counts = Counter(l for _, l in out)
        order = rng.permutation(len(out))
        trimmed, excess = [], len(out) - n_total
        for i in order:
            p, l = out[i]
            if excess > 0 and counts[l] > 1:
                counts[l] -= 1
                excess -= 1
                continue
            trimmed.append((p, l))
        out = trimmed
    return out


def build_all_tasks(**kw):
    return [build_task(qt, name, **kw) for qt, name in TASKS]


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #

def _paths_fingerprint(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(p.encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


def cached_embeddings(encoder_name, task_name, split, paths, extract_fn,
                      cache_dir=EMBED_CACHE_DIR, verbose=True):
    """Extract embeddings for `paths`, memoized on disk.

    The cache key includes a fingerprint of the exact path list, so changing the
    split or subsample invalidates it instead of silently returning stale rows.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{encoder_name}__{task_name}__{split}__{_paths_fingerprint(paths)}"
    cache_path = os.path.join(cache_dir, key + ".npy")

    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(paths):
            if verbose:
                print(f"    cache hit: {os.path.basename(cache_path)} "
                      f"{emb.shape}", flush=True)
            return emb
        print(f"    cache size mismatch, re-extracting {cache_path}", flush=True)

    emb = extract_fn(paths)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape[0] != len(paths):
        raise RuntimeError(
            f"{encoder_name}/{task_name}/{split}: extractor returned "
            f"{emb.shape[0]} embeddings for {len(paths)} paths")
    np.save(cache_path, emb)
    return emb


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def run_knn(train_emb, train_labels, test_emb, test_labels, k=20,
            weights="distance"):
    """kNN readout.

    Defaults to distance weighting: with 34 imbalanced Organ classes, uniform
    weighting over k=20 lets the head classes win nearly every vote.
    """
    k = min(k, len(train_labels))
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights=weights)
    knn.fit(train_emb, train_labels)
    preds = knn.predict(test_emb)
    return {
        "acc": accuracy_score(test_labels, preds),
        "balanced_acc": balanced_accuracy_score(test_labels, preds),
        "macro_f1": f1_score(test_labels, preds, average="macro",
                             zero_division=0),
        "weighted_f1": f1_score(test_labels, preds, average="weighted",
                                zero_division=0),
        "preds": preds,
    }


def run_linear_probe(train_emb, train_labels, test_emb, test_labels,
                     seed=42, max_iter=300):
    """Multinomial logistic-regression probe on standardized features.

    kNN failing only shows the labels are not recoverable from local cosine
    neighbourhoods. A linear probe answers the different question of whether the
    information is present in the representation at all, which is what
    distinguishes "the encoder learned nothing about organs" from "kNN is the
    wrong readout". Features are standardized (kNN's cosine metric is
    scale-free, a linear model is not) and classes are balanced, so the probe
    cannot win by predicting the prior.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_emb)
    Xtr = scaler.transform(train_emb)
    Xte = scaler.transform(test_emb)

    clf = LogisticRegression(
        max_iter=max_iter, class_weight="balanced", random_state=seed,
    )
    clf.fit(Xtr, train_labels)
    preds = clf.predict(Xte)
    return {
        "acc": accuracy_score(test_labels, preds),
        "balanced_acc": balanced_accuracy_score(test_labels, preds),
        "macro_f1": f1_score(test_labels, preds, average="macro",
                             zero_division=0),
    }


def evaluate(task, train_emb, test_emb, k=20, weights="distance"):
    """kNN metrics plus the majority baseline for context."""
    res = run_knn(train_emb, task.train_labels, test_emb, task.test_labels,
                  k=k, weights=weights)
    res.pop("preds", None)
    res["majority"] = task.majority_baseline
    res["gain_over_majority"] = res["acc"] - task.majority_baseline
    res["n_classes"] = len(task.classes)
    res["n_train"] = len(task.train_labels)
    res["n_test"] = len(task.test_labels)
    return res


def format_results(all_results, tasks):
    """Render {encoder: {task: metrics}} as a comparison table."""
    lines = []
    task_names = [t.name for t in tasks]
    by_name = {t.name: t for t in tasks}

    for metric, label in [("acc", "Accuracy"),
                          ("balanced_acc", "Balanced accuracy"),
                          ("macro_f1", "Macro-F1")]:
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"M3D frozen-feature kNN (k=20, cosine, distance-weighted) "
                     f"— {label}")
        lines.append("=" * 78)
        header = f"{'Encoder':<22}" + "".join(f"{n:>13}" for n in task_names)
        lines.append(header)
        lines.append("-" * len(header))

        if metric == "acc":
            base = f"{'majority baseline':<22}"
            for n in task_names:
                base += f"{100 * by_name[n].majority_baseline:>12.2f}%"
            lines.append(base)
            lines.append("-" * len(header))

        for enc, per_task in all_results.items():
            row = f"{enc:<22}"
            for n in task_names:
                v = per_task.get(n, {}).get(metric)
                row += f"{100 * v:>12.2f}%" if v is not None else f"{'--':>13}"
            lines.append(row)

    lines.append("")
    lines.append("Task sizes: " + ", ".join(
        f"{t.name} {len(t.train_labels)}/{len(t.test_labels)} "
        f"({len(t.classes)} cls)" for t in tasks))
    return "\n".join(lines)


if __name__ == "__main__":
    print("Building tasks under the case-disjoint protocol...\n")
    for t in build_all_tasks():
        print(t.describe())
        print()
