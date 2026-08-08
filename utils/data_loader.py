"""Stage 0-1: Audit CSVs and group MCQs by image path."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import csv
import re
from collections import defaultdict
from typing import Optional

from config import ALL_CSVS, QTYPE_LABELS, TURN_ORDER, IMPRESSION_MARKERS


def load_csv(paths=None):
    if paths is None:
        paths = ALL_CSVS
    if isinstance(paths, (str, type(ALL_CSVS[0]))):
        paths = [paths]
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def audit(rows):
    """Stage 0: Print schema report."""
    print(f"Total rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys())}")

    # Unique images
    images = {r["Image Path"] for r in rows}
    print(f"Unique images: {len(images)}")

    # Qtype distribution
    qtypes = defaultdict(int)
    for r in rows:
        qtypes[r["Question Type"]] += 1
    print("\nQuestion Type distribution:")
    for qt in sorted(qtypes):
        label = QTYPE_LABELS.get(qt, "unknown")
        print(f"  Type {qt} ({label}): {qtypes[qt]}")

    # Impression separability
    marker_re = re.compile(IMPRESSION_MARKERS)
    seen = set()
    total = separable = 0
    for r in rows:
        img = r["Image Path"]
        if img in seen:
            continue
        seen.add(img)
        total += 1
        if marker_re.search(r["Text"]):
            separable += 1
    pct = 100 * separable / total if total else 0
    print(f"\nReports with separable impression: {separable}/{total} ({pct:.1f}%)")
    if pct < 40:
        print("  WARNING: Low separability — pipeline will use LLM-based splitting")
    return {"n_rows": len(rows), "n_images": len(images), "separable_pct": pct}


def split_report(text: str) -> tuple[str, Optional[str]]:
    """Split report into (findings, impression).

    Returns (full_text, None) when no impression marker is found.
    """
    marker_re = re.compile(IMPRESSION_MARKERS)
    m = marker_re.search(text)
    if m:
        findings = text[: m.start()].strip()
        impression = text[m.start():].strip()
        if findings and impression:
            return findings, impression
    return text.strip(), None


def group_by_image(rows) -> list[dict]:
    """Stage 1: Group rows by Image Path, order turns by qtype."""
    groups = defaultdict(lambda: {"image_path": "", "text": "", "turns": []})
    for row in rows:
        img = row["Image Path"]
        g = groups[img]
        g["image_path"] = img
        g["text"] = row["Text"]
        g["turns"].append({
            "qtype": row["Question Type"],
            "question": row["Question"],
            "choices": {
                "A": row["Choice A"],
                "B": row["Choice B"],
                "C": row["Choice C"],
                "D": row["Choice D"],
            },
            "gt_answer": row["Answer"],
            "gt_choice": row["Answer Choice"],
        })

    result = []
    for img, g in groups.items():
        # Sort turns by reading order
        g["turns"].sort(key=lambda t: TURN_ORDER.index(t["qtype"])
                        if t["qtype"] in TURN_ORDER else 99)
        # Split report
        findings, impression = split_report(g["text"])
        g["findings"] = findings
        g["impression"] = impression
        result.append(g)

    return result


if __name__ == "__main__":
    rows = load_csv()
    audit(rows)
