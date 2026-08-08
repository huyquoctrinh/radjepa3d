#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from typing import Dict, List, Any


METRICS = ["DIR", "DIST", "EXT", "VOL", "COMP", "Unknown"]


def normalize_question(q: str) -> str:
    q = (q or "").lower()
    q = q.replace("–", "-").replace("—", "-")
    q = q.replace("cm³", "cm3").replace("cm^3", "cm3").replace("cm\\u00b3", "cm3")
    q = re.sub(r"([abcd])\.", r"\1. ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


RANGE_RE = re.compile(r"\b\d+(\.\d+)?\s*-\s*\d+(\.\d+)?\b")
UNIT_RE  = re.compile(r"\b\d+(\.\d+)?\s*(cm3|cc|ml)\b")
PCT_RE   = re.compile(r"\b\d+(\.\d+)?\s*%\b")
RATIO_RE = re.compile(r"\b\d+(\.\d+)?\s*:\s*\d+(\.\d+)?\b")
NUM_THRESHOLD_CUE_RE = re.compile(
    r"\b(above|below|greater than|less than|exceed|exceeds|exceeding|under|over)\b.*\b\d+(\.\d+)?\b"
)
TWO_VOLUME_OF_RE = re.compile(r"volume of .* volume of")
CM_LINEAR_RE = re.compile(r"(?<!\d)\bcm\b(?!\d)")
MM_LINEAR_RE = re.compile(r"(?<!\d)\bmm\b(?!\d)")


def classify_metric(question: str) -> str:
    q = normalize_question(question)

    if any(k in q for k in [
        "center-to-center", "center to center",
        "distance", "euclidean",
        "how far", "farther", "closer", "closest", "nearest",
        "proximity",
    ]):
        return "DIST"

    if any(k in q for k in [
        "superior", "inferior", "anterior", "posterior",
        "medial", "lateral", "left of", "right of",
        "relative position", "located", "positioned",
    ]):
        return "DIR"
    if ("above the" in q or "below the" in q) and not NUM_THRESHOLD_CUE_RE.search(q):
        return "DIR"

    has_extent_cue = any(k in q for k in [
        "bounding box", "bbox", "x-axis", "y-axis", "z-axis",
        "width", "height", "depth", "extent", "diameter", "radius", "size",
    ]) or bool(CM_LINEAR_RE.search(q)) or bool(MM_LINEAR_RE.search(q))

    has_prop_cue = any(k in q for k in [
        "ratio", "proportion", "fraction", "percentage", "percent",
        "times larger", "times the volume",
    ]) or bool(RATIO_RE.search(q)) or bool(PCT_RE.search(q))

    if has_extent_cue and not has_prop_cue:
        return "EXT"

    if has_prop_cue:
        return "COMP"

    if any(k in q for k in [
        "compare", "comparison", "larger than", "smaller than",
        "greater than", "less than", "approximately equal", "same size",
        "largest", "smallest", "increase", "decrease", "changed",
        "previous scan", "across scans",
    ]):
        return "COMP"

    if TWO_VOLUME_OF_RE.search(q):
        return "COMP"

    vol_keywords = [
        "estimate", "volume", "within", "range",
        "threshold", "cutoff", "combined", "total", "sum", "average",
    ]
    if any(k in q for k in vol_keywords) and (
        "volume" in q or UNIT_RE.search(q) or RANGE_RE.search(q) or NUM_THRESHOLD_CUE_RE.search(q)
    ):
        return "VOL"

    if RANGE_RE.search(q) or UNIT_RE.search(q) or NUM_THRESHOLD_CUE_RE.search(q):
        return "VOL"

    return "Unknown"
