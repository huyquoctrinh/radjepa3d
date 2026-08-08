"""Stage 4: Quality control — schema check, leakage detection, fabrication detection."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import re
from config import QTYPE_LABELS

MEASUREMENT_RE = re.compile(r"\d+\.?\d*\s*(?:mm|cm|HU|cc|mL)", re.IGNORECASE)


def check_schema(conv: dict) -> list[str]:
    """Verify the generated conversation matches the expected schema."""
    flags = []
    if "turns" not in conv:
        flags.append("missing_turns_key")
        return flags

    for i, turn in enumerate(conv["turns"]):
        prefix = f"turn_{i}"
        for key in ("turn", "type", "doctor", "think", "tool_calls", "response"):
            if key not in turn:
                flags.append(f"{prefix}_missing_{key}")
        if "tool_calls" in turn and not isinstance(turn["tool_calls"], list):
            flags.append(f"{prefix}_tool_calls_not_list")
    return flags


def check_leakage(conv: dict, impression: str | None) -> list[str]:
    """Detect if the model leaked the withheld impression into reasoning."""
    flags = []
    if not impression:
        return flags

    imp_lower = impression.lower()
    imp_words = set(w for w in re.findall(r"\b\w+\b", imp_lower) if len(w) > 4)

    for i, turn in enumerate(conv.get("turns", [])):
        think = turn.get("think", "").lower()
        if not think:
            continue
        think_words = set(re.findall(r"\b\w+\b", think))
        overlap = imp_words & think_words
        overlap_ratio = len(overlap) / len(imp_words) if imp_words else 0
        if overlap_ratio > 0.5:
            flags.append(f"turn_{i}_possible_leakage(overlap={overlap_ratio:.0%})")

    return flags


def check_fabrication(conv: dict, findings: str) -> list[str]:
    """Flag measurements in reasoning that don't appear in the source findings."""
    flags = []
    findings_measurements = set(
        m.group().lower() for m in MEASUREMENT_RE.finditer(findings)
    )

    for i, turn in enumerate(conv.get("turns", [])):
        text = turn.get("think", "") + " " + turn.get("response", "")
        for m in MEASUREMENT_RE.finditer(text):
            val = m.group().lower()
            if val not in findings_measurements:
                flags.append(f"turn_{i}_fabricated_measurement({val})")

    return flags


def format_with_think_tags(conv: dict) -> list[dict]:
    """Post-process: merge think + response into agent field with <think> tags."""
    formatted = []
    for turn in conv.get("turns", []):
        think = turn.get("think", "")
        response = turn.get("response", "")
        agent_text = f"<think>\n{think}\n</think>\n\n{response}"
        formatted.append({
            "turn": turn.get("turn"),
            "type": turn.get("type"),
            "doctor": turn.get("doctor"),
            "tool_calls": turn.get("tool_calls", []),
            "agent": agent_text,
        })
    return formatted


def run_qc(conv: dict, findings: str, impression: str | None) -> dict:
    """Run all QC checks on a generated conversation."""
    schema_flags = check_schema(conv)
    leakage_flags = check_leakage(conv, impression)
    fabrication_flags = check_fabrication(conv, findings)

    all_flags = schema_flags + leakage_flags + fabrication_flags
    return {
        "parse_ok": len(schema_flags) == 0,
        "n_turns": len(conv.get("turns", [])),
        "flags": all_flags,
    }


def print_qc_report(results: list[dict]):
    """Print aggregate QC report."""
    total = len(results)
    if total == 0:
        print("No results to report.")
        return

    parse_ok = sum(1 for r in results if r["qc"]["parse_ok"])
    flagged = sum(1 for r in results if r["qc"]["flags"])
    leakage = sum(1 for r in results
                  if any("leakage" in f for f in r["qc"]["flags"]))
    fabrication = sum(1 for r in results
                      if any("fabricated" in f for f in r["qc"]["flags"]))

    print(f"\n{'='*60}")
    print(f"QC Report — {total} conversations")
    print(f"{'='*60}")
    print(f"  Clean-parse rate:       {parse_ok}/{total} ({100*parse_ok/total:.1f}%)")
    print(f"  Flagged (any issue):    {flagged}/{total} ({100*flagged/total:.1f}%)")
    print(f"  Answer-leakage:         {leakage}/{total} ({100*leakage/total:.1f}%)")
    print(f"  Fabricated measurement: {fabrication}/{total} ({100*fabrication/total:.1f}%)")

    avg_turns = sum(r["qc"]["n_turns"] for r in results) / total
    print(f"  Avg turns per conv:     {avg_turns:.1f}")

    # List samples with issues
    problem_ids = [r["id"] for r in results if r["qc"]["flags"]]
    if problem_ids:
        print(f"\n  Flagged IDs: {problem_ids[:20]}")
        if len(problem_ids) > 20:
            print(f"  ... and {len(problem_ids) - 20} more")
