"""INSPECT CTPA dataset + split integrity checks.

Labels live in /data2/woa9224/inspect2:
  train_split.csv / val_split.csv / test_split.csv
      21 binary findings per impression_id (embolism, opacities, nodules,
      effusions, ...), prevalence 4.8%-26.0%. Splits are ID-disjoint.
  Final_Impressions_Labels.csv
      1/6/12-month mortality and readmission, 12-month PH, pe_acute,
      pe_subsegmentalonly, pe_positive.

WARNING -- UNVERIFIED PATIENT DISJOINTNESS. Every local CSV keys on
impression_id only; no patient identifier is present anywhere in this copy. A
patient with two CTPA studies can therefore straddle train and test, and for
retrieval/kNN evaluation that is the failure mode that inflates results most --
exactly what happened with M3D-Cap, where 77-86% of val cases also appeared in
train. Obtain patient IDs from the official INSPECT metadata and run
check_patient_disjoint() before reporting any number from these splits.
"""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = "/data2/woa9224/inspect2"
SPLITS = {s: os.path.join(ROOT, f"{s}_split.csv")
          for s in ("train", "val", "test")}
OUTCOME_CSV = os.path.join(ROOT, "Final_Impressions_Labels.csv")

# Per-encoder HU windows. Volumes are stored as raw clipped HU so each of
# these can be applied at load time instead of being baked in at extraction.
HU_WINDOWS = {
    "swinunetr":      (-175.0, 250.0),
    "voco_chest":     (-1000.0, 500.0),
    "voco_abdomen":   (-175.0, 250.0),
    "models_genesis": (-1000.0, 1000.0),
    "wide":           (-1024.0, 3071.0),   # full stored range
    "lung":           (-1350.0, 150.0),
    "mediastinum":    (-125.0, 225.0),
}


def load_split(split, npy_dir=None):
    """Return [(impression_id, label_vector)], optionally filtered to on-disk npy."""
    rows = list(csv.DictReader(open(SPLITS[split])))
    keys = [k for k in rows[0] if k != "impression_id"]
    out = []
    for r in rows:
        pid = r["impression_id"]
        if npy_dir and not os.path.exists(os.path.join(npy_dir, f"{pid}.npy")):
            continue
        out.append((pid, np.array([int(r[k]) for k in keys], dtype=np.float32)))
    return out, keys


def load_outcomes():
    """impression_id -> dict of outcome labels (mortality, readmission, PE)."""
    out = {}
    for r in csv.DictReader(open(OUTCOME_CSV)):
        pid = r.pop("impression_id")
        parsed = {}
        for k, v in r.items():
            v = v.strip()
            if v in ("TRUE", "FALSE"):
                parsed[k] = int(v == "TRUE")
            elif v.isdigit():
                parsed[k] = int(v)
        out[pid] = parsed
    return out


def check_patient_disjoint(id_to_patient):
    """Verify the provided splits do not share patients.

    id_to_patient maps impression_id -> patient_id and must come from the
    official INSPECT metadata; this copy of the data does not contain it.
    Returns {(split_a, split_b): n_shared_patients}.
    """
    members = {}
    for s in SPLITS:
        rows = list(csv.DictReader(open(SPLITS[s])))
        pats = {id_to_patient[r["impression_id"]] for r in rows
                if r["impression_id"] in id_to_patient}
        members[s] = pats

    report = {}
    names = list(members)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report[(a, b)] = len(members[a] & members[b])
    missing = sum(
        1 for s in SPLITS
        for r in csv.DictReader(open(SPLITS[s]))
        if r["impression_id"] not in id_to_patient)
    if missing:
        print(f"warning: {missing} impression_ids have no patient mapping")
    return report


class InspectCTPA(Dataset):
    """Pre-extracted INSPECT volumes, windowed to [0,1] on load.

    Volumes on disk are (1, D, H, W) int16 HU. `window` selects the HU range
    mapped to [0,1] -- pass the key matching the encoder being evaluated so
    each model sees the intensity distribution it was pretrained on.
    """

    def __init__(self, npy_dir, split="train", window="wide",
                 labels=True, transform=None):
        self.npy_dir = npy_dir
        self.window = HU_WINDOWS[window] if isinstance(window, str) else window
        self.transform = transform
        self.samples, self.label_names = (
            load_split(split, npy_dir) if labels
            else ([(os.path.basename(p)[:-4], None)
                   for p in sorted(os.listdir(npy_dir))
                   if p.endswith(".npy")], []))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, y = self.samples[idx]
        # A failed load must not silently become a zero volume paired with a
        # real label -- that quietly poisons both training and evaluation.
        vol = np.load(os.path.join(self.npy_dir, f"{pid}.npy"))
        lo, hi = self.window
        vol = np.clip((vol.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        x = torch.from_numpy(vol)
        if self.transform is not None:
            x = self.transform(x)
        return (x, torch.from_numpy(y)) if y is not None else x


if __name__ == "__main__":
    import sys
    npy_dir = sys.argv[1] if len(sys.argv) > 1 else None
    for s in SPLITS:
        rows, keys = load_split(s, npy_dir)
        print(f"{s:6s} {len(rows):6d} volumes"
              + (f"  ({len(keys)} labels)" if keys else ""))
    o = load_outcomes()
    print(f"outcomes for {len(o)} impressions: {sorted(next(iter(o.values())))}")
