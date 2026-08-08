"""Dataset for loading pre-extracted 3D CT volumes from .npy files."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MergedSSLDataset(Dataset):
    """M3D-Cap + INSPECT CTPA for joint SSL pretraining.

    The two corpora do not share an intensity convention. M3D-Cap is built from
    JPEG slices already min-max scaled to [0,1] per volume, with no recoverable
    HU; INSPECT stores real int16 Hounsfield units. To avoid handing the encoder
    two different conventions, the *same procedure* -- per-volume min-max -- is
    applied to both. For INSPECT this is nearly a fixed [-1024, 3071] window
    anyway, since ~80% of volumes contain both air and dense bone/contrast and
    therefore hit both clip bounds.

    A residual domain gap remains: M3D's JPEGs were windowed by whoever authored
    them, so identical tissue still maps to different values across the two
    sources. That is inherent to the merge and cannot be normalized away -- it
    is a reason to watch for the encoder partitioning by source rather than by
    anatomy.

    `inspect_repeat` oversamples INSPECT, which is ~5x smaller (23k vs 120k).
    Default 1 (no oversampling) keeps the natural 84/16 mix.
    """

    def __init__(self, m3d_dataset, inspect_dir, inspect_repeat=1):
        self.m3d = m3d_dataset
        insp = sorted(os.path.join(inspect_dir, f)
                      for f in os.listdir(inspect_dir)
                      if f.endswith(".npy") and not f.startswith("_"))
        self.inspect = insp * max(1, int(inspect_repeat))
        self.n_m3d = len(self.m3d)

    def __len__(self):
        return self.n_m3d + len(self.inspect)

    def source_counts(self):
        return {"m3d": self.n_m3d, "inspect": len(self.inspect)}

    def __getitem__(self, idx):
        if idx < self.n_m3d:
            return self.m3d[idx]
        vol = np.load(self.inspect[idx - self.n_m3d]).astype(np.float32)
        lo, hi = float(vol.min()), float(vol.max())
        vol = (vol - lo) / max(hi - lo, 1e-6)     # same per-volume min-max
        return torch.from_numpy(vol)


class InspectSSLDataset(Dataset):
    """INSPECT CTPA volumes for SSL pretraining.

    Volumes on disk are (1, D, H, W) int16 Hounsfield units, unlike M3D's
    per-volume min-max [0,1]. A fixed HU window is applied here so the
    normalization is absolute and identical across scans -- two volumes with the
    same tissue map to the same value, which per-volume min-max cannot
    guarantee. The default window spans lung through bone so a single encoder
    sees both parenchyma and mediastinum.

    A volume that fails to load raises rather than yielding zeros: a silent
    all-zero sample is a valid-looking training target and would quietly
    corrupt the JEPA objective.
    """

    def __init__(self, npy_dir, hu_window=(-1000.0, 1000.0), ids=None):
        self.npy_dir = npy_dir
        self.lo, self.hi = hu_window
        if ids is not None:
            self.samples = [os.path.join(npy_dir, f"{i}.npy") for i in ids]
            self.samples = [p for p in self.samples if os.path.exists(p)]
        else:
            self.samples = sorted(
                os.path.join(npy_dir, f) for f in os.listdir(npy_dir)
                if f.endswith(".npy") and not f.startswith("_"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol = np.load(self.samples[idx]).astype(np.float32)
        vol = np.clip((vol - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        return torch.from_numpy(vol)


class M3DCTDataset(Dataset):

    def __init__(self, config, split="train"):
        self.config = config
        self.volume_size = config.volume_size
        self.npy_dir = config.npy_dir

        with open(config.data_json) as f:
            data = json.load(f)
        entries = data[split]

        self.samples = []
        for entry in entries:
            img_path = entry["image"]
            # M3D_Cap_npy/ct_case/007789/Axial_Zoomed___thin_cuts.npy
            # -> extracted_M3D_Cap/ct_case/007789/Axial_Zoomed___thin_cuts.npy
            parts = img_path.split("/")
            npy_path = os.path.join(self.npy_dir, parts[1], parts[2], parts[3])
            if os.path.exists(npy_path):
                self.samples.append(npy_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            volume = np.load(self.samples[idx]).astype(np.float32)
            return torch.from_numpy(volume)
        except Exception:
            return torch.zeros(1, *self.volume_size)
