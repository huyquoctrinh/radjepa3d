"""Benchmark StanfordMIMI Merlin (I3-ResNet CT encoder) on M3D VQA tasks.

Frozen-feature kNN (k=20) on Plane / Phase / Organ — same protocol as
vjepa_benchmark, for direct comparison against the V-JEPA SSL encoders.

Caveat: M3D ships normalized (1,32,256,256) npy (no HU / spacing), so Merlin's
HU-window + spacing steps cannot be applied; volumes are best-effort resized to
Merlin's 224x224x160 input with intensities already in [0,1].

Usage:
    CUDA_VISIBLE_DEVICES=4 python benchmark_merlin.py
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

# Merlin package (cloned repo) + our benchmark helpers
sys.path.insert(0, "/tmp/claude-1079/-data2-gdp2988-3d-cot/"
                   "694ec789-20df-4d2a-921f-32cddeec2d71/scratchpad/Merlin")
from merlin import Merlin
from vjepa_benchmark import CSV_FILES, load_labels, img_path_to_npy, run_knn

NPY_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"
_CACHE = {}  # npy path -> 2048-d embedding (np.float32)


def _prep(vol):
    """(1,32,256,256) normalized npy -> (1,1,224,224,160) tensor for Merlin."""
    t = torch.from_numpy(vol.astype(np.float32))          # (C,D,H,W)=(1,32,256,256)
    t = t.permute(0, 2, 3, 1).unsqueeze(0)                # (1,1,256,256,32)=(N,C,H,W,D)
    t = F.interpolate(t, size=(224, 224, 160), mode="trilinear", align_corners=False)
    return t.squeeze(0)                                   # (1,224,224,160)


@torch.no_grad()
def extract(model, paths, batch_size=4, device="cuda"):
    todo = [p for p in paths if p not in _CACHE]
    for i in range(0, len(todo), batch_size):
        batch_paths = todo[i:i + batch_size]
        vols = []
        for p in batch_paths:
            try:
                vols.append(_prep(np.load(p)))
            except Exception:
                vols.append(torch.zeros(1, 224, 224, 160))
        batch = torch.stack(vols).to(device)              # (B,1,224,224,160)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)                            # (1,B,2048)
        emb = out.squeeze(0).float().cpu().numpy()
        for p, e in zip(batch_paths, emb):
            _CACHE[p] = e
        if (i // batch_size) % 25 == 0:
            print(f"  extracted {min(i + batch_size, len(todo))}/{len(todo)} new",
                  flush=True)
    return np.stack([_CACHE[p] for p in paths])


def main():
    print("Loading Merlin (ImageEmbedding=True)...", flush=True)
    model = Merlin(ImageEmbedding=True).cuda().eval()

    results = {}
    for qtype, task in [("1", "Plane"), ("2", "Phase"), ("3", "Organ")]:
        train_labels = load_labels(CSV_FILES["train"], qtype)
        test_labels = load_labels(CSV_FILES["val"], qtype)

        tr_paths, tr_labs = [], []
        for img, lab in train_labels.items():
            npy = img_path_to_npy(img, NPY_DIR)
            if os.path.exists(npy):
                tr_paths.append(npy); tr_labs.append(lab)
        te_paths, te_labs = [], []
        for img, lab in test_labels.items():
            npy = img_path_to_npy(img, NPY_DIR)
            if os.path.exists(npy):
                te_paths.append(npy); te_labs.append(lab)

        common = sorted(set(tr_labs) & set(te_labs))
        tr_paths = [p for p, l in zip(tr_paths, tr_labs) if l in common][:5000]
        tr_labs = [l for l in tr_labs if l in common][:5000]
        te_paths = [p for p, l in zip(te_paths, te_labs) if l in common]
        te_labs = [l for l in te_labs if l in common]

        print(f"[{task}] train={len(tr_paths)} test={len(te_paths)} "
              f"classes={len(common)}", flush=True)
        tr_emb = extract(model, tr_paths)
        te_emb = extract(model, te_paths)
        acc, f1, _ = run_knn(tr_emb, tr_labs, te_emb, te_labs)
        results[task] = acc
        print(f"  {task}: kNN acc={acc:.4f} f1={f1:.4f}", flush=True)

    print("\n" + "=" * 50)
    print("Merlin — M3D kNN (k=20)")
    print("=" * 50)
    mean = sum(results.values()) / len(results)
    for task in ["Plane", "Phase", "Organ"]:
        print(f"  {task:<8} {100*results[task]:.2f}%")
    print(f"  {'Mean':<8} {100*mean:.2f}%")


if __name__ == "__main__":
    main()
