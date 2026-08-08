"""Frozen-feature evaluation on INSPECT CTPA: 21 multi-label findings.

Differs from m3d_eval.py in two ways that matter.

Multi-label, not multi-class. Each scan carries any subset of 21 findings
(prevalence 4.8%-26.0%), so accuracy is meaningless -- predicting all-negative
scores ~85%. Metrics are macro AUROC (chance = 0.500 regardless of prevalence)
and mAP (chance = prevalence, so it is reported per label).

Real Hounsfield units. Volumes are stored as int16 HU, so each encoder is fed
its own pretraining window rather than a shared [0,1] min-max. This is the
comparison M3D-Cap could not support: there, every baseline ran out of
distribution while RadJepa trained on exactly the evaluated representation.

    Swin UNETR      [-175, 250]
    VoCo (chest)    [-1000, 500]
    Models Genesis  [-1000, 1000]
    RadJepa/INSPECT [-1000, 1000]   (its pretraining window)

CAVEAT -- the train/val/test splits ship without patient identifiers, so their
patient-disjointness is unverified. Comparing checkpoints of one model is fine
(any leakage is shared), but cross-encoder numbers should not be reported until
inspect_data.check_patient_disjoint() has been run with the official metadata.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test/eval_inspect.py --encoders radjepa-inspect:99,radjepa-inspect:199
    CUDA_VISIBLE_DEVICES=0 python test/eval_inspect.py --encoders swinunetr-ssl,voco-b,models-genesis
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import hashlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

torch.backends.cudnn.enabled = False

import encoders as E
import inspect_data as ID

NPY_DIR = "/data2/gdp2988/3d_cot/extracted_INSPECT/32x256x256"
CACHE = "/data2/gdp2988/3d_cot/checkpoints/embeddings_inspect"
RADJEPA_CKPT_DIR = ("/data2/gdp2988/3d_cot/checkpoints/"
                    "radjepa_inspect_hybrid_token")
MERGED_CKPT_DIR = ("/data2/gdp2988/3d_cot/checkpoints/"
                   "radjepa_merged_hybrid_token")

# Which HU window each encoder was pretrained on.
ENCODER_WINDOW = {
    "swinunetr-ssl": "swinunetr",
    "voco-b": "voco_chest",
    "models-genesis": "models_genesis",
}
DEFAULT_WINDOW = "models_genesis"       # [-1000, 1000], RadJepa's window


def fingerprint(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(p.encode())
    return h.hexdigest()[:12]


def cached(name, split, paths, extract):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{name}__{split}__{fingerprint(paths)}.npy")
    if os.path.exists(f):
        emb = np.load(f)
        if emb.shape[0] == len(paths):
            return emb
    emb = np.asarray(extract(paths), dtype=np.float32)
    if emb.shape[0] != len(paths):
        raise RuntimeError(f"{name}/{split}: {emb.shape[0]} embeddings "
                           f"for {len(paths)} paths")
    np.save(f, emb)
    return emb


def build_radjepa_inspect(epoch, batch_size=32, device="cuda",
                          ckpt_dir=None):
    """Load a checkpoint from the INSPECT pretraining run."""
    from load_checkpoint import load_encoder, encoder_extract_fn

    root = ckpt_dir or RADJEPA_CKPT_DIR
    path = (os.path.join(root, f"checkpoint_epoch{epoch:04d}.pt")
            if isinstance(epoch, int) else os.path.join(root, epoch))

    enc, cfg, meta = load_encoder(
        path, device=device, router_mode="token", batch_size=batch_size)

    return encoder_extract_fn(enc, device=device, batch_size=batch_size)


def make_loader(window):
    """Path -> windowed float tensor, matching E.batched_extract's contract."""
    lo, hi = ID.HU_WINDOWS[window]

    def load(path):
        vol = np.load(path).astype(np.float32)
        return torch.from_numpy(np.clip((vol - lo) / (hi - lo), 0.0, 1.0))
    return load


def evaluate(tr_emb, tr_y, te_emb, te_y, names, k=20):
    """Macro AUROC / mAP for kNN and a linear probe."""
    out = {}

    # kNN: score each label by the mean label vector of its k nearest
    # neighbours under cosine distance.
    a = tr_emb / (np.linalg.norm(tr_emb, axis=1, keepdims=True) + 1e-8)
    b = te_emb / (np.linalg.norm(te_emb, axis=1, keepdims=True) + 1e-8)
    sim = b @ a.T
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    knn_scores = tr_y[idx].mean(axis=1)

    scaler = StandardScaler().fit(tr_emb)
    Xtr, Xte = scaler.transform(tr_emb), scaler.transform(te_emb)
    lp_scores = np.zeros_like(knn_scores)
    for j in range(tr_y.shape[1]):
        if tr_y[:, j].sum() < 5 or te_y[:, j].sum() < 1:
            continue
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(Xtr, tr_y[:, j])
        lp_scores[:, j] = clf.predict_proba(Xte)[:, 1]

    for tag, s in [("knn", knn_scores), ("lp", lp_scores)]:
        aurocs, aps = [], []
        per = {}
        for j, nm in enumerate(names):
            if te_y[:, j].sum() == 0 or te_y[:, j].sum() == len(te_y):
                continue
            au = roc_auc_score(te_y[:, j], s[:, j])
            ap = average_precision_score(te_y[:, j], s[:, j])
            aurocs.append(au); aps.append(ap); per[nm] = (au, ap)
        out[tag] = {"macro_auroc": float(np.mean(aurocs)),
                    "mAP": float(np.mean(aps)), "per_label": per}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", required=True,
                   help="Comma-separated. 'radjepa-inspect:<epoch>' for this "
                        "run's checkpoints, or a name from encoders.py")
    p.add_argument("--npy-dir", default=NPY_DIR)
    p.add_argument("--max-train", type=int, default=8000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    train, names = ID.load_split("train", args.npy_dir)
    test, _ = ID.load_split("test", args.npy_dir)
    if args.max_train and len(train) > args.max_train:
        rng = np.random.RandomState(42)
        sel = rng.permutation(len(train))[:args.max_train]
        train = [train[i] for i in sel]

    tr_paths = [os.path.join(args.npy_dir, f"{i}.npy") for i, _ in train]
    te_paths = [os.path.join(args.npy_dir, f"{i}.npy") for i, _ in test]
    tr_y = np.stack([y for _, y in train])
    te_y = np.stack([y for _, y in test])
    prev = te_y.mean(0)
    print(f"train={len(tr_paths)} test={len(te_paths)} labels={len(names)}")
    print(f"chance: AUROC 0.500 | mAP = prevalence "
          f"(mean {prev.mean():.3f})\n", flush=True)

    results = {}
    for spec in [s.strip() for s in args.encoders.split(",")]:
        print("=" * 66)
        print(spec)
        print("=" * 66, flush=True)
        try:
            if spec.startswith(("radjepa-inspect:", "radjepa-merged:")):
                ep = spec.split(":", 1)[1]
                ep = int(ep) if ep.isdigit() else ep
                root = (MERGED_CKPT_DIR if spec.startswith("radjepa-merged:")
                        else RADJEPA_CKPT_DIR)
                extract = build_radjepa_inspect(ep, args.batch_size,
                                                ckpt_dir=root)
                window = DEFAULT_WINDOW
            else:
                spec_d = E.BASELINE_VARIANTS[spec]
                extract = E.build_baseline(
                    spec_d, batch_size=spec_d.get("batch_size", 8))
                window = ENCODER_WINDOW.get(spec, DEFAULT_WINDOW)
            print(f"  HU window: {window} {ID.HU_WINDOWS[window]}", flush=True)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        # Feed each encoder its own pretraining window.
        E.load_volume = make_loader(window)
        key = f"{spec.replace(':', '-')}__{window}"
        tr = cached(key, "train", tr_paths, extract)
        te = cached(key, "test", te_paths, extract)
        r = evaluate(tr, tr_y, te, te_y, names, k=args.k)
        results[spec] = r
        print(f"  kNN: macroAUROC={r['knn']['macro_auroc']:.4f} "
              f"mAP={r['knn']['mAP']:.4f}")
        print(f"  LP : macroAUROC={r['lp']['macro_auroc']:.4f} "
              f"mAP={r['lp']['mAP']:.4f}", flush=True)
        del extract
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"{'Encoder':<26}{'kNN AUROC':>12}{'kNN mAP':>10}"
          f"{'LP AUROC':>12}{'LP mAP':>10}")
    print("-" * 78)
    for k_, r in results.items():
        print(f"{k_:<26}{r['knn']['macro_auroc']:>12.4f}{r['knn']['mAP']:>10.4f}"
              f"{r['lp']['macro_auroc']:>12.4f}{r['lp']['mAP']:>10.4f}")
    print("\nSplits' patient-disjointness is UNVERIFIED -- see inspect_data.py")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
