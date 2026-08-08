"""Pre-extract INSPECT CTPA volumes from NIfTI to compact .npy.

Source: /data2/woa9224/inspect2/CTPA -- 23,227 volumes, 2.0 TB gzipped NIfTI,
512x512 in-plane, depth 233-556, spacing ~0.5-0.8 xy / 0.6-1.3 z, int16 HU.

KEY DIFFERENCE FROM extract_npy.py: this keeps Hounsfield units.

M3D-Cap ships JPEG slices, so extract_npy.py had no choice but to min-max scale
each volume to [0,1], which destroys HU and forces every pretrained baseline
(Swin UNETR, VoCo, Models Genesis, Merlin) to run out of distribution while
RadJepa trains on exactly that representation. INSPECT has real HU, so we store
clipped int16 HU and let each consumer apply its own window at load time:

    Swin UNETR      [-175, 250]
    VoCo (chest)    [-1000, 500]
    Models Genesis  [-1000, 1000]
    RadJepa         whatever you choose -- see hu_to_unit()

int16 costs the same 2 bytes/voxel as float16, so this is free.

Usage:
    python extract_inspect.py --out /path/to/extracted_INSPECT --workers 32
    python extract_inspect.py --out ... --shape 64 256 256
    python extract_inspect.py --out ... --limit 20 --workers 4   # smoke test
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np

SRC = "/data2/woa9224/inspect2/CTPA"

# Air is -1000 HU; -1024 is the usual int16 floor. Dense bone / contrast tops
# out around 3071. Clipping here loses nothing clinically and bounds the range
# so int16 storage is exact.
HU_MIN, HU_MAX = -1024, 3071

# Everything below this is air/table; used to find the body for cropping.
BODY_THRESHOLD = -500


def hu_to_unit(vol, lo, hi):
    """Window HU to [0,1]. Apply at load time, per encoder -- not here."""
    return np.clip((vol.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _crop_body(vol):
    """Trim air margins so the resize spends resolution on anatomy.

    Conservative: only crops rows/cols/slices that are entirely below the body
    threshold, so it cannot cut into tissue.
    """
    mask = vol > BODY_THRESHOLD
    if not mask.any():
        return vol
    out = []
    for axis in range(3):
        idx = np.where(mask.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        out.append((int(idx[0]), int(idx[-1]) + 1))
    (d0, d1), (h0, h1), (w0, w1) = out
    return vol[d0:d1, h0:h1, w0:w1]


def _resize(vol, shape):
    """Trilinear resample to `shape`, done in float32 then returned as HU."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=tuple(shape), mode="trilinear",
                      align_corners=False)
    return t[0, 0].numpy()


def process(args):
    """Decode one NIfTI once and emit every requested resolution.

    Decoding a 0.77 GB gzipped volume costs ~1.6 s and dominates the job, so
    reading the 2 TB source once and resizing N times is far cheaper than N
    separate passes.
    """
    path, targets, do_crop = args           # targets: [(out_dir, shape), ...]
    pid = os.path.basename(path)[:-7] if path.endswith(".nii.gz") else \
        os.path.splitext(os.path.basename(path))[0]

    pending = [(d, s) for d, s in targets
               if not os.path.exists(os.path.join(d, f"{pid}.npy"))]
    if not pending:
        return "skip", pid

    try:
        import nibabel as nib
        img = nib.load(path)
        vol = np.asanyarray(img.dataobj)
        if vol.ndim != 3:
            return f"bad_ndim:{vol.ndim}", pid

        # NIfTI here is (H, W, D); move depth first to match (D, H, W).
        vol = np.transpose(vol, (2, 0, 1))

        vol = np.clip(vol, HU_MIN, HU_MAX)
        if do_crop:
            vol = _crop_body(vol)
            if min(vol.shape) < 8:
                return "too_small_after_crop", pid

        for out_dir, shape in pending:
            v = _resize(vol, shape)
            v = np.clip(np.rint(v), HU_MIN, HU_MAX).astype(np.int16)
            v = v[None]                      # (1, D, H, W), matches M3D layout
            out_path = os.path.join(out_dir, f"{pid}.npy")
            tmp = out_path + ".tmp.npy"
            np.save(tmp, v)
            os.replace(tmp, out_path)        # atomic; safe to interrupt
        return "ok", pid
    except Exception as e:
        return f"error:{type(e).__name__}:{e}", pid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", required=True,
                   help="Base output directory; one subdir per resolution")
    p.add_argument("--shape", type=int, nargs=3, action="append",
                   default=None, metavar=("D", "H", "W"),
                   help="Repeat to extract several resolutions in one pass")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N volumes (smoke test)")
    p.add_argument("--no-crop", action="store_true",
                   help="Skip body cropping (keep the full field of view)")
    args = p.parse_args()

    import glob
    files = sorted(glob.glob(os.path.join(args.src, "*.nii.gz")))
    if args.limit:
        files = files[:args.limit]

    shapes = [tuple(s) for s in (args.shape or [[32, 256, 256]])]
    targets = []
    total_gb = 0.0
    for d, h, w in shapes:
        sub = os.path.join(args.out, f"{d}x{h}x{w}")
        os.makedirs(sub, exist_ok=True)
        targets.append((sub, (d, h, w)))
        total_gb += len(files) * 2 * d * h * w / 1e9

    print(f"source      : {args.src}  ({len(files)} volumes)")
    print(f"output      : {args.out}")
    for sub, (d, h, w) in targets:
        print(f"  -> {sub}  (1, {d}, {h}, {w}) int16 HU "
              f"[{HU_MIN}, {HU_MAX}]  {len(files) * 2 * d * h * w / 1e9:.0f} GB")
    print(f"crop body   : {not args.no_crop}")
    print(f"est. total  : {total_gb:.0f} GB")
    print(f"workers     : {args.workers}\n", flush=True)

    todo = [(f, targets, not args.no_crop) for f in files]
    stats = Counter()
    failures = []
    try:
        from tqdm import tqdm
        it = tqdm(total=len(todo), desc="extracting")
    except ImportError:
        it = None

    with Pool(processes=args.workers) as pool:
        for status, pid in pool.imap_unordered(process, todo, chunksize=4):
            key = status.split(":")[0]
            stats[key] += 1
            if key not in ("ok", "skip"):
                failures.append((pid, status))
            if it:
                it.update(1)
    if it:
        it.close()

    print("\nDone.")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    if failures:
        log = os.path.join(args.out, "_failures.txt")
        with open(log, "w") as f:
            for pid, s in failures:
                f.write(f"{pid}\t{s}\n")
        print(f"  wrote {len(failures)} failures to {log}")


if __name__ == "__main__":
    main()
