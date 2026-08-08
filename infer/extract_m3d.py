"""Pre-extract 3D CT volumes from zip archives to .npy files."""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import io
import json
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import monai.transforms as mtf

DATA_JSON = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/M3D_Cap.json"
CT_CASE_DIR = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/ct_case"
CT_QUIZZE_DIR = "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/ct_quizze"
OUT_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"

VOLUME_SIZE = [32, 256, 256]

transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=VOLUME_SIZE, mode="bilinear"),
])


def build_zip_index():
    """Map (ct_type, case_id) -> zip_path."""
    case_to_zip = {}
    for ct_dir in [CT_CASE_DIR, CT_QUIZZE_DIR]:
        ct_type = os.path.basename(ct_dir)
        zip_files = sorted(Path(ct_dir).glob("*.zip"))
        for zf_path in zip_files:
            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    dirs = set()
                    for name in zf.namelist():
                        top = name.split("/")[0]
                        if top:
                            dirs.add(top)
                    for case_id in dirs:
                        case_to_zip[(ct_type, case_id)] = str(zf_path)
            except Exception as e:
                print(f"Error scanning {zf_path}: {e}")
    return case_to_zip


def load_volume_from_zip(zip_path, case_id, series):
    """Load a single volume from a zip archive and return as numpy array."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"{case_id}/{series}/"
        image_files = []
        for name in zf.namelist():
            if name.startswith(prefix) and (name.endswith(".jpeg") or name.endswith(".png")):
                fname = name.split("/")[-1]
                try:
                    num = int(os.path.splitext(fname)[0])
                    image_files.append((num, name))
                except ValueError:
                    continue
        image_files.sort(key=lambda x: x[0])

        if not image_files:
            return None

        slices = []
        for _, name in image_files:
            try:
                img = Image.open(io.BytesIO(zf.read(name))).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                slices.append(arr[None])
            except Exception:
                continue

        if not slices:
            return None

        shapes = [s.shape for s in slices]
        most_common = Counter(shapes).most_common(1)[0][0]
        slices = [s for s in slices if s.shape == most_common]

        if not slices:
            return None

        volume = np.stack(slices, axis=1)  # (1, D, H, W)
        volume = volume - volume.min()
        volume = volume / np.clip(volume.max(), a_min=1e-8, a_max=None)

        volume = transform(volume)
        return np.array(volume, dtype=np.float16)


def process_sample(args):
    """Process a single sample: extract from zip -> save as .npy."""
    img_path, case_to_zip = args
    parts = img_path.split("/")
    ct_type = parts[1]
    case_id = parts[2]
    series = parts[3].replace(".npy", "")

    # Output path mirrors the original structure
    out_path = os.path.join(OUT_DIR, ct_type, case_id, f"{series}.npy")

    if os.path.exists(out_path):
        return "skip"

    zip_path = case_to_zip.get((ct_type, case_id))
    if zip_path is None:
        return "no_zip"

    try:
        volume = load_volume_from_zip(zip_path, case_id, series)
        if volume is None:
            return "no_data"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, volume)
        return "ok"
    except Exception as e:
        return f"error:{e}"


def main():
    print("Building zip index...")
    case_to_zip = build_zip_index()
    print(f"  Indexed {len(case_to_zip)} cases")

    print("Loading sample list from M3D_Cap.json...")
    with open(DATA_JSON) as f:
        data = json.load(f)

    all_images = set()
    for split in ["train", "validation", "test"]:
        for entry in data[split]:
            all_images.add(entry["image"])
    samples = sorted(all_images)
    print(f"  {len(samples)} unique volumes to extract")

    # Check how many already done
    already = sum(1 for s in samples
                  if os.path.exists(os.path.join(
                      OUT_DIR, s.replace("M3D_Cap_npy/", "").replace(".npy", "") + ".npy")
                  ))
    # Simpler check
    existing = 0
    for s in samples:
        parts = s.split("/")
        out_path = os.path.join(OUT_DIR, parts[1], parts[2],
                                parts[3].replace(".npy", "") + ".npy")
        if os.path.exists(out_path):
            existing += 1
    print(f"  {existing} already extracted, {len(samples) - existing} remaining")

    args_list = [(s, case_to_zip) for s in samples]

    print(f"Extracting with 32 workers...")
    stats = Counter()
    with Pool(processes=32) as pool:
        for result in tqdm(pool.imap_unordered(process_sample, args_list),
                           total=len(args_list), desc="Extracting"):
            stats[result] += 1

    print(f"\nDone!")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
