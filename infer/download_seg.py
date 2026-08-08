"""Download and extract all M3D-Seg zips."""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from huggingface_hub import hf_hub_download

os.environ["HF_HOME"] = "/data2/gdp2988/.cache/huggingface"

OUT_DIR = "/data2/gdp2988/3d_cot/dataset/m3d_seg/M3D_Seg"
REPO = "GoodBaiBai88/M3D-Seg"

zips = [f"M3D_Seg/{i:04d}.zip" for i in range(25)]
zips.append("M3D_Seg/0024_1.zip")
zips.append("M3D_Seg/0024_2.zip")

os.makedirs(OUT_DIR, exist_ok=True)

for i, zname in enumerate(zips):
    code = zname.split("/")[1].replace(".zip", "")
    extract_dir = os.path.join(OUT_DIR, code)

    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"[{i+1}/{len(zips)}] {code} already extracted, skipping")
        continue

    print(f"[{i+1}/{len(zips)}] Downloading {zname}...", flush=True)
    path = hf_hub_download(REPO, zname, repo_type="dataset",
                           local_dir=OUT_DIR + "/_dl")

    print(f"  Extracting...", flush=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.system(f"unzip -q -o {path} -d {extract_dir}")

    # Clean zip after extract
    os.remove(path)
    print(f"  Done: {extract_dir}", flush=True)

print("All done!")
