"""Extract text files from M3D-Cap zips to extracted_M3D_Cap/."""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zipfile
from pathlib import Path
from tqdm import tqdm

CT_DIRS = [
    "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/ct_case",
    "/data2/gdp2988/3d_cot/dataset/M3D-Cap/M3D_Cap/ct_quizze",
]
OUT_DIR = "/data2/gdp2988/3d_cot/extracted_M3D_Cap"


def extract_study_findings(text):
    idx = text.find("study_findings:")
    if idx >= 0:
        return text[idx + len("study_findings:"):].replace("\n", " ").strip()
    idx = text.find("discussion:")
    if idx >= 0:
        return text[idx + len("discussion:"):].replace("\n", " ").strip()
    return text.replace("\n", " ").strip()


def main():
    total = 0
    for ct_dir in CT_DIRS:
        ct_type = os.path.basename(ct_dir)
        zip_files = sorted(Path(ct_dir).glob("*.zip"))
        for zf_path in tqdm(zip_files, desc=ct_type):
            with zipfile.ZipFile(zf_path, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith("text.txt"):
                        continue
                    case_id = name.split("/")[0]
                    out_path = os.path.join(OUT_DIR, ct_type, case_id, "text.txt")
                    if os.path.exists(out_path):
                        continue
                    try:
                        raw = zf.read(name).decode("utf-8", errors="replace")
                        findings = extract_study_findings(raw)
                        if len(findings) < 10:
                            continue
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "w") as f:
                            f.write(findings)
                        total += 1
                    except Exception:
                        continue
    print(f"Extracted {total} text files")


if __name__ == "__main__":
    main()
