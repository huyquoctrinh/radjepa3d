"""Shared configuration for the 3D-CoT dataset generation pipeline."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA_DIR = Path("/data2/gdp2988/3d_cot/dataset/m3d_vqa_text")
OUT_DIR = Path("/data2/gdp2988/3d_cot/dataset/cot_v1")
ALL_CSVS = [
    DATA_DIR / "M3D_VQA_train.csv",
    DATA_DIR / "M3D_VQA_val.csv",
    DATA_DIR / "M3D_VQA_test.csv",
]

# Question type semantics from M3D-VQA
QTYPE_LABELS = {
    "1": "plane",
    "2": "phase",
    "3": "organ",
    "4": "abnormality",
    "5": "location",
}

# Turn ordering: walk the radiologist's reading order
TURN_ORDER = ["1", "2", "3", "4", "5"]

# Impression-splitting markers (regex pattern)
IMPRESSION_MARKERS = (
    r"(?:^|\.\s+|\n)"
    r"(?:Impression|IMPRESSION|Conclusion|CONCLUSION|Final Diagnosis|"
    r"FINAL DIAGNOSIS|Assessment|ASSESSMENT)\s*[:\-]"
)

# Generation params
TEMPERATURE = 0.4
TOP_P = 0.9
MAX_NEW_TOKENS = 2048
