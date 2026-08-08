"""Run ablation benchmarks under a consistent protocol.

Entry point for bench_ablations_clean -- the library stays in src/
because other scripts import ROUTER_MODE and build() from it.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test/bench_ablations.py --suite A
    CUDA_VISIBLE_DEVICES=0 python test/bench_ablations.py --suite A,B --epoch 99
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_ablations_clean import main

if __name__ == "__main__":
    main()
