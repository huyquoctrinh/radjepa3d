"""M3D Benchmark: Closed-ended VQA, Open-ended VQA, and Captioning.

Entry point for mllms.benchmark -- the library stays in mllms/
because bench_spatialmed and merge_shards import from it.

Usage:
    python test/bench_mllm.py --checkpoint stage2_best.pt --mode closed
    python test/bench_mllm.py --checkpoint stage2_best.pt --mode open
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mllms.benchmark import main

if __name__ == "__main__":
    main()
