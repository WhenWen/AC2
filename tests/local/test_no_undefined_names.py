"""Static undefined-name (pyflakes F821) guard for the difficulty-sampling touchpoints.

The difficulty blocks live inside ray_trainer.py's fit loop, which the torch-free local suite can't
execute — so a rename typo like `_seq_scores` vs `_seq_reward` is
invisible to the other tests. pyflakes catches undefined names without running the code. Skips
cleanly if pyflakes isn't installed (so the stdlib suite stays dependency-free); run under the torch
venv where it IS installed:

    python tests/local/test_no_undefined_names.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILES = [
    "src/verl/verl/trainer/ppo/ray_trainer.py",
    "src/verl/verl/trainer/ppo/difficulty.py",
    "src/verl/verl/workers/utils/losses.py",
    "src/ac2/viz/parse_fig_data.py",
    "src/ac2/viz/single_run_dashboard.py",
]

try:
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
except ImportError:
    print("SKIP: pyflakes not installed (run under /tmp/ds-venv). No undefined-name check performed.")
    sys.exit(0)

import io

undefined = []
for rel in FILES:
    path = os.path.join(ROOT, rel)
    src = open(path).read()
    out, err = io.StringIO(), io.StringIO()
    check(src, rel, Reporter(out, err))
    for line in out.getvalue().splitlines():
        if "undefined name" in line:
            undefined.append(line)

if undefined:
    print("FAIL: undefined names found:")
    print("\n".join(undefined))
    sys.exit(1)
print(f"PASS no-undefined-names: {len(FILES)} modules clean (F821)")
