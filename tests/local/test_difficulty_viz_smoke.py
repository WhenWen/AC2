"""Viz smoke: render a synthetic difficulty-sampling fig_data through the REAL render_dashboard
and assert the PDF has 3 pages (and 2 pages for a uniform run)."""
import json, os, sys, tempfile
sys.path.insert(0, "src/ac2/viz")

steps = list(range(1, 11))
def series(base, jitter):
    return [base + jitter * (i % 3) for i in range(10)]

per_step = {
    "steps": steps,
    # corrected (standard) + raw counterparts for a few audited keys
    "critic__score__mean": series(0.55, 0.01),
    "difficulty_raw__critic__score__mean": series(0.42, 0.01),
    "response_length__mean": series(20000, 500),
    "difficulty_raw__response_length__mean": series(24000, 500),
    "critic__advantages__mean": series(0.0, 0.002),
    "difficulty_raw__critic__advantages__mean": series(-0.01, 0.002),
    # sampler health
    "difficulty__ess_frac": series(0.3, 0.02),
    "difficulty__c_mean": series(1.0, 0.0),
    "difficulty__c_max": series(9.0, 0.5),
    "difficulty__floored_frac": series(0.3, 0.01),
    # minimal extras used by page-1 panels (most will render N/A -- fine)
    "actor__entropy": series(0.15, 0.005),
}
fig_data = {"schema": "riemann_v7_fig_data", "run_id": "smoke", "title": "smoke",
            "per_step": per_step, "config": {}, "manifest": {}, "warnings": []}

tmp = tempfile.mkdtemp()
fd = os.path.join(tmp, "fig.json")
json.dump(fig_data, open(fd, "w"))

import render_dashboard as RD
paths = RD.render(fd, os.path.join(tmp, "analysis"), experiment_folder="smoke")
from pypdf import PdfReader
n = len(PdfReader(paths["pdf"]).pages)
print("difficulty run pages:", n)
assert n == 3, f"expected 3 pages, got {n}"

# uniform run: strip difficulty keys -> must stay 2 pages
per_step2 = {k: v for k, v in per_step.items() if "difficulty" not in k}
fig_data["per_step"] = per_step2
fd2 = os.path.join(tmp, "fig2.json")
json.dump(fig_data, open(fd2, "w"))
paths2 = RD.render(fd2, os.path.join(tmp, "analysis2"), experiment_folder="smoke2")
n2 = len(PdfReader(paths2["pdf"]).pages)
print("uniform run pages:", n2)
assert n2 == 2, f"expected 2 pages, got {n2}"
print("VIZ SMOKE PASSED")
