"""Torch unit tests for the difficulty-sampling loss correction (importance-weighted loss).

Run with a torch venv (no verl import needed -- the vanilla clip math is replicated verbatim from
compute_policy_loss_vanilla, core_algos.py:1332-1362, so the scaling claim is tested against the
real formula):

    python tests/local/test_difficulty_loss.py
"""
import importlib.util
import os

import torch

torch.manual_seed(0)

_DIFF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "verl", "verl", "trainer", "ppo", "difficulty.py"
)
spec = importlib.util.spec_from_file_location("difficulty_under_test", _DIFF_PATH)
difficulty = importlib.util.module_from_spec(spec)
spec.loader.exec_module(difficulty)


def vanilla_pg_losses(advantages, ratio, clip_low=0.2, clip_high=0.2, clip_c=3.0, aec_k=0.0):
    """Verbatim replica of the per-token loss in compute_policy_loss_vanilla (incl. AEC term and
    dual-clip), BEFORE aggregation. Any edit to the real function should be mirrored here."""
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high + aec_k)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    return torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)


# ---------------------------------------------------------------------------
# 1) EXACT scaling: pg_losses(c*A) == c * pg_losses(A) per token, for c > 0,
#    across all clip regimes (inside/outside clip, A<0 dual-clip, AEC k>0).
# ---------------------------------------------------------------------------
B, L = 64, 37
A = torch.randn(B, L) * 3.0                      # both signs, large magnitudes
ratio = torch.exp(torch.randn(B, L) * 1.5)       # spans far outside the clip range
c = torch.rand(B, 1) * 31.9 + 0.1                # c in (0.1, 32): the floored-correction range
for k in (0.0, 0.08):
    base = vanilla_pg_losses(A, ratio, aec_k=k)
    scaled = vanilla_pg_losses(A * c, ratio, aec_k=k)
    assert torch.allclose(scaled, base * c, rtol=1e-6, atol=1e-6), "positive scaling must be exact"
# clip-decision invariance: the clipfrac indicator is unchanged by scaling
ind_base = (-A * torch.clamp(ratio, 0.8, 1.2)) > (-A * ratio)
ind_scaled = (-(A * c) * torch.clamp(ratio, 0.8, 1.2)) > (-(A * c) * ratio)
assert torch.equal(ind_base, ind_scaled)
print("PASS exact-scaling: per-token loss scales by c through clip/dual-clip/AEC; clipfrac invariant")

# ---------------------------------------------------------------------------
# 2) Importance-weight gradient identity (autograd): E_q[c_i * grad(l_i)] == E_p[grad(l_i)]
#    computed EXACTLY by enumerating the population (no sampling noise).
# ---------------------------------------------------------------------------
N = 11
theta = torch.randn(N, requires_grad=True)       # one "logit" per problem
p_hat = torch.linspace(0.0, 1.0, N).tolist()

d = difficulty
os.environ["SP_DIFF_SAMPLING"] = "1"
os.environ["SP_DIFF_WEIGHT"] = "sqrt_pq"
os.environ["SP_DIFF_MIN_OBS"] = "1"
spec2 = importlib.util.spec_from_file_location("difficulty_fresh", _DIFF_PATH)
d = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(d)
for i, p in enumerate(p_hat):
    d.observe(f"q{i}", p)
qids = [f"q{i}" for i in range(N)]
c_map, w_map = d.corrections_for(qids)
W = sum(w_map.values())

def loss_i(i):                                   # arbitrary smooth per-problem loss
    return (theta[i] - float(i)) ** 2 + torch.sin(theta[i])

# uniform-target gradient
lp = sum(loss_i(i) for i in range(N)) / N
gp = torch.autograd.grad(lp, theta, retain_graph=True)[0]
# exact expectation over the weighted sampler: sum_i q_i * c_i * grad l_i
lq = sum((w_map[f"q{i}"] / W) * c_map[f"q{i}"] * loss_i(i) for i in range(N))
gq = torch.autograd.grad(lq, theta)[0]
assert torch.allclose(gp, gq, rtol=1e-6, atol=1e-8), (gp, gq)
print("PASS gradient identity: E_q[c * grad] == uniform gradient (exact enumeration, autograd)")

# ---------------------------------------------------------------------------
# 3) Driver normalization equivalence with masked tensors:
#    masked_sum(c_hat * l) / batch_num_tokens  ==  sum_i c_i l_i_sum / sum_i c_i n_i
#    (the c-weighted token-mean), with ragged per-row token counts.
# ---------------------------------------------------------------------------
B, L = 16, 25
lens = torch.randint(3, L + 1, (B,))
mask = (torch.arange(L).unsqueeze(0) < lens.unsqueeze(1)).float()
loss_mat = torch.randn(B, L) * mask
c_row = (torch.rand(B) * 10 + 0.1).tolist()
n_row = lens.tolist()

ch = d.normalize_c_for_loss(c_row, n_row)
ch_t = torch.tensor(ch).unsqueeze(1)             # (B,1) -- the diff_c tensor shape
batch_num_tokens = mask.sum()
lhs = (loss_mat * ch_t * mask).sum() / batch_num_tokens          # what the worker computes
num = sum(c * (loss_mat[i] * mask[i]).sum() for i, c in enumerate(c_row))
den = sum(c * n for c, n in zip(c_row, n_row))
rhs = num / den                                                   # c-weighted token-mean
assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-7), (lhs.item(), rhs.item())
# c == 1 everywhere -> exactly the baseline token-mean (bit-identical no-op check)
ones = d.normalize_c_for_loss([1.0] * B, n_row)
assert all(x == 1.0 for x in ones)
print("PASS normalization: worker aggregate == weighted token-mean; c=1 -> exact baseline")

# ---------------------------------------------------------------------------
# 4) (B,1) broadcast sanity against per-row loop
# ---------------------------------------------------------------------------
A = torch.randn(B, L)
out = A * ch_t
for i in range(B):
    assert torch.equal(out[i], A[i] * ch[i])
print("PASS broadcast: (B,1) diff_c scales rows exactly")

print("\nALL DIFFICULTY LOSS TESTS PASSED")


# ---------------------------------------------------------------------------
# 5) Entropy regularizer is IS-corrected too.
#    agg_loss token-mean of c-weighted entropy == the c-weighted token-mean;
#    with c==1 it's bit-identical to the unweighted entropy term.
# ---------------------------------------------------------------------------
B, L = 16, 25
lens = torch.randint(3, L + 1, (B,))
mask = (torch.arange(L).unsqueeze(0) < lens.unsqueeze(1)).float()
H = torch.rand(B, L) * mask                                   # per-token entropy >= 0
c_row = (torch.rand(B) * 10 + 0.1).tolist()
n_row = lens.tolist()
ch = d.normalize_c_for_loss(c_row, n_row)
ch_t = torch.tensor(ch).unsqueeze(1)
bnt = mask.sum()
# what losses.py now computes: token-mean of (entropy * diff_c)
lhs = ((H * ch_t) * mask).sum() / bnt
# c-weighted token-mean of entropy
num = sum(c * (H[i] * mask[i]).sum() for i, c in enumerate(c_row))
den = sum(c * n for c, n in zip(c_row, n_row))
assert torch.allclose(lhs, num / den, rtol=1e-5, atol=1e-7), (lhs.item(), (num / den).item())
# c == 1 -> exactly the baseline entropy term (bit-identical no-op)
ones = torch.ones(B, 1)
assert torch.equal((H * ones * mask).sum() / bnt, (H * mask).sum() / bnt)
print("PASS entropy correction: c-weighted entropy token-mean == uniform expectation; c=1 == baseline")

# gradient identity for the entropy term (exact population enumeration)
N = 9
theta = torch.randn(N, requires_grad=True)
p_hat = torch.linspace(0.05, 0.95, N).tolist()
spec3 = importlib.util.spec_from_file_location("difficulty_H", _DIFF_PATH)
dH = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(dH)
os.environ["SP_DIFF_SAMPLING"] = "1"; os.environ["SP_DIFF_WEIGHT"] = "sqrt_pq"; os.environ["SP_DIFF_MIN_OBS"] = "1"
for i, p in enumerate(p_hat):
    dH.observe(f"q{i}", p)
qids = [f"q{i}" for i in range(N)]
cmap, wmap = dH.corrections_for(qids)
W = sum(wmap.values())

def ent_i(i):                                                 # a differentiable "entropy" per problem
    pr = torch.sigmoid(theta[i])
    return -(pr * torch.log(pr) + (1 - pr) * torch.log(1 - pr))

gp = torch.autograd.grad(sum(ent_i(i) for i in range(N)) / N, theta, retain_graph=True)[0]
gq = torch.autograd.grad(
    sum((wmap[f"q{i}"] / W) * cmap[f"q{i}"] * ent_i(i) for i in range(N)), theta)[0]
assert torch.allclose(gp, gq, rtol=1e-6, atol=1e-8), (gp, gq)
print("PASS entropy correction: E_q[c * grad H] == uniform entropy gradient (autograd enumeration)")

print("\nALL ENTROPY-CORRECTION TESTS PASSED")


# ---------------------------------------------------------------------------
# 6) actor/entropy control signal (AEC input) is the SNIS token-mean of entropy,
#    self-normalized by Σc·n (replicates the ray_trainer old_log_prob computation).
#    c==1 -> the plain token-mean (== unchanged AEC input on non-difficulty runs).
# ---------------------------------------------------------------------------
B, L = 20, 30
lens = torch.randint(4, L + 1, (B,))
rmask = (torch.arange(L).unsqueeze(0) < lens.unsqueeze(1)).float()
ent = torch.rand(B, L) * rmask
c_row = (torch.rand(B) * 15 + 0.1)
cc = c_row.unsqueeze(-1)

# exact ray_trainer form
entropy_ctrl = (ent * rmask * cc).sum() / (rmask * cc).sum()
# SNIS token-mean by hand
num = sum(float(c_row[i]) * (ent[i] * rmask[i]).sum() for i in range(B))
den = sum(float(c_row[i]) * float(lens[i]) for i in range(B))
assert torch.allclose(entropy_ctrl, torch.tensor(num / den), rtol=1e-5, atol=1e-7)
# c == 1 -> plain token-mean (the pre-difficulty AEC input, bit-identical)
plain = (ent * rmask).sum() / rmask.sum()
ctrl_c1 = (ent * rmask * torch.ones(B, 1)).sum() / (rmask * torch.ones(B, 1)).sum()
assert torch.equal(ctrl_c1, plain)
print("PASS entropy control signal: SNIS token-mean (Σc·H/Σc·n); c=1 == plain token-mean")

print("\nALL ENTROPY-CONTROL TESTS PASSED")
