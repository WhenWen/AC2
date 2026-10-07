"""No-group TD (sp_q_td_enable) — the advantage arithmetic, via sp_segment.

The mode writes the probe value into the sp_q_seg_* stamps at bound = prefix_len on
SINGLETON uid groups. compute_sp_segment_advantage's singleton-split path must then give

    A_i = 0            on [0, prefix_len)   (a1 = q - qbar with qbar = q, and the span is
                                             mask-0 prefix anyway)
    A_i = r_i - q_i    on [prefix_len, end)

and a mixed batch's UNSTAMPED groups (full-lane 16-groups, cold_scratch) must stay
bit-identical to grpo at norm_adv_by_std_in_grpo=False -- the control estimator untouched
everywhere the mechanism does not apply.

CPU only, no torch.distributed, no engine.
"""
import numpy as np
import torch

from verl.trainer.ppo import core_algos


def _batch(rewards, resp_len=10, prefix_lens=None):
    bsz = len(rewards)
    tlr = torch.zeros(bsz, resp_len, dtype=torch.float32)
    mask = torch.zeros(bsz, resp_len, dtype=torch.float32)
    for i, r in enumerate(rewards):
        tlr[i, -1] = float(r)
        p = 0 if prefix_lens is None else int(prefix_lens[i])
        mask[i, p:] = 1.0
    return tlr, mask


def _td_extra(qvals, bounds):
    out = []
    for q, b in zip(qvals, bounds):
        ei = {}
        if q is not None:
            ei["sp_q_seg_value"] = float(q)
            ei["sp_q_seg_bound"] = int(b)
            ei["sp_q_seg_invalid"] = 0
            ei["sp_q_seg_lane"] = "short"
        else:
            ei["sp_q_seg_invalid"] = 1
        ei["sp_q_td_row"] = 1
        out.append(ei)
    return np.array(out, dtype=object)


def test_td_singleton_is_r_minus_q_on_generated_span():
    rewards = [0.7, 0.2, 1.0]
    qvals = [0.3, 0.6, 1.0]
    prefix_lens = [4, 0, 7]
    tlr, mask = _batch(rewards, prefix_lens=prefix_lens)
    index = np.array(["u0", "u1", "u2"])  # every TD copy is its own uid

    adv, ret = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _td_extra(qvals, prefix_lens)},
        seg_min_valid=1,
    )
    for i, (r, q, p) in enumerate(zip(rewards, qvals, prefix_lens)):
        want = r - q
        # generated span carries r - q exactly
        got = adv[i, p:]
        assert torch.allclose(got, torch.full_like(got, want), atol=1e-6, rtol=0), (i, got, want)
        # a1 on the prefix span is q - qbar = 0 for a singleton
        if p:
            assert torch.equal(adv[i, :p], torch.zeros(p)), (i, adv[i, :p])


def test_td_invalid_stamp_falls_back_to_grpo_singleton():
    """A TD row whose probe overflowed carries sp_q_seg_invalid=1: the estimator's
    fallback is the grpo singleton convention (baseline 0.0 -> raw reward). The RUN never
    trains that row -- the reward site drops it (mask 0) -- but the estimator-level
    behavior is pinned here so a change shows up as a test failure, not a silent
    convention flip."""
    tlr, mask = _batch([0.5])
    index = np.array(["u0"])
    adv, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _td_extra([None], [0])},
        seg_min_valid=1,
    )
    want, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    assert torch.equal(adv, want)


def test_mixed_batch_leaves_unstamped_groups_bit_identical_to_grpo():
    """2 TD singletons + one 4-member full-lane group with no stamps: the group's rows
    must match grpo exactly (same tensors, same values), TD rows must be r - q."""
    rewards = [0.7, 0.1, 0.0, 0.29, 0.57, 1.0]
    prefix_lens = [3, 5, 0, 0, 0, 0]
    tlr, mask = _batch(rewards, prefix_lens=prefix_lens)
    index = np.array(["t0", "t1", "g", "g", "g", "g"])

    extra = list(_td_extra([0.4, 0.3], [3, 5]))
    extra += [{} for _ in range(4)]
    extra = np.array(extra, dtype=object)

    adv, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": extra},
        seg_min_valid=1,
    )

    want, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    # unstamped group rows: bit-identical to grpo
    assert torch.equal(adv[2:], want[2:])
    # TD rows: r - q on their generated spans
    assert torch.allclose(adv[0, 3:], torch.full((7,), 0.7 - 0.4)), adv[0, 3:]
    assert torch.allclose(adv[1, 5:], torch.full((5,), 0.1 - 0.3)), adv[1, 5:]
