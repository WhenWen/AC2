"""sp_segment advantage estimator — invariance and correctness.

With no seg rows the estimator must be BIT-IDENTICAL to
compute_grpo_outcome_advantage at norm_adv_by_std_in_grpo=False. Same discipline as "an
empty prefix reduces exactly to SingleTurnAgentLoop": the fallback path is what every
unsplit row in a real batch takes, so an approximate match there is a silent, permanent
regression on the majority of the batch.

CPU only, no torch.distributed, no engine.
"""
import numpy as np
import torch

from verl.trainer.ppo import core_algos


def _batch(rewards, resp_len=10, gen_from=0):
    """token_level_rewards carrying one scalar per row, plus an all-ones response mask."""
    bsz = len(rewards)
    tlr = torch.zeros(bsz, resp_len, dtype=torch.float32)
    for i, r in enumerate(rewards):
        tlr[i, -1] = float(r)
    mask = torch.zeros(bsz, resp_len, dtype=torch.float32)
    mask[:, gen_from:] = 1.0
    return tlr, mask


def _extra(n, seg=None, bound=None, lane="full"):
    """extra_info rows; seg[i] None -> that row carries no valid interior value."""
    out = []
    for i in range(n):
        ei = {}
        if seg is not None and seg[i] is not None:
            ei["sp_q_seg_value"] = float(seg[i])
            ei["sp_q_seg_bound"] = int(bound[i] if isinstance(bound, list) else bound)
            ei["sp_q_seg_invalid"] = 0
            ei["sp_q_seg_lane"] = lane
        out.append(ei)
    return np.array(out, dtype=object)


def test_no_seg_rows_is_bit_identical_to_grpo():
    rewards = [0.0, 0.14, 0.29, 1.0, 0.57, 0.43, 0.71, 0.86]
    tlr, mask = _batch(rewards)
    index = np.array(["g0"] * 4 + ["g1"] * 4)

    want, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    # no extra_info at all
    got_a, got_b = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index, non_tensor_batch=None,
    )
    assert torch.equal(got_a, want), (got_a, want)
    assert torch.equal(got_b, want)

    # extra_info present but every row unsplittable
    got_a2, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(8, seg=[None] * 8)},
    )
    assert torch.equal(got_a2, want)


def test_singleton_group_matches_grpo_zero_mean_convention():
    """grpo gives a group of one mean 0.0 (not r_i). The fallback must copy that exactly,
    or every degenerate group silently changes sign convention."""
    tlr, mask = _batch([0.43])
    index = np.array(["only"])
    want, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    got, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index, non_tensor_batch=None,
    )
    assert torch.equal(got, want)


def test_invalid_flag_zero_is_honoured():
    """Regression: `int(ei.get(...,1) or 1)` would read a valid 0 as invalid and disable the
    method wholesale, while every log line still looked healthy."""
    n = 8
    tlr, mask = _batch([0.1] * 4 + [0.9] * 4)
    index = np.array(["g"] * n)
    segs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    got, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(n, seg=segs, bound=4)},
    )
    grpo, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    assert not torch.equal(got, grpo), "seg values were ignored -- the split never happened"
    assert core_algos.SP_SEGMENT_LAST_STATS["q/seg_groups_split"] == 1.0


def test_two_segment_values_and_boundary():
    n = 8
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]
    segs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    bound = 4
    tlr, mask = _batch(rewards, resp_len=10)
    index = np.array(["g"] * n)
    got, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(n, seg=segs, bound=bound)},
    )
    qbar = sum(segs) / len(segs)
    # Tolerance, not exact equality, and deliberately so. The estimator does its arithmetic
    # on float32 tensor elements while the expectation here is float64, so a row whose true
    # a2 is 0.0 lands ~3e-9 off. atol=0 is BELOW float32 granularity and would be gating on
    # rounding, not on correctness -- the bit-identity test above is where exactness is
    # meaningful, because there both sides run the same float32 ops. Report the worst
    # deviation so a real error cannot hide inside the tolerance.
    F32_TOL = 1e-6
    worst = 0.0
    for i in range(n):
        a1 = segs[i] - qbar
        a2 = rewards[i] - segs[i]
        d1 = float((got[i, :bound] - a1).abs().max())
        d2 = float((got[i, bound:] - a2).abs().max())
        worst = max(worst, d1, d2)
        assert d1 < F32_TOL, (i, "seg1", d1)
        assert d2 < F32_TOL, (i, "seg2", d2)
    assert worst < F32_TOL, worst
    print(f"    [seg values] worst |d| = {worst:.3e} (float32 eps ~1.2e-07)")
    # the two segment scalars sum to r_i - qbar, not r_i - rbar
    for i in range(n):
        assert abs((float(got[i, 0]) + float(got[i, -1])) - (rewards[i] - qbar)) < F32_TOL


def test_small_group_is_still_split_no_fallback():
    """NO group-level fallback (2026-08-27). A group with only 2 valid interior values is
    segmented on those 2 -- routing it to the control's estimator is what diluted the arm to
    a third of its intended dose."""
    n = 8
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tlr, mask = _batch(rewards)
    index = np.array(["g"] * n)
    segs = [0.1, 0.9, None, None, None, None, None, None]  # |S_u| = 2
    got, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(n, seg=segs, bound=4)},
    )
    grpo, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    assert not torch.equal(got, grpo), "small group must STILL be segmented"
    st = core_algos.SP_SEGMENT_LAST_STATS
    assert st["q/seg_groups_split"] == 1.0
    assert st["q/seg_group_fallback_frac"] == 0.0, st["q/seg_group_fallback_frac"]
    # the two seg rows carry A1 = q_i - qbar around qbar = 0.5
    assert abs(float(got[0, 0]) - (0.1 - 0.5)) < 1e-6
    assert abs(float(got[1, 0]) - (0.9 - 0.5)) < 1e-6
    # rows with no interior value keep the single-segment form (degenerate, not a fallback)
    assert abs(float(got[2, 0]) - (rewards[2] - sum(rewards) / n)) < 1e-6


def test_grid_collapse_gives_zero_a1_and_is_counted_not_routed():
    """A collapsed group keeps the segmented estimator: A1 == 0 exactly (the honest statement
    that the midpoint carries no information here) while A2 = r_i - q_i still contributes.
    The collapse is COUNTED but must not change behaviour."""
    n = 8
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    q = 0.7
    bound = 4
    tlr, mask = _batch(rewards, resp_len=10)
    index = np.array(["g"] * n)
    got, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(n, seg=[q] * n, bound=bound)},
    )
    grpo, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    assert not torch.equal(got, grpo), "collapsed group must NOT be routed to the control"
    st = core_algos.SP_SEGMENT_LAST_STATS
    assert st["q/seg_collapsed_frac"] == 1.0, "collapse must still be measured"
    assert st["q/seg_group_fallback_frac"] == 0.0, "and must not cause a fallback"
    assert st["q/seg_groups_split"] == 1.0
    for i in range(n):
        # segment 1: exactly zero
        assert float(got[i, :bound].abs().max()) < 1e-6, (i, float(got[i, 0]))
        # segment 2: still r_i - q_i
        assert abs(float(got[i, bound]) - (rewards[i] - q)) < 1e-6, i


def test_masked_prefix_tokens_stay_zero():
    """The prefix rides in the response tensor with response_mask=0; it must receive no
    advantage regardless of which segment its indices fall in."""
    n = 8
    tlr, mask = _batch([0.5] * n, resp_len=12, gen_from=3)
    index = np.array(["g"] * n)
    segs = [round(0.1 * i, 2) for i in range(n)]
    got, _ = core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(n, seg=segs, bound=6)},
    )
    assert torch.all(got[:, :3] == 0.0), got[:, :3]


def test_lane_split_metrics_are_separate():
    """~2/3 of split rows are short-lane, where segment 2 is Q-minus-Q; the pooled number
    must not be the only one available."""
    n = 8
    tlr, mask = _batch([0.2] * n)
    index = np.array(["g"] * n)
    segs = [round(0.1 * i, 2) for i in range(n)]
    core_algos.compute_sp_segment_advantage(
        token_level_rewards=tlr, response_mask=mask, index=index,
        non_tensor_batch={"extra_info": _extra(n, seg=segs, bound=4, lane="short")},
    )
    st = core_algos.SP_SEGMENT_LAST_STATS
    assert st["q/seg_rows_short"] == float(n)
    assert st["q/seg_rows_full"] == 0.0
    assert st["q/seg_rows"] == float(n)
    for k in ("q/seg_rho_within", "q/seg_mid_mae", "q/seg2_bias",
              "q/seg1_adv_sd", "q/seg2_adv_sd", "q/seg_adv_xcorr",
              "q/seg1_token_frac", "q/seg2_token_frac"):
        assert k in st and f"{k}_short" in st and f"{k}_full" in st, k
