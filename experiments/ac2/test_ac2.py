"""Pure-logic tests for the AC2 critic additions. No torch, no GPU, no cluster.

Covers the decision logic that would otherwise first be exercised on a multi-node allocation:

  1. `QLrController` -- the critic LR halving ladder (schedule, floor alert, streak reset,
     immediate halve on non-finite, resume via replay_step).
  2. `Q_PROMPT_VARIANTS` -- the registry, and the properties each variant must hold.
  3. The tier-1 reference correctness gate -- that `judge_pass == 0` entries are refused,
     `judge_pass == 1` accepted, and a MISSING flag treated as passing (frozen seeds and the
     judged_correct admission path are correct by construction and carry no flag).
  4. The cold-bootstrap draw -- empty buffer yields zero replay qids plus `n_replay` fill
     indices disjoint from the statement lane.
  5+. Wiring checks on runner.py and run_attach.sh: no separate-Q resharder, pinned
     off switches, critic config reaching the dashboard, the judge revision pin, split
     readiness thresholds, the TP>1 AOT preflight, the response-budget group, and checkpoint
     retention.

Run:  python experiments/ac2/test_ac2.py
   or: pytest experiments/ac2/test_ac2.py

Sources are loaded by AST surgery rather than by importing `verl` (which needs torch/ray), so
these run anywhere -- including a laptop -- and fail loudly if the shapes they assume change.
"""
from __future__ import annotations

import ast
import os
import re
import sys

SP_Q = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "verl", "verl", "trainer", "ppo", "sp_q_readiness.py",
)


def _load_names(path: str, class_names=(), assign_prefixes=()):
    """exec only the named classes / prefixed module assignments from `path`."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                name = getattr(t, "id", "")
                if name and any(name.startswith(p) for p in assign_prefixes):
                    keep.append(node)
                    break
    ns: dict = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])),
                 "<sp_q>", "exec"), ns)
    return ns


# ---------------------------------------------------------------- 1. the LR ladder
def test_ladder_schedule_and_floor():
    Q = _load_names(SP_Q, class_names=("QLrController",))["QLrController"]
    c = Q(lr=2e-6)

    # one breach arms the streak but does not reduce (patience 2)
    s = c.apply_step(delta_q=5.0, delta_ppo1=1.0, delta_ppo2=1.0,
                     q_phase_skipped=False, q_non_finite=False)
    assert s["breach"] and s["breach_streak_after"] == 1 and not s["reduced"]
    assert abs(s["rho"] - 2.5) < 1e-9, s["rho"]          # 5 / (1 + 1), the SUM not the net
    assert c.lr == 2e-6

    # the second consecutive breach halves and clears the streak
    s = c.apply_step(delta_q=5.0, delta_ppo1=1.0, delta_ppo2=1.0,
                     q_phase_skipped=False, q_non_finite=False)
    assert s["reduced"] and s["breach_streak_after"] == 0 and c.lr == 1e-6

    # a non-breach resets the streak: two isolated breaches must NOT reduce
    for _ in range(2):
        c.apply_step(delta_q=5.0, delta_ppo1=1.0, delta_ppo2=1.0,
                     q_phase_skipped=False, q_non_finite=False)   # breach
        s = c.apply_step(delta_q=0.5, delta_ppo1=1.0, delta_ppo2=1.0,
                         q_phase_skipped=False, q_non_finite=False)  # not a breach
        assert not s["breach"] and s["breach_streak_after"] == 0
    assert c.lr == 1e-6, "isolated breaches must not walk the ladder down"

    # non-finite halves IMMEDIATELY, no streak needed -> reaches the floor
    s = c.apply_step(delta_q=None, delta_ppo1=1.0, delta_ppo2=1.0,
                     q_phase_skipped=False, q_non_finite=True)
    assert s["reduced"] and c.lr == 5e-7

    # at the floor: further breaches alert only, never go below
    for _ in range(4):
        s = c.apply_step(delta_q=5.0, delta_ppo1=1.0, delta_ppo2=1.0,
                         q_phase_skipped=False, q_non_finite=False)
    assert c.lr == 5e-7 and s["floor_alert"] and not s["reduced"]

    # a skipped (empty-sample) phase is a no-op: streak reset, no reduction, rho None
    s = c.apply_step(delta_q=None, delta_ppo1=0.0, delta_ppo2=0.0,
                     q_phase_skipped=True, q_non_finite=False)
    assert s["rho"] is None and not s["reduced"] and s["breach_streak_after"] == 0

    # from lr 2e-6 the halving schedule is exactly 2e-6 -> 1e-6 -> 5e-7
    c2 = Q(lr=2e-6)
    seen = [c2.lr]
    for _ in range(10):
        c2.apply_step(delta_q=9.0, delta_ppo1=1.0, delta_ppo2=1.0,
                      q_phase_skipped=False, q_non_finite=False)
        if c2.lr != seen[-1]:
            seen.append(c2.lr)
    assert seen == [2e-6, 1e-6, 5e-7], seen


def test_ladder_resume_is_exact():
    Q = _load_names(SP_Q, class_names=("QLrController",))["QLrController"]
    live = Q(lr=2e-6)
    sections = [
        live.apply_step(delta_q=d, delta_ppo1=1.0, delta_ppo2=1.0,
                        q_phase_skipped=False, q_non_finite=False)
        for d in (5.0, 5.0, 0.1, 5.0)
    ]
    # a fresh controller replaying the delta log must land on the same state
    resumed = Q(lr=2e-6)
    for sec in sections:
        resumed.replay_step(sec)
    assert (resumed.lr, resumed.breach_streak) == (live.lr, live.breach_streak)
    # and so must one restored from q_state.json
    from_state = Q(lr=2e-6)
    from_state.load_state_dict(live.state_dict())
    assert (from_state.lr, from_state.breach_streak) == (live.lr, live.breach_streak)


# ------------------------------------------------------------- 2. the prompt variants
def test_prompt_variants():
    ns = _load_names(SP_Q, assign_prefixes=("Q_INSTRUCTION", "Q_PROMPT_VARIANTS"))
    V = ns["Q_PROMPT_VARIANTS"]
    assert set(V) == {"legacy", "reward_horizon"}, sorted(V)

    for name, (with_ref, no_ref) in V.items():
        # the answer-format contract must be identical across variants: the target ids are
        # built from the prefill, so a variant that changed it would silently break teaching.
        for text in (with_ref, no_ref):
            assert "Q value: z" in text
            assert "one of 0, 0.1, ..., 1" in text
            assert text.rstrip().endswith("`Q value:`.")
        assert "{reference_proof}" in with_ref, name
        assert "{reference_proof}" not in no_ref, name

    legacy_ref, legacy_noref = V["legacy"]
    new_ref, new_noref = V["reward_horizon"]

    # the stale hardcoded budget is gone from the new variant...
    for text in (new_ref, new_noref):
        assert "50K" not in text, "reward_horizon must not hardcode a token budget"
        assert "thinking trace" not in text, "the budget parenthetical went with the clause"
        # ...but a HORIZON is still stated: the labels are censored by the rollout limit, so a
        # horizon-free question asks about an outcome that was never measured.
        assert "remaining token budget" in text, "reward_horizon must still state a horizon"
        # ...and it asks for rubric credit, not a probability, because z is a mean points/7.
        assert "rubric credit" in text, "the prompt must name the quantity z actually is"
        assert "probability" not in text, (
            "z is a mean continuous points/7 reward, not a probability -- asking for a "
            "probability is the mismatch this variant exists to fix"
        )
    # the legacy wording is untouched -- resuming runs depend on it byte-for-byte
    assert "within totally 50K tokens" in legacy_ref and "probability" in legacy_noref


# ------------------------------------------------- 3. the tier-1 reference correctness gate
def test_reference_requires_judge_pass():
    """Re-implements trajectory_ref_proof's gate over a fake buffer.

    Asserts the decision table directly; the surrounding method is a cache + a scan.
    """
    src = open(SP_Q, encoding="utf-8").read()
    assert 'judge_pass", 1)) == 0' in src, (
        "trajectory_ref_proof must consult meta.judge_pass and DEFAULT IT TO PASSING; the "
        "guard's shape changed -- re-read the method before trusting this test"
    )

    def accepts(meta, require_pass=True):
        return not (require_pass and int((meta or {}).get("judge_pass", 1)) == 0)

    assert accepts({"judge_pass": 1}) is True                      # judged correct -> usable
    assert accepts({"judge_pass": 0}) is False                     # FAILED -> refused
    assert accepts({}) is True                                     # frozen seed / judged_correct
    assert accepts(None) is True                                   # no meta at all
    assert accepts({"judge_pass": 0}, require_pass=False) is True   # knob off = old behaviour
    # the field really is written by the admission path this run uses
    replay_src = open(os.path.join(os.path.dirname(SP_Q), "sp_replay.py"), encoding="utf-8").read()
    assert '"judge_pass": int(passes[i])' in replay_src, (
        "_admit_global no longer records judge_pass; the reference gate would silently "
        "accept everything (missing == passing)"
    )


# ----------------------------------------------------------- 4. the cold-bootstrap draw
def test_cold_bootstrap_fill_is_disjoint_and_full():
    """The empty-buffer branch must return zero qids and exactly n_replay fill indices,
    disjoint from the statement lane -- the dataset indexes fill[rslot - len(qids)] for
    every replay slot, so a short list would IndexError mid-epoch."""
    # stdlib `random` stands in for numpy's Generator here: this asserts the branch's
    # INVARIANTS (count, distinctness, disjointness from the statement lane), which do not
    # depend on which RNG produced the draw. Keeps the test runnable without numpy.
    import random

    num_statements, n_orig, n_replay = 5227, 32, 96
    used = set(random.Random(0).sample(range(num_statements), n_orig))
    candidates = [i for i in range(num_statements) if i not in used]
    assert len(candidates) >= n_replay
    fill = random.Random(1).sample(candidates, n_replay)

    qids: list = []
    assert len(qids) == 0
    assert len(fill) == n_replay
    assert not (set(fill) & used), "fill must not reuse this step's statement problems"
    assert len(set(fill)) == n_replay, "fill must be distinct problems"
    for rslot in range(n_replay):                       # every replay slot resolves
        assert 0 <= rslot - len(qids) < len(fill)

    src = open(os.path.join(os.path.dirname(SP_Q), "sp_replay.py"), encoding="utf-8").read()
    assert "cold bootstrap needs" in src, "the short-fill guard is gone from sp_replay"
    assert 'if not self.cold_bootstrap:' in src, "the empty-buffer branch is no longer gated"


# --------------------------------------------------- 5. the relaunch topology check
def test_relaunch_does_not_call_the_separate_q_resharder():
    """The attach must NOT invoke the separate-Q reshard helper: it hard-requires q_model/, which
    this tied-Q run never writes, so every relaunch after checkpoint 1 would fail on "source
    q_model missing". Assert the wiring directly."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach.sh"), encoding="utf-8").read()
    # Check for an INVOCATION, not a mention: the refusal message names the helper on purpose,
    # so a bare substring test would flag its own explanatory text.
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))
    for pattern in (r"bash\s+\S*reshard_to_ws16\.sh", r"^\s*RESHARD_WS=", r"RESHARD_EXP="):
        assert not re.search(pattern, code, re.M), (
            f"the attach invokes the separate-Q resharder ({pattern}); it requires q_model/ "
            "and this run writes sp_q_optim/ -- every relaunch after checkpoint 1 will fail"
        )
    # and it must decide from the checkpoint's real world size
    assert "sp_q_optim/rank_*.pt" in attach, "no world-size probe in the topology check"
    assert "REFUSING: checkpoint" in attach, "a topology mismatch must refuse, not proceed"


# ------------------------------------------- 6. the science switches are pinned, not omitted
def test_off_switches_are_pinned_and_asserted():
    """SP_LENPEN_ENABLE is read from the process environment by the reward module, and the
    allocation does not use --export=NONE, so omitting it is not the same as setting it to 0."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach.sh"), encoding="utf-8").read()
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    assert '"SP_LENPEN_ENABLE=0"' in attach, "the attach must PIN the penalty off, not omit it"
    assert 'assert not _truthy("SP_LENPEN_ENABLE")' in runner, "and the runner must assert it"
    assert 'env_vars["SP_LENPEN_ENABLE"] = "0"' in runner, (
        'the forward loop only carries truthy values, so "0" must be set explicitly or the Ray '
        "actors read whatever the node inherited"
    )
    # the reward module really does read it from the environment (so the risk was real)
    reward = os.path.join(os.path.dirname(os.path.dirname(here)), "src", "ac2",
                          "rewards", "ds4_finegrained_judge.py")
    if os.path.exists(reward):
        assert 'os.environ.get("SP_LENPEN_ENABLE"' in open(reward, encoding="utf-8").read()


# ------------------------- 8. manifest config -> dashboard, END TO END
def test_q_config_reaches_the_dashboard_from_a_real_config_shape():
    """Both sides must carry the critic config: the INPUT side (_load_config_yaml extracting the
    sp_q_* keys from the manifest) and the OUTPUT side (_derive_config surfacing them). If either
    is missing, every lookup returns None and the panels silently fall back to hardcoded
    constants. Drive the actual flattener + deriver over a config shaped like the runner's, and
    assert the value the panel would draw."""
    import importlib.util
    import tempfile
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pfd = os.path.join(os.path.dirname(root), "src", "ac2", "viz", "parse_fig_data.py")
    if not os.path.exists(pfd):
        return  # viz not present in this checkout
    # import the module standalone (it is stdlib-only by design)
    spec = importlib.util.spec_from_file_location("_pfd", pfd)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pfd"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # a dependency appeared; skip rather than lie
        print(f"  (skipped: parse_fig_data not importable here: {e})")
        return

    cfg_yaml = """
trainer:
  experiment_name: ac2
  nnodes: 8
data:
  train_batch_size: 128
  max_response_length: 75000
  sp_q_ready_thresh: 0.18
  sp_q_ready_thresh_global: 0.2
  sp_q_ready_thresh_problem: 0.18
  sp_q_fifo_cap: 1920
  sp_q_train_n: 768
  sp_q_budget_g: 10000
  sp_q_train_noref: 1
  sp_q_prompt_variant: reward_horizon
  sp_q_ref_require_pass: 1
  sp_q_lr_ladder: 1
  sp_q_lr_ratio_max: 1.0
  sp_q_lr_floor: 5e-07
"""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "config.yaml")
        open(path, "w").write(cfg_yaml)
        from pathlib import Path as _P
        flat = mod._load_config_yaml(_P(path), set())

    if flat:
        # INPUT side, end to end: the flattener must actually carry the Q knobs.
        assert flat.get("sp_q_ready_thresh") in (0.18, "0.18"), (
            "_load_config_yaml drops data.sp_q_ready_thresh -- the dashboard falls back to "
            f"the panel's hardcoded fallback instead of the run's own pin "
            f"(got {flat.get('sp_q_ready_thresh')!r})"
        )
        assert flat.get("sp_q_lr_ladder") in (1, "1"), "the ladder regime never reaches the parser"
        assert flat.get("sp_q_fifo_cap") in (1920, "1920")
        # the SPLIT readiness thresholds, each on its own -- a missing one silently falls back
        # to the single knob and the panel then draws a gate the run never used
        assert flat.get("sp_q_ready_thresh_global") in (0.2, "0.2"), \
            "_load_config_yaml drops data.sp_q_ready_thresh_global"
        assert flat.get("sp_q_ready_thresh_problem") in (0.18, "0.18"), \
            "_load_config_yaml drops data.sp_q_ready_thresh_problem"
    else:
        # pyyaml absent (laptop): the flattener returns {} and cannot be exercised, so assert the
        # extraction EXISTS in source and keep the deriver half of the test honest below.
        psrc = open(pfd, encoding="utf-8").read()
        for key in ("sp_q_ready_thresh", "sp_q_ready_thresh_global", "sp_q_ready_thresh_problem",
                    "sp_q_fifo_cap", "sp_q_lr_ladder", "sp_q_budget_g"):
            assert f'"{key}": g("data", "{key}")' in psrc, (
                f"_load_config_yaml does not extract data.{key}; _derive_config's lookup for it "
                "returns None and the dashboard silently uses the default constant"
            )
        print("  (pyyaml absent: input side checked by source inspection)")
        flat = {
            "sp_q_ready_thresh": 0.18, "sp_q_ready_thresh_global": 0.2,
            "sp_q_ready_thresh_problem": 0.18, "sp_q_fifo_cap": 1920, "sp_q_train_n": 768,
            "sp_q_lr_ladder": 1, "sp_q_lr_ratio_max": 1.0, "sp_q_lr_floor": 5e-07,
        }

    # OUTPUT side: _derive_config must surface them under the *_default names the panels read
    derived = mod._derive_config(flat, {}, None, None)
    assert float(derived["sp_q_ready_thresh_default"]) == 0.18, derived["sp_q_ready_thresh_default"]
    # the split thresholds must reach the panels SEPARATELY
    assert float(derived["sp_q_ready_thresh_global_default"]) == 0.2
    assert float(derived["sp_q_ready_thresh_problem_default"]) == 0.18
    assert int(derived["sp_q_lr_ladder_default"]) == 1
    assert float(derived["sp_q_lr_ratio_max_default"]) == 1.0
    assert int(derived["sp_q_fifo_cap_default"]) == 1920
    # and the panel's own fallback logic must then pick the LADDER regime, not the movement cap
    assert derived.get("sp_q_rho_cap_default") is None, (
        "a ladder run must not report a movement-cap value; the displacement panel would draw "
        "the wrong reference line"
    )

    # A run that sets only the single legacy threshold knob must still derive the OLD behaviour.
    one_knob = mod._derive_config({"sp_q_ready_thresh": 0.18}, {}, None, None)
    assert float(one_knob["sp_q_ready_thresh_global_default"]) == 0.18, (
        "a run that sets only the single knob must have BOTH gates fall back to it, or the "
        "dashboard draws a threshold the run never used"
    )
    assert float(one_knob["sp_q_ready_thresh_problem_default"]) == 0.18

    legacy = mod._derive_config({}, {}, None, None)
    assert legacy["sp_q_ready_thresh_default"] is None, (
        "absent config must stay None so the panel applies its documented 0.18 fallback"
    )
    assert legacy["sp_q_lr_ladder_default"] is None
    # ...and a legacy run that logged q/s but never q/lr_current must NOT be called a ladder run
    inferred = mod._derive_config({}, {1: {"q/s": 0.4}}, None, None)
    assert not inferred["sp_q_lr_ladder_default"]
    # ...while one that logged q/lr_current is inferred as a ladder run even with no config keys
    inferred2 = mod._derive_config({}, {1: {"q/lr_current": 2e-6}}, None, None)
    assert inferred2["sp_q_lr_ladder_default"] == 1


# ------------------- 9. the rejection metric must not depend on WHERE in the step it counts
def test_rejection_metric_is_a_delta_not_an_order_dependent_reset():
    """The wave rejects references BEFORE observe_and_update runs, and the per-entry cache means
    a later lookup does not re-count. A reset inside observe_and_update therefore zeroed exactly
    the rejections that had happened. The metric must be a delta of the cumulative total."""
    src = open(SP_Q, encoding="utf-8").read()
    assert "ref_rejected_unpassed_step" not in src, (
        "the order-dependent per-step counter is back; the wave's rejections will be reset "
        "before they are ever reported"
    )
    assert "_ref_rej_total_at_step_start" in src, "no per-step window snapshot"
    # the window must close exactly once per step, in append_delta (after the update)
    assert src.count("self._ref_rej_total_at_step_start = self.ref_rejected_unpassed_total") >= 2, \
        "the window is never closed in append_delta and/or not re-based on resume"
    # the cumulative total must still survive a relaunch (it is the only monotone series)
    assert '"ref_rejected_unpassed_total": self.ref_rejected_unpassed_total' in src, \
        "the cumulative counter is not saved into q_state.json; it would sawtooth on resume"
    assert 'st.get("ref_rejected_unpassed_total"' in src, "...and is never restored"
    # simulate: increments during the wave must survive into the reported per-step number
    total, window = 0, 0
    total += 7                                        # wave rejects 7 (pre-reward)
    reported = total - window                         # metric read at observe/admission time
    assert reported == 7, "wave rejections must be visible in the per-step metric"
    window = total                                    # append_delta closes the step
    reported_next = total - window
    assert reported_next == 0, "a step with no new rejections must report 0, not the total"


# ------------------------- 10. step-cache equivalence: refs resolved before eviction
def test_refs_are_prewarmed_before_replay_eviction():
    """A postwave/postreward resume rehydrates the wave RESULTS but not the proof cache, and the
    replay update (which evicts) runs BEFORE Q admission -- so a resumed step could lose the
    source entry and silently train on a different reference than the uninterrupted run."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rt = os.path.join(os.path.dirname(root), "src", "verl", "verl", "trainer", "ppo",
                      "ray_trainer.py")
    if not os.path.exists(rt):
        return
    src = open(rt, encoding="utf-8").read()
    assert "prewarm_trajectory_refs" in src, "no pre-warm call at the reward site"
    i_pre = src.index("prewarm_trajectory_refs")
    i_replay = src.index("_sp_replay.observe_and_update")
    assert i_pre < i_replay, (
        "the pre-warm must run BEFORE the replay update: eviction can remove the entry whose "
        "proof a Q record needs, and then a resumed step and an uninterrupted step disagree"
    )
    q_src = open(SP_Q, encoding="utf-8").read()
    assert "def prewarm_trajectory_refs" in q_src


# ------------------------------- 11. the judge model is fixed, and its revision is pinned
def test_judge_is_the_lineage_model_at_a_pinned_revision():
    """The judge IS the reward function. The run keeps one fixed judge model so rewards stay on
    one scale, and pins the REVISION so a hub-cache re-download cannot move it. Both halves
    matter: the right repo at a floating revision is not reproducible."""
    here = os.path.dirname(os.path.abspath(__file__))
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    attach = open(os.path.join(here, "run_attach.sh"), encoding="utf-8").read()
    lineage_rev = "60d8d70770c6776ff598c94bb586a859a38244f1"

    # the pinned judge model -- NOT a newer release
    assert '"deepseek-ai/DeepSeek-V4-Flash"' in runner, "runner no longer defaults to the reported runs' judge"
    # Strip comments before checking for the newer judge release: a comment may name it, and a
    # bare substring test would flag that explanation (the same trap as the resharder test
    # above). Only CODE may not mention it.
    def _code(text, marker):
        return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(marker))
    assert "DeepSeek-V4-Flash-0731" not in _code(runner, "#"), \
        "the 0731 judge is back in runner CODE (not just a comment)"
    assert "DeepSeek-V4-Flash-0731" not in _code(attach, "#"), \
        "the 0731 judge is back in attach CODE (not just a comment)"
    assert "deepseek-ai/DeepSeek-V4-Flash}" in attach or \
           'JUDGE_HF_REPO="${SP_JUDGE_HF_REPO:-deepseek-ai/DeepSeek-V4-Flash}"' in attach

    # and the exact revision, pinned rather than resolved from refs/main
    assert lineage_rev in runner, "the runner does not pin the reported runs' judge revision"
    assert lineage_rev in attach, "the attach does not pin the reported runs' judge revision"
    # the resolver must PREFER the pin over refs/main
    i_pin = runner.index("if _JUDGE_REVISION:")
    i_ref = runner.index('ref = repo_dir / "refs" / "main"')
    assert i_pin < i_ref, (
        "refs/main is consulted before the pinned revision; a re-download would silently change "
        "the reward function"
    )
    # ...and the pin must be recorded per run so a later launch can detect drift
    assert '"revision": _JUDGE_REVISION' in runner, \
        "the resolved revision is not written into judge_snapshot.json"
    assert "SP_ALLOW_JUDGE_DRIFT" in runner, "no escape hatch documented for a deliberate change"


# ---------------------- 12. the two readiness gates are separate and independent
def test_readiness_gates_are_split_and_independent():
    """The single readiness threshold is split into a GLOBAL (pooled MAE5) gate and a
    PER-PROBLEM (fresh error) gate. Two things must hold: each check reads its OWN threshold,
    and a run that sets only the legacy knob still gets the old one-number behaviour."""
    src = open(SP_Q, encoding="utf-8").read()

    # each check must consult its own attribute -- not the shared legacy one
    assert "mae5 < self.ready_thresh_global" in src, "the pooled gate does not use the global threshold"
    assert 'prb["error"] < self.ready_thresh_problem' in src, \
        "the per-problem gate does not use the per-problem threshold"
    # and the old single-knob comparison must be GONE from the flip block
    i_flip = src.index("transitions = []")
    flip_block = src[i_flip:i_flip + 700]
    assert "self.ready_thresh:" not in flip_block, (
        "the flip block still compares against the shared legacy threshold somewhere; one of "
        "the two gates is not actually split"
    )
    # both default to the legacy knob, so a run that sets only that knob is unchanged
    assert 'cfg.get("sp_q_ready_thresh_global", self.ready_thresh)' in src
    assert 'cfg.get("sp_q_ready_thresh_problem", self.ready_thresh)' in src
    # both are surfaced per step and persisted
    for key in ('"q/ready_thresh_global"', '"q/ready_thresh_problem"', '"q/global_gate_open"',
                '"ready_thresh_global": self.ready_thresh_global'):
        assert key in src, f"{key} missing -- the split is not observable/persisted"

    # the gates must be INDEPENDENT: simulate the decision table
    def flips(mae5, err, g=0.2, pp=0.18):
        return (mae5 is not None and mae5 < g) and (err < pp)
    assert flips(0.19, 0.10) is True, "global open + problem good -> ready"
    assert flips(0.19, 0.19) is False, (
        "an error of 0.19 passes the GLOBAL threshold but must fail the per-problem one -- this "
        "is the case the old single-knob gate could not express"
    )
    assert flips(0.25, 0.05) is False, "global gate closed -> nothing flips, however good the probe"
    assert flips(None, 0.05) is False, "incomplete 5-step window -> nothing flips"
    # and with one knob for both, the split must degenerate to the old behaviour
    assert flips(0.19, 0.19, g=0.18, pp=0.18) is False
    assert flips(0.17, 0.17, g=0.18, pp=0.18) is True


# --------------------------------- 13. TP>1 rollout is gated on the flashinfer AOT preflight
def test_tp_gt_1_is_gated_on_the_aot_preflight():
    """SP_ROLLOUT_TP=4 is only startable because the flashinfer modules are promoted to AOT
    first. Wire the two together: a future edit that keeps TP=4 but drops (or softens) the
    preflight reintroduces a wedge that shows up as a 0%-util hang minutes into engine init,
    which is exactly the failure nobody diagnoses correctly under time pressure."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach.sh"), encoding="utf-8").read()
    runner = open(os.path.join(here, "runner.py"), encoding="utf-8").read()
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))

    m = re.search(r'"SP_ROLLOUT_TP=(\d+)"', code)
    assert m, "the attach must PIN SP_ROLLOUT_TP, not inherit the runner default"
    tp = int(m.group(1))
    assert 8 % tp == 0, f"TP={tp} does not divide the 8 GPUs per node"

    # The runner default is deliberately NOT tp: it is 1, the only value that is safe without the
    # preflight. The attach is what pins 4. So a launch that bypasses the attach runs SLOWER than
    # intended rather than wedging on a cold flashinfer cache -- the failure mode you can read off
    # a throughput number instead of a 0%-util hang. Cost of that choice: the effective TP lives in
    # exactly one place, so read run_attach.sh, never runner.py, to know what a run used.
    dm = re.search(r'ROLLOUT_TP\s*=\s*_env\("SP_ROLLOUT_TP",\s*(\d+)', runner)
    assert dm, "runner.py no longer reads SP_ROLLOUT_TP"
    assert int(dm.group(1)) == 1, (
        f"the runner default is {dm.group(1)}, not 1; a launch that bypasses the attach would then "
        "take the TP>1 path with no guarantee the AOT preflight ever ran"
    )

    if tp > 1:
        assert "flashinfer_aot_warm.py" in code, (
            "TP>1 without the AOT preflight: every rank JIT-rebuilds flashinfer's trtllm_comm "
            "under one FileLock, so engine init burns TP x build-time at 0% GPU util and wedges"
        )
        assert "[aot] READY" in code, "the preflight must check for the READY marker"
        # and it must be FATAL -- a warning would be silently ignored on every relaunch
        gate = code[code.index("flashinfer_aot_warm.py"):]
        gate = gate[:gate.index("SEED=")] if "SEED=" in gate else gate
        assert re.search(r"^\s*exit 1", gate, re.M), (
            "the AOT preflight only warns; it must abort the launch"
        )

    # the warm script really does emit that marker and cover the module vLLM's fusion pass needs
    warm = os.path.join(os.path.dirname(here), "ac2",
                        "flashinfer_aot_warm.py")
    if os.path.exists(warm):
        w = open(warm, encoding="utf-8").read()
        assert "[aot] READY" in w, "the preflight greps for a marker the warm script never prints"
        assert "trtllm_comm" in w, "the warm script skips the allreduce module, which is the one "\
                                   "the AllReduceRMSFusionPass builds at TP>1"


# --------------------------------------- 14. the context group is internally consistent at 75k
def test_context_group_moves_together():
    """The five context budgets are coupled by arithmetic the runner asserts at import
    (C_Q == 2048 + response + 1248) and by a memory invariant it does NOT check. Editing the
    response budget alone is the easy mistake -- it crashes at import if you are lucky and
    silently mis-packs if you are not -- so pin the whole group's arithmetic here."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach.sh"), encoding="utf-8").read()
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))
    v = {}
    for k in ("SP_MAX_RESPONSE_LEN", "SP_Q_CTX_LIMIT", "SP_Q_MAX_TOKEN_LEN",
              "SP_PPO_MAX_TOKEN_LEN", "SP_ACTOR_GPU_MEM_UTIL"):
        m = re.search(rf'"{k}=([0-9.]+)"', code)
        assert m, f"the attach must PIN {k} -- the context group moves together or not at all"
        v[k] = float(m.group(1))

    resp, prompt, head = v["SP_MAX_RESPONSE_LEN"], 2048, 1248
    assert resp == 75000, f"response budget is {resp:.0f}, expected 75000"
    # (a) the equality runner.py asserts at import
    assert v["SP_Q_CTX_LIMIT"] == prompt + resp + head, (
        f"C_Q ({v['SP_Q_CTX_LIMIT']:.0f}) != 2048+{resp:.0f}+1248 -- runner.py raises on this"
    )
    # (b) Q prompts reach C_Q, so their budget must clear it (the runner's own default is C_Q+64)
    assert v["SP_Q_MAX_TOKEN_LEN"] == v["SP_Q_CTX_LIMIT"] + 64, (
        f"SP_Q_MAX_TOKEN_LEN ({v['SP_Q_MAX_TOKEN_LEN']:.0f}) must be C_Q+64 "
        f"({v['SP_Q_CTX_LIMIT'] + 64:.0f}); Q prompts reach C_Q and would be truncated"
    )

    # (c) the memory invariant NOTHING checks at runtime: the PPO per-GPU budget must admit
    # exactly ONE maximum-length row. Two would OOM the actor update at this model and batch;
    # fewer than one would mean a full row cannot be formed at all.
    full_row = prompt + resp
    ratio = v["SP_PPO_MAX_TOKEN_LEN"] / full_row
    assert ratio < 2, (
        f"SP_PPO_MAX_TOKEN_LEN is {ratio:.2f}x a full {full_row:.0f}-token row -- two max-length "
        "rows can pack into one GPU micro-batch and the actor update OOMs"
    )
    assert ratio > 0.9, (
        f"SP_PPO_MAX_TOKEN_LEN is only {ratio:.2f}x a full row; well below 1 starts costing "
        "throughput for no memory benefit"
    )

    # (d) the arena must hold enough full-length sequences to be worth the rollout batch, and
    # SP_ACTOR_MAX_NUM_SEQS must not be the binding constraint (else it, not KV, caps concurrency)
    tp = int(re.search(r'"SP_ROLLOUT_TP=(\d+)"', code).group(1))
    KV_PER_TOKEN, WEIGHTS, OVERHEAD, TOTAL = 2 * 36 * 8 * 128 * 2, int(4.02e9) * 2, 2 << 30, 85520809984
    per_gpu = v["SP_ACTOR_GPU_MEM_UTIL"] * TOTAL - OVERHEAD
    kv_seqs = int((tp * per_gpu - WEIGHTS) // (KV_PER_TOKEN * v["SP_Q_CTX_LIMIT"]))
    assert kv_seqs >= 8, (
        f"the arena holds only {kv_seqs} full-length sequences per TP={tp} engine; the rollout "
        "will serialize and generation time will not match the benchmark"
    )
    cm = re.search(r'"SP_ACTOR_MAX_NUM_SEQS=(\d+)"', code) or re.search(
        r'_env\("SP_ACTOR_MAX_NUM_SEQS",[^,]*?(\d+)\), int\)|"rollout_max_num_seqs", (\d+)\)',
        open(os.path.join(here, "runner.py"), encoding="utf-8").read())
    cap = int(next(g for g in (cm.groups() if cm else ()) if g) if cm else 96)
    assert cap > kv_seqs, (
        f"SP_ACTOR_MAX_NUM_SEQS={cap} is at or below the KV-bound {kv_seqs}; it has silently "
        "become the concurrency limit, so the documented 'KV binds first' reasoning is stale"
    )


# ------------------- 15. checkpoint retention keeps every 10th step's WEIGHTS, permanently
def test_retention_keeps_every_tenth_step():
    """The rolling window alone makes mid-run steps unrecoverable: keep_last=5 with save_freq=1
    means step 137's weights are gone by step 143, so nothing later can re-evaluate or branch
    from it. Exercise the REAL fork function (not a reimplementation) against a fake checkpoint
    tree, because the failure mode is silent and destructive -- by the time anyone notices, the
    weights are already deleted."""
    import contextlib, io, json as _json, shutil, tempfile, types

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = os.path.join(root, "src", "verl", "verl", "trainer", "ppo", "ray_trainer.py")
    assert os.path.exists(src), (
        f"cannot find the verl fork at {src}; this test silently passed for one revision "
        "because the path was one directory short -- fail loudly instead"
    )

    # Import ONLY the method's source, not the module (ray_trainer imports torch/ray).
    text = open(src, encoding="utf-8").read()
    start = text.index("    def _retain_recent_and_best(self, train_reward):")
    end = text.index("\n    def ", start + 10)
    ns = {"os": os, "json": _json}
    exec("import os, json\nclass T:\n" + text[start:end], ns)
    fn = ns["T"]._retain_recent_and_best

    def build(tmp, steps):
        for st in steps:
            for sub in ("actor", "sp_q_optim"):
                os.makedirs(os.path.join(tmp, f"global_step_{st}", sub), exist_ok=True)
                open(os.path.join(tmp, f"global_step_{st}", sub, "w.pt"), "w").close()
            open(os.path.join(tmp, f"global_step_{st}", "data.pt"), "w").close()

    def surviving(steps, cur, keep_last=5, every=None, reward=0.1):
        tmp = tempfile.mkdtemp()
        try:
            build(tmp, steps)
            cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(
                default_local_dir=tmp, get=lambda k, d=None: {"max_actor_ckpt_to_keep": keep_last}.get(k, d)))
            self_ = types.SimpleNamespace(config=cfg, global_steps=cur)
            old = os.environ.get("SP_CKPT_KEEP_EVERY")
            if every is None:
                os.environ.pop("SP_CKPT_KEEP_EVERY", None)      # exercise the DEFAULT
            else:
                os.environ["SP_CKPT_KEEP_EVERY"] = str(every)
            try:
                # the real function logs one line per pruned dir; keep the suite readable
                with contextlib.redirect_stdout(io.StringIO()):
                    fn(self_, reward)
            finally:
                os.environ.pop("SP_CKPT_KEEP_EVERY", None)
                if old is not None:
                    os.environ["SP_CKPT_KEEP_EVERY"] = old
            w = {st for st in steps
                 if os.path.isdir(os.path.join(tmp, f"global_step_{st}", "actor"))}
            q = {st for st in steps
                 if os.path.isdir(os.path.join(tmp, f"global_step_{st}", "sp_q_optim"))}
            d = {st for st in steps
                 if os.path.exists(os.path.join(tmp, f"global_step_{st}", "data.pt"))}
            assert q == w, (
                f"sp_q_optim did not rotate with the weights it belongs to: actor={sorted(w)} "
                f"sp_q_optim={sorted(q)}; a tied-Q optimizer is as heavy as the model"
            )
            return w, d
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    steps = list(range(1, 44))
    # DEFAULT (no env var set) must already retain the stride -- "every active run", no opt-in
    w, d = surviving(steps, 43)
    assert {10, 20, 30, 40} <= w, f"multiples of 10 were pruned by default: kept {sorted(w)}"
    assert {39, 40, 41, 42, 43} <= w, f"the last-5 window was broken: kept {sorted(w)}"
    assert 43 in w and 7 not in w, f"a non-multiple outside the window survived: {sorted(w)}"
    # the small per-step state is NEVER pruned -- the dashboard reads every step
    assert d == set(steps), "data.pt was pruned; the dashboard would lose history"
    # (sp_q_optim/weights co-rotation is asserted inside surviving() on every call)

    # explicit stride
    w5, _ = surviving(steps, 43, every=5)
    assert {5, 10, 15, 20, 25, 30, 35, 40} <= w5, f"stride 5 not honoured: {sorted(w5)}"
    # OFF switch restores last-N + best
    w0, _ = surviving(steps, 43, every=0)
    assert 10 not in w0 and 20 not in w0, f"SP_CKPT_KEEP_EVERY=0 still retained a stride: {sorted(w0)}"
    assert {39, 40, 41, 42, 43} <= w0
    # a malformed value must fall back to the DEFAULT, not silently disable retention
    wbad, _ = surviving(steps, 43, every="ten")
    assert {10, 20, 30, 40} <= wbad, f"a bad SP_CKPT_KEEP_EVERY disabled the stride: {sorted(wbad)}"


# ---------------------------------- 16. the AOT warm must share the engines' flashinfer dir
def test_aot_preflight_shares_the_engine_workspace():
    """flashinfer derives FLASHINFER_AOT_DIR from FLASHINFER_WORKSPACE_BASE. If the warm step
    runs without the value the ENGINES get, it promotes into a different directory, prints
    '[aot] READY' truthfully about that one, and every rank still JIT-builds -- a gate that
    reports success while protecting nothing. It must be EXPORTED (the ENVS array only reaches
    the driver) and the two must read the same variable so they cannot drift."""
    here = os.path.dirname(os.path.abspath(__file__))
    attach = open(os.path.join(here, "run_attach.sh"), encoding="utf-8").read()
    code = "\n".join(l for l in attach.splitlines() if not l.lstrip().startswith("#"))
    if "flashinfer_aot_warm.py" not in code:
        return                                    # TP=1: no preflight to check

    m = re.search(r"^\s*export FLASHINFER_WORKSPACE_BASE=", code, re.M)
    assert m, ("FLASHINFER_WORKSPACE_BASE is not exported before the warm step, so it warms "
               "flashinfer's HOME-based default while the engines read the pinned cache dir")
    assert m.start() < code.index("flashinfer_aot_warm.py"), (
        "the export comes AFTER the warm step; it must precede it"
    )
    assert '"FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"' in code, (
        "ENVS pins a second literal path instead of reusing the exported variable -- the warm "
        "step and the engines can silently diverge"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
