"""AC2 training entry point (ac2).

Builds the verl PPO/Hydra configuration from SP_* environment variables, asserts the AC2
switches, writes the manifest and calls `verl.trainer.main_ppo.run_ppo`. It is started by
run_attach.sh, which pins every value a run used; the module defaults below are a
smaller self-consistent configuration (batch 128, 50k response budget), so read the attach,
not these defaults, to know what a run used.

  * Policy and critic V share one set of weights (Qwen3-4B-Thinking-2507). The critic has its
    own AdamW optimizer and takes one step between the two actor minibatches (SP_Q_INTERLEAVE);
    its LR is halved after SP_Q_LR_BREACH_PATIENCE consecutive steps with
    ||dV|| / (||dPPO_1|| + ||dPPO_2||) > SP_Q_LR_RATIO_MAX, down to SP_Q_LR_FLOOR.
  * Each step trains on SP_REPLAY_N prefixes cut from the replay buffer; the other fresh
    problems are rolled out once to refill it and are not trained on (SP_SCRATCH_INFLOW_ONLY).
  * A problem is ready when the 5-step critic error is below SP_Q_READY_THRESH_GLOBAL and its
    own latest error is below SP_Q_READY_THRESH_PROBLEM. Ready prefixes get action chunks of at
    most b = SP_Q_BUDGET_G new tokens scored by the critic, except 1/SP_Q_AUDIT_DEN of ready
    problems, which are audited with full rollouts.
  * The critic prompt asks for the expected rubric credit within the remaining token budget;
    the with- and without-reference prompts are both trained.
  * Cold start: replay buffer, critic buffer and reference bank start empty
    (build_cold_artifacts.py); step 1's replay slots become trained `cold_scratch` rows.
  * Reward = judge points / 7, no length penalty; the judge is DeepSeek-V4-Flash at a pinned
    revision, recorded per run in run_data/judge_snapshot.json.
"""

import json
import os
import shutil
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

import verl.trainer.main_ppo as main_ppo
import ac2.rewards.ds4_finegrained_judge as ds4_judge_reward
from ac2.utils.experiment_utils import manifest_dump

try:
    from ac2.site_env import site_config
    _SITE = site_config()
except Exception:
    _SITE = {
        "hf_hub_cache": os.environ.get("AC2_HF_HUB_CACHE", ""),
        "hf_home": os.environ.get("AC2_HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "cache_root": os.path.expandvars("${AC2_SCRATCH}/.cache"),
        "wandb_mode": os.environ.get("AC2_WANDB_MODE", "offline"),
    }

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = os.environ.get("SP_EXPERIMENT_NAME", EXP_DIR.name)
RUN_DATA_NAME = os.environ.get("SP_RUN_DATA_DIR", "run_data")
REPO_ROOT = EXP_DIR.parents[1]
MANIFEST_DIR = EXP_DIR / "manifest"
RUN_DATA = EXP_DIR / RUN_DATA_NAME
METRICS_PATH = RUN_DATA / "metrics.jsonl"
WANDB_ENTITY = os.environ.get("AC2_WANDB_ENTITY", "")
# A from-scratch run inherits nothing: no weights, no optimizer state, no step counter and no
# seed data. By default every path this run reads state from is its own.

REWARD_FILE = ds4_judge_reward.__file__
FINEGRAINED_TEMPLATE = str(Path(REWARD_FILE).parent / "templates" / "finegrained_noref_judge.txt")
VAL_JUDGE_TEMPLATE = str(Path(REWARD_FILE).parent / "templates" / "imo_proofautograder.txt")
VAL_MAP_PATH = os.path.expanduser(
    os.environ.get("SP_VAL_MAP", os.path.join(
        os.environ.get("SELF_PLAY_DATA_DIR", "~/data/fineproofs"), "val_map.json")))
VERL_CONFIG_DIR = str(Path(main_ppo.__file__).parent / "config")

NNODES = int(os.environ.get("NNODES", 8))
N_GPUS_PER_NODE = int(os.environ.get("N_GPUS_PER_NODE", 8))
DATA_DIR = os.environ.get("SELF_PLAY_DATA_DIR", f"{os.environ['HOME']}/data/fineproofs")


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default


def _truthy(name, default="0"):
    return os.environ.get(name, default) in ("1", "true", "True")


# ---- hard gates: every switch that changes the method must be set explicitly ----
assert _truthy("SP_REPLAY_ENABLE"), (
    "This run requires SP_REPLAY_ENABLE=1 (the Q mechanism rides on the replay dataset)."
)
assert _truthy("SP_Q_ENABLE"), (
    "This run requires SP_Q_ENABLE=1; running without it would silently train GRPO on replayed prefixes "
    "from scratch."
)
assert not _truthy("SP_Q_SEPARATE"), (
    "This run uses a WEIGHT-TIED critic: one set of weights, two optimizers. "
    "SP_Q_SEPARATE=1 would switch to a separate critic model (separate theta_Q), which has no "
    "shared weights to interleave into and no rho to price -- exactly the thing this run "
    "exists to reverse."
)
assert _truthy("SP_Q_INTERLEAVE"), (
    "This run requires SP_Q_INTERLEAVE=1: the interleaved update order is PPO(M1) -> Q-only -> PPO(M2) "
    "with the Q step left applied. Without it the trainer runs the older capture-at-theta_0 "
    "phase and the LR ladder would price a displacement that was never applied in that order."
)
assert _truthy("SP_Q_LR_LADDER"), (
    "This run requires SP_Q_LR_LADDER=1. Without it the interleaved Q step is "
    "taken at a CONSTANT lr with no size control at all -- the movement cap that used to "
    "bound it does not exist on this path."
)
assert not _truthy("SP_DIFF_SAMPLING")
# No length penalty. The reward module (ds4_finegrained_judge) reads SP_LENPEN_ENABLE from the
# process environment. The attach pins it to 0; assert it here so an inherited export from
# another experiment's shell cannot silently reinstate a shaped reward. It is the only switch
# that turns the penalty on.
assert not _truthy("SP_LENPEN_ENABLE"), (
    "This run runs with NO length penalty, but SP_LENPEN_ENABLE is set to "
    f"{os.environ.get('SP_LENPEN_ENABLE')!r} in this environment. That would multiply every "
    "correct proof's reward by a length term and change what Q fits. Unset it (the attach "
    "script pins SP_LENPEN_ENABLE=0) or run a different experiment."
)

# ---- replay-buffer knobs; the seed is EMPTY for a cold start ----
REPLAY_SEED_DIR   = os.environ.get("SP_REPLAY_SEED_DIR", str(EXP_DIR / "replay_seed_cold"))
REPLAY_COLD       = _truthy("SP_REPLAY_COLD_BOOTSTRAP", "1")
REPLAY_N          = _env("SP_REPLAY_N", 96, int)
REPLAY_CAPACITY   = _env("SP_REPLAY_CAPACITY", 3, int)
REPLAY_CUT_LOW    = _env("SP_REPLAY_CUT_LOW", 0.0, float)
REPLAY_CUT_HIGH   = _env("SP_REPLAY_CUT_HIGH", 0.90, float)
REPLAY_CUT_GRAIN  = _env("SP_REPLAY_CUT_GRAIN", 10000, int)
REPLAY_EMA_COEF   = _env("SP_REPLAY_EMA_COEF", 0.95, float)
REPLAY_RNG_SEED   = _env("SP_REPLAY_RNG_SEED", 714001, int)
if not (Path(REPLAY_SEED_DIR) / "replay_buffer_manifest.json").exists():
    raise FileNotFoundError(
        f"replay seed missing at {REPLAY_SEED_DIR} — run build_cold_artifacts.py first (this run "
        "starts from a genuinely EMPTY buffer; sp_replay's loader still requires a manifest "
        "and one empty shard)"
    )
# A cold start inherits no trajectory. A non-empty seed here would quietly restore a warm
# buffer while the logs and the state file still claim a cold start.
if REPLAY_COLD:
    _seed_n = sum(
        1
        for _p in sorted((Path(REPLAY_SEED_DIR) / "replay_buffer").glob("shard_*.jsonl"))
        for _l in open(_p, encoding="utf-8")
        if _l.strip()
    )
    if _seed_n:
        raise ValueError(
            f"SP_REPLAY_COLD_BOOTSTRAP=1 but {REPLAY_SEED_DIR} holds {_seed_n} entries. This run "
            "starts from an EMPTY buffer: point SP_REPLAY_SEED_DIR at the cold seed, or set "
            "SP_REPLAY_COLD_BOOTSTRAP=0 to run the warm-seed configuration deliberately."
        )
    print(f"[replay] COLD BOOTSTRAP: empty buffer at {REPLAY_SEED_DIR}. Step 1 has NO replay "
          f"rows — its 96 replay slots become trained cold_scratch rows on fresh problems; the "
          f"32-row inflow lane fills the buffer from step 1's own trajectories.", flush=True)

# ---- replay management policy (one global FIFO buffer) ----
REPLAY_BUCKETING  = os.environ.get("SP_REPLAY_BUCKETING", "global")
REPLAY_ADMISSION  = os.environ.get("SP_REPLAY_ADMISSION", "ungated")
REPLAY_ROTATION   = os.environ.get("SP_REPLAY_ROTATION", "global_fifo")
REPLAY_BOUND      = _env("SP_REPLAY_BOUND", 128, int)
# "entry" (uniform over stored trajectories; the default) or "question" (uniform over distinct
# problems, then one of that problem's trajectories). The attach scripts set "question".
REPLAY_GLOBAL_SAMPLING = os.environ.get("SP_REPLAY_GLOBAL_SAMPLING", "entry")
REPLAY_STALE      = _env("SP_REPLAY_STALE_STEPS", 0, int)
INFLOW_ONLY       = _truthy("SP_SCRATCH_INFLOW_ONLY", "1")

# ---- generative-Q (critic) knobs ----
Q_SEED_DIR    = os.environ.get("SP_Q_SEED_DIR", str(EXP_DIR))       # cold: this run's own empty seed
Q_BANK_DIR    = os.environ.get("SP_Q_BANK_DIR", str(EXP_DIR))       # cold: fills add-once online
Q_BUDGET_G    = _env("SP_Q_BUDGET_G", 10000, int)
# ---- readiness: two thresholds ------------------------------------------------------------
# The gate has two checks:
#   GLOBAL  -- pooled 5-step model-wide MAE < thresh   ("is Q calibrated overall?")
#   PROBLEM -- this problem's fresh probe error < thresh ("is Q right about THIS problem?")
# They are different questions, so they get different values: 0.20 global / 0.18 per-problem.
# The global gate is the LOOSER one: it decides only whether readiness may open at all, and
# with a cold critic buffer, a cold bank and the base policy a tighter global gate risks a run
# in which it never opens and the mechanism is never exercised. The per-problem gate decides
# whether a specific problem's continuations are cut to an action chunk and scored by Q instead
# of the judge; that is where a wrong call actually costs reward, so it stays at 0.18.
# SP_Q_READY_THRESH remains the single legacy knob and the default for both.
Q_READY_THRESH = _env("SP_Q_READY_THRESH", 0.18, float)
Q_READY_THRESH_GLOBAL = _env("SP_Q_READY_THRESH_GLOBAL", 0.2, float)
Q_READY_THRESH_PROBLEM = _env("SP_Q_READY_THRESH_PROBLEM", Q_READY_THRESH, float)
# Critic-buffer capacity in RECORDS. 1,920 = a 20-step horizon at 96 replay groups per step
# (the default batch shape); because the cap is a record count, the horizon it represents
# shrinks as SP_REPLAY_N grows, so re-derive it if the batch shape changes.
Q_FIFO_CAP    = _env("SP_Q_FIFO_CAP", 1920, int)
Q_MIN_VALID   = _env("SP_Q_MIN_VALID", 8, int)          # valid rewards a group needs to give a target
Q_TRAIN_N     = _env("SP_Q_TRAIN_N", 768, int)          # prefix-target pairs per critic update
Q_RNG_SEED    = _env("SP_Q_RNG_SEED", 804001, int)
Q_AUDIT_CUT   = _env("SP_Q_AUDIT_CUT", 1, int)
Q_AUDIT_DEN   = _env("SP_Q_AUDIT_DEN", 4, int)          # audit 1/den of the ready problems
# Train BOTH prompt variants. A record with a resolvable reference contributes ref + noref
# rows at weight 1/2 each; a record with NO reference contributes the noref row at weight 1
# instead of being skipped entirely. That last clause is why this matters with a cold
# reference bank: ref-only training would starve on exactly the problems the run has not
# solved yet. Makes q/loss_noref a real number rather than a constant 0, at up to 2x the
# Q-phase rows (768 records -> <=1536 rows, which divide evenly over 32 or 64 ranks).
Q_TRAIN_NOREF = _env("SP_Q_TRAIN_NOREF", 1, int)
# Critic prompt wording. "reward_horizon" asks for the expected rubric credit (0 = none,
# 1 = full, in between = partial) within the remaining token budget, which is what the target
# z is: the grid-rounded mean of the group's points/7 rewards, censored by the rollout limit.
# "legacy" asks for the probability of a correct proof within a hardcoded 50K tokens. This IS
# the trained Q format: q_state.json records it and a resume under a different variant is
# refused.
Q_PROMPT_VARIANT = os.environ.get("SP_Q_PROMPT_VARIANT", "reward_horizon")
# Tier-1 references must be judged-correct trajectories. Default ON in the harness; pinned here
# so it appears in the manifest and cannot drift silently.
Q_REF_REQUIRE_PASS = _env("SP_Q_REF_REQUIRE_PASS", 1, int)
Q_GRAD_CLIP   = _env("SP_Q_GRAD_CLIP", 0.2, float)      # critic gradient clip
# ---- critic LR ladder. On the interleaved path these replace SP_Q_LR_BASE + SP_Q_RHO_CAP. ----
Q_LR_INITIAL  = _env("SP_Q_LR_INITIAL", 2e-6, float)
Q_LR_FLOOR    = _env("SP_Q_LR_FLOOR", 5e-7, float)
Q_LR_RATIO_MAX = _env("SP_Q_LR_RATIO_MAX", 1.0, float)
Q_LR_PATIENCE = _env("SP_Q_LR_BREACH_PATIENCE", 2, int)
Q_LR_FACTOR   = _env("SP_Q_LR_REDUCTION_FACTOR", 0.5, float)
Q_INTERLEAVE_AFTER = _env("SP_Q_INTERLEAVE_AFTER", 1, int)   # critic step after PPO minibatch 1
for _mf, _dir in (("q_seed_manifest.json", Q_SEED_DIR), ("reference_bank_manifest.json", Q_BANK_DIR)):
    if not (Path(_dir) / _mf).exists():
        raise FileNotFoundError(
            f"{_mf} missing in {_dir} — run build_cold_artifacts.py first (this run starts "
            "from an EMPTY Q FIFO and reference bank; the loaders still require manifests)"
        )
# ...and they must be EMPTY, not merely present, for a cold start: a stale non-empty q_seed/
# or reference_bank/ left over from a copied directory would seed Q from another policy's
# data while every log line still said "cold". Manifest existence alone would not catch that.
if REPLAY_COLD:
    for _label, _dir, _sub in (("q seed", Q_SEED_DIR, "q_seed"),
                               ("reference bank", Q_BANK_DIR, "reference_bank")):
        _n = sum(
            1
            for _p in sorted((Path(_dir) / _sub).glob("shard_*.jsonl"))
            for _l in open(_p, encoding="utf-8")
            if _l.strip()
        )
        if _n:
            raise ValueError(
                f"SP_REPLAY_COLD_BOOTSTRAP=1 but the {_label} at {_dir}/{_sub} holds {_n} "
                "records. this run starts from EMPTY Q artifacts — rebuild them with "
                "build_cold_artifacts.py, or set SP_REPLAY_COLD_BOOTSTRAP=0 to run a "
                "warm-start configuration deliberately."
            )
    print("[q] COLD artifacts verified EMPTY: q_seed 0 records, reference_bank 0 records",
          flush=True)

TOTAL_STEPS_DEFAULT = 500
_target_file = RUN_DATA / ".target_total_steps"
TOTAL_STEPS = (int(_target_file.read_text().strip()) if _target_file.exists()
               else _env("SP_TOTAL_STEPS", TOTAL_STEPS_DEFAULT, int))
TRAIN_BATCH       = _env("SP_TRAIN_BATCH_SIZE", 128, int)
ROLLOUT_N         = _env("SP_ROLLOUT_N", 16, int)
PPO_MINI          = min(_env("SP_PPO_MINI_BATCH", 48, int), TRAIN_BATCH)
LR                = _env("SP_LR", 1.4142135623730951e-6, float)
MAX_PROMPT_LEN    = _env("SP_MAX_PROMPT_LEN", 2048, int)
MAX_RESPONSE_LEN  = _env("SP_MAX_RESPONSE_LEN", 50000, int)
ROLLOUT_MAXLEN    = MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1248
Q_CTX_LIMIT       = _env("SP_Q_CTX_LIMIT", ROLLOUT_MAXLEN, int)
assert Q_CTX_LIMIT == ROLLOUT_MAXLEN, (
    f"C_Q ({Q_CTX_LIMIT}) must equal rollout max_model_len ({ROLLOUT_MAXLEN}) — one "
    "authoritative limit"
)
VAL_N             = _env("SP_VAL_N", 16, int)
TEST_FREQ         = _env("SP_TEST_FREQ", 10, int)
SAVE_FREQ         = _env("SP_SAVE_FREQ", 1, int)
# From-scratch default: validate BEFORE training so the base model's IMO-ProofBench number is
# this run's own step-0 baseline. The attach script's validation gate overrides it to False on
# every later attach.
VAL_BEFORE_TRAIN  = _env("SP_VAL_BEFORE_TRAIN", "True")
VAL_ONLY          = _env("SP_VAL_ONLY", "False")

print(f"[q] TIED weights + O_Q; seed={Q_SEED_DIR} bank={Q_BANK_DIR} g={Q_BUDGET_G} "
      f"thresh_global={Q_READY_THRESH_GLOBAL} thresh_problem={Q_READY_THRESH_PROBLEM} "
      f"fifo={Q_FIFO_CAP} train_n={Q_TRAIN_N} clip={Q_GRAD_CLIP} "
      f"ctx={Q_CTX_LIMIT} rng={Q_RNG_SEED} audit_cut={Q_AUDIT_CUT} audit_den={Q_AUDIT_DEN} "
      f"train_noref={Q_TRAIN_NOREF} prompt={Q_PROMPT_VARIANT} "
      f"ref_require_pass={Q_REF_REQUIRE_PASS}", flush=True)
print(f"[q] order=PPO(M1)->Q->PPO(M2) after_mb={Q_INTERLEAVE_AFTER}; ladder lr0="
      f"{Q_LR_INITIAL:.4g} floor={Q_LR_FLOOR:.4g} ratio_max={Q_LR_RATIO_MAX} "
      f"patience={Q_LR_PATIENCE} factor={Q_LR_FACTOR}", flush=True)
print(f"[replay] seed_dir={REPLAY_SEED_DIR} n={REPLAY_N}/{TRAIN_BATCH} cap={REPLAY_CAPACITY} "
      f"cut=[{REPLAY_CUT_LOW},{REPLAY_CUT_HIGH}] grain={REPLAY_CUT_GRAIN} "
      f"rng={REPLAY_RNG_SEED}", flush=True)
print(f"[replay-policy] bucketing={REPLAY_BUCKETING} admission={REPLAY_ADMISSION} "
      f"rotation={REPLAY_ROTATION} bound={REPLAY_BOUND} stale={REPLAY_STALE} "
      f"inflow_only={INFLOW_ONLY}", flush=True)
if INFLOW_ONLY:
    assert REPLAY_N % PPO_MINI == 0, (
        f"SP_SCRATCH_INFLOW_ONLY: ppo_mini ({PPO_MINI}) must divide the trained batch "
        f"(= SP_REPLAY_N = {REPLAY_N}); e.g. 96 trained rows -> mini 48"
    )
    # The interleaved order needs a second minibatch to exist: with one minibatch the Q step lands
    # after the whole PPO update (the worker reports `late`) and delta_PPO2 is 0.
    assert REPLAY_N // PPO_MINI >= Q_INTERLEAVE_AFTER + 1, (
        f"SP_Q_INTERLEAVE_AFTER={Q_INTERLEAVE_AFTER} needs at least "
        f"{Q_INTERLEAVE_AFTER + 1} PPO minibatches, but the trained batch {REPLAY_N} / mini "
        f"{PPO_MINI} = {REPLAY_N // PPO_MINI}. The interleaved order is PPO(M1) -> Q -> PPO(M2)."
    )


def _latest_ckpt_step() -> int:
    f = RUN_DATA / "checkpoints" / "latest_checkpointed_iteration.txt"
    try:
        return int(f.read_text().strip())
    except (OSError, ValueError):
        return 0


# ---- from-scratch gate ------------------------------------------------------------------
# A from-scratch run starts from the BASE model. A grafted checkpoint would look like an
# ordinary resume and silently turn it into a branch run, so the only two legal states are
# "no checkpoints at all" (first attach) and "checkpoints this run wrote itself" (any
# relaunch). A checkpoint tree that appeared without a metrics history is the graft case, and
# it is refused unless SP_ALLOW_GRAFT=1 (which a deliberate branch run sets).
_resume_step = _latest_ckpt_step()
_have_history = METRICS_PATH.exists() and METRICS_PATH.stat().st_size > 0
if _resume_step > 0 and not _have_history:
    if not _truthy("SP_ALLOW_GRAFT"):
        raise RuntimeError(
            f"run_data/checkpoints has step {_resume_step} but {METRICS_PATH.name} is empty: "
            "this looks like a GRAFTED checkpoint, and this run trains from scratch "
            "— the base model with a fresh optimizer and a fresh clock. If you really are "
            "recovering a checkpoint tree whose metrics file was lost, set SP_ALLOW_GRAFT=1 "
            "and say so in the run log."
        )
    print(f"[resume] SP_ALLOW_GRAFT=1: adopting step {_resume_step} without a metrics history",
          flush=True)
print(f"[resume] latest ckpt step = {_resume_step} ({'fresh start' if _resume_step == 0 else 'resume'}); "
      f"target = {TOTAL_STEPS}; train_batch={TRAIN_BATCH} ppo_mini={PPO_MINI} lr={LR:.6g} "
      f"exp_name={EXP_NAME}", flush=True)

ACTOR_MEM_UTIL    = _env("SP_ACTOR_GPU_MEM_UTIL", 0.6, float)
ACTOR_MODEL       = _env("SP_ACTOR_MODEL", "Qwen/Qwen3-4B-Thinking-2507")


# ---- judge: DeepSeek-V4-Flash at a pinned revision ----------------------------------------
# The judge IS the reward function, so it is held fixed at one model and revision; that is
# what keeps rewards comparable across relaunches and across runs graded by the same judge.
_JUDGE_REPO = os.environ.get("SP_JUDGE_HF_REPO", "deepseek-ai/DeepSeek-V4-Flash")
# Pinning the REVISION rather than following the cache's mutable refs/main guarantees the same
# judge weights even if someone re-downloads the repo. "" = follow refs/main.
_JUDGE_REVISION = os.environ.get(
    "SP_JUDGE_REVISION", "60d8d70770c6776ff598c94bb586a859a38244f1"
)


def _resolve_judge_model() -> str:
    """Resolve the judge to a concrete local snapshot dir.

    The run is HF_HUB_OFFLINE=1, so a repo id that is not in the hub cache does not download
    — vLLM either dies deep in engine init or, worse, resolves some other cached revision. Pin
    the snapshot here, at launch, where the error is legible.

    Resolution order: `SP_JUDGE_MODEL` (an explicit path) wins; else the pinned
    `SP_JUDGE_REVISION` under the repo's cache dir; else the cache's mutable `refs/main`."""
    explicit = os.environ.get("SP_JUDGE_MODEL")
    if explicit:
        return explicit
    hub = os.environ.get("HF_HUB_CACHE") or _SITE.get("hf_hub_cache", "")
    repo_dir = Path(hub) / ("models--" + _JUDGE_REPO.replace("/", "--"))
    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"judge {_JUDGE_REPO} is not in the hub cache ({repo_dir}). Fetch it on a node with "
            f"network first, e.g.\n"
            f"  HF_HUB_ENABLE_HF_TRANSFER=1 hf download {_JUDGE_REPO} --local-dir-use-symlinks False\n"
            f"then re-attach. Or pass SP_JUDGE_MODEL=<snapshot dir> explicitly."
        )
    if _JUDGE_REVISION:
        snap = repo_dir / "snapshots" / _JUDGE_REVISION
        if not snap.is_dir():
            _have = sorted(p.name for p in (repo_dir / "snapshots").glob("*") if p.is_dir())
            raise FileNotFoundError(
                f"pinned judge revision {_JUDGE_REVISION[:12]} is not in the cache under "
                f"{repo_dir}/snapshots (present: {_have}). This is the revision all reported runs "
                "were graded with, so silently using another one would move the reward scale. "
                "Fetch it (`hf download " + _JUDGE_REPO + " --revision " + _JUDGE_REVISION + "`), "
                "or set SP_JUDGE_REVISION= (empty) to follow refs/main and accept that the "
                "reward is no longer comparable to the reported runs."
            )
        print(f"[judge] using PINNED revision {_JUDGE_REVISION[:12]} (comparable to the reported runs)",
              flush=True)
    else:
        ref = repo_dir / "refs" / "main"
        print("[judge] WARNING: SP_JUDGE_REVISION is empty -> following the cache's MUTABLE "
              "refs/main; reward comparability with the reported runs is not guaranteed", flush=True)
        if ref.is_file():
            snap = repo_dir / "snapshots" / ref.read_text().strip()
            if not snap.is_dir():
                raise FileNotFoundError(
                    f"{ref} points at {snap.name} but {snap} does not exist: the cache entry is "
                    "incomplete (interrupted download?). Re-fetch the judge."
                )
        else:
            snaps = sorted(p for p in (repo_dir / "snapshots").glob("*") if p.is_dir())
            if len(snaps) != 1:
                raise FileNotFoundError(
                    f"{repo_dir} has no refs/main and {len(snaps)} snapshots; cannot pick one "
                    "unambiguously. Pass SP_JUDGE_MODEL=<snapshot dir>."
                )
            snap = snaps[0]
    cfg_path = snap / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"judge snapshot {snap} has no config.json (incomplete download)")
    with open(cfg_path, encoding="utf-8") as f:
        jcfg = json.load(f)
    print(f"[judge] {_JUDGE_REPO} -> {snap}\n"
          f"[judge] model_type={jcfg.get('model_type')} arch={jcfg.get('architectures')} "
          f"layers={jcfg.get('num_hidden_layers')} experts={jcfg.get('n_routed_experts')} "
          f"quant={(jcfg.get('quantization_config') or {}).get('quant_method')}", flush=True)
    return str(snap)


def _pin_judge_snapshot(resolved: str) -> str:
    """Record the resolved judge snapshot on first launch and refuse a silent change later.

    `refs/main` in the hub cache is MUTABLE: re-downloading the repo mid-run repoints it, and
    since the judge IS the reward function, that would change the reward halfway through with
    nothing in the metrics saying so. Pin it on the first launch and compare thereafter.
    Override with SP_ALLOW_JUDGE_DRIFT=1 (and write down why)."""
    pin_path = RUN_DATA / "judge_snapshot.json"
    payload = {"repo": _JUDGE_REPO, "revision": _JUDGE_REVISION, "resolved": resolved}
    if pin_path.exists():
        try:
            was = json.loads(pin_path.read_text())
        except (OSError, ValueError):
            was = None
        if was and was.get("resolved") != resolved:
            if not _truthy("SP_ALLOW_JUDGE_DRIFT"):
                raise ValueError(
                    "JUDGE DRIFT: this run was launched against\n"
                    f"    {was.get('resolved')}\n"
                    f"but the cache now resolves {_JUDGE_REPO} to\n    {resolved}\n"
                    "The judge is the reward function, so continuing would change the reward "
                    "mid-run and make the metrics before and after incomparable. Restore the "
                    "pinned snapshot, pass SP_JUDGE_MODEL=<that path>, or set "
                    "SP_ALLOW_JUDGE_DRIFT=1 if you intend the change and will note the step."
                )
            print(f"[judge] WARNING: SP_ALLOW_JUDGE_DRIFT=1 -- reward model changed from "
                  f"{was.get('resolved')} to {resolved} at this relaunch", flush=True)
        else:
            print(f"[judge] snapshot pin OK (unchanged since first launch)", flush=True)
    else:
        RUN_DATA.mkdir(parents=True, exist_ok=True)
        pin_path.write_text(json.dumps(payload, indent=2))
        print(f"[judge] snapshot PINNED for this run -> {pin_path}", flush=True)
    return resolved


JUDGE_MODEL       = _pin_judge_snapshot(_resolve_judge_model())
JUDGE_TP          = _env("SP_JUDGE_TP", 8, int)
JUDGE_MAXLEN      = _env("SP_JUDGE_MAXLEN", 98304, int)
JUDGE_RESP_LEN    = _env("SP_JUDGE_RESP_LEN", 40960, int)
JUDGE_MAX_TOKENS  = _env("SP_JUDGE_MAX_TOKENS", 40000, int)
JUDGE_REASONING   = _env("SP_JUDGE_REASONING", "high")
JUDGE_MEM_UTIL    = _env("SP_JUDGE_GPU_MEM_UTIL", 0.80, float)
JUDGE_ENFORCE_EAGER = _env("SP_JUDGE_ENFORCE_EAGER", "False")
JUDGE_STANDALONE  = _truthy("SP_JUDGE_STANDALONE")
JUDGE_CUDAGRAPH_MODE = _env("SP_JUDGE_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
PASS_POINTS_MIN   = _env("SP_PASS_POINTS_MIN", 6, int)
PPO_MAX_TOKEN_LEN = _env("SP_PPO_MAX_TOKEN_LEN", 51200, int)
ENTROPY_COEFF     = _env("SP_ENTROPY_COEFF", 0.0, float)
LOG_PROB_MAX_TOKEN_LEN = _env("SP_LOG_PROB_MAX_TOKEN_LEN", 3 * PPO_MAX_TOKEN_LEN, int)
USE_FUSED_KERNELS = _env("SP_USE_FUSED_KERNELS", "True")
ROLLOUT_TP        = _env("SP_ROLLOUT_TP", 1, int)   # the attach scripts PIN 4 (throughput); 1 here so a
                                                   # bypassed attach cannot skip the AOT preflight
ACTOR_MAX_NUM_SEQS = _env("SP_ACTOR_MAX_NUM_SEQS", _SITE.get("rollout_max_num_seqs", 96), int)
MAX_NUM_BATCHED_TOKENS = _env("SP_MAX_NUM_BATCHED_TOKENS", 32768, int)
REWARD_MAX_NUM_SEQS = _env("SP_REWARD_MAX_NUM_SEQS", 256, int)
REWARD_MAX_NUM_BATCHED_TOKENS = _env("SP_REWARD_MAX_NUM_BATCHED_TOKENS", 16384, int)
TEMPERATURE       = _env("SP_TEMPERATURE", 0.8, float)
VAL_TOP_P         = _env("SP_VAL_TOP_P", 0.95, float)
VAL_TOP_K         = _env("SP_VAL_TOP_K", 20, int)
MAX_CKPT_KEEP     = _env("SP_MAX_CKPT_KEEP", 5, int)
REWARD_NUM_WORKERS = _env("SP_REWARD_NUM_WORKERS", 8, int)
Q_MAX_TOKEN_LEN   = _env("SP_Q_MAX_TOKEN_LEN", Q_CTX_LIMIT + 64, int)

OVERRIDES = [
    # ---- GRPO, no KL (unchanged) ----
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "algorithm.norm_adv_by_std_in_grpo=False",
    # ---- reward = async DS4 fine-grained judge ----
    f"reward.custom_reward_function.path={REWARD_FILE}",
    "reward.custom_reward_function.name=compute_score",
    f"+reward.custom_reward_function.reward_kwargs.judge_max_tokens={JUDGE_MAX_TOKENS}",
    f"+reward.custom_reward_function.reward_kwargs.judge_reasoning_effort={JUDGE_REASONING}",
    "+reward.custom_reward_function.reward_kwargs.judge_temperature=1.0",
    "+reward.custom_reward_function.reward_kwargs.judge_top_p=1.0",
    "+reward.custom_reward_function.reward_kwargs.judge_top_k=-1",
    "+reward.custom_reward_function.reward_kwargs.judge_seed=42",
    "+reward.custom_reward_function.reward_kwargs.judge_payload_style=deepseek_v4",
    f"+reward.custom_reward_function.reward_kwargs.pass_points_min={PASS_POINTS_MIN}",
    f"+reward.custom_reward_function.reward_kwargs.finegrained_template_path={FINEGRAINED_TEMPLATE}",
    f"+reward.custom_reward_function.reward_kwargs.val_judge_template_path={VAL_JUDGE_TEMPLATE}",
    f"+reward.custom_reward_function.reward_kwargs.val_map_path={VAL_MAP_PATH}",
    "reward.reward_manager.name=naive",
    f"reward.num_workers={REWARD_NUM_WORKERS}",
    # ---- colocated judge: DeepSeek-V4-Flash @ the pinned revision ----
    "reward.reward_model.enable=True",
    "reward.reward_model.enable_resource_pool={}".format(JUDGE_STANDALONE),
    f"reward.reward_model.model_path={JUDGE_MODEL}",
    "reward.reward_model.rollout.name=vllm",
    f"reward.reward_model.rollout.tensor_model_parallel_size={JUDGE_TP}",
    f"reward.reward_model.rollout.gpu_memory_utilization={JUDGE_MEM_UTIL}",
    f"reward.reward_model.rollout.max_num_seqs={REWARD_MAX_NUM_SEQS}",
    f"reward.reward_model.rollout.max_num_batched_tokens={REWARD_MAX_NUM_BATCHED_TOKENS}",
    f"reward.reward_model.rollout.max_model_len={JUDGE_MAXLEN}",
    f"reward.reward_model.rollout.response_length={JUDGE_RESP_LEN}",
    "reward.reward_model.rollout.free_cache_engine=True",
    f"reward.reward_model.rollout.enforce_eager={JUDGE_ENFORCE_EAGER}",
    "reward.reward_model.rollout.enable_chunked_prefill=True",
    "reward.reward_model.rollout.enable_prefix_caching=True",
    "+reward.reward_model.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4",
    "+reward.reward_model.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4",
    "+reward.reward_model.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8",
    "+reward.reward_model.rollout.engine_kwargs.vllm.block_size=256",
    "+reward.reward_model.rollout.engine_kwargs.vllm.moe_backend=marlin",
    "+reward.reward_model.rollout.engine_kwargs.vllm.async_scheduling=True",
    # ---- data: the Q-routing replay dataset ----
    f"data.train_files={DATA_DIR}/train.parquet",
    f"data.val_files={DATA_DIR}/test.parquet",
    f"data.train_batch_size={TRAIN_BATCH}",
    f"data.max_prompt_length={MAX_PROMPT_LEN}",
    f"data.max_response_length={MAX_RESPONSE_LEN}",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    f"data.custom_cls.path={EXP_DIR / 'q_dataset.py'}",
    "data.custom_cls.name=SPQReadinessDataset",
    "data.shuffle=False",
    "data.dataloader_num_workers=0",
    f"+data.sp_replay_seed_dir={REPLAY_SEED_DIR}",
    f"+data.sp_replay_delta_dir={RUN_DATA}",
    f"+data.sp_replay_n={REPLAY_N}",
    f"+data.sp_replay_capacity={REPLAY_CAPACITY}",
    f"+data.sp_replay_cut_low={REPLAY_CUT_LOW}",
    f"+data.sp_replay_cut_high={REPLAY_CUT_HIGH}",
    f"+data.sp_replay_cut_grain={REPLAY_CUT_GRAIN}",
    f"+data.sp_replay_ema_coef={REPLAY_EMA_COEF}",
    f"+data.sp_replay_rng_seed={REPLAY_RNG_SEED}",
    f"+data.sp_replay_bucketing={REPLAY_BUCKETING}",
    f"+data.sp_replay_admission={REPLAY_ADMISSION}",
    f"+data.sp_replay_rotation={REPLAY_ROTATION}",
    f"+data.sp_replay_bound={REPLAY_BOUND}",
    f"+data.sp_replay_global_sampling={REPLAY_GLOBAL_SAMPLING}",
    f"+data.sp_replay_stale_steps={REPLAY_STALE}",
    # cold start: tolerate the empty buffer instead of raising, and stand trained cold_scratch
    # rows in for the missing replay rows until the inflow lane has admitted something.
    f"+data.sp_replay_cold_bootstrap={int(REPLAY_COLD)}",
    # ---- generative-Q knobs ----
    f"+data.sp_q_seed_dir={Q_SEED_DIR}",
    f"+data.sp_q_bank_dir={Q_BANK_DIR}",
    f"+data.sp_q_budget_g={Q_BUDGET_G}",
    f"+data.sp_q_ready_thresh={Q_READY_THRESH}",
    f"+data.sp_q_ready_thresh_global={Q_READY_THRESH_GLOBAL}",
    f"+data.sp_q_ready_thresh_problem={Q_READY_THRESH_PROBLEM}",
    f"+data.sp_q_fifo_cap={Q_FIFO_CAP}",
    f"+data.sp_q_min_valid={Q_MIN_VALID}",
    f"+data.sp_q_train_n={Q_TRAIN_N}",
    f"+data.sp_q_ctx_limit={Q_CTX_LIMIT}",
    f"+data.sp_q_rng_seed={Q_RNG_SEED}",
    f"+data.sp_q_audit_cut={Q_AUDIT_CUT}",
    f"+data.sp_q_audit_den={Q_AUDIT_DEN}",
    f"+data.sp_q_train_noref={Q_TRAIN_NOREF}",
    f"+data.sp_q_prompt_variant={Q_PROMPT_VARIANT}",
    f"+data.sp_q_ref_require_pass={Q_REF_REQUIRE_PASS}",
    # Mirror the ENV-gated critic-LR knobs into the config for PROVENANCE ONLY -- nothing reads
    # them from here. Without this they appear in no manifest, and the dashboard cannot tell which
    # critic size-control regime (LR ladder or movement cap) the run used and draws the wrong
    # reference line. Keep them in sync with the ENVS block in run_attach.sh.
    f"+data.sp_q_lr_ladder={1 if _truthy('SP_Q_LR_LADDER') else 0}",
    f"+data.sp_q_lr_initial={Q_LR_INITIAL}",
    f"+data.sp_q_lr_floor={Q_LR_FLOOR}",
    f"+data.sp_q_lr_ratio_max={Q_LR_RATIO_MAX}",
    f"+data.sp_q_interleave={1 if _truthy('SP_Q_INTERLEAVE') else 0}",
    # ---- model / actor: BASE weights, fresh optimizer ----
    f"actor_rollout_ref.model.path={ACTOR_MODEL}",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    f"actor_rollout_ref.actor.optim.lr={LR}",
    "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
    "actor_rollout_ref.actor.optim.weight_decay=0.01",
    f"actor_rollout_ref.actor.ppo_mini_batch_size={PPO_MINI}",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.kl_loss_coef=0.0",
    f"actor_rollout_ref.actor.entropy_coeff={ENTROPY_COEFF}",
    "actor_rollout_ref.actor.grad_clip=0.3",
    "actor_rollout_ref.actor.clip_ratio=0.2",
    "actor_rollout_ref.actor.clip_ratio_low=0.2",
    "actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.actor.clip_ratio_c=3.0",
    "actor_rollout_ref.actor.use_dynamic_bsz=True",
    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={PPO_MAX_TOKEN_LEN}",
    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={LOG_PROB_MAX_TOKEN_LEN}",
    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={LOG_PROB_MAX_TOKEN_LEN}",
    f"actor_rollout_ref.model.use_fused_kernels={USE_FUSED_KERNELS}",
    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    # ---- rollout + the Q agent loops ----
    "actor_rollout_ref.rollout.name=vllm",
    f"actor_rollout_ref.rollout.n={ROLLOUT_N}",
    f"actor_rollout_ref.rollout.max_model_len={ROLLOUT_MAXLEN}",
    f"actor_rollout_ref.rollout.temperature={TEMPERATURE}",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.top_k=-1",
    f"actor_rollout_ref.rollout.tensor_model_parallel_size={ROLLOUT_TP}",
    f"actor_rollout_ref.rollout.max_num_seqs={ACTOR_MAX_NUM_SEQS}",
    f"actor_rollout_ref.rollout.max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS}",
    f"actor_rollout_ref.rollout.gpu_memory_utilization={ACTOR_MEM_UTIL}",
    "actor_rollout_ref.rollout.enable_chunked_prefill=True",
    "actor_rollout_ref.rollout.calculate_log_probs=True",
    f"actor_rollout_ref.rollout.agent.agent_loop_config_path={EXP_DIR / 'sp_agent_loops.yaml'}",
    f"actor_rollout_ref.rollout.val_kwargs.n={VAL_N}",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    f"actor_rollout_ref.rollout.val_kwargs.temperature={TEMPERATURE}",
    f"actor_rollout_ref.rollout.val_kwargs.top_p={VAL_TOP_P}",
    f"actor_rollout_ref.rollout.val_kwargs.top_k={VAL_TOP_K}",
    # ---- trainer / logging ----
    "trainer.logger=[console,wandb,file]",
    "trainer.project_name=self-play",
    f"trainer.experiment_name={EXP_NAME}",
    f"trainer.default_local_dir={RUN_DATA / 'checkpoints'}",
    f"trainer.rollout_data_dir={RUN_DATA / 'rollouts'}",
    f"trainer.validation_data_dir={RUN_DATA / 'val_rollouts'}",
    "trainer.log_val_generations=16",
    f"trainer.n_gpus_per_node={N_GPUS_PER_NODE}",
    f"trainer.nnodes={NNODES}",
    f"trainer.val_before_train={VAL_BEFORE_TRAIN}",
    f"trainer.val_only={VAL_ONLY}",
    f"trainer.test_freq={TEST_FREQ}",
    f"trainer.save_freq={SAVE_FREQ}",
    f"trainer.max_actor_ckpt_to_keep={MAX_CKPT_KEEP}",
    "trainer.total_epochs=9999",
    f"trainer.total_training_steps={TOTAL_STEPS}",
    "trainer.resume_mode=auto",
]
if JUDGE_STANDALONE:
    OVERRIDES += [
        "reward.reward_model.nnodes=1",
        f"reward.reward_model.n_gpus_per_node={JUDGE_TP}",
    ]

_JUDGE_COMPILATION_CONFIG = json.dumps({
    "mode": 0,
    "cudagraph_mode": JUDGE_CUDAGRAPH_MODE,
    "cudagraph_capture_sizes": [
        1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
        120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224,
        232, 240, 248, 256,
    ],
})


def _engine_cache_env():
    """Cache/telemetry env for all Ray actors."""
    xdg = os.environ.get("XDG_CACHE_HOME", _SITE.get("cache_root", os.path.expandvars("${AC2_SCRATCH}/.cache")))
    d = {
        "XDG_CACHE_HOME": xdg,
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", f"{xdg}/triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"{xdg}/inductor"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{xdg}/vllm"),
        "TORCH_HOME": os.environ.get("TORCH_HOME", f"{xdg}/torch"),
        "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH", f"{xdg}/nv"),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR", f"{xdg}/torch_extensions"),
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", f"{xdg}/matplotlib"),
        "WANDB_CACHE_DIR": os.environ.get("WANDB_CACHE_DIR", f"{xdg}/wandb"),
        "WANDB_DATA_DIR": os.environ.get("WANDB_DATA_DIR", f"{xdg}/wandb-data"),
        "WANDB_MODE": os.environ.get("WANDB_MODE", _SITE.get("wandb_mode", "offline")),
        "FLASHINFER_WORKSPACE_BASE": os.environ.get(
            "FLASHINFER_WORKSPACE_BASE", os.path.dirname(xdg)
        ),
        "VLLM_DISABLE_COMPILE_CACHE": os.environ.get("VLLM_DISABLE_COMPILE_CACHE", "1"),
        "VLLM_DEEP_GEMM_WARMUP": os.environ.get("VLLM_DEEP_GEMM_WARMUP", "skip"),
        "VLLM_ENGINE_READY_TIMEOUT_S": os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "3600"),
        "VLLM_RPC_TIMEOUT": os.environ.get("VLLM_RPC_TIMEOUT", "600000"),
        "TILELANG_CLEANUP_TEMP_FILES": os.environ.get("TILELANG_CLEANUP_TEMP_FILES", "1"),
        "TILELANG_CACHE_DIR": os.environ.get("TILELANG_CACHE_DIR", f"{xdg}/tilelang"),
        "TILELANG_TMP_DIR": os.environ.get("TILELANG_TMP_DIR", f"{xdg}/tilelang/tmp"),
        "TRTLLM_DG_CACHE_DIR": os.environ.get("TRTLLM_DG_CACHE_DIR", f"{xdg}/trtllm_dg"),
        "SP_DG_JIT_CACHE_BASE": os.environ.get("SP_DG_JIT_CACHE_BASE", f"{xdg}/dg_jit_nodes"),
        "DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
    }
    return d


def build_config():
    with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=OVERRIDES)
    main_ppo.auto_set_device(cfg)
    cfg = main_ppo.migrate_legacy_reward_impl(cfg)

    hf_hub_cache = os.environ.get("HF_HUB_CACHE", _SITE.get("hf_hub_cache", ""))
    hf_home = os.environ.get("HF_HOME", _SITE.get("hf_home", os.path.expanduser("~/.cache/huggingface")))
    env_vars = {
        "VERL_FILE_LOGGER_PATH": str(METRICS_PATH),
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_RUN_ID": EXP_NAME,
        "WANDB_RESUME": os.environ.get("WANDB_RESUME", "allow"),
        "HF_HOME": hf_home,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        **_engine_cache_env(),
    }
    if os.environ.get("WANDB_MODE", "online") != "offline":
        try:
            import netrc as _netrc_mod
            _wb = _netrc_mod.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
            if _wb and _wb[2]:
                env_vars["WANDB_API_KEY"] = _wb[2]
        except Exception as _e:
            print(f"[wandb] could not read ~/.netrc api.wandb.ai key ({_e}); relying on actor env")
    if hf_hub_cache:
        env_vars["HF_HUB_CACHE"] = hf_hub_cache
    _step_cache = os.environ.get("VERL_STEP_CACHE_DIR")
    if _step_cache:
        env_vars["VERL_STEP_CACHE_DIR"] = _step_cache
    for _k in (
        "SP_JUDGE_HTTP_TOTAL_TIMEOUT", "SP_JUDGE_HTTP_CONN_LIMIT", "SP_JUDGE_MAX_INFLIGHT",
        "SP_ROUTER_HTTP_TOTAL_TIMEOUT",
        "VERL_STAGGER_ENGINE_INIT",
        "SP_PASS_POINTS_MIN",
        "SP_Q_READY_REQUIRE_BANK",
        "SP_ADAPTIVE_ENTROPY", "SP_AEC_TARGET_H", "SP_AEC_DELTA",
        "SP_AEC_KMAX", "SP_AEC_KMIN", "SP_AEC_KINIT",
        "SP_DIFF_SAMPLING",
        "SP_KEEP_BEST_CKPT",
        # Retention stride (fork default 10 = keep every 10th step's weights forever).
        # Only the OFF-switch needs forwarding: unset means the actor falls back to the
        # fork default, which is the intended behaviour. Forwarded so SP_CKPT_KEEP_EVERY=0
        # actually reaches the driver -- the forward loop carries "0" (a non-empty string).
        "SP_CKPT_KEEP_EVERY",
        "SP_REPLAY_ENABLE",
        # generative-Q gates + worker-read knobs (Ray actors only see runtime_env)
        "SP_Q_ENABLE", "SP_Q_GRAD_CLIP", "SP_Q_MAX_TOKEN_LEN",
        # Interleaved critic step + LR ladder. EVERY one of these is read inside the
        # TaskRunner Ray actor or the actor workers, never on the login shell: a
        # missing forward means the flag silently no-ops and the run falls back to
        # the capture-and-cap critic update at a constant LR.
        "SP_Q_INTERLEAVE", "SP_Q_INTERLEAVE_AFTER",
        "SP_Q_LR_LADDER", "SP_Q_LR_INITIAL", "SP_Q_LR_FLOOR", "SP_Q_LR_RATIO_MAX",
        "SP_Q_LR_BREACH_PATIENCE", "SP_Q_LR_REDUCTION_FACTOR", "SP_Q_LADDER_START_STEP",
        # separate-Q knobs are deliberately NOT forwarded: this runner asserts SP_Q_SEPARATE off.
        "SP_SCRATCH_INFLOW_ONLY", "SP_Q_REQUIRE_NONZERO", "SP_REPLAY_COLD_BOOTSTRAP",
        "SP_Q_RESET_STEPS", "SP_REPLAY_RESEED_STEPS", "SP_REPLAY_POLICY_OVERRIDE",
        "SP_REPLAY_GLOBAL_SAMPLING",
        "SP_ROLLOUT_BACKFILL", "SP_ROLLOUT_BACKFILL_DIM",
        "SP_ROLLOUT_PRIORITY",
        "SP_Q_DUMP_WAVE",
    ):
        _v = os.environ.get(_k)
        if _v:
            env_vars[_k] = _v
    # Pin the OFF switches unconditionally. The `if _v` loop above cannot carry a "0" through
    # (falsy), so relying on it would leave the Ray actors reading whatever the node inherited.
    env_vars["SP_LENPEN_ENABLE"] = "0"
    env_vars["SP_DIFF_SAMPLING"] = "0"
    env_vars["SP_Q_SEPARATE"] = "0"
    # pin the Q-phase knobs even when not explicitly exported
    env_vars.setdefault("SP_Q_MAX_TOKEN_LEN", str(Q_MAX_TOKEN_LEN))
    env_vars.setdefault("SP_Q_GRAD_CLIP", str(Q_GRAD_CLIP))
    env_vars.setdefault("SP_Q_INTERLEAVE_AFTER", str(Q_INTERLEAVE_AFTER))
    env_vars.setdefault("SP_Q_LR_INITIAL", str(Q_LR_INITIAL))
    env_vars.setdefault("SP_Q_LR_FLOOR", str(Q_LR_FLOOR))
    env_vars.setdefault("SP_Q_LR_RATIO_MAX", str(Q_LR_RATIO_MAX))
    env_vars.setdefault("SP_Q_LR_BREACH_PATIENCE", str(Q_LR_PATIENCE))
    env_vars.setdefault("SP_Q_LR_REDUCTION_FACTOR", str(Q_LR_FACTOR))
    for _k in (
        "CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT", "CUDACXX",
        "TRTLLM_DG_NVCC_COMPILER", "CPATH", "NVCC_PREPEND_FLAGS",
    ):
        _v = os.environ.get(_k)
        if _v:
            env_vars[_k] = _v
    with open_dict(cfg):
        cfg.ray_kwargs.ray_init.runtime_env = {"env_vars": env_vars}
        cfg.reward.reward_model.rollout.engine_kwargs.vllm.compilation_config = _JUDGE_COMPILATION_CONFIG
        if _truthy("SP_RAY_LOCAL"):
            cfg.ray_kwargs.ray_init.address = "local"
            print("[ray] SP_RAY_LOCAL=1 -> ray.init(address='local')", flush=True)
    return cfg


def main():
    cfg = build_config()
    # EVERY experiment-local file that shapes a step, not just the runner + sbatch. q_dataset.py
    # builds the batch (including the cold_scratch fill), sp_agent_loops.yaml selects the agent
    # loops, run_attach.sh pins ~60 env vars that the config does not capture, and
    # build_cold_artifacts.py defines the cold start. A manifest without them cannot reproduce a
    # step.
    _extra = [
        Path(__file__).resolve(),
        EXP_DIR / "submit_4node.sbatch",
        EXP_DIR / "q_dataset.py",
        EXP_DIR / "sp_agent_loops.yaml",
        EXP_DIR / "run_attach.sh",
        EXP_DIR / "build_cold_artifacts.py",
        EXP_DIR / "setup.sh",
    ]
    _extra = [p for p in _extra if p.exists()]
    manifest_dump(MANIFEST_DIR, cfg, src_dir=REPO_ROOT / "src", extra_files=_extra)
    # manifest_dump OVERWRITES config.yaml / git_commit.txt / code.zip, so after a relaunch the
    # canonical manifest describes the LATEST launch -- not necessarily the code that produced
    # earlier steps. Keep the canonical copy (tools read it) AND archive a per-launch snapshot
    # keyed by the step being resumed from, so provenance for step N survives later relaunches.
    _snap = MANIFEST_DIR / "launches" / f"resume_from_{_resume_step:06d}_attempt"
    _n = 1
    while (_snap.parent / f"{_snap.name}{_n}").exists():
        _n += 1
    _snap_dir = _snap.parent / f"{_snap.name}{_n}"
    try:
        _snap_dir.mkdir(parents=True, exist_ok=True)
        for _f in ("config.yaml", "git_commit.txt", "uv_freeze.txt", "code.zip"):
            _src = MANIFEST_DIR / _f
            if _src.exists():
                shutil.copy2(_src, _snap_dir / _f)
        print(f"[manifest] per-launch snapshot -> {_snap_dir}", flush=True)
    except OSError as _e:
        print(f"[manifest] WARNING: could not archive the per-launch snapshot ({_e}); the "
              f"canonical manifest/ still describes THIS launch only", flush=True)
    RUN_DATA.mkdir(parents=True, exist_ok=True)
    for _cache_dir in _engine_cache_env().values():
        if isinstance(_cache_dir, str) and _cache_dir.startswith("/"):
            Path(_cache_dir).mkdir(parents=True, exist_ok=True)
    main_ppo.run_ppo(cfg)


if __name__ == "__main__":
    main()
