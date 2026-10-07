"""Difficulty sampling -- driver-side state.

Reweight the training-problem sampler by a per-problem difficulty weight w, sample problems
proportional to w, and apply an importance-sampling correction c = u/q (∝ 1/w) to both the training
loss and the corrected metrics, so every tracked expectation matches uniform (weight-1) sampling.
Difficulty is a running EMA pass rate p-hat per problem, keyed by a STABLE problem id (qid = sha1 of
the problem statement) -- NOT `uid`, which is a fresh uuid per draw and never matches across steps.

Roles of this module (all driver-process; workers never import mutable state from here):
  * qid helpers (`qid_from_text` / `qid_from_messages`) -- a marker-extraction hash of the
    statement, shared with the offline pruning analysis so offline results key-match at runtime.
  * per-qid stats {p-hat, n_obs, demoted} + EMA update from each step's per-problem pass fraction.
  * weight forms (SP_DIFF_WEIGHT): `step` ({1, w_low} with hysteresis), `sqrt_p` (raw-reward
    variance optimum), `sqrt_pq` (advantage-variance optimum, recommended) -- all floored.
  * draw FIFO: the sampler records (row_idx, qid, c) AT DRAW TIME so the correction matches the
    weights actually used to draw, even when the dataloader prefetches ahead; the fit loop pops one
    batch per step and cross-checks row identity, falling back to current-state c on mismatch.
  * `normalize_c_for_loss`: rescale per-row c so the c-weighted global token count equals the raw
    token count -- then the worker can keep dividing by the unchanged `batch_num_tokens` and the
    result IS the c-weighted token-mean.
  * persistence next to checkpoints (difficulty_state.json), like aec_k.json.

Env-gated: SP_DIFF_SAMPLING=1 turns it on; everything is a no-op otherwise. The env must reach the
DRIVER via runtime_env.env_vars -- though unlike AEC, only the driver needs it
(workers just see a diff_c tensor in the batch).
"""
import glob
import hashlib
import json
import math
import os
import random
from collections import deque

_S = {
    "inited": False,
    "version": 0,              # bumped on every observe()/load_state(); samplers cache against it
    "enable": False,
    "weight_form": "sqrt_pq",  # step | sqrt_p | sqrt_pq
    "w_floor": 1.0 / 32.0,     # floor for the continuous forms (bounds c <= 32 after Sum(w)=N norm)
    "ema_alpha": 0.8,          # EMA weight on the LATEST obs (new 0.8 / old 0.2); tracks current difficulty
    "min_obs": 0,              # obs required before a problem's p̂ affects its weight (0 = use it
                               # immediately; the EMA-from-prior already smooths a single noisy read)
    "p_prior": 0.5,            # p-hat prior: every problem starts here; first obs EMAs FROM it
    "pass_thresh": 0.999,      # per-rollout score >= this counts as a pass (i.e. int(score) == 1)
    # step-form knobs
    "step_thresh": 15.0 / 16.0,   # demote when p-hat >= this ...
    "step_hyst_low": 13.0 / 16.0, # ... promote back only when p-hat <= this (hysteresis band)
    "step_w_low": 1.0 / 16.0,     # demoted weight
    # state
    "stats": {},               # qid -> {"p": float, "n": int, "demoted": bool}
    "fifo": deque(),           # draw-time records: (row_idx, qid, c) pushed by the sampler
}


def _init():
    if _S["inited"]:
        return
    _S["enable"] = os.environ.get("SP_DIFF_SAMPLING", "0") in ("1", "true", "True")
    _S["weight_form"] = os.environ.get("SP_DIFF_WEIGHT", "sqrt_pq")
    _S["w_floor"] = float(os.environ.get("SP_DIFF_W_FLOOR", str(1.0 / 32.0)))
    _S["ema_alpha"] = float(os.environ.get("SP_DIFF_EMA_ALPHA", "0.8"))
    _S["min_obs"] = int(os.environ.get("SP_DIFF_MIN_OBS", "0"))
    _S["p_prior"] = float(os.environ.get("SP_DIFF_P_PRIOR", "0.5"))
    _S["pass_thresh"] = float(os.environ.get("SP_DIFF_PASS_THRESH", "0.999"))
    _S["step_thresh"] = float(os.environ.get("SP_DIFF_THRESH", str(15.0 / 16.0)))
    _S["step_hyst_low"] = float(os.environ.get("SP_DIFF_HYST_LOW", str(13.0 / 16.0)))
    _S["step_w_low"] = float(os.environ.get("SP_DIFF_W_LOW", str(1.0 / 16.0)))
    assert _S["weight_form"] in ("step", "sqrt_p", "sqrt_pq", "safe"), _S["weight_form"]
    assert 0.0 < _S["w_floor"] <= 1.0, _S["w_floor"]
    _S["inited"] = True


def enabled():
    _init()
    return _S["enable"]


def pass_threshold():
    """Per-rollout score >= this counts as a pass for the p-hat update."""
    _init()
    return _S["pass_thresh"]


def row_pass(prover_judge_score=None, correctness_judge_score=None, score=None):
    """Binary pass (0/1) for ONE rollout, from the BINARY judge verdict — NOT the shaped reward.

    Difficulty is "can the policy solve this problem", so the pass signal must be the raw judge
    outcome, not `score`: with a correct-only length penalty (prover_judge.py) a correct-but-long
    proof has `prover_judge_score == 1` yet `score < 1`, and thresholding the shaped `score` would
    wrongly log it as a failure and over-weight the problem. Preference (matches the offline
    analysis and warm-start): the binary judge fields first, shaped `score >= pass_thresh` only as a last-resort
    fallback for runs/rows that carry neither."""
    _init()
    for v in (prover_judge_score, correctness_judge_score):
        if v is not None:
            try:
                return 1 if int(round(float(v))) >= 1 else 0
            except (TypeError, ValueError):
                pass
    if score is not None:
        try:
            return 1 if float(score) >= _S["pass_thresh"] else 0
        except (TypeError, ValueError):
            pass
    return 0


# ---------------------------------------------------------------------------
# qid: stable problem identity. MUST stay in sync with the offline analysis's qid().
# ---------------------------------------------------------------------------

def qid_from_text(text):
    """sha1 of the problem statement: the slice between the fixed prompt-template markers, falling
    back to the whole text. Identical across re-draws of the same question (uid is NOT)."""
    text = text or ""
    a = text.find("mathematical problem:")
    b = text.find("Solve the problem")
    core = text[a:b] if (a != -1 and b != -1 and b > a) else text
    return hashlib.sha1(core.encode("utf-8", "ignore")).hexdigest()


def qid_from_messages(messages):
    """qid from a chat-format prompt (the dataset's raw_prompt / prompt column): concatenate the
    string contents (user statement lives there) and hash via qid_from_text."""
    parts = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # multimodal segments
            parts.extend(seg.get("text", "") for seg in c if isinstance(seg, dict) and seg.get("type") == "text")
    return qid_from_text("\n".join(parts))


# ---------------------------------------------------------------------------
# weights + corrections
# ---------------------------------------------------------------------------

def _w_of_p(p, demoted=False):
    """Weight as a pure function of a pass rate (and the step-form hysteresis flag)."""
    p = min(max(p, 0.0), 1.0)
    form = _S["weight_form"]
    if form == "step":
        return _S["step_w_low"] if demoted else 1.0
    if form == "sqrt_p":
        return max(math.sqrt(p), _S["w_floor"])
    if form == "safe":
        # difficulty-safed: FULL weight (1/2) to problems the policy solves < 1/2 the time; for the
        # easier half (p >= 1/2) use the standard max{sqrt(p(1-p)), 1/32} (sqrt_pq floored at 1/32).
        # Continuous at p=1/2 (both branches give 1/2); unlike sqrt_pq it does NOT down-weight the
        # very-hard (p -> 0) problems -- they stay at the max weight 1/2.
        if p < 0.5:
            return 0.5
        return max(math.sqrt(p * (1.0 - p)), _S["w_floor"])
    return max(math.sqrt(p * (1.0 - p)), _S["w_floor"])  # sqrt_pq


def _w_max():
    """Weight of a maximally-uncertain problem = weight at the prior p-hat (1/2 by default).
    Kept as the sampler's "unseen weight" metric."""
    return _w_of_p(_S["p_prior"])


def weight(qid):
    """Current sampling weight for one problem. Every problem carries the p_prior (1/2) until it
    has min_obs observations: unseen and under-observed problems weigh exactly as p-hat = prior,
    so a FRESH state has all-equal weights => c == 1 (uniform PPO), with no special-case branch."""
    _init()
    st = _S["stats"].get(qid)
    if st is None or st["n"] < _S["min_obs"]:
        return _w_of_p(_S["p_prior"])
    return _w_of_p(st["p"], demoted=st["demoted"])


def observe(qid, pass_frac, count=1):
    """EMA-update one problem's p-hat from this step's pass fraction (passes/rollout_n),
    starting from the p_prior (1/2) on first sight. Also maintains the step-form hysteresis
    flag. `count` is kept at 1 per step-occurrence regardless of rollout_n."""
    _init()
    pass_frac = min(max(float(pass_frac), 0.0), 1.0)
    st = _S["stats"].get(qid)
    if st is None:
        # start every problem at the maximum-uncertainty prior and EMA from there: one 16/16
        # fluke lands at (1-a)*prior + a, not at 1.0 -- built-in smoothing on top of min_obs.
        st = {"p": _S["p_prior"], "n": 0, "demoted": False}
        _S["stats"][qid] = st
    a = _S["ema_alpha"]
    st["p"] = (1.0 - a) * st["p"] + a * pass_frac
    st["n"] += int(count)
    if st["n"] >= _S["min_obs"]:
        if not st["demoted"] and st["p"] >= _S["step_thresh"]:
            st["demoted"] = True
        elif st["demoted"] and st["p"] <= _S["step_hyst_low"]:
            st["demoted"] = False
    _S["version"] += 1


def corrections_for(qids, row_weights=None):
    """c_i = u_i/q_i for a set of DISTINCT problems currently in the dataset. With target uniform
    over the N distinct problems and q ∝ w, c_i = W/(N*w_i) = w_bar/w_i, so E_q[c] = 1 exactly.
    Returns (dict qid->c, dict qid->w). `qids` must be the distinct-problem universe (the sampler
    passes its full row->qid map's distinct keys), NOT a sampled batch."""
    _init()
    ws = {q: weight(q) for q in set(qids)}
    n = len(ws)
    if n == 0:
        return {}, {}
    w_bar = sum(ws.values()) / n
    return {q: w_bar / w for q, w in ws.items()}, ws


# ---------------------------------------------------------------------------
# draw FIFO: draw-time corrections, in draw order (== dataloader batch order)
# ---------------------------------------------------------------------------

def push_draw(row_idx, qid, c):
    _S["fifo"].append((int(row_idx), qid, float(c)))


def pop_draws(k):
    """Pop k draw records (oldest first). Returns fewer than k if the FIFO is short (e.g. right
    after a resume, when prefetched draws from the previous process are gone) -- caller must
    fall back to current-state corrections in that case."""
    out = []
    while _S["fifo"] and len(out) < k:
        out.append(_S["fifo"].popleft())
    return out


def fifo_len():
    return len(_S["fifo"])


def clear_fifo():
    """Drop all pending draw records -- the fit loop's self-heal after a resume/replay mismatch."""
    _S["fifo"].clear()


# ---------------------------------------------------------------------------
# loss-side normalization: weighted token-mean with an unchanged denominator
# ---------------------------------------------------------------------------

def normalize_c_for_loss(cs, n_tokens):
    """Given per-row corrections c_i and per-row response token counts n_i (over the FULL global
    batch, driver-side), return c-hat_i = c_i * (Σ n) / (Σ c n). Then
        Σ_i c-hat_i Σ_t ℓ_it / Σ_i n_i  ==  Σ_i c_i Σ_t ℓ_it / Σ_i c_i n_i,
    i.e. workers keep dividing by the untouched global `batch_num_tokens` and the aggregate is
    exactly the self-normalized weighted token-mean. No worker-side denominator plumbing."""
    assert len(cs) == len(n_tokens) and len(cs) > 0
    tot = float(sum(n_tokens))
    wtot = float(sum(c * n for c, n in zip(cs, n_tokens)))
    if wtot <= 0.0 or tot <= 0.0:
        return [1.0] * len(cs)
    s = tot / wtot
    return [c * s for c in cs]


def ess(cs):
    """Effective sample size of a corrected batch: (Σc)^2 / Σc^2. Equals len(cs) when all c equal."""
    if not cs:
        return 0.0
    a = sum(cs)
    b = sum(c * c for c in cs)
    return (a * a) / b if b > 0 else 0.0


def snis_mean(values, cs):
    """Self-normalized IS estimate of a per-problem mean under the uniform target."""
    if not values:
        return float("nan")
    num = sum(c * v for c, v in zip(cs, values))
    den = sum(cs)
    return num / den if den > 0 else float("nan")


def snis_token_mean(token_sums, token_counts, cs):
    """SNIS estimate of a TOKEN-level mean (e.g. advantages): Σ c_i s_i / Σ c_i n_i, where s_i is
    row i's masked token sum and n_i its token count -- the c-weighted token-mean as an
    estimator."""
    if not token_sums:
        return float("nan")
    num = sum(c * s for c, s in zip(cs, token_sums))
    den = sum(c * n for c, n in zip(cs, token_counts))
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Metric audit: replace the standard train-metric keys with their SNIS-corrected
# estimates (so dashboards + cross-run comparisons read uniform-comparable numbers by
# default) and preserve the sampled-distribution values under difficulty_raw/<key>.
# ---------------------------------------------------------------------------

# sequence-level means: metric key -> name of the per-row values list in `rows`
_SEQ_MEAN_KEYS = {
    "critic/score/mean": "seq_score",              # non-aborted subset (mirrors raw computation)
    "critic/rewards/mean": "seq_reward",           # non-aborted subset
    "response_length/mean": "response_length",
    "response_length/clip_ratio": "resp_clip",
    "response_length_non_aborted/mean": "response_length_na",
    "response_length_non_aborted/clip_ratio": "resp_clip_na",
    "response/aborted_ratio": "aborted",
    "prompt_length/mean": "prompt_length",
    "prompt_length/clip_ratio": "prompt_clip",
    "num_turns/mean": "num_turns",
    "tool_call_counts/mean": "tool_calls",
}
# token-level means: metric key -> (per-row token-sum list, per-row token-count list)
_TOK_MEAN_KEYS = {
    "critic/advantages/mean": ("adv_sum", "n_tokens"),
    "critic/returns/mean": ("ret_sum", "n_tokens"),
}
# NEVER corrected (documented): max/min (order statistics, not population-mean estimators),
# critic/values/* + vf_explained_var (variance diagnostics), actor/entropy (the AEC driver signal
# -- its logged trace must equal what the controller consumed), optimizer/perf/timing metrics.


def apply_metric_corrections(metrics, cs, rows):
    """Overwrite each auditable metric key in `metrics` with its SNIS-corrected estimate and move
    the raw (sampled-distribution) value to difficulty_raw/<key>. `cs` = per-row draw-time
    corrections; `rows` = dict of per-row lists as named in _SEQ_MEAN_KEYS/_TOK_MEAN_KEYS
    (subset entries like seq_score come with a parallel `<name>__cs` list of that subset's c's).
    Missing rows entries or absent metric keys are skipped. Returns the corrected key list."""
    _init()
    done = []
    for key, name in _SEQ_MEAN_KEYS.items():
        vals = rows.get(name)
        if vals is None or key not in metrics or len(vals) == 0:
            continue
        sub_cs = rows.get(name + "__cs", cs)
        assert len(sub_cs) == len(vals), (key, len(sub_cs), len(vals))
        metrics[f"difficulty_raw/{key}"] = metrics[key]
        metrics[key] = snis_mean([float(v) for v in vals], sub_cs)
        done.append(key)
    for key, (sname, nname) in _TOK_MEAN_KEYS.items():
        sums, cnts = rows.get(sname), rows.get(nname)
        if sums is None or cnts is None or key not in metrics or len(sums) == 0:
            continue
        metrics[f"difficulty_raw/{key}"] = metrics[key]
        metrics[key] = snis_token_mean(
            [float(s) for s in sums], [float(n) for n in cnts], cs
        )
        done.append(key)
    return done


# ---------------------------------------------------------------------------
# persistence (mirrors aec_k.json; note _init() FIRST in load, as in aec.set_k:
# a later lazy _init() must never clobber loaded state)
# ---------------------------------------------------------------------------

def save_state(ckpt_dir):
    """DRIVER: persist per-qid stats next to a checkpoint. No-op when disabled. Atomic."""
    if not enabled():
        return
    try:
        p = os.path.join(ckpt_dir, "difficulty_state.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"stats": _S["stats"], "weight_form": _S["weight_form"]}, f)
        os.replace(tmp, p)
        print(f"[difficulty] saved {len(_S['stats'])} qid stats -> {p}", flush=True)
    except Exception as e:  # persistence must never crash training
        print(f"[difficulty] save_state failed: {e}", flush=True)


def load_state(ckpt_dir):
    """DRIVER: restore per-qid stats from a checkpoint dir. Returns count restored, or None."""
    if not enabled():
        return None
    try:
        p = os.path.join(ckpt_dir, "difficulty_state.json")
        if not os.path.exists(p):
            print(f"[difficulty] no persisted state at {p}; starting fresh", flush=True)
            return None
        with open(p) as f:
            d = json.load(f)
        stats = d.get("stats", {})
        for st in stats.values():  # tolerate older files without the flag
            st.setdefault("demoted", False)
            st["p"] = float(st["p"])
            st["n"] = int(st["n"])
        _S["stats"] = stats
        _S["version"] += 1
        print(f"[difficulty] resumed {len(stats)} qid stats from {p}", flush=True)
        return len(stats)
    except Exception as e:
        print(f"[difficulty] load_state failed: {e}", flush=True)
        return None


def warmstart_from_rollouts(rollout_dir, max_step=None):
    """DRIVER: warm-start p-hat from a run's HISTORICAL rollout dumps when grafting difficulty
    sampling onto a pre-difficulty checkpoint (no difficulty_state.json). Replays
    <rollout_dir>/<step>.jsonl in step order through observe(), so the EMA weights recent steps
    exactly as if difficulty sampling had been on all along. Row semantics (same as the offline
    analysis): qid = statement hash of the row's `input` text; pass = per-rollout judge score >= pass_thresh
    (prefer `prover_judge_score`, fall back to `score`). Rows without a score are skipped.

    `max_step` caps which files are replayed (use the RESUMED global step, so rollout files from
    beyond an older resume point are ignored). Env-gated via SP_DIFF_WARMSTART (default on).
    Returns the number of (step, qid) observations replayed, or None if disabled / no files.
    Never raises -- a corrupt line or file is skipped (this must not block a resume)."""
    if not enabled():
        return None
    if os.environ.get("SP_DIFF_WARMSTART", "1") not in ("1", "true", "True"):
        print("[difficulty] warm-start disabled via SP_DIFF_WARMSTART", flush=True)
        return None
    try:
        files = []
        for p in glob.glob(os.path.join(rollout_dir, "*.jsonl")):
            stem = os.path.basename(p).split(".")[0]
            if not stem.isdigit():
                continue
            step = int(stem)
            if max_step is not None and step > int(max_step):
                continue
            files.append((step, p))
        files.sort()
        if not files:
            print(f"[difficulty] warm-start: no rollout files under {rollout_dir}", flush=True)
            return None
        n_obs = 0
        for step, path in files:
            per = {}
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Replay-prefix runs: rollouts conditioned on a partial-proof prefix are
                        # systematically easier and must NOT feed the from-scratch p-hat (they
                        # would deflate every replayed problem's difficulty). Dumps written with
                        # the sp_replay hook carry sp_prefix_len; untagged rows count as prefix-free.
                        try:
                            if int(float(row.get("sp_prefix_len") or 0)) > 0:
                                continue
                        except (TypeError, ValueError):
                            pass
                        # binary pass via the shared helper (same as the live update): prefer the
                        # judge verdict, fall back to shaped score only if that's all the row has.
                        if (row.get("prover_judge_score") is None
                                and row.get("correctness_judge_score") is None
                                and row.get("score") is None):
                            continue
                        q = qid_from_text(row.get("input") or "")
                        a = per.setdefault(q, [0, 0])
                        a[0] += row_pass(
                            prover_judge_score=row.get("prover_judge_score"),
                            correctness_judge_score=row.get("correctness_judge_score"),
                            score=row.get("score"),
                        )
                        a[1] += 1
            except OSError:
                continue
            for q, (npass, ntot) in per.items():
                observe(q, npass / ntot)
                n_obs += 1
        print(
            f"[difficulty] warm-started p-hat from {len(files)} rollout files "
            f"(steps {files[0][0]}..{files[-1][0]}): {n_obs} step-observations over "
            f"{len(_S['stats'])} distinct qids",
            flush=True,
        )
        return n_obs
    except Exception as e:
        print(f"[difficulty] warm-start failed (continuing fresh): {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# weighted sampler (duck-typed torch Sampler: __iter__/__len__ + the torchdata
# Stateful protocol; deliberately torch-free so it is fully unit-testable)
# ---------------------------------------------------------------------------

class DifficultyWeightedSampler:
    """Samples dataset ROW indices with replacement, proportional to the current per-problem
    difficulty weight w (split evenly across duplicate rows of the same qid, so a problem's total
    draw probability is w_qid/W regardless of how many rows carry it -- and c then depends only on
    the qid). At each draw it pushes (row_idx, qid, c) into the module FIFO, so the correction the
    fit loop applies is the one matching the weights ACTUALLY used to draw, even when the
    dataloader prefetches several batches ahead.

    `get_row_messages(dataset, i)` -> the chat messages used for qid hashing; default reads
    dataset.dataframe[i][dataset.prompt_key] (verl RLHFDataset layout).

    Resume: implements state_dict/load_state_dict (RNG + position) so StatefulDataLoader restores
    it without naive index replay. Draws are i.i.d.-with-replacement so there is no epoch
    permutation to preserve; any FIFO misalignment after a resume is self-healed by the fit loop
    (row-idx cross-check -> clear + recompute from current state).
    """

    def __init__(self, dataset, seed=None, get_row_messages=None):
        _init()
        self._n = len(dataset)
        assert self._n > 0, "empty dataset"
        get = get_row_messages or (lambda ds, i: ds.dataframe[i][ds.prompt_key])
        self._row_qid = [qid_from_messages(get(dataset, i)) for i in range(self._n)]
        dup = {}
        for q in self._row_qid:
            dup[q] = dup.get(q, 0) + 1
        self._dup = dup                      # qid -> number of rows carrying it
        self._distinct = sorted(dup.keys())  # the uniform target universe (N distinct problems)
        self._rng = random.Random(seed if seed is not None else 0)
        self._pos = 0
        self._cache_version = -1
        self._row_w = None                   # per-row weights (w_qid / dup)
        self._c_map = None                   # qid -> correction at the cached version
        print(
            f"[difficulty] sampler over {self._n} rows, {len(self._distinct)} distinct qids "
            f"({self._n - len(self._distinct)} duplicate rows)",
            flush=True,
        )

    def _refresh(self):
        if self._cache_version == _S["version"]:
            return
        c_map, w_map = corrections_for(self._distinct)
        self._c_map = c_map
        self._row_w = [w_map[q] / self._dup[q] for q in self._row_qid]
        self._cache_version = _S["version"]

    def __len__(self):
        return self._n  # mirrors RandomSampler: one "epoch" = dataset-size draws

    def __iter__(self):
        # drop_last=True can discard the tail of an epoch's draws; reset the FIFO at each epoch
        # start so a fresh epoch never inherits stale draw records from the dropped tail (the fit
        # loop's row-id cross-check would otherwise trip the fallback for the first batch). Draws
        # are i.i.d.-with-replacement, so there is no ordering to preserve across the reset.
        clear_fifo()
        for _ in range(self._n):
            self._refresh()  # per-draw version check is O(1); weights rebuilt only after observe()
            i = self._rng.choices(range(self._n), weights=self._row_w, k=1)[0]
            q = self._row_qid[i]
            push_draw(i, q, self._c_map[q])
            self._pos += 1
            yield i

    def correction(self, qid):
        """Current-state correction for one problem -- the fit loop's fallback when the draw FIFO
        is unusable (resume replay). Unknown qid -> 1.0 (neutral)."""
        self._refresh()
        return self._c_map.get(qid, 1.0)

    def metrics(self):
        """Sampler-health numbers for the driver's difficulty/* metrics."""
        self._refresh()
        wmax = _w_max()
        n_floor = sum(1 for q in self._distinct if weight(q) <= _S["w_floor"] * (1 + 1e-9))
        n_seen = sum(1 for q in self._distinct if q in _S["stats"])
        # Advantage/gradient variance ratio (difficulty vs uniform): rho = Σ c_i v_i / Σ v_i with
        # v_i = p_i(1-p_i). This is the WITHIN-GROUP (advantage) variance term that w=√(p(1-p))
        # minimizes -- < 1 means reduction, and it CREDITS sampling the high-variance mid-difficulty
        # problems (unlike a Kish ESS on the corrections, which assumes equal per-sample variance and
        # only sees weight unevenness). Population-level over the distinct universe, using the SAME
        # p-hat the weights use. Derivation: Var(mu_hat) = (1/nNB)Σ c_i p_i(1-p_i) + (cross-problem);
        # only the first (within-group) term reaches the gradient, so it is the one we track here. The
        # cross-problem metric-scalar variance is deliberately NOT logged.
        _num = _den = 0.0
        _pr, _mo = _S["p_prior"], _S["min_obs"]
        for q in self._distinct:
            st = _S["stats"].get(q)
            p_eff = st["p"] if (st is not None and st["n"] >= _mo) else _pr
            vi = p_eff * (1.0 - p_eff)
            _num += self._c_map.get(q, 1.0) * vi
            _den += vi
        var_ratio = (_num / _den) if _den > 0 else 1.0
        # Record WHICH weight form is active this step (metrics.jsonl is one continuous file across
        # grafts, so log the choice per step to know retroactively which variant produced each step):
        # step=0, sqrt_p=1, sqrt_pq=2, safe=3.
        _wform_code = {"step": 0, "sqrt_p": 1, "sqrt_pq": 2, "safe": 3}.get(_S["weight_form"], -1)
        return {
            "difficulty/distinct_problems": len(self._distinct),
            "difficulty/seen_problems": n_seen,
            "difficulty/floored_frac": n_floor / max(len(self._distinct), 1),
            "difficulty/mean_weight": sum(self._row_w) / max(self._n, 1),  # mean per-row weight
            "difficulty/max_c": max(self._c_map.values()) if self._c_map else 1.0,
            "difficulty/unseen_weight": wmax,
            "difficulty/var_ratio": var_ratio,  # advantage-variance ratio vs uniform (<1 = reduction)
            "difficulty/weight_form_code": _wform_code,  # 0=step 1=sqrt_p 2=sqrt_pq 3=safe
        }

    # --- torchdata Stateful protocol ---
    def state_dict(self):
        return {"rng_state": self._rng.getstate(), "pos": self._pos}

    def load_state_dict(self, sd):
        try:
            self._rng.setstate(tuple(
                tuple(x) if isinstance(x, list) else x for x in sd["rng_state"]
            ))
        except Exception:
            pass  # RNG restore is best-effort: draws are i.i.d., a reseed is statistically fine
        self._pos = int(sd.get("pos", 0))
