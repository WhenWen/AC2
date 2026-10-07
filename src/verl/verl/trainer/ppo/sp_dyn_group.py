"""Dynamic group size for the prover rollout.

WHAT IT DOES
------------
Normal training draws a fixed number of completions per prompt (``rollout.n``).
This module replaces that with a doubling schedule that stops early on prompts
which already produced a high reward completion.

    k = K_MIN
    round 1: draw 2**k completions for every trained prompt
    loop:
        score every new completion (Q wave for capped no-proof short rows,
        judge for everything else -- exactly the production reward path)
        a prompt stops if any of its completions so far has score >= THRESH
        if k == K_MAX: stop every remaining prompt
        k += 1
        draw 2**k - 2**(k-1) more completions for the prompts still running

With the defaults K_MIN=3 and K_MAX=5 the cumulative group sizes are 8, 16, 32
and the per round increments are 8, 8, 16.

WHAT "SCORE" IS
---------------
Every completion ends with exactly one scalar, the reward function's ``score``,
and it has exactly one generator per completion:

  * a ``short``-routed completion that hit the g cap with no ``<proof>`` is
    scored by the actor's own Q function in the pre-judge wave, and the reward
    function returns that grid value {0, 0.1, ..., 1} without a judge call
    (``ds4_finegrained_judge.py``, the ``sp_q_route_taken == "q_consumed"``
    branch);
  * a completion with no ``<proof>`` is scored 0 without a judge call;
  * everything else is judged, ``score = points/7`` with points in {0,1,6,7}.

So "Q >= 0.7 or reward >= 0.7" is one test on one number. There is no per
completion choice between a Q value and a reward. Q also saves no judge calls:
a consumed row has no proof, so the judge would have returned 0 without a call
anyway.

Measured on step 99 of a no-audit branch of AC2 (3264 rows): 1045 scored by Q
(values 0/0.1/0.9/1.0 on 310/215/224/296 rows) and 2219 by the judge path
(points 0/1/6/7 on 1131/135/166/787 rows). Under the 8/16/32 schedule with
threshold 0.7 the expected draw count is 18.0 per group against the fixed 16,
and 108 of 192 groups stop after the first round.

WHAT THIS MODULE DOES NOT DO
----------------------------
It holds the schedule and the predicate only, with no verl imports, so it can
be unit tested on a laptop. The integration (round loop, wave, judge, replica
sleep and wake, cache semantics) lives in
``RayPPOTrainer._sp_dyn_generate_wave_reward`` (ray_trainer.py).

ENVIRONMENT
-----------
    SP_DYN_GROUP     1 to enable, anything else leaves training unchanged
    SP_DYN_K_MIN     starting exponent, default 3
    SP_DYN_K_MAX     final exponent, default 5
    SP_DYN_THRESH    high reward threshold on ``score``, default 0.7

Every one of these must also be added to the forward list the experiment's
runner.py builds for Ray ``runtime_env.env_vars``. The trainer runs inside a
Ray actor and an actor does not inherit the driver environment, so a variable
that is only exported by the launch script reads as empty here and the schedule
silently does not engage.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Set


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return os.environ.get("SP_DYN_GROUP", "0") in ("1", "true", "True")


def k_min() -> int:
    return _env_int("SP_DYN_K_MIN", 3)


def k_max() -> int:
    return _env_int("SP_DYN_K_MAX", 5)


def threshold() -> float:
    return _env_float("SP_DYN_THRESH", 0.7)


def cumulative_sizes() -> List[int]:
    """Group size after each round. Defaults give [8, 16, 32]."""
    return [2 ** k for k in range(k_min(), k_max() + 1)]


def increments() -> List[int]:
    """Completions to draw in each round. Defaults give [8, 8, 16]."""
    sizes = cumulative_sizes()
    out = [sizes[0]]
    for prev, cur in zip(sizes, sizes[1:]):
        out.append(cur - prev)
    return out


def n_rounds() -> int:
    return len(increments())


def describe() -> str:
    return (
        "[sp_dyn] dynamic group size: k from %d to %d, cumulative %s, increments %s, "
        "threshold on score=%.3f"
        % (k_min(), k_max(), cumulative_sizes(), increments(), threshold())
    )


# --------------------------------------------------------------------------- predicate


def is_high_reward(score: float) -> bool:
    """One completion counts as a high reward action when its unified score
    reaches the threshold. The score is provenance agnostic: a Q grid value on a
    capped partial and a judge points/7 on a finished proof are compared to the
    same number."""
    return float(score) >= threshold()


def satisfied_rows(row_of_seq: Sequence[int], scores: Sequence[float]) -> Set[int]:
    """Source rows holding at least one high reward completion.

    ``row_of_seq[j]`` is the source row of the j th completion, ``scores[j]`` its
    unified score. The two sequences are parallel, one entry per completion.
    """
    if len(row_of_seq) != len(scores):
        raise ValueError(
            "row_of_seq and scores must be parallel: got %d and %d"
            % (len(row_of_seq), len(scores))
        )
    out: Set[int] = set()
    for src, s in zip(row_of_seq, scores):
        if src not in out and is_high_reward(s):
            out.add(int(src))
    return out


# --------------------------------------------------------------------------- schedule


def round_counts(stamps: Sequence[int], rnd: int, satisfied: Iterable[int]) -> List[int]:
    """Repeat count per source row for round ``rnd`` (0 based).

    ``stamps`` are the trainer's three valued ``sp_rollout_n`` values read from
    ``extra_info``: positive = an inflow row with that many rollouts, 0 = a
    trained (replay or cold scratch) row that takes the schedule, negative = a
    row dropped from the step. The result is a plain repeat count (0 = drop)
    to hand straight to ``DataProto.sample_level_repeat``.

    Inflow rows are generated and judged once, in round 1, and never take part
    in the schedule: they are single rollouts that feed the replay buffer and
    are removed from the trained batch after the reward anyway.
    """
    inc = increments()
    if not 0 <= rnd < len(inc):
        raise ValueError("round %d outside the schedule of %d rounds" % (rnd, len(inc)))
    done = set(int(i) for i in satisfied)
    out: List[int] = []
    for i, st in enumerate(stamps):
        st = int(st)
        if st < 0:
            out.append(0)
        elif st > 0:
            out.append(st if rnd == 0 else 0)
        else:
            out.append(0 if i in done else inc[rnd])
    return out


def expand_rows(counts: Sequence[int]) -> List[int]:
    """Source row index of every completion ``sample_level_repeat(counts)``
    produces, in the order it produces them (all copies of row 0, then row 1,
    and so on)."""
    return [i for i, c in enumerate(counts) for _ in range(max(int(c), 0))]


def pad_to_multiple(counts: List[int], multiple: int, stamps: Sequence[int]) -> List[int]:
    """Raise the round's sequence total to a multiple of ``multiple``.

    DataProto refuses an uneven chunk (world size 32 at tensor parallel 4
    chunks by 8). Every default increment is a multiple of 8 and inflow rows
    only appear in round 1, so with the default schedule this is a guard. The
    padding goes to trained rows that are still live in this round, round
    robin, so no extra completion lands on a stopped group or an inflow row.
    """
    if multiple <= 1:
        return counts
    # Only TRAINED rows count toward the total: inflow rows are dropped before the PPO
    # update, and it is the post-drop total that the two-minibatch update divides.
    total = sum(c for i, c in enumerate(counts) if c > 0 and int(stamps[i]) == 0)
    if total % multiple == 0:
        return counts
    live = [i for i, c in enumerate(counts) if c > 0 and int(stamps[i]) == 0]
    if not live:
        raise ValueError("cannot pad a round with no live trained row")
    need = multiple - (total % multiple)
    out = list(counts)
    for j in range(need):
        out[live[j % len(live)]] += 1
    return out


def total_counts(per_round: Sequence[Sequence[int]]) -> List[int]:
    """Completions each source row ended up with across all rounds. Used for
    the per step log line and the ``sp_dyn/*`` metrics, not for a repeat: the
    driver batch is assembled round by round, so there is no second repeat site
    to feed."""
    if not per_round:
        return []
    out = [0] * len(per_round[0])
    for counts in per_round:
        if len(counts) != len(out):
            raise ValueError("per round count vectors differ in length")
        for i, c in enumerate(counts):
            out[i] += max(int(c), 0)
    return out


def summarize(per_round: Sequence[Sequence[int]], stamps: Sequence[int]) -> dict:
    """Metrics for one step: how many trained groups stopped after each round and
    the mean group size, so the schedule's cost is visible in the dashboard."""
    tot = total_counts(per_round)
    trained = [i for i, st in enumerate(stamps) if int(st) == 0]
    sizes = cumulative_sizes()
    stopped_at = {s: 0 for s in sizes}
    for i in trained:
        stopped_at[min(sizes, key=lambda s: abs(s - tot[i]))] += 1
    m = {"sp_dyn/groups": len(trained),
         "sp_dyn/mean_group_size": (sum(tot[i] for i in trained) / len(trained)) if trained else 0.0,
         "sp_dyn/rounds_run": len(per_round)}
    for s in sizes:
        m["sp_dyn/stopped_at_%d" % s] = stopped_at[s]
    return m
