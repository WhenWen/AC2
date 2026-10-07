"""Q-quality monitoring dataset for ac2 (and any sp_q run).

One record per (step, uid) group -- i.e. one problem attempt at one training step, with all
of its rollout.n completions. Answers three questions per record:

  1. WHICH PROBLEM       -- ``uid`` (per-step group id) + ``qid`` (stable problem id).
  2. READINESS AT THE TIME -- the gate state for THIS problem at THIS step: whether it was
     ready, the step it became ready, this step's fresh probe (prediction / target z /
     error) and the target's provenance, plus the two thresholds and the global pooled
     MAE the gate was compared against.
  3. THE 16 OUTCOMES     -- every completion with its LANE, so Q-scored and full-length
     judged rollouts are separable:
       * ``q_consumed``  -- the short lane cut it at prefix+g and Q supplied the score.
                            ``score`` IS the Q value (ds4_finegrained_judge substitutes it).
       * ``audit_full``  -- audit lane: ran the FULL budget, real judge reward. One member
                            per audit group additionally carries the counterfactual Q that
                            WOULD have been consumed (``audit_cut`` block).
       * ``judged``      -- ran to a natural end / had an extractable proof, so the final
                            judge scored it even though the group was ready.
     Note the route is a GROUP property: a group is entirely audit, entirely short (whose
     members then split judged/q_consumed), or entirely full (not ready). You never get an
     audit member and a Q member in the same group -- the Q-vs-audit comparison is BETWEEN
     groups on the same problem, or WITHIN an audit group via ``audit_cut``.

Sources (all under ``run_data/``, all read streaming; the rollout dumps are ~600MB/step and
are never held in memory):
  * ``rollouts/<step>.jsonl``    -- per completion: uid, qid, score, sp_q_route,
                                    sp_q_route_taken, sp_q_invalid, sp_prefix_len,
                                    response_length. The ONLY source of per-completion
                                    terminal judge rewards.
  * ``q_wave/<step>.jsonl``      -- every Q call (probe | consumed | audit_cut) with the
                                    value Q returned. Requires SP_Q_DUMP_WAVE=1.
  * ``q_state_deltas.jsonl``     -- per step: probe records (prediction/z/error/provenance),
                                    audit_pairs, readiness transitions, pooled mae5.

Usage (on the cluster, inside the repo venv):
    python experiments/ac2/q_monitor_dataset.py --schema
    python experiments/ac2/q_monitor_dataset.py --out analysis/q_monitor
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT_RUN = os.path.join(
    os.environ.get("AC2_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "experiments/ac2/run_data",
)

# Gate constants for this arm (run_attach.sh). Recorded on every row so a reader
# never has to go find the launch script to interpret an error against its threshold.
READY_THRESH_GLOBAL = float(os.environ.get("SP_Q_READY_THRESH_GLOBAL", 0.2))
READY_THRESH_PROBLEM = float(os.environ.get("SP_Q_READY_THRESH_PROBLEM", 0.18))
ROLLOUT_N = int(os.environ.get("SP_ROLLOUT_N", 16))

# Fields pulled from each rollout row. Everything else (input/output text, the two token-id
# arrays that dominate the file) is dropped immediately.
ROLLOUT_FIELDS = (
    "uid", "qid", "score", "acc", "step",
    "sp_q_route", "sp_q_route_taken", "sp_q_consumed", "sp_q_invalid",
    "sp_prefix_len", "response_length", "prompt_length",
    "prover_judge_score", "judge_http_error", "judge_parse_failed",
)


def _f(x):
    """float or None -- dumps stringify some fields via default=str, so accept both."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x, default=0):
    v = _f(x)
    return default if v is None else int(v)


def steps_available(run_dir: str) -> list[int]:
    d = os.path.join(run_dir, "rollouts")
    out = []
    for fn in os.listdir(d):
        if fn.endswith(".jsonl"):
            try:
                out.append(int(fn[: -len(".jsonl")]))
            except ValueError:
                continue
    return sorted(out)


def read_deltas(run_dir: str) -> tuple[dict, dict]:
    """Per-step Q state, plus the monotone qid -> step-it-became-ready map.

    Returns ({step: {"probes": {uid: rec}, "audit": {uid: rec}, "mae5": float|None,
                     "transitions": [qid...]}},
             {qid: step_became_ready})
    """
    path = os.path.join(run_dir, "q_state_deltas.jsonl")
    by_step: dict[int, dict] = {}
    ready_since: dict[str, int] = {}
    if not os.path.exists(path):
        return by_step, ready_since
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail on a live run
            gstep = int(d["dataset_step"]) + 1  # deltas are 0-based; dumps are 1-based
            probes = {}
            for p in d.get("probes") or []:
                probes[str(p.get("uid"))] = {
                    "prediction": _f(p.get("prediction")),
                    "z": _f(p.get("z")),
                    "error": _f(p.get("error")),
                    "source_tag": p.get("source_tag"),
                    "variant": p.get("variant"),
                    "route": p.get("route"),
                    "fail_reason": p.get("fail_reason"),
                    "qid": p.get("qid"),
                }
            audit = {}
            for a in d.get("audit_pairs") or []:
                audit[str(a.get("uid"))] = {
                    "q_at_cut": _f(a.get("q_at_cut")),
                    "terminal_reward": _f(a.get("terminal_reward")),
                    "variant": a.get("variant"),
                    "fail_reason": a.get("fail_reason"),
                }
            trans = [str(q) for q in (d.get("transitions") or [])]
            by_step[gstep] = {
                "probes": probes, "audit": audit, "mae5": _f(d.get("mae5")),
                "transitions": trans,
            }
            for qid in trans:
                ready_since.setdefault(qid, gstep)
    return by_step, ready_since


def read_wave(run_dir: str, global_step: int) -> dict:
    """Q calls for one step, keyed by (kind, uid).

    THE FILENAMES DO NOT AGREE ACROSS DUMPS. ``q_wave/<N>.jsonl`` is named by
    ``sp_dataset_step`` (0-based; ray_trainer's wave site reads
    ``int(extra[0]["sp_dataset_step"])``), while ``rollouts/<N>.jsonl`` is named by
    ``global_steps`` (1-based) and ``q_state_deltas`` carries ``dataset_step``. So the wave
    for global step S lives in ``q_wave/(S-1).jsonl``. Joining on the raw filename silently
    pairs each group with the NEXT step's Q calls -- which still looks plausible (the group
    count per step is constant) and only shows up as audit_cut rows disagreeing with the
    delta's audit_pairs.

    Never parses ctx_token_ids: every scalar field is written BEFORE "gen_text", so the line
    prefix up to it is a complete object.
    """
    path = os.path.join(run_dir, "q_wave", "%d.jsonl" % (global_step - 1))
    out: dict[tuple, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            cut = line.find(', "gen_text"')
            head = (line[:cut] + "}") if cut > 0 else line.strip()
            try:
                d = json.loads(head)
            except json.JSONDecodeError:
                continue
            kind, uid = d.get("kind"), str(d.get("uid"))
            rec = {
                "value": _f(d.get("value")),
                "variant": d.get("variant"),
                "overflow": bool(d.get("overflow")),
                "fail_reason": d.get("fail_reason"),
                "qid": d.get("qid"),
            }
            if kind == "consumed":
                out.setdefault(("consumed", uid), []).append(rec)
            else:
                out[(kind, uid)] = rec
    return out


def read_rollouts(run_dir: str, step: int) -> dict:
    """Group one step's completions by uid. Streams; token-id arrays are dropped per line."""
    path = os.path.join(run_dir, "rollouts", "%d.jsonl" % step)
    groups: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = {k: d.get(k) for k in ROLLOUT_FIELDS}
            row["_idx"] = idx
            groups[str(row.get("uid"))].append(row)
            del d
    return groups


def lane_of(row: dict) -> str:
    """Which scoring path actually produced this completion's score."""
    if str(row.get("sp_q_route_taken") or "") == "q_consumed":
        return "q_consumed"
    if str(row.get("sp_q_route") or "full") == "audit":
        return "audit_full"
    return "judged"


def build_record(step, uid, rows, wave, dstep, ready_since) -> dict:
    rows = sorted(rows, key=lambda r: r["_idx"])
    probe = (dstep.get("probes") or {}).get(uid)
    audit = (dstep.get("audit") or {}).get(uid)
    wave_probe = wave.get(("probe", uid))
    wave_audit = wave.get(("audit_cut", uid))

    qid = None
    for src in (rows[0].get("qid"), (probe or {}).get("qid"), (wave_probe or {}).get("qid")):
        if src not in (None, "", "None"):
            qid = str(src)
            break

    route = str(rows[0].get("sp_q_route") or "full")
    # Readiness is exactly "the group was routed off the full lane": route_plan_for_step
    # draws audit/short ONLY from slots whose qid passed the gate.
    ready_now = route in ("short", "audit")
    since = ready_since.get(qid) if qid else None

    completions = []
    for j, r in enumerate(rows):
        lane = lane_of(r)
        invalid = _i(r.get("sp_q_invalid"), 0)
        score = _f(r.get("score"))
        q_val = score if (lane == "q_consumed" and not invalid) else None
        judge = None if lane == "q_consumed" else score
        completions.append({
            "i": j,
            "lane": lane,
            "score": score,
            "q_value": q_val,
            "judge_reward": judge,
            "q_invalid": invalid,
            "prefix_len": _i(r.get("sp_prefix_len"), 0),
            "response_length": _i(r.get("response_length"), 0),
            "judge_pass": _f(r.get("prover_judge_score")),
        })

    qs = [c["q_value"] for c in completions if c["q_value"] is not None]
    js = [c["judge_reward"] for c in completions if c["judge_reward"] is not None]
    lanes = Counter(c["lane"] for c in completions)

    rec = {
        "step": step,
        "uid": uid,
        "qid": qid,
        "readiness": {
            "ready": ready_now,
            "route": route,
            # ready_since_step is a fact about the PROBLEM (readiness is monotone), so it is
            # carried on every record for that qid -- including steps BEFORE it went ready,
            # where it is legitimately in the future. steps_ready is the per-record age and
            # must not go negative there; None means "not ready yet at this step".
            "ready_since_step": since,
            "steps_ready": (step - since) if (since is not None and since <= step) else None,
            "became_ready_this_step": bool(qid and since == step),
            "probe_prediction": (probe or {}).get("prediction"),
            "probe_z": (probe or {}).get("z"),
            "probe_error": (probe or {}).get("error"),
            "probe_source_tag": (probe or {}).get("source_tag"),
            "probe_variant": (probe or {}).get("variant"),
            "probe_overflow": bool((wave_probe or {}).get("overflow")) if wave_probe else None,
            "thresh_problem": READY_THRESH_PROBLEM,
            "thresh_global": READY_THRESH_GLOBAL,
            "mae5_global": dstep.get("mae5"),
            "gate_open_global": (dstep.get("mae5") is not None
                                 and dstep["mae5"] < READY_THRESH_GLOBAL),
        },
        "n_completions": len(completions),
        "is_replay_group": len(completions) > 1,  # scratch/inflow rows run rollout_n=1
        "completions": completions,
        "summary": {
            "lanes": dict(lanes),
            "n_q": len(qs),
            "n_judged": len(js),
            "q_mean": (sum(qs) / len(qs)) if qs else None,
            "q_std": statistics.pstdev(qs) if len(qs) > 1 else (0.0 if qs else None),
            "judge_mean": (sum(js) / len(js)) if js else None,
            "judge_std": statistics.pstdev(js) if len(js) > 1 else (0.0 if js else None),
            "q_invalid": sum(c["q_invalid"] for c in completions),
            "q_values": qs,
            "judge_rewards": js,
        },
        # Only populated for audit groups: the counterfactual Q at the cut state paired
        # with the SAME rollout's realized terminal reward. The one within-group
        # Q-vs-ground-truth comparison the design provides.
        "audit_cut": None,
    }
    if audit or wave_audit:
        rec["audit_cut"] = {
            "q_at_cut": (audit or {}).get("q_at_cut",
                                          (wave_audit or {}).get("value")),
            "terminal_reward": (audit or {}).get("terminal_reward"),
            "variant": (audit or {}).get("variant", (wave_audit or {}).get("variant")),
            "fail_reason": (audit or {}).get("fail_reason"),
            "overflow": bool((wave_audit or {}).get("overflow")) if wave_audit else None,
        }
        qc, tr = rec["audit_cut"]["q_at_cut"], rec["audit_cut"]["terminal_reward"]
        rec["audit_cut"]["abs_error"] = abs(qc - tr) if (qc is not None and tr is not None) else None
    return rec


PAIR_COLS = (
    "step", "uid", "qid", "q_at_cut", "terminal_reward", "abs_error", "signed_error",
    "variant", "overflow", "fail_reason",
    "ready_since_step", "steps_ready", "probe_prediction", "probe_z", "probe_error",
    "probe_source_tag", "mae5_global", "n_completions", "group_judge_mean",
    "group_judge_std", "group_pass_rate",
)


def emit_audit_pairs(dataset_path: str, out_dir: str) -> int:
    """Flatten the one calibration pair per audit group into its own small file.

    These are the rows behind dashboard panel 48 -- (Q that WOULD have been consumed at the
    cut, realized terminal reward of that SAME rollout) -- plus the readiness context the
    panel drops. Derived from the dataset rather than re-read from the dumps, so it costs a
    single pass over a few tens of MB instead of another 23GB.
    """
    import csv

    js_path = os.path.join(out_dir, "q_audit_pairs.jsonl")
    csv_path = os.path.join(out_dir, "q_audit_pairs.csv")
    n = 0
    with open(dataset_path, encoding="utf-8") as f, \
            open(js_path, "w", encoding="utf-8") as js, \
            open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=list(PAIR_COLS))
        w.writeheader()
        for line in f:
            rec = json.loads(line)
            ac = rec.get("audit_cut")
            if not ac or ac.get("q_at_cut") is None or ac.get("terminal_reward") is None:
                continue
            rd, sm = rec["readiness"], rec["summary"]
            js_rewards = sm.get("judge_rewards") or []
            row = {
                "step": rec["step"], "uid": rec["uid"], "qid": rec["qid"],
                "q_at_cut": ac["q_at_cut"], "terminal_reward": ac["terminal_reward"],
                "abs_error": ac.get("abs_error"),
                "signed_error": ac["q_at_cut"] - ac["terminal_reward"],
                "variant": ac.get("variant"), "overflow": ac.get("overflow"),
                "fail_reason": ac.get("fail_reason"),
                "ready_since_step": rd.get("ready_since_step"),
                "steps_ready": rd.get("steps_ready"),
                "probe_prediction": rd.get("probe_prediction"),
                "probe_z": rd.get("probe_z"),
                "probe_error": rd.get("probe_error"),
                "probe_source_tag": rd.get("probe_source_tag"),
                "mae5_global": rd.get("mae5_global"),
                "n_completions": rec["n_completions"],
                "group_judge_mean": sm.get("judge_mean"),
                "group_judge_std": sm.get("judge_std"),
                # the group's realized pass rate under the fine-grained judge (score==1.0
                # is a full-rubric proof); the natural x-axis for "was Q right about this
                # problem" when the single paired reward is too noisy on its own
                "group_pass_rate": (sum(1 for v in js_rewards if v is not None and v >= 1.0)
                                    / len(js_rewards)) if js_rewards else None,
            }
            js.write(json.dumps(row) + "\n")
            w.writerow(row)
            n += 1
    print("wrote %s and %s (%d pairs)" % (js_path, csv_path, n))
    return n


def schema_report(run_dir: str) -> None:
    steps = steps_available(run_dir)
    if not steps:
        print("no rollout dumps under %s" % run_dir)
        return
    s = steps[len(steps) // 2]
    print("run_dir      : %s" % run_dir)
    print("steps        : %d..%d (n=%d)" % (steps[0], steps[-1], len(steps)))
    with open(os.path.join(run_dir, "rollouts", "%d.jsonl" % s), encoding="utf-8") as f:
        row = json.loads(f.readline())
    big = {k for k, v in row.items() if isinstance(v, (list, str)) and len(v) > 200}
    print("\nrollout keys (step %d), '*' = present of the ones we read:" % s)
    for k in sorted(row):
        mark = "*" if k in ROLLOUT_FIELDS else " "
        val = "<%d-elem/char>" % len(row[k]) if k in big else repr(row[k])[:70]
        print("  %s %-24s %s" % (mark, k, val))
    missing = [k for k in ROLLOUT_FIELDS if k not in row]
    print("\nMISSING requested fields: %s" % (missing or "none"))
    by_step, ready_since = read_deltas(run_dir)
    print("\nalignment check (global step S <-> q_wave/(S-1).jsonl <-> dataset_step S-1):")
    print("  %-6s %-28s %-22s %s" % ("step", "wave rows", "delta", "verdict"))
    for s2 in steps[len(steps) // 2: len(steps) // 2 + 4]:
        wv = read_wave(run_dir, s2)
        d = by_step.get(s2, {})
        n_cut = sum(1 for k in wv if k[0] == "audit_cut")
        n_pair = len(d.get("audit") or {})
        # every audit_pair is built from an audit_cut wave row, so cut >= pairs must hold;
        # cut < pairs is the tell that the two files are off by a step.
        print("  %-6d %-28s %-22s %s"
              % (s2, dict(Counter(k[0] for k in wv)),
                 "probes=%d pairs=%d" % (len(d.get("probes") or {}), n_pair),
                 "ok" if n_cut >= n_pair else "MISALIGNED (cut<pairs)"))
    print("\nqids ever ready: %d" % len(ready_since))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--out", default=None, help="output dir (default <run-dir>/../analysis)")
    ap.add_argument("--steps", default=None, help="e.g. 10-38 or 5,7,9")
    ap.add_argument("--schema", action="store_true", help="print source schemas and exit")
    ap.add_argument("--pairs-from", default=None,
                    help="skip extraction; re-emit the flat audit-pairs files from an "
                         "existing q_monitor.jsonl")
    args = ap.parse_args()

    run_dir = args.run_dir
    if args.schema:
        schema_report(run_dir)
        return 0

    out_dir = args.out or os.path.join(os.path.dirname(os.path.normpath(run_dir)), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q_monitor.jsonl")

    if args.pairs_from:
        emit_audit_pairs(args.pairs_from, out_dir)
        return 0

    steps = steps_available(run_dir)
    if args.steps:
        if "-" in args.steps:
            a, b = args.steps.split("-", 1)
            steps = [s for s in steps if int(a) <= s <= int(b)]
        else:
            want = {int(x) for x in args.steps.split(",")}
            steps = [s for s in steps if s in want]

    by_step, ready_since = read_deltas(run_dir)
    tot = Counter()
    per_step_rows = []

    with open(out_path, "w", encoding="utf-8") as out:
        for step in steps:
            groups = read_rollouts(run_dir, step)
            wave = read_wave(run_dir, step)
            dstep = by_step.get(step, {})
            n_rec = 0
            lanes = Counter()
            for uid in sorted(groups):
                rec = build_record(step, uid, groups[uid], wave, dstep, ready_since)
                out.write(json.dumps(rec) + "\n")
                n_rec += 1
                lanes.update(rec["summary"]["lanes"])
                tot["records"] += 1
                tot["ready_groups"] += int(rec["readiness"]["ready"])
                tot["audit_groups"] += int(rec["readiness"]["route"] == "audit")
                tot["audit_cut_pairs"] += int(
                    bool(rec["audit_cut"]) and rec["audit_cut"].get("abs_error") is not None)
            tot.update(lanes)
            per_step_rows.append({
                "step": step, "groups": n_rec, "lanes": dict(lanes),
                "mae5": dstep.get("mae5"),
                "ready_groups": sum(1 for u in groups
                                    if str(groups[u][0].get("sp_q_route") or "full") != "full"),
            })
            print("step %-4d groups=%-5d lanes=%s" % (step, n_rec, dict(lanes)), flush=True)

    summary = {
        "run_dir": run_dir,
        "steps": steps,
        "totals": dict(tot),
        "per_step": per_step_rows,
        "thresholds": {"global": READY_THRESH_GLOBAL, "problem": READY_THRESH_PROBLEM},
        "qids_ever_ready": len(ready_since),
    }
    sum_path = os.path.join(out_dir, "q_monitor_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s (%d records)" % (out_path, tot["records"]))
    print("wrote %s" % sum_path)
    emit_audit_pairs(out_path, out_dir)
    print("totals: %s" % dict(tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
