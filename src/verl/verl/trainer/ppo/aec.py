"""Adaptive entropy control (MAI-Thinking-1, Eqs 7-8) -- driver-synchronized variant.

Drives verl's global actor entropy (entropy_agg from the old_log_prob step = the logged
`actor/entropy`) toward target H* by relaxing ONLY the PPO upper clip bound:
    upper = 1 + clip_ratio_high + k,   lower = 1 - clip_ratio_low (unchanged)
    per step:  k <- clip(k + delta * sign(H* - H), k_min, k_max),  init k = k_init (default 0).

Synchronization: k is computed ONCE on the driver (single global entropy_agg -> single k) and
broadcast to every worker via tu.assign_non_tensor(batch_td, aec_k=k) -> data["aec_k"] in ppo_loss,
which calls set_k(). The clip constant is thus identical on all workers by construction -- no
worker-side reduction. k is floored at k_min (default 0): with k_min=0 it is a pure entropy FLOOR
(no-op while H >= H*). 1+eps form (matches verl's 1+clip_ratio_high) so k=0 == plain PPO clipping.
Env-gated (SP_ADAPTIVE_ENTROPY + SP_AEC_TARGET_H/DELTA/KMAX/KMIN/KINIT)."""
import os
import json

_S = {"inited": False, "enable": False, "target_h": 0.155, "delta": 0.01,
      "kmax": 0.08, "kmin": 0.0, "k": 0.0}


def _init():
    if _S["inited"]:
        return
    _S["enable"] = os.environ.get("SP_ADAPTIVE_ENTROPY", "0") in ("1", "true", "True")
    _S["target_h"] = float(os.environ.get("SP_AEC_TARGET_H", "0.155"))
    _S["delta"] = float(os.environ.get("SP_AEC_DELTA", "0.01"))
    _S["kmax"] = float(os.environ.get("SP_AEC_KMAX", "0.08"))
    _S["kmin"] = float(os.environ.get("SP_AEC_KMIN", "0.0"))
    # Initial integral seed. Default 0 reproduces the baseline; set >0 (e.g. = kmax) to START with a
    # relaxed upper clip -- used when re-enabling entropy mid-run after a collapse so the floor is
    # active immediately instead of integrating up from 0. Overridden by load_k on resume whenever a
    # persisted aec_k.json exists (so KINIT seeds only the very first AEC step, not later restarts).
    _S["k"] = float(os.environ.get("SP_AEC_KINIT", "0.0"))
    _S["inited"] = True


def enabled():
    _init()
    return _S["enable"]


def driver_step_k(H):
    """DRIVER only: integral step from the single global actor entropy H (entropy_agg). Returns k."""
    if not enabled():
        return 0.0
    sign = 1.0 if H < _S["target_h"] else (-1.0 if H > _S["target_h"] else 0.0)
    _S["k"] = min(max(_S["k"] + _S["delta"] * sign, _S["kmin"]), _S["kmax"])
    print(f"[aec] H={H:.4f} target={_S['target_h']:.3f} sign={sign:+.0f} -> k={_S['k']:.4f}", flush=True)
    return _S["k"]


def set_k(k):
    """WORKER: adopt the driver's clip constant (identical across workers). _init() FIRST so a
    lazy enabled()/current_k() later in the same call cannot re-run _init() and clobber k back to
    SP_AEC_KINIT (the first ppo_loss micro-batch on a fresh worker would otherwise ignore aec_k)."""
    _init()
    _S["k"] = float(k)


def current_k():
    """Upper-clip relaxation used by compute_policy_loss_vanilla; 0.0 unless AEC on."""
    return float(_S["k"]) if enabled() else 0.0


def save_k(ckpt_dir):
    """DRIVER: persist the integral k next to a checkpoint so it survives resume / job restarts.
    No-op when AEC is off. Atomic (tmp + os.replace) so a mid-write crash can't leave a torn file."""
    if not enabled():
        return
    try:
        p = os.path.join(ckpt_dir, "aec_k.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"k": float(_S["k"]), "target_h": _S["target_h"]}, f)
        os.replace(tmp, p)
        print(f"[aec] saved k={_S['k']:.4f} -> {p}", flush=True)
    except Exception as e:  # persistence must never crash training
        print(f"[aec] save_k failed: {e}", flush=True)


def load_k(ckpt_dir):
    """DRIVER: restore the integral k from a checkpoint dir. Returns the loaded k, or None if AEC
    is off / no file. Seeds _S["k"] so the next driver_step_k continues from the persisted value."""
    if not enabled():
        return None
    try:
        p = os.path.join(ckpt_dir, "aec_k.json")
        if not os.path.exists(p):
            print(f"[aec] no persisted k at {p}; starting k={_S['k']:.4f}", flush=True)
            return None
        with open(p) as f:
            d = json.load(f)
        _S["k"] = float(d["k"])
        print(f"[aec] resumed k={_S['k']:.4f} from {p}", flush=True)
        return _S["k"]
    except Exception as e:
        print(f"[aec] load_k failed: {e}", flush=True)
        return None
