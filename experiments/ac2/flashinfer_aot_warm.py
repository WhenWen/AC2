#!/usr/bin/env python3
"""Pre-build flashinfer's JIT modules ONCE, single-process, and promote them to the AOT
directory so no vLLM rank ever JIT-builds them again.

WHY. vLLM's AllReduceRMSFusionPass -> get_fi_ar_workspace ->
flashinfer trtllm_lamport_initialize -> get_trtllm_comm_module -> JitSpec.build_and_load,
which does `with FileLock(...): self.build(); self.load()` -- UNCONDITIONALLY, once per
rank. Every rank of a TP group serializes on that one lock and each re-runs ninja (they
race on build.ninja / .ninja_log in a shared dir, so it recompiles rather than no-op'ing).
Wall time therefore scales as TP x single-build-time:
    TP=2 ~4 min, TP=4 ~8-10 min, TP=8 >17 min
and at TP=8 the ranks waiting at the c10d store barrier hit their 600s timeout first, so
the engine WEDGES (`waitForInput ... timed out after 600000ms`) and never recovers.
0% GPU util throughout, because the work is nvcc on the CPU.

THE FIX. JitSpec.is_aot is just `aot_path.exists()`, and build_and_load short-circuits to
`self.load(self.aot_path)` when true -- no lock, no ninja, no rebuild, and the SAME
compiled artifact. So build each module once here (single process => no contention) and
copy it into FLASHINFER_AOT_DIR. Compilation stays fully enabled; only the redundant
per-rank rebuild goes away. Idempotent: re-running is a no-op.
"""
import os, shutil, sys, time

REQUIRED = ("trtllm_comm", "sampling")


def _getters():
    """Module-name -> zero-arg callable that forces the JIT build."""
    out = {}
    try:
        from flashinfer.comm.trtllm_ar import get_trtllm_comm_module
        out["trtllm_comm"] = get_trtllm_comm_module
    except Exception as e:                                    # pragma: no cover
        print(f"[aot] WARN cannot import trtllm_comm getter: {e}")
    try:
        from flashinfer.sampling import get_sampling_module
        out["sampling"] = get_sampling_module
    except Exception as e:
        print(f"[aot] WARN cannot import sampling getter: {e}")
    return out


def main():
    from flashinfer.jit import env as jit_env
    aot_dir = jit_env.FLASHINFER_AOT_DIR
    jit_dir = jit_env.FLASHINFER_JIT_DIR
    print(f"[aot] AOT dir: {aot_dir}")
    print(f"[aot] JIT dir: {jit_dir}")

    getters = _getters()
    missing = [m for m in REQUIRED if not os.path.exists(os.path.join(aot_dir, m, f"{m}.so"))]
    if not missing:
        print("[aot] all required modules already AOT -- nothing to do")
    for m in missing:
        g = getters.get(m)
        if g is None:
            print(f"[aot] SKIP {m}: no getter available in this flashinfer build")
            continue
        print(f"[aot] building {m} (single process, no lock contention) ...", flush=True)
        t0 = time.time()
        g()
        print(f"[aot]   built in {time.time() - t0:.1f}s", flush=True)
        src = os.path.join(jit_dir, m, f"{m}.so")
        if not os.path.exists(src):
            print(f"[aot]   ERROR built but {src} not found; not promoting")
            continue
        dst_dir = os.path.join(aot_dir, m)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{m}.so")
        tmp = dst + ".tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)                                  # atomic
        print(f"[aot]   promoted -> {dst} ({os.path.getsize(dst)} bytes)")

    # prove the fast path: every getter must now return ~instantly
    ok = True
    for m in REQUIRED:
        p = os.path.join(aot_dir, m, f"{m}.so")
        if not os.path.exists(p):
            print(f"[aot] MISSING {p}"); ok = False; continue
        g = getters.get(m)
        if g is None:
            continue
        t0 = time.time(); g(); dt = time.time() - t0
        flag = "OK" if dt < 5 else "SLOW (still JIT-ing?)"
        print(f"[aot] {m}: getter returned in {dt:.2f}s -> {flag}")
        ok = ok and dt < 5
    print("[aot] READY" if ok else "[aot] NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
