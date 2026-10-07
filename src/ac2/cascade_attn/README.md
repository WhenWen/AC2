# Grouped cascade attention (optional)

An optional attention kernel for the rollout engine. In AC2, the 16 continuations of a group
share the same replayed prefix. Cascade attention computes attention over that shared prefix once
per group and combines it with each continuation's own suffix attention, instead of attending to
the full prefix separately for every continuation. The result is mathematically equivalent to
standard attention (up to floating-point differences), so it changes rollout speed only, not what
is trained.

| File | Role |
|---|---|
| `vllm_grouped_cascade.py` | vLLM general plugin (entry point `sp_grouped_cascade` in `pyproject.toml`). vLLM loads it in every engine and worker process; `register()` does nothing unless `SP_GROUPED_CASCADE=1`. |
| `grouped_cascade.py` | Group planning and the cascade attention computation (shared-prefix pass plus per-continuation suffix pass, merged by log-sum-exp). |
| `fused_grouped_attn.py` | Fused and overlapped variants of the grouped attention computation. |

The main run does not set `SP_GROUPED_CASCADE=1`; three branch runs in the paper did.
The throughput gain was small, and the other runs use standard attention.
