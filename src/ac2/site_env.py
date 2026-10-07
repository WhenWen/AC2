"""Site configuration defaults for the experiment runners.

Site-specific paths come from the environment (the repository's .env; see .env.example). The
runners read cache and Hugging Face locations from here instead of hardcoding them, and every
value can still be overridden by its own variable at the point of use (e.g. ``$HF_HUB_CACHE``),
so this only sets *defaults*.

Shell usage:  eval "$(python -m ac2.site_env export)"
  -> exports HF_HOME, HF_HUB_CACHE, SP_CACHE_ROOT and WANDB_MODE (only non-empty ones), without
     clobbering variables that are already set.
"""

import os
import sys


def _env(var: str, default: str = "") -> str:
    return os.environ.get(var) or default


def site_config() -> dict:
    """Return the site defaults derived from the AC2_* variables."""
    scratch = _env("AC2_SCRATCH")
    hf_home = _env("AC2_HF_HOME", os.path.join(scratch, "hf_home") if scratch else "")
    return {
        "hf_home": hf_home,
        # empty -> HF_HOME/hub
        "hf_hub_cache": _env("AC2_HF_HUB_CACHE"),
        "cache_root": os.path.join(scratch, ".cache") if scratch else "",
        "data_dir": _env("AC2_DATA_DIR"),
        "wandb_mode": _env("AC2_WANDB_MODE", "offline"),
        # 80 GB GPUs: one 50k-token sequence per backward (98304 OOMs the actor update), and
        # 96 concurrent rollout sequences per engine.
        "ppo_max_token_len": 51200,
        "rollout_max_num_seqs": 96,
    }


_ENV_KEYS = {  # site-config key -> env var it feeds
    "hf_hub_cache": "HF_HUB_CACHE",
    "hf_home": "HF_HOME",
    "cache_root": "SP_CACHE_ROOT",
    "wandb_mode": "WANDB_MODE",
}


def _print_shell_exports() -> None:
    """Emit `export VAR=val` lines for a launcher to eval. Skips empty values and never clobbers a
    var already set in the environment (so a launcher/CLI override always wins)."""
    cfg = site_config()
    for key, env in _ENV_KEYS.items():
        val = cfg.get(key, "")
        if val and not os.environ.get(env):
            print(f'export {env}="{val}"')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        _print_shell_exports()
    else:
        import json
        print(json.dumps(site_config(), indent=2))
