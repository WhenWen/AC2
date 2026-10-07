#!/usr/bin/env bash
# install.sh — build the AC2 training environment with uv.
#
# Pinned stack (CUDA 12.9): Python 3.12, torch 2.11.0+cu129, vllm 0.23.0+cu129,
# transformers 5.10.2, flashinfer 0.6.12, triton 3.6.0, flash-attn 2.8.1 (SOURCE
# build — no torch-2.11 wheel exists), flash-linear-attention 0.5.0, plus vendored
# verl (editable, --no-deps). This stack supports a colocated DeepSeek-V4-Flash judge.
#
# The script creates (or rebuilds in place) a single venv, `.venv`. Rebuild it only
# when no run is executing from it: a live run's Ray spawns fresh workers from the
# venv path mid-run, so swapping it under a run corrupts the run. Rebuilds are fast
# once uv's wheel cache is warm (incl. the flash-attn built wheel).
#
# Prerequisite: CUDA 12.9 driver/toolkit. Wheels install on a GPU-less login node;
# the ONE exception is flash-attn, which must be COMPILED on a compute node
# (login nodes OOM-kill the nvcc jobs) — see INSTALL_FA below.
# Usage:
#   bash install.sh                     # everything except flash-attn build
#   INSTALL_FA=sdist bash install.sh    # + build flash-attn in-place (compute node!)
# On clusters: run the base install on the login node, then
#   srun --overlap --jobid=<alloc_jobid> -w <node> bash -c 'INSTALL_FA=sdist FA_ONLY=1 bash install.sh'
#
# Override VENV_DIR / VERL_DIR to relocate the venv or the verl checkout.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
VERL_DIR="${VERL_DIR:-$REPO_ROOT/src/verl}"
INSTALL_FA="${INSTALL_FA:-probe}"   # probe (wheel if one appears) | sdist (source build) | skip
FA_ONLY="${FA_ONLY:-0}"             # 1 = only do the flash-attn step into an EXISTING venv

# The exact vLLM release wheel this stack was validated with
# (cp38-abi3: one wheel for all pythons >= 3.8).
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256="8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"

command -v uv >/dev/null || { echo "ERROR: uv not on PATH (https://docs.astral.sh/uv/)" >&2; exit 1; }
[ -d "$VERL_DIR" ] || { echo "ERROR: vendored verl not found at $VERL_DIR" >&2; exit 1; }

install_flash_attn() {
  # flash-attn 2.8.1 for torch 2.11: no prebuilt wheel exists (Dao-AILab releases
  # stop at torch2.10 for 2.8.x). Probe first in case one appears,
  # else compile the sdist. The build needs a COMPUTE node (login OOM-kills cicc/nvcc)
  # with the system CUDA 12.9 toolkit; ~15 min at MAX_JOBS=64 on 96 cores.
  case "$INSTALL_FA" in
    skip) echo "[install] flash-attn: SKIPPED (INSTALL_FA=skip)"; return 0 ;;
  esac
  local fa_ver=2.8.1
  for abi in TRUE FALSE; do
    local u="https://github.com/Dao-AILab/flash-attention/releases/download/v${fa_ver}/flash_attn-${fa_ver}%2Bcu12torch2.11cxx11abi${abi}-cp312-cp312-linux_x86_64.whl"
    if curl -sfIL --max-time 15 "$u" >/dev/null 2>&1; then
      echo "[install] flash-attn: found prebuilt wheel (abi$abi)"
      uv pip install --python "$VENV_DIR/bin/python" "$u"
      return 0
    fi
  done
  if [ "$INSTALL_FA" != "sdist" ]; then
    echo "[install] flash-attn: NO torch-2.11 wheel; NOT building (login node?). Run the"
    echo "          compute-node step:  srun --overlap --jobid=<alloc_jobid> -w <node> \\"
    echo "            bash -c 'INSTALL_FA=sdist FA_ONLY=1 VENV_DIR=$VENV_DIR bash $REPO_ROOT/install.sh'"
    return 0
  fi
  echo "[install] flash-attn: building ${fa_ver} from sdist (compute node expected)"
  uv pip install --python "$VENV_DIR/bin/python" -q ninja psutil packaging setuptools wheel einops
  local bdir="${FA_BUILD_DIR:-$REPO_ROOT/.fa_build}"
  mkdir -p "$bdir"
  if [ ! -f "$bdir/flash_attn-${fa_ver}.tar.gz" ]; then
    local sdist_url
    sdist_url=$(curl -fsS "https://pypi.org/pypi/flash-attn/${fa_ver}/json" \
      | "$VENV_DIR/bin/python" -c "import json,sys; d=json.load(sys.stdin); print([u['url'] for u in d['urls'] if u['packagetype']=='sdist'][0])")
    curl -fsSL -o "$bdir/flash_attn-${fa_ver}.tar.gz" "$sdist_url"
  fi
  # Respect a pre-set CUDA_HOME (where /usr/local/cuda-12.9 does not exist, load the site's
  # CUDA 12.9 toolkit module, which sets CUDA_HOME, before running this).
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export PATH="$CUDA_HOME/bin:$PATH"
  local ns="$VENV_DIR/lib/python3.12/site-packages/nvidia"
  export CPATH="$ns/cublas/include:$ns/curand/include:$ns/cusolver/include:$ns/cusparse/include:$ns/cuda_nvrtc/include:$ns/nvjitlink/include"
  export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS="${MAX_JOBS:-64}" FLASH_ATTENTION_FORCE_BUILD=TRUE
  uv pip install --offline --python "$VENV_DIR/bin/python" --no-deps --no-build-isolation \
    "$bdir/flash_attn-${fa_ver}.tar.gz" 2>/dev/null \
    || uv pip install --python "$VENV_DIR/bin/python" --no-deps --no-build-isolation \
         "$bdir/flash_attn-${fa_ver}.tar.gz"
  "$VENV_DIR/bin/python" -c "import flash_attn; print('VERIFY flash_attn', flash_attn.__version__)"
}

if [ "$FA_ONLY" = "1" ]; then
  [ -x "$VENV_DIR/bin/python" ] || { echo "ERROR: FA_ONLY=1 but no venv at $VENV_DIR" >&2; exit 1; }
  install_flash_attn
  exit 0
fi

# In-place update support (see the header): --clear an existing venv, but
# REFUSE if any process is running from it (a live run's Ray spawns fresh workers
# from this path mid-run). INSTALL_FORCE=1 overrides the guard.
UV_VENV_ARGS=()
if [ -d "$VENV_DIR" ]; then
  if pgrep -f "$VENV_DIR/" >/dev/null 2>&1 && [ "${INSTALL_FORCE:-0}" != "1" ]; then
    echo "ERROR: processes are running from $VENV_DIR (pgrep -f '$VENV_DIR/'):" >&2
    pgrep -af "$VENV_DIR/" | head -5 >&2
    echo "Drain them first, or INSTALL_FORCE=1 to override." >&2
    exit 1
  fi
  echo "[install] existing venv at $VENV_DIR -> in-place rebuild (--clear)"
  UV_VENV_ARGS+=(--clear)
fi
uv venv --python 3.12 "${UV_VENV_ARGS[@]}" "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Core GPU stack. The vLLM release wheel is installed by exact URL + sha (the
# validated artifact); torch/flashinfer/triton pins come from the same validated
# runtime. transformers 5.10.2 was verified to parse the DeepSeek-V4-Flash
# config — do NOT bump past 5.12.x blindly: 5.13.0's deepseek_v4 config
# validator is broken.
uv pip install --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
    "torch==2.11.0+cu129" \
    transformers==5.10.2 \
    flashinfer-python==0.6.12 \
    flashinfer-cubin==0.6.12 \
    triton==3.6.0

# verl runtime deps — explicit because verl installs --no-deps below (its
# setup.py caps vllm too low, which would fight the pin above).
uv pip install --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    accelerate \
    datasets \
    peft \
    hf-transfer \
    "pyarrow>=15" \
    pandas \
    "tensordict>=0.8,<=0.10,!=0.9" \
    torchdata \
    "ray[default]" \
    codetiming \
    hydra-core \
    pylatexenc \
    qwen-vl-utils \
    wandb \
    dill \
    pybind11 \
    liger-kernel \
    mathruler \
    "nvidia-ml-py>=12.560.30" \
    "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" \
    "pydantic>=2.9" \
    "grpcio>=1.62.1"

# vendored verl, editable + --no-deps (our pins win over verl's).
uv pip install --no-deps -e "$VERL_DIR"

# flash-linear-attention (qwen3_next deltanet fast kernel; triton-based, torch-2.11 ok).
uv pip install flash-linear-attention==0.5.0

# math-verify (HF LaTeX/Sympy answer grader) — the DeepScaleR-style 1/0 reward.
uv pip install math-verify

# ac2 extras (aiohttp = judge reward client; matplotlib = dashboards).
uv pip install aiohttp matplotlib numpy pyyaml

# cupy — verl's NCCL checkpoint-engine backend (colocated weight publish).
# NOTE: uv venvs ship NO pip — always add packages via `uv pip install --python <venv>`.
uv pip install --no-deps cupy-cuda12x fastrlock

# our ac2 package, editable + --no-deps (deps are the pinned stack above).
uv pip install --no-deps -e "$REPO_ROOT"

# flash-attn last (may be a compute-node step; see function above).
install_flash_attn

# verify — imports don't need a GPU, so this is safe on a login node.
python - << 'PY'
import importlib.metadata as md
import torch, vllm, transformers, flashinfer, fla  # noqa: F401
assert vllm.__version__ == "0.23.0", vllm.__version__
assert md.version("vllm") == "0.23.0+cu129", md.version("vllm")
assert torch.__version__ == "2.11.0+cu129" and torch.version.cuda == "12.9"
assert transformers.__version__ == "5.10.2"
assert flashinfer.__version__ == "0.6.12" and md.version("triton") == "3.6.0"
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer  # noqa: F401  (DS4 judge support)
print("VERIFY torch", torch.__version__, "| vllm", md.version("vllm"),
      "| tf", transformers.__version__, "| flashinfer", flashinfer.__version__, "| fla ok")
try:
    import flash_attn
    print("VERIFY flash_attn", flash_attn.__version__)
except ImportError:
    print("VERIFY flash_attn MISSING — run the compute-node FA step before training")
PY
python -c "import verl; print('VERIFY verl import ok at', verl.__file__)"
python -c "import ac2; print('VERIFY ac2 import ok at', ac2.__file__)"
python -c "import verl.checkpoint_engine.nccl_checkpoint_engine; from verl.checkpoint_engine.base import CheckpointEngineRegistry as R; \
print('VERIFY nccl checkpoint engine registered:', R.get('nccl').__name__)"

echo "[install] DONE. Activate with:  source $VENV_DIR/bin/activate"
