#!/usr/bin/env bash
# setup.sh -- per-node runtime environment for this experiment. Source it; do not execute it.
#
#   source setup.sh cuda-compat     before activating the virtual environment
#   source setup.sh cuda-toolkit    after activating the virtual environment
#
# The launch scripts source it in every compute-node process: the Ray daemons started by
# submit_<N>node.sbatch and the driver (and the flashinfer warm-up) started by run_attach.sh.
# Site paths come from the repository's .env (see .env.example).
case "${1:-}" in
cuda-compat)
# Before activating the venv.
#
# (1) OLDER DRIVERS. The cu129 stack needs a driver that supports CUDA 12.9. On a node with an
#     older driver (< 575), a plain torch op dies with "CUDA error: the provided PTX was
#     compiled with an unsupported toolchain" unless the CUDA 12.9 forward-compatibility libs
#     ($AC2_CUDA_COMPAT) are on LD_LIBRARY_PATH. Forward compatibility only applies when the
#     kernel driver is OLDER than those libs, so on a newer driver they break CUDA outright
#     (torch.cuda.is_available() False, which surfaces later as verl building
#     "cpu:gloo,cpu:nccl" -> "ValueError: Duplicate device type cpu"). One allocation can mix
#     driver versions, so this is decided per node.
#
# (2) JIT HEADERS. flashinfer/DeepGEMM JIT-compile at engine init and need the CUDA C++ core
#     headers; without them the judge dies with "fatal error: nv/target: No such file or
#     directory". They ship in the VENV (nvidia/cuda_cccl/include), which the cuda-toolkit
#     section does not add -- its list covers cublas/curand/cufft/cusolver/cusparse/nvrtc/nvjitlink only.
_COMPAT="${AC2_CUDA_COMPAT:-}"
_drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$_COMPAT" ] && [ -n "$_drv" ] && [ "$_drv" -lt 575 ] 2>/dev/null && [ -f "$_COMPAT/libcuda.so.1" ]; then
  export LD_LIBRARY_PATH="$_COMPAT:${LD_LIBRARY_PATH:-}"
  echo "[cuda-shim] $(hostname): driver $_drv < 575 -> CUDA 12.9 forward-compat libs"
else
  echo "[cuda-shim] $(hostname): driver ${_drv:-unknown} -> native CUDA (no compat)"
fi

_CH=/usr/local/cuda-12.9
[ -x "$_CH/bin/nvcc" ] || _CH=/usr/local/cuda
export CUDA_HOME="$_CH" CUDA_PATH="$_CH" CUDACXX="$_CH/bin/nvcc" TRTLLM_DG_NVCC_COMPILER="$_CH/bin/nvcc"
case ":$PATH:" in *":$_CH/bin:"*) : ;; *) export PATH="$_CH/bin:$PATH" ;; esac

_NS="${AC2_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/.venv/lib/python3.12/site-packages/nvidia"
_INC=()
for c in cuda_cccl cuda_runtime cublas curand cufft cusolver cusparse cusparselt cuda_nvrtc nvjitlink; do
  [ -d "$_NS/$c/include" ] && _INC+=("$_NS/$c/include")
done
[ -d "$_CH/include" ] && _INC+=("$_CH/include")
if [ "${#_INC[@]}" -gt 0 ]; then
  export CPATH="$(IFS=:; echo "${_INC[*]}")${CPATH:+:$CPATH}"
  export NVCC_PREPEND_FLAGS="$(printf -- '-I%s ' "${_INC[@]}")${NVCC_PREPEND_FLAGS:-}"
fi

# flashinfer writes its JIT workspace under ~/.cache by default, which fails with Errno 122 on a
# home directory at quota. The Ray runtime_env sets this for the workers, but the driver-srun
# AOT preflight runs OUTSIDE that env, so it must be set here.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${AC2_SCRATCH}/.cache/ds4_vllm_023/flashinfer}"

# HOME REDIRECT. When the home directory is not writable or ~/.cache is missing, libraries that
# call expanduser("~/.cache") directly ignore XDG_CACHE_HOME, so setting the cache vars is not
# enough: at TP=4 a full home killed Worker_TP0 with Errno 122 mid-init. Pointing HOME at
# scratch makes every ~-based default land somewhere writable. Scoped to compute-node engine
# processes; the login shell keeps the real HOME.
if [ -n "${SP_SCRATCH_HOME:-}" ] || [ ! -w "${HOME:-/}" ] || [ ! -e "$HOME/.cache" ]; then
  export HOME="${SP_SCRATCH_HOME:-${AC2_SCRATCH}/home}"
  mkdir -p "$HOME/.cache" 2>/dev/null || true
fi
  ;;
cuda-toolkit)
#
# At engine init, flashinfer/vLLM JIT-compile the DS4 FP8 block-scale GEMM kernels whenever
# the warm JIT cache misses (e.g. after a .venv rebuild, or a kernel/shape not previously
# warmed). nvcc then needs the CUDA toolkit AND the pip `nvidia/*/include` headers
# (cublasLt.h, curand, cusolver, ...) on its include path; without them the build fails with
# `fatal error: cublasLt.h: No such file`.
#
# Source this AFTER activating the venv (it reads the venv's site-packages/nvidia includes).
# Evaluated per-node so it picks whatever CUDA toolkit that node actually has. Idempotent.
_VENV="${VIRTUAL_ENV:-${AC2_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/.venv}"
# Toolkit preference: $CUDA_HOME_PREF -> system /usr/local/cuda-12.9 -> a user-space 12.9
# install in $HOME -> whatever /usr/local/cuda points at. The DeepGEMM DS4 fp8 kernels need
# nvcc >= 12.9 ("NVCC compilation failed" with 12.6), and an install on an inode-limited
# filesystem can end up truncated -> "fatal error: cuda/std/utility".
if [ -n "${CUDA_HOME_PREF:-}" ] && [ -x "$CUDA_HOME_PREF/bin/nvcc" ]; then _CH="$CUDA_HOME_PREF"
elif [ -x /usr/local/cuda-12.9/bin/nvcc ]; then _CH=/usr/local/cuda-12.9
elif [ -x "$HOME/cuda-12.9/bin/nvcc" ]; then _CH="$HOME/cuda-12.9"
else _CH=/usr/local/cuda; fi
export CUDA_HOME="$_CH" CUDA_PATH="$_CH" CUDAToolkit_ROOT="$_CH" CUDACXX="$_CH/bin/nvcc" TRTLLM_DG_NVCC_COMPILER="$_CH/bin/nvcc"
case ":$PATH:" in *":$_CH/bin:"*) : ;; *) export PATH="$_CH/bin:$PATH" ;; esac
_NS="$_VENV/lib/python3.12/site-packages/nvidia"
_INC=()
for c in cublas curand cufft cusolver cusparse cusparselt cuda_nvrtc nvjitlink; do
  [ -d "$_NS/$c/include" ] && _INC+=("$_NS/$c/include")
done
if [ "${#_INC[@]}" -gt 0 ]; then
  export CPATH="$(IFS=:; echo "${_INC[*]}")"
  export NVCC_PREPEND_FLAGS="$(printf -- '-I%s ' "${_INC[@]}")"
fi
  ;;
*)
  echo "usage: source setup.sh cuda-compat|cuda-toolkit" >&2
  return 1 2>/dev/null || exit 1
  ;;
esac
