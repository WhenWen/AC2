#!/usr/bin/env bash
# submit_4node.sbatch -- 4-node allocation (4 x 8 GPUs) for ac2: starts a Ray head and
# workers and writes alloc_<JOBID>.info for run_attach.sh.
# Submit with scripts/sbatch_env.sh, which fills in the ${AC2_*} site settings.
#SBATCH --job-name=ac2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=${AC2_CPUS_PER_NODE}
#SBATCH --mem=${AC2_MEM_PER_NODE}
#SBATCH --time=${AC2_TIME_LIMIT}
#SBATCH --account=${AC2_ACCOUNT}
#SBATCH --partition=${AC2_PARTITION}
#SBATCH --qos=${AC2_QOS}
#SBATCH --mail-user=${AC2_MAIL_USER}
#SBATCH --output=${AC2_SCRATCH}/logs/ac2-%j.out
#SBATCH --error=${AC2_SCRATCH}/logs/ac2-%j.err

set -xuo pipefail    # deliberately NO -e

REPO_ROOT="${AC2_REPO}"
EXP="$REPO_ROOT/experiments/ac2"
VENV="$REPO_ROOT/.venv"
SETUP_SH="$EXP/setup.sh"
# setup.sh cuda-compat MUST precede ray start on every node: engines inherit Ray's environment,
# so on a node whose driver needs the forward-compatibility libs, every judge engine would
# otherwise die with 'PTX ... unsupported toolchain'.
ACTIVATE="unset ROCR_VISIBLE_DEVICES; source $SETUP_SH cuda-compat; source $VENV/bin/activate; source $SETUP_SH cuda-toolkit"

# SLURM 23.11+ propagates --cpus-per-task into srun via SLURM_TRES_PER_TASK, which then
# conflicts with the explicit --cpus-per-task on the ray srun steps and kills the job
# seconds in. Keep the old variable and drop the new one.
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE}"
unset SLURM_TRES_PER_TASK
export SLURM_CPU_BIND=none

"$VENV/bin/python" -c "import vllm; assert vllm.__version__=='0.23.0', vllm.__version__" \
  || { echo "FATAL: $VENV is not the pinned vllm 0.23.0 stack (run install.sh)"; exit 1; }

mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 --mem=0 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then read -ra ADDR <<<"$head_node_ip"; head_node_ip="${ADDR[1]:-${ADDR[0]}}"; fi
port=6379
ip_head="$head_node_ip:$port"

# (A) Ray HEAD
srun --nodes=1 --ntasks=1 --mem=0 -w "$head_node" \
    bash -c "$ACTIVATE && ray start --head \
        --node-ip-address=$head_node_ip --port=$port \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=8 --block" &
sleep 10

# (B) Ray WORKERS
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i="${nodes_array[$i]}"
    srun --nodes=1 --ntasks=1 --mem=0 -w "$node_i" \
        bash -c "$ACTIVATE && ray start --address=$ip_head \
            --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=8 --block" &
    sleep 5
done

# (C) Publish the address only after every node's GPUs are registered: run_attach.sh reads
# this file, so writing it early would let the driver attach to a half-formed cluster.
WANT_GPUS=$(( SLURM_JOB_NUM_NODES * 8 ))
READY=0
for _i in $(seq 1 60); do
  GOT=$(source "$VENV/bin/activate" && RAY_ADDRESS="$ip_head" ray status 2>/dev/null \
        | sed -n 's@.*/\([0-9.]*\) GPU$@\1@p' | head -1 | cut -d. -f1)
  if [ -n "$GOT" ] && [ "$GOT" -ge "$WANT_GPUS" ] 2>/dev/null; then
    echo "[alloc] ray cluster ready: $GOT/$WANT_GPUS GPUs"; READY=1; break
  fi
  sleep 5
done
if [ "$READY" != "1" ]; then
  echo "[alloc] FATAL: ray reported ${GOT:-0}/$WANT_GPUS GPUs after 300s; no alloc_*.info written"
  sleep infinity
fi

cat > "$EXP/alloc_${SLURM_JOB_ID}.info" <<INFO
JOBID=$SLURM_JOB_ID
HEAD_NODE=$head_node
HEAD_IP=$head_node_ip
RAY_ADDRESS=$ip_head
NODES='${nodes_array[*]}'
INFO
echo "[alloc] ready: nodes (${nodes_array[*]}), Ray at $ip_head; attach with run_attach.sh $SLURM_JOB_ID"
sleep infinity
