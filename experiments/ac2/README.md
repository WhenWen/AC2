# AC2 (main run)

`ac2`

**Paper reference:** the AC2 run (blue line) throughout the paper: Fig. 1, right panel; Fig. 2, first three panels (ready fraction and critic error); Fig. 3 and Fig. 6, reference line; Fig. 7 and Fig. 9; Table 4; Table 3; App. B (Fig. 19). Checkpoints of this run are the starting point of the value-function probes (Sec. 4.4, Fig. 4; App. A.4) and of the variants branched at steps 20, 40, 50 and 130.

## Description

Qwen3-4B-Thinking-2507 is trained with AC2 on FineProofs-RL, starting from the base model with an empty replay buffer $\mathcal{B}$, an empty critic buffer of prefix–target pairs and an empty reference-solution bank. The critic $V^\pi_\theta$ shares all parameters with the policy and is queried through the value prompt (App. C); it answers on the grid $\{0, 0.1, \dots, 1\}$ and is fit by next-token prediction to the rounded group-mean value $\mathrm{mean}_j(v_j)$, under both the with-reference and the without-reference prompt. Each step applies the first actor minibatch update, one critic update with a separate AdamW optimizer, and the second actor minibatch update. The critic learning rate is halved (down to a floor) after two consecutive steps in which the critic's parameter displacement exceeds $\sqrt{2}$ times the summed displacement of the two actor updates.

Each step draws $n_{\text{refill}} = 192$ fresh problems, each rolled out once to refill $\mathcal{B}$ (these rollouts are not trained on), and $n_{\text{batch}} = 192$ replayed prefixes, obtained by cutting a stored trajectory at a multiple of 10,000 tokens in $[0, 0.9L]$. From each prefix the policy samples a group of $g = 16$ continuations. Unready problems receive full rollouts scored by the judge. Ready problems receive action chunks of at most $b = 10{,}000$ new tokens whose endpoints are scored by the critic, except for $\lceil \alpha n \rceil$ of the $n$ ready problems ($\alpha = 1/4$), which are audited with full rollouts. A problem becomes ready when the critic error averaged over the last five steps is below $\tau_{\text{global}} = 0.20$, its own latest error is below $\tau_{\text{local}} = 0.18$, the critic has made a nonzero prediction on it, and it has a reference solution in the bank (a rollout with at least 6/7 judge points). The last condition, condition (3) of Sec. 3.1 (`SP_Q_READY_REQUIRE_BANK=1`), was enabled partway through this run: the logged metric `q/ready_require_bank` goes from 0 to 1 at step 42 and stays at 1. The condition only gates new transitions to ready, so problems that became ready during steps 7–41 without a solved reference remained ready. Branches taken at steps 20 and 40 inherit this pre-step-42 readiness state. Advantages are $\hat A_i = v_i - \mathrm{mean}_j(v_j)$ without standard-deviation normalization, the KL coefficient is 0, and the upper clipping parameter $\epsilon_{\text{high}}$ follows the adaptive entropy rule of App. C.

The run uses 4 nodes × 8 GPUs (32 GPUs), with the DeepSeek-V4-Flash judge colocated on the training nodes (one tensor-parallel-8 replica per node). Decoding-cost records cover steps 1–202 (Table 3).

## Configuration

Values are taken from `run_attach.sh` and `runner.py`.

| Parameter | Value | Launch variable / source |
|---|---|---|
| Initialization | `Qwen/Qwen3-4B-Thinking-2507`; empty $\mathcal{B}$, critic buffer and reference bank | `SP_ACTOR_MODEL` (runner default), `SP_REPLAY_COLD_BOOTSTRAP=1` |
| Group size $g$ | 16 | `SP_ROLLOUT_N` (runner default) |
| Action-chunk length $b$ | 10,000 new tokens | `SP_Q_BUDGET_G=10000` |
| Audit fraction $\alpha$ | 1/4 of ready problems, rounded up | `SP_Q_AUDIT_DEN=4` |
| $\tau_{\text{global}}$ / $\tau_{\text{local}}$ | 0.20 / 0.18 | `SP_Q_READY_THRESH_GLOBAL`, `SP_Q_READY_THRESH_PROBLEM` |
| Readiness window | 5 steps | `sp_q_readiness.py` |
| Additional readiness conditions | nonzero critic prediction; reference solution in bank | `SP_Q_REQUIRE_NONZERO=1`; `SP_Q_READY_REQUIRE_BANK=1` (active from step 42, see Description; pass `SP_Q_READY_REQUIRE_BANK=0` to reproduce steps 1–41) |
| $n_{\text{batch}}$ / $n_{\text{refill}}$ | 192 / 192 | `SP_REPLAY_N=192`, `SP_TRAIN_BATCH_SIZE=384`, `SP_SCRATCH_INFLOW_ONLY=1` |
| Replay buffer | global FIFO of 256 trajectories, admission regardless of correctness, uniform over distinct problems | `SP_REPLAY_BOUND=256`, `SP_REPLAY_ADMISSION=ungated`, `SP_REPLAY_GLOBAL_SAMPLING=question` |
| Prefix cut grid / maximum fraction | 10,000 tokens / 0.90 | `SP_REPLAY_CUT_GRAIN`, `SP_REPLAY_CUT_HIGH` |
| Response budget (prefix + new tokens) | 50,000 tokens; prompt 2,048 | `SP_MAX_RESPONSE_LEN=50000` |
| Actor optimizer | LR $2\times10^{-6}$, no warmup, weight decay 0.01, gradient clip 0.3 | `SP_LR=2e-6` |
| Actor minibatches | 2 per step, 96 groups (1,536 continuations) each | `SP_PPO_MINI_BATCH=96` |
| PPO clipping | $\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}^{\text{base}}=0.28$, dual-clip 3 | `runner.py` |
| Adaptive entropy | $H^\star=0.28$, $\delta_H=0.02$, $k\in[-0.08, 0.08]$, initial $k=0.06$ | `SP_AEC_*` |
| KL / entropy bonus | 0 / 0 | `use_kl_loss=False`, `SP_ENTROPY_COEFF=0.0` |
| Critic learning rate | initial $2\sqrt{2}\times10^{-6}$, floor $5\sqrt{2}\times10^{-7}$, halving factor 0.5, ratio threshold $\sqrt{2}$, patience 2 | `SP_Q_LR_*` |
| Critic buffer / pairs per update / valid rewards per target | 1,920 / 768 / 8 | `SP_Q_FIFO_CAP`, `SP_Q_TRAIN_N`, `SP_Q_MIN_VALID` |
| Critic gradient clip | 0.2 | `SP_Q_GRAD_CLIP` |
| Critic prompts | both prompt variants trained; references must be judge-passing | `SP_Q_TRAIN_NOREF=1`, `SP_Q_REF_REQUIRE_PASS=1`, `SP_Q_PROMPT_VARIANT=reward_horizon` |
| Critic decoding | greedy, at most 4 tokens | `sp_q_agent` loop (`sp_agent_loops.yaml`) |
| Training sampling | temperature 0.8, top-$p$ 1, top-$k$ unrestricted | `runner.py` |
| Evaluation | IMO-ProofBench, 16 samples per problem every 10 steps; temperature 0.8, top-$p$ 0.95, top-$k$ 20; 50,000-token limit | `SP_TEST_FREQ=10`, `runner.py` |
| Judge | DeepSeek-V4-Flash at pinned revision `60d8d707`; reward = points/7; pass = points $\ge$ 6 | `SP_JUDGE_REVISION`, `SP_PASS_POINTS_MIN=6` |
| Hardware | 4 nodes × 8 GPUs; rollout tensor parallelism 4 | `submit_4node.sbatch`, `SP_ROLLOUT_TP=4` |
| Step target | 500 (stopped earlier; the paper uses steps up to 202) | `SP_TOTAL_STEPS=500` |

## Files

| File | Description |
|---|---|
| `runner.py` | Training entry point. Builds the verl PPO/Hydra configuration from environment variables, asserts the AC2 switches (replay, critic, interleaved critic update, learning-rate controller), writes the manifest and calls `verl.trainer.main_ppo.run_ppo`. |
| `q_dataset.py` | Training dataset class (`SPQReadinessDataset`): replay-prefix sampling, the empty-buffer fill at the first step, and per-prefix routing to full rollout, action chunk or audit. |
| `sp_agent_loops.yaml` | Agent-loop registry: the prefix-continuation loop (honours the action-chunk cap) and the critic-query loop (greedy, at most 4 tokens). |
| `build_cold_artifacts.py` | Writes the empty replay-buffer seed, critic-buffer seed and reference bank that the loaders require; run once before the first launch. |
| `run_attach.sh` | Launch script: preflight checks, all training environment variables, and the driver launch into a running allocation. |
| `submit_4node.sbatch` | SLURM allocation: 4 nodes × 8 GPUs with a Ray head and workers; writes `alloc_<JOBID>.info` for `run_attach.sh`. |
| `compose_dryrun.sh` | Composes the Hydra configuration with the launch environment on a CPU node, to validate settings before requesting GPUs. |
| `refresh_dashboard.sh` | Render the training dashboard PDF incrementally with `ac2.viz`. |
| `q_monitor_dataset.py` | Builds a per-(step, group) critic-monitoring dataset from the rollout dumps, critic-call dumps and critic-state deltas in `run_data/`. |
| `q_monitor_report.py` | Validates that dataset and prints critic-quality summaries (routing shares, agreement with the judge, audit calibration). |
| `q_mae_analysis.py` | Critic error against problem difficulty and prefix length, with sampling-noise floors. |
| `q_group_structure.py` | Between-group and within-group structure of critic values (intraclass correlation, design effect). |
| `test_ac2.py` | CPU-only tests of the critic learning-rate controller, the critic prompt variants, the reference-solution correctness gate and the empty-buffer bootstrap. `test_context_group_moves_together` expects a 75,000-token response budget and fails against the shipped 50,000-token launch scripts. |
| `q_build_common.py` | Shard-writing and checksum helpers imported by `build_cold_artifacts.py`. |
| `flashinfer_aot_warm.py` | Ahead-of-time compilation of the flashinfer kernels, run before every launch (required with rollout tensor parallelism > 1). |
| `setup.sh` | Per-node runtime environment, sourced by the launch scripts on every node: `setup.sh cuda-compat` adds the CUDA 12.9 forward-compatibility libraries (`AC2_CUDA_COMPAT`) on nodes with older drivers, `setup.sh cuda-toolkit` selects the CUDA toolkit and include paths for the judge's JIT-compiled kernels. |

## Running

`runner.py` is the entry point; it is started by `run_attach.sh` inside a running SLURM allocation.

1. Install the environment with `install.sh` at the repository root. Prepare the data with `python -m ac2.data.prepare_fineproofs --out-dir <DATA_DIR> --split both` and `python -m ac2.data.build_rubric_map --out <DATA_DIR>/rubric_map.json --val-out <DATA_DIR>/val_map.json`. Place `Qwen/Qwen3-4B-Thinking-2507` and `deepseek-ai/DeepSeek-V4-Flash` (revision `60d8d707`) in the Hugging Face cache (`AC2_HF_HOME`); the run is offline (`HF_HUB_OFFLINE=1`). Copy `.env.example` to `.env` and fill in the site settings.
2. Create the empty artifacts once: `python build_cold_artifacts.py --out .`
3. Submit the allocation (`bash scripts/sbatch_env.sh experiments/ac2/submit_4node.sbatch`) and, once it has written `alloc_<JOBID>.info`, attach the driver with `bash run_attach.sh <JOBID> [KEY=VAL ...]`. Trailing `KEY=VAL` arguments override the pinned environment variables.
4. Optionally run `refresh_dashboard.sh` to render the dashboard.

Site-specific values (repository and scratch paths, data directory, Hugging Face cache, SLURM account, partition and QOS) come from `.env`. The attach and allocation scripts also use two helper files in this folder: `setup.sh` (per-node CUDA environment for the judge's JIT-compiled kernels) and, because rollout tensor parallelism is 4, `flashinfer_aot_warm.py` (flashinfer ahead-of-time warm-up run before every launch). The attach script refuses to start if the pinned judge revision is not cached, if the replay seed, critic seed or reference bank is non-empty on a fresh start, or if a checkpoint's world size differs from the allocation's (checkpoints are not resharded, so a run must resume on the same number of GPUs). `python test_ac2.py` runs the CPU tests (no GPU needed).

## Results

From the paper:

- AC2 first exceeds GRPO's peak mean score of 18.50% at 0.79e20 Decoding FLOPs, compared with 1.99e20 for GRPO ($2.5\times$ compute efficiency), surpassing it at step 90 versus step 120 for GRPO.
- Peak mean score 20.57%, compared with 18.50% for GRPO.
- Mean score 16.41% at step 50 and 16.88% at step 60 (reported in the chunk-size and correct-only comparisons).
- The global readiness threshold is first crossed at the end of step 7; the fraction of sampled problems that are ready reaches about 70% by step 200.
- Per-step GPU hours against Decoding FLOPs over 196 complete steps (steps 1–202): $\widehat H = 6.11 + 27.41\,(D/10^{18})$, Pearson $r = 0.874$.
- Where complete records exist, the mean-length cost estimate falls 5.51% below the exact decoding cost for this run.
