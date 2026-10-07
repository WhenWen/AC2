# AC2: Actor-Critic with Action Chunking

Code for **[Trust the Critic More](https://arxiv.org/abs/2609.39247)**
(Kaiyue Wen\*, Luke Bailey\*, Arvind Mahankali, Tengyu Ma; Stanford University; \*equal contribution).

[[Paper](https://arxiv.org/abs/2609.39247)]

<p align="center">
  <a href="assets/figure1_animation.mp4"><img src="assets/figure1_animation.gif" width="100%" alt="Advantage estimation in GRPO and AC2, and IMO-ProofBench mean score against decoding FLOPs"></a>
</p>
<p align="center"><em>Left: advantage estimation in GRPO and AC2. Right: IMO-ProofBench mean score against
Decoding FLOPs for AC2, GRPO and Prefix GRPO (Fig. 1 of the paper). <a href="assets/figure1_animation.mp4">MP4</a></em></p>

Standard RL algorithms for language models, such as GRPO, give every token of a long rollout the
same advantage, determined by its terminal reward. AC2 instead assigns credit to **action
chunks**: short continuations of prefixes of earlier trajectories. A learned critic
$V^\pi_\theta$, which shares its parameters with the policy, scores the state reached at the end of
each chunk, so the policy is updated without rolling every trajectory out to completion. Three
design choices make critic-based credit assignment reliable:

1. **Local readiness.** Critic-based updates are used on a problem only when the critic is
   accurate on that problem, in addition to a global accuracy gate.
2. **Reference solutions.** When available, the critic is prompted with a reference solution taken
   from an earlier successful rollout.
3. **Long chunks.** Credit is assigned over chunks of $b = 10{,}000$ new tokens rather than over
   individual tokens.

We train Qwen3-4B-Thinking-2507 on FineProofs-RL and evaluate on IMO-ProofBench. AC2 exceeds
GRPO's peak validation score of 18.5% with $2.5\times$ fewer decoding FLOPs, and reaches a peak of
20.57%.

## Repository layout

```
.
├── install.sh                 # builds the pinned training environment (.venv) with uv
├── pyproject.toml             # installs src/ac2 as the `ac2` package
├── src/
│   ├── verl/                  # verl 0.8.0 (commit 19c6af5) with the AC2 additions; see "Code map"
│   └── ac2/             # data preparation, judge rewards, dashboards, utilities
├── experiments/ac2/           # the AC2 main run (launch scripts, runner, dataset, analysis)
├── scripts/                   # sbatch_env.sh (SLURM submission with site settings); metric export
├── tests/                     # CPU unit tests for the AC2 components
├── assets/                    # Figure 1 animation (MP4 and GIF preview)
├── .env.example               # site configuration template (paths, SLURM accounts); copy to .env
├── LICENSE                    # Apache License 2.0
└── NOTICE                     # attribution for vendored verl and adapted judge prompts
```

`experiments/ac2/README.md` describes the run in the paper's terms, lists its exact
configuration, explains every file, and quotes the results the paper reports for it.

## Code map

The AC2 components are added to the vendored verl as new `sp_*` modules and as changes to its
PPO trainer. They are switched on and configured by `SP_*` environment variables, which the
launch script (`experiments/ac2/run_attach.sh`) sets. With none of them set, the trainer runs verl's GRPO. Paths are
relative to `src/verl/verl/`.

| Paper concept | Code |
|---|---|
| One AC2 step: problem sampling, policy sampling, advantage estimation, actor and critic updates (Algorithm 1) | `trainer/ppo/ray_trainer.py` (the modified verl PPO trainer) |
| Replay buffer $\mathcal{B}$; $n_{\text{refill}}$ fresh problems and $n_{\text{batch}}$ replayed trajectories per step; random prefix cuts | `trainer/ppo/sp_replay.py` (`ReplayHarness`), and `experiments/ac2/q_dataset.py`, which mixes fresh problems with replayed prefixes |
| Continuing a replayed prefix $s$: a chunk of $b$ new tokens on ready problems, a full continuation otherwise | `experimental/agent_loop/prefix_agent_loop.py`, registered in `experiments/ac2/sp_agent_loops.yaml` |
| Critic $V^\pi_\theta$: the policy prompted with the problem, the partial attempt and, when available, a reference solution; it decodes a value on the grid $\{0, 0.1, \dots, 1\}$ greedily | `experimental/agent_loop/sp_q_agent_loop.py` (critic calls); `trainer/ppo/sp_q_readiness.py` (prompt construction, value parsing, critic buffer) |
| Endpoint values $v_i$: the critic's value at the end of the chunk, or the terminal reward when the chunk finishes the trajectory | `ray_trainer.py` (`_sp_q_run_wave_and_stamp`, `_sp_q_reward_site`) |
| Advantage $\hat A_i = v_i - \mathrm{mean}_j(v_j)$; the single-endpoint estimator of AC2 w/o Group & Audit | `trainer/ppo/core_algos.py` (GRPO outcome advantage without std normalization; `compute_sp_segment_advantage`) |
| Global and local readiness ($\tau_{\text{global}}$, $\tau_{\text{local}}$); auditing a fraction $\alpha$ of ready problems with full rollouts | `trainer/ppo/sp_q_readiness.py` (`QHarness.is_ready`, audit-lane selection, readiness bookkeeping) |
| Critic update: fit $V^\pi_\theta(s)$ to $\mathrm{mean}_i(v_i)$ with a second optimizer over the shared weights, interleaved with the actor's PPO minibatches, under a halving learning-rate schedule | `ray_trainer.py` (`_sp_q_interleave_*`); `workers/engine_workers.py` (critic optimizer step); `sp_q_readiness.py` (`QLrController`) |
| A fixed global batch on a GPU count that does not divide it | `trainer/ppo/sp_dp_pad.py` |
| Decoding-FLOPs accounting (App. B) | `scripts/export_paper_metrics.py`; see `scripts/PAPER_METRICS.md` |

Adaptive entropy control (`trainer/ppo/aec.py`) is enabled in all runs and sets the PPO upper
clip bound as described in App. C. The modified trainer also imports a few
additions that are disabled in every run reported here: difficulty-weighted sampling
(`trainer/ppo/difficulty.py`; only its stable problem-ID helper is used), dynamic group size
(`trainer/ppo/sp_dyn_group.py`), and an on-policy distillation teacher
(`trainer/ppo/nitrobrew_teacher.py`, `trainer/distillation/*/nitrobrew_loss.py`).

`src/ac2/` holds the code outside the RL framework:

| Module | Contents |
|---|---|
| `ac2.data` | `prepare_fineproofs` (FineProofs-RL and IMO-ProofBench parquet files); `build_rubric_map` (training rubrics, and validation reference solutions and grading guidelines) |
| `ac2.rewards` | `ds4_finegrained_judge` (DeepSeek-V4-Flash proof judge, 0–7 points), built on `prover_judge` and `qednano_rubric_judge`; prompt templates in `rewards/templates/` |
| `ac2.viz` | Training dashboards (`parse_fig_data`, `merge_fig_data`, `render_dashboard`); see `src/ac2/viz/README.md` |
| `ac2.cascade_attn` | Optional vLLM plugin for grouped cascade attention over the shared prefix of a group (`SP_GROUPED_CASCADE=1`). Off in the main run; the throughput gain was small. See `src/ac2/cascade_attn/README.md` |
| `ac2.utils` | `experiment_utils.manifest_dump` (reproducibility manifest) |
| `ac2.site_env` | Cache and path defaults read by the runners (site values from `.env`) |

## Installation

`install.sh` pins the training stack: Python 3.12, CUDA 12.9, PyTorch 2.11.0+cu129, vLLM
0.23.0+cu129, transformers 5.10.2, FlashInfer 0.6.12, flash-attn 2.8.1, and the vendored verl
installed editable with `--no-deps`. It requires [uv](https://docs.astral.sh/uv/).

```bash
bash install.sh                     # creates .venv; everything except the flash-attn build
INSTALL_FA=sdist bash install.sh    # also compiles flash-attn (run on a GPU node)
```

## Data and judge

```bash
python -m ac2.data.prepare_fineproofs --out-dir $AC2_DATA_DIR --split both
python -m ac2.data.build_rubric_map --out $AC2_DATA_DIR/rubric_map.json \
    --val-out $AC2_DATA_DIR/val_map.json
```

The first command writes the FineProofs-RL training set (`lm-provers/FineProofs-RL`, about 5,200
olympiad proof problems) and the IMO-ProofBench validation set (60 problems). The second writes
the training rubrics and, in `val_map.json`, the IMO-ProofBench reference solutions and grading
guidelines.

Proofs are graded from 0 to 7 by DeepSeek-V4-Flash, which runs on the training nodes alongside
the policy (`src/ac2/rewards/ds4_finegrained_judge.py`), and the reward is `points / 7`.
The training judge sees only the problem and the candidate proof
(`rewards/templates/finegrained_noref_judge.txt`). The validation judge also receives the
reference solution and the problem-specific rubric (`rewards/templates/imo_proofautograder.txt`).

## Running an experiment

The experiment folder `experiments/ac2` contains:

- `runner.py`, the entry point. It builds the verl/Hydra configuration in-process, writes a
  reproducibility manifest (`manifest/`: resolved config, code snapshot, package versions), and
  launches training through `verl.trainer.main_ppo`.
- `submit_<n>node.sbatch`, which requests an allocation of `<n>` nodes × 8 GPUs, starts Ray on it
  and writes `alloc_<JOBID>.info` once every node has joined.
- `run_attach.sh`, the launch script. It checks preconditions, sets the run's `SP_*` variables
  (the hyperparameters) and starts `runner.py` on the allocation's head node.
- `setup.sh`, the per-node runtime environment (CUDA toolkit and compatibility libraries) that the
  launch scripts source on every node, and `flashinfer_aot_warm.py`, a kernel warm-up run before
  each launch.
- `q_dataset.py` (the training dataset class), `sp_agent_loops.yaml` (rollout and
  critic agent loops), and `build_cold_artifacts.py`, which creates the empty replay buffer, critic
  buffer and reference bank a run starts from.

Site-specific settings (repository, scratch and data paths, the Hugging Face cache, SLURM
account, partition, QOS and node resources, the job e-mail address, the W&B entity) are read
from a `.env` file at the repository root:

```bash
cp .env.example .env        # then fill in your paths and SLURM settings
bash scripts/sbatch_env.sh experiments/ac2/submit_4node.sbatch
bash experiments/ac2/run_attach.sh <ALLOC_JOBID>    # once alloc_<ALLOC_JOBID>.info exists
```

The launch scripts load `.env` themselves. SLURM does not expand environment variables in
`#SBATCH` lines, so `scripts/sbatch_env.sh` fills them in before calling `sbatch`. Outputs go to
`experiments/ac2/run_data/` (checkpoints, rollouts, `metrics.jsonl`).

## Experiments and the paper

`experiments/ac2` is the AC2 main run: Qwen3-4B-Thinking-2507 trained on FineProofs-RL from the
base model with empty initial buffers, $g=16$, $b=10{,}000$, auditing $\alpha=1/4$, and global and
local readiness, evaluated on IMO-ProofBench with 16 samples per problem every 10 steps. The KL
coefficient is 0 and the response budget 50,000 tokens. It is the AC2 run in Fig. 1, Fig. 2,
Fig. 3, Table 4 and Fig. 19; figure, table, section and appendix numbers refer to the
[arXiv version](https://arxiv.org/abs/2609.39247). Its README lists the full configuration and
the results the paper reports for it.

## Citation

```bibtex
@article{wen2026trust,
  title   = {Trust the Critic More},
  author  = {Wen, Kaiyue and Bailey, Luke and Mahankali, Arvind and Ma, Tengyu},
  journal = {arXiv preprint arXiv:2609.39247},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.39247}
}
```

## Acknowledgements

This repository builds on [verl](https://github.com/volcengine/verl) (Apache-2.0). The vendored
copy in `src/verl` keeps its original license and notice files, and `src/verl/README.md` lists the
files we added or modified. The judge prompts in `src/ac2/rewards/templates/` are adapted from the
ProofAutoGrader prompt of [IMO-ProofBench](https://github.com/google-deepmind/superhuman) and the
grader prompt of [QED-Nano](https://github.com/CMU-AIRe/QED-Nano) (both Apache-2.0); see `NOTICE`.

## License

This code is released under the [Apache License 2.0](LICENSE).
