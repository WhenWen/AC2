# Judge prompt templates

Prompt templates read by the reward modules in `src/ac2/rewards/`. The files are used verbatim
(`ds4_finegrained_judge.py` verifies the checksum of `finegrained_noref_judge.txt` when it loads it),
so do not edit them in place.

| Template | Used for | Source |
|---|---|---|
| `finegrained_noref_judge.txt` | Training reward (DeepSeek-V4-Flash judge, 0–7 points, no reference solution) | No-reference variant of the ProofAutoGrader prompt of IMO-ProofBench ([google-deepmind/superhuman](https://github.com/google-deepmind/superhuman), Apache-2.0) |
| `imo_proofautograder.txt` | Validation on IMO-ProofBench (with reference solution and grading guidelines) | ProofAutoGrader prompt of IMO-ProofBench ([google-deepmind/superhuman](https://github.com/google-deepmind/superhuman), Apache-2.0) |
| `qednano_rubric_judge.txt` | Rubric-based grading (`qednano_rubric_judge.py`) | Grader prompt of QED-Nano ([CMU-AIRe/QED-Nano](https://github.com/CMU-AIRe/QED-Nano), Apache-2.0), itself adapted from ProofBench |
| `prover_judge.txt` | Binary proof judge (`prover_judge.py`) | This project |
