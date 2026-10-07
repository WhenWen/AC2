# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by the AC2 authors (2026) to implement AC2; see src/verl/README.md for the list of changed files.

import math
import unittest

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import (
    calculate_debug_metrics,
    calculate_train_inference_mismatch_metrics,
)


class TestMetrics(unittest.TestCase):
    def test_calculate_debug_metrics(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                        [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                        [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
                    ]
                ),
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)
        print(metrics)
        assert metrics["training/rollout_probs_diff_valid"] == 1


class TestTrainInferenceMismatchMetrics(unittest.TestCase):
    """Train/inference logprob mismatch diagnostics (self-play addition)."""

    @staticmethod
    def _batch(actor_log_probs, rollout_log_probs, response_mask):
        bsz, resp_len = actor_log_probs.shape
        return DataProto.from_dict(
            {
                "old_log_probs": actor_log_probs,
                "rollout_log_probs": rollout_log_probs,
                "response_mask": response_mask,
                "responses": torch.zeros(bsz, resp_len, dtype=torch.long),
            }
        )

    def test_perfect_agreement_is_zero_diff(self):
        lp = torch.log(torch.tensor([[0.5, 0.25, 0.1], [0.8, 0.2, 0.4]]))
        mask = torch.ones(2, 3, dtype=torch.long)
        m = calculate_train_inference_mismatch_metrics(self._batch(lp, lp.clone(), mask))

        assert m["train_inference/old_available"] == 1
        assert m["train_inference/logprob_diff_abs_mean"] == 0.0
        assert m["train_inference/logprob_diff_abs_max"] == 0.0
        assert m["train_inference/prob_diff_abs_max"] == 0.0
        # same distribution -> identical per-token NLL, and perplexity = exp(loss)
        assert math.isclose(
            m["train_inference/trainer_next_token_loss"],
            m["train_inference/rollout_next_token_loss"],
            rel_tol=1e-6,
        )
        assert math.isclose(
            m["train_inference/trainer_next_token_perplexity"],
            math.exp(m["train_inference/trainer_next_token_loss"]),
            rel_tol=1e-6,
        )

    def test_known_diff_values(self):
        # actor prob 0.5 vs rollout prob 0.25 everywhere -> logprob diff = ln2, prob diff = 0.25
        actor = torch.log(torch.tensor([[0.5, 0.5]]))
        rollout = torch.log(torch.tensor([[0.25, 0.25]]))
        mask = torch.ones(1, 2, dtype=torch.long)
        m = calculate_train_inference_mismatch_metrics(self._batch(actor, rollout, mask))

        assert math.isclose(m["train_inference/logprob_diff_abs_mean"], math.log(2), rel_tol=1e-5)
        assert math.isclose(m["train_inference/prob_diff_abs_mean"], 0.25, rel_tol=1e-5)
        assert math.isclose(m["train_inference/prob_diff_abs_p99"], 0.25, rel_tol=1e-5)
        assert math.isclose(m["train_inference/rollout_next_token_loss"], -math.log(0.25), rel_tol=1e-5)

    def test_masked_tokens_are_ignored(self):
        # token 1 disagrees wildly but is masked; only token 0 (perfect agreement) counts
        actor = torch.log(torch.tensor([[0.5, 0.9]]))
        rollout = torch.log(torch.tensor([[0.5, 0.01]]))
        mask = torch.tensor([[1, 0]], dtype=torch.long)
        m = calculate_train_inference_mismatch_metrics(self._batch(actor, rollout, mask))

        assert m["train_inference/old_available"] == 1
        assert m["train_inference/logprob_diff_abs_max"] == 0.0
        assert m["train_inference/prob_diff_abs_max"] == 0.0

    def test_all_masked_is_unavailable(self):
        lp = torch.log(torch.tensor([[0.5, 0.5]]))
        mask = torch.zeros(1, 2, dtype=torch.long)
        m = calculate_train_inference_mismatch_metrics(self._batch(lp, lp.clone(), mask))
        assert m == {"train_inference/old_available": 0}

    def test_missing_rollout_log_probs_is_unavailable(self):
        lp = torch.log(torch.tensor([[0.5, 0.5]]))
        data = DataProto.from_dict(
            {
                "old_log_probs": lp,
                "response_mask": torch.ones(1, 2, dtype=torch.long),
                "responses": torch.zeros(1, 2, dtype=torch.long),
            }
        )
        m = calculate_train_inference_mismatch_metrics(data)
        assert m == {"train_inference/old_available": 0}


if __name__ == "__main__":
    unittest.main()
