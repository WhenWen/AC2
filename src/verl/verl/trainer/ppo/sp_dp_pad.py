"""sp_dp_pad -- run a fixed global batch shape on a data-parallel size that does not divide it.

Why this exists (the AC2 batch shape on 7 nodes = 56 GPUs):

    pre-drop rows   192 inflow x1 + 192 trained x16 = 3264      3264 % 56 = 16
    trained rows    192 x 16                        = 3072      3072 % 56 = 48
    PPO minibatch    96 x 16                        = 1536      1536 % 56 = 24

None of the three has a factor of 7, so no Ulysses / FSDP re-split makes them divide: every
dp-dispatched call on the actor (the seqlen balancer, compute_log_prob, compute_ref_log_prob,
update_actor) asserts `len(batch) % dp == 0`, and the worker's minibatch iterator asserts both
`mini_batch_size % dp == 0` and `per_rank_rows % per_rank_mini == 0`.

The fix keeps the SCIENCE shape byte-for-byte (384 / 192 / 96, two PPO minibatches of exactly
1536 real sequences, the interleaved order PPO(M1) -> Q -> PPO(M2)) and only changes how rows are laid
onto ranks:

  * PAD.  Right before a dp-dispatched call, the batch is padded to a multiple of dp with
    duplicates of its SHORTEST row whose `response_mask` is zeroed. The PPO loss is
    `token-mean` with an all-reduced global token count (agg_loss / forward_backward_batch), so
    a zero-mask row contributes exactly 0 to the numerator AND the denominator: the gradient
    is identical to the unpadded batch, not approximately. Forward-only calls (old/ref
    log-prob) simply have the pad rows' outputs dropped. Pad rows are marked in
    `non_tensor_batch[PAD_KEY]`; nothing between the dispatch boundaries ever sees them --
    the judge, the replay admission, the Q reward site, advantage estimation, the rollout dump
    and the checkpoint all run on the real rows only.

  * BALANCE.  The seqlen balancer runs on the padded batch (so its equal-size partitions are
    exactly the dispatch chunks) and the pad POSITIONS in the balanced order are remembered, so
    that re-padding for the log-prob calls puts each pad back into the chunk the balancer gave
    it. Losing the plan (a mid-step resume) only degrades balance, never correctness.

  * MINIBATCH IDS.  update_actor cannot split each rank's rows positionally into minibatches:
    3080 padded / 56 = 55 rows per rank does not halve, and 1536 real rows per minibatch is
    not a multiple of 56. Instead the DRIVER assigns every row (pad rows included) a minibatch
    id (`sp_minibatch_id`, a per-row tensor that rides through the ordinary chunked dispatch),
    apportioned so that every minibatch holds exactly `ppo_mini_batch_size` REAL sequences
    globally and every rank holds at least one row of every minibatch (so all ranks run the
    same number of optimizer steps and the interleaved Q step still lands after minibatch 1).
    The worker builds its minibatches from the ids (tensordict_utils.make_iterator_by_ids).

Gated by SP_DP_PAD=1 and additionally inert whenever the shape already divides: an 8-node run
of the same config takes the pre-existing code path unchanged, and a run that did NOT opt in
still fails loudly on a non-divisible shape (a silent pad would mask a misconfiguration).

Everything at module top is pure Python so the apportionment can be unit-tested without torch;
the DataProto helpers import torch/numpy lazily.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

PAD_KEY = "sp_dp_pad"
MINIBATCH_ID_KEY = "sp_minibatch_id"


def enabled() -> bool:
    return os.environ.get("SP_DP_PAD", "0") in ("1", "true", "True")


def pad_count(n_rows: int, dp_size: int) -> int:
    """Rows to append so that n_rows + pad is a multiple of dp_size (0 when it already is)."""
    assert dp_size >= 1, dp_size
    return (-int(n_rows)) % int(dp_size)


def minibatch_ids(pad_flags: Sequence[int], dp_size: int, num_mini_batch: int) -> list[int]:
    """Driver-side minibatch membership for a padded, dp-chunked batch.

    `pad_flags[i]` is 1 for a pad row, 0 for a real row; len must be a multiple of `dp_size`
    and chunk c (= rank c's rows after the equal chunk dispatch) is positions
    [c*slots, (c+1)*slots). Returns one minibatch id per row such that

      * every minibatch holds exactly n_real / num_mini_batch REAL rows globally,
      * every chunk holds >= 1 REAL row of every minibatch (same optimizer-step count on all
        ranks; the interleaved Q step keeps its slot),
      * a chunk's real rows are apportioned as evenly as its count allows (floor / floor+1),
        with the +1 remainders handed to minibatches round-robin across chunks -- the total
        number of remainders is a multiple of num_mini_batch, so the cycle closes exactly,
      * pad rows are spread over the minibatches that hold the fewest rows in their chunk.
    """
    flags = [int(bool(f)) for f in pad_flags]
    n = len(flags)
    dp = int(dp_size)
    k = int(num_mini_batch)
    assert dp >= 1 and k >= 1, (dp, k)
    assert n % dp == 0, f"padded batch of {n} rows is not a multiple of dp={dp}"
    slots = n // dp
    n_real = n - sum(flags)
    assert n_real % k == 0, (
        f"{n_real} real rows cannot form {k} equal minibatches (the runner's "
        f"REPLAY_N % PPO_MINI == 0 assertion should have caught this)"
    )

    real_per_chunk = [slots - sum(flags[c * slots : (c + 1) * slots]) for c in range(dp)]
    bad = [c for c, r in enumerate(real_per_chunk) if r < k]
    assert not bad, (
        f"chunks {bad} hold fewer than {k} real rows ({[real_per_chunk[c] for c in bad]}); every "
        f"rank needs at least one real row per minibatch"
    )

    # per-chunk apportionment: floor share to every minibatch, remainders round-robin
    alloc: list[list[int]] = []
    cursor = 0
    for r in real_per_chunk:
        base, rem = divmod(r, k)
        counts = [base] * k
        for j in range(rem):
            counts[(cursor + j) % k] += 1
        cursor = (cursor + rem) % k
        alloc.append(counts)
    assert cursor == 0, cursor  # sum(rem) is a multiple of k because n_real is
    for j in range(k):
        got = sum(a[j] for a in alloc)
        assert got == n_real // k, (j, got, n_real // k)

    ids = [0] * n
    for c in range(dp):
        counts = list(alloc[c])
        real_pos = [c * slots + i for i in range(slots) if flags[c * slots + i] == 0]
        pad_pos = [c * slots + i for i in range(slots) if flags[c * slots + i] == 1]
        j = 0
        for p in real_pos:
            while counts[j] == 0:
                j += 1
            ids[p] = j
            counts[j] -= 1
        # pads: fill the minibatches that have the fewest rows in this chunk first
        per_mb = list(alloc[c])
        for p in pad_pos:
            j = min(range(k), key=lambda m: (per_mb[m], m))
            ids[p] = j
            per_mb[j] += 1
    return ids


def describe(n_predrop: int, n_trained: int, mini_rows: int, dp_size: int) -> str:
    """One line for the launcher / dry-run log: what the padding will do at this dp."""
    p1, p2 = pad_count(n_predrop, dp_size), pad_count(n_trained, dp_size)
    k = n_trained // mini_rows if mini_rows else 0
    active = (p1 or p2 or (mini_rows % dp_size != 0))
    return (
        f"[sp_dp_pad] dp={dp_size}: pre-drop {n_predrop} rows -> +{p1} pad "
        f"({(n_predrop + p1) // dp_size}/rank); trained {n_trained} rows -> +{p2} pad "
        f"({(n_trained + p2) // dp_size}/rank); {k} driver-assigned minibatches of {mini_rows} "
        f"real seqs; {'ACTIVE' if active else 'inert (shape divides)'}"
    )


# --------------------------------------------------------------------------------------------
# DataProto helpers (torch / numpy imported lazily)
# --------------------------------------------------------------------------------------------

def _shortest_row(batch) -> int:
    if batch.batch is not None and "attention_mask" in batch.batch.keys():
        return int(batch.batch["attention_mask"].sum(-1).argmin().item())
    return 0


def pad_flags(batch):
    """np.ndarray[int] of pad markers (zeros when the batch was never padded)."""
    import numpy as np

    v = batch.non_tensor_batch.get(PAD_KEY)
    if v is None:
        return np.zeros(len(batch), dtype=np.int64)
    return np.asarray(v).astype(np.int64)


def real_mask(batch):
    return pad_flags(batch) == 0


def plan_fits(positions, n_real: int, dp_size: int) -> bool:
    """A remembered pad plan is reusable only for the SAME real-row count."""
    if positions is None:
        return False
    n_pad = pad_count(n_real, dp_size)
    return len(positions) == n_pad and (n_pad == 0 or int(max(positions)) < n_real + n_pad)


def pad_batch(batch, dp_size: int, positions=None):
    """Return a NEW DataProto padded to a multiple of dp_size (or `batch` itself if it divides).

    Pad rows duplicate the shortest real row (cheapest forward) with `response_mask` zeroed and
    `non_tensor_batch[PAD_KEY] = 1`. With `positions` (row indices in the padded order, from a
    previous `strip`) the pads are placed there; otherwise they are appended. The input is not
    mutated; the caller keeps its unpadded batch.
    """
    import numpy as np
    import torch

    from verl.protocol import DataProto

    n_real = len(batch)
    n_pad = pad_count(n_real, dp_size)
    if n_pad == 0:
        return batch
    src = _shortest_row(batch)
    pad_part = batch.select_idxs([src]).repeat(repeat_times=n_pad, interleave=True)
    pad_part.meta_info = {}  # concat merges meta_info by equality; tensors there would trip it
    if pad_part.batch is not None and "response_mask" in pad_part.batch.keys():
        pad_part.batch["response_mask"] = torch.zeros_like(pad_part.batch["response_mask"])
    # shallow view of the real rows so the marker key is ours (concat copies the tensors once)
    real = DataProto(batch=batch.batch, non_tensor_batch=dict(batch.non_tensor_batch),
                     meta_info=batch.meta_info)
    real.non_tensor_batch[PAD_KEY] = np.zeros(n_real, dtype=np.int64)
    pad_part.non_tensor_batch[PAD_KEY] = np.ones(n_pad, dtype=np.int64)
    padded = DataProto.concat([real, pad_part])
    if positions is not None and plan_fits(positions, n_real, dp_size):
        n = n_real + n_pad
        is_pad = np.zeros(n, dtype=bool)
        is_pad[np.asarray(positions, dtype=np.int64)] = True
        order = np.empty(n, dtype=np.int64)
        order[~is_pad] = np.arange(n_real)
        order[is_pad] = n_real + np.arange(n_pad)
        padded.reorder(torch.as_tensor(order))
    return padded


def strip(batch):
    """(real-rows-only DataProto, pad positions in the given order | None).

    Drops PAD_KEY from the result so the stripped batch is byte-identical to the unpadded flow.
    A batch that carries no marker (never padded) is returned as is with None.
    """
    import numpy as np

    v = batch.non_tensor_batch.get(PAD_KEY)
    if v is None:
        return batch, None
    flags = np.asarray(v).astype(np.int64)
    positions = np.nonzero(flags == 1)[0]
    real = batch.select_idxs(flags == 0)
    real.non_tensor_batch.pop(PAD_KEY, None)
    return real, positions
