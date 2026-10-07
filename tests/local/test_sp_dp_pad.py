"""sp_dp_pad -- padded data-parallel dispatch for a fixed batch shape on a non-dividing world size.

The pure-Python parts (pad_count, minibatch_ids, describe) run without torch; the DataProto
round-trip tests skip when torch/numpy are not installed (the login-node venv has them).

Shapes under test are the AC2 batch on 7 nodes: 384/192/96 x n=16 -> pre-drop 3264 rows,
trained 3072, minibatch 1536, on 7 nodes (dp=56) and 8 nodes (dp=64, must be inert).
"""
import random

import pytest

from verl.trainer.ppo import sp_dp_pad as sp

PREDROP, TRAINED, MINI = 192 * 1 + 192 * 16, 192 * 16, 96 * 16


def _chunks(flags, dp):
    slots = len(flags) // dp
    return [list(range(c * slots, (c + 1) * slots)) for c in range(dp)]


def _check_ids(flags, dp, k, ids):
    n = len(flags)
    assert len(ids) == n
    n_real = n - sum(flags)
    # every minibatch holds exactly n_real/k REAL rows globally
    for m in range(k):
        got = sum(1 for i in range(n) if flags[i] == 0 and ids[i] == m)
        assert got == n_real // k, (m, got, n_real // k)
    # every chunk (rank) holds >= 1 REAL row of every minibatch, and ids are in range
    for rows in _chunks(flags, dp):
        for m in range(k):
            assert any(flags[i] == 0 and ids[i] == m for i in rows), (rows[0], m)
    assert set(ids) <= set(range(k))


def test_pad_count():
    assert sp.pad_count(PREDROP, 56) == 40 and (PREDROP + 40) % 56 == 0
    assert sp.pad_count(TRAINED, 56) == 8 and (TRAINED + 8) % 56 == 0
    assert sp.pad_count(PREDROP, 64) == 0
    assert sp.pad_count(TRAINED, 64) == 0
    assert sp.pad_count(0, 7) == 0
    assert sp.pad_count(1, 7) == 6


def test_minibatch_ids_at_dp56():
    n_pad = sp.pad_count(TRAINED, 56)          # 8 pads appended -> 3080 = 56 x 55
    flags = [0] * TRAINED + [1] * n_pad
    ids = sp.minibatch_ids(flags, 56, 2)
    _check_ids(flags, 56, 2, ids)
    # per-rank split is 28/27 or 27/28 (55 slots): never more uneven than that
    for rows in _chunks(flags, 56):
        c0 = sum(1 for i in rows if ids[i] == 0)
        assert c0 in (27, 28), c0


def test_minibatch_ids_dividing_shape_is_exact_halves():
    flags = [0] * TRAINED                       # dp=64: 48 per rank, 24/24
    ids = sp.minibatch_ids(flags, 64, 2)
    _check_ids(flags, 64, 2, ids)
    for rows in _chunks(flags, 64):
        assert sum(1 for i in rows if ids[i] == 0) == 24


def test_minibatch_ids_pads_anywhere_in_the_order():
    # pads at balancer-chosen positions, not appended
    rng = random.Random(831001)
    n_pad = sp.pad_count(TRAINED, 56)
    flags = [0] * (TRAINED + n_pad)
    for p in rng.sample(range(TRAINED + n_pad), n_pad):
        flags[p] = 1
    ids = sp.minibatch_ids(flags, 56, 2)
    _check_ids(flags, 56, 2, ids)


@pytest.mark.parametrize("seed", range(20))
def test_minibatch_ids_random_shapes(seed):
    rng = random.Random(seed)
    dp = rng.choice([3, 5, 7, 12, 56])
    k = rng.choice([2, 3, 4])
    groups = rng.randint(dp * k // 4 + 4, 400)      # real rows = groups*k, a multiple of k
    n_real = groups * k
    n_pad = sp.pad_count(n_real, dp)
    n = n_real + n_pad
    if n // dp - 1 < k:                            # a chunk with a pad must still hold k real rows
        pytest.skip("shape too small for the >=1-real-row-per-minibatch-per-rank invariant")
    flags = [0] * n
    for p in rng.sample(range(n), n_pad):
        flags[p] = 1
    ids = sp.minibatch_ids(flags, dp, k)
    _check_ids(flags, dp, k, ids)


def test_minibatch_ids_refuses_impossible_shapes():
    with pytest.raises(AssertionError):
        sp.minibatch_ids([0] * 10, 4, 2)           # 10 not a multiple of dp
    with pytest.raises(AssertionError):
        sp.minibatch_ids([0] * 9 + [1] * 3, 4, 2)  # 9 real rows, 2 minibatches
    with pytest.raises(AssertionError):
        sp.minibatch_ids([0, 0, 1, 1], 2, 2)       # chunk 1 has no real row


def test_describe_flags_activity():
    active = sp.describe(PREDROP, TRAINED, MINI, 56)
    inert = sp.describe(PREDROP, TRAINED, MINI, 64)
    assert "ACTIVE" in active and "+40 pad" in active and "+8 pad" in active
    assert "inert" in inert and "+0 pad" in inert


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("SP_DP_PAD", raising=False)
    assert not sp.enabled()
    monkeypatch.setenv("SP_DP_PAD", "1")
    assert sp.enabled()
    monkeypatch.setenv("SP_DP_PAD", "0")
    assert not sp.enabled()


# ---------------------------------------------------------------------------------------------
# DataProto round trips (need torch + numpy)
# ---------------------------------------------------------------------------------------------

def _proto(n, seqlen=8, resp=4):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from verl.protocol import DataProto

    lens = torch.randint(1, seqlen + 1, (n,))
    attn = (torch.arange(seqlen).unsqueeze(0) < lens.unsqueeze(1)).long()
    resp_mask = (torch.arange(resp).unsqueeze(0) < torch.clamp(lens, max=resp).unsqueeze(1)).long()
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(n).unsqueeze(1).repeat(1, seqlen),
            "attention_mask": attn,
            "response_mask": resp_mask,
        },
        non_tensors={"uid": np.array([f"u{i}" for i in range(n)], dtype=object)},
        meta_info={"global_token_num": list(range(n))},
    )


def test_pad_batch_strip_roundtrip_appended():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    torch.manual_seed(0)
    b = _proto(30)
    p = sp.pad_batch(b, 7)                          # 30 -> 35
    assert len(p) == 35 and len(b) == 30            # caller's batch untouched
    flags = sp.pad_flags(p)
    assert flags.sum() == 5 and list(flags[:30]) == [0] * 30
    # pads duplicate the SHORTEST row and carry a zero response mask
    src = int(b.batch["attention_mask"].sum(-1).argmin())
    assert torch.equal(p.batch["input_ids"][30], b.batch["input_ids"][src])
    assert int(p.batch["response_mask"][30:].sum()) == 0
    assert p.meta_info["global_token_num"] == b.meta_info["global_token_num"]
    real, pos = sp.strip(p)
    assert len(real) == 30 and list(pos) == list(range(30, 35))
    assert sp.PAD_KEY not in real.non_tensor_batch
    assert torch.equal(real.batch["input_ids"], b.batch["input_ids"])
    assert list(real.non_tensor_batch["uid"]) == list(b.non_tensor_batch["uid"])
    # keep-mask unpad of a per-row output lines up with the real rows
    keep = sp.real_mask(p)
    out = p.select_idxs(keep)
    assert len(out) == 30 and torch.equal(out.batch["input_ids"], b.batch["input_ids"])


def test_pad_batch_honours_a_plan_and_strip_recovers_it():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    torch.manual_seed(1)
    b = _proto(30)
    plan = np.array([0, 7, 8, 20, 34])
    p = sp.pad_batch(b, 7, positions=plan)
    flags = sp.pad_flags(p)
    assert list(np.nonzero(flags)[0]) == list(plan)
    real, pos = sp.strip(p)
    assert list(pos) == list(plan)
    assert torch.equal(real.batch["input_ids"], b.batch["input_ids"])   # order of real rows preserved
    # a plan for a different length is ignored (pads appended instead)
    assert not sp.plan_fits(plan, 29, 7)
    q = sp.pad_batch(_proto(29), 7, positions=plan)
    assert list(np.nonzero(sp.pad_flags(q))[0]) == list(range(29, 35))


def test_pad_batch_is_identity_when_it_divides():
    pytest.importorskip("torch")
    b = _proto(28)
    assert sp.pad_batch(b, 7) is b
    real, pos = sp.strip(b)
    assert real is b and pos is None


def test_make_iterator_by_ids_groups_rows():
    torch = pytest.importorskip("torch")
    from tensordict import TensorDict

    from verl.utils.tensordict_utils import make_iterator_by_ids

    td = TensorDict({"x": torch.arange(7).unsqueeze(1)}, batch_size=[7])
    ids = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    got = [mb["x"].flatten().tolist() for mb in make_iterator_by_ids(td, ids, 2, epochs=2)]
    assert got == [[0, 2, 5], [1, 3, 4, 6]] * 2
    with pytest.raises(AssertionError):
        list(make_iterator_by_ids(td, torch.zeros(7, dtype=torch.long), 2, 1))  # minibatch 1 empty
