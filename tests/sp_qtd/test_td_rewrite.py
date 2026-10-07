"""sp_td_rewrite_{gen,driver}_copies — the per-copy prefix/uid rewrite.

Contract: after an interleaved repeat, copy j of a TD slot gets prefix traj[:cuts[j]] and
cap caps[j] (gen side) and its own extra_info dict with sp_prefix_len = cuts[j], its own
uid for j >= 1, and sp_q_td_row = 1 (driver side); non-TD rows are untouched, byte for
byte. The two rewrites use the same counts, so the gen batch's row r and the driver
batch's row r describe the same request.

CPU only; operates on plain dict/np.object arrays like the live non_tensor_batch.
"""
import numpy as np
import pytest

from verl.trainer.ppo.ray_trainer import (
    _sp_td_copy_ranges,
    sp_td_mirror_driver_keys,
    sp_td_rewrite_driver_copies,
    sp_td_rewrite_gen_copies,
)


class _Proto:
    """gen DataProto stand-in: .non_tensor_batch dict of np object arrays, .batch None."""
    def __init__(self, ntb):
        self.non_tensor_batch = ntb
        self.batch = None


def _obj(rows):
    a = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        a[i] = r
    return a


def _repeat(rows, counts):
    out = []
    for r, c in zip(rows, counts):
        out.extend([r] * c)  # repeat duplicates REFERENCES, like DataProto.repeat
    return out


def test_copy_ranges_are_contiguous_interleaved():
    assert _sp_td_copy_ranges([1, 3, 2]) == [(0, 0, 1), (1, 1, 3), (2, 4, 2)]


def _mk_batches(n_sib=4):
    traj = list(range(100))
    cuts = [0, 10, 20, 30]
    caps = [50, 50, 50, 40]
    # source rows: statement (n=1), TD replay slot (n=4), plain replay slot (n=4)
    pre_gen = {
        "sp_td_cuts": _obj([[], cuts, []]),
        "sp_td_caps": _obj([[], caps, []]),
        "sp_td_traj_ids": _obj([[], traj, []]),
    }
    counts = [1, n_sib, n_sib]
    gen_rows_prefix = _repeat([[], traj[:cuts[0]], [7, 7]], counts)
    gen_rows_cap = _repeat([0, caps[0], 60], counts)
    out_gen = _Proto({
        "sp_prefix_token_ids": _obj(gen_rows_prefix),
        "sp_q_max_new_tokens": _obj(gen_rows_cap),
    })
    ei_stmt = {"sp_q_td_group": 0, "sp_prefix_len": 0}
    ei_td = {"sp_q_td_group": 1, "sp_td_cuts": cuts, "sp_td_len": len(traj),
             "sp_prefix_len": cuts[0], "sp_cut_fraction": 0.0, "sp_q_route": "short"}
    ei_rep = {"sp_q_td_group": 0, "sp_prefix_len": 7}
    driver = {
        "uid": _obj([f"u{i}" for i in range(sum(counts))]),
        "extra_info": _obj(_repeat([ei_stmt, ei_td, ei_rep], counts)),
    }
    return pre_gen, out_gen, driver, counts, traj, cuts, caps


def test_gen_rewrite_assigns_per_copy_prefix_and_cap():
    pre_gen, out_gen, _, counts, traj, cuts, caps = _mk_batches()
    n = sp_td_rewrite_gen_copies(pre_gen, out_gen, counts)
    assert n == 1
    prefix = out_gen.non_tensor_batch["sp_prefix_token_ids"]
    cap = out_gen.non_tensor_batch["sp_q_max_new_tokens"]
    # TD copies occupy positions 1..4
    for j in range(4):
        assert prefix[1 + j] == traj[: cuts[j]], (j, len(prefix[1 + j]))
        assert cap[1 + j] == caps[j]
    # neighbors untouched
    assert prefix[0] == [] and cap[0] == 0
    assert all(prefix[5 + j] == [7, 7] and cap[5 + j] == 60 for j in range(4))


def test_driver_rewrite_singleton_uids_and_per_copy_extra():
    _, _, driver, counts, traj, cuts, _ = _mk_batches()
    n = sp_td_rewrite_driver_copies(driver, counts)
    assert n == 1
    extra = driver["extra_info"]
    uids = driver["uid"]
    td = [extra[1 + j] for j in range(4)]
    # per-copy dicts, not shared references
    assert len({id(e) for e in td}) == 4
    for j, e in enumerate(td):
        assert e["sp_q_td_row"] == 1
        assert e["sp_prefix_len"] == cuts[j]
        assert e["sp_cut_fraction"] == pytest.approx(cuts[j] / len(traj))
    # fresh uid for copies j >= 1; copy 0 keeps the slot's
    td_uids = [uids[1 + j] for j in range(4)]
    assert td_uids[0] == "u1"
    assert len(set(td_uids)) == 4, td_uids
    # non-TD rows untouched: statement + the plain replay slot's copies still share dicts
    assert extra[0]["sp_q_td_group"] == 0 and "sp_q_td_row" not in extra[0]
    rep = [extra[5 + j] for j in range(4)]
    assert len({id(e) for e in rep}) == 1
    assert [uids[5 + j] for j in range(4)] == ["u5", "u6", "u7", "u8"]


def test_count_mismatch_is_refused():
    pre_gen, out_gen, driver, _, _, _, _ = _mk_batches()
    bad_counts = [1, 3, 4]  # slot stamped 4 cuts but repeats 3 copies
    with pytest.raises(AssertionError):
        sp_td_rewrite_gen_copies(pre_gen, out_gen, bad_counts)
    with pytest.raises(AssertionError):
        sp_td_rewrite_driver_copies(driver, bad_counts)


def test_no_td_key_is_a_noop():
    _, out_gen, _, counts, _, _, _ = _mk_batches()
    assert sp_td_rewrite_gen_copies({"sp_td_cuts": None}, out_gen, counts) == 0
    assert sp_td_rewrite_gen_copies({}, out_gen, counts) == 0


def test_union_passes_after_mirror_of_rewritten_keys():
    """Failure seen in a live run: the gen output comes back from Ray with PICKLED
    copies of uid/extra_info; after the driver rewrite the two sides differ and
    union_numpy_dict asserts. The mirror must make the real union pass and keep the
    driver's per-copy bookkeeping."""
    import copy
    from verl.protocol import union_numpy_dict

    counts = [1, 4]
    ei_td = {"sp_q_td_group": 1, "sp_td_cuts": [0, 10, 20, 30], "sp_td_len": 100,
             "sp_prefix_len": 0, "sp_cut_fraction": 0.0}
    src_extra = [{"x": 1}, ei_td]
    src_uid = ["s0", "t0"]
    driver = {"uid": _obj(_repeat(src_uid, counts)),
              "extra_info": _obj(_repeat(src_extra, counts)),
              "qid": _obj(_repeat(["q0", "q1"], counts))}
    gen = {k: copy.deepcopy(v) for k, v in driver.items()}   # the Ray round-trip
    gen["responses_len"] = _obj([5] * 5)                       # gen-only key must survive

    sp_td_rewrite_driver_copies(driver, counts)
    with pytest.raises(AssertionError):
        union_numpy_dict(dict(driver), dict(gen))

    assert sp_td_mirror_driver_keys(driver, gen) == ["uid", "extra_info"]
    merged = union_numpy_dict(dict(driver), dict(gen))
    assert set(merged) == {"uid", "extra_info", "qid", "responses_len"}
    assert merged["uid"][0] == "s0" and merged["uid"][1] == "t0"
    assert len({merged["uid"][i] for i in range(1, 5)}) == 4
    assert [merged["extra_info"][i]["sp_prefix_len"] for i in range(1, 5)] == [0, 10, 20, 30]
    assert all(merged["extra_info"][i]["sp_q_td_row"] == 1 for i in range(1, 5))
    assert merged["extra_info"][0] == {"x": 1}


def test_mirror_is_noop_without_shared_keys():
    driver = {"uid": _obj(["a"]), "extra_info": _obj([{}])}
    gen = {"responses_len": _obj([1])}
    assert sp_td_mirror_driver_keys(driver, gen) == []
    assert set(gen) == {"responses_len"}
