"""Learned index tests: exactness for every trained key, insertion-point
semantics, and behaviour across parameter choices (fixed seed)."""
import os
import random
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from engine.learned_index import BlockLearnedIndex, LearnedIndex  # noqa: E402
from keygen import gen_clustered, gen_uniform, gen_zipf  # noqa: E402


def sorted_unique(rng, n, span):
    keys = set()
    while len(keys) < n:
        keys.add(rng.randrange(span))
    return sorted(keys)


def test_all_keys_found():
    rng = random.Random(42)
    keys = sorted_unique(rng, 5000, 100_000)
    idx = LearnedIndex(keys, level1_size=64, threshold=16).build()
    for k in keys:
        assert idx.search(k) is True
        assert idx.lookup(k) == keys.index(k)


def test_membership_matches_set():
    rng = random.Random(7)
    keys = sorted_unique(rng, 3000, 50_000)
    key_set = set(keys)
    idx = LearnedIndex(keys, level1_size=32, threshold=8).build()
    probes = [rng.randrange(50_000) for _ in range(2000)]
    for k in probes:
        assert idx.search(k) == (k in key_set)


def test_lookup_insertion_point():
    keys = list(range(0, 1000, 2))  # even numbers
    idx = LearnedIndex(keys, level1_size=16, threshold=4).build()
    for k in range(0, 1000):
        expect = keys.index(k) if k in keys else len([x for x in keys if x < k])
        assert idx.lookup(k) == expect


def test_small_data_single_segment():
    keys = [3, 7, 9, 42, 101]
    idx = LearnedIndex(keys, level1_size=100, threshold=2).build()
    for k in keys:
        assert idx.search(k)
    assert idx.search(5) is False


def test_threshold_sweep_keeps_exactness():
    rng = random.Random(123)
    keys = sorted_unique(rng, 4000, 200_000)
    for threshold in (4, 16, 64):
        idx = LearnedIndex(keys, level1_size=32, threshold=threshold).build()
        assert all(idx.search(k) for k in keys)
        assert idx.memory_estimate() > 0


def test_memory_estimate_and_repr():
    keys = list(range(1000))
    idx = LearnedIndex(keys).build()
    assert idx.memory_estimate() > 0
    assert "LearnedIndex" in repr(idx)


def test_not_built_raises():
    idx = LearnedIndex([1, 2, 3])
    with pytest.raises(RuntimeError):
        idx.lookup(1)


# ----------------------------------------------------------------------
# BlockLearnedIndex
# ----------------------------------------------------------------------
def test_block_index_exact():
    """Every trained key is found, with the correct insertion position."""
    for keys in (
        sorted_unique(random.Random(1), 5000, 100_000),
        list(gen_clustered(np.random.default_rng(2), 5000).tolist()),
        list(gen_zipf(np.random.default_rng(3), 5000).tolist()),
    ):
        for bs, extra in ((32, None), (128, None), (128, 2)):
            idx = BlockLearnedIndex(keys, level1_size=64, threshold=16,
                                    block_size=bs, block_extra=extra).build()
            for k in keys:
                assert idx.search(k) is True
                assert idx.lookup(k) == keys.index(k)


def test_block_membership_matches_set():
    rng = random.Random(7)
    keys = sorted_unique(rng, 3000, 50_000)
    key_set = set(keys)
    idx = BlockLearnedIndex(keys, level1_size=32, threshold=8, block_size=64).build()
    probes = [rng.randrange(50_000) for _ in range(2000)]
    for k in probes:
        assert idx.search(k) == (k in key_set)


def test_block_error_bounded():
    """Predicted block never strays beyond block_extra of the true block."""
    rng = random.Random(11)
    keys = sorted_unique(rng, 4000, 200_000)
    idx = BlockLearnedIndex(keys, level1_size=64, threshold=16, block_size=64).build()
    nblocks = idx._nblocks
    for k in keys:
        pred_b = idx.predict_block(k)
        true_b = min(nblocks - 1, keys.index(k) // idx.block_size)
        assert abs(pred_b - true_b) <= idx.block_extra


def test_predict_within_threshold():
    """|predict(k) - bisect_left(k)| <= threshold + 1 for trained keys."""
    rng = random.Random(5)
    keys = sorted_unique(rng, 4000, 100_000)
    for televel, th in ((16, 8), (16, 16), (32, 4)):
        idx = LearnedIndex(keys, level1_size=televel, threshold=th).build()
        for k in keys:
            assert abs(idx.predict(k) - keys.index(k)) <= th + 1


def test_predict_bounded_on_gap_keys():
    """|predict(q) - bisect_left(q)| <= threshold + 1 for *absent* queries too.

    Regression test for the gap-extrapolation bug: a query key can fall between
    two fitted pieces or past a piece's last trained key, where the linear model
    would extrapolate unboundedly.  The index must fall back to an exact
    segment-scoped search so the exact-search window always contains truth.
    """
    import bisect
    for gen in (gen_uniform, gen_clustered, gen_zipf):
        keys = gen(np.random.default_rng(2024), 20_000).tolist()
        idx = LearnedIndex(keys, level1_size=64, threshold=16).build()
        rng = np.random.default_rng(999)
        probes = rng.integers(0, 2**48, 10_000)
        for q in probes:
            truth = bisect.bisect_left(keys, int(q))
            assert abs(idx.predict(int(q)) - truth) <= idx.threshold + 1


# ----------------------------------------------------------------------
# keygen reproducibility
# ----------------------------------------------------------------------
@pytest.mark.parametrize("gen", [gen_uniform, gen_clustered, gen_zipf])
def test_keygen_seeded_reproducible(gen):
    a = gen(np.random.default_rng(42), 5000)
    b = gen(np.random.default_rng(42), 5000)
    assert np.array_equal(a, b)
    c = gen(np.random.default_rng(43), 5000)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("gen", [gen_uniform, gen_clustered, gen_zipf])
def test_keygen_sorted_unique_in_range(gen):
    keys = gen(np.random.default_rng(1), 5000)
    assert len(keys) == 5000
    assert keys[0] >= 0 and keys[-1] < 2**48
    assert all(keys[i] < keys[i + 1] for i in range(len(keys) - 1))
