"""Learned index tests: exactness for every trained key, insertion-point
semantics, and behaviour across parameter choices (fixed seed)."""
import random

import pytest

from engine.learned_index import LearnedIndex


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
