"""B+ tree unit tests: fuzzing against a sorted list, splits, tombstones,
range scans and duplicate handling (fixed seed for reproducibility)."""
import random

import pytest

from engine.btree import BTree, DuplicateKeyError, MemNodeStore


def make_tree(order=8):
    return BTree(order, MemNodeStore())


def test_insert_get_entries():
    tree = make_tree()
    for k in range(100):
        tree.insert(k, k * 10)
    assert tree.get(0) == 0
    assert tree.get(99) == 990
    assert tree.get(50) == 500
    assert tree.get(101) is None
    assert list(tree.entries()) == [(k, k * 10) for k in range(100)]
    assert len(tree) == 100


def test_duplicate_key_raises():
    tree = make_tree()
    tree.insert(1, 100)
    with pytest.raises(DuplicateKeyError):
        tree.insert(1, 200)


def test_split_ascending():
    tree = make_tree(order=4)
    for k in range(50):
        tree.insert(k, k)
    assert len(tree) == 50
    assert list(tree.entries()) == [(k, k) for k in range(50)]
    assert tree.height() >= 2


def test_split_descending():
    tree = make_tree(order=4)
    for k in range(50, 0, -1):
        tree.insert(k, k)
    assert list(tree.entries()) == [(k, k) for k in range(1, 51)]


def test_tombstone_delete():
    tree = make_tree(order=8)
    for k in range(100):
        tree.insert(k, k)
    for k in range(0, 100, 2):
        assert tree.delete(k) is True
    assert len(tree) == 50
    for k in range(0, 100, 2):
        assert tree.get(k) is None
    assert tree.delete(0) is False  # already gone


def test_tombstone_resurrection():
    tree = make_tree(order=8)
    tree.insert(5, 1)
    tree.delete(5)
    tree.insert(5, 2)  # resurrect, no duplicate error
    assert tree.get(5) == 2
    assert len(tree) == 1


def test_range_scan():
    tree = make_tree(order=8)
    for k in range(100):
        tree.insert(k, k)
    assert tree.range_scan(10, 20) == [(k, k) for k in range(10, 21)]
    assert tree.range_scan(90) == [(k, k) for k in range(90, 100)]
    assert tree.range_scan(None, 4) == [(k, k) for k in range(5)]
    assert tree.range_scan() == [(k, k) for k in range(100)]
    tree.delete(15)
    assert 15 not in [k for k, _ in tree.range_scan(10, 20)]


def test_fuzz_vs_sorted_list():
    rng = random.Random(42)
    tree = make_tree(order=16)
    keys = []
    for _ in range(2000):
        k = rng.randrange(100000)
        if k in keys:
            continue
        tree.insert(k, k)
        keys.append(k)
    keys.sort()
    assert list(tree.entries()) == [(k, k) for k in keys]
    for k in keys:
        assert tree.get(k) == k
    assert tree.get(999999) is None
    # delete a random half, compare again
    rng.shuffle(keys)
    for k in keys[: len(keys) // 2]:
        tree.delete(k)
    remaining = sorted(keys[len(keys) // 2 :])
    assert list(tree.entries()) == [(k, k) for k in remaining]
