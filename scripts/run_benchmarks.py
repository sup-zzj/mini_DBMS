"""Index structure micro-benchmark: sorted array vs. B+ tree vs. learned
index (two-level RMI).

Question the experiment answers
-------------------------------
For a static, read-only, in-memory workload — the setting where learned
indexes are supposed to shine — does a learned index actually beat a plain
sorted array or a textbook B+ tree?

Method
------
* N = 100_000 unique sorted keys (uniform 48-bit space), Q = 50_000 point
  lookups (half present, half absent), fixed seed.
* Variants:
    - ``sorted_array``  bisect_left over a numpy int64 array
    - ``btree``         B+ tree (order 32, in-memory node store)
    - ``learned_e16``   two-level RMI, threshold 16 (default)
* A threshold sweep (4 / 16 / 64) shows the latency / model-size trade-off.

Everything is wall-clock; run-to-run jitter on a shared machine is small
but real.  See README for the honest interpretation of the results.
"""

from __future__ import annotations

import bisect
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.btree import BTree, MemNodeStore
from engine.learned_index import LearnedIndex

SEED = 42
N_KEYS = 100_000
N_QUERIES = 50_000
BTREE_ORDER = 32
LEARNED_LEVEL1 = 64

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


# ----------------------------------------------------------------------
# memory estimates
# ----------------------------------------------------------------------
def btree_memory(store: MemNodeStore) -> int:
    """Approximate bytes held by the B+ tree node store."""
    total = 0
    for node in store.nodes.values():
        if node["t"] == 1:  # leaf
            total += 8 * len(node["k"]) * 2 + 128
        else:  # internal
            total += 8 * len(node["k"]) + 8 * len(node["c"]) + 128
    return total


def time_lookups(fn, queries: np.ndarray) -> float:
    """Return nanoseconds per query (caller passes ints)."""
    start = time.perf_counter_ns()
    for q in queries:
        fn(int(q))
    elapsed = time.perf_counter_ns() - start
    return elapsed / len(queries)


def build_btree(keys: np.ndarray, order: int = BTREE_ORDER) -> Tuple[BTree, MemNodeStore]:
    store = MemNodeStore()
    tree = BTree(order, store)
    for k in keys:
        tree.insert(int(k), int(k))
    return tree, store


def build_learned(keys: np.ndarray, threshold: int) -> LearnedIndex:
    idx = LearnedIndex(keys, LEARNED_LEVEL1, threshold)
    idx.build()
    return idx


def measure(keys, present, absent, build, lookup, memory_bytes, extra: dict) -> dict:
    t0 = time.perf_counter()
    handle = build()
    build_s = time.perf_counter() - t0
    all_q = np.concatenate([present, absent])
    return {
        "build_s": round(build_s, 4),
        "lookup_ns": round(time_lookups(lookup(handle), all_q), 1),
        "present_ns": round(time_lookups(lookup(handle), present), 1),
        "absent_ns": round(time_lookups(lookup(handle), absent), 1),
        "memory_bytes": memory_bytes,
        "memory_mb": round(memory_bytes / 2**20, 3),
        **extra,
    }


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------
    # data set + query set
    # ---------------------------------------------------------------
    keys = np.sort(rng.choice(2**48, N_KEYS, replace=False))
    key_set = set(map(int, keys))
    present = rng.choice(keys, N_QUERIES // 2, replace=False)
    absent_vals = [int(v) for v in rng.integers(0, 2**48, N_QUERIES) if v not in key_set]
    absent = np.asarray(absent_vals[: N_QUERIES // 2], dtype=np.int64)

    variants: List[dict] = []

    # 1) sorted array
    arr = keys
    variants.append({
        "name": "sorted_array",
        **measure(
            keys, present, absent,
            build=lambda: arr,
            lookup=lambda h: lambda q: bisect.bisect_left(h, q),
            memory_bytes=int(arr.nbytes),
            extra={},
        ),
    })

    # 2) B+ tree
    tree, store = build_btree(keys)
    variants.append({
        "name": "btree",
        **measure(
            keys, present, absent,
            build=lambda: build_btree(keys)[0],
            lookup=lambda h: h.get,
            memory_bytes=btree_memory(store),
            extra={"nodes": len(store), "height": tree.height()},
        ),
    })

    # 3) learned index (default threshold 16)
    li = build_learned(keys, 16)
    variants.append({
        "name": "learned_e16",
        **measure(
            keys, present, absent,
            build=lambda: build_learned(keys, 16),
            lookup=lambda h: h.lookup,
            # honest accounting: the model still needs the key array for the
            # final exact check, so report model + keys together
            memory_bytes=int(li.memory_estimate()) + int(keys.nbytes),
            extra={
                "model_bytes": int(li.memory_estimate()),
                "pieces": sum(len(p) for p in li._pieces),
            },
        ),
    })

    # threshold sweep: model size vs. lookup latency
    sweep: Dict[str, dict] = {}
    for th in (4, 16, 64):
        li_th = build_learned(keys, th)
        sweep[str(th)] = {
            "lookup_ns": round(
                time_lookups(li_th.lookup, np.concatenate([present, absent])), 1
            ),
            "memory_bytes": int(li_th.memory_estimate()) + int(keys.nbytes),
            "model_bytes": int(li_th.memory_estimate()),
            "memory_mb": round((li_th.memory_estimate() + keys.nbytes) / 2**20, 3),
            "pieces": sum(len(p) for p in li_th._pieces),
        }

    payload = {
        "meta": {
            "n_keys": N_KEYS,
            "n_queries": N_QUERIES,
            "seed": SEED,
            "btree_order": BTREE_ORDER,
            "learned_level1": LEARNED_LEVEL1,
            "key_space": 2**48,
        },
        "variants": variants,
        "threshold_sweep": sweep,
    }

    path = os.path.join(RESULTS, "index_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"index benchmark -> {path}")
    print(f"  keys={N_KEYS} queries={N_QUERIES} seed={SEED}")
    for v in variants:
        print(
            f"  {v['name']:<12} build={v['build_s']:>7.2f}s  "
            f"lookup={v['lookup_ns']:>7.1f} ns  "
            f"(present={v['present_ns']:.1f}, absent={v['absent_ns']:.1f})  "
            f"mem={v['memory_mb']:>8.2f} MB"
        )
    print("  threshold sweep (lookup ns / memory MB / pieces):")
    for th, d in sweep.items():
        print(f"    E={th:>3}  {d['lookup_ns']:>7.1f} ns  {d['memory_mb']:>7.2f} MB  {d['pieces']} pieces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
