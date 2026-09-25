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

import argparse
import bisect
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.btree import BTree, MemNodeStore
from engine.learned_index import BlockLearnedIndex, LearnedIndex
from keygen import gen_clustered, gen_uniform, gen_zipf

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


def _clean(obj):
    """Recursively replace NaN/Inf floats with None so JSON stays valid."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _build_queries(keys: np.ndarray, rng: np.random.Generator, n_queries: int) -> Tuple[np.ndarray, np.ndarray]:
    """Present (keys) and absent (random non-keys) point-lookup sets."""
    key_set = set(map(int, keys))
    present = rng.choice(keys, n_queries // 2, replace=False)
    absent_vals = [int(v) for v in rng.integers(0, 2**48, n_queries) if v not in key_set]
    absent = np.asarray(absent_vals[: n_queries // 2], dtype=np.int64)
    return present, absent


def _prediction_error_stats(idx: LearnedIndex, queries: np.ndarray) -> dict:
    """Distribution of ``|predict(q) - bisect_left(q)|`` over ``queries``.

    The exact-search window is ``[predicted ± (threshold+1)]`` (inclusive), so a
    prediction is *out of window* only when its error exceeds ``threshold+1``.
    ``over_threshold_frac`` measures exactly that and must be 0 for every query:
    fitted pieces bound their error by ``threshold`` and gap keys fall back to an
    exact segment-scoped search, so the window always contains the truth.
    """
    errors = [abs(idx.predict(int(q)) - bisect.bisect_left(idx.keys, int(q))) for q in queries]
    arr = np.asarray(errors, dtype=np.int64)
    return {
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 3),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p99": int(np.percentile(arr, 99)),
        "over_threshold_frac": round(float((arr > idx.threshold + 1).mean()), 6),
    }


def _block_error_stats(idx: BlockLearnedIndex, queries: np.ndarray) -> dict:
    """Block-level prediction error: true block vs. predicted block."""
    nblocks = idx._nblocks
    errs = []
    for q in queries:
        truth_block = bisect.bisect_left(idx.keys, int(q)) // idx.block_size
        errs.append(abs(idx.predict_block(int(q)) - min(nblocks - 1, truth_block)))
    arr = np.asarray(errs, dtype=np.int64)
    return {
        "block_error_max": int(arr.max()),
        "block_over_frac": round(float((arr > idx.block_extra).mean()), 6),
    }


def estimate_comparisons(
    variant: str,
    n_keys: int,
    level1_size: int = LEARNED_LEVEL1,
    threshold: int = 16,
    block_size: int | None = None,
    block_extra: int | None = None,
) -> float:
    """Analytical estimate of expected key-comparisons per lookup.

    ``learned_*`` = level-1 boundary bisect (log2 of fanout) + window bisect
    (log2 of the search window).  The block variant's window is the candidate
    block span; the point variant's window is ``2*(threshold+1)+1`` keys.
    """
    if variant == "sorted":
        return round(math.log2(n_keys), 2)
    if variant == "btree":
        return 4.0  # measured height on 100k keys, order 32
    if variant == "learned_point":
        win = 2 * (threshold + 1) + 1
        return round(math.log2(level1_size) + math.log2(win), 2)
    if variant == "learned_block":
        win = (2 * (block_extra or 1) + 1) * (block_size or 64)
        return round(math.log2(level1_size) + math.log2(win), 2)
    raise ValueError(f"unknown variant {variant}")


# ----------------------------------------------------------------------
# experiment 1: classic index-structure comparison (backward compatible)
# ----------------------------------------------------------------------
def run_classic(seed: int, n_keys: int = N_KEYS, n_queries: int = N_QUERIES) -> int:
    rng = np.random.default_rng(seed)
    keys = np.sort(rng.choice(2**48, n_keys, replace=False))
    present, absent = _build_queries(keys, rng, n_queries)

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

    payload = _clean({
        "meta": {
            "n_keys": n_keys,
            "n_queries": n_queries,
            "seed": seed,
            "btree_order": BTREE_ORDER,
            "learned_level1": LEARNED_LEVEL1,
            "key_space": 2**48,
        },
        "variants": variants,
        "threshold_sweep": sweep,
    })

    path = os.path.join(RESULTS, "index_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"index benchmark -> {path}")
    print(f"  keys={n_keys} queries={n_queries} seed={seed}")
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


# ----------------------------------------------------------------------
# experiment A: distribution sensitivity
# ----------------------------------------------------------------------
DISTRIBUTIONS: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "uniform": lambda rng, n: gen_uniform(rng, n),
    "clustered": lambda rng, n: gen_clustered(rng, n),
    "zipf": lambda rng, n: gen_zipf(rng, n),
}


def run_distribution(distribs: List[str], seed: int, n_keys: int = N_KEYS,
                     n_queries: int = N_QUERIES) -> int:
    """Compare all index variants per key distribution and report prediction-
    error stats.  Writes ``results/index_distribution.json``."""
    per_dist: List[dict] = []
    for name in distribs:
        rng = np.random.default_rng(seed)
        keys = DISTRIBUTIONS[name](rng, n_keys)
        present, absent = _build_queries(keys, rng, n_queries)
        all_q = np.concatenate([present, absent])

        variants: List[dict] = []
        # sorted
        arr = keys
        variants.append({"name": "sorted_array", **measure(
            keys, present, absent,
            build=lambda: arr,
            lookup=lambda h: lambda q: bisect.bisect_left(h, q),
            memory_bytes=int(arr.nbytes), extra={})})
        # btree
        tree, store = build_btree(keys)
        variants.append({"name": "btree", **measure(
            keys, present, absent,
            build=lambda: build_btree(keys)[0],
            lookup=lambda h: h.get,
            memory_bytes=btree_memory(store),
            extra={"nodes": len(store), "height": tree.height()})})
        # learned point index
        li = build_learned(keys, 16)
        variants.append({"name": "learned_e16", **measure(
            keys, present, absent,
            build=lambda: build_learned(keys, 16),
            lookup=lambda h: h.lookup,
            memory_bytes=int(li.memory_estimate()) + int(keys.nbytes),
            extra={
                "model_bytes": int(li.memory_estimate()),
                "pieces": sum(len(p) for p in li._pieces),
                "error": _prediction_error_stats(li, all_q),
            })})
        # learned block-routing index
        bi = BlockLearnedIndex(keys, LEARNED_LEVEL1, 16, block_size=64).build()
        variants.append({"name": "learned_block", **measure(
            keys, present, absent,
            build=lambda: BlockLearnedIndex(keys, LEARNED_LEVEL1, 16, block_size=64).build(),
            lookup=lambda h: h.lookup,
            memory_bytes=int(bi.memory_estimate()) + int(keys.nbytes),
            extra={
                "model_bytes": int(bi.memory_estimate()),
                "pieces": sum(len(p) for p in bi._pieces),
                "block_size": bi.block_size,
                "block_extra": bi.block_extra,
                "error": _prediction_error_stats(bi, all_q),
                "block_error": _block_error_stats(bi, all_q),
            })})
        per_dist.append({"distribution": name, "variants": variants})

    payload = _clean({
        "meta": {"n_keys": n_keys, "n_queries": n_queries, "seed": seed,
                 "btree_order": BTREE_ORDER, "learned_level1": LEARNED_LEVEL1,
                 "key_space": 2**48},
        "per_dist": per_dist,
    })
    path = os.path.join(RESULTS, "index_distribution.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"distribution benchmark -> {path}")
    for d in per_dist:
        print(f"  [{d['distribution']}]")
        for v in d["variants"]:
            err = v.get("error", {})
            print(f"    {v['name']:<14} lookup={v['lookup_ns']:>7.1f} ns  "
                  f"pieces={v.get('pieces', '-'):>6}  "
                  f"err.max={err.get('max', '-'):>5}  "
                  f"over={err.get('over_threshold_frac', '-'):<7}")
    return 0


# ----------------------------------------------------------------------
# experiment B: block-routing sweep
# ----------------------------------------------------------------------
def run_block(block_sizes: List[int], seed: int, dist: str = "uniform",
              n_keys: int = N_KEYS, n_queries: int = N_QUERIES) -> int:
    """Sweep ``block_size`` for the block-routing learned index and report
    latency / memory / error bound / estimated comparisons against the fixed
    reference variants.  Writes ``results/index_block.json``."""
    rng = np.random.default_rng(seed)
    keys = DISTRIBUTIONS[dist](rng, n_keys)
    present, absent = _build_queries(keys, rng, n_queries)
    all_q = np.concatenate([present, absent])

    # fixed references (measured once)
    refs: Dict[str, dict] = {}
    arr = keys
    refs["sorted"] = {"name": "sorted_array", "lookup_ns": round(
        time_lookups(lambda q: bisect.bisect_left(arr, q), all_q), 1),
        "memory_bytes": int(arr.nbytes),
        "comparisons": estimate_comparisons("sorted", n_keys)}
    tree, store = build_btree(keys)
    refs["btree"] = {"name": "btree", "lookup_ns": round(time_lookups(tree.get, all_q), 1),
        "memory_bytes": btree_memory(store), "height": tree.height(),
        "comparisons": estimate_comparisons("btree", n_keys)}
    li = build_learned(keys, 16)
    refs["learned_point"] = {"name": "learned_e16", "lookup_ns": round(time_lookups(li.lookup, all_q), 1),
        "memory_bytes": int(li.memory_estimate()) + int(keys.nbytes),
        "model_bytes": int(li.memory_estimate()),
        "comparisons": estimate_comparisons("learned_point", n_keys)}

    sweep: List[dict] = []
    for bs in block_sizes:
        bi = BlockLearnedIndex(keys, LEARNED_LEVEL1, 16, block_size=bs).build()
        sweep.append({
            "block_size": bs,
            "block_extra": bi.block_extra,
            "lookup_ns": round(time_lookups(bi.lookup, all_q), 1),
            "memory_bytes": int(bi.memory_estimate()) + int(keys.nbytes),
            "model_bytes": int(bi.memory_estimate()),
            "pieces": sum(len(p) for p in bi._pieces),
            "block_error_max": _block_error_stats(bi, all_q)["block_error_max"],
            "block_over_frac": _block_error_stats(bi, all_q)["block_over_frac"],
            "comparisons": estimate_comparisons(
                "learned_block", n_keys, block_size=bs, block_extra=bi.block_extra),
        })

    payload = _clean({
        "meta": {"n_keys": n_keys, "n_queries": n_queries, "seed": seed,
                 "distribution": dist, "btree_order": BTREE_ORDER,
                 "learned_level1": LEARNED_LEVEL1, "key_space": 2**48,
                 "threshold": 16},
        "references": refs,
        "block_sweep": sweep,
    })
    path = os.path.join(RESULTS, "index_block.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"block-routing benchmark -> {path}")
    print(f"  distribution={dist} seed={seed}")
    for name, r in refs.items():
        print(f"    ref {name:<14} lookup={r['lookup_ns']:>7.1f} ns  "
              f"cmp~{r['comparisons']:>5}")
    for s in sweep:
        print(f"    B={s['block_size']:>4} extra={s['block_extra']:>2}  "
              f"lookup={s['lookup_ns']:>7.1f} ns  "
              f"err.max={s['block_error_max']:>3}  "
              f"cmp~{s['comparisons']:>5}  mem={s['memory_bytes'] / 2**20:>6.2f} MB")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="索引结构微基准：经典对比 / 分布敏感性 / 页路由块大小扫描")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="随机种子（默认 42，可复现）")
    parser.add_argument("--n-keys", type=int, default=N_KEYS,
                        help="键数量（默认 100000）")
    parser.add_argument("--n-queries", type=int, default=N_QUERIES,
                        help="点查数量（默认 50000）")
    parser.add_argument("--distribution", nargs="+",
                        choices=list(DISTRIBUTIONS), default=None,
                        help="分布敏感性实验要跑的分布（uniform/clustered/zipf）；"
                             "不指定则只跑经典对比实验")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=None,
                        help="页路由实验的块大小列表（默认不跑）；"
                             "可与 --distribution 结合指定数据分布")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    dist = args.distribution[0] if args.distribution else "uniform"

    ran = False
    if args.distribution is None and args.block_sizes is None:
        # backward-compatible default: classic index comparison only
        run_classic(args.seed, args.n_keys, args.n_queries)
        ran = True
    if args.distribution is not None:
        run_distribution(args.distribution, args.seed, args.n_keys, args.n_queries)
        ran = True
    if args.block_sizes is not None:
        run_block(args.block_sizes, args.seed, dist, args.n_keys, args.n_queries)
        ran = True
    return 0 if ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
