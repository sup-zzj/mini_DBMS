"""Buffer-pool replacement-policy benchmark.

Question the experiment answers
-------------------------------
How much does the eviction policy matter for the hit ratio under a skewed
(Zipf) page-access workload, and how far is each online policy from the
offline optimum (Belady / MIN)?

Method
------
* D = 1_000 distinct pages, Q = 100_000 page accesses drawn from a Zipf
  distribution (s = 1.2, the typical skewed web/DB access pattern), fixed
  seed.
* Pool capacities: 5 % / 10 % / 20 % / 50 % / 100 % of the page set.
* Policies:
    - ``lru`` / ``clock`` / ``random`` run through the *real* buffer pool
      (in-memory disks, real pinning and hit counting);
    - ``opt`` is Belady's offline optimum simulated separately (it needs
      the future, so it cannot run in the pool) — included as the
      theoretical reference line.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.buffer_pool import BufferPool
from engine.page import DiskManager

SEED = 42
N_PAGES = 1_000
N_ACCESS = 100_000
ZIPF_S = 1.2
CAPACITY_PCTS = [5, 10, 20, 50, 100]
POLICIES = ["lru", "clock", "random"]

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def zipf_accesses(n_pages: int, n: int, s: float, seed: int) -> np.ndarray:
    """Zipf-distributed page ids (rank 0 is the hottest)."""
    rng = np.random.default_rng(seed)
    ranks = rng.zipf(s, size=n) - 1
    return (ranks % n_pages).astype(np.int64)


def pool_hit_ratio(policy: str, capacity: int, accesses: np.ndarray) -> float:
    """Run the real buffer pool over the access trace; return hit ratio."""
    pool = BufferPool(capacity, policy=policy, seed=SEED)
    pool.register_relation("r", DiskManager("mem://bench", in_memory=True))
    for p in accesses:
        page = pool.fetch_page("r", int(p))
        pool.unpin(page)
    ratio = pool.hit_ratio()
    pool.reset_stats()
    return ratio


def opt_hit_ratio(accesses: np.ndarray, capacity: int) -> float:
    """Belady's offline optimum: evict the resident page whose next use is
    furthest in the future."""
    n = len(accesses)
    nxt = np.full(n, n, dtype=np.int64)  # next occurrence of accesses[i]
    last = {}
    for i in range(n - 1, -1, -1):
        p = int(accesses[i])
        nxt[i] = last.get(p, n)
        last[p] = i
    next_use = {}  # resident page -> next use position (n = never again)
    hits = 0
    for i, p in enumerate(accesses):
        if p in next_use:
            hits += 1
            next_use[p] = int(nxt[i])
            continue
        if len(next_use) == capacity:
            victim = max(next_use, key=next_use.get)
            del next_use[victim]
        next_use[p] = int(nxt[i])
    return hits / n


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    accesses = zipf_accesses(N_PAGES, N_ACCESS, ZIPF_S, SEED)
    capacities = [int(N_PAGES * p / 100) for p in CAPACITY_PCTS]

    results: dict = {}
    wall: dict = {}
    for policy in POLICIES + ["opt"]:
        results[policy] = {}
        wall[policy] = {}
        for cap in capacities:
            t0 = time.perf_counter()
            ratio = (
                opt_hit_ratio(accesses, cap)
                if policy == "opt"
                else pool_hit_ratio(policy, cap, accesses)
            )
            wall[policy][str(cap)] = round(time.perf_counter() - t0, 3)
            results[policy][str(cap)] = round(float(ratio), 5)
            print(f"  {policy:<7} cap={cap:>4} ({cap / N_PAGES * 100:>3.0f}%)  hit={ratio:.3f}")

    payload = {
        "meta": {
            "n_pages": N_PAGES,
            "n_access": N_ACCESS,
            "zipf_s": ZIPF_S,
            "seed": SEED,
            "capacity_pct": CAPACITY_PCTS,
            "note": "opt = Belady offline optimum, simulated separately (needs the future)",
        },
        "hit_ratio": results,
        "wall_s": wall,
    }
    path = os.path.join(RESULTS, "buffer_pool_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"buffer benchmark -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
