"""Reproducible key-distribution generators for the learned-index benchmarks.

Each generator draws ``n`` *sorted, unique* integer keys from a ``keyspace``
and returns a ``numpy.int64`` array.  All functions are pure and accept an
explicit ``numpy.random.Generator`` so callers can fix a seed once and get
bit-for-bit reproducible data sets.

Distributions
-------------
* ``gen_uniform``   — i.i.d. uniform sampling (the classic benchmark setup).
* ``gen_clustered`` — draws ``n_clusters`` centers then samples keys densely
  around each center with bounded jitter; models "hot ranges" in real data.
* ``gen_zipf``      — samples keys concentrated in the low / popular ranks of
  a zipf popularity distribution; models skew / long-tail data.

Every generator returns **exactly** ``n`` keys (a fill loop tops up the
shortfall left by de-duplication) with ``0 <= key < keyspace`` and strictly
increasing order.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _finish_unique(rng: np.random.Generator, n: int, keyspace: int,
                   sampler: Callable[[int], np.ndarray]) -> np.ndarray:
    """Collect exactly ``n`` unique keys by pulling from ``sampler`` until the
    unique set reaches size ``n`` (deterministic given ``rng``).  Returns a
    sorted ``int64`` array."""
    got = np.empty(0, dtype=np.int64)
    cur = 0
    while cur < n:
        batch = sampler(n - cur)
        got = np.unique(np.concatenate([got, batch]))
        got = np.sort(got)
        cur = len(got)
    if len(got) > n:
        got = got[:n]
    if got.size == 0 or got[0] < 0 or got[-1] >= keyspace:
        raise ValueError("key generator produced out-of-range keys")
    return got


def gen_uniform(rng: np.random.Generator, n: int, keyspace: int = 2**48) -> np.ndarray:
    """``n`` unique keys drawn uniformly from ``[0, keyspace)``, sorted."""
    if keyspace < n:
        raise ValueError("keyspace must be >= n for unique keys")
    keys = rng.choice(np.int64(keyspace), n, replace=False)
    return np.sort(keys).astype(np.int64)


def gen_clustered(
    rng: np.random.Generator,
    n: int,
    n_clusters: int = 200,
    spread: int = 2000,
    keyspace: int = 2**48,
) -> np.ndarray:
    """Keys clustered around ``n_clusters`` random centers with a few hundred
    usable slots around each; models "hot ranges" with lots of repeats."""
    if keyspace < n:
        raise ValueError("keyspace must be >= n for unique keys")
    centers = rng.choice(np.int64(keyspace), n_clusters, replace=False)

    def _sample(k: int) -> np.ndarray:
        cluster_idx = rng.integers(0, len(centers), size=k)
        jitter = rng.integers(-spread, spread, size=k, dtype=np.int64)
        return (centers[cluster_idx] + jitter) % np.int64(keyspace)

    return _finish_unique(rng, n, keyspace, _sample)


def gen_zipf(rng: np.random.Generator, n: int, s: float = 1.2, keyspace: int = 2**48) -> np.ndarray:
    """Zipf-skewed keys: maps a zipf popularity sample to a dense low-key range
    so popular ranks concentrate near zero and the tail stretches to the full
    keyspace (skew / long-tail data).

    The rank->key map ``floor(rank * keyspace / n)`` is computed without
    int64 overflow by splitting the division (``rank <= n`` after clamping)."""
    if keyspace < n:
        raise ValueError("keyspace must be >= n for unique keys")
    span = np.int64(keyspace)
    q = span // np.int64(n)
    r = span % np.int64(n)

    def _sample(k: int) -> np.ndarray:
        ranks = rng.zipf(s, size=k).astype(np.int64)
        ranks = np.minimum(ranks, np.int64(n))  # clamp tail ranks
        base = ranks * q + (ranks * r) // np.int64(n)
        jitter = rng.integers(0, np.int64(n), size=k, dtype=np.int64)
        return (base + jitter) % span

    return _finish_unique(rng, n, keyspace, _sample)