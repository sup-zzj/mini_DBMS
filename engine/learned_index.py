"""A two-level Recursive Model Index (RMI) over sorted keys.

The learned-index literature (Kraska et al., *The Case for Learned Index
Structures*, SIGMOD 2018) proposes replacing a search tree with a *model*
that predicts a key's position, plus a small bounded search to correct the
prediction.  This module is the honest, minimal version used by the
benchmarks:

* level 1: ``level1_size`` equal-width segments; a key is routed to a
  segment by binary search over the segment-boundary keys;
* level 2: within each segment, a greedy piecewise-linear fit — every
  piece predicts positions with error <= ``threshold``;
* a final exact search over the window ``[predicted - threshold - 1,
  predicted + threshold + 1]``.  Because the fitted pieces bound the error,
  the window always contains the true position, so lookups stay exact.

The index is **static** — it is built once over a fixed key set and only
answers lookups.  Inserts are deliberately unsupported (documented
limitation; a real learned index needs delta buffers / remapping).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# bytes per stored scalar used by memory_estimate (8-byte int/float slots)
_ITEM_BYTES = 8


@dataclass
class _Piece:
    """One linear segment: ``predicted_pos = start_pos + slope * (key - start_key)``."""

    start_key: int
    start_pos: int
    slope: float


class LearnedIndex:
    """A static two-level RMI.

    Parameters
    ----------
    keys : sequence of int
        Sorted, unique keys (the data set the index is trained on).
    level1_size : int
        Number of level-1 segments (fanout of the top model).
    threshold : int
        Maximum absolute position error allowed for a level-2 piece; the
        exact search window is derived from it.
    """

    def __init__(self, keys, level1_size: int = 64, threshold: int = 16) -> None:
        self.keys = np.asarray(keys, dtype=np.int64)
        if self.keys.ndim != 1 or len(self.keys) == 0:
            raise ValueError("keys must be a non-empty 1-D array")
        self.level1_size = max(1, int(level1_size))
        self.threshold = max(1, int(threshold))
        self._built = False
        self._boundaries: Optional[np.ndarray] = None
        self._seg_starts: List[int] = []
        self._pieces: List[List[_Piece]] = []
        self._piece_keys: List[List[int]] = []  # cached piece start keys

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------
    def build(self) -> "LearnedIndex":
        n = len(self.keys)
        s = min(self.level1_size, n)
        # level-1 boundaries at equally spaced *positions*
        positions = np.linspace(0, n - 1, s + 1).astype(np.int64)
        self._boundaries = self.keys[positions]
        self._seg_starts = [(i * n) // s for i in range(s + 1)]
        self._pieces = []
        self._piece_keys = []
        for i in range(s):
            pieces = self._fit_pieces(self._seg_starts[i], self._seg_starts[i + 1])
            self._pieces.append(pieces)
            self._piece_keys.append([p.start_key for p in pieces])
        self._built = True
        return self

    def _fit_pieces(self, lo: int, hi: int) -> List[_Piece]:
        """Greedy piecewise-linear fit over ``keys[lo:hi]`` with max error
        <= threshold.  O(piece_len^2) worst case per piece; for sorted keys
        each piece typically covers the whole segment in one pass."""
        pieces: List[_Piece] = []
        keys = self.keys
        i = lo
        while i < hi:
            start_key, start_pos = int(keys[i]), i
            j = i
            while j + 1 < hi:
                j += 1
                if keys[j] == start_key:
                    slope = 0.0
                else:
                    slope = (j - start_pos) / (keys[j] - start_key)
                if self._max_err(start_key, start_pos, slope, i, j) > self.threshold:
                    j -= 1
                    break
            if j == i:
                slope = 0.0
            elif keys[j] != start_key:
                slope = (j - start_pos) / (keys[j] - start_key)
            else:
                slope = 0.0
            pieces.append(_Piece(start_key, start_pos, slope))
            i = j + 1
        return pieces

    def _max_err(self, start_key: int, start_pos: int, slope: float, i: int, j: int) -> float:
        keys = self.keys[i : j + 1]
        pred = start_pos + slope * (keys - start_key)
        actual = np.arange(i, j + 1)
        return float(np.max(np.abs(pred - actual)))

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    def lookup(self, key: int) -> int:
        """Return the insertion position of ``key`` (``keys[pos] >= key``),
        the same semantics as :func:`bisect.bisect_left`.

        The hot path is pure Python ``bisect`` (no numpy per-call overhead)
        so lookups are comparable with the other index variants on equal
        implementation footing."""
        if not self._built:
            raise RuntimeError("call build() before lookup()")
        keys = self.keys
        # level 1: route to a segment by boundary keys
        seg = bisect.bisect_right(self._boundaries, key) - 1
        if seg < 0:
            seg = 0
        elif seg >= len(self._pieces):
            seg = len(self._pieces) - 1
        # level 2: pick a piece, predict, then correct with a bounded search
        pieces = self._pieces[seg]
        if pieces:
            idx = bisect.bisect_right(self._piece_keys[seg], key) - 1
            piece = pieces[max(0, idx)]
            predicted = int(piece.start_pos + piece.slope * (key - piece.start_key))
        else:
            predicted = self._seg_starts[seg]
        lo = max(0, predicted - self.threshold - 1)
        hi = min(len(keys) - 1, predicted + self.threshold + 1)
        return bisect.bisect_left(keys, key, lo, hi + 1)

    def search(self, key: int) -> bool:
        """Membership test — True iff ``key`` is present."""
        pos = self.lookup(key)
        return pos < len(self.keys) and int(self.keys[pos]) == key

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def memory_estimate(self) -> float:
        """Approximate resident bytes of the index structures (excludes the
        key array itself, which all three benchmark variants share)."""
        if not self._built:
            return 0.0
        total = 0.0
        total += self._boundaries.nbytes  # level-1 boundary keys
        total += len(self._seg_starts) * _ITEM_BYTES
        n_pieces = sum(len(p) for p in self._pieces)
        total += n_pieces * 3 * _ITEM_BYTES  # start_key / start_pos / slope
        total += (len(self._pieces) + 1) * _ITEM_BYTES  # piece offsets
        return total

    def __repr__(self) -> str:
        n = sum(len(p) for p in self._pieces) if self._built else 0
        return (
            f"LearnedIndex(keys={len(self.keys)}, segments={self.level1_size}, "
            f"pieces={n}, threshold={self.threshold})"
        )
