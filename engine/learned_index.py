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

Exactness
---------
The greedy fit bounds the prediction error by ``threshold`` **only over the
keys each piece was fitted on**.  A query key may fall in a *gap* — between
two pieces, or after a piece's last trained key (skewed / clustered data
produces many such gaps).  For those keys the linear model would extrapolate
unboundedly, so lookups detect them and fall back to a segment-scoped
``bisect`` (the true position of any key routed to a segment is provably
inside ``[seg_starts[seg], seg_starts[seg+1]]``).  Lookups therefore stay
exact for *every* key, trained or not.
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
        self._piece_start_poss: List[List[int]] = []  # piece start positions
        self._piece_last_poss: List[List[int]] = []  # last fitted position per piece

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
        self._piece_start_poss = []
        for i in range(s):
            pieces = self._fit_pieces(self._seg_starts[i], self._seg_starts[i + 1])
            self._pieces.append(pieces)
            self._piece_keys.append([p.start_key for p in pieces])
            self._piece_start_poss.append([p.start_pos for p in pieces])
            # last trained position per piece: previous to the next piece's
            # start, or the segment end for the final piece
            seg_hi = self._seg_starts[i + 1]
            self._piece_last_poss.append(
                [pieces[k + 1].start_pos - 1 if k + 1 < len(pieces) else seg_hi - 1
                 for k in range(len(pieces))]
            )
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
    def _route_and_predict(self, key: int) -> int:
        """Route ``key`` through both models and return a predicted position
        whose error against the true insertion point is ``<= threshold + 1`` for
        *every* key, not only the trained ones.

        Keys that fall inside a fitted piece are interpolated (error bound by
        ``threshold``).  Keys in a *gap* — between two adjacent pieces, or beyond
        a piece's last trained key (skewed / clustered data produces many such
        gaps) — would otherwise make the linear model extrapolate unboundedly, so
        they fall back to a segment-scoped ``bisect``, return the exact position.
        The raw prediction is also clamped to the selected piece's fitted position
        span so it always points into a valid array region.
        """
        # level 1: route to a segment by boundary keys
        seg = bisect.bisect_right(self._boundaries, key) - 1
        if seg < 0:
            seg = 0
        elif seg >= len(self._pieces):
            seg = len(self._pieces) - 1
        lo_pos = self._seg_starts[seg]
        hi_pos = self._seg_starts[seg + 1]
        pieces = self._pieces[seg]
        if not pieces:
            return lo_pos
        idx = max(0, bisect.bisect_right(self._piece_keys[seg], key) - 1)
        # fitted key range of the selected piece
        key_lo = self._piece_keys[seg][idx]
        if idx + 1 < len(pieces):
            key_hi = self._piece_keys[seg][idx + 1] - 1
        else:
            key_hi = int(self.keys[hi_pos - 1])
        if key < key_lo or key > key_hi:
            # gap key: fall back to the exact segment-scoped search
            return bisect.bisect_left(self.keys, key, lo_pos, hi_pos)
        piece = pieces[idx]
        predicted = int(piece.start_pos + piece.slope * (key - piece.start_key))
        pos_lo = self._piece_start_poss[seg][idx]
        pos_hi = self._piece_last_poss[seg][idx]
        return min(pos_hi, max(pos_lo, predicted))

    def predict(self, key: int) -> int:
        """Return the raw model-predicted insertion position of ``key``
        (before the bounded exact search).  Exposed so benchmarks can measure
        the prediction-error distribution ``|predict(key) - bisect(key)|``."""
        if not self._built:
            raise RuntimeError("call build() before predict()")
        return self._route_and_predict(key)

    def lookup(self, key: int) -> int:
        """Return the insertion position of ``key`` (``keys[pos] >= key``),
        the same semantics as :func:`bisect.bisect_left`.

        The hot path is pure Python ``bisect`` (no numpy per-call overhead)
        so lookups are comparable with the other index variants on equal
        implementation footing."""
        if not self._built:
            raise RuntimeError("call build() before lookup()")
        keys = self.keys
        predicted = self._route_and_predict(key)
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


class BlockLearnedIndex(LearnedIndex):
    """A *block-routing* learned index: the model predicts the block (page)
    a key lives in, then a small bounded search runs only inside that block's
    candidate window.

    This mirrors the setting where learned indexes are supposed to win (Kraska
    et al., SIGMOD 2018): the model's job is to *locate the page*, avoiding a
    full binary search that would walk many cache-line unfriendly pages, rather
    than to pin-point the exact in-array position.

    Exactness
    ---------
    The base RMI guarantees ``|predict - truth| <= threshold + 1``.  A key
    whose true position is at offset ``truth`` lives in block
    ``truth // block_size``.  If the predicted position ``p`` is in block
    ``pb = p // block_size``, then the true block index differs from ``pb`` by
    at most ``ceil((threshold + 1) / block_size)``.  Choosing
    ``block_extra = (threshold + block_size) // block_size + 1`` therefore
    guarantees the candidate block window always contains the true block, so
    lookups stay exact.
    """

    def __init__(
        self,
        keys,
        level1_size: int = 64,
        threshold: int = 16,
        block_size: int = 64,
        block_extra: Optional[int] = None,
    ) -> None:
        super().__init__(keys, level1_size, threshold)
        self.block_size = max(1, int(block_size))
        if block_extra is None:
            block_extra = (self.threshold + self.block_size) // self.block_size + 1
        self.block_extra = max(0, int(block_extra))
        self._nblocks: int = 0

    def build(self) -> "BlockLearnedIndex":
        super().build()
        # ceil: include the final partial block so its keys are searchable
        self._nblocks = (len(self.keys) + self.block_size - 1) // self.block_size
        return self

    def predict_block(self, key: int) -> int:
        """Return the block the model predicts ``key`` lives in."""
        if not self._built:
            raise RuntimeError("call build() before predict_block()")
        pred = self._route_and_predict(key)
        pb = pred // self.block_size
        return max(0, min(self._nblocks - 1, pb))

    def lookup(self, key: int) -> int:
        """Route to a predicted block, then run the exact ``bisect_left``
        only inside the candidate block window that provably contains the key."""
        if not self._built:
            raise RuntimeError("call build() before lookup()")
        keys = self.keys
        pb = self.predict_block(key)
        # Candidate block window [lo_b, hi_b]; the true block is guaranteed here.
        lo_b = max(0, pb - self.block_extra)
        hi_b = min(self._nblocks - 1, pb + self.block_extra)
        lo = max(0, lo_b * self.block_size)
        hi = min(len(keys) - 1, (hi_b + 1) * self.block_size - 1)
        return bisect.bisect_left(keys, key, lo, hi + 1)

    def memory_estimate(self) -> float:
        base = super().memory_estimate()
        if not self._built:
            return base
        return base + self._nblocks * _ITEM_BYTES  # block offsets

    def __repr__(self) -> str:
        n = sum(len(p) for p in self._pieces) if self._built else 0
        return (
            f"BlockLearnedIndex(keys={len(self.keys)}, segments={self.level1_size}, "
            f"blocks={self._nblocks}, block_size={self.block_size}, "
            f"block_extra={self.block_extra}, pieces={n}, threshold={self.threshold})"
        )
