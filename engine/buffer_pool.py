"""Buffer pool: the in-memory cache between the engine and disk.

Responsibilities
----------------
* Cache fixed-size pages keyed by ``(relation, page_id)``.
* Reference counting ("pinning"): a page with ``pin_count > 0`` cannot be
  evicted.  The transaction layer pins every page it touches so that the
  *no-steal* policy holds (uncommitted changes never reach disk).
* Dirty-page tracking: pages modified in memory are flushed lazily on
  eviction or by an explicit flush (checkpoint / shutdown).
* Pluggable eviction policies: exact LRU, clock (approx. LRU), random.
* Hit/miss and I/O statistics used by the buffer-pool benchmark.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Dict, List, Optional

from .page import PAGE_SIZE, DiskManager, Page


class BufferPoolFullError(RuntimeError):
    """Raised when no evictable (unpinned) frame exists."""


class BufferPool:
    """A capacity-bounded page cache over one or more relation files.

    Parameters
    ----------
    capacity : int
        Maximum number of resident pages.
    policy : {"clock", "lru", "random"}
        Eviction policy.  ``clock`` approximates LRU with a reference bit.
    seed : int
        Seed for the random policy (reproducible experiments).
    """

    def __init__(
        self,
        capacity: int,
        page_size: int = PAGE_SIZE,
        policy: str = "clock",
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("buffer pool capacity must be >= 1")
        self.capacity = capacity
        self.page_size = page_size
        self.policy = policy
        self._rng = random.Random(seed)
        self.disks: Dict[str, DiskManager] = {}
        self.pages: Dict[tuple, Page] = {}
        self._lru: OrderedDict[tuple, None] = OrderedDict()
        self._hand = 0  # clock hand (index into a snapshot)
        self._stats = {"hits": 0, "misses": 0, "reads": 0, "writes": 0}

    # ------------------------------------------------------------------
    # relation registration
    # ------------------------------------------------------------------
    def register_relation(self, name: str, disk: DiskManager) -> None:
        if name in self.disks:
            raise ValueError(f"relation {name!r} already registered")
        self.disks[name] = disk

    def unregister_relation(self, name: str) -> None:
        """Drop a relation: discard resident pages without write-back."""
        for key in [k for k in self.pages if k[0] == name]:
            del self.pages[key]
        self.disks.pop(name, None)

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def allocate_page(self, rel: str) -> Page:
        """Allocate a fresh page id from ``rel``'s disk manager and return
        the (pinned, zeroed) page resident in the pool.  Evicts a victim
        first if the pool is already full."""
        disk = self._disk(rel)
        page_id = disk.allocate_page()
        page = Page(rel, page_id)
        page.pin_count = 1
        if len(self.pages) >= self.capacity:
            self._evict()
        self._admit(page)
        return page

    def fetch_page(self, rel: str, page_id: int) -> Page:
        """Return the page, pinning it.  Misses are served from disk."""
        key = (rel, page_id)
        page = self.pages.get(key)
        if page is not None:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
            page = self._load(key)
        page.pin_count += 1
        if self.policy == "lru":
            self._lru.move_to_end(key)
        elif self.policy == "clock":
            page.ref = True
        return page

    def unpin(self, page: Page, dirty: bool = False) -> None:
        """Release one reference.  Mark dirty so it is written back later."""
        if dirty:
            page.dirty = True
        if page.pin_count > 0:
            page.pin_count -= 1
        if self.policy == "lru" and page.pin_count == 0:
            self._lru.move_to_end(page.key)

    def flush_page(self, page: Page) -> None:
        """Write a dirty page back to disk immediately."""
        if page.dirty:
            disk = self._disk(page.rel)
            disk.write_page(page.page_id, bytes(page.buffer))
            page.dirty = False
            self._stats["writes"] += 1

    def flush_all(self) -> int:
        """Write back every dirty page; returns the number of pages written."""
        written = 0
        for page in list(self.pages.values()):
            if page.dirty:
                self.flush_page(page)
                written += 1
        return written

    def delete_page(self, rel: str, page_id: int, flush: bool = False) -> None:
        """Permanently remove a page.  By default the buffer is discarded
        (DROP TABLE semantics); with ``flush=True`` dirty content is written
        before removal."""
        key = (rel, page_id)
        page = self.pages.pop(key, None)
        if page is not None and flush and page.dirty:
            self.flush_page(page)
        if page is not None:
            self._lru.pop(key, None)
        self._disk(rel).deallocate_page(page_id)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _disk(self, rel: str) -> DiskManager:
        try:
            return self.disks[rel]
        except KeyError:
            raise KeyError(f"no disk registered for relation {rel!r}") from None

    def _load(self, key: tuple) -> Page:
        rel, page_id = key
        if len(self.pages) >= self.capacity:
            self._evict()
        page = Page(rel, page_id)
        page.buffer[:] = self._disk(rel).read_page(page_id)
        self._stats["reads"] += 1
        self._admit(page)
        return page

    def _admit(self, page: Page) -> None:
        self.pages[page.key] = page
        if self.policy == "lru":
            self._lru[page.key] = None

    def _evict(self) -> None:
        victim = None
        if self.policy == "lru":
            for key in self._lru:
                if self.pages[key].pin_count == 0:
                    victim = self.pages[key]
                    break
        elif self.policy == "random":
            candidates = [p for p in self.pages.values() if p.pin_count == 0]
            if candidates:
                victim = self._rng.choice(candidates)
        else:  # clock
            victim = self._clock_victim()
        if victim is None:
            raise BufferPoolFullError(
                "no unpinned frame available; pool too small for the "
                "active working set"
            )
        self.flush_page(victim) if victim.dirty else None
        del self.pages[victim.key]
        self._lru.pop(victim.key, None)

    def _clock_victim(self) -> Optional[Page]:
        frames = list(self.pages.values())
        n = len(frames)
        if n == 0:
            return None
        # two revolutions: the first clears reference bits, the second
        # actually evicts.  A single revolution would fail when every
        # frame had been recently touched (ref == True).
        for _ in range(2 * n):
            page = frames[self._hand % n]
            self._hand = (self._hand + 1) % n
            if page.pin_count > 0:
                continue
            if page.ref:
                page.ref = False
                continue
            return page
        return None

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0

    def hit_ratio(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total else 0.0

    @property
    def resident(self) -> int:
        return len(self.pages)

    def __repr__(self) -> str:
        return (
            f"BufferPool(capacity={self.capacity}, policy={self.policy!r}, "
            f"resident={self.resident})"
        )
