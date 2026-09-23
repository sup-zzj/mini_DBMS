"""Fixed-size page buffers and the per-relation disk manager.

This is the lowest layer of the storage engine:

* ``PAGE_SIZE`` is the default physical page size in bytes.
* :class:`Page` is an in-memory wrapper around a fixed-size byte buffer.
  Pages are owned by the buffer pool and are never created by callers.
* :class:`DiskManager` owns a single data file (one per relation) and is
  responsible for page allocation and raw page I/O.  A free list enables
  reuse of deallocated pages.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

PAGE_SIZE = 4096


class Page:
    """An in-memory page.

    Attributes
    ----------
    rel : str
        Name of the relation (table / index) this page belongs to.
    page_id : int
        Identifier of the page *within* ``rel`` (block index).
    buffer : bytearray
        The raw ``PAGE_SIZE``-byte buffer.
    pin_count : int
        Number of holders currently using the page.  A page with a positive
        pin count can never be evicted.
    dirty : bool
        Whether the in-memory buffer differs from the copy on disk.
    ref : bool
        Reference bit used by the clock (approx. LRU) replacer.
    """

    __slots__ = ("rel", "page_id", "buffer", "pin_count", "dirty", "ref")

    def __init__(self, rel: str, page_id: int) -> None:
        self.rel = rel
        self.page_id = page_id
        self.buffer = bytearray(PAGE_SIZE)
        self.pin_count = 0
        self.dirty = False
        self.ref = False

    @property
    def key(self) -> "tuple[str, int]":
        return (self.rel, self.page_id)

    def __repr__(self) -> str:
        return (
            f"Page(rel={self.rel!r}, id={self.page_id}, "
            f"pin={self.pin_count}, dirty={self.dirty})"
        )


class DiskManager:
    """Raw page storage for a single relation file.

    The file is a flat sequence of ``page_size`` blocks; ``page_id`` equals
    the block index.  ``in_memory=True`` keeps buffers in a dict instead of
    touching the file system (used by the buffer-pool benchmarks).
    """

    def __init__(
        self,
        path: str,
        page_size: int = PAGE_SIZE,
        in_memory: bool = False,
    ) -> None:
        self.path = path
        self.page_size = page_size
        self.in_memory = in_memory
        self._num_pages = 0
        self._free: List[int] = []
        self._mem: Dict[int, bytearray] = {}
        if not in_memory:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            if not os.path.exists(path):
                with open(path, "wb"):
                    pass
            self._fh = open(path, "r+b")
            self._num_pages = os.path.getsize(path) // page_size

    # ------------------------------------------------------------------
    # page allocation
    # ------------------------------------------------------------------
    def allocate_page(self) -> int:
        """Return a page id.  Reuses a freed id when possible."""
        if self._free:
            return self._free.pop()
        pid = self._num_pages
        self._num_pages += 1
        return pid

    def deallocate_page(self, page_id: int) -> None:
        """Return ``page_id`` to the free list; shrink the file when it was
        the trailing block."""
        if self.in_memory:
            self._mem.pop(page_id, None)
        if page_id == self._num_pages - 1:
            self._num_pages -= 1
        elif page_id not in self._free:
            self._free.append(page_id)

    # ------------------------------------------------------------------
    # raw I/O
    # ------------------------------------------------------------------
    def read_page(self, page_id: int) -> bytes:
        """Read one page from disk.  Out-of-range ids read as zero pages."""
        if self.in_memory:
            buf = self._mem.get(page_id)
            if buf is None:
                return b"\x00" * self.page_size
            return bytes(buf)
        self._fh.seek(page_id * self.page_size)
        data = self._fh.read(self.page_size)
        if len(data) < self.page_size:
            data += b"\x00" * (self.page_size - len(data))
        return data

    def write_page(self, page_id: int, data: bytes) -> None:
        """Write one page to disk (buffered in the OS; call :meth:`sync`
        to force it out)."""
        if len(data) != self.page_size:
            raise ValueError(f"page size mismatch: got {len(data)} bytes")
        if self.in_memory:
            self._mem[page_id] = bytearray(data)
            return
        self._fh.seek(page_id * self.page_size)
        self._fh.write(data)

    def sync(self) -> None:
        """Force buffered writes to stable storage (fsync)."""
        if not self.in_memory:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self.in_memory:
            self._fh.close()

    @property
    def num_pages(self) -> int:
        return self._num_pages

    def __repr__(self) -> str:
        return f"DiskManager(path={self.path!r}, pages={self._num_pages}, free={len(self._free)})"
