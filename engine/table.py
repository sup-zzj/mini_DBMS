"""Heap table storage: slotted pages over the buffer pool.

Row layout
----------
Rows are packed into fixed-size pages using the classic *slotted page*
scheme.  A page header points to a slot array growing from the front of
the page and to a free-space area growing from the back::

    +---------------------------------------------------------------+
    | header | slot0 | slot1 | ... |  free space  | rec2 | rec1 |    |
    +---------------------------------------------------------------+
              ^ slot_end  (grows up)   ^ free_start (grows down)

Each slot is ``(offset, length)``; the high bit of ``length`` marks a
tombstone (deleted) record while the offset is kept, so an undone delete
only has to clear the flag.  The heap is append-only — deleted space is
reclaimed by dropping the table (no vacuum in v1).

A record id is ``(page_id << 32) | slot_index``, which is what the B+ tree
index stores as its value.

Row encoding (little-endian, column order from the catalog)::

    INT  -> <q      (8 bytes)
    REAL -> <d      (8 bytes)
    TEXT -> <I len  + utf-8 bytes
"""

from __future__ import annotations

import struct
from typing import Callable, Iterator, List, Optional, Tuple

from .buffer_pool import BufferPool
from .catalog import Column, INT, REAL, TEXT
from .page import PAGE_SIZE

PAGE_TYPE_HEAP = 1
PAGE_TYPE_BTREE_LEAF = 2
PAGE_TYPE_BTREE_INTERNAL = 3

HEAP_HEADER_SIZE = 32
SLOT_SIZE = 8
INVALID_RECORD_ID = 0

# slot length high bit = tombstone flag; low 31 bits = payload length
SLOT_DELETED = 0x80000000
SLOT_LEN_MASK = 0x7FFFFFFF

# header field offsets (u8 / u32 / u16 helpers below)
_OFF_TYPE = 0
_OFF_NEXT = 1
_OFF_SLOTS = 5
_OFF_FREE_START = 7
_OFF_SLOT_END = 11

ModifyHook = Callable[[str, int], None]


def _hdr(page, offset: int, fmt: str, value=None):
    size = struct.calcsize(fmt)
    if value is None:
        return struct.unpack_from(fmt, page.buffer, offset)[0]
    struct.pack_into(fmt, page.buffer, offset, value)
    return None


# ----------------------------------------------------------------------
# row encoding
# ----------------------------------------------------------------------
def encode_row(columns: List[Column], values: List) -> bytes:
    """Pack a row according to the table's column list."""
    out = bytearray()
    for col, val in zip(columns, values):
        if col.type == INT:
            out += struct.pack("<q", int(val))
        elif col.type == REAL:
            out += struct.pack("<d", float(val))
        elif col.type == TEXT:
            raw = str(val).encode("utf-8")
            out += struct.pack("<I", len(raw)) + raw
        else:
            raise ValueError(f"unsupported column type {col.type!r}")
    return bytes(out)


def decode_row(columns: List[Column], data: bytes) -> List:
    """Unpack a row, returning the value list (and validating the tail)."""
    values: List = []
    pos = 0
    for col in columns:
        if col.type == INT:
            values.append(struct.unpack_from("<q", data, pos)[0])
            pos += 8
        elif col.type == REAL:
            values.append(struct.unpack_from("<d", data, pos)[0])
            pos += 8
        elif col.type == TEXT:
            (n,) = struct.unpack_from("<I", data, pos)
            pos += 4
            values.append(data[pos : pos + n].decode("utf-8"))
            pos += n
        else:
            raise ValueError(f"unsupported column type {col.type!r}")
    if pos != len(data):
        raise ValueError(f"trailing bytes in row encoding ({len(data) - pos})")
    return values


# ----------------------------------------------------------------------
# heap page helpers
# ----------------------------------------------------------------------
def init_heap_page(page) -> None:
    page.buffer[_OFF_TYPE] = PAGE_TYPE_HEAP
    _hdr(page, _OFF_NEXT, "<I", 0)
    _hdr(page, _OFF_SLOTS, "<H", 0)
    _hdr(page, _OFF_FREE_START, "<I", PAGE_SIZE)
    _hdr(page, _OFF_SLOT_END, "<I", HEAP_HEADER_SIZE)


def _slot_at(page, slot: int) -> Tuple[int, int, bool]:
    """Return ``(offset, live_length, deleted)`` for a slot."""
    base = HEAP_HEADER_SIZE + slot * SLOT_SIZE
    offset, raw = struct.unpack_from("<II", page.buffer, base)
    return offset, raw & SLOT_LEN_MASK, bool(raw & SLOT_DELETED)


def _set_slot(page, slot: int, offset: int, raw_length: int) -> None:
    base = HEAP_HEADER_SIZE + slot * SLOT_SIZE
    struct.pack_into("<II", page.buffer, base, offset, raw_length)


def _live_raw(length: int) -> int:
    return length & SLOT_LEN_MASK


def _tomb_raw(length: int) -> int:
    return (length & SLOT_LEN_MASK) | SLOT_DELETED


def _fits(page, row_len: int) -> bool:
    free_start = _hdr(page, _OFF_FREE_START, "<I")
    slot_end = _hdr(page, _OFF_SLOT_END, "<I")
    return slot_end + SLOT_SIZE <= free_start - row_len


def _append_slot(page, row: bytes) -> int:
    """Append a record to a page; caller must have checked ``_fits``."""
    free_start = _hdr(page, _OFF_FREE_START, "<I")
    slot_count = _hdr(page, _OFF_SLOTS, "<H")
    offset = free_start - len(row)
    page.buffer[offset : offset + len(row)] = row
    _set_slot(page, slot_count, offset, len(row))
    _hdr(page, _OFF_FREE_START, "<I", offset)
    _hdr(page, _OFF_SLOT_END, "<I", HEAP_HEADER_SIZE + (slot_count + 1) * SLOT_SIZE)
    _hdr(page, _OFF_SLOTS, "<H", slot_count + 1)
    return slot_count


def _read_row_at(page, slot: int) -> Optional[bytes]:
    offset, length, deleted = _slot_at(page, slot)
    if length == 0 or deleted:
        return None  # empty slot / tombstone
    return bytes(page.buffer[offset : offset + length])


# ----------------------------------------------------------------------
# table operations (all operate through the shared buffer pool)
# ----------------------------------------------------------------------
def create_table_pages(pool: BufferPool, rel: str) -> int:
    """Allocate the heap head page for a new table; returns its page id."""
    page = pool.allocate_page(rel)
    init_heap_page(page)
    root = page.page_id
    pool.unpin(page)
    return root


def insert_row(
    pool: BufferPool,
    rel: str,
    root_page: int,
    row: bytes,
    on_modify: Optional[ModifyHook] = None,
) -> int:
    """Append a row, allocating pages as needed.  Returns the record id."""
    cur = root_page
    while True:
        page = pool.fetch_page(rel, cur)
        if _fits(page, len(row)):
            slot = _append_slot(page, row)
            page.dirty = True
            record_id = (page.page_id << 32) | slot
            pool.unpin(page)
            if on_modify:
                on_modify(rel, page.page_id)
            return record_id
        nxt = _hdr(page, _OFF_NEXT, "<I")
        if nxt == 0:
            fresh = pool.allocate_page(rel)
            init_heap_page(fresh)
            _hdr(page, _OFF_NEXT, "<I", fresh.page_id)
            page.dirty = True
            slot = _append_slot(fresh, row)
            fresh.dirty = True
            record_id = (fresh.page_id << 32) | slot
            pool.unpin(fresh)
            pool.unpin(page)
            if on_modify:
                on_modify(rel, page.page_id)
                on_modify(rel, fresh.page_id)
            return record_id
        pool.unpin(page)
        cur = nxt


def read_row(pool: BufferPool, rel: str, record_id: int) -> Optional[bytes]:
    """Read a row by record id; None if tombstoned / out of range."""
    page_id = record_id >> 32
    slot = record_id & 0xFFFFFFFF
    page = pool.fetch_page(rel, page_id)
    try:
        count = _hdr(page, _OFF_SLOTS, "<H")
        if slot >= count:
            return None
        return _read_row_at(page, slot)
    finally:
        pool.unpin(page)


def tombstone(pool: BufferPool, rel: str, record_id: int, on_modify: Optional[ModifyHook] = None) -> bool:
    """Mark a record deleted (set the deleted bit, keep the offset so an
    undo can restore it in place).  Returns False if already gone."""
    page_id = record_id >> 32
    slot = record_id & 0xFFFFFFFF
    page = pool.fetch_page(rel, page_id)
    try:
        count = _hdr(page, _OFF_SLOTS, "<H")
        if slot >= count:
            return False
        offset, length, deleted = _slot_at(page, slot)
        if length == 0 or deleted:
            return False
        _set_slot(page, slot, offset, _tomb_raw(length))
        page.dirty = True
    finally:
        pool.unpin(page)
    if on_modify:
        on_modify(rel, page_id)
    return True


def restore_row(
    pool: BufferPool,
    rel: str,
    record_id: int,
    row: bytes,
    on_modify: Optional[ModifyHook] = None,
) -> bool:
    """Undo a tombstone: clear the deleted bit (the row bytes are still
    in place)."""
    page_id = record_id >> 32
    slot = record_id & 0xFFFFFFFF
    page = pool.fetch_page(rel, page_id)
    try:
        count = _hdr(page, _OFF_SLOTS, "<H")
        if slot >= count:
            return False
        offset, length, deleted = _slot_at(page, slot)
        if length == 0 or not deleted:
            return False  # never tombstoned; refuse to clobber
        _set_slot(page, slot, offset, _live_raw(length))
        page.dirty = True
    finally:
        pool.unpin(page)
    if on_modify:
        on_modify(rel, page_id)
    return True


def insert_row_at(
    pool: BufferPool,
    rel: str,
    record_id: int,
    row: bytes,
    on_modify: Optional[ModifyHook] = None,
) -> bool:
    """Redo helper: materialise a row at a specific ``(page, slot)``.

    Used by crash recovery.  Idempotent — if the slot already holds a live
    row the call is a no-op; a tombstoned slot is resurrected; a fresh
    zero page is initialised as a heap page on the fly.
    """
    page_id = record_id >> 32
    slot = record_id & 0xFFFFFFFF
    page = pool.fetch_page(rel, page_id)
    try:
        if _hdr(page, _OFF_TYPE, "<B") != PAGE_TYPE_HEAP:
            init_heap_page(page)
        count = _hdr(page, _OFF_SLOTS, "<H")
        if slot >= count:
            # extend the slot array with tombstones up to the target slot
            for s in range(count, slot):
                _set_slot(page, s, 0, 0)
            _hdr(page, _OFF_SLOTS, "<H", slot + 1)
            _hdr(page, _OFF_SLOT_END, "<I", HEAP_HEADER_SIZE + (slot + 1) * SLOT_SIZE)
            offset, length, deleted = 0, 0, False
        else:
            offset, length, deleted = _slot_at(page, slot)
        if length != 0 and not deleted:
            return False  # already applied
        if length == 0 and offset == 0 and not deleted:
            # never written: place the row in the free-space area
            free_start = _hdr(page, _OFF_FREE_START, "<I")
            offset = free_start - len(row)
            page.buffer[offset : offset + len(row)] = row
            _hdr(page, _OFF_FREE_START, "<I", offset)
        _set_slot(page, slot, offset, len(row))
        page.dirty = True
    finally:
        pool.unpin(page)
    if on_modify:
        on_modify(rel, page_id)
    return True


def scan(pool: BufferPool, rel: str, root_page: int) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(record_id, row_bytes)`` for every live row, in insertion
    order, following the page chain.

    The chain sentinel (next = 0) may equal the root page id (a fresh
    table's root is page 0), so the first page is always visited and a
    ``seen`` set breaks any accidental self-loop.
    """
    cur = root_page
    seen = set()
    first = True
    while (cur != 0 or first) and cur not in seen:
        first = False
        seen.add(cur)
        page = pool.fetch_page(rel, cur)
        try:
            count = _hdr(page, _OFF_SLOTS, "<H")
            for slot in range(count):
                row = _read_row_at(page, slot)
                if row is not None:
                    yield ((cur << 32) | slot, row)
            cur = _hdr(page, _OFF_NEXT, "<I")
        finally:
            pool.unpin(page)


def set_next_page(pool: BufferPool, rel: str, page_id: int, next_page: int) -> None:
    """Rewrite a heap page's next-page pointer (used by chain repair after
    crash recovery)."""
    page = pool.fetch_page(rel, page_id)
    try:
        _hdr(page, _OFF_NEXT, "<I", next_page)
        page.dirty = True
    finally:
        pool.unpin(page)


def drop_table_pages(pool: BufferPool, rel: str, root_page: int) -> None:
    """Free every page in the chain (DROP TABLE).  See :func:`scan` for the
    root-page/sentinel note."""
    cur = root_page
    seen = set()
    first = True
    while (cur != 0 or first) and cur not in seen:
        first = False
        seen.add(cur)
        page = pool.fetch_page(rel, cur)
        try:
            nxt = _hdr(page, _OFF_NEXT, "<I")
        finally:
            pool.unpin(page)
        pool.delete_page(rel, cur)
        cur = nxt
