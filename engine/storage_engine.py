"""The storage engine: ties catalog, buffer pool, WAL, heap and B+ tree
together and exposes the SQL-ish surface used by the frontend.

Design recap
------------
* **no-steal / no-force** — every page a transaction touches is pinned
  (see ``_pin_hook``), so uncommitted changes never reach disk; commits
  only fsync the WAL, dirty pages flush lazily later.
* Every mutating operation appends its redo record to the WAL as it runs
  and registers an undo entry in the transaction.  Rollback replays the
  undo log in reverse (index first, then heap) so data and indexes stay
  consistent.
* UPDATE is DELETE-old + INSERT-new.
* Crash recovery replays the committed redo records, repairs every heap
  page chain (page ids are assigned in insertion order), then rebuilds all
  indexes from a full heap scan.  WAL is truncated only after a checkpoint.
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from .btree import BTree, DuplicateKeyError, PoolNodeStore
from .buffer_pool import BufferPool
from .catalog import Catalog, Column, INT, REAL, TEXT, IndexInfo, TableMeta
from .page import PAGE_SIZE, DiskManager
from .table import (
    PAGE_TYPE_HEAP,
    create_table_pages,
    decode_row,
    drop_table_pages,
    encode_row,
    insert_row,
    insert_row_at,
    read_row,
    restore_row,
    scan,
    set_next_page,
    tombstone,
)
from .txn import Transaction, TxnManager
from .wal import OP_DELETE, OP_INSERT, WAL

DEFAULT_ORDER = 32  # B+ tree order used for on-disk indexes

_OPS = {"=": lambda a, b: a == b, "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


class StorageEngine:
    """One database directory = one engine instance."""

    def __init__(
        self,
        data_dir: str,
        pool_capacity: int = 64,
        policy: str = "clock",
        seed: int = 0,
        order: int = DEFAULT_ORDER,
    ) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.order = order
        self.catalog = Catalog(os.path.join(data_dir, "catalog.json"))
        self.pool = BufferPool(pool_capacity, PAGE_SIZE, policy, seed)
        self.wal = WAL(os.path.join(data_dir, "wal.log"))
        self.txn_mgr = TxnManager(self.wal)
        self._recover()

    # ------------------------------------------------------------------
    # relation / file plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _index_rel(table: str, index: str) -> str:
        return f"{table}__{index}"

    def _rel_path(self, rel: str) -> str:
        return os.path.join(self.data_dir, rel + ".dat")

    def _register_disk(self, rel: str) -> None:
        if rel not in self.pool.disks:
            self.pool.register_relation(rel, DiskManager(self._rel_path(rel)))

    # ------------------------------------------------------------------
    # key (de)serialisation for WAL + B+ tree
    # ------------------------------------------------------------------
    @staticmethod
    def _key_bytes(col: Column, key_obj) -> bytes:
        if col.type == INT:
            return struct.pack("<q", int(key_obj))
        if col.type == TEXT:
            return str(key_obj).encode("utf-8")
        raise ValueError(f"column {col.name!r}: {col.type} keys are not supported")

    @staticmethod
    def _key_obj(col: Column, raw: bytes):
        if col.type == INT:
            return struct.unpack("<q", raw)[0]
        return raw.decode("utf-8")

    def _coerce(self, col: Column, value):
        if col.type == INT:
            if isinstance(value, str):
                value = value.strip()
            return int(value)
        if col.type == REAL:
            return float(value)
        return str(value)

    # ------------------------------------------------------------------
    # transaction helpers
    # ------------------------------------------------------------------
    def begin(self) -> Transaction:
        return self.txn_mgr.begin()

    def commit(self, txn: Transaction) -> None:
        self._release_pins(txn)
        self.txn_mgr.commit(txn)

    def rollback(self, txn: Transaction) -> None:
        self._undo(txn)
        self._release_pins(txn)
        self.txn_mgr.abort(txn)

    def _pin_hook(self, txn: Transaction):
        """Return an on_modify callback that pins the touched page for the
        lifetime of the transaction (the no-steal mechanism)."""

        def hook(rel: str, page_id: int) -> None:
            self.pool.fetch_page(rel, page_id)  # pin; deliberately not unpinned
            txn.pinned_pages.add((rel, page_id))

        return hook

    def _release_pins(self, txn: Transaction) -> None:
        for rel, page_id in txn.pinned_pages:
            page = self.pool.fetch_page(rel, page_id)
            self.pool.unpin(page)
        txn.pinned_pages.clear()

    def _undo(self, txn: Transaction) -> None:
        hook = self._pin_hook(txn)
        for entry in reversed(txn.undo_log):
            table = entry["table"]
            meta = self.catalog.get_table(table)
            values = decode_row(meta.columns, entry["row"])
            if entry["op"] == "insert":
                # remove the index entry only if it still points at the row
                # this insert created (a later duplicate-key failure may
                # have left the index pointing at an older row)
                for idx in meta.indexes:
                    tree = self._open_tree(meta, idx, txn)
                    col = meta.column(idx.column)
                    key = values[self._col_index(meta, col.name)]
                    if tree.get(key) == entry["record_id"]:
                        tree.delete(key)
                tombstone(self.pool, table, entry["record_id"], on_modify=hook)
            else:  # delete -> restore the old row and re-insert index entries
                restore_row(self.pool, table, entry["record_id"], entry["row"], on_modify=hook)
                for idx in meta.indexes:
                    tree = self._open_tree(meta, idx, txn)
                    col = meta.column(idx.column)
                    tree.insert(values[self._col_index(meta, col.name)], entry["record_id"])

    # ------------------------------------------------------------------
    # index plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _col_index(meta: TableMeta, name: str) -> int:
        for i, col in enumerate(meta.columns):
            if col.name.lower() == name.lower():
                return i
        raise ValueError(f"unknown column {name!r}")

    def _open_tree(self, meta: TableMeta, idx, txn: Optional[Transaction]) -> BTree:
        store = PoolNodeStore(
            self.pool,
            self._index_rel(meta.name, idx.name),
            on_modify=self._pin_hook(txn) if txn else None,
        )
        return BTree(self.order, store, idx.root_page)

    def _index_insert(self, txn: Optional[Transaction], meta: TableMeta, values: List, record_id: int) -> None:
        for idx in meta.indexes:
            key = values[self._col_index(meta, idx.column)]
            self._open_tree(meta, idx, txn).insert(key, record_id)

    def _index_delete(self, txn: Optional[Transaction], meta: TableMeta, values: List) -> None:
        for idx in meta.indexes:
            key = values[self._col_index(meta, idx.column)]
            self._open_tree(meta, idx, txn).delete(key)

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------
    def create_table(self, name: str, columns: Sequence[Tuple[str, str, bool]]) -> None:
        """Create a table; ``columns`` are ``(name, type, primary_key)``."""
        if self.catalog.has_table(name):
            raise ValueError(f"table {name!r} already exists")
        names = [c[0].lower() for c in columns]
        if len(set(names)) != len(names):
            raise ValueError("duplicate column name")
        pks = [c for c in columns if c[2]]
        if len(pks) != 1:
            raise ValueError("a table must have exactly one primary key")
        pk_type = pks[0][1]
        if pk_type not in (INT, TEXT):
            raise ValueError("primary key column must be INT or TEXT")
        meta = TableMeta(name, [Column(c[0], c[1], c[2]) for c in columns])
        self._register_disk(name)
        meta.root_page = create_table_pages(self.pool, name)
        self.catalog.create_table(meta)
        self.create_index(f"pk_{name}", name, pks[0][0])

    def create_index(self, name: str, table: str, column: str) -> None:
        meta = self.catalog.get_table(table)
        col = meta.column(column)
        if col is None:
            raise ValueError(f"table {table!r} has no column {column!r}")
        if col.type not in (INT, TEXT):
            raise ValueError(f"index column {column!r} must be INT or TEXT")
        idx = IndexInfo(name, table, col.name, "btree", 0, False)
        self.catalog.add_index(idx)
        self._rebuild_index(meta, idx)

    def drop_table(self, name: str) -> None:
        meta = self.catalog.get_table(name)
        for idx in list(meta.indexes):
            rel = self._index_rel(name, idx.name)
            self.pool.unregister_relation(rel)
            path = self._rel_path(rel)
            if os.path.exists(path):
                os.remove(path)
        drop_table_pages(self.pool, name, meta.root_page)
        self.pool.unregister_relation(name)
        if os.path.exists(self._rel_path(name)):
            os.remove(self._rel_path(name))
        self.catalog.drop_table(name)

    def tables(self) -> List[str]:
        return list(self.catalog.tables)

    def describe(self, name: str) -> List[Tuple[str, str, bool]]:
        meta = self.catalog.get_table(name)
        return [(c.name, c.type, c.primary_key) for c in meta.columns]

    # ------------------------------------------------------------------
    # DML
    # ------------------------------------------------------------------
    def insert(self, txn: Transaction, table: str, values: Sequence) -> None:
        meta = self.catalog.get_table(table)
        if len(values) != len(meta.columns):
            raise ValueError(
                f"expected {len(meta.columns)} values, got {len(values)}"
            )
        coerced = [self._coerce(col, v) for col, v in zip(meta.columns, values)]
        row = encode_row(meta.columns, coerced)
        record_id = insert_row(
            self.pool, table, meta.root_page, row, on_modify=self._pin_hook(txn)
        )
        pk = meta.primary_key
        pk_value = coerced[self._col_index(meta, pk.name)]
        self.wal.insert(txn.txn_id, table, self._key_bytes(pk, pk_value), record_id, row)
        # register the undo entry before the fallible index insert so a
        # duplicate-key failure still gets cleaned up on rollback
        txn.undo_log.append(
            {"op": "insert", "table": table, "record_id": record_id, "row": row}
        )
        self._index_insert(txn, meta, coerced, record_id)  # raises DuplicateKeyError

    def delete(self, txn: Transaction, table: str, pred) -> int:
        meta = self.catalog.get_table(table)
        hook = self._pin_hook(txn)
        count = 0
        for record_id, row in self._candidates(meta, pred):
            values = decode_row(meta.columns, row)
            pk = meta.primary_key
            pk_value = values[self._col_index(meta, pk.name)]
            self.wal.delete(
                txn.txn_id, table, self._key_bytes(pk, pk_value), record_id, row
            )
            self._index_delete(txn, meta, values)
            tombstone(self.pool, table, record_id, on_modify=hook)
            txn.undo_log.append(
                {"op": "delete", "table": table, "record_id": record_id, "row": row}
            )
            count += 1
        return count

    def update(self, txn: Transaction, table: str, set_col: str, set_val, pred) -> int:
        meta = self.catalog.get_table(table)
        set_col_obj = meta.column(set_col)
        if set_col_obj is None:
            raise ValueError(f"unknown column {set_col!r}")
        new_val = self._coerce(set_col_obj, set_val)
        set_idx = self._col_index(meta, set_col_obj.name)
        hook = self._pin_hook(txn)
        count = 0
        for record_id, row in self._candidates(meta, pred):
            values = decode_row(meta.columns, row)
            pk = meta.primary_key
            pk_value = values[self._col_index(meta, pk.name)]
            # phase 1: retire the old row
            self.wal.delete(
                txn.txn_id, table, self._key_bytes(pk, pk_value), record_id, row
            )
            self._index_delete(txn, meta, values)
            tombstone(self.pool, table, record_id, on_modify=hook)
            txn.undo_log.append(
                {"op": "delete", "table": table, "record_id": record_id, "row": row}
            )
            # phase 2: append the new row
            new_values = list(values)
            new_values[set_idx] = new_val
            new_row = encode_row(meta.columns, new_values)
            new_rid = insert_row(
                self.pool, table, meta.root_page, new_row, on_modify=hook
            )
            pk_value = new_values[self._col_index(meta, pk.name)]
            self.wal.insert(
                txn.txn_id, table, self._key_bytes(pk, pk_value), new_rid, new_row
            )
            txn.undo_log.append(
                {"op": "insert", "table": table, "record_id": new_rid, "row": new_row}
            )
            self._index_insert(txn, meta, new_values, new_rid)
            count += 1
        return count

    def select(
        self,
        table: str,
        pred=None,
        order_col: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
    ) -> List[dict]:
        meta = self.catalog.get_table(table)
        names = [c.name for c in meta.columns]
        rows = [
            dict(zip(names, decode_row(meta.columns, row)))
            for _, row in self._candidates(meta, pred)
        ]
        if order_col is not None:
            idx = self._col_index(meta, order_col)
            rows.sort(key=lambda r: list(r.values())[idx], reverse=desc)
        if limit is not None and limit >= 0:
            rows = rows[:limit]
        return rows

    # ------------------------------------------------------------------
    # predicate evaluation / candidate generation
    # ------------------------------------------------------------------
    def _candidates(self, meta: TableMeta, pred) -> List[Tuple[int, bytes]]:
        """Return ``(record_id, row_bytes)`` for rows matching ``pred``
        (None = all rows).  Uses an index on the predicate column when one
        exists; otherwise falls back to a full heap scan."""
        if pred is None:
            return list(scan(self.pool, meta.name, meta.root_page))
        column, op, value = pred
        col = meta.column(column)
        if col is None:
            raise ValueError(f"unknown column {column!r}")
        idx = next((i for i in meta.indexes if i.column == col.name), None)
        cmp_value = self._coerce(col, value)
        if idx is not None and op != "!=":
            tree = self._open_tree(meta, idx, None)
            if op == "=":
                key = cmp_value
                rid = tree.get(key)
                pairs = []
                if rid is not None:
                    row = read_row(self.pool, meta.name, rid)
                    if row is not None:
                        pairs = [(rid, row)]
                return pairs
            lo = None
            hi = None
            if op in ("<", "<="):
                hi = cmp_value
            elif op in (">", ">="):
                lo = cmp_value
            # range_scan yields (key, record_id)
            return [(record_id, read_row(self.pool, meta.name, record_id))
                    for _, record_id in tree.range_scan(lo, hi)]
        # full scan + filter
        ci = self._col_index(meta, col.name)
        fn = _OPS[op]
        out = []
        for record_id, row in scan(self.pool, meta.name, meta.root_page):
            values = decode_row(meta.columns, row)
            if fn(values[ci], cmp_value):
                out.append((record_id, row))
        return out

    # ------------------------------------------------------------------
    # autocommit wrappers (each statement runs in its own transaction)
    # ------------------------------------------------------------------
    def execute_insert(self, table: str, values: Sequence) -> None:
        txn = self.begin()
        try:
            self.insert(txn, table, values)
        except BaseException:
            self.rollback(txn)
            raise
        self.commit(txn)

    def execute_delete(self, table: str, pred) -> int:
        txn = self.begin()
        try:
            count = self.delete(txn, table, pred)
        except BaseException:
            self.rollback(txn)
            raise
        self.commit(txn)
        return count

    def execute_update(self, table: str, set_col: str, set_val, pred) -> int:
        txn = self.begin()
        try:
            count = self.update(txn, table, set_col, set_val, pred)
        except BaseException:
            self.rollback(txn)
            raise
        self.commit(txn)
        return count

    # ------------------------------------------------------------------
    # durability: checkpoint / close
    # ------------------------------------------------------------------
    def checkpoint(self) -> int:
        """Flush every dirty page, fsync all relation files, then truncate
        the WAL.  Returns the number of pages written."""
        written = self.pool.flush_all()
        for disk in self.pool.disks.values():
            disk.sync()
        self.wal.truncate()
        return written

    def close(self) -> None:
        self.pool.flush_all()
        for disk in self.pool.disks.values():
            disk.sync()
        self.wal.sync()
        for disk in self.pool.disks.values():
            disk.close()
        self.wal.close()

    def discard(self) -> None:
        """Simulate a process crash: drop every dirty page and close the
        file handles **without** flushing, so the on-disk state is exactly
        what was previously fsynced (WAL commits + flushed pages)."""
        for disk in self.pool.disks.values():
            disk.close()
        self.pool.disks.clear()
        self.pool.pages.clear()
        self.pool._lru.clear()
        self.wal.close()

    # ------------------------------------------------------------------
    # crash recovery (called from __init__)
    # ------------------------------------------------------------------
    def _recover(self) -> None:
        for name in self.catalog.tables:
            self._register_disk(name)
        for name, meta in self.catalog.tables.items():
            for idx in meta.indexes:
                if idx.built and idx.root_page:
                    self._register_disk(self._index_rel(name, idx.name))

        result = self.wal.recover()
        for rec in result.records:
            if rec.txn_id not in result.committed:
                continue
            if rec.op == OP_INSERT:
                insert_row_at(self.pool, rec.table, rec.record_id, rec.row)
            elif rec.op == OP_DELETE:
                tombstone(self.pool, rec.table, rec.record_id)

        for name in self.catalog.tables:
            self._repair_chain(name)
        for name, meta in self.catalog.tables.items():
            for idx in list(meta.indexes):
                self._rebuild_index(meta, idx)

        # Everything is on disk now: safe to reset the log.
        self.checkpoint()

    def _repair_chain(self, table: str) -> None:
        """Re-link every heap page by page-id order (heap pages are
        allocated in chain order, so id order == chain order).  Non-heap
        pages (uncommitted orphans, unallocated tails) are skipped."""
        meta = self.catalog.get_table(table)
        root = meta.root_page
        disk = self.pool.disks[table]
        prev = root
        for pid in range(root + 1, disk.num_pages):
            page = self.pool.fetch_page(table, pid)
            try:
                is_heap = page.buffer[0] == PAGE_TYPE_HEAP
            finally:
                self.pool.unpin(page)
            if not is_heap:
                continue
            set_next_page(self.pool, table, prev, pid)
            prev = pid
        set_next_page(self.pool, table, prev, 0)

    def _rebuild_index(self, meta: TableMeta, idx) -> None:
        """Recreate an index from scratch by scanning the heap (used on
        first build and after crash recovery, when on-disk index pages may
        be stale)."""
        rel = self._index_rel(meta.name, idx.name)
        self.pool.unregister_relation(rel)
        if os.path.exists(self._rel_path(rel)):
            os.remove(self._rel_path(rel))
        self._register_disk(rel)
        tree = BTree(self.order, PoolNodeStore(self.pool, rel), None)
        col = meta.column(idx.column)
        ci = self._col_index(meta, col.name)
        entries = []
        for record_id, row in scan(self.pool, meta.name, meta.root_page):
            values = decode_row(meta.columns, row)
            entries.append((values[ci], record_id))
        entries.sort(key=lambda kv: kv[0])
        for key, record_id in entries:
            tree.insert(key, record_id)
        idx.root_page = tree.root
        idx.built = True
        self.catalog.save_table(meta)

    def __repr__(self) -> str:
        return (
            f"StorageEngine(data_dir={self.data_dir!r}, tables={list(self.catalog.tables)}, "
            f"pool={self.pool})"
        )
