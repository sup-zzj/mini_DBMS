"""Transaction state and the transaction manager.

Transactions are tracked in memory only.  Durability comes from the WAL:
the storage engine appends redo records (INSERT / DELETE) as operations
run, and the manager writes BEGIN / COMMIT / ABORT markers.  The engine's
*no-steal* guarantee — every page a transaction touches is pinned, so
uncommitted changes can never be flushed to disk — makes rollback a pure
in-memory undo of the transaction's own operations.

The manager is deliberately dumb: it assigns ids, tracks active
transactions and writes commit/abort markers.  The actual undo execution
lives in :mod:`engine.storage_engine`, which owns the heap and the indexes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .wal import WAL

ACTIVE, COMMITTED, ABORTED = "active", "committed", "aborted"


@dataclass
class Transaction:
    """A single transaction.

    ``undo_log`` entries are plain dicts with an ``op`` of ``"insert"`` or
    ``"delete"`` plus the table, record id and the affected row bytes — the
    storage engine decodes them (column values, index keys) at rollback
    time.  ``pinned_pages`` holds ``(relation, page_id)`` pairs kept in the
    buffer pool for the duration of the transaction (no-steal).
    """

    txn_id: int
    status: str = ACTIVE
    undo_log: List[dict] = field(default_factory=list)
    pinned_pages: Set[Tuple[str, int]] = field(default_factory=set)

    def __repr__(self) -> str:
        return f"Transaction(id={self.txn_id}, status={self.status!r})"


class TxnManager:
    """Assigns transaction ids and writes the WAL lifecycle markers."""

    def __init__(self, wal: WAL) -> None:
        self.wal = wal
        self._next = 0
        self.active: Dict[int, Transaction] = {}

    def begin(self) -> Transaction:
        txn = Transaction(self._next)
        self._next += 1
        self.wal.begin(txn.txn_id)
        self.active[txn.txn_id] = txn
        return txn

    def commit(self, txn: Transaction) -> None:
        """Persist the commit: append the COMMIT record and fsync the WAL
        (no-force — dirty pages may still be in the buffer pool)."""
        self.wal.commit(txn.txn_id)
        self.wal.sync()
        txn.status = COMMITTED
        self.active.pop(txn.txn_id, None)

    def abort(self, txn: Transaction) -> None:
        self.wal.abort(txn.txn_id)
        self.wal.sync()
        txn.status = ABORTED
        self.active.pop(txn.txn_id, None)

    def __len__(self) -> int:
        return len(self.active)
