"""Write-ahead log (WAL).

A single append-only log per database is used to guarantee durability of
committed transactions.  The engine follows a *no-steal / no-force* policy:

* **no-steal** — pages containing uncommitted changes are pinned and never
  written to disk before commit, so a crash never leaves partial uncommitted
  data on disk;
* **no-force** — committed pages may still be dirty in the buffer pool when
  the commit returns, so the WAL is the only source that guarantees the
  commit survives a crash.

Recovery therefore only has to *redo* the writes of transactions whose
COMMIT record reached the log; uncommitted transactions are simply absent
from disk and are dropped.

Record format (little-endian, appended to the log file)::

    [u32 crc32][u16 length][payload]
    payload = [u64 txn_id][u8 op][body]

    op 1 BEGIN        body: -
    op 2 INSERT       body: [u16 table_len][table utf8][u16 key_len][key bytes][u64 record_id][u32 row_len][row bytes]
    op 3 DELETE       body: [u16 table_len][table utf8][u16 key_len][key bytes][u64 record_id][u32 row_len][old row bytes]
    op 4 COMMIT       body: -
    op 5 ABORT        body: -

The CRC covers ``length + payload``; a torn tail (partial write) is
detected as a length/CRC mismatch and recovery stops there.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

WAL_MAGIC = b"MINIWAL1"

# op codes
OP_BEGIN = 1
OP_INSERT = 2
OP_DELETE = 3
OP_COMMIT = 4
OP_ABORT = 5

OP_NAMES = {
    OP_BEGIN: "BEGIN",
    OP_INSERT: "INSERT",
    OP_DELETE: "DELETE",
    OP_COMMIT: "COMMIT",
    OP_ABORT: "ABORT",
}


@dataclass(frozen=True)
class LogRecord:
    """One decoded log record.

    ``table`` / ``key`` / ``record_id`` / ``row`` are only populated for
    INSERT and DELETE records.  ``key`` is the raw serialised key bytes
    (the storage engine knows how to decode them per column type).
    """

    op: int
    txn_id: int
    table: str = ""
    key: bytes = b""
    record_id: int = 0
    row: bytes = b""

    @property
    def op_name(self) -> str:
        return OP_NAMES[self.op]


@dataclass
class RecoveryResult:
    """Result of :meth:`WAL.recover`."""

    records: List[LogRecord]  # all valid records, in log order
    committed: Set[int]  # txn ids whose COMMIT record was found
    truncated: bool  # True if the log ended with a torn/corrupt record


def _pack_body(record: LogRecord) -> bytes:
    body = struct.pack("<QB", record.txn_id, record.op)
    if record.op in (OP_INSERT, OP_DELETE):
        table = record.table.encode("utf-8")
        body += struct.pack("<H", len(table)) + table
        body += struct.pack("<H", len(record.key)) + record.key
        body += struct.pack("<Q", record.record_id)
        body += struct.pack("<I", len(record.row)) + record.row
    return body


class WAL:
    """Append-only, checksummed redo log."""

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a+b")
        if self._fh.tell() == 0:
            self._fh.write(WAL_MAGIC)
            self._fh.flush()

    # ------------------------------------------------------------------
    # append
    # ------------------------------------------------------------------
    def _append(self, record: LogRecord) -> None:
        payload = _pack_body(record)
        length = len(payload)
        crc = zlib.crc32(payload)
        self._fh.write(struct.pack("<IH", crc, length))
        self._fh.write(payload)

    def begin(self, txn_id: int) -> None:
        self._append(LogRecord(OP_BEGIN, txn_id))

    def insert(self, txn_id: int, table: str, key: bytes, record_id: int, row: bytes) -> None:
        self._append(LogRecord(OP_INSERT, txn_id, table, key, record_id, row))

    def delete(self, txn_id: int, table: str, key: bytes, record_id: int, row: bytes) -> None:
        self._append(LogRecord(OP_DELETE, txn_id, table, key, record_id, row))

    def commit(self, txn_id: int) -> None:
        self._append(LogRecord(OP_COMMIT, txn_id))

    def abort(self, txn_id: int) -> None:
        self._append(LogRecord(OP_ABORT, txn_id))

    def sync(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    # ------------------------------------------------------------------
    # recovery / maintenance
    # ------------------------------------------------------------------
    def recover(self) -> RecoveryResult:
        """Decode the whole log.  Stops (and sets ``truncated``) at the
        first record whose checksum fails or whose length overruns EOF."""
        self._fh.seek(len(WAL_MAGIC))
        records: List[LogRecord] = []
        committed: Set[int] = set()
        truncated = False
        header = self._fh.read(6)
        while header:
            if len(header) < 6:
                truncated = True
                break
            crc, length = struct.unpack("<IH", header)
            if length > 1 << 20:  # sanity bound
                truncated = True
                break
            payload = self._fh.read(length)
            if len(payload) < length:
                truncated = True
                break
            if zlib.crc32(payload) != crc:
                truncated = True
                break
            record = self._decode(payload)
            if record is None:
                truncated = True
                break
            if record.op == OP_COMMIT:
                committed.add(record.txn_id)
            records.append(record)
            header = self._fh.read(6)
        return RecoveryResult(records, committed, truncated)

    def truncate(self) -> None:
        """Reset the log (used after a checkpoint that flushed all dirty
        pages to disk)."""
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(WAL_MAGIC)
        self._fh.flush()

    @staticmethod
    def _decode(payload: bytes) -> Optional[LogRecord]:
        try:
            txn_id, op = struct.unpack("<QB", payload[:9])
            if op in (OP_INSERT, OP_DELETE):
                pos = 9
                (tlen,) = struct.unpack("<H", payload[pos : pos + 2])
                pos += 2
                table = payload[pos : pos + tlen].decode("utf-8")
                pos += tlen
                (klen,) = struct.unpack("<H", payload[pos : pos + 2])
                pos += 2
                key = payload[pos : pos + klen]
                pos += klen
                (record_id,) = struct.unpack("<Q", payload[pos : pos + 8])
                pos += 8
                (rlen,) = struct.unpack("<I", payload[pos : pos + 4])
                pos += 4
                row = payload[pos : pos + rlen]
                return LogRecord(op, txn_id, table, key, record_id, row)
            if op in (OP_BEGIN, OP_COMMIT, OP_ABORT):
                return LogRecord(op, txn_id)
            return None
        except (struct.error, UnicodeDecodeError, IndexError):
            return None

    def close(self) -> None:
        self._fh.close()

    def __repr__(self) -> str:
        return f"WAL(path={self.path!r})"
