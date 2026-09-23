"""WAL unit tests: round-trip, committed-txn filtering, CRC corruption and
torn-tail detection."""
import os
import struct
import zlib

from engine.wal import OP_ABORT, OP_BEGIN, OP_COMMIT, OP_DELETE, OP_INSERT, WAL, WAL_MAGIC


def test_roundtrip(tmp_path):
    path = str(tmp_path / "wal.log")
    wal = WAL(path)
    wal.begin(1)
    wal.insert(1, "t", b"\x01\x00", 5, b"rowdata")
    wal.commit(1)
    wal.begin(2)
    wal.delete(2, "t", b"\x02\x00", 9, b"oldrow")
    wal.abort(2)
    wal.close()

    wal = WAL(path)
    result = wal.recover()
    assert result.committed == {1}
    assert not result.truncated
    ops = [r.op for r in result.records]
    assert ops == [OP_BEGIN, OP_INSERT, OP_COMMIT, OP_BEGIN, OP_DELETE, OP_ABORT]
    insert_rec = result.records[1]
    assert insert_rec.table == "t"
    assert insert_rec.key == b"\x01\x00"
    assert insert_rec.record_id == 5
    assert insert_rec.row == b"rowdata"
    delete_rec = result.records[4]
    assert delete_rec.row == b"oldrow"
    wal.close()


def test_crc_corruption_stops_recovery(tmp_path):
    path = str(tmp_path / "wal.log")
    wal = WAL(path)
    wal.begin(1)
    wal.insert(1, "t", b"\x01", 1, b"a" * 32)
    wal.commit(1)
    wal.begin(2)
    wal.insert(2, "t", b"\x02", 2, b"b" * 32)
    wal.commit(2)
    wal.close()

    # flip the op byte inside the second transaction's BEGIN record so the
    # checksum fails and recovery stops before any record of txn 2 appears
    with open(path, "r+b") as fh:
        fh.seek(len(WAL_MAGIC))
        data = fh.read()
    # block sizes: BEGIN = 6 + 9, INSERT = 6 + (9 + 2 + 1 + 2 + 1 + 8 + 4 + 32)
    begin2_op = (6 + 9) + (6 + 59) + (6 + 9) + 6 + 8  # past txn1's 3 blocks, into BEGIN2's op byte
    corrupted = bytearray(data)
    corrupted[begin2_op] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(WAL_MAGIC + bytes(corrupted))

    wal = WAL(path)
    result = wal.recover()
    assert result.committed == {1}
    assert result.truncated
    assert all(r.txn_id != 2 for r in result.records)
    wal.close()


def test_torn_tail(tmp_path):
    path = str(tmp_path / "wal.log")
    wal = WAL(path)
    wal.begin(1)
    wal.insert(1, "t", b"\x01", 1, b"x" * 16)
    wal.commit(1)
    wal.close()
    # append a half-written record: header claims length but payload is short
    payload = struct.pack("<QB", 99, OP_INSERT) + struct.pack("<H", 1) + b"t" + struct.pack("<H", 1) + b"\x01" + struct.pack("<Q", 3) + struct.pack("<I", 8) + b"short!!"
    header = struct.pack("<IH", zlib.crc32(payload), len(payload))
    with open(path, "ab") as fh:
        fh.write(header + payload[: len(payload) - 3])  # tear the tail

    wal = WAL(path)
    result = wal.recover()
    assert result.committed == {1}
    assert result.truncated
    wal.close()


def test_truncate_resets(tmp_path):
    path = str(tmp_path / "wal.log")
    wal = WAL(path)
    wal.begin(1)
    wal.insert(1, "t", b"\x01", 1, b"r")
    wal.commit(1)
    wal.truncate()
    result = wal.recover()
    assert result.committed == set()
    assert result.records == []
    wal.close()
