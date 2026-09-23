"""End-to-end storage engine tests: DDL/DML through the public API,
index-assisted predicate evaluation, checkpoint/reopen idempotence and a
crash-recovery drill (committed survives, uncommitted disappears)."""
import pytest

from engine import DuplicateKeyError, INT, StorageEngine
from engine.table import scan


def make_engine(tmp_path, **kw):
    return StorageEngine(str(tmp_path / "db"), **kw)


def make_users(e):
    e.create_table(
        "users",
        [("id", "INT", True), ("name", "TEXT", False), ("score", "REAL", False)],
    )
    e.create_index("idx_name", "users", "name")


def seed_users(e, n=20):
    for i in range(n):
        e.execute_insert("users", [i, f"u{i}", float(i % 7)])


def test_ddl_roundtrip(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    assert e.tables() == ["users"]
    assert e.describe("users") == [
        ("id", "INT", True),
        ("name", "TEXT", False),
        ("score", "REAL", False),
    ]
    with pytest.raises(ValueError):
        e.create_table("users", [("id", "INT", True)])
    e.close()
    # reopen: catalog survives
    e2 = make_engine(tmp_path)
    assert e2.tables() == ["users"]
    e2.close()


def test_insert_select_predicate_order_limit(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    seed_users(e)
    assert len(e.select("users")) == 20
    # equality via pk index
    assert e.select("users", ("id", "=", 5)) == [
        {"id": 5, "name": "u5", "score": 5.0}
    ]
    # range via pk index
    ids = [r["id"] for r in e.select("users", ("id", ">=", 15))]
    assert ids == [15, 16, 17, 18, 19]
    # inequality falls back to a scan
    assert len(e.select("users", ("name", "!=", "u0"))) == 19
    # order + limit
    rows = e.select("users", order_col="score", desc=True, limit=3)
    assert rows[0]["score"] == 6.0
    assert len(rows) == 3
    e.close()


def test_update_and_delete(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    seed_users(e, 5)
    n = e.execute_update("users", "score", 99.0, ("id", "<=", 2))
    assert n == 3
    assert all(
        r["score"] == 99.0 for r in e.select("users", ("id", "<=", 2))
    )
    n = e.execute_delete("users", ("name", "=", "u4"))
    assert n == 1
    assert e.select("users", ("id", "=", 4)) == []
    assert len(e.select("users")) == 4
    e.close()


def test_secondary_index_range(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    seed_users(e)
    # range predicate on the indexed TEXT column; rows come back in the
    # index's key order (lexicographic byte order), not insertion order
    rows = e.select("users", ("name", ">=", "u15"))
    lex_names = sorted(n for n in (f"u{i}" for i in range(20)) if n >= "u15")
    expected = [int(n[1:]) for n in lex_names]
    assert [r["id"] for r in rows] == expected
    e.close()


def test_checkpoint_reopen_idempotent(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    seed_users(e, 10)
    e.checkpoint()
    n_written = e.checkpoint()
    assert n_written == 0  # nothing dirty left
    e.close()
    e2 = make_engine(tmp_path)
    assert len(e2.select("users")) == 10
    e2.execute_insert("users", [10, "u10", 0.0])
    e2.close()
    e3 = make_engine(tmp_path)
    assert len(e3.select("users")) == 11
    e3.close()


def test_crash_recovery_committed_survives_uncommitted_lost(tmp_path):
    # phase 1: commit A, leave B uncommitted, then crash without flushing
    e = make_engine(tmp_path)
    make_users(e)
    e.execute_insert("users", [1, "a", 1.0])
    e.execute_insert("users", [2, "b", 2.0])
    txn = e.begin()
    e.insert(txn, "users", [3, "c", 3.0])  # never committed
    e.discard()  # simulate a crash: no flush, WAL keeps only A's commit

    # phase 2: reopen -> recovery replays committed redo and rebuilds indexes
    e2 = make_engine(tmp_path)
    rows = e2.select("users", order_col="id")
    assert [r["id"] for r in rows] == [1, 2]
    meta = e2.catalog.get_table("users")
    for idx in meta.indexes:
        tree = e2._open_tree(meta, idx, None)
        assert len(tree) == 2
        # the rolled-back key must be absent from the index for that column
        key = 3 if meta.column(idx.column).type == INT else "c"
        assert tree.get(key) is None
    assert len(list(scan(e2.pool, "users", meta.root_page))) == 2
    e2.close()


def test_crash_recovery_uncommitted_delete_lost(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    e.execute_insert("users", [1, "a", 1.0])
    txn = e.begin()
    e.delete(txn, "users", ("id", "=", 1))
    e.discard()
    e2 = make_engine(tmp_path)
    assert e2.select("users", ("id", "=", 1)) == [{"id": 1, "name": "a", "score": 1.0}]
    e2.close()


def test_drop_table(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    seed_users(e, 3)
    e.drop_table("users")
    assert e.tables() == []
    with pytest.raises(ValueError):
        e.select("users")
    e.close()
