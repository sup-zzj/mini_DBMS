"""Transaction tests through the storage engine: commit persistence,
rollback consistency (data AND index), and duplicate-key rollback cleanup."""
import pytest

from engine import DuplicateKeyError, StorageEngine
from engine.table import scan


def make_engine(tmp_path):
    return StorageEngine(str(tmp_path / "db"))


def make_users(e):
    e.create_table("users", [("id", "INT", True), ("name", "TEXT", False), ("score", "REAL", False)])
    e.create_index("idx_name", "users", "name")
    return e.catalog.get_table("users")


def test_commit_persists(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    txn = e.begin()
    e.insert(txn, "users", [1, "alice", 9.5])
    e.insert(txn, "users", [2, "bob", 7.0])
    e.commit(txn)
    assert len(e.select("users")) == 2
    # index must contain both keys too
    meta = e.catalog.get_table("users")
    for idx in meta.indexes:
        tree = e._open_tree(meta, idx, None)
        assert len(tree) == 2
    e.close()


def test_rollback_removes_everything(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    e.execute_insert("users", [1, "alice", 9.5])
    txn = e.begin()
    e.insert(txn, "users", [2, "bob", 7.0])
    e.delete(txn, "users", ("name", "=", "alice"))
    e.rollback(txn)
    # nothing from the txn may be visible, in data or in any index
    assert len(e.select("users")) == 1
    meta = e.catalog.get_table("users")
    trees = {idx.column: e._open_tree(meta, idx, None) for idx in meta.indexes}
    assert trees["id"].get(2) is None  # rolled-back pk absent
    assert trees["name"].get("bob") is None  # rolled-back secondary absent
    assert trees["name"].get("alice") is not None  # original row restored
    e.close()


def test_update_rollback_restores_original(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    e.execute_insert("users", [1, "alice", 9.5])
    txn = e.begin()
    e.update(txn, "users", "score", 0.0, ("id", "=", 1))
    assert e.select("users")[0]["score"] == 0.0
    e.rollback(txn)
    assert e.select("users")[0]["score"] == 9.5
    e.close()


def test_duplicate_key_rollback_leaves_no_orphan(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    e.execute_insert("users", [1, "alice", 9.5])
    # dup on the primary key
    with pytest.raises(DuplicateKeyError):
        e.execute_insert("users", [1, "bob", 0.0])
    rows = list(scan(e.pool, "users", e.catalog.get_table("users").root_page))
    assert len(rows) == 1
    assert len(e.select("users")) == 1
    # dup on the secondary (unique) index — the heap row and the pk entry
    # inserted before the failure must also be rolled back
    with pytest.raises(DuplicateKeyError):
        e.execute_insert("users", [2, "alice", 7.0])
    rows = list(scan(e.pool, "users", e.catalog.get_table("users").root_page))
    assert len(rows) == 1
    assert e.select("users", ("id", "=", 2)) == []
    e.close()


def test_abort_wal_record_present(tmp_path):
    e = make_engine(tmp_path)
    make_users(e)
    txn = e.begin()
    e.insert(txn, "users", [7, "x", 1.0])
    e.rollback(txn)
    result = e.wal.recover()
    assert result.records[-1].op == 5  # OP_ABORT
    e.close()
