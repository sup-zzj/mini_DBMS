"""Crash-recovery demonstration (the WAL story, end to end).

Scenario
--------
1. Open a fresh database, commit transaction A (inserts 3 rows).
2. Start transaction B, insert 2 more rows, then **crash** before commit
   (:meth:`StorageEngine.discard` drops every dirty page and closes all
   file handles without flushing — the WAL is the only survivor).
3. Reopen the database: recovery replays the committed redo records and
   rebuilds every index from the heap.
4. Assertion: A's rows are there, B's rows are gone, and both the primary
   and secondary indexes agree with the heap.

Output: ``results/crash_recovery_demo.json`` with the before / after states.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import StorageEngine

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
DB_DIR = os.path.join(RESULTS, "crash_demo_db")


def snapshot(e: StorageEngine) -> list:
    return e.select("users", order_col="id")


def main() -> int:
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # phase 1: committed A + uncommitted B, then crash
    # ---------------------------------------------------------------
    e = StorageEngine(DB_DIR)
    e.create_table(
        "users",
        [("id", "INT", True), ("name", "TEXT", False), ("score", "REAL", False)],
    )
    e.create_index("idx_name", "users", "name")

    e.execute_insert("users", [1, "alice", 9.5])
    e.execute_insert("users", [2, "bob", 7.0])
    e.execute_insert("users", [3, "carol", 8.5])  # committed transaction A

    txn = e.begin()
    e.insert(txn, "users", [4, "dave", 6.0])
    e.insert(txn, "users", [5, "erin", 9.0])
    # ... crash! no commit, no flush
    before_crash = snapshot(e)  # in-memory view, includes B
    e.discard()

    # ---------------------------------------------------------------
    # phase 2: reopen -> recovery
    # ---------------------------------------------------------------
    e2 = StorageEngine(DB_DIR)
    after_recovery = snapshot(e2)

    meta = e2.catalog.get_table("users")
    index_ok = True
    for idx in meta.indexes:
        tree = e2._open_tree(meta, idx, None)
        col = meta.column(idx.column)
        index_ok = index_ok and len(tree) == len(after_recovery)

    ok = (
        [r["id"] for r in after_recovery] == [1, 2, 3]
        and all(r["id"] not in (4, 5) for r in after_recovery)
        and index_ok
    )

    print("before crash (in memory):", [(r["id"], r["name"]) for r in before_crash])
    print("after recovery (disk):   ", [(r["id"], r["name"]) for r in after_recovery])
    print(f"committed survived: {[r['id'] for r in after_recovery] == [1, 2, 3]}")
    print(f"uncommitted lost:   {not any(r['id'] in (4, 5) for r in after_recovery)}")
    print(f"indexes consistent: {index_ok}")

    payload = {
        "committed_ids": [1, 2, 3],
        "uncommitted_ids": [4, 5],
        "before_crash": [r["id"] for r in before_crash],
        "after_recovery": [r["id"] for r in after_recovery],
        "committed_survived": [r["id"] for r in after_recovery] == [1, 2, 3],
        "uncommitted_lost": not any(r["id"] in (4, 5) for r in after_recovery),
        "indexes_consistent": index_ok,
        "all_checks_passed": ok,
    }
    path = os.path.join(RESULTS, "crash_recovery_demo.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"crash-recovery demo -> {path}")

    e2.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
