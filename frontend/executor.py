"""Execute parsed statements against the storage engine.

Results are returned as plain tuples for the REPL to render::

    ("msg", "已创建表 users")          # single-line feedback
    ("table", [headers], [rows])       # tabular result

An explicit transaction (BEGIN ... COMMIT/ROLLBACK) is tracked here: while
one is open, INSERT / UPDATE / DELETE run inside it instead of using the
engine's autocommit wrappers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from engine import StorageEngine
from engine.txn import Transaction

_MSG = "msg"
_TABLE = "table"


class Executor:
    def __init__(self, engine: StorageEngine) -> None:
        self.engine = engine
        self._txn: Optional[Transaction] = None

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def execute(self, ast: dict) -> Tuple:
        op = ast["op"]
        if op == "noop":
            return (_MSG, "")
        handler = getattr(self, "do_" + op)
        return handler(ast)

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------
    def do_create_table(self, ast: dict) -> Tuple:
        cols = [(c["name"], c["type"], c["primary_key"]) for c in ast["columns"]]
        self.engine.create_table(ast["name"], cols)
        return (_MSG, f"已创建表 {ast['name']}")

    def do_create_index(self, ast: dict) -> Tuple:
        self.engine.create_index(ast["name"], ast["table"], ast["column"])
        return (_MSG, f"已创建索引 {ast['name']} ON {ast['table']}({ast['column']})")

    def do_drop_table(self, ast: dict) -> Tuple:
        self.engine.drop_table(ast["name"])
        return (_MSG, f"已删除表 {ast['name']}")

    def do_show_tables(self, ast: dict) -> Tuple:
        names = self.engine.tables()
        return (_TABLE, ["表名"], [[n] for n in names])

    def do_describe(self, ast: dict) -> Tuple:
        cols = self.engine.describe(ast["table"])
        rows = [[n, t, "是" if pk else ""] for n, t, pk in cols]
        return (_TABLE, ["列名", "类型", "主键"], rows)

    # ------------------------------------------------------------------
    # DML
    # ------------------------------------------------------------------
    def do_insert(self, ast: dict) -> Tuple:
        table = ast["table"]
        values = ast["values"]
        if ast["columns"] is not None:
            values = self._map_columns(table, ast["columns"], values)
        if self._txn is not None:
            self.engine.insert(self._txn, table, values)
        else:
            self.engine.execute_insert(table, values)
        return (_MSG, "已插入 1 行")

    def do_delete(self, ast: dict) -> Tuple:
        pred = ast["where"]
        if self._txn is not None:
            n = self.engine.delete(self._txn, ast["table"], pred)
        else:
            n = self.engine.execute_delete(ast["table"], pred)
        return (_MSG, f"已删除 {n} 行")

    def do_update(self, ast: dict) -> Tuple:
        column, value = ast["set"]
        pred = ast["where"]
        if pred is None:
            raise ValueError("UPDATE 需要 WHERE 条件（防止全表误改）")
        if self._txn is not None:
            n = self.engine.update(self._txn, ast["table"], column, value, pred)
        else:
            n = self.engine.execute_update(ast["table"], column, value, pred)
        return (_MSG, f"已更新 {n} 行")

    def do_select(self, ast: dict) -> Tuple:
        rows = self.engine.select(
            ast["table"],
            pred=ast["where"],
            order_col=ast["order"][0] if ast["order"] else None,
            desc=ast["order"][1] == "desc" if ast["order"] else False,
            limit=ast["limit"],
        )
        names = ast["columns"]
        if names == ["*"]:
            headers = list(rows[0].keys()) if rows else self._table_headers(ast["table"])
            data = [list(r.values()) for r in rows]
        else:
            headers = names
            data = [[r[n] for n in names] for r in rows]
        return (_TABLE, headers, data)

    # ------------------------------------------------------------------
    # transactions
    # ------------------------------------------------------------------
    def do_begin(self, ast: dict) -> Tuple:
        if self._txn is not None:
            raise ValueError("已在事务中，请先 COMMIT 或 ROLLBACK")
        self._txn = self.engine.begin()
        return (_MSG, "已开始事务")

    def do_commit(self, ast: dict) -> Tuple:
        if self._txn is None:
            raise ValueError("没有活动事务")
        self.engine.commit(self._txn)
        self._txn = None
        return (_MSG, "事务已提交")

    def do_rollback(self, ast: dict) -> Tuple:
        if self._txn is None:
            raise ValueError("没有活动事务")
        self.engine.rollback(self._txn)
        self._txn = None
        return (_MSG, "事务已回滚")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _table_headers(self, table: str) -> List[str]:
        return [c[0] for c in self.engine.describe(table)]

    def _map_columns(self, table: str, columns: List[str], values: List) -> List:
        """Map an explicit column list onto catalog column order."""
        meta = self.engine.catalog.get_table(table)
        if len(columns) != len(meta.columns):
            raise ValueError(
                f"列数不匹配：表 {table} 有 {len(meta.columns)} 列，指定了 {len(columns)} 列"
            )
        ordered = [None] * len(meta.columns)
        used = set()
        for col, val in zip(columns, values):
            idx = None
            for i, c in enumerate(meta.columns):
                if c.name.lower() == col.lower():
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"未知列 {col!r}")
            if idx in used:
                raise ValueError(f"重复指定列 {col!r}")
            used.add(idx)
            ordered[idx] = val
        if len(used) != len(meta.columns):
            raise ValueError("INSERT 必须覆盖所有列（本引擎无默认值）")
        return ordered
