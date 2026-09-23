"""System catalog: schema and index metadata persisted as JSON.

The catalog is the engine's memory of *what exists*:

* tables: column list (name / type / primary-key flag), heap head page id;
* indexes: name, indexed column, root page id (for B+ tree indexes).

It is loaded at engine start and rewritten on every schema change.  Page
ids stored here are the anchors the buffer pool + disk manager operate on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

INT, REAL, TEXT = "INT", "REAL", "TEXT"
COLUMN_TYPES = (INT, REAL, TEXT)


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "primary_key": self.primary_key}

    @classmethod
    def from_dict(cls, d: dict) -> "Column":
        return cls(d["name"], d["type"].upper(), bool(d.get("primary_key", False)))


@dataclass
class IndexInfo:
    name: str
    table: str
    column: str
    kind: str = "btree"  # v1: only unique B+ tree indexes
    root_page: int = 0
    built: bool = False  # False until the tree is created/rebuilt on disk

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "table": self.table,
            "column": self.column,
            "kind": self.kind,
            "root_page": self.root_page,
            "built": self.built,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndexInfo":
        return cls(
            d["name"],
            d["table"],
            d["column"],
            d.get("kind", "btree"),
            int(d.get("root_page", 0)),
            bool(d.get("built", False)),
        )


@dataclass
class TableMeta:
    name: str
    columns: List[Column] = field(default_factory=list)
    indexes: List[IndexInfo] = field(default_factory=list)
    root_page: int = 0

    def column(self, name: str) -> Optional[Column]:
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None

    @property
    def primary_key(self) -> Optional[Column]:
        for col in self.columns:
            if col.primary_key:
                return col
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "indexes": [i.to_dict() for i in self.indexes],
            "root_page": self.root_page,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableMeta":
        return cls(
            d["name"],
            [Column.from_dict(c) for c in d["columns"]],
            [IndexInfo.from_dict(i) for i in d.get("indexes", [])],
            int(d.get("root_page", 0)),
        )


class Catalog:
    """JSON-backed catalog for a single database directory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.tables: Dict[str, TableMeta] = {}
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.tables = {
                name: TableMeta.from_dict(d) for name, d in data.get("tables", {}).items()
            }

    def save(self) -> None:
        payload = {"tables": {n: t.to_dict() for n, t in self.tables.items()}}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    # table / index operations
    # ------------------------------------------------------------------
    def create_table(self, meta: TableMeta) -> None:
        if meta.name in self.tables:
            raise ValueError(f"table {meta.name!r} already exists")
        self.tables[meta.name] = meta
        self.save()

    def drop_table(self, name: str) -> None:
        self.tables.pop(name, None)
        self.save()

    def get_table(self, name: str) -> TableMeta:
        try:
            return self.tables[name]
        except KeyError:
            raise ValueError(f"unknown table {name!r}") from None

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def add_index(self, index: IndexInfo) -> None:
        table = self.get_table(index.table)
        for existing in table.indexes:
            if existing.column == index.column:
                raise ValueError(
                    f"table {index.table!r} already has an index on column "
                    f"{index.column!r}"
                )
        table.indexes.append(index)
        self.save()

    def drop_index(self, table: str, name: str) -> Optional[IndexInfo]:
        meta = self.get_table(table)
        for i, idx in enumerate(meta.indexes):
            if idx.name == name:
                return meta.indexes.pop(i)
        return None

    def save_table(self, meta: TableMeta) -> None:
        """Persist page-id updates (e.g. a rebuilt index root)."""
        self.save()

    def __repr__(self) -> str:
        return f"Catalog(tables={list(self.tables)})"
