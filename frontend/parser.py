"""Recursive-descent parser producing plain-dict ASTs.

Supported grammar (mini_SQL)::

    statement      := create_table | create_index | drop_table
                    | insert | select | update | delete
                    | begin | commit | rollback | show_tables | describe
    create_table   := CREATE TABLE ident '(' column_def (',' column_def)* ')'
    column_def     := ident (INT | REAL | TEXT) [PRIMARY KEY]
    create_index   := CREATE INDEX ident ON ident '(' ident ')'
    drop_table     := DROP TABLE ident
    insert         := INSERT INTO ident ['(' ident (',' ident)* ')']
                      VALUES '(' value (',' value)* ')'
    select         := SELECT ( '*' | ident (',' ident)* ) FROM ident
                      [WHERE predicate] [ORDER BY ident (ASC | DESC)] [LIMIT int]
    update         := UPDATE ident SET ident '=' value [WHERE predicate]
    delete         := DELETE FROM ident [WHERE predicate]
    predicate      := ident ( '=' | '!=' | '<' | '<=' | '>' | '>=' ) value
    begin/commit/rollback := BEGIN | COMMIT | ROLLBACK
    show_tables    := SHOW TABLES
    describe       := DESCRIBE ident
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .tokenizer import Token, tokenize

_COLUMN_TYPES = {"INT", "REAL", "TEXT"}
_OPS = ("=", "!=", "<", "<=", ">", ">=")


class ParseError(ValueError):
    pass


class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # ------------------------------------------------------------------
    # token helpers
    # ------------------------------------------------------------------
    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def fail(self, tok: Token, expected: str) -> None:
        raise ParseError(f"expected {expected}, got {tok.value!r} at position {tok.pos}")

    def accept_keyword(self, kw: str) -> bool:
        if self.peek().kind == "KEYWORD" and self.peek().value == kw:
            self.pos += 1
            return True
        return False

    def expect_keyword(self, kw: str) -> None:
        if not self.accept_keyword(kw):
            self.fail(self.peek(), kw)

    def accept_symbol(self, sym: str) -> bool:
        if self.peek().kind == "SYMBOL" and self.peek().value == sym:
            self.pos += 1
            return True
        return False

    def expect_symbol(self, sym: str) -> None:
        if not self.accept_symbol(sym):
            self.fail(self.peek(), repr(sym))

    def expect_ident(self) -> str:
        tok = self.next()
        if tok.kind != "IDENT":
            self.fail(tok, "identifier")
        return tok.value

    def expect_value(self):
        """A literal value: NUMBER, STRING, or a bare '-' NUMBER."""
        if self.peek().kind == "SYMBOL" and self.peek().value == "-":
            self.pos += 1
            tok = self.next()
            if tok.kind != "NUMBER":
                self.fail(tok, "number after '-'")
            return -tok.value
        tok = self.next()
        if tok.kind not in ("NUMBER", "STRING"):
            self.fail(tok, "literal value")
        return tok.value

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def parse(self) -> dict:
        tok = self.peek()
        if tok.kind == "EOF":
            return {"op": "noop"}
        if tok.kind != "KEYWORD":
            self.fail(tok, "statement")
        method = getattr(self, "parse_" + tok.value.lower(), None)
        if method is None:
            self.fail(tok, f"statement (unsupported keyword {tok.value!r})")
        ast = method()
        self.accept_symbol(";")  # optional terminator
        if self.peek().kind != "EOF":
            self.fail(self.peek(), "end of statement")
        return ast

    # ------------------------------------------------------------------
    # CREATE / DROP
    # ------------------------------------------------------------------
    def parse_create(self) -> dict:
        self.next()  # CREATE
        if self.accept_keyword("TABLE"):
            name = self.expect_ident()
            self.expect_symbol("(")
            columns: List[dict] = []
            while True:
                col = self.expect_ident()
                type_tok = self.next()
                if type_tok.kind != "KEYWORD" or type_tok.value not in _COLUMN_TYPES:
                    self.fail(type_tok, "column type (INT, REAL, TEXT)")
                primary = self.accept_keyword("PRIMARY") and self.accept_keyword("KEY")
                if primary:
                    pass
                elif self.peek().kind == "KEYWORD" and self.peek().value == "PRIMARY":
                    self.pos += 1
                    self.expect_keyword("KEY")
                    primary = True
                columns.append({"name": col, "type": type_tok.value, "primary_key": bool(primary)})
                if not self.accept_symbol(","):
                    break
            self.expect_symbol(")")
            return {"op": "create_table", "name": name, "columns": columns}
        self.expect_keyword("INDEX")
        index_name = self.expect_ident()
        self.expect_keyword("ON")
        table = self.expect_ident()
        self.expect_symbol("(")
        column = self.expect_ident()
        self.expect_symbol(")")
        return {"op": "create_index", "name": index_name, "table": table, "column": column}

    def parse_drop(self) -> dict:
        self.next()  # DROP
        self.expect_keyword("TABLE")
        return {"op": "drop_table", "name": self.expect_ident()}

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------
    def parse_insert(self) -> dict:
        self.next()  # INSERT
        self.expect_keyword("INTO")
        table = self.expect_ident()
        columns: Optional[List[str]] = None
        if self.accept_symbol("("):
            columns = []
            while True:
                columns.append(self.expect_ident())
                if not self.accept_symbol(","):
                    break
            self.expect_symbol(")")
        self.expect_keyword("VALUES")
        self.expect_symbol("(")
        values = [self.expect_value()]
        while self.accept_symbol(","):
            values.append(self.expect_value())
        self.expect_symbol(")")
        return {"op": "insert", "table": table, "columns": columns, "values": values}

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------
    def parse_select(self) -> dict:
        self.next()  # SELECT
        cols: List[str] = []
        if self.accept_symbol("*"):
            cols = ["*"]
        else:
            cols.append(self.expect_ident())
            while self.accept_symbol(","):
                cols.append(self.expect_ident())
        self.expect_keyword("FROM")
        table = self.expect_ident()
        where = self._parse_where()
        order = None
        limit = None
        if self.accept_keyword("ORDER"):
            self.expect_keyword("BY")
            col = self.expect_ident()
            desc = False
            if self.accept_keyword("DESC"):
                desc = True
            elif self.accept_keyword("ASC"):
                desc = False
            order = (col, "desc" if desc else "asc")
        if self.accept_keyword("LIMIT"):
            tok = self.next()
            if tok.kind != "NUMBER" or not isinstance(tok.value, int):
                self.fail(tok, "integer after LIMIT")
            limit = tok.value
        return {"op": "select", "columns": cols, "table": table,
                "where": where, "order": order, "limit": limit}

    def _parse_where(self) -> Optional[Tuple[str, str, object]]:
        if not self.accept_keyword("WHERE"):
            return None
        column = self.expect_ident()
        op_tok = self.next()
        if op_tok.kind != "SYMBOL" or op_tok.value not in _OPS:
            self.fail(op_tok, "comparison operator")
        return (column, op_tok.value, self.expect_value())

    # ------------------------------------------------------------------
    # UPDATE / DELETE
    # ------------------------------------------------------------------
    def parse_update(self) -> dict:
        self.next()  # UPDATE
        table = self.expect_ident()
        self.expect_keyword("SET")
        column = self.expect_ident()
        self.expect_symbol("=")
        value = self.expect_value()
        return {"op": "update", "table": table, "set": (column, value),
                "where": self._parse_where()}

    def parse_delete(self) -> dict:
        self.next()  # DELETE
        self.expect_keyword("FROM")
        table = self.expect_ident()
        return {"op": "delete", "table": table, "where": self._parse_where()}

    # ------------------------------------------------------------------
    # transactions / meta
    # ------------------------------------------------------------------
    def parse_begin(self) -> dict:
        self.next()
        return {"op": "begin"}

    def parse_commit(self) -> dict:
        self.next()
        return {"op": "commit"}

    def parse_rollback(self) -> dict:
        self.next()
        return {"op": "rollback"}

    def parse_show(self) -> dict:
        self.next()  # SHOW
        self.expect_keyword("TABLES")
        return {"op": "show_tables"}

    def parse_describe(self) -> dict:
        self.next()  # DESCRIBE
        return {"op": "describe", "table": self.expect_ident()}


def parse_sql(text: str) -> dict:
    return Parser(tokenize(text)).parse()
