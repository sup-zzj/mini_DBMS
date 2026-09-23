"""A hand-written tokenizer for the mini_SQL dialect.

No external dependencies (no PLY / lex).  Token kinds::

    KEYWORD  statement / clause keywords, upper-cased
    IDENT    table / column names
    NUMBER   int or float literals
    STRING   '...' or "..." literals (doubled quotes escape a quote)
    SYMBOL   ( ) , ; = < > <= >= != *
    EOF

``#`` starts a comment that runs to the end of the line.
"""

from __future__ import annotations

from typing import List, NamedTuple

KEYWORDS = {
    "BEGIN", "BY", "COMMIT", "CREATE", "DELETE", "DESC", "DESCRIBE",
    "DROP", "FROM", "IN", "INDEX", "INSERT", "INT", "INTO", "KEY",
    "LIMIT", "ON", "ORDER", "PRIMARY", "REAL", "ROLLBACK", "SELECT",
    "SET", "SHOW", "TABLE", "TABLES", "TEXT", "UPDATE", "VALUES",
    "WHERE",
}

_SYMBOLS = "(),;=<>*"


class Token(NamedTuple):
    kind: str      # KEYWORD | IDENT | NUMBER | STRING | SYMBOL | EOF
    value: object
    pos: int       # character offset in the source


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch in ("'", '"'):
            quote, j, buf = ch, i + 1, []
            while j < n:
                if text[j] == quote:
                    if j + 1 < n and text[j + 1] == quote:
                        buf.append(quote)
                        j += 2
                        continue
                    j += 1
                    break
                buf.append(text[j])
                j += 1
            tokens.append(Token("STRING", "".join(buf), i))
            i = j
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            raw = text[i:j]
            tokens.append(Token("NUMBER", float(raw) if "." in raw else int(raw), i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            if word.upper() in KEYWORDS:
                tokens.append(Token("KEYWORD", word.upper(), i))
            else:
                tokens.append(Token("IDENT", word, i))
            i = j
            continue
        two = text[i : i + 2]
        if two in ("<=", ">=", "!="):
            tokens.append(Token("SYMBOL", two, i))
            i += 2
            continue
        if ch in _SYMBOLS:
            tokens.append(Token("SYMBOL", ch, i))
            i += 1
            continue
        raise ValueError(f"unexpected character {ch!r} at position {i}")
    tokens.append(Token("EOF", None, n))
    return tokens
