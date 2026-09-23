"""Prompt construction and response parsing for the LLM layer.

Two responsibilities:
* turn the live catalog into compact, schema-aware prompt text so the
  model hallucinates less (tables / columns / types actually exist);
* normalise the model's raw reply into one executable SQL statement.
"""

from __future__ import annotations

from typing import Tuple

from engine.catalog import Catalog

from .client import LLMError

_GRAMMAR = (
    "支持的语句：CREATE TABLE / CREATE INDEX / DROP TABLE / "
    "INSERT INTO ... VALUES / SELECT ... FROM ... [WHERE col op value] "
    "[ORDER BY col ASC|DESC] [LIMIT n] / UPDATE ... SET ... WHERE ... / "
    "DELETE FROM ... WHERE ... / BEGIN / COMMIT / ROLLBACK / SHOW TABLES / DESCRIBE。"
    "谓词 op 仅限 = != < <= > >=；字符串字面量用单引号。"
)

_NL2SQL_SYSTEM = (
    "[NL2SQL]\n"
    "你是 mini_DBMS（一个教学型迷你关系数据库）的 SQL 翻译器。\n"
    "只输出一条可执行的 SQL，不要任何解释、前后缀或 markdown 围栏。\n"
    "语法白名单：" + _GRAMMAR + "\n"
    "示例：\n"
    "用户：查询 users 表里 id 大于 10 的所有行\n"
    "SQL：SELECT * FROM users WHERE id > 10\n"
    "用户：向 users 表插入一行，id=1，name='alice'\n"
    "SQL：INSERT INTO users (id, name) VALUES (1, 'alice')\n"
    "如果需求无法用白名单语法表达，输出一行：ERROR: <原因>"
)


def build_schema_text(catalog: Catalog) -> str:
    """Serialize the catalog into compact Chinese schema text."""
    if not catalog.tables:
        return "（当前数据库为空库，没有任何表）"
    lines = []
    for meta in catalog.tables.values():
        cols = []
        for col in meta.columns:
            tag = " 主键" if col.primary_key else ""
            cols.append(f"{col.name} {col.type}{tag}")
        idx = [f"{ix.name}({ix.column})" for ix in meta.indexes]
        suffix = f"；索引: {', '.join(idx)}" if idx else ""
        lines.append(f"表 {meta.name}: 列 {', '.join(cols)}{suffix}")
    return "\n".join(lines)


def build_nl_to_sql_prompt(schema_text: str, nl: str) -> Tuple[str, str]:
    """Return ``(system, user)`` for an NL→SQL completion."""
    user = f"当前数据库 schema：\n{schema_text}\n\n用户的自然语言请求：{nl}\n\n只输出 SQL："
    return _NL2SQL_SYSTEM, user


def parse_sql_response(text: str) -> str:
    """Strip fences / trailing semicolons and return a single statement.

    Raises :class:`llm.client.LLMError` on empty input.
    """
    if not text or not text.strip():
        raise LLMError("LLM 返回空响应")
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
    t = " ".join(t.split())
    while t.endswith(";"):
        t = t[:-1].rstrip()
    return t
