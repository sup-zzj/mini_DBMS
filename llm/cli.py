"""REPL-facing commands for the LLM layer.

``NL`` is the only path that touches the engine, and it goes through the
frontend parser as a guard: an LLM suggestion that does not parse is
rejected *before* execution.  ``ADVISE`` and ``TUNE`` only produce text.
"""

from __future__ import annotations

from typing import Tuple

from engine import StorageEngine
from frontend.executor import Executor
from frontend.parser import ParseError, Parser
from frontend.tokenizer import tokenize

from .advisor import advise
from .client import LLMClient, LLMError
from .prompts import build_nl_to_sql_prompt, build_schema_text, parse_sql_response
from .tuning import tune

_LEARNED_HINT = (
    "（注意：README 实验表明学习式索引查询约慢 2.2×、内存与数组相近，请谨慎选型）"
)


def run_nl(engine: StorageEngine, exec_: Executor, client: LLMClient, text: str) -> Tuple:
    """Natural language -> validated SQL -> engine execution."""
    schema = build_schema_text(engine.catalog)
    system, user = build_nl_to_sql_prompt(schema, text)
    reply = client.complete(system, user)
    sql = parse_sql_response(reply)
    try:
        ast = Parser(tokenize(sql)).parse()
    except ParseError as exc:
        raise LLMError(
            f"LLM 生成的 SQL 未通过语法护栏：{exc}\nLLM 原文：{reply}"
        ) from None
    return exec_.execute(ast)


def run_advise(client: LLMClient, workload: str) -> str:
    """Workload description -> index suggestion text."""
    adv = advise(client, workload)
    tag = " [兜底判定]" if adv.fallback else ""
    hint = f"\n{_LEARNED_HINT}" if adv.index_type == "learned" else ""
    return f"建议索引类型：{adv.index_type}{tag}\n理由：{adv.reason}{hint}"


def run_tune(client: LLMClient, json_path: str) -> str:
    """Benchmark JSON -> tuning interpretation text."""
    return tune(client, json_path)
