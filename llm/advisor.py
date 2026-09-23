"""Data-distribution-aware index advisor.

Asks the LLM to pick ``btree`` / ``learned`` / ``none`` for a workload
described in natural language.  When the model's reply is not valid JSON,
a keyword heuristic takes over so the REPL command never dies; the
``fallback`` flag tells the caller the advice was heuristic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .client import LLMClient, LLMError

_ADVISE_SYSTEM = (
    "[ADVISE]\n"
    "你是数据库索引选型顾问。根据用户描述的工作负载，只输出一行 JSON，"
    "不要任何其他内容：\n"
    '{"index_type": "btree" | "learned" | "none", "reason": "一句话理由"}\n'
    "选择规则：\n"
    "- 点查 / 范围扫描 / 排序 → btree\n"
    "- 只读批量扫描、数据分布可被低阶模型刻画 → learned\n"
    "- 写多读少、数据频繁变化 → none（不建索引）"
)


@dataclass
class Advice:
    index_type: str  # "btree" | "learned" | "none"
    reason: str
    fallback: bool = False


def _fallback_advise(workload: str) -> Advice:
    low = workload.lower()
    if any(k in low for k in ("insert", "update", "delete")) or "写" in workload:
        return Advice("none", "写多读少，索引维护成本高于收益", True)
    if "learned" in low or "只读" in workload or "批量" in workload:
        return Advice("learned", "只读批量扫描，适合学习式索引", True)
    return Advice("btree", "点查/范围/排序场景，B+ 树更稳", True)


def advise(client: LLMClient, workload: str) -> Advice:
    """Return an index-type suggestion for the workload description."""
    try:
        reply = client.complete(_ADVISE_SYSTEM, workload)
    except LLMError:
        reply = ""
    if not reply or not reply.strip():
        return _fallback_advise(workload)
    try:
        data = json.loads(reply.strip().strip("`"))
        itype = str(data["index_type"]).strip().lower()
        if itype not in ("btree", "learned", "none"):
            raise ValueError(itype)
        return Advice(itype, str(data.get("reason", "")))
    except (ValueError, TypeError, json.JSONDecodeError):
        return _fallback_advise(workload)
