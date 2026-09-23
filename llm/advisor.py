"""Data-distribution-aware index advisor.

Asks the LLM to pick ``btree`` / ``learned`` / ``none`` for a workload
described in natural language.  When the model's reply is not valid JSON,
a keyword heuristic takes over so the REPL command never dies; the
``fallback`` flag tells the caller the advice was heuristic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from .client import LLMClient, LLMError

_ADVISE_SYSTEM = (
    "[ADVISE]\n"
    "你是数据库索引选型顾问。结合库统计与实验基准事实判断，"
    "理由里尽量引用实测数字（如查询耗时 µs、内存 MB、构建耗时 s）。"
    "只输出一行 JSON，不要任何其他内容：\n"
    '{"index_type": "btree" | "learned" | "none", "reason": "一句话理由"}\n'
    "选择规则：\n"
    "- 点查 / 范围扫描 / 排序 → btree\n"
    "- 只读批量扫描、数据分布可被低阶模型刻画 → learned\n"
    "- 写多读少、数据频繁变化 → none（不建索引）"
)

_BENCHMARK_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "index_benchmark.json",
)


@dataclass
class Advice:
    index_type: str  # "btree" | "learned" | "none"
    reason: str
    fallback: bool = False


def build_advise_prompt(
    workload: str, stats_text: str, benchmark_text: str
) -> Tuple[str, str]:
    """Compose the (system, user) prompt pair for the advisor.

    The user prompt embeds real catalog statistics and measured benchmark
    facts so the LLM can ground its suggestion in data and evidence.
    """
    user = (
        "当前库统计：\n"
        f"{stats_text}\n\n"
        "实验基准事实（本引擎实测）：\n"
        f"{benchmark_text}\n\n"
        f"工作负载：{workload}\n\n"
        "只输出 JSON："
    )
    return _ADVISE_SYSTEM, user


def _table_stats(engine) -> str:
    """One compact line per catalog table: columns, row count, indexes."""
    lines = []
    for meta in engine.catalog.tables.values():
        cols = "；".join(
            f"{c.name} {c.type}{' PK' if c.primary_key else ''}"
            for c in meta.columns
        )
        try:
            n = len(engine.select(meta.name))
        except Exception:
            n = -1
        idx = "；".join(f"{i.name}({i.column})" for i in meta.indexes) or "无"
        lines.append(f"表 {meta.name}（行数≈{n}）：列 {cols}；索引：{idx}")
    return "\n".join(lines) if lines else "（空库）"


def _benchmark_text(path: Optional[str] = None) -> str:
    """Compact measured-benchmark facts from ``results/index_benchmark.json``.

    Returns a fallback notice when the file is missing or unparsable.
    """
    p = path or _BENCHMARK_DEFAULT
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        variants = data.get("variants") or []
    except (OSError, ValueError, json.JSONDecodeError):
        return "（无基准数据，仅按工作负载与库统计判断）"
    if not variants:
        return "（无基准数据，仅按工作负载与库统计判断）"
    lines = []
    by_name = {}
    for v in variants:
        name = v.get("name", "?")
        by_name[name] = v
        parts = [f"lookup {v.get('lookup_ns', 0.0):.2f}µs",
                 f"内存 {v.get('memory_mb', 0.0):.2f}MB"]
        if "build_s" in v:
            parts.append(f"构建 {v['build_s']:.2f}s")
        lines.append(f"{name}: {', '.join(parts)}")
    sa = by_name.get("sorted_array")
    le = by_name.get("learned_e16")
    if sa is not None and le is not None and sa.get("lookup_ns"):
        ratio = le["lookup_ns"] / sa["lookup_ns"]
        lines.append(f"结论：learned 比 sorted_array 慢约 {ratio:.1f}×")
    return "\n".join(lines)


def _fallback_advise(workload: str) -> Advice:
    low = workload.lower()
    if any(k in low for k in ("insert", "update", "delete")) or "写" in workload:
        return Advice("none", "写多读少，索引维护成本高于收益", True)
    if "learned" in low or "只读" in workload or "批量" in workload:
        return Advice("learned", "只读批量扫描，适合学习式索引", True)
    return Advice("btree", "点查/范围/排序场景，B+ 树更稳", True)


def advise(
    client: LLMClient,
    workload: str,
    engine=None,
    benchmark: Optional[str] = None,
) -> Advice:
    """Return an index-type suggestion for the workload description.

    When ``engine`` is given, real catalog statistics are embedded into the
    prompt; ``benchmark`` overrides the default ``results/index_benchmark.json``
    path used for measured facts.  JSON parsing and the heuristic fallback
    behave exactly as before.
    """
    stats = _table_stats(engine) if engine is not None else "（无库统计）"
    bench = _benchmark_text(benchmark)
    system, user = build_advise_prompt(workload, stats, bench)
    try:
        reply = client.complete(system, user)
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
