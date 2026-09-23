"""Tuning reporter: turn ``results/*.json`` benchmark artifacts into a
human-readable "why is this index plan optimal" explanation.

The mock backend returns ``""`` for ``[TUNE]`` prompts, so ``tune()``
falls back to a deterministic, data-driven template that reads the
numbers themselves — the demo works offline without any model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .client import LLMClient

_TUNE_SYSTEM = (
    "[TUNE]\n"
    "你是数据库性能实验解读助手。根据给定的基准数据，输出 3-6 句话的中文解读："
    "哪个方案在什么指标上最优、为什么、有什么权衡与局限。"
    "不要输出 markdown 表格，用自然段落。"
)


def _summary_text(data: Dict[str, Any]) -> str:
    """Compact, flat text of a benchmark JSON (meta + variants)."""
    lines = []
    meta = data.get("meta", {})
    if meta:
        lines.append("meta: " + json.dumps(meta, ensure_ascii=False))
    for v in data.get("variants", []):
        cells = ", ".join(f"{k}={v[k]}" for k in v if k != "name")
        lines.append(f"variant {v.get('name', '?')}: {cells}")
    return "\n".join(lines)


def _template_report(data: Dict[str, Any]) -> str:
    """Deterministic report: pick the fastest lookup variant by the data."""
    variants = data.get("variants", [])
    if not variants:
        return "该 JSON 不包含 variants，无法生成解读。"
    best = min(variants, key=lambda v: v.get("lookup_ns", float("inf")))
    lookup = best.get("lookup_ns", 0.0)
    memory = best.get("memory_mb", 0.0)
    build = best.get("build_s", 0.0)
    return (
        f"综合 {best.get('name')} 的 lookup_ns={lookup:.1f} 最小，"
        "是当前实验中最快的查找方案。\n"
        f"内存占用 {memory:.2f} MB，构建耗时 {build:.2f}s。\n"
        "解读：索引选型是 算法复杂度 × 常数因子 × 内存代价 的权衡；"
        "该结论与 README 实验解读一致。"
    )


def tune(client: LLMClient, json_path: str) -> str:
    """Load ``json_path`` and produce a tuning interpretation."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到实验数据: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    text = _summary_text(data)
    reply = client.complete(_TUNE_SYSTEM, text)
    if reply and reply.strip():
        return reply.strip()
    return _template_report(data)
