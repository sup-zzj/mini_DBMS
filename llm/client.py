"""LLM client abstraction: deterministic mock + OpenAI-compatible remote.

This module is the *only* boundary between mini_DBMS and an external LLM.
The engine never imports it; only the REPL layer (``app.py`` / ``llm.cli``)
does.  ``create_client`` picks the backend from the environment so the
whole project stays testable offline (mock) while supporting a real model
(DeepSeek etc.) when ``MINI_DBMS_API_KEY`` is set.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, Optional

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
ENV_BASE = "MINI_DBMS_API_BASE"
ENV_KEY = "MINI_DBMS_API_KEY"
ENV_MODEL = "MINI_DBMS_MODEL"

_SQL_TABLE_RE = re.compile(r"表\s+(\w+)")


class LLMError(RuntimeError):
    """Raised when the LLM backend fails (network, protocol, bad output)."""


class LLMClient(ABC):
    """Uniform chat-completion interface."""

    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        """Return the assistant reply for the (system, user) prompt pair."""


class MockLLMClient(LLMClient):
    """Deterministic backend for tests and offline demos.

    Match order:
    1. exact keyword rules injected via the constructor (user-text substrings);
    2. system-prompt markers ``[NL2SQL]`` / ``[ADVISE]`` / ``[TUNE]``.

    ``[TUNE]`` deliberately returns ``""`` so :mod:`llm.tuning` falls back
    to its data-driven template.
    """

    name = "mock"

    def __init__(self, rules: Optional[Dict[str, str]] = None) -> None:
        self.rules = rules or {}

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        for keyword, reply in self.rules.items():
            if keyword.lower() in user.lower():
                return reply
        if "[NL2SQL]" in system:
            m = _SQL_TABLE_RE.search(user)
            if m and ("查询" in user or "select" in user.lower()):
                return f"SELECT * FROM {m.group(1)} LIMIT 10"
            return ""
        if "[ADVISE]" in system:
            # The user prompt now embeds catalog stats and benchmark facts
            # (may contain insert/update/learned/写 ...).  Only the workload
            # segment after the "工作负载：" marker drives keyword matching.
            marker = "工作负载："
            if marker in user:
                low = user.split(marker, 1)[1].lower()
            else:
                low = user.lower()
            if any(k in low for k in ("insert", "update", "delete")) or "写" in low:
                return '{"index_type": "none", "reason": "写多读少，索引维护成本高"}'
            if "learned" in low or "只读" in low or "批量" in low:
                return '{"index_type": "learned", "reason": "只读批量扫描适合学习式索引"}'
            return '{"index_type": "btree", "reason": "点查/范围/排序场景，B+ 树更稳"}'
        return ""


class OpenAICompatClient(LLMClient):
    """Real backend: POST ``{base_url}/chat/completions`` (OpenAI-compatible).

    Uses only the standard library (``urllib``) to keep the project
    dependency-free, consistent with the engine itself.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"openai-compat({model})"

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM 请求失败: {exc}") from None
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 响应格式异常: {exc}") from None


def create_client() -> LLMClient:
    """Factory: real backend when ``MINI_DBMS_API_KEY`` is set, else mock."""
    api_key = os.environ.get(ENV_KEY, "").strip()
    if api_key:
        return OpenAICompatClient(
            api_key=api_key,
            base_url=os.environ.get(ENV_BASE, DEFAULT_BASE_URL),
            model=os.environ.get(ENV_MODEL, DEFAULT_MODEL),
        )
    return MockLLMClient()
