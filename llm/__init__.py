"""LLM 层：自然语言接口（不进存储正确性路径）。

三块能力：
* NL→SQL —— ``llm.cli.run_nl``（经 frontend 语法护栏后执行）
* 索引选择器 —— ``llm.advisor.advise``（ADVISE 命令）
* 实验调优解读 —— ``llm.tuning.tune``（TUNE 命令）

对外异常统一为 :class:`llm.client.LLMError`。
"""

from .client import (
    LLMClient,
    LLMError,
    MockLLMClient,
    OpenAICompatClient,
    create_client,
)
from .advisor import Advice, advise

__all__ = [
    "LLMClient",
    "LLMError",
    "MockLLMClient",
    "OpenAICompatClient",
    "create_client",
    "Advice",
    "advise",
]
