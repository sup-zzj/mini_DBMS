# mini_DBMS Phase 2：LLM 层设计（2026-09-23）

## 背景与目标

Phase 1 已交付一个从零手写的迷你关系型存储引擎（堆表 / B+ 树 / WAL 事务 / 缓冲池 / 学习式索引），并产出了三组可复现基准实验。README 中预告了 Phase 2 —— **LLM 增强数据库**。

本阶段目标（三块功能，用户已确认全做）：

1. **NL→SQL**：自然语言查询经 LLM 生成 mini_SQL，再走本引擎执行；
2. **SQL 辅助调优（TUNE）**：把 `results/*.json` 实验产物喂给 LLM，生成"为何该索引方案在此数据分布下最优"的解读；
3. **数据分布感知的索引选择器（ADVISE）**：用自然语言描述工作负载特征 → 提示选 B+ 树还是学习式索引（结合实验 1 的负面结论做约束）。

核心价值：把"引擎能力"与"LLM 判断"解耦——引擎保持确定性、可测试；LLM 只做模式识别与自然语言接口，**不进入存储正确性路径**。

## 决策记录

| 维度 | 决策 |
| --- | --- |
| 范围 | 三块全做 |
| LLM 接入 | Mock + OpenAI 兼容远程 API 双后端（工厂按环境变量选择） |
| 交互方式 | REPL 内联命令：`NL ...` / `ADVISE ...` / `TUNE [path]` |
| 实现方案 | 方案 A：独立 `llm/` 包 + `app.py` 薄集成，引擎零改动 |

## 核心约束

- **护栏**：LLM 生成的 SQL 必须先经 `frontend` 的 tokenizer + parser 验证，验证失败则不执行；
- **零第三方依赖**：真实客户端用标准库 `urllib` 调 OpenAI 兼容接口（与 Phase 1 的零依赖风格一致）；
- **离线可测**：Mock 后端确定性输出，全部测试无 key、无网络；
- **退出码约定不变**：0 正常 / 1 启动失败 / 2 参数错误；
- 环境变量：`MINI_DBMS_API_BASE` / `MINI_DBMS_API_KEY` / `MINI_DBMS_MODEL`（默认 `deepseek-chat`）。

## 架构

```
mini_DBMS/
├── llm/                  # 新增：LLM 层（不进存储正确性路径）
│   ├── __init__.py
│   ├── client.py         # LLMClient ABC + MockLLMClient + OpenAICompatClient + create_client 工厂
│   ├── prompts.py        # schema 序列化 + 提示词构建 + 响应解析
│   ├── advisor.py        # ADVISE：数据分布感知的索引选择器
│   ├── tuning.py         # TUNE：读 results/*.json 生成解读
│   └── cli.py            # NL / ADVISE / TUNE 三个命令实现（编排，含护栏）
├── app.py                # 加三个命令入口（薄路由）
└── tests/
    ├── test_llm_client.py
    ├── test_nl_to_sql.py
    ├── test_advisor.py
    └── test_tuning.py
```

数据流：`REPL 输入 → 命令前缀路由 → llm/cli.py → client.complete() → 结果解析 → （NL 场景）frontend 验证 → engine 执行 → 渲染`。

## 模块设计

### 1. `llm/client.py`

```python
class LLMClient(ABC):
    """统一对话补全接口。"""
    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str: ...

class MockLLMClient(LLMClient):
    """确定性后端：内置关键词规则表 → 预设 SQL / 建议 / 解读模板。
    用于单元测试与离线演示；不发起任何网络请求。"""

class OpenAICompatClient(LLMClient):
    """真实后端：POST {base_url}/chat/completions（OpenAI 兼容格式）。
    配置来自 MINI_DBMS_API_BASE / MINI_DBMS_API_KEY / MINI_DBMS_MODEL。"""

def create_client() -> LLMClient:
    """工厂：有 MINI_DBMS_API_KEY 用真实后端，否则回退 Mock。"""
```

- `OpenAICompatClient` 用 `urllib.request`，超时、HTTP 错误、JSON 解析失败均抛 `LLMError`（自定义异常，`llm/__init__.py` 导出）；
- 失败语义：`complete()` 只抛异常不返回半成品；上层捕获后转 REPL 错误消息。

### 2. `llm/prompts.py`

- `build_schema_text(catalog: Catalog) -> str`：遍历 `catalog.tables`，序列化为紧凑文本（表、列、类型、主键、已有索引）；无表时输出"（空库）"；
- `build_nl_to_sql_prompt(schema_text, nl) -> tuple[system, user]`：system 定义角色与语法白名单（`CREATE / INSERT / SELECT / UPDATE / DELETE / BEGIN / COMMIT / ROLLBACK / SHOW / DESCRIBE / DROP`，谓词 op 白名单 `= != < <= > >=`）；user 给出 schema 与 NL 请求；含 2 个 few-shot 示例（SELECT 与 INSERT）；
- `parse_sql_response(text) -> str`：剥 markdown ```sql 围栏（`sql`/`SQL` 或不带语言标签均可）、取首个语句、去尾部多余分号；解析失败抛 `LLMError`。

### 3. `llm/advisor.py`

- `advise(client, workload_desc: str, stats: dict) -> Advice`，`Advice` 为 dataclass：`index_type: str`（`"btree" | "learned" | "none"`）、`reason: str`、`fallback: bool`；
- 提示词要求 LLM 只输出一行 JSON：`{"index_type": "...", "reason": "..."}`；
- JSON 解析失败 → 兜底关键词启发式：点查/范围/排序 → `btree`；只读批量/全表扫 → `learned`；写多 → `none`；命中兜底时 `fallback=True`；
- 输出时附加 README 实验结论的诚实提示（learned 内存略省但查询约慢 2.2×）。

### 4. `llm/tuning.py`

- `tune(client, json_path: str) -> str`：读取 JSON → 汇总成紧凑表格文本（variant / 指标 / 值）→ 提示词要求生成"为何该索引方案最优 + 权衡 + 局限"解读 → 返回报告文本；
- Mock 模式：内置模板基于数据本身生成确定性解读（比较各 variant 指标，选最优并给出理由），不依赖 LLM 也能演示。

### 5. `llm/cli.py`

- `run_nl(engine, client, text) -> result`：build schema → prompt → complete → `parse_sql_response` → **护栏：`Parser(tokenize(sql)).parse()` 验证** → `Executor.execute` 返回结果；验证失败抛 `LLMError`（携带 LLM 原文）；
- `run_advise(engine, client, workload_desc) -> str`：汇总库统计（表数、每表行数可经 `select` 计数）→ `advise` → 渲染建议文本；
- `run_tune(client, json_path) -> str`：`tune` 渲染报告。

### 6. `app.py` 改动（薄）

- REPL 主循环中，在 SQL 解析**之前**按前缀路由：`NL ` / `ADVISE` / `TUNE ` 开头 → 对应 `llm.cli` 函数；
- 启动时打印当前 LLM 后端（`LLM: mock` 或 `LLM: openai-compat (model)`）；
- `LLMError` 捕获后打印 `错误: ...` 不崩 REPL；`TUNE` 缺省路径时取 `results/` 下最新 `*.json`；
- `_HELP` 增加三个命令的说明。

## 错误处理

| 场景 | 行为 |
| --- | --- |
| LLM 输出非 SQL / 语法非法 | 护栏拦截，报错并展示 LLM 原文，不执行 |
| 网络超时 / HTTP 错误（真实后端） | `LLMError` → REPL 打印错误，不崩溃 |
| JSON 解析失败（ADVISE） | 兜底启发式 + `fallback=True` 标注 |
| 文件不存在（TUNE） | 明确报错，提示可用路径 |

## 测试策略（全部离线、零 key）

- `test_llm_client.py`：Mock 确定性断言；`OpenAICompatClient` 用 monkeypatch 假响应验证请求体（URL、model、消息结构）与响应解析；
- `test_nl_to_sql.py`：schema 文本内容、围栏剥离、非法 SQL 拒执行（护栏）、解析失败含原文；
- `test_advisor.py`：正常 JSON 响应、坏 JSON 兜底、兜底标注；
- `test_tuning.py`：用 `results/index_benchmark.json` 的真实产物生成报告，断言含各 variant 名；
- 验收：`pytest` 全绿；`python app.py` 冒烟：Mock 模式下 `NL` / `ADVISE` / `TUNE` 三条命令均可演示。

## 交付物

1. `llm/` 包五个模块 + `__init__.py`；
2. `app.py` 三个命令 + 帮助文本 + 后端打印；
3. 四个测试文件；
4. README 的 Phase 2 章节由"预告"改为"已实现"（含双后端配置说明与命令示例）。

## 参考资料

- README「实验解读」：学习式索引约慢 2.2×、内存 0.77 vs 0.76 MB（keys 数组计入）、E=4 时 3538 段爆炸——ADVISE 的诚实提示与 TUNE 的解读均以这些数字为准。
- `engine/catalog.py`：`Catalog.tables`（`TableMeta.columns / indexes / primary_key`）——schema 序列化来源。
- `frontend/`：`tokenize` / `Parser` / `ParseError`——护栏依赖。
