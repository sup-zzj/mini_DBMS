# mini_DBMS

一个**从零手写的小型关系型存储引擎 + 实验研究项目**：堆表（slotted page）、B+ 树索引、带 WAL 的 ACID 事务（no-steal / no-force）、可插拔 LRU / Clock / Random 缓冲池、以及一个诚实的**学习式索引（Learned Index）**实现——并配有三组可复现的基准实验。

> **定位**：这不是一个"能用"的数据库，而是一个**教科书知识到工程代码的完整落地**。每个核心概念（预写日志、崩溃恢复、脏页钉住、B+ 树分裂、学习式索引）都有对应代码、单元测试、和一份"解释为什么"的 README。适合面试讲深度。

> **诚实声明**：本项目的基准实验报告了**负面结果**——学习式索引在 10 万整数点查场景下并不比二分查找快、也没省内存。这类"结论与论文宣传相悖"的实验，恰恰是最能体现工程判断力的面试素材。

---

## 目录

- [功能一览](#功能一览)
- [快速开始](#快速开始)
- [架构](#架构)
- [关键设计决策](#关键设计决策)
- [实验与解读](#实验与解读)
- [复现约定](#复现约定)
- [已知局限](#已知局限)
- [Phase 2：LLM 结合](#phase-2llm-结合)
- [参考资料](#参考资料)

---

## 功能一览

| 层 | 组件 | 说明 |
|---|---|---|
| 物理 | `page.py` | 4 KB 定长页 + 每表一个数据文件的 `DiskManager`（空闲页复用） |
| 缓冲 | `buffer_pool.py` | 容量受限页缓存；LRU / Clock(≈LRU) / Random 三种替换策略；pin 引用计数防驱逐；脏页延迟写回；命中率统计 |
| 日志 | `wal.py` | 追加式预写日志，CRC32 校验，崩溃恢复按已提交事务 redo；no-steal / no-force |
| 事务 | `txn.py` | BEGIN / COMMIT / ROLLBACK；内存 undo 日志；事务触达的脏页被 pin 住（no-steal） |
| 存储 | `table.py` | slotted page 堆表；紧凑二进制行编码；墓碑删除（保留 offset 便于 undo 原位恢复） |
| 目录 | `catalog.py` | 表/列/索引元数据（JSON 持久化） |
| 索引 | `btree.py` | B+ 树，可插拔 NodeStore（内存 dict / 序列化进 4KB 页）；分裂、墓碑删除、叶内压缩 |
| 学习式 | `learned_index.py` | 两级 RMI（Recursive Model Index）：等分路由 + 贪心分段线性拟合 + 误差界内精确搜索；gap key 自动回退段内二分，任意 key 精确；含 `BlockLearnedIndex` 页路由变体；静态只读 |
| 编排 | `storage_engine.py` | DDL / DML / 事务 / 崩溃恢复 / checkpoint 全链路 |
| 前端 | `frontend/` | 手写 tokenizer + 递归下降 parser + executor；`app.py` 为 REPL |
| 实验 | `scripts/` | 五组基准（索引对比 / 分布敏感性 / 页路由 / 缓冲池 / 崩溃恢复）+ 出图（PNG/PDF） |
| LLM 自然语言接口 | `llm/` 包：NL→SQL（语法护栏）、ADVISE 索引选型、TUNE 调优解读；Mock / OpenAI 兼容双后端 |

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 单元测试（71 个，含 B+ 树 fuzz、WAL 损坏恢复、事务回滚、崩溃恢复、LLM 层双后端/护栏/选型/调优）
python -m pytest

# 2) 进入 REPL，用 SQL 操作数据库
python app.py --data-dir ./db
# 示例：
#   CREATE TABLE users (id INT PRIMARY KEY, name TEXT, score REAL);
#   INSERT INTO users VALUES (1, 'alice', 9.5);
#   SELECT * FROM users WHERE score >= 9.0 ORDER BY id DESC LIMIT 10;
#   BEGIN; INSERT INTO users VALUES (2, 'bob', 7.0); ROLLBACK;
#   SHOW TABLES; DESCRIBE users;   (输入 exit / quit / \q 退出)

# 3) 复现实验
python scripts/run_benchmarks.py                     # 实验1 索引对比 -> results/index_benchmark.json
python scripts/run_benchmarks.py --distribution uniform clustered zipf   # 实验4 分布敏感性 -> results/index_distribution.json
python scripts/run_benchmarks.py --block-sizes 32 64 128 256            # 实验5 页路由 -> results/index_block.json
python scripts/bench_buffer.py          # 缓冲池替换策略实验 -> results/buffer_pool_benchmark.json
python scripts/crash_recovery_demo.py   # 崩溃恢复演示 -> results/crash_recovery_demo.json
python scripts/make_plots.py            # 出版级图表 -> figures/*.png + *.pdf
```

### 支持的 SQL（mini_SQL）

`CREATE TABLE`（每表必须且仅有一个主键，类型限 INT/TEXT）、`CREATE INDEX`（二级索引，唯一）、`DROP TABLE`、`INSERT`、`SELECT`（`WHERE` 六种比较运算符 / `ORDER BY` / `LIMIT`）、`UPDATE`、`DELETE`、`BEGIN` / `COMMIT` / `ROLLBACK`、`SHOW TABLES`、`DESCRIBE`。

---

## 架构

```
 +---------------------------------------------------+
 |  llm/  NL→SQL（经 frontend 护栏）· ADVISE · TUNE   |
 |  Mock / OpenAI 兼容双后端（不进存储正确性路径）      |
 +------------------------+--------------------------+
                           |  app.py（REPL）路由
                 +---------v-----------+
                 |  frontend/  tokenizer · parser · executor · REPL  |
                 +-------------+-------------+
                               |  mini_SQL AST
                 +-------------v-------------+
                 |   StorageEngine (storage_engine.py)             |
                 |   DDL · DML · 事务 · 崩溃恢复 · checkpoint      |
                 +---+---------+-----------+---------+-------------+
                     |         |           |         |
              +------v--+  +---v----+  +---v----+  +-v------------+
              | catalog |  | btree  |  | table  |  | txn (+undo)  |
              | (JSON)  |  | B+树   |  | 堆表   |  +------+-------+
              +---------+  +---+----+  +---+----+         |
                              |            |              |
                 +------------v------------v-------+      |
                 |         BufferPool  (LRU/Clock/Random) |  |
                 +------------+------------+-------+      |
                              |            |              |
                 +------------v------------v-------+      |
                 |  WAL (redo, CRC32, no-steal/no-force)  |<-
                 +-------------------+-----------------+
                                     |
                 +-------------------v-----------------+
                 |   DiskManager：每表一个 .dat 文件 + wal.log  |
                 +-------------------------------------+
```

**无 WAL 时的隐患**（为什么需要日志）：事务修改的页先写内存缓冲池；若在落盘前崩溃，已"提交"的数据会丢失（no-force 的代价）；若在提交前把未提交的脏页刷出去又崩溃，磁盘上会出现半个事务（no-steal 要避免的）。WAL 是两者之间的桥梁——**提交只保证日志里有一条 COMMIT 记录**，恢复时据此 redo。

---

## 关键设计决策

### 1. 事务：no-steal / no-force + 内存 undo

- **no-steal**：事务每修改一个页，`_pin_hook` 就把该页 pin 住直到事务结束，因此**未提交的脏页永远不会被替换策略写出**。崩溃模拟（`discard()`）后磁盘上只有已提交数据。
- **no-force**：提交时**不强制刷页**，只 `fsync` WAL 的 COMMIT 记录；脏页延迟到 checkpoint 或淘汰时落盘。
- **undo**：每个修改操作先写 WAL redo 记录、再登记内存 undo 条目（`insert`/`delete`），回滚时逆序执行。`UPDATE = DELETE 旧行 + INSERT 新行`，所以 WAL 只有两种操作码，undo 逻辑统一。

### 2. 堆表：slotted page + 墓碑删除

行被紧凑二进制编码（INT→`<q` 8 字节，REAL→`<d`，TEXT→`<I` 长度 + UTF-8），塞进 4 KB 定长页。槽位 `(offset, raw_len)` 的 raw_len 高位置 1 即墓碑，**保留 offset**——回滚 undo 时原位清除墓碑位即可恢复，无需重排页面。

### 3. WAL：CRC 校验 + 崩溃恢复

记录格式 `[u32 crc32][u16 length][payload]`，payload 含 `txn_id` + 操作码 + 数据体。恢复流程：

1. 顺序扫描日志，CRC 失败或截断即停（`truncated = True`）；
2. 只重放**已提交事务**的 INSERT/DELETE 到堆表；
3. 按页号升序修复堆页链（页号序 == 插入序）；
4. **全量重建所有索引**（磁盘上的索引页可能残留未提交状态）；
5. checkpoint：刷全部脏页 + fsync + 截断 WAL。

### 4. B+ 树

节点以 JSON 序列化进 4 KB 页（可插拔 `NodeStore`：测试和索引基准用纯内存 dict）。键升序插入触发叶节点分裂；删除走墓碑 + 叶内压缩。**刻意不实现跨节点 underflow 合并**（见[已知局限](#已知局限)）。

### 5. 学习式索引（两级 RMI）

Kraska 等人（SIGMOD 2018）主张用"预测位置 + 小范围搜索"的模型替代搜索树。本项目的最小诚实实现：

- **level 1**：等分 `S` 段，二分边界键路由到段；
- **level 2**：段内贪心分段线性拟合，每段预测误差 ≤ 阈值 `E`；
- **最终**：在 `[predicted ± (E+1)]` 窗口内精确二分。

因拟合误差有界，窗口必含真实位置，查询保持精确。**静态只读**——不支持插入（真实场景需要 delta buffer / 重映射）。

---

## 实验与解读

### 实验 1：索引结构对比（`run_benchmarks.py`）

设定：`seed=42`，`N = 100_000` 个有序唯一整数（48 位空间），`Q = 50_000` 次点查（一半命中、一半未命中）。机器为单台 Windows，数值为一次运行结果，**可复现但会有小幅抖动**。

| 变体 | 构建时间 | 查询延迟（ns/次） | 命中 | 未命中 | 内存 |
|---|---|---|---|---|---|
| sorted_array（bisect） | 0.00 s | **1793** | 1855 | 1452 | 0.76 MB |
| btree（order 32，6638 节点，高 4） | 0.21 s | 3485 | 3484 | 2517 | 2.43 MB |
| learned_e16（两级 RMI） | 1.80 s | 3868 | 3750 | 3499 | 0.77 MB（模型仅 10 KB + keys 数组 0.76 MB） |

阈值扫描（延迟 ns / 内存 / 线性段数）：

| 误差界 E | 延迟 | 内存 | 段数 |
|---|---|---|---|
| 4 | 3970 | 0.85 MB | 3538 |
| 16 | **3757** | 0.77 MB | 349 |
| 64 | 4146 | 0.77 MB | 67 |

**解读（诚实负面结论）**：

1. **学习式索引没有赢**。10 万整数上，`bisect` 最快；学习式索引约慢 2.2 倍。原因：二分在 10 万元素上只需 ~7 次比较，而 RMI 的路由 + 分段 + 窗口搜索合计 ~20 次比较并多几层 Python 调用。预测省掉的比较次数，少于它引入的额外开销。
2. **"省内存"也站不住**。模型本身只有 10 KB，但**它无法脱离 keys 数组回答成员查询**——最终精确检查还是要落到数据上。把 keys 数组算进去，学习式索引与 sorted_array 内存几乎相同（0.77 vs 0.76 MB），B+ 树反而贵 3 倍。
3. **构建是有代价的**：贪心分段拟合花 1.8 s（与 bisect 的 0 s 对比），阈值越小段数越爆炸（E=4 时 3538 段）。
4. 学习式索引的价值在**超大 key、cache 敏感、数据分布可被低阶模型刻画**的场景——10 万内存整数恰恰是最不利的场景。**索引结构选型永远是"算法复杂度 × 常数因子 × 内存代价"的权衡，而不是追新。**

### 实验 2：缓冲池替换策略（`bench_buffer.py`）

设定：`seed=42`，Zipf（s=1.2）工作负载，`D = 1_000` 个页，`Q = 100_000` 次访问；容量为页集的 5% / 10% / 20% / 50% / 100%。LRU / Clock / Random 走**真实缓冲池**（真实 pinning 与命中计数）；OPT 是 Belady 离线最优（需要未来，单独模拟）作为理论参考。

| 容量 | 5% | 10% | 20% | 50% | 100% |
|---|---|---|---|---|---|
| LRU | 0.486 | 0.572 | 0.664 | 0.822 | 0.990 |
| CLOCK | 0.482 | 0.568 | 0.662 | 0.820 | 0.990 |
| RANDOM | 0.432 | 0.523 | 0.624 | 0.795 | 0.990 |
| OPT（参考） | 0.658 | 0.737 | 0.817 | 0.926 | 0.990 |

**解读**：

1. **Clock 用一个参考位逼近 LRU**：全程差距 ≤ 0.5 个百分点，这是经典结论——CLOCK 的工程意义在于 O(1) 开销近似 LRU。
2. **Random 是显著下限**：比 LRU 低约 3~5 个百分点（容量越小差距越大，5% 容量时达 5.4pp）。
3. **离线最优与在线策略的差距就是信息差**：OPT 比 LRU 高 15~17 个百分点（5%~20% 小容量时），容量增大后收窄到约 10pp（50% 时）；所有策略在 100% 容量处收敛（只剩每页首次访问的 compulsory miss）。
4. 结论：在偏斜负载下，**容量比策略重要得多**——把缓冲池从 5% 加到 10% 带来的提升（+8.6pp）远大于换掉 LRU（+5.4pp）。

### 实验 3：崩溃恢复演示（`crash_recovery_demo.py`）

提交事务 A（插入 3 行）→ 开启事务 B（再插 2 行）→ `discard()` 模拟进程崩溃（**不 flush、关句柄**）→ 重开数据库。

结果：恢复后磁盘上只剩 A 的 3 行；B 的 2 行既不在堆表也不在任何索引中；主键索引与二级索引与堆表完全一致。**这验证了 no-steal/no-force + WAL redo + 索引全量重建的端到端正确性。**

### 实验 4：数据分布敏感性（`run_benchmarks.py --distribution ...`）

设定：同样的 `seed=42 / N=100_000 / Q=50_000`，但数据集换成三种可复现的 key 分布（`scripts/keygen.py`）：**uniform**（均匀）、**clustered**（低熵聚簇，贴近"频繁访问区间"的真实数据）、**zipf**（偏斜）。核心观测对象：① 查询延迟是否随分布改变；② 模型复杂度（线性段数）如何响应数据熵。

| 分布 | pieces（E=16） | learned 延迟 ns | learned_block 延迟 ns | 预测误差 max | 误差越界率 |
|---|---|---|---|---|---|
| uniform | 349 | 2340 | 3156 | 17 | **0.0** |
| clustered | 438 | 2297 | 3272 | 17 | **0.0** |
| zipf | 638 | 2306 | 3159 | 17 | **0.0** |

**解读（正确性修复 + 诚实负面结论）**：

1. **先修了一个真实 bug，再做实验**。第一版 RMI 只保证"**训练过的 key**"误差 ≤ E，但查询可能落在 gap 里（两段之间、或段内最后一个训练 key 之后）——低熵分布下 gap 极多，线性模型会无界外推（曾测到 `err.max ≈ 1.15e12`、越界率高达 47.8%）。修复：检测到 key 落在 gap 即回退到**段内二分**（段位置范围由 level-1 边界保证）。修复后三种分布 `err.max` 恒为 17（= E+1，即窗口边界）、**越界率恒为 0**，`lookup()` 对任意 key 逐位置精确。
2. **模型复杂度与数据熵**：pieces 数量 uniform 349 < clustered 438 < zipf 638。**zipf 需要最多的线性段**——它头重脚轻，头部一个点一个斜率、尾部一马平川，贪心拟合用大量短段去逼近头部陡坡。这直接打脸"低熵数据=更好拟合=更少模型"的天真预期：**偏斜数据只是把复杂度从"数据本身"转移到"模型里"。**
3. **延迟对分布几乎不敏感**：三种分布下 learned 都在 ~2.3 μs，说明瓶颈在 Python 解释器 + 调用链（路由、选段、窗口搜索），不在比较次数。这也解释了为什么实验 1 里它赢不了 bisect。
4. 诚实结论：**分布敏感性实验的主要收获是正确性 bug 的暴露与修复**（gap 外推），性能结论与实验 1 一致——学习式索引的优势场景不在这台机器的内存整数点查上。

### 实验 5：页路由视角（`run_benchmarks.py --block-sizes ...`）

设定：回到 Kraska 2018 论文的原始主张——模型的任务是**定位页（block）**而非精确位置，页内再用小范围二分。`B` = 页内 key 数（32/64/128/256），`block_extra = (E+B)//B + 1 = 2` 保证候选页窗口必含真页。

| 页大小 B | lookup ns | 预测页误差 max | 页越界率 | 模型+页偏移内存 |
|---|---|---|---|---|
| 32 | 3018 | 1 | **0.0** | 0.80 MB |
| 64 | 3070 | 1 | **0.0** | 0.78 MB |
| 128 | 3248 | 1 | **0.0** | 0.78 MB |
| 256 | 3329 | 1 | **0.0** | 0.78 MB |
| （参考）bisect | 1411 | — | — | 0.76 MB |

**解读**：

1. **页路由正确性成立**：预测页误差最大 1（因为位置误差 E+1=17 对 64-key 页最多跨 1 页），页越界率恒为 0——候选页窗口的选取公式在数学上是紧的。
2. **延迟反而随 B 增大而变慢**（3018 → 3329 ns）。这与论文叙事相反：论文说"块定位省寻道"，但这是**磁盘/外存假设**——页越大、页内二分比较越多，而内存里页定位 + 页内二分的总比较次数（`log2(seg) + log2(2·B·extra+1)` 约为 13~16 次）本来就多于 plain bisect 的 ~16.6 次的理论下限没兑现、Python 常数还更高。
3. 诚实结论：**页路由在"内存、Python、比较次数不是瓶颈"的环境里，恰好是论文优势条件最不成立的场景**。它的价值要等 cache-line 敏感的 C++ 实现 + 磁盘页寻道才显现——这本身就是重要的工程判断：**把论文结论迁移到自己的硬件/语言栈时必须重估假设。**

---

## 复现约定

- 全部随机实验固定 `seed=42`（缓冲池的 Random 策略、Zipf 抽样、索引数据集）。
- 退出码约定：`0` 成功 / `1` 运行错误 / `2` 用法错误（`app.py`）。
- 脚本把结构化结果写入 `results/*.json`，图表输出到 `figures/*.{png,pdf}`，均为可审计的中间产物。

---

## 已知局限

- B+ 树**无跨节点 underflow 合并**：删除只在叶内压缩，空间靠重建索引回收（README 已声明，代码行为与之一致）。
- 键列仅限 INT / TEXT（REAL 拒绝）；每表必须且仅一个主键；二级索引唯一。
- 堆表为追加式，**无 vacuum**；重复删除会让页变稀。
- 无并发控制（单连接串行）；锁粒度到"整库"。
- 索引节点用 JSON 序列化进页（教学清晰优先，未做紧凑二进制）。
- 学习式索引为静态只读；段拟合 `O(段长²)` 最坏复杂度。

---

## Phase 2：LLM 结合

在引擎之上新增一层 **LLM 自然语言接口**（`llm/` 包），三块能力：

1. **NL→SQL**：`NL 查询 users 的所有用户` 自然语言生成 mini_SQL，经 `frontend` 语法护栏验证后执行；提示词注入实时 schema（表/列/类型/索引）降低幻觉。非法 SQL 直接拒绝、不执行。
2. **SQL 辅助调优**：`TUNE [results/index_benchmark.json]` 把实验产物喂给 LLM，生成"为何该索引方案在此数据分布下最优"的解读；Mock 后端退化为数据驱动模板，离线可演示。
3. **数据分布感知的索引选择器**：`ADVISE 大量点查 user id` 把**真实库统计**（表/列/行数/已有索引）与**实验基准实测数字**（查询耗时、内存）注入提示词，让 LLM 基于数据与实测证据给出 `btree / learned / none` 建议，并结合负面结论输出诚实提示（learned 约慢 2.2×）。

**双后端**（`llm/client.py`）：

| 后端 | 触发条件 | 用途 |
| --- | --- | --- |
| Mock（确定性） | 未设置 `MINI_DBMS_API_KEY` | 测试、离线演示，零网络零 key |
| OpenAI 兼容 | 设置 `MINI_DBMS_API_KEY` | 接 DeepSeek / Qwen 等，`base_url`/`model` 可配 |

环境变量：`MINI_DBMS_API_BASE`（默认 `https://api.deepseek.com`）、`MINI_DBMS_API_KEY`、`MINI_DBMS_MODEL`（默认 `deepseek-chat`）。

**边界声明**：LLM 只做模式识别与自然语言接口，**不进入存储正确性路径**——引擎保持确定性、可测试；LLM 输出经 `frontend` 解析器验证后才执行，非法输入拒绝并回显原文。

### Mock 模式快速演示

默认**未设置** `MINI_DBMS_API_KEY` 时走 **Mock 后端**（确定性、零网络零 key），以下会话可直接照做：

```text
mini_db> CREATE TABLE users (id INT PRIMARY KEY, name TEXT, score REAL);
已创建表 users
mini_db> INSERT INTO users VALUES (1, 'alice', 9.5);
已插入 1 行
mini_db> INSERT INTO users VALUES (2, 'bob', 7.0);
已插入 1 行
mini_db> NL 查询 users 的所有用户
id | name  | score
---+-------+------
1  | alice | 9.5
2  | bob   | 7.0
共 2 行
mini_db> ADVISE 大量点查 user id
建议索引类型：btree
理由：点查/范围/排序场景，B+ 树更稳
mini_db> TUNE results/index_benchmark.json
综合 sorted_array 的 lookup_ns=1793.3 最小，是当前实验中最快的查找方案。
内存占用 0.76 MB，构建耗时 0.00s。
解读：索引选型是 算法复杂度 × 常数因子 × 内存代价 的权衡；该结论与 README 实验解读一致。
```

> 说明：Mock 的 `NL` 只识别"表 表名 + 查询/select"模式，生成 `SELECT * FROM <表> LIMIT 10`；`ADVISE` 只按工作负载关键词在 btree / learned / none 间选择（数据感知提示词照常注入）；`TUNE` 退化为数据驱动模板。设置 `MINI_DBMS_API_KEY` 后同一会话即切换到 OpenAI 兼容真实后端（`base_url`/`model` 可用环境变量配置）。

---

## 参考资料

- Kraska, Beutel, Chi, Dean et al. *The Case for Learned Index Structures*, SIGMOD 2018.
- Hellerstein, Stonebraker, Hamilton. *Architecture of a Database System*, Foundations and Trends in Databases, 2007.
- 经典教材：*Database System Concepts*（Silberschatz 等）、*Readings in Database Systems*（"Red Book"）。
- Graefe. *Write-Ahead Logging*（postgres 文档实现说明）与 Belady, L.A. *A Study of Replacement Algorithms for a Virtual-Storage Computer*, 1966.
