"""mini_DBMS 交互式 REPL（命令行前端）。

用法::

    python app.py --data-dir data --pool-size 64 --policy clock

退出码约定:
    0  正常退出（exit / EOF）
    1  启动失败（数据目录损坏、catalog 无法解析等）
    2  命令行参数错误（argparse 默认行为）
"""

from __future__ import annotations

import argparse
import os
import sys

from engine import StorageEngine
from frontend.executor import Executor
from frontend.parser import ParseError, Parser
from frontend.tokenizer import tokenize

from llm.client import LLMError, create_client
from llm import cli as llm_cli

BANNER = r"""
mini_DBMS — 学习型存储引擎（B+树 / WAL 事务 / 缓冲池 / 学习式索引）
输入 SQL 语句，一行一条；exit / quit / \q 退出。
"""


def _render(result) -> None:
    kind = result[0]
    if kind == "msg":
        if result[1]:
            print(result[1])
        return
    if kind == "table":
        _, headers, rows = result
        if not rows:
            print("（无结果）")
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        pad = "  "
        print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
        print(f"共 {len(rows)} 行")


def main(argv=None) -> int:
    argp = argparse.ArgumentParser(
        prog="mini_dbms",
        description="学习型存储引擎 REPL（B+树 / WAL 事务 / 缓冲池 / 学习式索引）",
    )
    argp.add_argument("--data-dir", default=os.path.join(os.getcwd(), "data"),
                      help="数据库数据目录（默认 ./data）")
    argp.add_argument("--pool-size", type=int, default=64,
                      help="缓冲池容量（页数，默认 64）")
    argp.add_argument("--policy", choices=("clock", "lru", "random"), default="clock",
                      help="缓冲池替换策略（默认 clock）")
    argp.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    args = argp.parse_args(argv)

    try:
        engine = StorageEngine(args.data_dir, args.pool_size, args.policy, args.seed)
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    exec_ = Executor(engine)
    client = create_client()
    print(BANNER)
    print(f"数据目录: {os.path.abspath(args.data_dir)}  缓冲池: {args.pool_size} 页  策略: {args.policy}  LLM 后端: {client.name}")
    print("输入 help 查看支持的语句。\n")

    while True:
        try:
            line = input("mini_db> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("exit", "quit", "\\q"):
            break
        if low in ("help", "\\?"):
            print(_HELP)
            continue
        if low.startswith("nl "):
            try:
                result = llm_cli.run_nl(engine, exec_, client, line[3:].strip())
            except (LLMError, ValueError, KeyError) as exc:
                print(f"错误: {exc}")
            else:
                _render(result)
            continue
        if low == "advise" or low.startswith("advise "):
            try:
                print(llm_cli.run_advise(client, line[len("advise"):].strip()))
            except LLMError as exc:
                print(f"错误: {exc}")
            continue
        if low == "tune" or low.startswith("tune "):
            try:
                path = line[len("tune"):].strip() or _latest_results_json()
                print(llm_cli.run_tune(client, path))
            except (LLMError, FileNotFoundError, ValueError) as exc:
                print(f"错误: {exc}")
            continue
        try:
            ast = Parser(tokenize(line)).parse()
            result = exec_.execute(ast)
            _render(result)
        except (ParseError, ValueError, KeyError) as exc:
            print(f"错误: {exc}")
        except Exception as exc:  # engine-level failures should never kill the REPL
            print(f"内部错误: {type(exc).__name__}: {exc}")

    engine.close()
    return 0


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _latest_results_json() -> str:
    """Path of the most recently modified results/*.json."""
    base = os.path.join(_REPO_ROOT, "results")
    if not os.path.isdir(base):
        raise FileNotFoundError("results/ 目录不存在")
    jsons = sorted(
        (os.path.join(base, f) for f in os.listdir(base) if f.endswith(".json")),
        key=os.path.getmtime,
    )
    if not jsons:
        raise FileNotFoundError("results/ 下没有 .json 文件")
    return jsons[-1]


_HELP = """支持的语句：
  CREATE TABLE t (id INT PRIMARY KEY, name TEXT, score REAL)
  CREATE INDEX idx ON t (col)
  DROP TABLE t
  INSERT INTO t VALUES (1, 'alice', 9.5)
  INSERT INTO t (name, id) VALUES ('bob', 2)
  SELECT * FROM t [WHERE col op value] [ORDER BY col ASC|DESC] [LIMIT n]
  UPDATE t SET col = value WHERE col op value
  DELETE FROM t WHERE col op value
  BEGIN / COMMIT / ROLLBACK
  SHOW TABLES / DESCRIBE t
  NL 查询 users 的所有用户           # 自然语言 → SQL → 执行（LLM 后端）
  ADVISE 大量点查 user id            # 数据分布感知的索引选择建议
  TUNE [results/x.json]             # 实验调优解读（默认取 results/ 下最新 json）
op: = != < <= > >=   注释: # 到行尾   字符串: '...' 或 "..."（双引号转义）"""


if __name__ == "__main__":
    sys.exit(main())
