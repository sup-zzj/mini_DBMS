"""tests/test_nl_to_sql.py — NL→SQL 命令编排与语法护栏"""
import pytest

from engine import INT, StorageEngine
from frontend.executor import Executor
from llm.client import LLMError, MockLLMClient
from llm.cli import run_advise, run_nl, run_tune


def _engine(tmp_path):
    return StorageEngine(str(tmp_path / "db"))


def test_run_nl_executes_valid_sql(tmp_path):
    engine = _engine(tmp_path)
    engine.create_table("users", [("id", INT, True), ("name", "TEXT", False)])
    engine.execute_insert("users", (1, "alice"))
    exec_ = Executor(engine)
    client = MockLLMClient()
    kind, headers, rows = run_nl(engine, exec_, client, "查询 users 的所有用户")
    assert kind == "table"
    assert rows == [[1, "alice"]]
    engine.close()


def test_run_nl_rejects_bad_sql(tmp_path):
    engine = _engine(tmp_path)
    exec_ = Executor(engine)
    client = MockLLMClient(rules={"查询": "DROP DATABASE x"})
    with pytest.raises(LLMError):
        run_nl(engine, exec_, client, "查询一下")
    engine.close()


def test_run_nl_rejects_empty_reply(tmp_path):
    engine = _engine(tmp_path)
    exec_ = Executor(engine)
    client = MockLLMClient(rules={"查询": ""})
    with pytest.raises(LLMError):
        run_nl(engine, exec_, client, "查询一下")
    engine.close()


def test_run_advise_text(tmp_path):
    engine = _engine(tmp_path)
    text = run_advise(MockLLMClient(), "大量点查 user id")
    assert "btree" in text
    engine.close()


def test_run_tune_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_tune(MockLLMClient(), str(tmp_path / "nope.json"))
