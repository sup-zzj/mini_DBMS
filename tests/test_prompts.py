"""tests/test_prompts.py — schema 序列化、NL→SQL 提示词、响应解析"""
import pytest

from engine import INT, StorageEngine
from llm.client import LLMError
from llm.prompts import build_nl_to_sql_prompt, build_schema_text, parse_sql_response


def _engine(tmp_path):
    return StorageEngine(str(tmp_path / "db"))


def test_build_schema_text_empty(tmp_path):
    engine = _engine(tmp_path)
    assert "空库" in build_schema_text(engine.catalog)
    engine.close()


def test_build_schema_text_lists_tables(tmp_path):
    engine = _engine(tmp_path)
    engine.create_table("users", [("id", INT, True), ("name", "TEXT", False)])
    text = build_schema_text(engine.catalog)
    assert "表 users" in text
    assert "id INT 主键" in text
    assert "name TEXT" in text
    engine.close()


def test_build_nl_to_sql_prompt_contains_schema_and_nl(tmp_path):
    engine = _engine(tmp_path)
    engine.create_table("users", [("id", INT, True)])
    system, user = build_nl_to_sql_prompt(build_schema_text(engine.catalog), "查 id=1")
    assert "[NL2SQL]" in system
    assert "表 users" in user
    assert "查 id=1" in user
    engine.close()


def test_parse_sql_response_fence():
    assert parse_sql_response("```sql\nSELECT * FROM t;\n```") == "SELECT * FROM t"


def test_parse_sql_response_plain():
    assert parse_sql_response("SELECT 1;") == "SELECT 1"


def test_parse_sql_response_empty():
    with pytest.raises(LLMError):
        parse_sql_response("   ")
