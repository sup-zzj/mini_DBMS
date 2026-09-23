"""tests/test_advisor.py — 索引选择器：JSON 解析 + 兜底启发式 + 数据感知"""
import json

from engine import INT, StorageEngine
from llm.advisor import advise, build_advise_prompt
from llm.client import MockLLMClient


def test_advise_btree_from_mock():
    client = MockLLMClient()
    adv = advise(client, "大量点查 user id")
    assert adv.index_type == "btree"
    assert not adv.fallback


def test_advise_learned():
    client = MockLLMClient()
    adv = advise(client, "只读批量扫描全表，数据分布简单")
    assert adv.index_type == "learned"
    assert not adv.fallback


def test_advise_write_heavy():
    client = MockLLMClient()
    adv = advise(client, "频繁 insert 和 update")
    assert adv.index_type == "none"


def test_advise_fallback_on_bad_json():
    client = MockLLMClient(rules={"查询": "not json"})
    adv = advise(client, "查询一下")
    assert adv.fallback
    assert adv.index_type == "btree"


def test_build_advise_prompt_contains_stats_and_benchmark():
    system, user = build_advise_prompt(
        "大量点查 user id",
        "表 users（行数≈3）：列 id INT PK；索引：pk_users(id)",
        "sorted_array: lookup 1793.30µs, 内存 0.76MB\n结论：learned 比 sorted_array 慢约 2.2×",
    )
    assert "表 users" in user
    assert "sorted_array" in user
    assert "工作负载：大量点查 user id" in user


def test_advise_data_aware_with_engine(tmp_path):
    engine = StorageEngine(str(tmp_path / "db"))
    engine.create_table("users", [("id", INT, True), ("name", "TEXT", False)])
    engine.create_index("idx_name", "users", "name")
    engine.execute_insert("users", (1, "alice"))
    engine.execute_insert("users", (2, "bob"))
    adv = advise(MockLLMClient(), "大量点查 user id", engine=engine)
    assert adv.index_type in ("btree", "learned", "none")
    engine.close()


def test_mock_advise_ignores_stats_keywords():
    client = MockLLMClient()
    reply = client.complete(
        "[ADVISE]...",
        "当前库统计：\n表 log（行数≈2）：列 update_time INT\n工作负载：点查 user id",
    )
    assert json.loads(reply)["index_type"] == "btree"
