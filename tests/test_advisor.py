"""tests/test_advisor.py — 索引选择器：JSON 解析 + 兜底启发式"""
from llm.advisor import advise
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
