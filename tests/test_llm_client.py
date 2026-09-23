"""tests/test_llm_client.py — LLM 客户端双后端（Mock 确定性 + OpenAI 兼容）"""
import json
import urllib.request

import pytest

from llm.client import LLMError, MockLLMClient, OpenAICompatClient, create_client


def test_mock_rules_are_deterministic():
    client = MockLLMClient(rules={"查询": "SELECT * FROM users LIMIT 10"})
    assert client.complete("sys", "帮我查询用户") == "SELECT * FROM users LIMIT 10"
    assert client.complete("sys", "帮我查询用户") == "SELECT * FROM users LIMIT 10"


def test_mock_nl2sql_select_from_schema():
    client = MockLLMClient()
    reply = client.complete("[NL2SQL]...", "表 users: 列 id INT 主键\n查询 id>10 的用户")
    assert reply == "SELECT * FROM users LIMIT 10"


def test_mock_advise_btree():
    client = MockLLMClient()
    reply = client.complete("[ADVISE]...", "大量点查 user id")
    assert json.loads(reply)["index_type"] == "btree"


def test_mock_tune_returns_empty():
    client = MockLLMClient()
    assert client.complete("[TUNE]...", "variant btree: lookup_ns=1") == ""


def test_create_client_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("MINI_DBMS_API_KEY", raising=False)
    assert isinstance(create_client(), MockLLMClient)


def test_create_client_with_key_uses_openai(monkeypatch):
    monkeypatch.setenv("MINI_DBMS_API_KEY", "sk-test")
    client = create_client()
    assert isinstance(client, OpenAICompatClient)
    assert client.model == "deepseek-chat"


def test_openai_compat_request_and_parse(monkeypatch):
    captured = {}

    class FakeResp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps(self._body).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = req.headers
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return FakeResp({"choices": [{"message": {"content": "  SELECT 1  "}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatClient("sk-test", base_url="https://example.com", model="m1")
    assert client.complete("sys", "usr") == "SELECT 1"
    assert captured["url"] == "https://example.com/chat/completions"
    assert captured["method"] == "POST"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["data"]["model"] == "m1"
    assert captured["data"]["messages"][1]["content"] == "usr"


def test_openai_compat_network_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    client = OpenAICompatClient("k", base_url="https://e.com", model="m")
    with pytest.raises(LLMError):
        client.complete("s", "u")


def test_openai_compat_malformed_response(monkeypatch):
    def fake_urlopen(req, timeout=None):
        class R:
            def read(self):
                return b'{"unexpected": true}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatClient("k", base_url="https://e.com", model="m")
    with pytest.raises(LLMError):
        client.complete("s", "u")
