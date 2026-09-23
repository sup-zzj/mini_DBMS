"""tests/test_tuning.py — TUNE：读 results/*.json 生成解读"""
import json

import pytest

from llm.client import MockLLMClient
from llm.tuning import tune


def _write_bench(tmp_path):
    data = {
        "meta": {"n_keys": 100000},
        "variants": [
            {"name": "sorted_array", "lookup_ns": 1793.3, "memory_mb": 0.763},
            {"name": "btree", "lookup_ns": 3484.6, "memory_mb": 2.434},
            {"name": "learned_e16", "lookup_ns": 3867.8, "memory_mb": 0.772},
        ],
    }
    p = tmp_path / "index_benchmark.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_tune_mock_template(tmp_path):
    path = _write_bench(tmp_path)
    report = tune(MockLLMClient(), path)
    assert "sorted_array" in report
    assert "lookup_ns" in report


def test_tune_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        tune(MockLLMClient(), str(tmp_path / "nope.json"))


def test_tune_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        tune(MockLLMClient(), str(p))
