"""Tests for FactorLibrary — 因子库测试"""

import importlib
import sys
import os
import types
import json
import tempfile
import pytest

_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [_src_path]
    sys.modules["src"] = src_pkg

spec = importlib.util.spec_from_file_location("src.factor_library",
                                               os.path.join(_src_path, "factor_library.py"))
_mod = importlib.util.module_from_spec(spec)
sys.modules["src.factor_library"] = _mod
spec.loader.exec_module(_mod)

FactorLibrary = _mod.FactorLibrary
FactorEntry = _mod.FactorEntry

# ──────── Test Data ────────

CODE_A = """
def calculate_factor(df):
    return df['close'].pct_change(20).rolling(5).mean()
"""

CODE_B = """
def calculate_factor(df):
    z = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    return -z
"""

CODE_C = """
def calculate_factor(df):
    return df['volume'] / df['volume'].rolling(20).mean()
"""

METRICS_A = {"IC": 0.05, "sharpe_ratio": 3.0, "annual_return": 0.80}
METRICS_B = {"IC": 0.02, "sharpe_ratio": 1.0, "annual_return": 0.10}
METRICS_C = {"IC": 0.03, "sharpe_ratio": 2.0, "annual_return": 0.40}


class TestFactorLibrary:
    def setup_method(self):
        self.tmp = tempfile.mktemp(suffix=".json")
        self.lib = FactorLibrary(storage_path=self.tmp)

    def teardown_method(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_add_factor(self):
        entry = self.lib.add(CODE_A, METRICS_A, hypothesis="momentum")
        assert entry.id != ""
        assert self.lib.size == 1

    def test_add_multiple(self):
        self.lib.add(CODE_A, METRICS_A)
        self.lib.add(CODE_B, METRICS_B)
        self.lib.add(CODE_C, METRICS_C)
        assert self.lib.size == 3

    def test_version_increment(self):
        e1 = self.lib.add(CODE_A, METRICS_A)
        e2 = self.lib.add(CODE_A, METRICS_A)  # same code
        assert e2.version == 2
        assert self.lib.size == 1  # same ID, updated

    def test_get_factor(self):
        entry = self.lib.add(CODE_A, METRICS_A)
        retrieved = self.lib.get(entry.id)
        assert retrieved is not None
        assert retrieved.factor_code == CODE_A

    def test_remove_factor(self):
        entry = self.lib.add(CODE_A, METRICS_A)
        assert self.lib.remove(entry.id) is True
        assert self.lib.size == 0

    def test_auto_tagging(self):
        entry = self.lib.add(CODE_A, METRICS_A)
        assert "momentum" in entry.tags or "moving_average" in entry.tags

        entry_b = self.lib.add(CODE_B, METRICS_B)
        assert "mean_reversion" in entry_b.tags or "volatility" in entry_b.tags

    def test_get_top_factors_by_sharpe(self):
        self.lib.add(CODE_A, METRICS_A, name="FactorA")
        self.lib.add(CODE_B, METRICS_B, name="FactorB")
        self.lib.add(CODE_C, METRICS_C, name="FactorC")
        top = self.lib.get_top_factors(n=2, sort_by="sharpe_ratio")
        assert len(top) == 2
        assert top[0].backtest_metrics["sharpe_ratio"] >= top[1].backtest_metrics["sharpe_ratio"]

    def test_get_top_factors_by_ic(self):
        self.lib.add(CODE_A, METRICS_A)
        self.lib.add(CODE_B, METRICS_B)
        top = self.lib.get_top_factors(n=1, sort_by="IC")
        assert top[0].backtest_metrics["IC"] == 0.05

    def test_search_by_keyword(self):
        self.lib.add(CODE_A, METRICS_A, hypothesis="20-day momentum factor")
        self.lib.add(CODE_B, METRICS_B, hypothesis="z-score mean reversion")
        results = self.lib.search(keyword="momentum")
        assert len(results) == 1

    def test_search_by_tags(self):
        self.lib.add(CODE_C, METRICS_C)
        results = self.lib.search(tags=["volume"])
        assert len(results) >= 1

    def test_search_by_min_sharpe(self):
        self.lib.add(CODE_A, METRICS_A)
        self.lib.add(CODE_B, METRICS_B)
        results = self.lib.search(min_sharpe=2.0)
        assert len(results) == 1

    def test_search_by_grade(self):
        self.lib.add(CODE_A, METRICS_A, review_grade="A")
        self.lib.add(CODE_B, METRICS_B, review_grade="C")
        results = self.lib.search(min_grade="B")
        assert len(results) == 1

    def test_find_similar(self):
        self.lib.add(CODE_A, METRICS_A)
        # Same code → similarity = 1.0
        similar = self.lib.find_similar(CODE_A, threshold=0.9)
        assert len(similar) >= 1
        assert similar[0][1] >= 0.9

    def test_find_similar_different(self):
        self.lib.add(CODE_A, METRICS_A)
        similar = self.lib.find_similar(CODE_B, threshold=0.99)
        assert len(similar) == 0  # very different code

    def test_persistence(self):
        self.lib.add(CODE_A, METRICS_A, name="Persistent")
        # Create new library from same file
        lib2 = FactorLibrary(storage_path=self.tmp)
        assert lib2.size == 1
        assert lib2.get_all()[0].name == "Persistent"

    def test_to_markdown_table(self):
        self.lib.add(CODE_A, METRICS_A, name="MomentumA", review_grade="A")
        md = self.lib.to_markdown_table()
        assert "MomentumA" in md
        assert "Rank" in md

    def test_entry_to_dict(self):
        entry = self.lib.add(CODE_A, METRICS_A, name="Test")
        d = entry.to_dict()
        assert d["name"] == "Test"
        assert d["backtest_metrics"]["IC"] == 0.05

    def test_entry_from_dict(self):
        d = {"id": "abc", "name": "FromDict", "version": 1, "factor_code": CODE_A,
             "hypothesis": "test", "tags": ["momentum"], "backtest_metrics": METRICS_A,
             "review_grade": "B", "review_score": 65.0, "created_at": 0, "iteration": 1,
             "run_id": "r1", "metadata": {}}
        entry = FactorEntry.from_dict(d)
        assert entry.name == "FromDict"
        assert entry.review_grade == "B"

    def test_empty_library(self):
        assert self.lib.size == 0
        assert self.lib.get_top_factors() == []
        assert self.lib.search() == []
