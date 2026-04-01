"""
Unit tests for the optimized pipeline.
测试完整管道组装。
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import (
    OptimizedPipeline, IterationResult, PipelineReport,
)


# ──────── IterationResult Tests ────────

class TestIterationResult:

    def test_default_not_skipped(self):
        """默认结果不是跳过状态。"""
        r = IterationResult(iteration=0)
        assert not r.skipped
        assert r.skip_reason == ""

    def test_skipped_result(self):
        """可以标记为跳过并附带原因。"""
        r = IterationResult(iteration=1, skipped=True, skip_reason="Too similar")
        assert r.skipped
        assert "similar" in r.skip_reason.lower()

    def test_verdict_without_debate(self):
        """没有辩论结果时 verdict 应为 N/A。"""
        r = IterationResult(iteration=0)
        assert r.verdict == "N/A"

    def test_passed_filter_without_result(self):
        """没有过滤结果时 passed_filter 应为 False。"""
        r = IterationResult(iteration=0)
        assert not r.passed_filter


# ──────── PipelineReport Tests ────────

class TestPipelineReport:

    def test_empty_report(self):
        """空报告的初始值应该正确。"""
        report = PipelineReport()
        assert report.iterations_completed == 0
        assert report.iterations_skipped == 0
        assert report.total_cost_usd == 0.0
        assert len(report.results) == 0

    def test_summary_format(self):
        """summary() 应返回可读的文本。"""
        report = PipelineReport(
            iterations_completed=25,
            iterations_skipped=5,
            total_cost_usd=1.85,
            total_duration_sec=3600,
        )
        text = report.summary()
        assert "25" in text
        assert "5" in text
        assert "1.85" in text

    def test_verdicts_tracking(self):
        """裁决统计应正确初始化。"""
        report = PipelineReport()
        assert report.verdicts["continue"] == 0
        assert report.verdicts["pivot"] == 0
        assert report.verdicts["neutral"] == 0


# ──────── Pipeline Initialization Tests ────────

class TestPipelineInit:

    def test_init_with_default_config(self):
        """用默认配置文件初始化应该成功。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        pipeline = OptimizedPipeline(config_path=config_path)
        assert pipeline.router is not None
        assert pipeline.alpha_filter is not None
        assert pipeline.debate_analyzer is not None
        assert pipeline.regime_detector is not None

    def test_init_with_missing_config(self):
        """配置文件不存在时应使用默认值，不崩溃。"""
        pipeline = OptimizedPipeline(config_path="nonexistent.yaml")
        assert pipeline.router is not None

    def test_accepted_factors_empty_initially(self):
        """初始因子库应为空。"""
        pipeline = OptimizedPipeline(config_path="nonexistent.yaml")
        assert len(pipeline.accepted_factors) == 0

    def test_cost_tracker_initialized(self):
        """成本追踪器应该被初始化。"""
        pipeline = OptimizedPipeline(config_path="nonexistent.yaml")
        assert pipeline.cost_tracker.get_today_cost() == 0.0


# ──────── Config Loading Tests ────────

class TestConfigLoading:

    def test_loads_yaml(self):
        """应能加载 YAML 配置。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        pipeline = OptimizedPipeline(config_path=config_path)
        assert pipeline.config.get("adaptive_mode") is not None

    def test_missing_file_returns_empty_dict(self):
        """文件不存在时返回空字典。"""
        pipeline = OptimizedPipeline(config_path="nonexistent.yaml")
        assert isinstance(pipeline.config, dict)


# ──────── Code Extraction Tests ────────

class TestExtractPythonCode:

    def test_extracts_from_markdown_block(self):
        """应从 ```python ... ``` 块中提取代码。"""
        llm_response = '''Here's the implementation:

```python
import numpy as np
import pandas as pd

def calculate_factor(df):
    return df['close'].pct_change(20)
```

This factor computes 20-day momentum.'''
        code = OptimizedPipeline._extract_python_code(llm_response)
        assert "def calculate_factor(df):" in code
        assert "```" not in code
        assert "Here's the implementation" not in code

    def test_extracts_from_generic_code_block(self):
        """应从 ``` ... ``` 块（无 python 标签）中提取代码。"""
        llm_response = '''```
def calculate_factor(df):
    return df['close'].rolling(10).mean()
```'''
        code = OptimizedPipeline._extract_python_code(llm_response)
        assert "def calculate_factor(df):" in code
        assert "```" not in code

    def test_returns_raw_if_no_block(self):
        """无代码块时返回原始文本（如果包含 def）。"""
        raw = "import pandas as pd\n\ndef calculate_factor(df):\n    return df['close']"
        code = OptimizedPipeline._extract_python_code(raw)
        assert "def calculate_factor(df):" in code

    def test_picks_longest_block(self):
        """多个代码块时选最长的。"""
        llm_response = '''```python
x = 1
```

```python
import numpy as np
import pandas as pd

def calculate_factor(df):
    vol = df['volume'].rolling(20).mean()
    return vol / df['close']
```'''
        code = OptimizedPipeline._extract_python_code(llm_response)
        assert "def calculate_factor(df):" in code
        assert "rolling(20)" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
