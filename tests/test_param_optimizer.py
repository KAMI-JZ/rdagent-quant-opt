"""
Unit tests for the parameter optimizer module.
测试参数自动优化器。
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.param_optimizer import (
    ParameterExtractor, ParameterOptimizer, ParamSpec, OptimizationResult,
)


# ──────── ParameterExtractor Tests ────────

class TestParameterExtractor:

    def test_extract_pct_change(self):
        """应识别 pct_change(N) 中的参数。"""
        ext = ParameterExtractor()
        params = ext.extract("df['close'].pct_change(20)")
        assert len(params) == 1
        assert params[0].original_value == 20
        assert params[0].min_value == 1
        assert params[0].max_value == 60

    def test_extract_rolling(self):
        """应识别 rolling(N) 中的参数。"""
        ext = ParameterExtractor()
        params = ext.extract("df['close'].rolling(30).mean()")
        assert len(params) == 1
        assert params[0].original_value == 30

    def test_extract_multiple_params(self):
        """应识别多个参数。"""
        ext = ParameterExtractor()
        code = "df['close'].pct_change(5) / df['close'].rolling(10).std()"
        params = ext.extract(code)
        assert len(params) == 2
        values = {p.original_value for p in params}
        assert 5 in values
        assert 10 in values

    def test_extract_shift(self):
        """应识别 shift(N) 中的参数。"""
        ext = ParameterExtractor()
        params = ext.extract("df['volume'].shift(3)")
        assert len(params) == 1
        assert params[0].original_value == 3

    def test_extract_no_params(self):
        """没有可调参数时返回空列表。"""
        ext = ParameterExtractor()
        params = ext.extract("x = 1 + 2")
        assert params == []

    def test_extract_syntax_error(self):
        """语法错误时返回空列表。"""
        ext = ParameterExtractor()
        params = ext.extract("def broken(:")
        assert params == []

    def test_integer_flag(self):
        """整数参数应标记为 is_integer=True。"""
        ext = ParameterExtractor()
        params = ext.extract("df['close'].rolling(20).mean()")
        assert params[0].is_integer is True


# ──────── ParameterOptimizer Tests ────────

class TestParameterOptimizer:

    def test_no_params_returns_original(self):
        """没有参数时应返回原始代码。"""
        opt = ParameterOptimizer()
        result = opt.optimize("x = 1", [])
        assert result.optimized_code == "x = 1"
        assert result.search_iterations == 0

    def test_substitute_single_param(self):
        """单参数替换应正确。"""
        code = "df['close'].pct_change(20)"
        params = [ParamSpec(name="pct_change_0", original_value=20,
                            min_value=1, max_value=60)]
        result = ParameterOptimizer.substitute_params(code, params, [10])
        assert "pct_change(10)" in result

    def test_substitute_multiple_params(self):
        """多参数替换应各自正确。"""
        code = "df['close'].pct_change(5) / df['close'].rolling(10).std()"
        params = [
            ParamSpec(name="pct_change_0", original_value=5, min_value=1, max_value=30),
            ParamSpec(name="rolling_1", original_value=10, min_value=2, max_value=60),
        ]
        result = ParameterOptimizer.substitute_params(code, params, [3, 20])
        assert "pct_change(3)" in result
        assert "rolling(20)" in result

    def test_optimize_with_custom_objective(self):
        """自定义目标函数应被调用。"""
        call_count = [0]

        def fake_objective(code: str) -> float:
            call_count[0] += 1
            # 偏好较大的 pct_change 参数
            if "pct_change(30)" in code:
                return 0.1
            return 0.0

        opt = ParameterOptimizer(objective_fn=fake_objective, max_evals=20)
        params = [ParamSpec(name="pct_change_0", original_value=5,
                            min_value=5, max_value=30, step=5)]
        result = opt.optimize("df['close'].pct_change(5)", params)
        assert call_count[0] > 0
        assert result.optimized_score >= result.original_score

    def test_optimization_result_fields(self):
        """结果应包含所有必要字段。"""
        opt = ParameterOptimizer(max_evals=5)
        params = [ParamSpec(name="rolling_0", original_value=10,
                            min_value=5, max_value=15, step=5)]
        result = opt.optimize("df['close'].rolling(10).mean()", params)
        assert isinstance(result, OptimizationResult)
        assert result.original_code != ""
        assert isinstance(result.original_params, dict)
        assert isinstance(result.optimized_params, dict)
        assert result.search_iterations > 0

    def test_grid_size_limited(self):
        """搜索网格应被 max_evals 限制。"""
        opt = ParameterOptimizer(max_evals=10)
        params = [
            ParamSpec(name="p1", original_value=5, min_value=1, max_value=100),
            ParamSpec(name="p2", original_value=10, min_value=1, max_value=100),
        ]
        grid = opt._generate_grid(params)
        assert len(grid) <= 10

    def test_objective_exception_handled(self):
        """目标函数抛异常时不应崩溃。"""
        def bad_objective(code: str) -> float:
            raise ValueError("Backtest failed")

        opt = ParameterOptimizer(objective_fn=bad_objective, max_evals=5)
        params = [ParamSpec(name="rolling_0", original_value=10,
                            min_value=5, max_value=15, step=5)]
        # 不应抛异常
        result = opt.optimize("df['close'].rolling(10).mean()", params)
        assert isinstance(result, OptimizationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
