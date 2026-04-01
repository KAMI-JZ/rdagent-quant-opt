"""
Anti-Alpha-Decay Filter for RD-Agent.

Prevents factor crowding and overfitting via three mechanisms:
1. AST similarity: rejects factors too similar to existing library
2. Complexity constraint: limits formula nesting depth
3. Hypothesis-factor alignment: verifies code implements stated hypothesis

Based on AlphaAgent (arXiv:2502.16789, KDD 2025).
"""

import ast
import logging
from dataclasses import dataclass
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    passed: bool
    factor_code: str
    similarity_score: float = 0.0
    complexity_depth: int = 0
    alignment_score: float = 0.0
    rejection_reasons: list[str] = None

    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []


class ASTSimilarityChecker:
    """
    Compares factor AST structures to detect redundant factors.
    Includes hyperbolic decay: older factors have relaxed similarity thresholds.
    Based on Lee (2025, arXiv:2512.11913): alpha(t) = K / (1 + λ*t).

    双曲衰减: 越老的因子，相似度阈值越宽松。
    原因: 老因子的预测力在衰减，和它们类似的新因子可能有更新的信息。
    """

    def __init__(self, similarity_threshold: float = 0.85, decay_lambda: float = 0.1):
        self.threshold = similarity_threshold
        self.decay_lambda = decay_lambda  # 衰减速率: 越大衰减越快
        self.factor_library: list[tuple[str, ast.AST, int]] = []  # (name, tree, age)
        self._current_iteration: int = 0  # 当前迭代数，用于计算因子年龄

    def _normalize_tree(self, tree: ast.AST) -> list[str]:
        """Extract structural tokens from AST, ignoring variable names."""
        tokens = []
        for node in ast.walk(tree):
            tokens.append(type(node).__name__)
            if isinstance(node, ast.BinOp):
                tokens.append(type(node.op).__name__)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                tokens.append(node.func.attr)
        return tokens

    def _jaccard_similarity(self, a: list[str], b: list[str]) -> float:
        set_a, set_b = set(a), set(b)
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _hyperbolic_decay(self, age: int) -> float:
        """
        Compute decay-adjusted threshold multiplier.
        双曲衰减: 因子越老，相似度阈值越高（越容易放行）。
        公式: multiplier = 1 + decay_lambda * age
        实际阈值 = base_threshold * multiplier（被 cap 在 1.0）
        """
        return 1.0 + self.decay_lambda * age

    def set_iteration(self, iteration: int):
        """Set current iteration for age calculation. 设置当前迭代数。"""
        self._current_iteration = iteration

    def add_factor(self, name: str, code: str, iteration: Optional[int] = None):
        """Register a factor in the library. 注册因子到库中。"""
        registered_at = iteration if iteration is not None else self._current_iteration
        try:
            tree = ast.parse(code)
            self.factor_library.append((name, tree, registered_at))
        except SyntaxError:
            logger.warning(f"Cannot parse factor '{name}', skipping library registration")

    def check(self, code: str) -> tuple[float, Optional[str]]:
        """
        Check similarity against library with hyperbolic decay.
        带双曲衰减的相似度检查。

        老因子的有效阈值更高 → 即使相似度高也更容易放行。
        Returns (max_effective_similarity, matched_name).
        """
        try:
            new_tree = ast.parse(code)
        except SyntaxError:
            return 0.0, None

        new_tokens = self._normalize_tree(new_tree)
        max_sim, match_name = 0.0, None

        for name, lib_tree, registered_at in self.factor_library:
            lib_tokens = self._normalize_tree(lib_tree)
            raw_sim = self._jaccard_similarity(new_tokens, lib_tokens)
            # 影响: 老因子的有效相似度被降低（除以衰减系数）
            # 例: 10 轮前注册的因子，原始相似度 0.90 → 有效相似度 0.90/2.0 = 0.45
            age = max(0, self._current_iteration - registered_at)
            decay_divisor = self._hyperbolic_decay(age)
            effective_sim = raw_sim / decay_divisor
            if effective_sim > max_sim:
                max_sim, match_name = effective_sim, name

        return max_sim, match_name


class ComplexityChecker:
    """Limits formula nesting depth to prevent overfitting."""

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def _get_max_depth(self, node: ast.AST, current: int = 0) -> int:
        depth = current
        if isinstance(node, (ast.BinOp, ast.Call, ast.Subscript, ast.IfExp)):
            current += 1
            depth = max(depth, current)
        for child in ast.iter_child_nodes(node):
            depth = max(depth, self._get_max_depth(child, current))
        return depth

    def check(self, code: str) -> int:
        """Returns max nesting depth. -1 if code can't be parsed."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return -1
        return self._get_max_depth(tree)


class AlignmentChecker:
    """Verifies generated code aligns with stated hypothesis via LLM."""

    def __init__(self, model: str = "deepseek/deepseek-chat"):
        self.model = model

    def check(self, hypothesis: str, code: str) -> tuple[float, str]:
        """Returns (alignment_score 0-1, explanation)."""
        prompt = (
            "You are a quantitative finance expert. Score how well this code "
            "implements the stated hypothesis on a scale of 0.0 to 1.0.\n\n"
            f"Hypothesis: {hypothesis}\n\nCode:\n```python\n{code}\n```\n\n"
            "Respond in exactly this format:\nSCORE: <float>\nREASON: <one line>"
        )
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150, temperature=0.1,
                timeout=30,  # 影响: 防止 API 卡死挂起整个流水线
            )
            text = response.choices[0].message.content.strip()
            # 影响: 安全解析，LLM 回复格式不对时不会崩溃
            score_lines = [l for l in text.split("\n") if l.startswith("SCORE:")]
            if not score_lines:
                logger.warning(f"Alignment response missing SCORE line: {text[:100]}")
                return 0.5, "LLM response format unexpected"
            try:
                score = float(score_lines[0].split(":")[1].strip())
            except (ValueError, IndexError):
                logger.warning(f"Cannot parse score from: {score_lines[0]}")
                return 0.5, "Score parse error"
            reason = text.split("REASON:")[-1].strip() if "REASON:" in text else ""
            return min(max(score, 0.0), 1.0), reason
        except Exception as e:
            logger.error(f"Alignment check failed: {e}")
            return 0.5, f"Check failed: {e}"


class AlphaDecayFilter:
    """
    Composite filter gate between Synthesis and Implementation.

    Rejects factors that are:
    - Too similar to existing library (AST similarity > threshold)
    - Too complex (nesting depth > max_depth)
    - Misaligned with hypothesis (alignment score < min_alignment)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_complexity_depth: int = 5,
        min_alignment_score: float = 0.6,
        alignment_model: str = "deepseek/deepseek-chat",
    ):
        self.ast_checker = ASTSimilarityChecker(similarity_threshold)
        self.complexity_checker = ComplexityChecker(max_complexity_depth)
        self.alignment_checker = AlignmentChecker(alignment_model)
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_complexity_depth
        self.min_alignment = min_alignment_score

    def add_existing_factor(self, name: str, code: str, iteration: Optional[int] = None):
        self.ast_checker.add_factor(name, code, iteration)

    def set_iteration(self, iteration: int):
        """Set current iteration for decay calculation. 设置当前迭代数。"""
        self.ast_checker.set_iteration(iteration)

    def evaluate(
        self, code: str, hypothesis: str = "", check_alignment: bool = True
    ) -> FilterResult:
        """Run all filter checks. Returns FilterResult with pass/fail and details."""
        reasons = []

        # 1. AST similarity
        sim_score, match_name = self.ast_checker.check(code)
        if sim_score >= self.similarity_threshold:
            reasons.append(
                f"Too similar to '{match_name}' (similarity={sim_score:.2f})"
            )

        # 2. Complexity
        depth = self.complexity_checker.check(code)
        if depth > self.max_depth:
            reasons.append(f"Too complex (depth={depth}, max={self.max_depth})")

        # 3. Alignment (optional, requires LLM call)
        alignment = 1.0
        if check_alignment and hypothesis:
            alignment, reason = self.alignment_checker.check(hypothesis, code)
            if alignment < self.min_alignment:
                reasons.append(f"Misaligned with hypothesis (score={alignment:.2f}): {reason}")

        return FilterResult(
            passed=len(reasons) == 0,
            factor_code=code,
            similarity_score=sim_score,
            complexity_depth=depth,
            alignment_score=alignment,
            rejection_reasons=reasons,
        )
