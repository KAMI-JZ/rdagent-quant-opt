"""
Self Optimizer — 自我优化引擎

自动分析项目代码和结构，识别:
- 死代码 / 未使用的模块
- 冗余文档 / 过时内容
- 代码质量问题 (过大文件、过深嵌套、重复代码)
- 可优化的 import 和依赖

生成优化建议，但不自动执行（安全第一，需用户确认）。
可选接入 LLM 生成深度分析。

基于用户需求: "自动优化自身代码和结构，自己写判断标准，
留存关键功能，删减长久不用的功能和多余文档。"
"""

import ast
import hashlib
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """问题严重程度"""
    LOW = "low"           # 建议优化
    MEDIUM = "medium"     # 应该处理
    HIGH = "high"         # 需要尽快处理
    CRITICAL = "critical" # 必须立即处理


class IssueCategory(Enum):
    """问题类别"""
    DEAD_CODE = "dead_code"           # 未使用的代码
    REDUNDANT_DOC = "redundant_doc"   # 冗余文档
    CODE_QUALITY = "code_quality"     # 代码质量
    STRUCTURE = "structure"           # 项目结构
    DEPENDENCY = "dependency"         # 依赖问题
    DUPLICATION = "duplication"       # 重复代码


@dataclass
class OptimizationIssue:
    """单个优化问题"""
    category: IssueCategory
    severity: IssueSeverity
    file_path: str
    description: str
    suggestion: str
    line_range: tuple[int, int] | None = None  # (start, end) 行号范围

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "file": self.file_path,
            "description": self.description,
            "suggestion": self.suggestion,
            "lines": self.line_range,
        }


@dataclass
class FileMetrics:
    """单个文件的代码指标"""
    path: str
    lines: int = 0
    functions: int = 0
    classes: int = 0
    imports: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    complexity_score: float = 0.0  # 简化的复杂度评分


@dataclass
class OptimizationReport:
    """优化分析报告"""
    issues: list[OptimizationIssue] = field(default_factory=list)
    file_metrics: list[FileMetrics] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def get_issues_by_severity(self, severity: IssueSeverity) -> list[OptimizationIssue]:
        return [i for i in self.issues if i.severity == severity]

    def get_issues_by_category(self, category: IssueCategory) -> list[OptimizationIssue]:
        return [i for i in self.issues if i.category == category]

    @property
    def critical_count(self) -> int:
        return len(self.get_issues_by_severity(IssueSeverity.CRITICAL))

    @property
    def high_count(self) -> int:
        return len(self.get_issues_by_severity(IssueSeverity.HIGH))

    def to_markdown(self) -> str:
        lines = ["# Self-Optimization Report\n"]

        # 总结
        lines.append(f"**Total issues**: {len(self.issues)}")
        lines.append(f"- Critical: {self.critical_count}")
        lines.append(f"- High: {self.high_count}")
        lines.append(f"- Medium: {len(self.get_issues_by_severity(IssueSeverity.MEDIUM))}")
        lines.append(f"- Low: {len(self.get_issues_by_severity(IssueSeverity.LOW))}")
        lines.append("")

        # 按严重程度分组
        for severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH,
                         IssueSeverity.MEDIUM, IssueSeverity.LOW]:
            issues = self.get_issues_by_severity(severity)
            if not issues:
                continue
            lines.append(f"## {severity.value.upper()} ({len(issues)})\n")
            for issue in issues:
                lines.append(f"- **[{issue.category.value}]** `{issue.file_path}`")
                lines.append(f"  {issue.description}")
                lines.append(f"  → {issue.suggestion}")
                lines.append("")

        return "\n".join(lines)


class CodeAnalyzer:
    """
    分析 Python 代码文件，识别代码质量和结构问题。
    """

    # 影响: 这些阈值定义了什么算"问题"，可调整
    MAX_FILE_LINES = 600        # 超过此行数的文件需要拆分
    MAX_FUNCTION_LINES = 80     # 超过此行数的函数需要重构
    MAX_IMPORTS = 20            # 超过此数量的 import 可能有依赖问题
    MIN_USEFUL_LINES = 5        # 少于此行数的文件可能是空壳

    def analyze_file(self, filepath: str) -> tuple[FileMetrics, list[OptimizationIssue]]:
        """分析单个 Python 文件，返回指标和问题列表。"""
        issues = []
        metrics = FileMetrics(path=filepath)

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                source_lines = content.split("\n")
        except (OSError, IOError) as e:
            issues.append(OptimizationIssue(
                category=IssueCategory.CODE_QUALITY,
                severity=IssueSeverity.HIGH,
                file_path=filepath,
                description=f"无法读取文件: {e}",
                suggestion="检查文件编码或权限。",
            ))
            return metrics, issues

        metrics.lines = len(source_lines)
        metrics.blank_lines = sum(1 for l in source_lines if l.strip() == "")
        metrics.comment_lines = sum(1 for l in source_lines if l.strip().startswith("#"))

        # 检查文件大小
        if metrics.lines > self.MAX_FILE_LINES:
            issues.append(OptimizationIssue(
                category=IssueCategory.CODE_QUALITY,
                severity=IssueSeverity.MEDIUM,
                file_path=filepath,
                description=f"文件过大 ({metrics.lines} 行，阈值 {self.MAX_FILE_LINES})。",
                suggestion="考虑拆分为多个模块。",
            ))

        if metrics.lines < self.MIN_USEFUL_LINES:
            issues.append(OptimizationIssue(
                category=IssueCategory.DEAD_CODE,
                severity=IssueSeverity.LOW,
                file_path=filepath,
                description=f"文件过小 ({metrics.lines} 行)，可能是空壳。",
                suggestion="检查是否有实际功能，如无则考虑删除。",
            ))

        # AST 分析
        try:
            tree = ast.parse(content)
        except SyntaxError:
            issues.append(OptimizationIssue(
                category=IssueCategory.CODE_QUALITY,
                severity=IssueSeverity.HIGH,
                file_path=filepath,
                description="语法错误，无法解析。",
                suggestion="修复语法错误。",
            ))
            return metrics, issues

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.functions += 1
                # 检查函数长度
                if hasattr(node, "end_lineno") and node.end_lineno:
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > self.MAX_FUNCTION_LINES:
                        issues.append(OptimizationIssue(
                            category=IssueCategory.CODE_QUALITY,
                            severity=IssueSeverity.LOW,
                            file_path=filepath,
                            description=f"函数 {node.name}() 过长 ({func_lines} 行)。",
                            suggestion=f"考虑拆分 {node.name}() 为更小的函数。",
                            line_range=(node.lineno, node.end_lineno),
                        ))
            elif isinstance(node, ast.ClassDef):
                metrics.classes += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics.imports += 1

        if metrics.imports > self.MAX_IMPORTS:
            issues.append(OptimizationIssue(
                category=IssueCategory.DEPENDENCY,
                severity=IssueSeverity.LOW,
                file_path=filepath,
                description=f"import 过多 ({metrics.imports}，阈值 {self.MAX_IMPORTS})。",
                suggestion="检查是否有未使用的 import。",
            ))

        return metrics, issues


class ImportAnalyzer:
    """
    分析模块间的 import 关系，找出未被引用的"孤岛"模块。
    """

    def analyze_imports(self, src_dir: str) -> dict[str, set[str]]:
        """
        扫描 src/ 下所有 .py 文件，构建 import 图。
        返回: {module_name: set(imported_by_modules)}
        """
        import_graph: dict[str, set[str]] = defaultdict(set)
        modules: set[str] = set()

        py_files = [f for f in os.listdir(src_dir) if f.endswith(".py") and f != "__init__.py"]

        for filename in py_files:
            module_name = filename[:-3]  # 去掉 .py
            modules.add(module_name)

            filepath = os.path.join(src_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                tree = ast.parse(content)
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # from .xxx import ... 或 from src.xxx import ...
                    imported = node.module.split(".")[-1]
                    if imported in modules or imported + ".py" in [f for f in py_files]:
                        import_graph[imported].add(module_name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name.split(".")[-1]
                        import_graph[imported].add(module_name)

        return dict(import_graph)

    def find_orphan_modules(self, src_dir: str, exclude: set[str] | None = None) -> list[str]:
        """
        找出未被其他模块 import 的孤岛模块。
        exclude: 排除列表（如 pipeline.py 是入口不需要被 import）
        """
        exclude = exclude or {"pipeline", "__init__"}
        import_graph = self.analyze_imports(src_dir)

        py_files = [f[:-3] for f in os.listdir(src_dir)
                     if f.endswith(".py") and f != "__init__.py"]

        orphans = []
        for module in py_files:
            if module in exclude:
                continue
            if module not in import_graph or len(import_graph[module]) == 0:
                orphans.append(module)

        return orphans


class DuplicationDetector:
    """
    简单的代码重复检测器（行级哈希比对）。
    """

    def __init__(self, min_duplicate_lines: int = 5):
        self.min_duplicate_lines = min_duplicate_lines

    def find_duplicates(self, filepaths: list[str]) -> list[OptimizationIssue]:
        """找出跨文件的重复代码块。"""
        # 收集每个文件的行哈希序列
        file_hashes: dict[str, list[tuple[str, int]]] = {}  # filepath -> [(hash, lineno)]

        for fp in filepaths:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except (OSError, IOError):
                continue

            hashes = []
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
                    h = hashlib.md5(stripped.encode()).hexdigest()[:8]
                    hashes.append((h, i))
            file_hashes[fp] = hashes

        # 滑动窗口比对 (简化版: 只检测连续 N 行完全相同)
        issues = []
        seen_blocks: dict[str, tuple[str, int]] = {}  # block_hash -> (filepath, start_line)
        fps = list(file_hashes.keys())

        for fp in fps:
            hashes = file_hashes[fp]
            if len(hashes) < self.min_duplicate_lines:
                continue
            for start in range(len(hashes) - self.min_duplicate_lines + 1):
                block = tuple(h for h, _ in hashes[start:start + self.min_duplicate_lines])
                block_hash = hashlib.md5(str(block).encode()).hexdigest()
                start_line = hashes[start][1]

                if block_hash in seen_blocks:
                    orig_fp, orig_line = seen_blocks[block_hash]
                    if orig_fp != fp:  # 跨文件重复
                        issues.append(OptimizationIssue(
                            category=IssueCategory.DUPLICATION,
                            severity=IssueSeverity.LOW,
                            file_path=fp,
                            description=(
                                f"与 {os.path.basename(orig_fp)}:{orig_line} "
                                f"有 {self.min_duplicate_lines}+ 行重复代码。"
                            ),
                            suggestion="考虑提取为共享工具函数。",
                            line_range=(start_line, start_line + self.min_duplicate_lines),
                        ))
                else:
                    seen_blocks[block_hash] = (fp, start_line)

        return issues


class DocAnalyzer:
    """
    分析文档文件，检测过时或冗余的文档。
    """

    # 影响: 这些文件被认为是核心文档，不建议删除
    CORE_DOCS = {"README.md", "CLAUDE.md", "CHANGELOG.md", "LICENSE", "CONTRIBUTING.md"}

    def analyze_docs(self, project_dir: str) -> list[OptimizationIssue]:
        """扫描项目根目录和 docs/ 下的文档文件。"""
        issues = []

        # 检查根目录 .md 文件
        for f in os.listdir(project_dir):
            if f.endswith(".md") and f not in self.CORE_DOCS:
                filepath = os.path.join(project_dir, f)
                size = os.path.getsize(filepath)
                if size < 100:
                    issues.append(OptimizationIssue(
                        category=IssueCategory.REDUNDANT_DOC,
                        severity=IssueSeverity.LOW,
                        file_path=filepath,
                        description=f"文档过小 ({size} bytes)，可能是空壳或占位符。",
                        suggestion="检查是否有实际内容，如无则删除。",
                    ))

        # 检查 docs/ 目录
        docs_dir = os.path.join(project_dir, "docs")
        if os.path.isdir(docs_dir):
            for f in os.listdir(docs_dir):
                if not f.endswith(".md"):
                    continue
                filepath = os.path.join(docs_dir, f)
                size = os.path.getsize(filepath)
                if size < 100:
                    issues.append(OptimizationIssue(
                        category=IssueCategory.REDUNDANT_DOC,
                        severity=IssueSeverity.LOW,
                        file_path=filepath,
                        description=f"文档过小 ({size} bytes)。",
                        suggestion="检查是否有实际内容。",
                    ))

        return issues


class SelfOptimizer:
    """
    自我优化引擎: 整合所有分析器，生成优化报告。

    用法:
        optimizer = SelfOptimizer("/path/to/project")
        report = optimizer.analyze()
        print(report.to_markdown())
    """

    def __init__(
        self,
        project_dir: str,
        src_subdir: str = "src",
        entry_modules: set[str] | None = None,
    ):
        """
        Args:
            project_dir: 项目根目录
            src_subdir: 源代码子目录名
            entry_modules: 入口模块（不需要被 import 的模块）
        """
        self.project_dir = project_dir
        self.src_dir = os.path.join(project_dir, src_subdir)
        self.entry_modules = entry_modules or {"pipeline", "__init__"}

        self.code_analyzer = CodeAnalyzer()
        self.import_analyzer = ImportAnalyzer()
        self.dup_detector = DuplicationDetector()
        self.doc_analyzer = DocAnalyzer()

    def analyze(self) -> OptimizationReport:
        """运行完整分析，生成报告。"""
        report = OptimizationReport()

        # 1. 代码文件分析
        if os.path.isdir(self.src_dir):
            py_files = [
                os.path.join(self.src_dir, f)
                for f in os.listdir(self.src_dir)
                if f.endswith(".py")
            ]
            for fp in py_files:
                metrics, issues = self.code_analyzer.analyze_file(fp)
                report.file_metrics.append(metrics)
                report.issues.extend(issues)

            # 2. 孤岛模块检测
            orphans = self.import_analyzer.find_orphan_modules(
                self.src_dir, exclude=self.entry_modules,
            )
            for orphan in orphans:
                report.issues.append(OptimizationIssue(
                    category=IssueCategory.DEAD_CODE,
                    severity=IssueSeverity.MEDIUM,
                    file_path=os.path.join(self.src_dir, f"{orphan}.py"),
                    description=f"模块 {orphan} 未被其他模块 import。",
                    suggestion="检查是否仍在使用，如确认无用则可删除。",
                ))

            # 3. 重复代码检测
            dup_issues = self.dup_detector.find_duplicates(py_files)
            report.issues.extend(dup_issues)

        # 4. 文档分析
        doc_issues = self.doc_analyzer.analyze_docs(self.project_dir)
        report.issues.extend(doc_issues)

        # 5. 生成总结
        report.summary = self._build_summary(report)

        logger.info(
            f"[SelfOptimizer] Analysis complete: {len(report.issues)} issues found"
        )

        return report

    def _build_summary(self, report: OptimizationReport) -> dict:
        """构建报告摘要。"""
        total_lines = sum(m.lines for m in report.file_metrics)
        total_functions = sum(m.functions for m in report.file_metrics)
        total_classes = sum(m.classes for m in report.file_metrics)

        category_counts = defaultdict(int)
        for issue in report.issues:
            category_counts[issue.category.value] += 1

        return {
            "total_files": len(report.file_metrics),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_issues": len(report.issues),
            "critical_issues": report.critical_count,
            "high_issues": report.high_count,
            "category_breakdown": dict(category_counts),
        }

    def get_actionable_items(self, report: OptimizationReport | None = None) -> list[str]:
        """获取可执行的优化建议列表（人类可读）。"""
        if report is None:
            report = self.analyze()

        items = []
        # 按优先级排序: critical > high > medium > low
        priority_order = [
            IssueSeverity.CRITICAL, IssueSeverity.HIGH,
            IssueSeverity.MEDIUM, IssueSeverity.LOW,
        ]

        for severity in priority_order:
            for issue in report.get_issues_by_severity(severity):
                items.append(
                    f"[{severity.value.upper()}] {issue.description} → {issue.suggestion}"
                )

        return items
