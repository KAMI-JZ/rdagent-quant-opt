"""
Tests for Self Optimizer — 自我优化引擎
"""

import sys
import os
import importlib.util
import tempfile
import shutil

# 绕过 src/__init__.py 的重依赖链
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if "src" not in sys.modules:
    sys.modules["src"] = type(sys)("src")
    sys.modules["src"].__path__ = [_src_dir]
    sys.modules["src"].__package__ = "src"

spec = importlib.util.spec_from_file_location(
    "src.self_optimizer", os.path.join(_src_dir, "self_optimizer.py")
)
so_mod = importlib.util.module_from_spec(spec)
sys.modules["src.self_optimizer"] = so_mod
spec.loader.exec_module(so_mod)

SelfOptimizer = so_mod.SelfOptimizer
CodeAnalyzer = so_mod.CodeAnalyzer
ImportAnalyzer = so_mod.ImportAnalyzer
DuplicationDetector = so_mod.DuplicationDetector
DocAnalyzer = so_mod.DocAnalyzer
OptimizationReport = so_mod.OptimizationReport
OptimizationIssue = so_mod.OptimizationIssue
IssueSeverity = so_mod.IssueSeverity
IssueCategory = so_mod.IssueCategory
FileMetrics = so_mod.FileMetrics

import pytest


# ──────────────────────────────────────────────
# Helper: 创建临时项目目录
# ──────────────────────────────────────────────

@pytest.fixture
def temp_project():
    """创建临时项目目录用于测试"""
    tmpdir = tempfile.mkdtemp()
    src_dir = os.path.join(tmpdir, "src")
    docs_dir = os.path.join(tmpdir, "docs")
    os.makedirs(src_dir)
    os.makedirs(docs_dir)

    # __init__.py
    with open(os.path.join(src_dir, "__init__.py"), "w") as f:
        f.write('"""Package init."""\nfrom .module_a import func_a\n')

    # module_a.py — 正常模块
    with open(os.path.join(src_dir, "module_a.py"), "w") as f:
        f.write(
            '"""Module A."""\nimport os\nimport sys\n\n'
            'def func_a():\n    """Do something."""\n    return 42\n\n'
            'def func_b():\n    return 100\n'
        )

    # module_b.py — 引用 module_a
    with open(os.path.join(src_dir, "module_b.py"), "w") as f:
        f.write(
            '"""Module B."""\nfrom .module_a import func_a\n\n'
            'def use_a():\n    return func_a() + 1\n'
        )

    # module_orphan.py — 孤岛模块 (未被任何人 import)
    with open(os.path.join(src_dir, "module_orphan.py"), "w") as f:
        f.write(
            '"""Orphan module."""\n\ndef orphan_func():\n    return "lonely"\n'
        )

    # module_empty.py — 几乎空的文件
    with open(os.path.join(src_dir, "module_empty.py"), "w") as f:
        f.write('"""Empty."""\n')

    # README.md — 核心文档
    with open(os.path.join(tmpdir, "README.md"), "w") as f:
        f.write("# Project\nSome content here.\n" * 10)

    # extra.md — 非核心文档
    with open(os.path.join(tmpdir, "extra.md"), "w") as f:
        f.write("x\n")  # 极小文档

    # docs/GUIDE.md — 有内容的文档
    with open(os.path.join(docs_dir, "GUIDE.md"), "w") as f:
        f.write("# Guide\n" + "content\n" * 20)

    # docs/TINY.md — 极小文档
    with open(os.path.join(docs_dir, "TINY.md"), "w") as f:
        f.write("x\n")

    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────
# CodeAnalyzer Tests
# ──────────────────────────────────────────────

class TestCodeAnalyzer:
    def test_normal_file(self, temp_project):
        analyzer = CodeAnalyzer()
        fp = os.path.join(temp_project, "src", "module_a.py")
        metrics, issues = analyzer.analyze_file(fp)
        assert metrics.lines > 0
        assert metrics.functions == 2
        assert metrics.imports == 2

    def test_empty_file(self, temp_project):
        analyzer = CodeAnalyzer()
        fp = os.path.join(temp_project, "src", "module_empty.py")
        metrics, issues = analyzer.analyze_file(fp)
        assert metrics.lines < 5
        # 应有"文件过小"的 issue
        assert any(i.category == IssueCategory.DEAD_CODE for i in issues)

    def test_large_file_threshold(self):
        """超过阈值的文件触发警告"""
        analyzer = CodeAnalyzer()
        analyzer.MAX_FILE_LINES = 10  # 设低阈值方便测试

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("# lines\n" * 20)
            fp = f.name

        try:
            metrics, issues = analyzer.analyze_file(fp)
            assert any(i.category == IssueCategory.CODE_QUALITY
                       and "过大" in i.description for i in issues)
        finally:
            os.unlink(fp)

    def test_syntax_error_file(self):
        """语法错误文件"""
        analyzer = CodeAnalyzer()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def broken(\n")
            fp = f.name

        try:
            metrics, issues = analyzer.analyze_file(fp)
            assert any("语法错误" in i.description for i in issues)
        finally:
            os.unlink(fp)

    def test_many_imports(self):
        """import 过多警告"""
        analyzer = CodeAnalyzer()
        analyzer.MAX_IMPORTS = 3  # 设低阈值

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import os\nimport sys\nimport json\nimport re\nimport math\n")
            fp = f.name

        try:
            metrics, issues = analyzer.analyze_file(fp)
            assert metrics.imports == 5
            assert any(i.category == IssueCategory.DEPENDENCY for i in issues)
        finally:
            os.unlink(fp)


# ──────────────────────────────────────────────
# ImportAnalyzer Tests
# ──────────────────────────────────────────────

class TestImportAnalyzer:
    def test_find_orphans(self, temp_project):
        analyzer = ImportAnalyzer()
        orphans = analyzer.find_orphan_modules(
            os.path.join(temp_project, "src"),
            exclude={"__init__"},
        )
        assert "module_orphan" in orphans
        assert "module_empty" in orphans  # 也没被 import

    def test_non_orphan_not_listed(self, temp_project):
        analyzer = ImportAnalyzer()
        orphans = analyzer.find_orphan_modules(
            os.path.join(temp_project, "src"),
            exclude={"__init__"},
        )
        assert "module_a" not in orphans  # module_b imports module_a

    def test_import_graph(self, temp_project):
        analyzer = ImportAnalyzer()
        graph = analyzer.analyze_imports(os.path.join(temp_project, "src"))
        assert "module_a" in graph
        assert "module_b" in graph["module_a"]


# ──────────────────────────────────────────────
# DuplicationDetector Tests
# ──────────────────────────────────────────────

class TestDuplicationDetector:
    def test_no_duplicates(self, temp_project):
        detector = DuplicationDetector(min_duplicate_lines=3)
        filepaths = [
            os.path.join(temp_project, "src", "module_a.py"),
            os.path.join(temp_project, "src", "module_b.py"),
        ]
        issues = detector.find_duplicates(filepaths)
        # 这两个文件不应有重复
        assert len(issues) == 0

    def test_detect_duplicates(self):
        """跨文件重复代码"""
        detector = DuplicationDetector(min_duplicate_lines=3)
        tmpdir = tempfile.mkdtemp()

        shared_code = "x = 1\ny = 2\nz = x + y\nresult = z * 2\nprint(result)\n"

        fp1 = os.path.join(tmpdir, "file1.py")
        fp2 = os.path.join(tmpdir, "file2.py")
        with open(fp1, "w") as f:
            f.write(f"# File 1\n{shared_code}")
        with open(fp2, "w") as f:
            f.write(f"# File 2\n{shared_code}")

        try:
            issues = detector.find_duplicates([fp1, fp2])
            assert len(issues) > 0
            assert issues[0].category == IssueCategory.DUPLICATION
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────
# DocAnalyzer Tests
# ──────────────────────────────────────────────

class TestDocAnalyzer:
    def test_detect_tiny_docs(self, temp_project):
        analyzer = DocAnalyzer()
        issues = analyzer.analyze_docs(temp_project)
        # extra.md 和 docs/TINY.md 应被标记
        tiny_files = [i.file_path for i in issues]
        assert any("extra.md" in f for f in tiny_files)
        assert any("TINY.md" in f for f in tiny_files)

    def test_core_docs_not_flagged(self, temp_project):
        analyzer = DocAnalyzer()
        issues = analyzer.analyze_docs(temp_project)
        # README.md 不应被标记（是核心文档且有内容）
        assert not any("README.md" in i.file_path for i in issues)


# ──────────────────────────────────────────────
# SelfOptimizer Integration Tests
# ──────────────────────────────────────────────

class TestSelfOptimizer:
    def test_full_analysis(self, temp_project):
        """完整分析流程"""
        optimizer = SelfOptimizer(temp_project)
        report = optimizer.analyze()
        assert isinstance(report, OptimizationReport)
        assert len(report.file_metrics) > 0
        assert len(report.issues) > 0
        assert report.summary["total_files"] > 0

    def test_orphan_detected(self, temp_project):
        """检测到孤岛模块"""
        optimizer = SelfOptimizer(temp_project)
        report = optimizer.analyze()
        orphan_issues = report.get_issues_by_category(IssueCategory.DEAD_CODE)
        orphan_files = [i.file_path for i in orphan_issues]
        assert any("module_orphan" in f for f in orphan_files)

    def test_markdown_output(self, temp_project):
        """Markdown 报告生成"""
        optimizer = SelfOptimizer(temp_project)
        report = optimizer.analyze()
        md = report.to_markdown()
        assert "# Self-Optimization Report" in md
        assert "Total issues" in md

    def test_actionable_items(self, temp_project):
        """可执行建议列表"""
        optimizer = SelfOptimizer(temp_project)
        items = optimizer.get_actionable_items()
        assert isinstance(items, list)
        assert len(items) > 0
        # 每项应有 severity 标签
        assert all(any(s in item for s in ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
                   for item in items)

    def test_summary_stats(self, temp_project):
        """摘要统计正确"""
        optimizer = SelfOptimizer(temp_project)
        report = optimizer.analyze()
        assert report.summary["total_functions"] >= 3  # func_a, func_b, use_a, orphan_func
        assert report.summary["total_issues"] == len(report.issues)


# ──────────────────────────────────────────────
# OptimizationReport Tests
# ──────────────────────────────────────────────

class TestOptimizationReport:
    def test_filter_by_severity(self):
        report = OptimizationReport(issues=[
            OptimizationIssue(IssueCategory.DEAD_CODE, IssueSeverity.HIGH, "a.py", "d", "s"),
            OptimizationIssue(IssueCategory.CODE_QUALITY, IssueSeverity.LOW, "b.py", "d", "s"),
        ])
        assert len(report.get_issues_by_severity(IssueSeverity.HIGH)) == 1
        assert len(report.get_issues_by_severity(IssueSeverity.LOW)) == 1
        assert len(report.get_issues_by_severity(IssueSeverity.CRITICAL)) == 0

    def test_filter_by_category(self):
        report = OptimizationReport(issues=[
            OptimizationIssue(IssueCategory.DEAD_CODE, IssueSeverity.HIGH, "a.py", "d", "s"),
            OptimizationIssue(IssueCategory.DEAD_CODE, IssueSeverity.LOW, "b.py", "d", "s"),
            OptimizationIssue(IssueCategory.STRUCTURE, IssueSeverity.MEDIUM, "c.py", "d", "s"),
        ])
        assert len(report.get_issues_by_category(IssueCategory.DEAD_CODE)) == 2

    def test_issue_to_dict(self):
        issue = OptimizationIssue(
            IssueCategory.DUPLICATION, IssueSeverity.LOW,
            "test.py", "desc", "sugg", (10, 20),
        )
        d = issue.to_dict()
        assert d["category"] == "duplication"
        assert d["lines"] == (10, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
