"""
Report Generator — 运行报告生成器

Generates structured Markdown reports after pipeline runs.
Zero LLM cost — pure data aggregation and analysis.

在每次管道运行结束后生成结构化 Markdown 报告：
- 运行总览（成本、耗时、通过率）
- 每轮迭代的关键指标表格
- IC/ICIR 趋势分析
- 辩论裁决分布
- 过滤器拦截分析
- 成本分解
- 最佳/最差迭代对比
- 可操作的调参建议
"""

import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Report generation settings. 报告生成配置"""
    output_dir: str = "logs"
    include_code: bool = False       # 是否在报告中包含因子代码（可能很长）
    include_hypothesis: bool = True  # 是否包含因子假设文本
    max_hypothesis_chars: int = 200  # 假设文本截断长度
    language: str = "bilingual"      # "en" / "zh" / "bilingual"


class ReportGenerator:
    """
    Generates Markdown reports from pipeline run results.
    从管道运行结果生成 Markdown 报告。

    Usage:
        from src.report_generator import ReportGenerator
        gen = ReportGenerator()
        report_path = gen.generate(pipeline_report, cost_summary)
        print(f"Report saved to {report_path}")
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    def generate(self, report, cost_summary: Optional[dict] = None,
                 run_label: str = "") -> str:
        """
        Generate a full Markdown report.
        生成完整的 Markdown 报告。

        Args:
            report: PipelineReport object from pipeline.run()
            cost_summary: Optional dict from CostTracker.get_summary()
            run_label: Optional label like "budget_30iter"

        Returns:
            str: path to the generated report file
        """
        if not run_label:
            run_label = datetime.now().strftime("%Y%m%d_%H%M%S")

        sections = [
            self._header(report, run_label),
            self._overview(report),
            self._iteration_table(report),
            self._metric_trends(report),
            self._verdict_analysis(report),
            self._filter_analysis(report),
            self._cost_breakdown(report, cost_summary),
            self._best_worst(report),
            self._recommendations(report),
            self._footer(),
        ]

        content = "\n\n".join(s for s in sections if s)

        # 保存报告
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"report_{run_label}.md"
        filepath = os.path.join(self.config.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # 同时保存结构化 JSON（方便程序化对比）
        json_path = os.path.join(self.config.output_dir, f"report_{run_label}.json")
        self._save_json(report, cost_summary, run_label, json_path)

        logger.info(f"[Report] Generated: {filepath}")
        return filepath

    # ────────────── Section Generators ──────────────

    def _header(self, report, run_label: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"# Pipeline Run Report / 管道运行报告\n\n"
            f"- **Run Label / 运行标签**: `{run_label}`\n"
            f"- **Generated / 生成时间**: {now}\n"
            f"- **Iterations / 迭代轮次**: {len(report.results)}"
        )

    def _overview(self, report) -> str:
        """运行总览"""
        total = len(report.results)
        completed = report.iterations_completed
        skipped = report.iterations_skipped
        pass_rate = (completed / total * 100) if total > 0 else 0

        minutes = report.total_duration_sec / 60
        avg_sec = report.total_duration_sec / total if total > 0 else 0

        lines = [
            "## Overview / 总览\n",
            "| Metric / 指标 | Value / 值 |",
            "|---|---|",
            f"| Completed / 完成 | {completed} / {total} ({pass_rate:.0f}%) |",
            f"| Skipped / 跳过 | {skipped} |",
            f"| Total Cost / 总成本 | ${report.total_cost_usd:.4f} |",
            f"| Duration / 总耗时 | {minutes:.1f} min |",
            f"| Avg per Iteration / 平均每轮 | {avg_sec:.1f}s |",
        ]

        if report.best_metrics:
            best_ic = report.best_metrics.get("IC", "N/A")
            best_iter = report.best_metrics.get("iteration", "?")
            lines.append(
                f"| Best IC / 最佳IC | {best_ic} (iter {best_iter}) |"
            )

        return "\n".join(lines)

    def _iteration_table(self, report) -> str:
        """每轮迭代的指标表格"""
        lines = [
            "## Iteration Details / 迭代明细\n",
            "| # | Status | IC | ICIR | Sharpe | Verdict | Cost | Time |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for r in report.results:
            if r.skipped:
                status = "SKIP"
                ic = icir = sharpe = "-"
            else:
                status = "OK"
                ic = f"{r.backtest_metrics.get('IC', 0):.4f}"
                icir = f"{r.backtest_metrics.get('ICIR', 0):.4f}"
                sharpe = f"{r.backtest_metrics.get('sharpe_ratio', 0):.2f}"

            verdict = r.verdict if not r.skipped else r.skip_reason[:30]
            lines.append(
                f"| {r.iteration} | {status} | {ic} | {icir} | {sharpe} "
                f"| {verdict} | ${r.cost_usd:.4f} | {r.duration_sec:.1f}s |"
            )

        return "\n".join(lines)

    def _metric_trends(self, report) -> str:
        """IC/ICIR 趋势（文本 sparkline）"""
        completed = [r for r in report.results if not r.skipped]
        if len(completed) < 2:
            return ""

        ics = [r.backtest_metrics.get("IC", 0) for r in completed]
        icirs = [r.backtest_metrics.get("ICIR", 0) for r in completed]

        lines = [
            "## Metric Trends / 指标趋势\n",
            f"**IC range**: {min(ics):.4f} ~ {max(ics):.4f}  "
            f"(mean={sum(ics)/len(ics):.4f})",
            f"**ICIR range**: {min(icirs):.4f} ~ {max(icirs):.4f}  "
            f"(mean={sum(icirs)/len(icirs):.4f})",
            "",
            "IC trend (each char = 1 iteration):",
            f"```",
            self._sparkline(ics),
            f"```",
        ]

        # 趋势判断
        if len(ics) >= 4:
            first_half = sum(ics[:len(ics)//2]) / (len(ics)//2)
            second_half = sum(ics[len(ics)//2:]) / (len(ics) - len(ics)//2)
            if second_half > first_half * 1.1:
                lines.append("Trend: **Improving** / 趋势: 上升 ↑")
            elif second_half < first_half * 0.9:
                lines.append("Trend: **Declining** / 趋势: 下降 ↓")
            else:
                lines.append("Trend: **Stable** / 趋势: 平稳 →")

        return "\n".join(lines)

    def _verdict_analysis(self, report) -> str:
        """辩论裁决分布"""
        verdicts = report.verdicts
        total_v = sum(verdicts.values())
        if total_v == 0:
            return ""

        lines = [
            "## Debate Verdicts / 辩论裁决\n",
            "| Verdict / 裁决 | Count / 次数 | Ratio / 占比 |",
            "|---|---|---|",
        ]

        for v in ["continue", "pivot", "neutral"]:
            count = verdicts.get(v, 0)
            pct = count / total_v * 100 if total_v > 0 else 0
            bar = "█" * int(pct / 5)  # 每5%一格
            label_zh = {"continue": "继续", "pivot": "转向", "neutral": "中立"}
            lines.append(f"| {v.upper()} / {label_zh.get(v, v)} | {count} | {pct:.0f}% {bar} |")

        # 诊断
        lines.append("")
        pivot_pct = verdicts.get("pivot", 0) / total_v * 100 if total_v > 0 else 0
        if pivot_pct > 60:
            lines.append(
                "> ⚠ PIVOT rate > 60%: the system is frequently changing direction. "
                "Consider stronger synthesis models or more focused prompts.\n"
                "> PIVOT 率 > 60%：系统频繁换方向，建议用更强的构思模型或更聚焦的 prompt。"
            )
        elif pivot_pct < 20 and total_v >= 5:
            lines.append(
                "> ⚠ PIVOT rate < 20%: the system rarely changes direction. "
                "This could indicate confirmation bias.\n"
                "> PIVOT 率 < 20%：系统很少换方向，可能存在确认偏差。"
            )

        return "\n".join(lines)

    def _filter_analysis(self, report) -> str:
        """过滤器拦截分析"""
        total = len(report.results)
        filtered = [r for r in report.results if r.skipped and r.filter_result is not None
                     and not r.filter_result.passed]
        other_skip = [r for r in report.results if r.skipped and r not in filtered]

        if total == 0:
            return ""

        filter_rate = len(filtered) / total * 100

        lines = [
            "## Filter Analysis / 过滤器分析\n",
            f"- **Filtered out / 被拦截**: {len(filtered)} / {total} ({filter_rate:.0f}%)",
            f"- **Other skips / 其他跳过**: {len(other_skip)}",
        ]

        # 拦截原因统计
        if filtered:
            reasons: dict[str, int] = {}
            for r in filtered:
                for reason in r.filter_result.rejection_reasons:
                    bucket = self._categorize_reason(reason)
                    reasons[bucket] = reasons.get(bucket, 0) + 1

            lines.append("\n**Rejection reasons / 拦截原因:**\n")
            lines.append("| Reason / 原因 | Count / 次数 |")
            lines.append("|---|---|")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {count} |")

        # 建议
        lines.append("")
        if filter_rate > 50:
            lines.append(
                "> ⚠ Filter rate > 50%: too many factors rejected. "
                "Consider lowering `similarity_threshold` (current: 0.85) "
                "or increasing `max_complexity_depth` (current: 5).\n"
                "> 过滤率 > 50%：太多因子被拒。建议降低相似度阈值或提高复杂度限制。"
            )
        elif filter_rate < 5 and total >= 10:
            lines.append(
                "> ℹ Filter rate < 5%: very few factors rejected. "
                "The filter may be too lenient — consider raising `similarity_threshold`.\n"
                "> 过滤率 < 5%：很少因子被拒，过滤器可能太宽松。"
            )

        return "\n".join(lines)

    def _cost_breakdown(self, report, cost_summary: Optional[dict]) -> str:
        """成本分解"""
        if not cost_summary:
            return (
                "## Cost Breakdown / 成本分解\n\n"
                f"Total: ${report.total_cost_usd:.4f}\n\n"
                "_Detailed breakdown not available (no CostTracker summary provided)._"
            )

        lines = [
            "## Cost Breakdown / 成本分解\n",
            f"**Total / 总计**: ${cost_summary.get('total_cost_usd', 0):.4f}",
            f"**Calls / LLM调用次数**: {cost_summary.get('total_calls', 0)}",
            f"**Budget remaining / 剩余预算**: "
            f"${cost_summary.get('budget_remaining_usd', 0):.4f}",
            "",
            "| Stage / 阶段 | Calls / 调用 | Cost / 成本 | Share / 占比 |",
            "|---|---|---|---|",
        ]

        breakdown = cost_summary.get("stage_breakdown", {})
        distribution = cost_summary.get("cost_distribution", {})

        for stage, data in sorted(breakdown.items()):
            calls = data.get("calls", 0)
            cost = data.get("cost_usd", 0)
            share = distribution.get(stage, "0%")
            stage_zh = {
                "synthesis": "因子构思", "implementation": "代码生成",
                "analysis": "结果分析", "specification": "模板组装",
            }
            label = f"{stage} / {stage_zh.get(stage, stage)}"
            lines.append(f"| {label} | {calls} | ${cost:.4f} | {share} |")

        # 成本效率指标
        total_cost = cost_summary.get("total_cost_usd", 0)
        completed = report.iterations_completed
        if completed > 0 and total_cost > 0:
            cost_per_iter = total_cost / completed
            lines.append(f"\n**Cost per completed iteration / 每完成一轮的成本**: "
                         f"${cost_per_iter:.4f}")

        return "\n".join(lines)

    def _best_worst(self, report) -> str:
        """最佳/最差迭代对比"""
        completed = [r for r in report.results if not r.skipped]
        if len(completed) < 2:
            return ""

        by_ic = sorted(completed, key=lambda r: r.backtest_metrics.get("IC", 0))
        worst = by_ic[0]
        best = by_ic[-1]

        lines = [
            "## Best vs Worst / 最佳 vs 最差\n",
            "| | Best / 最佳 | Worst / 最差 |",
            "|---|---|---|",
            f"| Iteration / 轮次 | {best.iteration} | {worst.iteration} |",
            f"| IC | {best.backtest_metrics.get('IC', 0):.4f} "
            f"| {worst.backtest_metrics.get('IC', 0):.4f} |",
            f"| ICIR | {best.backtest_metrics.get('ICIR', 0):.4f} "
            f"| {worst.backtest_metrics.get('ICIR', 0):.4f} |",
            f"| Sharpe | {best.backtest_metrics.get('sharpe_ratio', 0):.2f} "
            f"| {worst.backtest_metrics.get('sharpe_ratio', 0):.2f} |",
            f"| Verdict / 裁决 | {best.verdict} | {worst.verdict} |",
        ]

        if self.config.include_hypothesis:
            best_h = (best.hypothesis[:self.config.max_hypothesis_chars] + "..."
                      if len(best.hypothesis) > self.config.max_hypothesis_chars
                      else best.hypothesis)
            worst_h = (worst.hypothesis[:self.config.max_hypothesis_chars] + "..."
                       if len(worst.hypothesis) > self.config.max_hypothesis_chars
                       else worst.hypothesis)
            lines.append(f"| Hypothesis / 假设 | {best_h} | {worst_h} |")

        return "\n".join(lines)

    def _recommendations(self, report) -> str:
        """可操作的调参建议"""
        completed = [r for r in report.results if not r.skipped]
        total = len(report.results)
        recs = []

        # 1. 通过率检查
        pass_rate = len(completed) / total * 100 if total > 0 else 0
        if pass_rate < 50:
            recs.append(
                "- **Low pass rate / 通过率低** ({:.0f}%): "
                "Lower `alpha_filter.similarity_threshold` from 0.85 to 0.75, "
                "or increase `alpha_filter.max_complexity_depth` from 5 to 7.\n"
                "  通过率低于50%，建议降低相似度阈值或提高复杂度限制。".format(pass_rate)
            )

        # 2. IC 趋势检查
        if len(completed) >= 6:
            ics = [r.backtest_metrics.get("IC", 0) for r in completed]
            first_half = sum(ics[:len(ics)//2]) / (len(ics)//2)
            second_half = sum(ics[len(ics)//2:]) / (len(ics) - len(ics)//2)
            if second_half < first_half * 0.8:
                recs.append(
                    "- **IC declining / IC 下降**: Later iterations are worse. "
                    "Consider switching to `premium` mode for synthesis, "
                    "or enable trajectory evolution for cross-iteration learning.\n"
                    "  后半段 IC 明显下降，建议升级构思模型或启用轨迹进化。"
                )

        # 3. 辩论分布检查
        verdicts = report.verdicts
        total_v = sum(verdicts.values())
        if total_v > 0:
            pivot_rate = verdicts.get("pivot", 0) / total_v
            if pivot_rate > 0.7:
                recs.append(
                    "- **Excessive pivots / 过度转向** ({:.0f}%): "
                    "System can't find a good direction. "
                    "Try using experience memory or feeding more market context.\n"
                    "  PIVOT 率过高，系统找不到好方向，建议启用经验记忆。".format(pivot_rate * 100)
                )

        # 4. 成本效率检查
        if report.total_cost_usd > 0 and len(completed) > 0:
            cost_per = report.total_cost_usd / len(completed)
            if cost_per > 0.5:
                recs.append(
                    f"- **High cost per iteration / 每轮成本高** (${cost_per:.2f}): "
                    "Consider switching to `budget` or `optimized` mode.\n"
                    "  每轮成本较高，建议切换到更经济的模式。"
                )

        # 5. 模拟数据警告
        simulated = any(
            r.backtest_metrics.get("_simulated", False)
            for r in completed
        )
        if simulated:
            recs.append(
                "- **Simulated data / 模拟数据**: Backtest metrics are placeholders, "
                "not real Qlib results. Integrate with RD-Agent for meaningful analysis.\n"
                "  当前回测指标是占位数据，需要接入 RD-Agent/Qlib 才能获得真实结果。"
            )

        if not recs:
            return (
                "## Recommendations / 调参建议\n\n"
                "No issues detected. The run looks healthy.\n"
                "没有发现问题，运行状态良好。"
            )

        return "## Recommendations / 调参建议\n\n" + "\n".join(recs)

    def _footer(self) -> str:
        return (
            "---\n"
            "*Generated by RD-Agent Quant Optimizer Report Generator. "
            "Zero LLM cost — pure data analysis.*\n"
            "*由 RD-Agent Quant Optimizer 报告生成器自动生成，纯数据分析，无 LLM 成本。*"
        )

    # ────────────── Helpers ──────────────

    @staticmethod
    def _sparkline(values: list[float]) -> str:
        """Generate a text-based sparkline from values. 文本迷你图"""
        if not values:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        lo, hi = min(values), max(values)
        spread = hi - lo if hi != lo else 1
        return "".join(
            blocks[min(int((v - lo) / spread * 8), 8)]
            for v in values
        )

    @staticmethod
    def _categorize_reason(reason: str) -> str:
        """Categorize a rejection reason into a bucket. 将拦截原因归类"""
        reason_lower = reason.lower()
        if "similar" in reason_lower:
            return "Too similar / 过于相似"
        if "complex" in reason_lower or "depth" in reason_lower:
            return "Too complex / 过于复杂"
        if "align" in reason_lower:
            return "Misaligned / 假设不匹配"
        return "Other / 其他"

    def _save_json(self, report, cost_summary, run_label, filepath):
        """Save structured JSON for programmatic comparison. 保存结构化 JSON"""
        completed = [r for r in report.results if not r.skipped]
        ics = [r.backtest_metrics.get("IC", 0) for r in completed]

        data = {
            "run_label": run_label,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "iterations_total": len(report.results),
                "iterations_completed": report.iterations_completed,
                "iterations_skipped": report.iterations_skipped,
                "total_cost_usd": report.total_cost_usd,
                "total_duration_sec": report.total_duration_sec,
                "verdicts": report.verdicts,
                "best_metrics": report.best_metrics,
            },
            "metrics": {
                "ic_mean": sum(ics) / len(ics) if ics else 0,
                "ic_min": min(ics) if ics else 0,
                "ic_max": max(ics) if ics else 0,
                "ic_values": ics,
            },
            "cost_summary": cost_summary,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "skipped": r.skipped,
                    "ic": r.backtest_metrics.get("IC", 0) if not r.skipped else None,
                    "icir": r.backtest_metrics.get("ICIR", 0) if not r.skipped else None,
                    "sharpe": r.backtest_metrics.get("sharpe_ratio", 0) if not r.skipped else None,
                    "verdict": r.verdict,
                    "cost_usd": r.cost_usd,
                    "duration_sec": r.duration_sec,
                }
                for r in report.results
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
