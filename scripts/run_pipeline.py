#!/usr/bin/env python3
"""
Run Pipeline — 一键启动自动迭代因子研发管道

Usage:
    # 默认: 5 轮迭代，optimized 模式
    python scripts/run_pipeline.py

    # 自定义轮次和模式
    python scripts/run_pipeline.py --iterations 30 --profile premium

    # 预算模式（全 DeepSeek Chat）
    python scripts/run_pipeline.py --profile budget --iterations 10

    # 只跑 1 轮测试
    python scripts/run_pipeline.py --iterations 1 --verbose

环境要求:
    - DEEPSEEK_API_KEY 环境变量已设置
    - Qlib US 数据已安装（~/.qlib/qlib_data/us_data）
    - 在项目根目录下运行
"""

import argparse
import logging
import os
import sys
import time

# 确保能 import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_prerequisites():
    """检查运行前提条件。"""
    errors = []

    # 检查 API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        errors.append(
            "DEEPSEEK_API_KEY 未设置。\n"
            "  设置方法: export DEEPSEEK_API_KEY='sk-...'"
        )

    # 检查 Qlib 数据
    from pathlib import Path
    qlib_path = Path.home() / ".qlib" / "qlib_data" / "us_data"
    if not qlib_path.exists():
        errors.append(
            f"Qlib 数据未找到: {qlib_path}\n"
            "  安装方法: python scripts/get_data.py qlib_data "
            "--target_dir ~/.qlib/qlib_data/us_data --region us"
        )

    # 检查配置文件
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "default.yaml"
    )
    if not os.path.exists(config_path):
        errors.append(f"配置文件未找到: {config_path}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="RD-Agent Quant Optimizer — 自动迭代因子研发管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=5,
        help="迭代轮次 (默认: 5)"
    )
    parser.add_argument(
        "--profile", "-p", choices=["budget", "optimized", "premium"],
        default="optimized",
        help="模型预算模式 (默认: optimized)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="配置文件路径 (默认: configs/default.yaml)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="logs",
        help="输出目录 (默认: logs/)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="开启详细日志"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只检查环境，不实际运行"
    )
    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_pipeline")

    # 环境检查
    logger.info("=" * 60)
    logger.info("RD-Agent Quant Optimizer — 环境检查")
    logger.info("=" * 60)
    errors = check_prerequisites()
    if errors:
        logger.error("环境检查失败:")
        for e in errors:
            logger.error(f"  ✗ {e}")
        if not args.dry_run:
            logger.error("请修复以上问题后重新运行。")
            sys.exit(1)
    else:
        logger.info("  ✓ DEEPSEEK_API_KEY 已设置")
        logger.info("  ✓ Qlib 数据就绪")
        logger.info("  ✓ 配置文件就绪")

    if args.dry_run:
        logger.info("Dry run 完成，未发现阻断性问题。" if not errors else "Dry run 完成，有错误需修复。")
        return

    # 确定配置文件
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml"
        )

    # 启动管道
    logger.info("=" * 60)
    logger.info(f"启动管道: {args.iterations} 轮, {args.profile} 模式")
    logger.info("=" * 60)

    from src.pipeline import OptimizedPipeline
    import yaml

    # 加载配置并覆盖 adaptive_mode（必须在 pipeline 初始化前修改，
    # 因为路由表在 __init__ 时根据 adaptive_mode 一次性生成）
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["adaptive_mode"] = args.profile

    start_time = time.time()
    pipeline = OptimizedPipeline(config=config)

    # 运行
    report = pipeline.run(n_iterations=args.iterations)

    elapsed = time.time() - start_time

    # 输出结果
    logger.info("=" * 60)
    logger.info("运行完成!")
    logger.info("=" * 60)
    logger.info(report.summary())
    logger.info(f"总耗时: {elapsed:.1f}s")

    # 导出结果
    os.makedirs(args.output, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{args.profile}_{timestamp}")
    pipeline.export_results(report, output_dir)
    logger.info(f"结果已导出: {output_dir}/")

    # 打印最佳因子
    if report.best_metrics:
        logger.info(f"\n最佳因子 (iteration {report.best_metrics.get('iteration', '?')}):")
        logger.info(f"  IC     = {report.best_metrics.get('IC', 0):.6f}")
        logger.info(f"  ICIR   = {report.best_metrics.get('ICIR', 0):.6f}")
        logger.info(f"  Sharpe = {report.best_metrics.get('sharpe_ratio', 0):.4f}")

    # 返回非零退出码如果全部跳过
    if report.iterations_completed == 0:
        logger.warning("所有迭代均被跳过，请检查 LLM API 和配置。")
        sys.exit(1)


if __name__ == "__main__":
    main()
