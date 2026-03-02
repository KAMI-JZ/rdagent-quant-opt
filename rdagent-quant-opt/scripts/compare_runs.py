"""
Compare backtest results across different routing profiles.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --runs-dir logs/
"""

import os
import json
import glob
import argparse
from pathlib import Path


def load_run(run_dir: str) -> dict:
    """Load run configuration and results."""
    result = {"dir": run_dir}
    
    # Load config
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            result["config"] = json.load(f)
    
    # Load cost log
    cost_files = glob.glob(os.path.join(run_dir, "cost_log*.json"))
    if cost_files:
        with open(cost_files[0]) as f:
            result["cost"] = json.load(f)
    
    # Load RD-Agent results (if available)
    qlib_results = glob.glob(os.path.join(run_dir, "**/*result*.json"), recursive=True)
    if qlib_results:
        with open(qlib_results[0]) as f:
            result["backtest"] = json.load(f)
    
    return result


def compare_runs(runs_dir: str = "logs/"):
    """Compare all runs in the logs directory."""
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
    
    if not run_dirs:
        print("No runs found. Execute a backtest first:")
        print("  bash scripts/run_backtest.sh budget 30")
        return
    
    runs = [load_run(d) for d in run_dirs]
    
    # Print comparison table
    print(f"\n{'═'*80}")
    print(f"  BACKTEST COMPARISON ({len(runs)} runs)")
    print(f"{'═'*80}")
    print(f"{'Run':<35} {'Profile':<12} {'Cost':>8} {'Calls':>6}")
    print(f"{'─'*80}")
    
    for run in runs:
        name = os.path.basename(run["dir"])
        profile = run.get("config", {}).get("profile", "?")
        cost = run.get("cost", {}).get("summary", {}).get("total_cost_usd", "N/A")
        calls = run.get("cost", {}).get("summary", {}).get("total_calls", "N/A")
        
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else str(cost)
        print(f"  {name:<33} {profile:<12} {cost_str:>8} {str(calls):>6}")
    
    print(f"{'═'*80}")
    
    # Cost distribution comparison
    print(f"\n{'═'*80}")
    print(f"  COST DISTRIBUTION BY STAGE")
    print(f"{'═'*80}")
    
    for run in runs:
        name = os.path.basename(run["dir"])
        profile = run.get("config", {}).get("profile", "?")
        breakdown = run.get("cost", {}).get("summary", {}).get("stage_breakdown", {})
        
        if breakdown:
            total = sum(s["cost_usd"] for s in breakdown.values())
            print(f"\n  {name} ({profile}):")
            for stage, info in sorted(breakdown.items()):
                pct = (info["cost_usd"] / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                print(f"    {stage:20s} {bar} {pct:5.1f}% (${info['cost_usd']:.4f})")
    
    print(f"\n{'═'*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare backtest runs")
    parser.add_argument("--runs-dir", default="logs/", help="Directory containing run logs")
    args = parser.parse_args()
    compare_runs(args.runs_dir)
