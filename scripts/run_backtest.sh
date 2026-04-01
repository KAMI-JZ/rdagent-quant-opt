#!/bin/bash
# ═══════════════════════════════════════════════════
# RD-Agent Quant Optimizer - Backtest Runner
# ═══════════════════════════════════════════════════
set -e

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default profile
PROFILE="${1:-budget}"
ITERATIONS="${2:-30}"
SCENARIO="${3:-fin_factor}"

echo "╔══════════════════════════════════════════════╗"
echo "║  RD-Agent Quant Optimizer - Backtest         ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Profile:    $PROFILE"
echo "║  Iterations: $ITERATIONS"
echo "║  Scenario:   $SCENARIO"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ──────── Profile Configuration ────────
case "$PROFILE" in
    budget)
        echo "💰 Budget Profile: All DeepSeek Chat (adaptive=budget)"
        echo "   Estimated cost: ~\$0.5-1.5 for $ITERATIONS iterations"
        export CHAT_MODEL="deepseek/deepseek-chat"
        export ADAPTIVE_MODE="budget"
        export SYNTHESIS_MODEL="deepseek/deepseek-chat"
        export IMPLEMENTATION_MODEL="deepseek/deepseek-chat"
        export ANALYSIS_MODEL="deepseek/deepseek-chat"
        ;;
    optimized)
        echo "⚡ Optimized Profile: Reasoner for thinking, Chat for coding (adaptive=optimized)"
        echo "   Estimated cost: ~\$1.5-3.0 for $ITERATIONS iterations"
        export CHAT_MODEL="deepseek/deepseek-chat"
        export ADAPTIVE_MODE="optimized"
        export SYNTHESIS_MODEL="deepseek/deepseek-reasoner"
        export IMPLEMENTATION_MODEL="deepseek/deepseek-chat"
        export ANALYSIS_MODEL="deepseek/deepseek-reasoner"
        ;;
    premium)
        echo "👑 Premium Profile: Opus 4.6 for creativity, Reasoner for analysis, Chat for coding"
        echo "   Estimated cost: ~\$15-30 for $ITERATIONS iterations"
        export CHAT_MODEL="deepseek/deepseek-chat"
        export ADAPTIVE_MODE="premium"
        export SYNTHESIS_MODEL="anthropic/claude-opus-4-6"
        export IMPLEMENTATION_MODEL="deepseek/deepseek-chat"
        export ANALYSIS_MODEL="deepseek/deepseek-reasoner"
        ;;
    baseline)
        echo "📊 Baseline Profile: Single model (for comparison)"
        echo "   Uses CHAT_MODEL from .env for everything"
        ;;
    *)
        echo "❌ Unknown profile: $PROFILE"
        echo "   Available: budget, optimized, premium, baseline"
        exit 1
        ;;
esac

echo ""

# ──────── Pre-flight Checks ────────
echo "🔍 Pre-flight checks..."

# Check API key
if [ -z "$DEEPSEEK_API_KEY" ] && [ "$PROFILE" != "premium" ]; then
    echo "❌ DEEPSEEK_API_KEY not set. Edit .env first."
    exit 1
fi
echo "  ✅ API key configured"

# Check Docker
if command -v docker &> /dev/null && docker info &> /dev/null; then
    echo "  ✅ Docker running"
else
    echo "  ⚠️  Docker not running. RD-Agent needs Docker for code execution."
    echo "     Start Docker and try again."
    exit 1
fi

echo ""

# ──────── Prepare Qlib Data ────────
echo "📊 Preparing market data..."
python3 -c "
import os, sys
try:
    import qlib
    data_path = os.path.expanduser('~/.qlib/qlib_data/cn_data')
    if os.path.exists(data_path):
        print('  ✅ CSI 300 data available')
    else:
        print('  📥 Downloading CSI 300 data (first time only, ~5 min)...')
        os.system('python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn')
        print('  ✅ Data downloaded')
except ImportError:
    print('  ⚠️  Qlib not installed. Run: pip install pyqlib')
    sys.exit(1)
"

echo ""

# ──────── Create Run Directory ────────
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_${PROFILE}"
RUN_DIR="logs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "📁 Run directory: $RUN_DIR"

# Save run config
cat > "$RUN_DIR/config.json" << EOF
{
    "profile": "$PROFILE",
    "iterations": $ITERATIONS,
    "scenario": "$SCENARIO",
    "chat_model": "$CHAT_MODEL",
    "synthesis_model": "${SYNTHESIS_MODEL:-$CHAT_MODEL}",
    "implementation_model": "${IMPLEMENTATION_MODEL:-$CHAT_MODEL}",
    "analysis_model": "${ANALYSIS_MODEL:-$CHAT_MODEL}",
    "timestamp": "$(date -Iseconds)"
}
EOF

echo ""

# ──────── Run RD-Agent ────────
echo "🚀 Starting RD-Agent $SCENARIO ($ITERATIONS iterations)..."
echo "   This will take approximately $((ITERATIONS * 10)) - $((ITERATIONS * 30)) minutes."
echo "   Cost tracking: $RUN_DIR/cost_log.json"
echo ""
echo "───────────────────────────────────────────────"

# Set RD-Agent to use our patched backend
export BACKEND="rdagent.oai.backend.LiteLLMAPIBackend"
export LOG_DIR="$RUN_DIR"

# Run the backtest
rdagent "$SCENARIO" 2>&1 | tee "$RUN_DIR/output.log"

echo "───────────────────────────────────────────────"
echo ""

# ──────── Post-Run Analysis ────────
echo "📊 Generating analysis report..."

python3 << 'ANALYSIS_SCRIPT'
import json, os, glob

run_dir = os.environ.get("RUN_DIR", "logs/latest")

# Try to find cost log
cost_files = glob.glob(f"{run_dir}/cost_log*.json")
if cost_files:
    with open(cost_files[0]) as f:
        data = json.load(f)
    
    summary = data.get("summary", {})
    print(f"\n{'═'*50}")
    print(f"  COST SUMMARY")
    print(f"{'═'*50}")
    print(f"  Total cost:     ${summary.get('total_cost_usd', 'N/A')}")
    print(f"  Total API calls: {summary.get('total_calls', 'N/A')}")
    print(f"\n  Per-stage breakdown:")
    for stage, info in summary.get("stage_breakdown", {}).items():
        print(f"    {stage:20s}: ${info['cost_usd']:.4f} ({info['calls']} calls)")
    print(f"{'═'*50}\n")
else:
    print("  ℹ️  No cost log found (normal if using baseline profile)")

print(f"  📁 Full logs: {run_dir}/")
print(f"  📊 RD-Agent UI: rdagent ui --log_dir {run_dir}")
ANALYSIS_SCRIPT

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Backtest Complete!                       ║"
echo "║                                              ║"
echo "║  View results:                               ║"
echo "║    rdagent ui --log_dir $RUN_DIR             ║"
echo "║                                              ║"
echo "║  Compare profiles:                           ║"
echo "║    python3 scripts/compare_runs.py           ║"
echo "╚══════════════════════════════════════════════╝"
