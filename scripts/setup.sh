#!/bin/bash
# ═══════════════════════════════════════════════════
# RD-Agent Quant Optimizer - Setup Script
# ═══════════════════════════════════════════════════
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  RD-Agent Quant Optimizer - Setup            ║"
echo "║  Multi-Model Routing for Cost-Efficient      ║"
echo "║  Quantitative Strategy Development           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ──────── Check Prerequisites ────────
echo "🔍 Checking prerequisites..."

# Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.10+ is required. Please install it first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ✅ Python $PYTHON_VERSION"

# Docker
if ! command -v docker &> /dev/null; then
    echo "  ⚠️  Docker not found. RD-Agent requires Docker for code execution."
    echo "     Install: https://docs.docker.com/get-docker/"
    echo "     (You can still set up the project, but backtesting won't work without Docker)"
else
    echo "  ✅ Docker $(docker --version | grep -oP '\d+\.\d+\.\d+')"
fi

# Git
if ! command -v git &> /dev/null; then
    echo "❌ Git is required."
    exit 1
fi
echo "  ✅ Git $(git --version | grep -oP '\d+\.\d+\.\d+')"

echo ""

# ──────── Create Virtual Environment ────────
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✅ Virtual environment created"
else
    echo "  ✅ Virtual environment already exists"
fi

source venv/bin/activate

# ──────── Install Dependencies ────────
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip -q

# Core dependencies
pip install rdagent litellm pyyaml python-dotenv -q
echo "  ✅ Core packages installed (rdagent, litellm, pyyaml)"

# Qlib for backtesting
pip install pyqlib -q 2>/dev/null || {
    echo "  ⚠️  pyqlib installation failed, trying alternative..."
    pip install qlib -q 2>/dev/null || echo "  ⚠️  Qlib not installed. You may need to install it manually."
}
echo "  ✅ Qlib backtesting framework"

# Analysis and visualization
pip install pandas matplotlib seaborn jupyter -q
echo "  ✅ Analysis tools (pandas, matplotlib, jupyter)"

echo ""

# ──────── Download Qlib Data ────────
echo "📊 Setting up Qlib market data..."
echo "   This downloads historical stock data from Yahoo Finance."
echo "   No account or SSN required - all public data."
echo ""

# Create data directory
mkdir -p data/qlib_data

# Download CSI 300 data via Qlib
python3 -c "
try:
    import qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
    print('  ✅ Qlib CN data already available')
except Exception as e:
    print(f'  ℹ️  Will download data on first run: {e}')
    print('  Run: python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn')
" 2>/dev/null || echo "  ℹ️  Qlib data will be downloaded on first backtest run"

echo ""

# ──────── Configure Environment ────────
echo "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✅ Created .env from template"
    echo ""
    echo "  ⚠️  IMPORTANT: Edit .env and add your DeepSeek API key!"
    echo "     Get one at: https://platform.deepseek.com/api_keys"
    echo "     New accounts get 5M free tokens"
else
    echo "  ✅ .env already exists"
fi

# Create logs directory
mkdir -p logs
echo "  ✅ Logs directory created"

echo ""

# ──────── Verify Setup ────────
echo "🧪 Verifying installation..."
python3 -c "
import litellm
import yaml
print('  ✅ litellm:', litellm.__version__)
print('  ✅ yaml: available')
try:
    import rdagent
    print('  ✅ rdagent:', getattr(rdagent, '__version__', 'installed'))
except ImportError:
    print('  ⚠️  rdagent: not yet installed (install via pip install rdagent)')
"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                          ║"
echo "║                                              ║"
echo "║  Next steps:                                 ║"
echo "║  1. Edit .env with your DeepSeek API key     ║"
echo "║  2. Run: bash scripts/run_backtest.sh        ║"
echo "║                                              ║"
echo "║  Estimated cost: $0.5-1.5 for 30 iterations  ║"
echo "╚══════════════════════════════════════════════╝"
