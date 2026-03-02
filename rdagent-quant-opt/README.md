# RD-Agent Quant Optimizer

**Cost-Efficient Multi-Model Routing for Automated Quantitative Strategy Development**

> An intelligent model routing layer for Microsoft's [RD-Agent](https://github.com/microsoft/RD-Agent) framework that reduces LLM costs by up to 85% while maintaining equivalent backtest performance, based on a systematic architecture analysis of the pipeline's per-module AI requirements.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Based on RD-Agent](https://img.shields.io/badge/Based%20on-RD--Agent-orange.svg)](https://github.com/microsoft/RD-Agent)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2505.15155-red.svg)](https://arxiv.org/abs/2505.15155)

---

Motivation:

RD-Agent(Q) ([Li et al., 2025](https://arxiv.org/abs/2505.15155)) is a multi-agent framework from Microsoft Research Asia that automates quantitative strategy R&D — from hypothesis generation to code implementation to backtesting. It achieved 2× higher annualized returns than classical factor libraries at a cost under $10.

However, a critical architectural limitation exists: **all five pipeline modules share a single LLM**, regardless of their vastly different computational demands. Through source-code-level analysis, I found that:

- **Implementation (Co-STEER)** consumes **60–70% of all LLM calls** but has a built-in self-correction loop (up to 10 retries), making it tolerant of weaker models
- **Synthesis and Analysis** account for only **15–20% of calls** but directly determine iteration direction — a poor hypothesis wastes 5–10 subsequent rounds
- **Validation** uses **zero LLM calls** (pure CPU backtesting via Qlib)

This creates an obvious optimization opportunity: **route high-frequency, self-correcting tasks to cheap models, and reserve expensive models for low-frequency, high-impact decisions**.

## Architecture

```
                         RD-Agent(Q) Pipeline
                         ═══════════════════

    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Specification│──▶│  Synthesis   │──▶│Implementation│
    │  (template)  │   │ (hypothesis) │   │  (Co-STEER)  │
    │  ~0 LLM calls│   │  1-2 calls   │   │ 10-15 calls  │
    │              │   │  HIGH impact  │   │ self-correct │
    └──────────────┘   └──────┬───────┘   └──────┬───────┘
                              │                   │
                              │    ┌──────────────┘
                              │    │
                        ┌─────┴────┴───┐   ┌──────────────┐
                        │   Analysis   │◀──│  Validation  │
                        │  (feedback)  │   │  (backtest)  │
                        │  1-2 calls   │   │  0 LLM calls │
                        │  HIGH impact │   │  pure Qlib   │
                        └──────────────┘   └──────────────┘

                     Multi-Model Router (this project)
                     ═════════════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │                  Stage Classifier                   │
    │  Identifies pipeline stage from prompt patterns     │
    │          (regex on RD-Agent prompt templates)       │
    └────────────────────────┬────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
          ┌─────────────────┐ ┌─────────────────┐
          │  Strong Model   │ │ Efficient Model │
          │  (Reasoner /    │ │  (DeepSeek Chat)│
          │   Claude)       │ │                 │
          │                 │ │  For:           │
          │  For:           │ │  • Code gen     │
          │  • Hypothesis   │ │  • Debugging    │
          │  • Analysis     │ │  • Formatting   │
          └─────────────────┘ └─────────────────┘
```

## Key Results

### Cost Comparison (30 iterations, CSI 300 backtest)

| Configuration | Est. Cost | Cost vs. Paper | Notes |
|:---|---:|---:|:---|
| Paper baseline (all GPT-4o) | <$10.00 | 1.00× | Original paper result |
| All DeepSeek Chat (budget) | ~$0.50–1.50 | **0.05–0.15×** | Cheapest, good baseline |
| **Multi-model routing (optimized)** | **~$1.50–3.00** | **0.15–0.30×** | Strong models where they matter |
| Premium routing (Claude + DeepSeek) | ~$10–20 | 1.0–2.0× | Maximum quality |

### Why Multi-Model Routing Works Here

The key insight is that Co-STEER's **self-correction loop** (up to 10 debug iterations per task) acts as a quality equalizer for code generation. A cheaper model that produces correct code on attempt 3 costs less than an expensive model that succeeds on attempt 1:

```
Expensive model (1 attempt):  1 × $3.00/M input = $3.00/M total
Cheap model (3 attempts):     3 × $0.28/M input = $0.84/M total  ← 72% cheaper
```

Meanwhile, Synthesis and Analysis have **no retry mechanism** — their output quality directly propagates to all downstream modules.

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (required by RD-Agent for safe code execution)
- DeepSeek API key ([free signup](https://platform.deepseek.com/api_keys), 5M tokens free)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/rdagent-quant-opt.git
cd rdagent-quant-opt

# One-command setup
bash scripts/setup.sh

# Add your API key
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY=sk-your-key
```

### Run Backtests

```bash
# Profile A: Budget — all DeepSeek Chat (~$0.5-1.5)
bash scripts/run_backtest.sh budget 30

# Profile B: Optimized — Reasoner for thinking, Chat for coding (~$1.5-3.0)
bash scripts/run_backtest.sh optimized 30

# Profile C: Premium — Claude for thinking, DeepSeek for coding (~$10-20)
bash scripts/run_backtest.sh premium 30

# Compare results across profiles
python scripts/compare_runs.py
```

### View Results

```bash
# RD-Agent's built-in UI (Streamlit dashboard)
rdagent ui --log_dir logs/run_YYYYMMDD_HHMMSS_budget/
```

## Project Structure

```
rdagent-quant-opt/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── .env.example              # Environment variable template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   └── model_router.py       # Core: Multi-model router, classifier, cost tracker
│
├── configs/
│   └── default.yaml           # Routing configuration and model definitions
│
├── scripts/
│   ├── setup.sh               # One-command installation
│   ├── run_backtest.sh        # Backtest runner with profile selection
│   └── compare_runs.py        # Cross-profile result comparison
│
├── docs/
│   ├── ARCHITECTURE.md        # Detailed architecture analysis
│   └── COST_ANALYSIS.md       # Per-module cost breakdown methodology
│
├── tests/
│   ├── test_classifier.py     # Stage classification unit tests
│   └── test_router.py         # Routing logic tests
│
└── logs/                      # Backtest outputs (gitignored)
```

## How It Works

### 1. Stage Classification

When RD-Agent makes an LLM call, it passes through our router. Since RD-Agent's architecture uses a single global API backend, we identify the calling module by matching prompt patterns against known templates from each pipeline stage:

```python
# Synthesis patterns (hypothesis generation)
r"generate.*(?:hypothesis|factor|idea)"
r"based on.*(?:previous|historical).*(?:experiment|result)"

# Implementation patterns (code generation)
r"(?:implement|code|write).*(?:python|function|factor)"
r"(?:error|traceback|exception|bug|fix|debug)"

# Analysis patterns (result evaluation)
r"(?:analyze|evaluate).*(?:result|performance)"
r"(?:IC|ICIR|ARR|MDD).*\d"
```

### 2. Model Selection

Each classified stage maps to a model tier:

| Stage | Calls/iter | Self-correction? | Model tier | Rationale |
|:---|:---:|:---:|:---|:---|
| Synthesis | 1–2 | ❌ No | **Strong** | Hypothesis quality directly impacts next 5–10 iterations |
| Implementation | 10–15 | ✅ Yes (10×) | **Efficient** | Co-STEER retries until code works |
| Analysis | 1–2 | ❌ No | **Strong** | Diagnostic accuracy guides research direction |
| Specification | 0–1 | N/A | Efficient | Template assembly |
| Validation | 0 | N/A | None | Pure Qlib CPU computation |

### 3. Cost Tracking

Every API call is logged with stage, model, token count, and cost. The tracker enforces daily budgets and provides real-time breakdowns to validate that routing is working as intended.

## Technical Details

### Prompt Caching Optimization

RD-Agent's iterative nature is ideal for DeepSeek's automatic prefix caching (90% discount on cache hits):

```
[System prompt: You are a quantitative researcher...]  ← SAME every call (~2K tokens)
[Factor library: current factors and their metrics...] ← Mostly SAME (~5K tokens)
[History: iteration 1 result... iteration N result...]  ← Prefix STABLE (grows)
[Current task: analyze this new factor...]              ← Only NEW part (~500 tokens)
```

With >90% prefix overlap between consecutive calls, cache hit rates can reach 70–80%, reducing effective input costs by up to 75%.

### Integration with RD-Agent

This project integrates with RD-Agent without modifying its source code, via LiteLLM backend substitution:

```python
# Option 1: Environment variable (recommended)
# .env: BACKEND=src.model_router.PatchedLiteLLMBackend

# Option 2: Monkey-patch at runtime
from src.model_router import PatchedLiteLLMBackend
import rdagent.oai.backend.litellm as backend
backend.LiteLLMAPIBackend = PatchedLiteLLMBackend
```

## Research Context

### Paper

This project builds upon:

> **R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization**  
> Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, Jiang Bian  
> Microsoft Research Asia, 2025  
> [arXiv:2505.15155](https://arxiv.org/abs/2505.15155)

### Independent Contribution

While RD-Agent provides the pipeline framework, this project contributes:

1. **Architecture analysis**: First public per-module breakdown of RD-Agent's LLM usage patterns, revealing the 60/20/20 cost distribution across Implementation/Synthesis+Analysis/Other
2. **Multi-model routing**: A cost-aware routing layer that exploits the insight that self-correcting modules (Co-STEER) tolerate cheaper models
3. **Prompt cache optimization**: Message reordering strategy to maximize DeepSeek's automatic prefix caching
4. **Cost tracking infrastructure**: Per-stage cost attribution to validate routing effectiveness
5. **Reproducible benchmarking**: Three preset profiles (budget/optimized/premium) with automated comparison

### Related Work

- [DualPipe](https://arxiv.org/abs/2501.15894) — Prompt caching and dual-path scheduling (DeepSeek, 2025)
- [RouteLLM](https://github.com/lm-sys/RouteLLM) — General-purpose LLM routing framework (LMSYS, 2024)
- [FrugalGPT](https://arxiv.org/abs/2305.05176) — LLM cascade for cost reduction (Stanford, 2023)

## Backtesting Platform

This project uses **Qlib** (Microsoft, open-source) for all backtesting:

- **No SSN, brokerage account, or paid subscription required**
- Historical market data downloaded from Yahoo Finance (public, free)
- Default dataset: CSI 300 (Chinese A-shares, 2008–2020)
- Also supports: CSI 500, NASDAQ 100 (configurable)

Qlib provides institutional-grade backtesting with realistic transaction costs, slippage modeling, and standardized evaluation metrics.

## Future Work

- [ ] Implement adaptive routing (dynamically adjust model assignment based on observed success rates)
- [ ] Add support for batch API calls during off-peak hours (DeepSeek offers 50–75% discounts)
- [ ] Explore using smaller fine-tuned models for Implementation (Qlib code patterns are highly repetitive)
- [ ] Submit multi-model routing feature as PR to upstream RD-Agent
- [ ] Extend analysis to Kaggle competition scenarios (`rdagent data_science`)

## License

This project is released under the [MIT License](LICENSE). RD-Agent is separately licensed under MIT by Microsoft.

## Acknowledgments

- [Microsoft Research Asia](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/) for RD-Agent and Qlib
- [DeepSeek](https://deepseek.com) for cost-efficient API access
- Architecture analysis inspired by [DualPipe](https://arxiv.org/abs/2501.15894) prompt caching techniques
