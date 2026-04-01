# CLAUDE.md
# Project Context & Development Specification
# Claude Code will automatically read this file for context.

## Project Identity

**Name**: RD-Agent Quant Optimizer
**Author**: Joshua Zhou
**Purpose**: Cost-efficient multi-model routing + anti-alpha-decay + adversarial debate for automated quantitative strategy R&D
**Based on**: Microsoft RD-Agent (arXiv:2505.15155v2)
**Target**: GitHub open-source project + US college transfer application portfolio piece

---

## AI Behavioral Rules (Claude Code 行为准则)

### Safety First — 安全红线

1. **NEVER delete files outside `rdagent-quant-opt/` project directory**
2. **NEVER modify system files, registry, or Windows settings**
3. **NEVER commit `.env`, API keys, or any credential files**
4. **NEVER run `rm -rf`, `git reset --hard`, `git push --force` without explicit user approval**
5. **NEVER install system-level packages** — only project-level `pip install` in venv
6. **NEVER start background services/daemons** that persist after session ends
7. **NEVER access user's personal files** (Documents, Downloads, game folders, etc.)
8. All Docker containers must have resource limits (CPU, memory) to protect gaming performance

### Spec-Before-Code Protocol — 先规范再写码

Before writing ANY code, Claude MUST present to the user:

```
## 变更计划 (Change Plan)
**改什么**: [Which files, which functions]
**为什么**: [Why this change is needed]
**输入/输出**: [What goes in, what comes out]
**影响范围**: [What other parts of the system are affected]
**验收标准**: [How to verify it works]
**风险**: [What could go wrong]
**回滚方案**: [How to undo if something breaks]
**性能影响**: [Will it affect system/game performance? How?]
```

Wait for user confirmation before proceeding. Exception: trivial fixes (typo, formatting) can proceed directly.

### Communication Style — 沟通风格

- Always explain in **plain Chinese** what a change does and why
- For key code changes, add inline comment: `# 影响: xxx` or `# 功能: xxx`
- When presenting options, use numbered list with pros/cons
- Never assume the user understands technical jargon — explain it

### Development Workflow — 开发流程

1. **Spec** → Write requirements before code
2. **Test** → Write tests alongside or before implementation
3. **Implement** → Write the minimum code needed
4. **Verify** → Run tests, check for regressions
5. **Explain** → Tell the user what changed and why
6. **Commit** → Small, atomic commits with clear messages (only when user asks)

### Resource Protection — 资源保护

- Docker: limit to `--cpus=2 --memory=4g` to preserve gaming performance
- Background processes: always inform user, never auto-start
- Disk: warn if any operation will use > 1GB disk space
- Network: only access APIs explicitly configured in `.env`

---

## Architecture Overview (V2) — Chain-of-Alpha Variant

This project extends Microsoft's RD-Agent framework with a **dual-chain architecture**
inspired by Chain-of-Alpha (arXiv:2508.06312):

- **Factor Generation Chain**: Market Regime → Synthesis → Alpha Filter → Implementation
  (creative exploration, uses frontier/strong models)
- **Factor Optimization Chain**: Parameter Optimizer → Trajectory Evolution → Experience Memory
  (systematic refinement, zero LLM cost)

The two chains are connected by the **Bull-Bear Debate** system, which decides whether to
continue optimizing (stay in optimization chain) or pivot to a new hypothesis (restart
generation chain). This dual-chain design was independently developed and later validated
by Chain-of-Alpha (Aug 2025) and FactorEngine (Mar 2026).

### Independent Contributions:

### Contribution 1: Multi-Model Router + Adaptive Selector (COMPLETED)
- File: `src/model_router.py`
- Routes different pipeline stages to optimal LLM models
- **NEW: AdaptiveModelSelector** — 根据任务特征自动选模型:
  - 因子构思(高创造力) → frontier (Opus 4.6)
  - 结果分析(深度推理) → strong (DeepSeek Reasoner)
  - 代码生成(高频+重试) → efficient (DeepSeek Chat)
- MODEL_REGISTRY 集中管理所有模型，更新模型只需改注册表
- 三种预算模式: budget / optimized / premium
- Estimated cost reduction: 85% vs paper baseline (GPT-4o)

### Contribution 2: Anti-Alpha-Decay Filter (COMPLETED)
- File: `src/alpha_filter.py`
- AST similarity check: prevents factor crowding (rejects > 85% similar)
- Complexity constraint: limits nesting depth to 5 levels
- Hypothesis-factor alignment: LLM verifies code matches stated hypothesis
- **NEW: Hyperbolic decay** (Lee 2025): older factors have relaxed thresholds, alpha(t)=K/(1+λt)

### Contribution 3: Bull-Bear Adversarial Debate (COMPLETED)
- File: `src/debate_agents.py`
- Two adversarial agents (Bull/Bear) debate each iteration's direction
- Judge agent synthesizes CONTINUE/PIVOT/NEUTRAL verdict
- **NEW: Debate feedback loop**: verdict (PIVOT/CONTINUE) injects into next synthesis prompt
- Reduces confirmation bias, adds only 3 LLM calls per iteration

### Contribution 4: Market Regime-Aware RAG (COMPLETED)
- File: `src/market_regime.py`
- Classifies market: {bull/bear} × {high/low vol} × {trend/mean-revert}
- Injects regime context into hypothesis generation prompt
- Supports both rule-based (indicators) and LLM-based (text) detection

### Contribution 6: Experience Memory (COMPLETED)
- File: `src/experience_memory.py`
- Stores iteration results (hypothesis, code, metrics, verdict, lessons)
- Keyword-based retrieval: zero LLM cost
- Injects success/failure patterns into synthesis prompt
- Based on FactorMiner (arXiv:2602.14670) "Ralph Loop" concept

### Contribution 7: Parameter Optimizer (COMPLETED)
- File: `src/param_optimizer.py`
- Decouples formula structure (LLM) from parameter values (local search)
- AST-based extraction: identifies rolling(), pct_change(), shift() parameters
- Grid search optimization: zero LLM cost
- Based on FactorEngine (arXiv:2603.16365) logic/parameter decoupling

### Contribution 8: Trajectory Evolution (COMPLETED)
- File: `src/trajectory_evolution.py`
- Segments iterations into trajectories based on debate verdicts
- Mutation: replaces strategy keywords with alternatives
- Crossover: combines best elements from two trajectories
- Based on QuantaAlpha (arXiv:2602.07085) trajectory-level evolution

### Contribution 9: Report Generator (COMPLETED)
- File: `src/report_generator.py`
- Auto-generates Markdown + JSON reports after each pipeline run
- Sections: overview, iteration table, IC trends (sparkline), verdict distribution, filter analysis, cost breakdown, best/worst comparison, actionable recommendations
- Zero LLM cost — pure data aggregation
- Integrated into pipeline.run() — reports auto-saved to logs/

### Contribution 10: Real Backtesting Engine (COMPLETED)
- File: `src/qlib_backtester.py`
- Executes factor code against real S&P 500 data via Microsoft Qlib
- Computes IC (rank correlation), ICIR, annualized return, max drawdown, Sharpe ratio
- Long-short portfolio construction (top/bottom quintiles)
- Integrated into pipeline: replaces placeholder data with genuine metrics
- `create_backtest_objective()` connects to ParameterOptimizer for zero-LLM-cost parameter tuning
- Best results: IC=0.052, Sharpe=3.99, annual +83.4% (30-iter premium run)

### Contribution 11: Factor Translator (NEW — Phase 1)
- File: `src/factor_translator.py`
- Translates factor code → structured trading strategy (entry/exit/position rules)
- AST-based factor type detection (momentum/mean_reversion/volatility/volume/composite)
- Auto-generates rebalance frequency, stop-loss, take-profit based on backtest metrics
- Optional LLM-enhanced narrative generation
- 30 tests passing

### Contribution 12: Factor Reviewer (NEW — Phase 1)
- File: `src/factor_reviewer.py`
- Deep attribution analysis: decomposes returns into market beta vs. factor alpha
- IC stability analysis: decay rate, positive ratio, consecutive negative streaks
- Automated grading (A-F) with 0-100 scoring system
- Risk flag detection: overfitting, regime-dependent, data snooping, tail risk
- Optional LLM-enhanced deep analysis
- 29 tests passing

### Contribution 13: Factor Library (NEW — Phase 1)
- File: `src/factor_library.py`
- Centralized factor storage with JSON persistence
- Versioning, auto-tagging, ranking by IC/Sharpe/score
- Search by keyword, tags, metric ranges, grade
- AST-based similarity deduplication
- 19 tests passing

### Contribution 14: Position Guard (NEW — Phase 1)
- File: `src/position_guard.py`
- Enforces stop-loss, take-profit, trailing stop, min/max holding periods
- Position size caps, sector concentration limits
- Turnover monitoring and cost estimation
- 23 tests passing

## Current Project Structure

```
rdagent-quant-opt/
├── CLAUDE.md              ← THIS FILE
├── CHANGELOG.md           ← Version history
├── README.md
├── LICENSE (MIT)
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── CONTRIBUTING.md
│
├── src/
│   ├── __init__.py            ← v0.6.0, exports all modules
│   ├── model_router.py        ← Multi-model router + adaptive selector + cost tracker
│   ├── alpha_filter.py        ← AST similarity + complexity + alignment + hyperbolic decay
│   ├── debate_agents.py       ← Bull-Bear adversarial analysis (with feedback loop)
│   ├── market_regime.py       ← Market regime detection + RAG injection
│   ├── pipeline.py            ← Complete pipeline with debate feedback + memory
│   ├── data_provider.py       ← Unified data interface (Polygon + Yahoo), fallback for pipeline
│   ├── survivorship_bias.py   ← S&P 500 historical membership correction (integrated into backtester)
│   ├── data_validator.py      ← OHLCV data quality validation (integrated into backtester)
│   ├── experience_memory.py   ← Experience memory for cross-iteration learning
│   ├── param_optimizer.py     ← Logic/parameter decoupling optimizer
│   ├── trajectory_evolution.py← Trajectory-level mutation and crossover
│   ├── qlib_backtester.py     ← Real factor backtesting via Qlib (IC/ICIR/Sharpe)
│   ├── report_generator.py    ← Auto-generate Markdown + JSON reports
│   ├── factor_translator.py   ← NEW: Factor code → trading strategy translation
│   ├── factor_reviewer.py     ← NEW: Deep attribution analysis + grading
│   ├── factor_library.py      ← NEW: Factor storage, versioning, search, ranking
│   └── position_guard.py      ← NEW: Stop-loss, take-profit, turnover control
│
├── configs/
│   └── default.yaml        ← All module configs (data, models, filter, debate, regime)
│
├── scripts/
│   ├── setup.sh            ← One-command installation
│   ├── run_pipeline.py     ← One-command pipeline runner
│   ├── run_backtest.sh     ← Backtest runner with profile selection
│   ├── compare_runs.py     ← Cross-profile result comparison
│   ├── download_databento.py ← Databento exchange-direct data downloader
│   └── security_check.sh   ← 7-point pre-push privacy/security scan
│
├── tests/                       ← 303 unit + integration tests
│   ├── test_classifier.py       ← Stage classification (13 tests)
│   ├── test_router.py           ← Router + adaptive selector + cost (24 tests)
│   ├── test_alpha_filter.py     ← Alpha decay filter + hyperbolic decay (15 tests)
│   ├── test_debate.py           ← Debate agents (6 tests)
│   ├── test_market_regime.py    ← Market regime (9 tests)
│   ├── test_pipeline.py         ← Pipeline assembly (13 tests)
│   ├── test_report_generator.py ← Report generator (20 tests)
│   ├── test_experience_memory.py← Experience memory (17 tests)
│   ├── test_param_optimizer.py  ← Parameter optimizer (14 tests)
│   ├── test_trajectory.py       ← Trajectory evolution (16 tests)
│   ├── test_qlib_backtest.py    ← Real backtest integration (9 tests)
│   ├── test_survivorship.py     ← Survivorship bias (17 tests)
│   ├── test_data_validator.py   ← Data validation (22 tests)
│   ├── test_factor_translator.py← Factor translator (30 tests)
│   ├── test_factor_reviewer.py  ← Factor reviewer (29 tests)
│   ├── test_factor_library.py   ← Factor library (19 tests)
│   └── test_position_guard.py   ← Position guard (23 tests)
│
├── docs/
│   ├── ARCHITECTURE.md      ← Detailed architecture analysis
│   ├── COST_ANALYSIS.md     ← Per-module cost breakdown
│   ├── WORKFLOW.md          ← Collaboration workflow guide
│   ├── USER_MANUAL.md       ← Plain-language user manual (中文)
│   ├── DATA_SOURCES.md      ← Data source documentation
│   ├── ANTI_DECAY.md        ← Anti-alpha-decay filter explanation
│   ├── DEBATE_SYSTEM.md     ← Bull-Bear debate system explanation
│   └── SETUP_GUIDE.md      ← WSL + Qlib setup guide
│
├── data/                     ← Local data cache (gitignored)
│   └── sp500_changes.csv    ← S&P 500 historical membership (auto-downloaded)
│
└── logs/                     ← Backtest outputs (gitignored)
```

## Technical Constraints

- **Platform**: Must run on WSL (Ubuntu) on Windows. RD-Agent uses Linux-only APIs.
- **Docker**: Required by RD-Agent for safe code execution. Limit resources for gaming.
- **LLM Backend**: RD-Agent uses LiteLLM. Our router patches `LiteLLMAPIBackend`.
- **Backtesting**: Qlib (Microsoft) + custom data layer. Supports S&P 500 (US) and CSI 300 (China).
- **Data Sources**: Polygon.io (primary, professional-grade) + Yahoo Finance (fallback, free).
- **Budget**: DeepSeek API. Target: <$3 for 30 iterations.
- **Python**: 3.10+ required.
- **Privacy**: No personal data in code, no API keys in commits.

## Code Style & Conventions

- Python 3.10+ with type hints
- Logging via `logging` module (not print)
- Tests via pytest (202 tests, all passing)
- Config via YAML + environment variables
- All LLM interactions go through the router
- All data access through DataProvider interface
- Docstrings in English with Chinese annotations for key terms

## Key Papers

### Core References
1. RD-Agent(Q): arXiv:2505.15155v2 (Microsoft, 2025) — base framework
2. AlphaAgent: arXiv:2502.16789 (KDD 2025) — anti-alpha-decay filter
3. TradingAgents: arXiv:2412.20138 (Tauric Research, 2024) — adversarial debate
4. QuantaAlpha: arXiv:2602.07085 (2026) — trajectory evolution
5. DualPipe: arXiv:2501.15894 (DeepSeek, 2025) — prompt cache optimization

### Validating & Extending References (2025-2026)
6. AlphaLogics: arXiv:2603.20247 (Mar 2026) — market-logic-driven multi-agent factor generation
7. Chain-of-Alpha: arXiv:2508.06312 (Aug 2025) — dual-chain architecture (generation + optimization)
8. FactorMAD: ACM ICAIF 2025 — multi-agent debate for factor mining (validates our debate system)
9. FactorMiner: arXiv:2602.14670 (Feb 2026) — self-evolving agent with experience memory (validates experience_memory.py)
10. FactorEngine: arXiv:2603.16365 (Mar 2026) — logic/parameter decoupling (validates param_optimizer.py)
11. "The New Quant" Survey: arXiv:2510.05533 (Oct 2025) — comprehensive LLM+quant survey
12. Sentiment-Aware Alpha: arXiv:2508.04975 (Aug 2025) — LLM sentiment + formulaic alpha

### Contribution 5: Survivorship-Bias-Free Data Layer (COMPLETED + INTEGRATED)
- Files: `src/data_provider.py`, `src/survivorship_bias.py`, `src/data_validator.py`
- HybridProvider: Polygon.io (primary) + Yahoo Finance (fallback) — used as pipeline market data fallback
- SurvivorshipBiasCorrector: integrated into QlibBacktester._load_stock_data() — filters to point-in-time constituents
- DataValidator: integrated into QlibBacktester._load_stock_data() — validates OHLCV quality, removes bad stocks
- 6-point data quality validation with 0-1 scoring
- Target market: US S&P 500

### Pipeline Assembly (COMPLETED)
- File: `src/pipeline.py`
- Connects all components: Regime → Synthesis → Filter → Implementation → Validation → Debate
- Configurable via `configs/default.yaml`
- Cost tracking and result export

## Development Priority Order

1. ✅ Multi-model router (model_router.py)
2. ✅ AST similarity filter (alpha_filter.py)
3. ✅ Bull-Bear debate (debate_agents.py)
4. ✅ Market regime detector (market_regime.py)
5. ✅ Data layer upgrade (data_provider, survivorship_bias, data_validator)
6. ✅ Pipeline assembly (pipeline.py)
7. ✅ Full documentation (README, user manual, technical docs)
8. ✅ Real backtesting engine (qlib_backtester.py)
9. ✅ End-to-end pipeline wiring (param optimizer + trajectory evolution + real backtest)
10. 🔲 Run full 30-iteration backtest with real LLM + real data and collect results
11. 🔲 Push to GitHub + CI setup

## Environment Setup (WSL)

```bash
wsl
cd $PROJECT_ROOT
source venv/bin/activate  # or: source ~/myenv/bin/activate
export DEEPSEEK_API_KEY="sk-..."
```
