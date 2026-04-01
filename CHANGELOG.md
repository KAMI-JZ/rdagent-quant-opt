# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-03-31

### Added
- **Real Backtesting Engine** (`src/qlib_backtester.py`): Execute factor code against real S&P 500 data via Qlib. Computes IC, ICIR, annualized return, max drawdown, Sharpe ratio. Long-short portfolio (top/bottom quintiles).
- **Parameter Optimizer Integration**: Pipeline Step 5b now auto-extracts tunable params via AST and runs grid search optimization — zero LLM cost, can improve IC 5-30%.
- **Trajectory Evolution Integration**: Every 5 iterations, evolved hypothesis templates (mutation + crossover from top trajectories) are injected into synthesis prompt.
- **Real Market Indicators**: Step 1 now fetches real SPY data from Qlib for regime detection instead of hardcoded defaults.
- **Code Extraction**: Step 4 now auto-extracts pure Python from LLM markdown responses via `_extract_python_code()`.
- **Improved Code Generation Prompt**: Step 4 explicitly requests `def calculate_factor(df)` function format compatible with backtester.
- **End-to-end Run Script** (`scripts/run_pipeline.py`): One-command pipeline runner with `--profile`, `--iterations`, `--dry-run` options.
- 4 new tests for `_extract_python_code()` (total: 202 tests passing).
- 9 integration tests for real Qlib backtesting (`tests/test_qlib_backtest.py`).

### Changed
- Pipeline `_run_backtest()`: Replaced hardcoded stub with real Qlib backtester (graceful fallback if Qlib unavailable).
- Pipeline Step 2: Now injects trajectory evolution hints and experience memory context.
- README.md: Updated architecture diagram to dual-chain variant, added contributions 6-10, updated test count to 202.
- `model_router.py`: Updated DeepSeek V3.2 pricing ($0.28 input, $0.028 cache).

## [0.5.0] - 2026-03-29

### Added
- **Report Generator** (`src/report_generator.py`): Auto-generates Markdown + JSON reports with IC trends, verdict distribution, cost breakdown. Zero LLM cost.
- **Trajectory Evolution** (`src/trajectory_evolution.py`): Mutation and crossover on successful iteration trajectories. Based on QuantaAlpha.
- **Parameter Optimizer** (`src/param_optimizer.py`): AST-based parameter extraction + grid search. Based on FactorEngine.
- **Experience Memory** (`src/experience_memory.py`): Cross-iteration learning via keyword retrieval. Based on FactorMiner.
- 67 new tests (report_generator: 20, experience_memory: 17, param_optimizer: 14, trajectory: 16).

## [0.4.0] - 2026-03-28

### Added
- **Data Layer**: `data_provider.py` (Polygon + Yahoo hybrid), `survivorship_bias.py` (S&P 500 historical membership), `data_validator.py` (6-point quality check).
- **Alpha Filter Enhancements**: Hyperbolic decay (older factors have relaxed similarity thresholds).
- **Debate Feedback Loop**: Verdict (PIVOT/CONTINUE) now injected into next iteration's synthesis prompt.
- Security check script (`scripts/security_check.sh`).
- 39 new tests (survivorship: 17, data_validator: 22).

## [0.3.0] - 2026-03-27

### Added
- **Pipeline Assembly** (`src/pipeline.py`): Connects all components into iterative flow.
- **Market Regime Detector** (`src/market_regime.py`): Rule-based + LLM-based regime classification.
- **Bull-Bear Debate** (`src/debate_agents.py`): Adversarial analysis with Judge verdict.
- 28 new tests (pipeline: 13, market_regime: 9, debate: 6).

## [0.2.0] - 2026-03-26

### Added
- **Anti-Alpha-Decay Filter** (`src/alpha_filter.py`): AST similarity, complexity, alignment checks.
- 11 filter tests.

## [0.1.0] - 2026-03-25

### Added
- **Multi-Model Router** (`src/model_router.py`): Adaptive model selector, cost tracker, 3 budget profiles.
- **Stage Classifier**: Routes pipeline stages to optimal model tiers.
- Initial project structure, configs, README.
- 37 tests (classifier: 13, router: 24).
