"""
RD-Agent Quant Optimizer: Multi-Model Routing + Anti-Decay + Adversarial Debate
for Cost-Efficient Quantitative R&D.
"""

__version__ = "0.7.0"
__author__ = "Joshua Zhou"

from .model_router import (
    MultiModelRouter, PipelineStage, StageClassifier, CostTracker,
    AdaptiveModelSelector, MODEL_REGISTRY,
)
from .alpha_filter import AlphaDecayFilter, ASTSimilarityChecker, ComplexityChecker
from .debate_agents import DebateAnalyzer, Verdict, DebateResult
from .market_regime import MarketRegimeDetector, RegimeAwareSynthesis, MarketRegime
from .pipeline import OptimizedPipeline, IterationResult, PipelineReport
from .data_provider import DataProvider, PolygonProvider, YahooFallbackProvider, HybridProvider
from .survivorship_bias import SurvivorshipBiasCorrector
from .data_validator import DataValidator, ValidationReport, adjust_for_splits
from .experience_memory import ExperienceMemory, Experience
from .param_optimizer import ParameterExtractor, ParameterOptimizer, ParamSpec
from .trajectory_evolution import TrajectoryBuilder, TrajectoryEvolver, Trajectory
from .report_generator import ReportGenerator, ReportConfig
from .qlib_backtester import QlibBacktester, BacktestConfig, BacktestResult, create_backtest_objective
from .factor_translator import FactorTranslator, FactorAnalyzer, TradingStrategy
from .factor_reviewer import FactorReviewer, FactorReview, ReviewGrade
from .factor_library import FactorLibrary, FactorEntry
from .position_guard import PositionGuard, GuardConfig, Position, GuardResult
from .self_optimizer import SelfOptimizer, OptimizationReport, CodeAnalyzer
from .external_scout import (
    ScoutPipeline, ScoutConfig, ScoutResult, ArxivScout, GitHubScout,
)
from .fundamental_analyst import (
    FundamentalAnalyzer, FundamentalData, FundamentalSignal,
    SignalGenerator, FundamentalLLMAnalyst, Signal,
)
from .investment_principles import (
    kelly_criterion, mean_variance_optimize, risk_parity,
    black_litterman, drawdown_position_control, compute_position_advice,
    KellyResult, MVOResult, RiskParityResult, BlackLittermanResult,
)
