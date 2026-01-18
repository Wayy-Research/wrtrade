"""
wrtrade - Ultra-fast backtesting and trading framework.

Simple API:
    # One-line backtest
    result = wrt.backtest(my_signal, prices)
    print(result.sortino)

    # Portfolio with multiple signals
    portfolio = wrt.Portfolio([
        (trend_signal, 0.6),
        (momentum_signal, 0.4),
    ])
    result = portfolio.backtest(prices)

    # Validate
    p_value = wrt.validate(my_signal, prices)

    # Optimize
    portfolio.optimize(prices)
"""

# =============================================================================
# Core API - what most users need
# =============================================================================

from wrtrade.portfolio import (
    # Core data types
    Portfolio,
    Result,

    # Convenience functions
    backtest,
    validate,
    optimize,
)

# Metrics
from wrtrade.metrics import tear_sheet, calculate_all_metrics, calculate_all_rolling_metrics

# =============================================================================
# Full API - for advanced usage and customization
# =============================================================================

# Validation (permutation testing)
from wrtrade.permutation import PermutationTester, PermutationConfig

# Optimization (Kelly criterion)
from wrtrade.kelly import KellyOptimizer, HierarchicalKellyOptimizer, KellyConfig

# Deployment
from wrtrade.deploy import deploy, validate_strategy, DeployConfig

# Components (for building complex hierarchical portfolios)
from wrtrade.components import SignalComponent, CompositePortfolio

# Builder (legacy - prefer Portfolio class for new code)
from wrtrade.ndimensional_portfolio import NDimensionalPortfolioBuilder, AdvancedPortfolioManager

# Charts (wrchart integration)
from wrtrade.charts import (
    BacktestChart,
    price_chart,
    line_chart,
    area_chart,
    histogram,
    bar_chart,
    indicator_panel,
    plot_backtest,
)

__version__ = "2.1.0"

__all__ = [
    # === Core API (what most users need) ===
    'Portfolio',
    'Result',
    'backtest',
    'validate',
    'optimize',
    'tear_sheet',

    # === Full API (advanced usage) ===

    # Metrics
    'calculate_all_metrics',
    'calculate_all_rolling_metrics',

    # Validation
    'PermutationTester',
    'PermutationConfig',

    # Optimization
    'KellyOptimizer',
    'HierarchicalKellyOptimizer',
    'KellyConfig',

    # Deployment
    'deploy',
    'validate_strategy',
    'DeployConfig',

    # Components (for hierarchical portfolios)
    'SignalComponent',
    'CompositePortfolio',

    # Builder (legacy)
    'NDimensionalPortfolioBuilder',
    'AdvancedPortfolioManager',

    # Charts
    'BacktestChart',
    'price_chart',
    'line_chart',
    'area_chart',
    'histogram',
    'bar_chart',
    'indicator_panel',
    'plot_backtest',
]
