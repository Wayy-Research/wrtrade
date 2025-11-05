# Ultra-fast backtesting library using Polars with N-dimensional portfolios

# Core portfolio system
from wrtrade.portfolio import Portfolio
from wrtrade.components import SignalComponent, CompositePortfolio
from wrtrade.ndimensional_portfolio import NDimensionalPortfolioBuilder, AdvancedPortfolioManager

# Analysis and optimization
from wrtrade.metrics import tear_sheet, calculate_all_metrics, calculate_all_rolling_metrics
from wrtrade.permutation import PermutationTester, PermutationConfig
from wrtrade.kelly import KellyOptimizer, HierarchicalKellyOptimizer, KellyConfig

# Simple deployment
from wrtrade.deploy import deploy, validate_strategy, DeployConfig

__version__ = "2.0.0"

__all__ = [
    # Core
    'Portfolio',
    'SignalComponent',
    'CompositePortfolio',
    'NDimensionalPortfolioBuilder',
    'AdvancedPortfolioManager',

    # Metrics
    'tear_sheet',
    'calculate_all_metrics',
    'calculate_all_rolling_metrics',

    # Validation
    'PermutationTester',
    'PermutationConfig',
    'validate_strategy',

    # Optimization
    'KellyOptimizer',
    'HierarchicalKellyOptimizer',
    'KellyConfig',

    # Deployment
    'deploy',
    'DeployConfig',
]