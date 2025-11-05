# Ultra-fast backtesting library using Polars with N-dimensional portfolios

# Core portfolio system
from wrtrade.portfolio import Portfolio
from wrtrade.components import PortfolioComponent, SignalComponent, CompositePortfolio, AllocationWeights
from wrtrade.ndimensional_portfolio import NDimensionalPortfolioBuilder, AdvancedPortfolioManager, PortfolioBuilderConfig

# Analysis and optimization
from wrtrade.metrics import tear_sheet, calculate_all_metrics, calculate_all_rolling_metrics
from wrtrade.permutation import PermutationTester, PricePermutationGenerator, PermutationConfig
from wrtrade.kelly import KellyOptimizer, HierarchicalKellyOptimizer, KellyConfig

# Trading and deployment
from wrtrade.brokers import BrokerAdapter, AlpacaBrokerAdapter, BrokerFactory, TradingSession
from wrtrade.deployment import WRTradeDeploymentSystem, StrategyValidator, StrategyDeployer, DeploymentConfig

__version__ = "2.0.0"

__all__ = [
    # Legacy
    'Portfolio', 'tear_sheet', 'calculate_all_metrics', 'calculate_all_rolling_metrics',
    
    # N-dimensional portfolio system
    'PortfolioComponent', 'SignalComponent', 'CompositePortfolio', 'AllocationWeights',
    'NDimensionalPortfolioBuilder', 'AdvancedPortfolioManager', 'PortfolioBuilderConfig',
    
    # Permutation testing
    'PermutationTester', 'PricePermutationGenerator', 'PermutationConfig',
    
    # Kelly optimization
    'KellyOptimizer', 'HierarchicalKellyOptimizer', 'KellyConfig',
    
    # Trading infrastructure
    'BrokerAdapter', 'AlpacaBrokerAdapter', 'BrokerFactory', 'TradingSession',
    
    # Deployment system
    'WRTradeDeploymentSystem', 'StrategyValidator', 'StrategyDeployer', 'DeploymentConfig'
]