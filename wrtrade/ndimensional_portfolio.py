import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import logging

from wrtrade.components import PortfolioComponent, SignalComponent, CompositePortfolio, AllocationWeights
from wrtrade.trade import calculate_positions, calculate_returns
from wrtrade.metrics import calculate_all_metrics


logger = logging.getLogger(__name__)


@dataclass 
class PortfolioBuilderConfig:
    """Configuration for building N-dimensional portfolios."""
    default_weight: float = 1.0
    max_depth: int = 10
    kelly_optimization: bool = True
    rebalance_frequency: Optional[int] = 252  # Annual rebalancing by default
    risk_free_rate: float = 0.02


class NDimensionalPortfolioBuilder:
    """
    Builder class for constructing complex N-dimensional portfolios.
    Supports nested portfolios with arbitrary depth and composition.
    """
    
    def __init__(self, config: Optional[PortfolioBuilderConfig] = None):
        """Initialize portfolio builder with configuration."""
        self.config = config or PortfolioBuilderConfig()
        self._signal_registry = {}
        
    def register_signal(self, name: str, signal_func: Callable) -> 'NDimensionalPortfolioBuilder':
        """
        Register a signal generating function for reuse across portfolios.
        
        Args:
            name: Signal name for reference
            signal_func: Function that takes prices and returns signals
            
        Returns:
            Self for method chaining
        """
        self._signal_registry[name] = signal_func
        return self
        
    def create_signal_component(
        self, 
        name: str, 
        signal_func_or_name: Union[str, Callable],
        weight: float = None,
        **kwargs
    ) -> SignalComponent:
        """
        Create a signal component.
        
        Args:
            name: Component name
            signal_func_or_name: Signal function or registered name
            weight: Component weight (uses default if None)
            **kwargs: Additional SignalComponent parameters
            
        Returns:
            SignalComponent instance
        """
        if weight is None:
            weight = self.config.default_weight
            
        if isinstance(signal_func_or_name, str):
            if signal_func_or_name not in self._signal_registry:
                raise ValueError(f"Signal '{signal_func_or_name}' not registered")
            signal_func = self._signal_registry[signal_func_or_name]
        else:
            signal_func = signal_func_or_name
            
        return SignalComponent(name, signal_func, weight, **kwargs)
    
    def create_portfolio(
        self, 
        name: str,
        components: List[PortfolioComponent],
        weight: float = None,
        **kwargs
    ) -> CompositePortfolio:
        """
        Create a composite portfolio from components.
        
        Args:
            name: Portfolio name
            components: List of child components
            weight: Portfolio weight (uses default if None)
            **kwargs: Additional CompositePortfolio parameters
            
        Returns:
            CompositePortfolio instance
        """
        if weight is None:
            weight = self.config.default_weight
            
        portfolio_kwargs = {
            'rebalance_frequency': self.config.rebalance_frequency,
            'kelly_optimization': self.config.kelly_optimization,
            **kwargs
        }
        
        return CompositePortfolio(name, components, weight, **portfolio_kwargs)
    
    def create_balanced_portfolio(
        self, 
        name: str,
        components: List[PortfolioComponent],
        weight: float = None
    ) -> CompositePortfolio:
        """
        Create a portfolio with equal-weighted components.
        
        Args:
            name: Portfolio name
            components: List of components to balance
            weight: Portfolio weight
            
        Returns:
            CompositePortfolio with balanced components
        """
        if not components:
            raise ValueError("Cannot create balanced portfolio with no components")
            
        equal_weight = 1.0 / len(components)
        for component in components:
            component.set_weight(equal_weight)
            
        return self.create_portfolio(name, components, weight)
        
    def validate_portfolio_structure(self, portfolio: CompositePortfolio, depth: int = 0) -> bool:
        """
        Validate portfolio structure for circular references and max depth.
        
        Args:
            portfolio: Portfolio to validate
            depth: Current nesting depth
            
        Returns:
            True if valid, raises exception if invalid
        """
        if depth > self.config.max_depth:
            raise ValueError(f"Portfolio depth exceeds maximum of {self.config.max_depth}")
            
        seen_names = set()
        for component in portfolio.component_list:
            if component.name in seen_names:
                raise ValueError(f"Duplicate component name: {component.name}")
            seen_names.add(component.name)
            
            if isinstance(component, CompositePortfolio):
                # Recursively validate sub-portfolios at increased depth
                self.validate_portfolio_structure(component, depth + 1)
                
        return True


class AdvancedPortfolioManager:
    """
    Advanced portfolio manager that handles N-dimensional portfolios with
    sophisticated features like Kelly optimization, permutation testing,
    and performance attribution.
    """
    
    def __init__(self, root_portfolio: CompositePortfolio):
        """
        Initialize portfolio manager.
        
        Args:
            root_portfolio: The top-level portfolio to manage
        """
        self.root_portfolio = root_portfolio
        self._backtest_results = None
        self._attribution_results = None
        self._validation_results = None
        
    def backtest(self, prices: pl.Series) -> Dict[str, Any]:
        """
        Run comprehensive backtest on the portfolio.
        
        Args:
            prices: Price data for backtesting
            
        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting backtest for portfolio: {self.root_portfolio.name}")
        
        # Calculate portfolio returns
        portfolio_returns = self.root_portfolio.calculate_returns(prices)
        
        # Calculate benchmark returns (buy-and-hold)
        benchmark_returns = prices.log().diff().fill_null(0)
        
        # Get performance metrics
        portfolio_metrics = self.root_portfolio.get_performance_metrics()
        benchmark_metrics = calculate_all_metrics(benchmark_returns)
        
        # Calculate performance attribution
        attribution = self._calculate_performance_attribution()
        
        self._backtest_results = {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'portfolio_metrics': portfolio_metrics,
            'benchmark_metrics': benchmark_metrics,
            'attribution': attribution,
            'total_return': portfolio_returns.cum_sum()[-1],
            'benchmark_total_return': benchmark_returns.cum_sum()[-1],
            'excess_return': portfolio_returns.cum_sum()[-1] - benchmark_returns.cum_sum()[-1]
        }
        
        logger.info(f"Backtest completed. Total return: {self._backtest_results['total_return']:.4f}")
        return self._backtest_results
    
    def _calculate_performance_attribution(self) -> Dict[str, Any]:
        """
        Calculate performance attribution across all portfolio components.
        
        Returns:
            Attribution analysis results
        """
        attribution = {'component_contributions': {}}
        
        def collect_contributions(portfolio: CompositePortfolio, prefix: str = ""):
            for component in portfolio.component_list:
                full_name = f"{prefix}{component.name}" if prefix else component.name
                
                if hasattr(component, '_returns') and component._returns is not None:
                    contribution = component._returns.sum() * component.weight
                    attribution['component_contributions'][full_name] = float(contribution)
                    
                if isinstance(component, CompositePortfolio):
                    collect_contributions(component, f"{full_name}/")
        
        collect_contributions(self.root_portfolio)
        return attribution
    
    def optimize_kelly_weights(
        self, 
        prices: pl.Series,
        lookback_window: int = 252,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using Kelly criterion.
        
        Args:
            prices: Price data for optimization
            lookback_window: Historical window for calculating statistics
            min_weight: Minimum component weight
            max_weight: Maximum component weight
            
        Returns:
            Dictionary of optimal weights
        """
        # This is a placeholder for Kelly optimization logic
        # Will be implemented in the Kelly optimization system
        logger.info("Kelly optimization will be implemented in the Kelly system")
        
        # For now, return current weights
        return self.root_portfolio.get_component_weights()
    
    def run_permutation_test(
        self, 
        prices: pl.Series,
        n_permutations: int = 1000,
        metric: str = 'sortino_ratio'
    ) -> Dict[str, Any]:
        """
        Run permutation test to validate strategy significance.
        
        Args:
            prices: Original price data
            n_permutations: Number of permutations to test
            metric: Performance metric to test
            
        Returns:
            Permutation test results with p-value
        """
        # This will be implemented in the permutation testing system
        logger.info(f"Permutation test will be implemented with {n_permutations} permutations")
        
        return {
            'p_value': None,
            'metric_value': None,
            'permutation_distribution': None,
            'status': 'pending_implementation'
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio performance report.
        
        Returns:
            Detailed performance report
        """
        if self._backtest_results is None:
            raise ValueError("Must run backtest before generating report")
            
        report = {
            'portfolio_structure': self._get_structure_summary(),
            'performance_summary': self._get_performance_summary(),
            'risk_metrics': self._get_risk_metrics(),
            'attribution': self._backtest_results['attribution'],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _get_structure_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio structure."""
        def count_components(portfolio: CompositePortfolio) -> Dict[str, int]:
            counts = {'signals': 0, 'portfolios': 0, 'total': 0}
            
            for component in portfolio.component_list:
                counts['total'] += 1
                if isinstance(component, SignalComponent):
                    counts['signals'] += 1
                elif isinstance(component, CompositePortfolio):
                    counts['portfolios'] += 1
                    sub_counts = count_components(component)
                    counts['signals'] += sub_counts['signals']
                    counts['portfolios'] += sub_counts['portfolios']
                    counts['total'] += sub_counts['total']
                    
            return counts
        
        return count_components(self.root_portfolio)
    
    def _get_performance_summary(self) -> Dict[str, float]:
        """Get key performance metrics summary."""
        return {
            'total_return': self._backtest_results['total_return'],
            'excess_return': self._backtest_results['excess_return'],
            'sharpe_ratio': self._backtest_results['portfolio_metrics'].get('sortino_ratio', 0.0),
            'max_drawdown': self._backtest_results['portfolio_metrics'].get('max_drawdown', 0.0),
            'volatility': self._backtest_results['portfolio_metrics'].get('volatility', 0.0)
        }
    
    def _get_risk_metrics(self) -> Dict[str, float]:
        """Get risk-related metrics."""
        return {
            'volatility': self._backtest_results['portfolio_metrics'].get('volatility', 0.0),
            'max_drawdown': self._backtest_results['portfolio_metrics'].get('max_drawdown', 0.0),
            'downside_deviation': 0.0,  # Will be calculated
            'var_95': 0.0,  # Will be calculated
            'expected_shortfall': 0.0  # Will be calculated
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        performance = self._get_performance_summary()
        
        if performance['excess_return'] < 0:
            recommendations.append("Portfolio underperforming benchmark - consider strategy review")
            
        if performance['max_drawdown'] < -0.2:
            recommendations.append("High maximum drawdown - consider risk management improvements")
            
        if performance['volatility'] > 0.3:
            recommendations.append("High volatility - consider position sizing optimization")
            
        if len(recommendations) == 0:
            recommendations.append("Portfolio showing good performance characteristics")
            
        return recommendations
    
    def print_structure(self) -> None:
        """Print the complete portfolio structure."""
        print(f"\n=== Portfolio Structure: {self.root_portfolio.name} ===")
        self.root_portfolio.print_structure()
        print("=" * 50)