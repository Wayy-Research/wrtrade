import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings

from wrtrade.components import PortfolioComponent, CompositePortfolio


logger = logging.getLogger(__name__)


@dataclass
class KellyConfig:
    """Configuration for Kelly optimization."""
    lookback_window: int = 252
    min_weight: float = 0.0
    max_weight: float = 1.0  
    max_leverage: float = 1.0
    regularization_lambda: float = 0.01
    rebalance_frequency: int = 21  # Monthly
    risk_free_rate: float = 0.02
    min_observations: int = 60
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    use_constraints: bool = True
    allow_short: bool = False


class KellyOptimizer:
    """
    Kelly Criterion optimizer for portfolio weight allocation.
    Implements both discrete Kelly formula and continuous optimization
    for portfolio of strategies with correlation considerations.
    """
    
    def __init__(self, config: Optional[KellyConfig] = None):
        """Initialize Kelly optimizer with configuration."""
        self.config = config or KellyConfig()
        
    def calculate_discrete_kelly(
        self, 
        returns: pl.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate discrete Kelly fraction for single asset.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Kelly fraction (optimal position size)
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate
        
        returns_np = returns.to_numpy()
        
        if len(returns_np) < self.config.min_observations:
            logger.warning(f"Insufficient data for Kelly calculation: {len(returns_np)} < {self.config.min_observations}")
            return 0.0
        
        # Calculate excess returns
        daily_rf_rate = risk_free_rate / 252
        excess_returns = returns_np - daily_rf_rate
        
        # Calculate Kelly fraction: f* = μ/σ²
        mean_excess = np.mean(excess_returns)
        variance = np.var(excess_returns, ddof=1)
        
        if variance <= 0:
            logger.warning("Zero or negative variance in returns")
            return 0.0
        
        kelly_fraction = mean_excess / variance
        
        # Apply leverage and weight constraints
        kelly_fraction = np.clip(
            kelly_fraction, 
            self.config.min_weight if self.config.allow_short else 0,
            self.config.max_weight * self.config.max_leverage
        )
        
        return float(kelly_fraction)
    
    def calculate_portfolio_kelly(
        self,
        returns_matrix: np.ndarray,
        names: Optional[List[str]] = None,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Kelly-optimal weights for portfolio of assets.
        
        Args:
            returns_matrix: Matrix where each column is asset returns
            names: Asset names (default: indices)
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of optimal weights by asset name
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate
        
        n_assets = returns_matrix.shape[1]
        n_observations = returns_matrix.shape[0]
        
        if names is None:
            names = [f"asset_{i}" for i in range(n_assets)]
        
        if n_observations < self.config.min_observations:
            logger.warning(f"Insufficient observations: {n_observations} < {self.config.min_observations}")
            return {name: 1.0/n_assets for name in names}  # Equal weights
        
        # Calculate excess returns
        daily_rf_rate = risk_free_rate / 252
        excess_returns = returns_matrix - daily_rf_rate
        
        # Calculate mean returns and covariance matrix
        mean_returns = np.mean(excess_returns, axis=0)
        cov_matrix = np.cov(excess_returns.T)
        
        # Add regularization to covariance matrix for stability
        regularized_cov = cov_matrix + self.config.regularization_lambda * np.eye(n_assets)
        
        try:
            # Kelly optimal weights: w* = Σ⁻¹μ
            inv_cov = np.linalg.inv(regularized_cov)
            kelly_weights = inv_cov @ mean_returns
            
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using equal weights")
            kelly_weights = np.ones(n_assets) / n_assets
        
        # Apply constraints if enabled
        if self.config.use_constraints:
            kelly_weights = self._apply_weight_constraints(kelly_weights)
        else:
            # Simple clipping
            if not self.config.allow_short:
                kelly_weights = np.maximum(kelly_weights, 0)
            kelly_weights = np.clip(kelly_weights, self.config.min_weight, self.config.max_weight)
        
        # Normalize weights to respect leverage constraint
        total_leverage = np.sum(np.abs(kelly_weights))
        if total_leverage > self.config.max_leverage:
            kelly_weights = kelly_weights * (self.config.max_leverage / total_leverage)
        
        return dict(zip(names, kelly_weights))
    
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply sophisticated weight constraints using optimization.
        
        Args:
            weights: Unconstrained Kelly weights
            
        Returns:
            Constrained weights
        """
        n_assets = len(weights)
        
        # Objective: minimize distance from unconstrained Kelly weights
        def objective(x):
            return np.sum((x - weights) ** 2)
        
        # Constraints
        constraints = []
        
        # Leverage constraint
        if self.config.max_leverage < np.inf:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.config.max_leverage - np.sum(np.abs(x))
            })
        
        # Individual weight bounds
        bounds = []
        for i in range(n_assets):
            if self.config.allow_short:
                lower = -self.config.max_weight
            else:
                lower = self.config.min_weight
            bounds.append((lower, self.config.max_weight))
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0=weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tolerance
                }
            )
            
            if result.success:
                return result.x
            else:
                logger.warning(f"Optimization failed: {result.message}")
                
        except Exception as e:
            logger.warning(f"Error in constraint optimization: {e}")
        
        # Fallback to simple clipping
        constrained_weights = np.clip(weights, 
                                     -self.config.max_weight if self.config.allow_short else self.config.min_weight,
                                     self.config.max_weight)
        
        # Ensure leverage constraint
        total_leverage = np.sum(np.abs(constrained_weights))
        if total_leverage > self.config.max_leverage:
            constrained_weights = constrained_weights * (self.config.max_leverage / total_leverage)
            
        return constrained_weights
    
    def calculate_fractional_kelly(
        self, 
        weights: Dict[str, float],
        fraction: float = 0.25
    ) -> Dict[str, float]:
        """
        Apply fractional Kelly sizing to reduce volatility.
        
        Args:
            weights: Full Kelly weights
            fraction: Fraction of Kelly to use (e.g., 0.25 for quarter-Kelly)
            
        Returns:
            Fractional Kelly weights
        """
        return {name: weight * fraction for name, weight in weights.items()}
    
    def calculate_dynamic_fraction(
        self,
        returns_matrix: np.ndarray,
        target_volatility: float = 0.15
    ) -> float:
        """
        Calculate dynamic Kelly fraction based on target volatility.
        
        Args:
            returns_matrix: Historical returns matrix
            target_volatility: Target portfolio volatility (annualized)
            
        Returns:
            Dynamic Kelly fraction
        """
        # Calculate portfolio volatility with full Kelly
        full_kelly_weights = self.calculate_portfolio_kelly(returns_matrix)
        weights_array = np.array(list(full_kelly_weights.values()))
        
        cov_matrix = np.cov(returns_matrix.T)
        portfolio_variance = weights_array.T @ cov_matrix @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
        
        if portfolio_volatility <= 0:
            return 1.0
        
        # Scale to target volatility
        fraction = target_volatility / portfolio_volatility
        return min(fraction, 1.0)  # Don't exceed full Kelly


class HierarchicalKellyOptimizer:
    """
    Hierarchical Kelly optimization for N-dimensional portfolio structures.
    Applies Kelly optimization at each level of the portfolio hierarchy.
    """
    
    def __init__(self, config: Optional[KellyConfig] = None):
        """Initialize hierarchical Kelly optimizer."""
        self.config = config or KellyConfig()
        self.optimizer = KellyOptimizer(config)
        self._optimization_history = []
        
    def optimize_portfolio(
        self, 
        portfolio: CompositePortfolio,
        prices: pl.Series,
        rebalance: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize weights for entire portfolio hierarchy using Kelly criterion.
        
        Args:
            portfolio: Root portfolio to optimize
            prices: Price data for optimization
            rebalance: Whether to apply optimized weights to portfolio
            
        Returns:
            Optimization results with new weights and performance metrics
        """
        logger.info(f"Starting hierarchical Kelly optimization for {portfolio.name}")
        
        optimization_results = {
            'portfolio_name': portfolio.name,
            'level_optimizations': [],
            'total_components_optimized': 0,
            'optimization_timestamp': None
        }
        
        # Recursively optimize each level of the hierarchy
        self._optimize_portfolio_level(portfolio, prices, optimization_results)
        
        # Apply rebalancing if requested
        if rebalance:
            self._apply_optimization_results(portfolio, optimization_results)
        
        # Store in history
        self._optimization_history.append(optimization_results)
        
        logger.info(f"Hierarchical optimization completed. Optimized {optimization_results['total_components_optimized']} components")
        return optimization_results
    
    def _optimize_portfolio_level(
        self,
        portfolio: CompositePortfolio,
        prices: pl.Series,
        results: Dict[str, Any],
        level: int = 0
    ) -> None:
        """Optimize weights at a specific portfolio level."""
        if not portfolio.component_list:
            return
        
        logger.debug(f"Optimizing level {level}: {portfolio.name}")
        
        # Calculate returns for each component
        component_returns = {}
        returns_matrix = []
        component_names = []
        
        for component in portfolio.component_list:
            try:
                returns = component.calculate_returns(prices)
                component_returns[component.name] = returns
                returns_matrix.append(returns.to_numpy())
                component_names.append(component.name)
            except Exception as e:
                logger.warning(f"Could not calculate returns for {component.name}: {e}")
                continue
        
        if not returns_matrix:
            logger.warning(f"No valid components for optimization at level {level}")
            return
        
        # Create returns matrix (observations x assets)
        returns_matrix = np.column_stack(returns_matrix)
        
        # Calculate Kelly optimal weights
        try:
            kelly_weights = self.optimizer.calculate_portfolio_kelly(
                returns_matrix, component_names
            )
        except Exception as e:
            logger.error(f"Kelly optimization failed for {portfolio.name}: {e}")
            # Use equal weights as fallback
            kelly_weights = {name: 1.0/len(component_names) for name in component_names}
        
        # Calculate performance metrics
        level_metrics = self._calculate_level_metrics(
            returns_matrix, kelly_weights, component_names
        )
        
        # Store level results
        level_result = {
            'level': level,
            'portfolio_name': portfolio.name,
            'original_weights': {comp.name: comp.weight for comp in portfolio.component_list},
            'kelly_weights': kelly_weights,
            'metrics': level_metrics,
            'n_components': len(component_names)
        }
        
        results['level_optimizations'].append(level_result)
        results['total_components_optimized'] += len(component_names)
        
        # Recursively optimize sub-portfolios
        for component in portfolio.component_list:
            if isinstance(component, CompositePortfolio):
                self._optimize_portfolio_level(component, prices, results, level + 1)
    
    def _calculate_level_metrics(
        self,
        returns_matrix: np.ndarray,
        weights: Dict[str, float],
        names: List[str]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a portfolio level."""
        weights_array = np.array([weights.get(name, 0.0) for name in names])
        
        # Portfolio returns
        portfolio_returns = returns_matrix @ weights_array
        
        # Calculate metrics
        try:
            metrics = {
                'expected_return': np.mean(portfolio_returns) * 252,
                'volatility': np.std(portfolio_returns) * np.sqrt(252),
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'kelly_leverage': np.sum(np.abs(weights_array))
            }
            
            if metrics['volatility'] > 0:
                excess_return = metrics['expected_return'] - self.config.risk_free_rate
                metrics['sharpe_ratio'] = excess_return / metrics['volatility']
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            metrics['max_drawdown'] = np.min(drawdown)
            
        except Exception as e:
            logger.warning(f"Error calculating level metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def _apply_optimization_results(
        self, 
        portfolio: CompositePortfolio, 
        results: Dict[str, Any]
    ) -> None:
        """Apply optimization results to portfolio weights."""
        for level_result in results['level_optimizations']:
            if level_result['portfolio_name'] == portfolio.name:
                # Apply weights to this level
                portfolio.rebalance_components(level_result['kelly_weights'])
                logger.info(f"Applied Kelly weights to {portfolio.name}")
                break
        
        # Recursively apply to sub-portfolios
        for component in portfolio.component_list:
            if isinstance(component, CompositePortfolio):
                self._apply_optimization_results(component, results)
    
    def schedule_rebalancing(
        self,
        portfolio: CompositePortfolio,
        prices: pl.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Schedule periodic rebalancing using Kelly optimization.
        
        Args:
            portfolio: Portfolio to rebalance
            prices: Price data
            start_date: Start date for rebalancing
            end_date: End date for rebalancing
            
        Returns:
            List of rebalancing events
        """
        rebalancing_events = []
        
        price_length = len(prices)
        rebalance_freq = self.config.rebalance_frequency
        lookback = self.config.lookback_window
        
        # Schedule rebalancing dates
        rebalance_dates = list(range(lookback, price_length, rebalance_freq))
        
        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                # Use lookback window for optimization
                start_idx = max(0, rebalance_date - lookback)
                optimization_prices = prices[start_idx:rebalance_date]
                
                # Run optimization
                optimization_result = self.optimize_portfolio(
                    portfolio, optimization_prices, rebalance=False
                )
                
                rebalance_event = {
                    'date_index': rebalance_date,
                    'rebalance_number': i + 1,
                    'optimization_result': optimization_result,
                    'lookback_start': start_idx,
                    'lookback_end': rebalance_date
                }
                
                rebalancing_events.append(rebalance_event)
                
            except Exception as e:
                logger.error(f"Rebalancing failed at date {rebalance_date}: {e}")
                continue
        
        logger.info(f"Scheduled {len(rebalancing_events)} rebalancing events")
        return rebalancing_events
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all optimization runs."""
        return self._optimization_history.copy()
    
    def calculate_kelly_growth_rate(
        self, 
        returns_matrix: np.ndarray,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate Kelly growth rate (expected log return) for given weights.
        
        Args:
            returns_matrix: Historical returns matrix
            weights: Portfolio weights
            
        Returns:
            Expected geometric growth rate
        """
        component_names = list(weights.keys())
        weights_array = np.array([weights[name] for name in component_names])
        
        # Calculate portfolio returns
        portfolio_returns = returns_matrix @ weights_array
        
        # Kelly growth rate is expected log return
        # Approximation: E[log(1 + r)] ≈ E[r] - 0.5 * Var[r]
        mean_return = np.mean(portfolio_returns)
        variance = np.var(portfolio_returns)
        
        growth_rate = mean_return - 0.5 * variance
        return float(growth_rate)
    
    def compare_allocations(
        self,
        portfolio: CompositePortfolio,
        prices: pl.Series,
        allocation_methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare different allocation methods (equal weight, Kelly, etc.).
        
        Args:
            portfolio: Portfolio to analyze
            prices: Price data
            allocation_methods: List of methods to compare
            
        Returns:
            Comparison results
        """
        if allocation_methods is None:
            allocation_methods = ['equal_weight', 'kelly', 'kelly_quarter', 'market_cap']
        
        comparison_results = {
            'methods': {},
            'rankings': {},
            'summary': {}
        }
        
        # Calculate returns for each component
        component_returns = {}
        for component in portfolio.component_list:
            try:
                returns = component.calculate_returns(prices)
                component_returns[component.name] = returns.to_numpy()
            except:
                continue
        
        if not component_returns:
            return comparison_results
        
        returns_matrix = np.column_stack(list(component_returns.values()))
        component_names = list(component_returns.keys())
        
        # Test each allocation method
        for method in allocation_methods:
            try:
                if method == 'equal_weight':
                    weights = {name: 1.0/len(component_names) for name in component_names}
                elif method == 'kelly':
                    weights = self.optimizer.calculate_portfolio_kelly(returns_matrix, component_names)
                elif method == 'kelly_quarter':
                    full_kelly = self.optimizer.calculate_portfolio_kelly(returns_matrix, component_names)
                    weights = self.optimizer.calculate_fractional_kelly(full_kelly, 0.25)
                elif method == 'market_cap':
                    # Simple proxy - could be enhanced with real market cap data
                    weights = {name: 1.0/len(component_names) for name in component_names}
                else:
                    continue
                
                # Calculate performance metrics
                weights_array = np.array([weights[name] for name in component_names])
                portfolio_returns = returns_matrix @ weights_array
                
                metrics = {
                    'total_return': np.sum(portfolio_returns),
                    'volatility': np.std(portfolio_returns) * np.sqrt(252),
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'kelly_growth_rate': self.calculate_kelly_growth_rate(returns_matrix, weights)
                }
                
                if metrics['volatility'] > 0:
                    excess_return = np.mean(portfolio_returns) * 252 - self.config.risk_free_rate
                    metrics['sharpe_ratio'] = excess_return / metrics['volatility']
                
                # Calculate max drawdown
                cumulative = np.cumsum(portfolio_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = cumulative - running_max
                metrics['max_drawdown'] = np.min(drawdown)
                
                comparison_results['methods'][method] = {
                    'weights': weights,
                    'metrics': metrics
                }
                
            except Exception as e:
                logger.warning(f"Failed to test allocation method {method}: {e}")
                
        return comparison_results