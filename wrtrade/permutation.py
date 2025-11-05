import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from wrtrade.components import PortfolioComponent
from wrtrade.metrics import sortino_ratio, calculate_all_metrics


logger = logging.getLogger(__name__)


@dataclass
class PermutationConfig:
    """Configuration for permutation testing."""
    n_permutations: int = 1000
    start_index: int = 0
    preserve_gaps: bool = True
    parallel: bool = True
    n_workers: Optional[int] = None
    random_seed: Optional[int] = None


class PricePermutationGenerator:
    """
    Generates price permutations that preserve statistical properties
    while destroying temporal patterns, based on the methodology from
    permutation.md.
    """
    
    def __init__(self, config: Optional[PermutationConfig] = None):
        """Initialize permutation generator with configuration."""
        self.config = config or PermutationConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def generate_single_market_permutation(
        self, 
        prices: pl.Series,
        start_index: Optional[int] = None
    ) -> pl.Series:
        """
        Generate permutation of single market price data.
        
        Args:
            prices: Original price series (OHLC or close prices)
            start_index: Index to start permutation from
            
        Returns:
            Permuted price series with same statistical properties
        """
        if start_index is None:
            start_index = self.config.start_index
            
        prices_np = prices.to_numpy()
        n_bars = len(prices_np)
        
        if start_index >= n_bars:
            return prices.clone()
            
        # Convert to log prices for percentage calculations
        log_prices = np.log(prices_np)
        
        # Calculate relative prices (percentage moves from each bar's open)
        # For single series, we treat each price as close
        relative_prices = np.diff(log_prices)
        
        # Preserve first part of data before start_index
        permuted_log_prices = log_prices.copy()
        
        # Only permute data from start_index onwards
        if start_index > 0:
            # Keep original data before start_index unchanged
            permuted_log_prices[:start_index] = log_prices[:start_index]
        
        # Permute the relative price changes from start_index onwards only
        if start_index < n_bars - 1:
            changes_to_permute = relative_prices[start_index:]
            permuted_changes = np.random.permutation(changes_to_permute)
            
            # Reconstruct prices from permuted changes starting from start_index
            for i in range(len(permuted_changes)):
                permuted_log_prices[start_index + 1 + i] = (
                    permuted_log_prices[start_index] + np.sum(permuted_changes[:i+1])
                )
        
        # Convert back to normal prices
        permuted_prices = np.exp(permuted_log_prices)
        
        return pl.Series(permuted_prices)
    
    def generate_multi_market_permutation(
        self, 
        price_data: List[pl.Series],
        start_index: Optional[int] = None
    ) -> List[pl.Series]:
        """
        Generate permutation of multi-market price data preserving correlations.
        
        Args:
            price_data: List of price series for different markets
            start_index: Index to start permutation from
            
        Returns:
            List of permuted price series
        """
        if not price_data:
            return []
            
        if start_index is None:
            start_index = self.config.start_index
            
        n_markets = len(price_data)
        n_bars = len(price_data[0])
        
        # Verify all series have same length
        for i, series in enumerate(price_data):
            if len(series) != n_bars:
                raise ValueError(f"All price series must have same length. Series {i} has length {len(series)}, expected {n_bars}")
        
        # Convert all series to numpy arrays and log prices
        log_prices_all = []
        for series in price_data:
            prices_np = series.to_numpy()
            log_prices = np.log(prices_np)
            log_prices_all.append(log_prices)
        
        # Calculate relative price changes for all markets
        relative_changes_all = []
        for log_prices in log_prices_all:
            changes = np.diff(log_prices)
            relative_changes_all.append(changes)
        
        # Create permuted versions
        permuted_log_prices_all = [lp.copy() for lp in log_prices_all]
        
        # Permute changes while preserving cross-market relationships
        if start_index < n_bars - 1:
            # Get indices to permute
            indices_to_permute = list(range(start_index, n_bars - 1))
            permuted_indices = np.random.permutation(indices_to_permute)
            
            # Apply same permutation to all markets to preserve correlations
            for market_idx in range(n_markets):
                changes = relative_changes_all[market_idx]
                permuted_changes = changes[permuted_indices]
                
                # Reconstruct prices
                for i, change in enumerate(permuted_changes):
                    idx = start_index + 1 + i
                    permuted_log_prices_all[market_idx][idx] = (
                        permuted_log_prices_all[market_idx][idx - 1] + change
                    )
        
        # Convert back to normal prices and return as polars series
        permuted_series = []
        for log_prices in permuted_log_prices_all:
            normal_prices = np.exp(log_prices)
            permuted_series.append(pl.Series(normal_prices))
        
        return permuted_series
    
    def generate_ohlc_permutation(
        self, 
        ohlc_data: pl.DataFrame,
        start_index: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Generate permutation of OHLC data preserving intrabar relationships.
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close']
            start_index: Index to start permutation from
            
        Returns:
            Permuted OHLC DataFrame
        """
        if start_index is None:
            start_index = self.config.start_index
            
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in ohlc_data.columns:
                raise ValueError(f"OHLC data must contain column '{col}'")
        
        # Extract OHLC arrays
        opens = ohlc_data['open'].to_numpy()
        highs = ohlc_data['high'].to_numpy()
        lows = ohlc_data['low'].to_numpy()
        closes = ohlc_data['close'].to_numpy()
        
        n_bars = len(opens)
        log_opens = np.log(opens)
        
        # Calculate relative prices (percentage from open)
        rel_highs = np.log(highs) - log_opens
        rel_lows = np.log(lows) - log_opens  
        rel_closes = np.log(closes) - log_opens
        
        # Calculate gaps (open relative to previous close)
        gaps = np.zeros(n_bars)
        for i in range(1, n_bars):
            gaps[i] = log_opens[i] - np.log(closes[i-1])
        
        # Create permuted arrays
        perm_log_opens = log_opens.copy()
        perm_rel_highs = rel_highs.copy()
        perm_rel_lows = rel_lows.copy()
        perm_rel_closes = rel_closes.copy()
        perm_gaps = gaps.copy()
        
        # Permute intrabar data and gaps separately if beyond start_index
        if start_index < n_bars:
            # Permute intrabar relationships
            intrabar_indices = list(range(start_index, n_bars))
            perm_intrabar = np.random.permutation(intrabar_indices)
            
            perm_rel_highs[start_index:] = rel_highs[perm_intrabar]
            perm_rel_lows[start_index:] = rel_lows[perm_intrabar]
            perm_rel_closes[start_index:] = rel_closes[perm_intrabar]
            
            # Permute gaps separately if preserve_gaps is False
            if not self.config.preserve_gaps and start_index < n_bars - 1:
                gap_indices = list(range(start_index + 1, n_bars))
                perm_gap_indices = np.random.permutation(gap_indices)
                perm_gaps[start_index + 1:] = gaps[perm_gap_indices]
        
        # Reconstruct OHLC from permuted relative prices
        perm_opens = np.zeros(n_bars)
        perm_highs = np.zeros(n_bars)
        perm_lows = np.zeros(n_bars)
        perm_closes = np.zeros(n_bars)
        
        # First bar unchanged
        perm_opens[0] = opens[0]
        perm_highs[0] = np.exp(log_opens[0] + perm_rel_highs[0])
        perm_lows[0] = np.exp(log_opens[0] + perm_rel_lows[0])
        perm_closes[0] = np.exp(log_opens[0] + perm_rel_closes[0])
        
        # Subsequent bars
        for i in range(1, n_bars):
            if i < start_index:
                # Keep original
                perm_opens[i] = opens[i]
            else:
                # Use permuted gap
                perm_log_opens[i] = np.log(perm_closes[i-1]) + perm_gaps[i]
                perm_opens[i] = np.exp(perm_log_opens[i])
            
            # Calculate H, L, C from relative prices
            perm_highs[i] = np.exp(perm_log_opens[i] + perm_rel_highs[i])
            perm_lows[i] = np.exp(perm_log_opens[i] + perm_rel_lows[i])
            perm_closes[i] = np.exp(perm_log_opens[i] + perm_rel_closes[i])
        
        # Create permuted DataFrame
        permuted_df = pl.DataFrame({
            'open': perm_opens,
            'high': perm_highs,
            'low': perm_lows,
            'close': perm_closes
        })
        
        # Add any additional columns from original data
        for col in ohlc_data.columns:
            if col not in required_cols:
                permuted_df = permuted_df.with_columns(ohlc_data[col].alias(col))
        
        return permuted_df


class PermutationTester:
    """
    Performs statistical permutation tests on trading strategies
    to validate significance and detect overfitting.
    """
    
    def __init__(
        self, 
        config: Optional[PermutationConfig] = None,
        permutation_generator: Optional[PricePermutationGenerator] = None
    ):
        """Initialize permutation tester."""
        self.config = config or PermutationConfig()
        self.generator = permutation_generator or PricePermutationGenerator(config)
        
    def run_insample_test(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        metric: str = 'sortino_ratio',
        **strategy_kwargs
    ) -> Dict[str, Any]:
        """
        Run in-sample permutation test to detect overfitting.
        
        Args:
            prices: Price data (single series or list for multi-market)
            strategy_func: Function that takes prices and returns portfolio/signals
            metric: Performance metric to test ('sortino_ratio', 'gain_to_pain_ratio', etc.)
            **strategy_kwargs: Additional arguments for strategy function
            
        Returns:
            Dictionary with test results including p-value
        """
        logger.info(f"Starting in-sample permutation test with {self.config.n_permutations} permutations")
        
        # Calculate real strategy performance
        real_performance = self._calculate_strategy_performance(
            prices, strategy_func, metric, **strategy_kwargs
        )
        
        logger.info(f"Real strategy {metric}: {real_performance:.4f}")
        
        # Run permutation tests
        if self.config.parallel and self.config.n_permutations > 50:
            permutation_results = self._run_parallel_permutations(
                prices, strategy_func, metric, **strategy_kwargs
            )
        else:
            permutation_results = self._run_sequential_permutations(
                prices, strategy_func, metric, **strategy_kwargs
            )
        
        # Calculate p-value
        better_count = sum(1 for result in permutation_results if result >= real_performance)
        p_value = better_count / len(permutation_results)
        
        results = {
            'real_performance': real_performance,
            'metric': metric,
            'p_value': p_value,
            'permutation_results': permutation_results,
            'better_than_real': better_count,
            'total_permutations': len(permutation_results),
            'permutation_mean': np.mean(permutation_results),
            'permutation_std': np.std(permutation_results),
            'test_type': 'in_sample'
        }
        
        logger.info(f"Permutation test completed. P-value: {p_value:.4f}")
        return results
    
    def run_walkforward_test(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        train_window: int,
        metric: str = 'sortino_ratio',
        **strategy_kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward permutation test on out-of-sample data.
        
        Args:
            prices: Price data
            strategy_func: Function that takes prices and returns portfolio/signals
            train_window: Size of training window
            metric: Performance metric to test
            **strategy_kwargs: Additional arguments for strategy function
            
        Returns:
            Dictionary with test results including p-value
        """
        logger.info(f"Starting walk-forward permutation test")
        
        if isinstance(prices, list):
            total_length = len(prices[0])
        else:
            total_length = len(prices)
            
        if train_window >= total_length:
            raise ValueError("Train window must be smaller than total data length")
        
        # Use data after training window for testing
        test_start_idx = train_window
        
        # Calculate real walk-forward performance
        real_performance = self._calculate_walkforward_performance(
            prices, strategy_func, train_window, metric, **strategy_kwargs
        )
        
        logger.info(f"Real walk-forward {metric}: {real_performance:.4f}")
        
        # Run permutation tests on out-of-sample data only
        old_start_index = self.config.start_index
        self.config.start_index = test_start_idx
        self.generator.config.start_index = test_start_idx
        
        try:
            if self.config.parallel and self.config.n_permutations > 50:
                permutation_results = self._run_parallel_walkforward_permutations(
                    prices, strategy_func, train_window, metric, **strategy_kwargs
                )
            else:
                permutation_results = self._run_sequential_walkforward_permutations(
                    prices, strategy_func, train_window, metric, **strategy_kwargs
                )
        finally:
            # Restore original start index
            self.config.start_index = old_start_index
            self.generator.config.start_index = old_start_index
        
        # Calculate p-value
        better_count = sum(1 for result in permutation_results if result >= real_performance)
        p_value = better_count / len(permutation_results)
        
        results = {
            'real_performance': real_performance,
            'metric': metric,
            'p_value': p_value,
            'permutation_results': permutation_results,
            'better_than_real': better_count,
            'total_permutations': len(permutation_results),
            'permutation_mean': np.mean(permutation_results),
            'permutation_std': np.std(permutation_results),
            'train_window': train_window,
            'test_type': 'walk_forward'
        }
        
        logger.info(f"Walk-forward permutation test completed. P-value: {p_value:.4f}")
        return results
    
    def _calculate_strategy_performance(
        self, 
        prices: Union[pl.Series, List[pl.Series]], 
        strategy_func: Callable, 
        metric: str,
        **strategy_kwargs
    ) -> float:
        """Calculate strategy performance for given metric."""
        try:
            # Execute strategy function
            result = strategy_func(prices, **strategy_kwargs)
            
            # Extract returns based on result type
            if hasattr(result, 'calculate_returns'):
                # Portfolio component
                if isinstance(prices, list):
                    returns = result.calculate_returns(prices[0])  # Use first market for now
                else:
                    returns = result.calculate_returns(prices)
            elif hasattr(result, 'returns'):
                # Object with returns attribute
                returns = result.returns
            elif isinstance(result, pl.Series):
                # Direct return series
                returns = result
            else:
                raise ValueError(f"Unsupported strategy function result type: {type(result)}")
            
            # Calculate requested metric
            if metric == 'sortino_ratio':
                return sortino_ratio(returns)
            elif metric in ['gain_to_pain_ratio', 'volatility', 'max_drawdown']:
                metrics = calculate_all_metrics(returns)
                return metrics[metric]
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
        except Exception as e:
            logger.warning(f"Error calculating strategy performance: {e}")
            return 0.0
    
    def _calculate_walkforward_performance(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        train_window: int,
        metric: str,
        **strategy_kwargs
    ) -> float:
        """Calculate walk-forward performance using only out-of-sample data."""
        # This is a simplified version - full walk-forward would involve
        # retraining the strategy at each step
        try:
            result = strategy_func(prices, **strategy_kwargs)
            
            # Extract returns
            if hasattr(result, 'calculate_returns'):
                if isinstance(prices, list):
                    returns = result.calculate_returns(prices[0])
                else:
                    returns = result.calculate_returns(prices)
            elif hasattr(result, 'returns'):
                returns = result.returns
            elif isinstance(result, pl.Series):
                returns = result
            else:
                raise ValueError(f"Unsupported result type: {type(result)}")
            
            # Use only out-of-sample returns
            oos_returns = returns[train_window:]
            
            # Calculate metric on out-of-sample data
            if metric == 'sortino_ratio':
                return sortino_ratio(oos_returns)
            else:
                metrics = calculate_all_metrics(oos_returns)
                return metrics[metric]
                
        except Exception as e:
            logger.warning(f"Error calculating walk-forward performance: {e}")
            return 0.0
    
    def _run_sequential_permutations(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        metric: str,
        **strategy_kwargs
    ) -> List[float]:
        """Run permutation tests sequentially."""
        results = []
        
        for i in range(self.config.n_permutations):
            if i % 100 == 0:
                logger.info(f"Permutation {i}/{self.config.n_permutations}")
            
            # Generate permutation
            if isinstance(prices, list):
                perm_prices = self.generator.generate_multi_market_permutation(prices)
            else:
                perm_prices = self.generator.generate_single_market_permutation(prices)
            
            # Calculate performance on permuted data
            performance = self._calculate_strategy_performance(
                perm_prices, strategy_func, metric, **strategy_kwargs
            )
            results.append(performance)
        
        return results
    
    def _run_parallel_permutations(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        metric: str,
        **strategy_kwargs
    ) -> List[float]:
        """Run permutation tests in parallel."""
        n_workers = self.config.n_workers or min(mp.cpu_count(), self.config.n_permutations)
        
        def run_single_permutation(seed):
            # Create new generator with different seed for each worker
            local_generator = PricePermutationGenerator(
                PermutationConfig(random_seed=seed)
            )
            
            if isinstance(prices, list):
                perm_prices = local_generator.generate_multi_market_permutation(prices)
            else:
                perm_prices = local_generator.generate_single_market_permutation(prices)
            
            return self._calculate_strategy_performance(
                perm_prices, strategy_func, metric, **strategy_kwargs
            )
        
        seeds = np.random.randint(0, 10000, self.config.n_permutations)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(run_single_permutation, seeds))
        
        return results
    
    def _run_sequential_walkforward_permutations(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        train_window: int,
        metric: str,
        **strategy_kwargs
    ) -> List[float]:
        """Run walk-forward permutation tests sequentially."""
        results = []
        
        for i in range(self.config.n_permutations):
            if i % 50 == 0:
                logger.info(f"Walk-forward permutation {i}/{self.config.n_permutations}")
            
            # Generate permutation
            if isinstance(prices, list):
                perm_prices = self.generator.generate_multi_market_permutation(prices)
            else:
                perm_prices = self.generator.generate_single_market_permutation(prices)
            
            # Calculate walk-forward performance
            performance = self._calculate_walkforward_performance(
                perm_prices, strategy_func, train_window, metric, **strategy_kwargs
            )
            results.append(performance)
        
        return results
    
    def _run_parallel_walkforward_permutations(
        self,
        prices: Union[pl.Series, List[pl.Series]],
        strategy_func: Callable,
        train_window: int,
        metric: str,
        **strategy_kwargs
    ) -> List[float]:
        """Run walk-forward permutation tests in parallel."""
        # Similar to parallel implementation but with walk-forward logic
        return self._run_sequential_walkforward_permutations(
            prices, strategy_func, train_window, metric, **strategy_kwargs
        )