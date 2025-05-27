import polars as pl
import numpy as np
from typing import Dict, Any, Optional
from wrtrade.trade import calculate_positions, calculate_returns, apply_take_profit_stop_loss
from wrtrade.metrics import calculate_all_metrics, calculate_all_rolling_metrics


class Portfolio:
    """
    Ultra-fast vectorized portfolio backtesting using Polars.
    """
    
    def __init__(
        self,
        prices: pl.Series,
        signals: pl.Series,
        max_position: float = float('inf'),
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ):
        """
        Initialize portfolio with price and signal data.
        
        Args:
            prices: Polars Series of market prices
            signals: Polars Series of trading signals (-1, 0, 1)
            max_position: Maximum absolute position allowed
            take_profit: Take profit threshold (e.g., 0.05 for 5%)
            stop_loss: Stop loss threshold (e.g., 0.02 for 2%)
        """
        self.prices = prices
        self.signals = signals
        self.max_position = max_position
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
        # Results (calculated on demand)
        self._positions = None
        self._returns = None
        self._cumulative_returns = None
        self._metrics = None
        
    def calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate all portfolio performance metrics in one vectorized operation.
        
        Returns:
            Dictionary containing all performance results
        """
        # Calculate positions from signals
        self._positions = calculate_positions(self.signals, self.max_position)
        
        # Apply take profit/stop loss if specified
        if self.take_profit is not None or self.stop_loss is not None:
            self._positions = apply_take_profit_stop_loss(
                self.prices, self._positions, self.signals, 
                self.take_profit, self.stop_loss
            )
        
        # Calculate returns
        self._returns = calculate_returns(self.prices, self._positions)
        self._cumulative_returns = self._returns.cum_sum()
        
        # Calculate market returns for benchmarking
        market_returns = self.prices.log().diff().fill_null(0)
        
        # Calculate all metrics
        self._metrics = calculate_all_metrics(self._returns)
        
        return {
            'total_return': self._cumulative_returns[-1],
            'positions': self._positions,
            'returns': self._returns,
            'cumulative_returns': self._cumulative_returns,
            'metrics': self._metrics
        }
    
    @property
    def positions(self) -> pl.Series:
        """Get position series (calculates if not already done)."""
        if self._positions is None:
            self.calculate_performance()
        return self._positions
    
    @property 
    def returns(self) -> pl.Series:
        """Get return series (calculates if not already done)."""
        if self._returns is None:
            self.calculate_performance()
        return self._returns
    
    @property
    def cumulative_returns(self) -> pl.Series:
        """Get cumulative return series (calculates if not already done).""" 
        if self._cumulative_returns is None:
            self.calculate_performance()
        return self._cumulative_returns
    
    @property
    def metrics(self) -> Dict[str, float]:
        """Get performance metrics (calculates if not already done)."""
        if self._metrics is None:
            self.calculate_performance()
        return self._metrics
    
    def plot_results(self):
        """Plot portfolio performance."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        # Convert to pandas for matplotlib compatibility
        prices_pd = self.prices.to_pandas()
        positions_pd = self.positions.to_pandas() 
        cumret_pd = self.cumulative_returns.to_pandas()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Prices
        ax1.plot(prices_pd.index, prices_pd.values, label='Price')
        ax1.set_ylabel('Price')
        ax1.set_title('Market Price')
        ax1.legend()
        
        # Positions  
        ax2.plot(positions_pd.index, positions_pd.values, label='Position')
        ax2.set_ylabel('Position')
        ax2.set_title('Portfolio Position')
        ax2.legend()
        
        # Cumulative returns
        ax3.plot(cumret_pd.index, cumret_pd.values, label='Portfolio Returns') 
        ax3.set_ylabel('Cumulative Returns')
        ax3.set_title('Portfolio Cumulative Returns')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()