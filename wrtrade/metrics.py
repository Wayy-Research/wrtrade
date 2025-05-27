import polars as pl
import numpy as np
from typing import Dict, Callable


def volatility(returns: pl.Series) -> float:
    """Calculate annualized volatility of returns."""
    std_val = returns.std()
    if std_val is None:
        return 0.0
    return float(std_val * np.sqrt(252))


def sortino_ratio(returns: pl.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns.filter(excess_returns < 0)
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_deviation = downside_returns.std()
    if downside_deviation is None or downside_deviation == 0:
        return float('inf')
    
    excess_mean = excess_returns.mean()
    if excess_mean is None:
        return 0.0
    
    return float((excess_mean * 252) / (downside_deviation * np.sqrt(252)))


def gain_to_pain_ratio(returns: pl.Series) -> float:
    """Calculate gain to pain ratio (sum of gains / sum of losses)."""
    gains = returns.filter(returns > 0).sum()
    losses = returns.filter(returns < 0).sum()
    
    # Handle None values from sum()
    if gains is None:
        gains = 0.0
    if losses is None:
        losses = 0.0
    
    if losses == 0:
        return float('inf')
    
    return float(gains / abs(losses))


def max_drawdown(returns: pl.Series) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    cumulative = returns.cum_sum()
    running_max = cumulative.cum_max()
    drawdowns = cumulative - running_max
    return float(drawdowns.min())


def rolling_metric(returns: pl.Series, metric_func: Callable, window: int = 252) -> pl.Series:
    """
    Apply any metric function on a rolling basis - ultra fast with polars.
    
    Args:
        returns: Return series
        metric_func: Function that takes pl.Series and returns float
        window: Rolling window size
        
    Returns:
        Rolling metric series
    """
    def apply_metric(s):
        try:
            return metric_func(s)
        except:
            return np.nan
    
    return returns.rolling_map(apply_metric, window_size=window)


def rolling_volatility(returns: pl.Series, window: int = 252) -> pl.Series:
    """Fast rolling volatility using polars native operations."""
    rolling_std = returns.rolling_std(window_size=window)
    # Keep nulls for proper window behavior, only fill when multiplying
    return rolling_std * np.sqrt(252)


def rolling_max_drawdown(returns: pl.Series, window: int = 252) -> pl.Series:
    """Fast rolling max drawdown using polars native operations."""
    def calc_dd(s):
        cumulative = s.cum_sum()
        running_max = cumulative.cum_max()
        drawdowns = cumulative - running_max
        return float(drawdowns.min())
    
    return returns.rolling_map(calc_dd, window_size=window)


def calculate_all_metrics(returns: pl.Series) -> Dict[str, float]:
    """Calculate all key performance metrics."""
    return {
        'volatility': volatility(returns),
        'sortino_ratio': sortino_ratio(returns),
        'gain_to_pain_ratio': gain_to_pain_ratio(returns),
        'max_drawdown': max_drawdown(returns)
    }


def calculate_all_rolling_metrics(returns: pl.Series, window: int = 252) -> Dict[str, pl.Series]:
    """Calculate all metrics on rolling basis for plotting."""
    return {
        'rolling_volatility': rolling_volatility(returns, window),
        'rolling_sortino': rolling_metric(returns, sortino_ratio, window),
        'rolling_gain_to_pain': rolling_metric(returns, gain_to_pain_ratio, window),
        'rolling_max_drawdown': rolling_max_drawdown(returns, window)
    }


def tear_sheet(returns: pl.Series) -> None:
    """Print a formatted tear sheet of performance metrics."""
    metrics = calculate_all_metrics(returns)
    
    print("=" * 50)
    print("           PERFORMANCE TEAR SHEET")
    print("=" * 50)
    print()
    print(f"Volatility (Annualized):     {metrics['volatility']:.4f}")
    print(f"Sortino Ratio:               {metrics['sortino_ratio']:.4f}")
    print(f"Gain to Pain Ratio:          {metrics['gain_to_pain_ratio']:.4f}")
    print(f"Max Drawdown:                {metrics['max_drawdown']:.4f}")
    print()
    print("=" * 50)