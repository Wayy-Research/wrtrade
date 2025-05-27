import numpy as np
import pandas as pd
from typing import Dict, Any

class Metrics:
    def __init__(self, returns: pd.Series, market_returns: pd.Series):
        self.returns = returns
        self.market_returns = market_returns
        self.frequency = self._infer_frequency()
        self.annualization_factor = self._get_annualization_factor()

    def _infer_frequency(self) -> str:
        """Infer the frequency of the returns data using pd.infer_freq."""
        if isinstance(self.returns.index, pd.DatetimeIndex):
            freq = pd.infer_freq(self.returns.index)
            if freq is None:
                # If pd.infer_freq fails, fallback to daily
                return 'D'
            return freq
        else:
            # If index is not DatetimeIndex, assume daily data
            return 'D'

    def _get_annualization_factor(self) -> float:
        """Get the annualization factor based on the inferred frequency."""
        freq_map = {
            'A': 1,     # Annual
            'Q': np.sqrt(4),  # Quarterly
            'M': np.sqrt(12),  # Monthly
            'W': np.sqrt(52),  # Weekly
            'D': np.sqrt(252),  # Daily
            'B': np.sqrt(252),  # Business Day
            'H': np.sqrt(252 * 24),  # Hourly (assuming 24-hour trading)
            'T': np.sqrt(252 * 24 * 60),  # Minutely
            'S': np.sqrt(252 * 24 * 60 * 60),  # Secondly
        }
        base_freq = self.frequency[0]  # Get the base frequency (first character)
        return freq_map.get(base_freq, np.sqrt(252))  # Default to daily if not found

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return various performance metrics.
        """
        metrics = {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'information_ratio': self.information_ratio(),
            'max_drawdown': self.max_drawdown(),
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'alpha': self.alpha(),
            'beta': self.beta()
        }
        return metrics

    def total_return(self) -> float:
        """Calculate the total return of the strategy."""
        return self.returns.iloc[-1]

    def annualized_return(self) -> float:
        """Calculate the annualized return of the strategy."""
        total_periods = len(self.returns)
        periods_per_year = (self.annualization_factor ** 2)
        total_years = total_periods / periods_per_year
        return (1 + self.total_return()) ** (1 / total_years) - 1

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate the Sharpe ratio of the strategy."""
        periods_per_year = (self.annualization_factor ** 2)
        excess_returns = self.returns - risk_free_rate / periods_per_year
        return self.annualization_factor * excess_returns.mean() / excess_returns.std()

    def sortino_ratio(self, risk_free_rate: float = 0.02, target_return: float = 0) -> float:
        """Calculate the Sortino ratio of the strategy."""
        periods_per_year = (self.annualization_factor ** 2)
        excess_returns = self.returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < target_return]
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        return (excess_returns.mean() / downside_deviation) * self.annualization_factor if downside_deviation != 0 else 0

    def information_ratio(self) -> float:
        """Calculate the Information ratio of the strategy."""
        excess_returns = self.returns - self.market_returns
        tracking_error = excess_returns.std()
        return (excess_returns.mean() / tracking_error) * self.annualization_factor if tracking_error != 0 else 0

    def max_drawdown(self) -> float:
        """Calculate the maximum drawdown of the strategy."""
        cumulative_returns = self.returns.cumsum()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = cumulative_returns - peak
        return drawdown.min()

    def win_rate(self) -> float:
        """Calculate the win rate of the strategy."""
        wins = (self.returns > 0).sum()
        total_trades = len(self.returns)
        return wins / total_trades if total_trades > 0 else 0

    def profit_factor(self) -> float:
        """Calculate the profit factor of the strategy."""
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return gains / losses if losses != 0 else np.inf

    def alpha(self) -> float:
        """Calculate the alpha of the strategy."""
        X = self.market_returns.values.reshape(-1, 1)
        y = self.returns.values
        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
        alpha = self.returns.mean() - (beta * self.market_returns.mean())
        return alpha * self.annualization_factor  # Annualized alpha

    def beta(self) -> float:
        """Calculate the beta of the strategy."""
        covariance = np.cov(self.returns, self.market_returns)[0][1]
        market_variance = np.var(self.market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    def rolling_sharpe_ratio(self, window: int = 252, risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate the rolling Sharpe ratio of the strategy."""
        periods_per_year = (self.annualization_factor ** 2)
        excess_returns = self.returns - risk_free_rate / periods_per_year
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        return self.annualization_factor * rolling_mean / rolling_std

    def rolling_sortino_ratio(self, window: int = 252, risk_free_rate: float = 0.02, target_return: float = 0) -> pd.Series:
        """Calculate the rolling Sortino ratio of the strategy."""
        periods_per_year = (self.annualization_factor ** 2)
        excess_returns = self.returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns.where(excess_returns < target_return, 0)
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_downside_std = np.sqrt((downside_returns ** 2).rolling(window=window).mean())
        return self.annualization_factor * rolling_mean / rolling_downside_std

    def rolling_information_ratio(self, window: int = 252) -> pd.Series:
        """Calculate the rolling Information ratio of the strategy."""
        excess_returns = self.returns - self.market_returns
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        return self.annualization_factor * rolling_mean / rolling_std
