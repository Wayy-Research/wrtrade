from typing import Dict, Any
import pandas as pd

from wrtrade.portfolio import Portfolio
from wrtrade.strategy import Strategy

class Backtest:
    def __init__(self, market_data: pd.Series, strategy: Strategy, initial_capital: float):
        self.market_data = market_data
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        """
        signals = self.strategy.generate_signals(self.market_data)
        self.portfolio.generate_trades(self.market_data, signals)
        self.portfolio.calculate_returns(self.market_data)

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the backtest.
        """
        return self.portfolio.get_results() 