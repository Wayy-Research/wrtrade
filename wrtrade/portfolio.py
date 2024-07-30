import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from typing import Dict, Any

from wrtrade.trade import Trades

class Portfolio:
    def __init__(self,
                 market_data: Series, 
                 signals: Series, 
                 params: Dict[str, float] = {'stop': 0.025, 'tp': 0.07}) -> None:
        self.market_data = market_data
        self.signals = signals
        self.params = params
        self.market_returns = []
        self.trades = Trades(params=self.params)
        self.position = []
        self.cumulative_returns = []
        self.notional = []



    def calculate_market_returns(self) -> None:
        """
        Calculate the returns of the market based on the current market data.
        """
        self.market_returns = (np.log(self.market_data) - np.log(self.market_data.shift(1))).fillna(0)


    def calculate_portfolio_returns(self) -> None:
        """
        Calculate signal returns based on the current market data.
        
        :param market_data: Current market data
        """
        # generate trades
        self.trades.add_trades(self.market_data, self.signals)
        self.trades.create_ts(self.market_data.index)

        # close trades
        if np.array_equal(self.trades.ts.unique(), np.array([0, 1])):
            self.trades.close_trades(self.market_data, self.params)

        # calculate market returns
        self.calculate_market_returns()

        #generate positions
        self.position = self.trades.ts.cumsum()

        # Calculate the returns of the portfolio
        self.cumulative_returns = (self.position * self.market_returns).cumsum()


    def get_results(self) -> Dict[str, Any]:
        """
        Calculate and return the results of the backtest.

        :return: Dictionary containing performance metrics
        """
        return {
            'cumulative_return': self.cumulative_returns.iloc[-1],
            'trades': pd.DataFrame(self.trades.trades)
        }