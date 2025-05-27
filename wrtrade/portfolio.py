import pandas as pd
import numpy as np
import pprint
from typing import Dict, Any
from pandas import Series
from wrtrade.trade import TradeEngine
from wrtrade.metrics import Metrics


class Portfolio:
    def __init__(
        self,
        market_data: Series,
        signals: Series,
        close_trades: bool = False,
        params: Dict = {"max_position": 10, "take_profit": 0.05, "stop_loss": 0.02},
    ):
        self.market_data = market_data
        self.signals = signals
        self.close_trades = close_trades
        self.params = params
        self.trade_engine = TradeEngine(params)
        self.position = Series(index=market_data.index, dtype=float)
        self.cumulative_returns = Series(index=market_data.index, dtype=float)
        self.market_returns = Series(index=market_data.index, dtype=float)
        self.metrics = None
        self.results = None

    def calculate_market_returns(self) -> None:
        """Calculate the returns of the market based on the current market data."""
        self.market_returns = np.log(self.market_data).diff().fillna(0)

    def calculate_portfolio_returns(self) -> None:
        """Calculate signal returns based on the current market data."""
        self.calculate_market_returns()

        # Process signals using the vectorized TradeEngine
        self.trade_engine.process_signals(
            self.market_data.index, self.market_data, self.signals
        )

        # Apply take profit and stop loss
        if self.close_trades:
            self.trade_engine.apply_take_profit_stop_loss(
                self.market_data.index, self.market_data
            )

        # Get the position time series from the trade engine
        self.position = self.trade_engine.get_position()

        # Calculate the returns of the portfolio
        self.cumulative_returns = (
            self.position.shift(1) * self.market_returns
        ).cumsum()

    def get_results(self) -> Dict[str, Any]:
        """
        Calculate and return the results of the backtest.

        :return: Dictionary containing performance metrics
        """
        self.metrics = Metrics(self.cumulative_returns, self.market_returns)

        self.results = {
            "cumulative_return": self.cumulative_returns.iloc[-1],
            "trades": pd.DataFrame(
                [vars(trade) for trade in self.trade_engine.get_trades()]
            ),
            "position": self.position,
            "cumulative_returns": self.cumulative_returns,
            "metrics": self.metrics.calculate_metrics(),
        }

        return self.results
    
    def print_results(self) -> None:
        """
        Print the results of the backtest.
        """
        pp = pprint.PrettyPrinter(depth=2)
        pp.pprint(self.results)

    def plot_results(self) -> None:
        """
        Plot the results of the backtest.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Plot market data
        ax1.plot(self.market_data.index, self.market_data, label="Market Price")
        ax1.set_ylabel("Price")
        ax1.set_title("Market Price")
        ax1.legend()

        # Plot position
        ax2.plot(self.position.index, self.position, label="Position")
        ax2.set_ylabel("Position")
        ax2.set_title("Portfolio Position")
        ax2.legend()

        # Plot cumulative returns
        ax3.plot(
            self.cumulative_returns.index,
            self.cumulative_returns,
            label="Portfolio Returns",
        )
        ax3.set_ylabel("Cumulative Returns")
        ax3.set_title("Portfolio Cumulative Returns")
        ax3.legend()

        plt.tight_layout()
        plt.show()
