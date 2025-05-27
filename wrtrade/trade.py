import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from pandas import Series, DataFrame, Timestamp
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """
    A trade object that represents a trade in the market
    """

    dt: str
    price: float
    direction: int
    size: float = 0


class TradeEngine:
    def __init__(self, params: Dict[str, Any] = {}):
        self.params = params
        self.max_position = params.get("max_position", float("inf"))
        self.take_profit = params.get("take_profit")
        self.stop_loss = params.get("stop_loss")
        self.current_position = 0
        self.trades = {}  # Changed to dictionary
        self.position = None  # This will be a Series
        self.ts = Series(dtype=float)
        logger.info(f"Initialized TradeEngine with max_position: {self.max_position}")

    def process_signals(self, dates, prices: Series, signals: Series) -> None:
        # Ensure all series have the same index
        if isinstance(dates, pd.Index):
            common_index = dates.intersection(prices.index).intersection(signals.index)
        else:
            common_index = dates.index.intersection(prices.index).intersection(
                signals.index
            )

        if isinstance(dates, pd.Index):
            dates = pd.Series(index=dates, data=dates)

        dates = dates.loc[common_index]
        prices = prices.loc[common_index]
        signals = signals.loc[common_index]

        # Initialize position series
        self.position = pd.Series(0, index=common_index)
        self.ts = pd.Series(0, index=common_index)

        for i in range(len(signals)):
            signal = signals.iloc[i]
            date = dates.iloc[i]
            price = prices.iloc[i]

            # Calculate the desired position change
            # Only take a position if abs(current position + signal) <= max position
            if abs(self.current_position + signal) <= self.max_position:
                allowed_change = signal
            else:
                allowed_change = 0

            # Update position
            if allowed_change != 0:
                self.current_position += allowed_change
                self.position.iloc[i] = self.current_position
                self.ts.iloc[i] = self.current_position
                trade = Trade(
                    self._format_date(date),
                    price,
                    int(allowed_change),
                    abs(allowed_change),
                )
                self.trades[self._format_date(date)] = trade  # Use date as key
            else:
                self.position.iloc[i] = self.current_position
                self.ts.iloc[i] = self.current_position

        logger.info(f"Processed signals. Current position: {self.current_position}")

    def _format_date(self, dt):
        if isinstance(dt, (int, np.integer)):
            return str(dt)
        elif isinstance(dt, (pd.Timestamp, datetime.datetime)):
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return str(dt)

    def apply_take_profit_stop_loss(self, prices: Series) -> None:
        if self.take_profit is None and self.stop_loss is None:
            return

        # Iterate over the trades
        for date, trade in list(self.trades.items()):
            following_prices = prices.loc[prices.index > pd.to_datetime(trade.dt)]
            if len(following_prices) == 0:
                continue
            tp_exists = following_prices.gt(trade.price * (1 + self.take_profit)).any()
            sl_exists = following_prices.lt(trade.price * (1 - self.stop_loss)).any()

            tp = (following_prices * tp_exists).ne(0).idxmax()
            sl = (following_prices * sl_exists).ne(0).idxmax()

            if tp_exists or sl_exists:
                if min(tp, sl) == tp:
                    new_trade = Trade(
                        self._format_date(tp), tp.value, -trade.direction, trade.size
                    )
                    self.trades[self._format_date(tp)] = new_trade
                elif min(tp, sl) == sl:
                    new_trade = Trade(
                        self._format_date(sl), sl.value, -trade.direction, trade.size
                    )
                    self.trades[self._format_date(sl)] = new_trade

                self.position.loc[trade.dt:] -= trade.direction

    def get_position(self) -> Series:
        """Get the current position time series."""
        return self.position

    def get_trades(self) -> List[Trade]:
        """Get all executed trades."""
        return list(self.trades.values())

    def get_time_series(self) -> Series:
        """Get the time series of positions."""
        return self.ts

    def __iter__(self):
        return iter(self.trades.values())

    def __len__(self):
        return len(self.trades)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.trades[item]
        elif isinstance(item, int):
            return list(self.trades.values())[item]
        else:
            raise TypeError("Index must be a string (date) or an integer")
