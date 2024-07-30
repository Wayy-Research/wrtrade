import pandas as pd
from dataclasses import dataclass
from typing import Dict
from pandas import Series

@dataclass
class Trade:
    """
    A trade object that represents a trade in the market
    """
    dt: str
    price: float
    direction: int

class Trades:
    """
    A collection of trade objects
    """
    def __init__(self, params: Dict = {}) -> None:
        self.trades: list[Trade] = []
        self.ts: Series = Series()
        self.params = params


    def add_trades(self, market_data: Series, signals: Series) -> None:
        """
        Add a trade to the collection
        """
        for index, signal in signals.items():
            price = market_data.loc[index]
            if signal == 1:
                self.trades.append(Trade(index, price, signal))
            elif signal == -1:
                self.trades.append(Trade(index, price, signal))
    

    def create_ts(self, index: Series.index) -> None:
        """
        Create a time series of trades
        """
        self.ts = pd.Series(0, index=index)
        for trade in self.trades:
            self.ts.loc[trade.dt] = trade.direction


    def close_trades(self, market_data: Series, params: Dict[str, float]) -> None:
        """
        For signals without sells, close out the trades
        """
        for trade in self.trades:
            tmp = self.ts.loc[trade.dt:]
            for (i, val) in tmp.iloc[1:].items():
                if val <= trade.price - (trade.price * self.params['stop']):
                    self.ts[i] -= 1
                    break
                elif val >= trade.price + (trade.price * self.params['tp']):
                    self.ts[i] -= 1
                    break


    def __iter__(self):
        return iter(self.trades)


    def __len__(self):
        return len(self.trades)


    def __getitem__(self, item):
        return self.trades[item]