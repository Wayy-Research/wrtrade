# This strategy uses the Bollinger Bands and the CCI to generate buy and sell signals.
# It is based on the idea that when the price is below the lower Bollinger Band
# and the CCI is below -100, it is a good time to buy.
# It is based on the idea that when the price is above the upper Bollinger Band
# and the CCI is above 100, it is a good time to sell.

# We will pull our data from alpaca

# This strategy will produce a long only signal, so we will use the `close_trades` parameter
# to close the trades at the end of the day.

# get data from alpaca

from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime


crypto_client = CryptoHistoricalDataClient()

request_params = CryptoBarsRequest(
    symbol_or_symbols="BTC/USD",
    timeframe=TimeFrame.Minute,
    start=datetime(2024, 6, 2),
    end=datetime(2024, 7, 1),
)

bars = crypto_client.get_crypto_bars(request_params)

# convert to dataframe
df = bars.df

# cleanup
df.reset_index(inplace=True)
df.index = pd.to_datetime(df["timestamp"])
df.drop(columns=["timestamp", "symbol", "trade_count", "vwap"], inplace=True)
df.head()

market_data = df["close"]

# BB
df["SMA"] = df["close"].rolling(window=200).mean()
df["SD"] = df["close"].rolling(window=200).std()
df["BBU"] = df.SMA + (df.SD * 2)
df["BBL"] = df.SMA + (df.SD * 2)

# CCI
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
df["TPMA"] = df.rolling(window=200)["typical_price"].mean()
df["mean_devation"] = (df.typical_price - df.TPMA).abs().mean()
df["CCI"] = (df.typical_price - df.TPMA) / (0.015 * df.mean_devation)

df.dropna(inplace=True)

# make signals
from pandas import Series, DataFrame
from wrtrade.strategy import Strategy


class BBCCI(Strategy):

    def generate_signals(self, market_data: DataFrame) -> Series:
        """Returns a series of values in {1, 0, -1} for buy, hold, sell"""

        # generate entries
        signals = (
            (market_data.CCI < 200) & (market_data.close < market_data.BBL)
        ).astype(int)

        return signals


strat = BBCCI()
signals = strat.generate_signals(df)

from wrtrade.portfolio import Portfolio

p = Portfolio(market_data, signals)
p.calculate_portfolio_returns()
p.get_results()
