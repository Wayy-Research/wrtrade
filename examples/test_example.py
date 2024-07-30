import pandas as pd
import wrtrade as wrt


p = wrt.Portfolio(initial_capital=10000)

market_data = pd.Series([150.0, 160.0, 170.0])
signals = pd.Series([1, -1, 1])

p.generate_trades(market_data, signals)
p.calculate_returns(market_data)