import sys

sys.path.append('~/wayy-research/wrtrade/')

import pandas as pd
import numpy as np
import wrtrade as wrt
import matplotlib.pyplot as plt

# generate random market data and signals
index = pd.date_range(start='2022-01-01', end='2022-01-30')
market_data = pd.Series(index=index, data=np.random.normal(100, 10, size=30))
signals = pd.Series(index=index, data=np.random.choice([-1, 0,1], size=30))


p = wrt.Portfolio(market_data, signals) 
p.calculate_portfolio_returns()

# p.cumulative_returns.plot()
# plt.show()


p.get_results()
p.print_results()