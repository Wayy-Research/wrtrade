# wrtrade

A simple python backtesting library 

## Install

```
$ git clone https://github.com/Wayy-Research/wrtrade
$ cd wrtrade
$ pip install .
```

## Functionality

```
import wrtrade as wrt

market_data = pd.Series([150.0, 160.0, 170.0])
signals = pd.Series([1, -1, 1])

# args - market_data: pd.Series, signals: pd.Series
p = wrt.Portfolio(market_data, signals) 
p.calculate_portfolio_returns()
p.get_results()
```