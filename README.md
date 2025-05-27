# wrtrade

Ultra-fast Python backtesting library built with Polars for maximum performance on large datasets.

## Features

- ⚡ **Blazing Fast**: Built with Polars for 10x+ faster performance than pandas
- 🔧 **Simple API**: Minimal abstractions, maximum speed
- 📊 **Vectorized Operations**: No loops, pure vectorized calculations
- 📈 **Rolling Metrics**: Real-time rolling performance calculations for plotting
- 🎯 **Key Metrics**: Volatility, Sortino ratio, Gain-to-Pain ratio, Max Drawdown

## Install

```bash
git clone https://github.com/Wayy-Research/wrtrade
cd wrtrade
pip install .
```

## Quick Start

```python
import polars as pl
import numpy as np
import wrtrade as wrt

# Generate sample data with Polars
dates = pl.date_range(start='2022-01-01', end='2023-12-31', interval='1d')
prices = pl.Series('price', np.random.normal(100, 10, len(dates)))
signals = pl.Series('signal', np.random.choice([-1, 0, 1], len(dates)))

# Create portfolio and run backtest
portfolio = wrt.Portfolio(prices, signals, max_position=5)
results = portfolio.calculate_performance()

# View performance metrics
wrt.tear_sheet(portfolio.returns)

# Plot results
portfolio.plot_results()
```

## API Reference

### Portfolio Class

```python
Portfolio(
    prices: pl.Series,        # Price data
    signals: pl.Series,       # Trading signals (-1, 0, 1)
    max_position: float = inf,# Max position size
    take_profit: float = None,# Take profit threshold
    stop_loss: float = None   # Stop loss threshold
)
```

### Key Methods

```python
# Calculate all performance metrics
results = portfolio.calculate_performance()

# Access individual components
positions = portfolio.positions
returns = portfolio.returns
cumulative_returns = portfolio.cumulative_returns
metrics = portfolio.metrics

# Print formatted results
wrt.tear_sheet(portfolio.returns)

# Calculate rolling metrics for plotting
rolling_metrics = wrt.calculate_all_rolling_metrics(portfolio.returns, window=252)
```

### Performance Metrics

- **Volatility**: Annualized volatility of returns
- **Sortino Ratio**: Risk-adjusted return using downside deviation
- **Gain-to-Pain Ratio**: Sum of gains divided by sum of losses
- **Max Drawdown**: Maximum peak-to-trough decline

## Advanced Usage

### Custom Signal Generation

```python
# Moving average crossover example
def ma_crossover_signals(prices: pl.Series, short_window: int = 20, long_window: int = 50):
    short_ma = prices.rolling_mean(short_window)
    long_ma = prices.rolling_mean(long_window)
    
    signals = pl.when(short_ma > long_ma).then(1).otherwise(
        pl.when(short_ma < long_ma).then(-1).otherwise(0)
    )
    return signals

signals = ma_crossover_signals(prices)
portfolio = wrt.Portfolio(prices, signals)
```

### Rolling Metrics Analysis

```python
# Calculate 1-year rolling metrics
rolling_metrics = wrt.calculate_all_rolling_metrics(portfolio.returns, window=252)

# Plot rolling Sortino ratio
import matplotlib.pyplot as plt
rolling_sortino = rolling_metrics['rolling_sortino'].to_pandas()
plt.plot(rolling_sortino)
plt.title('Rolling Sortino Ratio')
plt.show()
```

### Multiple Strategy Comparison

```python
strategies = {}
for strategy_name, signals in signal_dict.items():
    portfolio = wrt.Portfolio(prices, signals)
    portfolio.calculate_performance()
    strategies[strategy_name] = portfolio.metrics

# Compare strategies
for name, metrics in strategies.items():
    print(f"{name}: Sortino = {metrics['sortino_ratio']:.2f}")
```

## Performance Benefits

Compared to pandas-based backtesting frameworks:

- 🚀 **10-50x faster** execution on large datasets
- 💾 **Lower memory usage** with Polars' columnar format
- ⚡ **Vectorized operations** eliminate Python loops
- 🔧 **Simple, focused API** reduces overhead

Perfect for:
- High-frequency strategy development
- Large dataset backtesting (millions of data points)
- Parameter optimization across multiple strategies
- Real-time strategy evaluation

## License

MIT License - see LICENSE file for details.
