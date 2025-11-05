# wrtrade

Advanced portfolio trading framework with ultra-fast backtesting, N-dimensional portfolio composition, Kelly optimization, permutation testing, and live trading deployment.

## Features

### Core Backtesting
- ⚡ **Blazing Fast**: Built with Polars for 10-50x faster performance than pandas
- 🔧 **Simple API**: Minimal abstractions, maximum speed
- 📊 **Vectorized Operations**: No loops, pure vectorized calculations
- 📈 **Performance Metrics**: Volatility, Sortino ratio, Gain-to-Pain ratio, Max Drawdown, and more

### Advanced Portfolio Management
- 🏗️ **N-Dimensional Portfolios**: Build hierarchical portfolios with unlimited nesting depth
- 🧩 **Component System**: Compose portfolios from signal components and sub-portfolios
- ⚖️ **Dynamic Rebalancing**: Automatic portfolio rebalancing with configurable frequencies
- 📊 **Performance Attribution**: Track contribution of each component to total returns

### Optimization & Validation
- 🎯 **Kelly Criterion**: Optimal position sizing using Kelly optimization at each portfolio level
- 🔄 **Hierarchical Optimization**: Apply Kelly optimization across entire portfolio tree
- 🎲 **Permutation Testing**: Statistical validation to detect overfitting with p-values
- 📉 **Walk-Forward Analysis**: Out-of-sample testing with rolling windows

### Live Trading
- 🤝 **Multi-Broker Support**: Unified interface for Alpaca (with more brokers coming)
- 🚀 **One-Click Deployment**: Deploy validated strategies to live/paper trading
- 🛡️ **Risk Management**: Position sizing limits, daily loss limits, auto-shutdown conditions
- 📡 **Real-Time Monitoring**: Track performance, P&L, and risk metrics in real-time
- 🔐 **Paper Trading**: Test strategies safely before deploying with real capital

## Install

```bash
git clone https://github.com/Wayy-Research/wrtrade
cd wrtrade
pip install -e .
```

## Quick Start

### Basic Backtesting

```python
import polars as pl
import numpy as np
import wrtrade as wrt
from datetime import date

# Generate sample data
dates = pl.date_range(date(2022, 1, 1), date(2023, 12, 31), "1d", eager=True).alias("date")
prices = pl.Series('price', np.random.normal(100, 10, dates.len()))
signals = pl.Series('signal', np.random.choice([-1, 0, 1], dates.len()))

# Simple backtest
portfolio = wrt.Portfolio(prices, signals, max_position=5)
results = portfolio.calculate_performance()

# View performance metrics
wrt.tear_sheet(portfolio.returns)
```

### N-Dimensional Portfolio Composition

```python
import wrtrade as wrt

# Create signal functions
def ma_crossover(prices, fast=10, slow=30):
    fast_ma = prices.rolling_mean(fast)
    slow_ma = prices.rolling_mean(slow)
    return (fast_ma > slow_ma).cast(int) - (fast_ma < slow_ma).cast(int)

def momentum(prices, lookback=14):
    returns = prices.pct_change(lookback)
    return (returns > 0.02).cast(int) - (returns < -0.02).cast(int)

# Build portfolio with builder pattern
builder = wrt.NDimensionalPortfolioBuilder()

# Register signals
builder.register_signal("ma_fast", lambda p: ma_crossover(p, 5, 20))
builder.register_signal("ma_slow", lambda p: ma_crossover(p, 20, 50))
builder.register_signal("momentum", lambda p: momentum(p, 14))

# Create signal components
ma_fast = builder.create_signal_component("MA_Fast", "ma_fast", weight=0.4)
ma_slow = builder.create_signal_component("MA_Slow", "ma_slow", weight=0.3)
mom = builder.create_signal_component("Momentum", "momentum", weight=0.3)

# Create composite portfolio
portfolio = builder.create_portfolio(
    "Multi_Strategy",
    [ma_fast, ma_slow, mom],
    kelly_optimization=True
)

# Run backtest
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)

# View structure and performance
manager.print_structure()
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']:.2f}")
```

## Advanced Features

### Kelly Criterion Optimization

Optimize position sizing using Kelly criterion for maximum geometric growth:

```python
from wrtrade import HierarchicalKellyOptimizer, KellyConfig

# Configure Kelly optimization
kelly_config = KellyConfig(
    lookback_window=252,      # 1 year lookback
    max_leverage=1.0,         # No leverage
    rebalance_frequency=21,   # Monthly rebalancing
    risk_free_rate=0.02       # 2% risk-free rate
)

# Create optimizer
optimizer = HierarchicalKellyOptimizer(kelly_config)

# Optimize entire portfolio hierarchy
optimization_results = optimizer.optimize_portfolio(
    portfolio,
    prices,
    rebalance=True  # Apply optimized weights
)

# View optimization results
for level in optimization_results['level_optimizations']:
    print(f"Level: {level['portfolio_name']}")
    print(f"  Kelly Weights: {level['kelly_weights']}")
    print(f"  Expected Return: {level['metrics']['expected_return']:.2%}")
    print(f"  Sharpe Ratio: {level['metrics']['sharpe_ratio']:.2f}")
```

### Permutation Testing

Validate strategy significance and detect overfitting:

```python
from wrtrade import PermutationTester, PermutationConfig

# Configure permutation testing
perm_config = PermutationConfig(
    n_permutations=1000,     # Run 1000 permutations
    parallel=True,           # Use multiprocessing
    random_seed=42
)

# Create tester
tester = PermutationTester(perm_config)

# Define strategy function
def my_strategy(prices):
    return portfolio

# Run in-sample test
results = tester.run_insample_test(
    prices,
    my_strategy,
    metric='sortino_ratio'
)

print(f"P-value: {results['p_value']:.4f}")
print(f"Real Performance: {results['real_performance']:.3f}")
print(f"Permutation Mean: {results['permutation_mean']:.3f}")

if results['p_value'] < 0.05:
    print("✓ Strategy is statistically significant!")
else:
    print("✗ Strategy may be overfitted")

# Run walk-forward out-of-sample test
oos_results = tester.run_walkforward_test(
    prices,
    my_strategy,
    train_window=252,  # 1 year training
    metric='sortino_ratio'
)

print(f"Out-of-Sample P-value: {oos_results['p_value']:.4f}")
```

### Hierarchical Portfolio Structures

Build complex nested portfolios:

```python
# Create sub-portfolios
trend_portfolio = builder.create_portfolio(
    "Trend_Following",
    [ma_fast, ma_slow],
    weight=0.6
)

mean_reversion_portfolio = builder.create_portfolio(
    "Mean_Reversion",
    [rsi_component, bollinger_component],
    weight=0.4
)

# Create master portfolio combining sub-portfolios
master_portfolio = builder.create_portfolio(
    "Master_Strategy",
    [trend_portfolio, mean_reversion_portfolio],
    kelly_optimization=True
)

# Visualize structure
manager = wrt.AdvancedPortfolioManager(master_portfolio)
manager.print_structure()

# Output:
# === Portfolio Structure: Master_Strategy ===
#   Trend_Following (weight: 0.600)
#     MA_Fast (weight: 0.500)
#     MA_Slow (weight: 0.500)
#   Mean_Reversion (weight: 0.400)
#     RSI (weight: 0.500)
#     Bollinger (weight: 0.500)
```

## Live Trading Deployment

Deploy validated strategies to paper or live trading with full risk management:

```python
from wrtrade import WRTradeDeploymentSystem, DeploymentConfig

# Initialize deployment system
deployment_system = WRTradeDeploymentSystem()

# Configure deployment
deployment_config = DeploymentConfig(
    strategy_name="My_Strategy",
    broker_name="alpaca",
    paper_trading=True,  # Start with paper trading

    # Validation requirements
    min_sortino_ratio=1.0,
    max_permutation_pvalue=0.05,
    min_backtest_period_days=252,

    # Risk management
    max_position_size=0.1,     # 10% max per position
    max_daily_loss=0.05,       # 5% daily loss limit

    # Optimization
    kelly_optimization=True,
    rebalance_frequency_days=21,  # Monthly rebalancing

    # Auto-shutdown conditions
    auto_shutdown_conditions={
        'max_drawdown': -0.15,
        'daily_loss_limit': -0.05,
        'consecutive_losing_days': 5
    }
)

# Broker credentials (Alpaca)
broker_config = {
    'broker': 'alpaca',
    'api_key': 'YOUR_API_KEY',
    'secret_key': 'YOUR_SECRET_KEY'
}

# Symbol mapping (component name -> trading symbol)
symbols_map = {
    'MA_Fast': 'AAPL',
    'MA_Slow': 'AAPL',
    'Momentum': 'AAPL'
}

# Deploy strategy (automatically validates before deploying)
deployment_id = await deployment_system.deploy_portfolio_strategy(
    portfolio=portfolio,
    historical_prices=prices,
    broker_config=broker_config,
    symbols_map=symbols_map,
    deployment_config=deployment_config
)

print(f"Strategy deployed with ID: {deployment_id}")

# Monitor deployment status
status = deployment_system.deployer.get_deployment_status(deployment_id)
print(f"Status: {status.status}")
print(f"Daily P&L: ${status.daily_pnl:.2f}")
print(f"Total Return: {status.total_return:.2%}")

# Stop deployment if needed
deployment_system.deployer.stop_deployment(deployment_id, reason="Manual stop")
```

### Deployment Features

- **Automatic Validation**: Strategies are validated before deployment
  - Performance requirements (Sortino ratio, max drawdown)
  - Permutation testing for statistical significance
  - Portfolio structure validation

- **Risk Management**: Built-in risk controls
  - Position size limits
  - Daily loss limits
  - Auto-shutdown on risk breach

- **Real-Time Monitoring**: Track live performance
  - P&L tracking
  - Position monitoring
  - Risk alerts

- **Paper Trading**: Test safely before live deployment
  - Full simulation with real market data
  - No real capital at risk

## Complete Workflow Example

Here's a complete end-to-end workflow from strategy development to deployment:

```python
import wrtrade as wrt
import polars as pl

# 1. Define strategy signals
def trend_signal(prices):
    ma_fast = prices.rolling_mean(10)
    ma_slow = prices.rolling_mean(30)
    return (ma_fast > ma_slow).cast(int) - (ma_fast < ma_slow).cast(int)

# 2. Build portfolio
builder = wrt.NDimensionalPortfolioBuilder()
builder.register_signal("trend", trend_signal)
component = builder.create_signal_component("Trend", "trend", weight=1.0)
portfolio = builder.create_portfolio("My_Strategy", [component])

# 3. Backtest
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)
print(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']:.2f}")

# 4. Optimize with Kelly
optimizer = wrt.HierarchicalKellyOptimizer()
optimizer.optimize_portfolio(portfolio, prices, rebalance=True)

# 5. Validate with permutation testing
tester = wrt.PermutationTester(wrt.PermutationConfig(n_permutations=1000))
perm_results = tester.run_insample_test(prices, lambda p: portfolio, 'sortino_ratio')
print(f"P-value: {perm_results['p_value']:.4f}")

# 6. Deploy to paper trading
if perm_results['p_value'] < 0.05:
    deployment_system = wrt.WRTradeDeploymentSystem()
    deployment_id = await deployment_system.deploy_portfolio_strategy(
        portfolio=portfolio,
        historical_prices=prices,
        broker_config={'broker': 'alpaca', 'api_key': 'KEY', 'secret_key': 'SECRET'},
        symbols_map={'Trend': 'AAPL'}
    )
    print(f"✓ Strategy deployed: {deployment_id}")
```

## API Reference

### Core Classes

- **Portfolio**: Basic portfolio for single signal backtesting
- **SignalComponent**: Wraps a signal function as a portfolio component
- **CompositePortfolio**: Portfolio containing multiple components or sub-portfolios
- **NDimensionalPortfolioBuilder**: Builder for creating complex portfolio hierarchies
- **AdvancedPortfolioManager**: Manages backtesting and analysis for composite portfolios

### Optimization

- **KellyOptimizer**: Kelly criterion optimization for single or multi-asset portfolios
- **HierarchicalKellyOptimizer**: Kelly optimization across entire portfolio tree
- **KellyConfig**: Configuration for Kelly optimization parameters

### Validation

- **PermutationTester**: Statistical validation through permutation testing
- **PricePermutationGenerator**: Generates permutations preserving statistical properties
- **PermutationConfig**: Configuration for permutation testing

### Deployment

- **WRTradeDeploymentSystem**: Main deployment system
- **StrategyValidator**: Validates strategies before deployment
- **StrategyDeployer**: Deploys and manages live strategies
- **DeploymentConfig**: Configuration for strategy deployment

### Brokers

- **BrokerAdapter**: Abstract base class for broker integrations
- **AlpacaBrokerAdapter**: Alpaca Markets broker integration
- **BrokerFactory**: Factory for creating broker adapters
- **TradingSession**: Manages trading session with risk controls

## Performance Benefits

Compared to pandas-based frameworks:

- 🚀 **10-50x faster** execution on large datasets
- 💾 **Lower memory usage** with Polars' columnar format
- ⚡ **Vectorized operations** eliminate Python loops
- 🎯 **Parallel permutation testing** with multiprocessing

Perfect for:
- Complex multi-strategy portfolios
- Large-scale backtesting (millions of data points)
- Statistical validation and optimization
- Production trading deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Support

For issues or questions, please open an issue on GitHub.
