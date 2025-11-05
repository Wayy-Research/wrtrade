# WRTrade Local Deployment Guide

This guide shows how to deploy and manage trading strategies locally using WRTrade with Alpaca broker integration.

## Prerequisites

1. **Python 3.8+** installed
2. **WRTrade** installed with Alpaca support:
   ```bash
   pip install wrtrade[alpaca]
   ```
3. **Alpaca account** (paper trading recommended for testing)
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Get API keys from your dashboard

## Quick Start

### 1. Initialize WRTrade Workspace
```bash
wrtrade init
```
This creates a `~/.wrtrade` directory for storing configurations, logs, and data.

### 2. Set Up Alpaca Credentials

Create environment variables (recommended) or update the strategy file:

**Option A: Environment Variables (Recommended)**
```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
```

**Option B: Update Strategy File**
Edit the credentials section in `deployable_ma_strategy.py` with your actual keys.

### 3. Deploy a Strategy

**Method 1: Command Line Deployment**
```bash
wrtrade strategy deploy examples/deployable_ma_strategy.py \
    --name my_ma_strategy \
    --broker alpaca \
    --symbols AAPL,TSLA,MSFT \
    --max-position 1000 \
    --cycle-minutes 5 \
    --kelly-optimization \
    --auto-restart
```

**Method 2: Configuration File Deployment**
```bash
wrtrade strategy deploy examples/deployable_ma_strategy.py \
    --config-file examples/alpaca_strategy_config.yaml \
    --name my_ma_strategy
```

### 4. Start the Strategy
```bash
wrtrade strategy start my_ma_strategy
```

### 5. Monitor the Strategy
```bash
# Check status
wrtrade strategy status my_ma_strategy

# View logs (last 50 lines)
wrtrade strategy logs my_ma_strategy

# Follow logs in real-time
wrtrade strategy logs my_ma_strategy --follow
```

## Available Commands

### Strategy Management
```bash
# List all strategies
wrtrade strategy status

# Stop a strategy
wrtrade strategy stop my_ma_strategy

# Restart a strategy  
wrtrade strategy restart my_ma_strategy

# Remove a strategy
wrtrade strategy remove my_ma_strategy

# Remove strategy and all data
wrtrade strategy remove my_ma_strategy --remove-data
```

### Broker Management
```bash
# List available brokers
wrtrade broker list

# Check WRTrade version
wrtrade version
```

## Configuration Options

The strategy configuration supports these options:

### Core Settings
- `name`: Strategy name (must be unique)
- `description`: Strategy description
- `strategy_file`: Path to Python strategy file
- `broker_name`: Broker to use ("alpaca")
- `symbols`: List of trading symbols

### Risk Management
- `max_position_size`: Maximum dollar amount per position
- `risk_per_trade`: Risk percentage of account per trade
- `max_daily_trades`: Maximum trades per day
- `stop_loss_percent`: Stop loss percentage
- `take_profit_percent`: Take profit percentage

### Execution
- `cycle_interval_minutes`: How often to run strategy (minutes)
- `market_hours_only`: Only trade during market hours
- `enable_kelly_optimization`: Use Kelly criterion for position sizing
- `enable_permutation_testing`: Enable statistical validation

### Monitoring
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `auto_restart`: Automatically restart on failure
- `max_restarts`: Maximum restart attempts

## Strategy Development

### Creating a New Strategy

1. **Copy the example strategy**:
   ```bash
   cp examples/deployable_ma_strategy.py my_custom_strategy.py
   ```

2. **Modify the signal functions** in the `_build_portfolio()` method

3. **Test standalone**:
   ```bash
   python my_custom_strategy.py --dry-run --symbols AAPL,TSLA
   ```

4. **Deploy with WRTrade**:
   ```bash
   wrtrade strategy deploy my_custom_strategy.py --name my_custom --broker alpaca --symbols AAPL,TSLA
   ```

### Required Strategy Interface

Your strategy must implement these key methods:
- `__init__(self, config: StrategyConfig, broker_adapter: AlpacaBrokerAdapter)`
- `async def start(self)`: Main strategy loop
- `def stop(self)`: Graceful shutdown
- `async def run_strategy_cycle(self)`: Single strategy execution cycle

## File Locations

- **Workspace**: `~/.wrtrade/`
- **Strategy configs**: `~/.wrtrade/strategies/{strategy_name}/config.yaml`
- **Logs**: `~/.wrtrade/logs/{strategy_name}.log`
- **Data**: `~/.wrtrade/data/{strategy_name}/`
- **Manager logs**: `~/.wrtrade/logs/strategy_manager.log`

## Troubleshooting

### Common Issues

1. **Strategy won't start**
   ```bash
   # Check logs for errors
   wrtrade strategy logs my_ma_strategy
   
   # Verify broker credentials
   wrtrade broker list
   ```

2. **Authentication errors**
   - Verify API keys are correct
   - Check if using paper trading URL for paper account
   - Ensure environment variables are set correctly

3. **Strategy keeps restarting**
   ```bash
   # Check error logs
   wrtrade strategy logs my_ma_strategy
   
   # Check strategy status
   wrtrade strategy status my_ma_strategy
   ```

4. **No trades being placed**
   - Check if signals are being generated (logs show signal values)
   - Verify position sizing calculations
   - Check daily trade limits
   - Ensure sufficient buying power

### Getting Help

1. **Check logs**: Always check the strategy logs first
2. **Status information**: Use `wrtrade strategy status` for detailed info
3. **Broker connectivity**: Verify broker connection with `wrtrade broker list`
4. **Test standalone**: Run the strategy file directly to test logic

## Safety Recommendations

1. **Always start with paper trading**
2. **Test thoroughly before using real money**
3. **Start with small position sizes**
4. **Monitor strategies closely when first deployed**
5. **Set appropriate risk limits**
6. **Use stop losses and take profits**
7. **Never risk more than you can afford to lose**

## Next Steps

- **Paper Trading**: Test your strategies thoroughly with paper trading
- **Risk Management**: Adjust position sizing and risk parameters
- **Performance Analysis**: Monitor strategy performance and adjust as needed
- **Scaling Up**: Once comfortable, gradually increase position sizes
- **Multiple Strategies**: Deploy multiple strategies with different approaches

## Support

For issues or questions:
- Check the logs first
- Review this guide
- Test with paper trading
- Start with simple strategies before complex ones