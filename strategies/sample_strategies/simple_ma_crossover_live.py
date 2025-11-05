"""
Simple Moving Average Crossover Strategy with Live Alpaca Trading.
This demonstrates integrating WRTrade portfolio components with real broker APIs.
"""

import asyncio
import sys
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from wrtrade.brokers_real import AlpacaBrokerAdapter, BrokerConfig, Order, OrderType, TimeInForce
from wrtrade.components import SignalComponent, CompositePortfolio
from wrtrade.kelly import KellyOptimizer, KellyConfig
from wrtrade.permutation import PermutationTester, PermutationConfig
from strategies.configs.alpaca_config import get_alpaca_credentials, BROKER_CONFIG, TRADING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTradingStrategy:
    """
    Live trading strategy that combines WRTrade components with real broker execution.
    """
    
    def __init__(self, broker_adapter: AlpacaBrokerAdapter, symbols: List[str]):
        """Initialize live trading strategy."""
        self.broker = broker_adapter
        self.symbols = symbols
        self.positions = {}
        self.last_signals = {}
        self.performance_history = []
        
        # Trading parameters
        self.max_position_size = TRADING_CONFIG['max_position_size']
        self.risk_per_trade = TRADING_CONFIG['risk_per_trade']
        
        # Create signal functions
        self.ma_fast_signal = self._create_ma_crossover_signal(fast_period=10, slow_period=30)
        self.ma_slow_signal = self._create_ma_crossover_signal(fast_period=20, slow_period=50)
        
        # Create portfolio components
        self.fast_ma_component = SignalComponent(
            "fast_ma_crossover", 
            self.ma_fast_signal, 
            weight=0.6
        )
        
        self.slow_ma_component = SignalComponent(
            "slow_ma_crossover", 
            self.ma_slow_signal, 
            weight=0.4
        )
        
        # Create composite portfolio
        self.portfolio = CompositePortfolio(
            "ma_crossover_portfolio",
            [self.fast_ma_component, self.slow_ma_component],
            weight=1.0,
            kelly_optimization=True
        )
        
        # Kelly optimizer for position sizing
        self.kelly_optimizer = KellyOptimizer(KellyConfig(
            min_weight=0.0,
            max_weight=1.0,
            max_leverage=1.0,
            risk_free_rate=0.02
        ))
        
    def _create_ma_crossover_signal(self, fast_period: int = 10, slow_period: int = 30):
        """Create moving average crossover signal function."""
        def signal_func(prices: pl.Series) -> pl.Series:
            if len(prices) < slow_period:
                return pl.Series([0] * len(prices))
            
            # Calculate moving averages
            fast_ma = prices.rolling_mean(window_size=fast_period)
            slow_ma = prices.rolling_mean(window_size=slow_period)
            
            # Generate crossover signals
            signals = []
            for i in range(len(prices)):
                if i < slow_period:
                    signals.append(0)
                else:
                    fast_current = fast_ma[i] if fast_ma[i] is not None else 0
                    slow_current = slow_ma[i] if slow_ma[i] is not None else 0
                    fast_prev = fast_ma[i-1] if i > 0 and fast_ma[i-1] is not None else 0
                    slow_prev = slow_ma[i-1] if i > 0 and slow_ma[i-1] is not None else 0
                    
                    # Bullish crossover
                    if fast_current > slow_current and fast_prev <= slow_prev:
                        signals.append(1)
                    # Bearish crossover
                    elif fast_current < slow_current and fast_prev >= slow_prev:
                        signals.append(-1)
                    else:
                        signals.append(0)
            
            return pl.Series(signals)
        
        return signal_func
    
    async def get_historical_prices(self, symbol: str, days: int = 100) -> pl.DataFrame:
        """Get historical price data for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            hist_data = await self.broker.get_historical_data(symbol, start_date, end_date)
            return hist_data
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pl.DataFrame()
    
    async def calculate_signals(self, symbol: str) -> Dict[str, float]:
        """Calculate trading signals for a symbol."""
        # Get historical data
        hist_data = await self.get_historical_prices(symbol)
        
        if hist_data.is_empty():
            logger.warning(f"No historical data for {symbol}")
            return {"composite_signal": 0.0, "fast_ma_signal": 0.0, "slow_ma_signal": 0.0}
        
        # Extract close prices
        if 'close' not in hist_data.columns:
            logger.warning(f"No close price data for {symbol}")
            return {"composite_signal": 0.0, "fast_ma_signal": 0.0, "slow_ma_signal": 0.0}
        
        prices = hist_data['close']
        
        # Generate portfolio signals
        portfolio_signals = self.portfolio.generate_signals(prices)
        
        # Get individual component signals
        fast_signals = self.fast_ma_component.generate_signals(prices)
        slow_signals = self.slow_ma_component.generate_signals(prices)
        
        # Return the latest signals
        return {
            "composite_signal": float(portfolio_signals[-1]) if len(portfolio_signals) > 0 else 0.0,
            "fast_ma_signal": float(fast_signals[-1]) if len(fast_signals) > 0 else 0.0,
            "slow_ma_signal": float(slow_signals[-1]) if len(slow_signals) > 0 else 0.0
        }
    
    async def calculate_position_size(self, symbol: str, signal_strength: float) -> int:
        """Calculate optimal position size using Kelly criterion."""
        try:
            # Get account info
            account_info = await self.broker.get_account_info()
            available_cash = account_info.cash
            
            # Get current market price
            market_data = await self.broker.get_market_data([symbol])
            if symbol not in market_data:
                return 0
            
            current_price = market_data[symbol].price
            
            # Calculate base position size from available cash and risk
            risk_amount = available_cash * self.risk_per_trade
            base_position_size = min(risk_amount / current_price, self.max_position_size / current_price)
            
            # Apply signal strength scaling
            scaled_position_size = base_position_size * abs(signal_strength)
            
            # Ensure we have at least 1 share if signal is strong enough
            if abs(signal_strength) > 0.5 and scaled_position_size < 1:
                scaled_position_size = 1
            
            return int(scaled_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    async def execute_trade(self, symbol: str, signal: float, position_size: int) -> bool:
        """Execute a trade based on signal."""
        if position_size == 0:
            logger.info(f"No trade for {symbol}: position size is 0")
            return False
        
        try:
            # Get current positions
            positions = await self.broker.get_positions()
            current_position = 0
            for pos in positions:
                if pos.symbol == symbol:
                    current_position = int(pos.quantity)
                    break
            
            # Determine trade action
            target_position = int(position_size * np.sign(signal))
            trade_quantity = target_position - current_position
            
            if abs(trade_quantity) < 1:
                logger.info(f"No trade needed for {symbol}: position already optimal")
                return False
            
            # Create and place order
            order = Order(
                symbol=symbol,
                quantity=abs(trade_quantity),
                side='buy' if trade_quantity > 0 else 'sell',
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Placing order: {order.side} {order.quantity} shares of {symbol}")
            success = await self.broker.place_order(order)
            
            if success:
                logger.info(f"Order placed successfully: {order.order_id}")
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': order.side,
                    'quantity': order.quantity,
                    'signal': signal,
                    'order_id': order.order_id
                })
                return True
            else:
                logger.error(f"Failed to place order for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    async def run_strategy_cycle(self):
        """Run one complete strategy cycle for all symbols."""
        logger.info(f"Running strategy cycle for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                logger.info(f"Processing {symbol}...")
                
                # Calculate signals
                signals = await self.calculate_signals(symbol)
                composite_signal = signals['composite_signal']
                
                # Skip if signal is too weak
                if abs(composite_signal) < 0.1:
                    logger.info(f"Signal too weak for {symbol}: {composite_signal:.3f}")
                    continue
                
                # Calculate position size
                position_size = await self.calculate_position_size(symbol, composite_signal)
                
                if position_size > 0:
                    logger.info(f"{symbol} - Signal: {composite_signal:.3f}, Position Size: {position_size}")
                    
                    # Execute trade if signal is strong enough
                    if abs(composite_signal) > 0.5:
                        success = await self.execute_trade(symbol, composite_signal, position_size)
                        if success:
                            logger.info(f"Trade executed successfully for {symbol}")
                    else:
                        logger.info(f"Signal not strong enough to trade {symbol}: {composite_signal:.3f}")
                
                # Small delay between symbols to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
    
    async def run_backtest(self, symbol: str, days: int = 252) -> Dict:
        """Run backtest on historical data."""
        logger.info(f"Running backtest for {symbol} over {days} days")
        
        try:
            # Get historical data
            hist_data = await self.get_historical_prices(symbol, days)
            if hist_data.is_empty():
                return {"error": "No historical data available"}
            
            prices = hist_data['close']
            
            # Generate signals and calculate returns
            self.portfolio.generate_signals(prices)
            portfolio_returns = self.portfolio.calculate_returns(prices)
            
            # Calculate performance metrics
            if portfolio_returns is not None and len(portfolio_returns) > 0:
                from wrtrade.metrics import calculate_all_metrics
                metrics = calculate_all_metrics(portfolio_returns)
                
                # Add additional backtest info
                metrics.update({
                    'total_return': float(portfolio_returns.sum()),
                    'num_trades': int((portfolio_returns != 0).sum()),
                    'win_rate': float((portfolio_returns > 0).sum() / len(portfolio_returns)),
                    'avg_return': float(portfolio_returns.mean()),
                    'period_days': days
                })
                
                return metrics
            else:
                return {"error": "Could not calculate returns"}
                
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of live trading."""
        if not self.performance_history:
            return {"message": "No trades executed yet"}
        
        total_trades = len(self.performance_history)
        buy_trades = sum(1 for trade in self.performance_history if trade['action'] == 'buy')
        sell_trades = total_trades - buy_trades
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'first_trade': self.performance_history[0]['timestamp'],
            'last_trade': self.performance_history[-1]['timestamp'],
            'symbols_traded': list(set(trade['symbol'] for trade in self.performance_history))
        }


async def main():
    """Main function to run the live trading strategy."""
    print("🚀 Live Trading Strategy with WRTrade + Alpaca")
    print("=" * 50)
    
    # Get credentials and create broker adapter
    credentials = get_alpaca_credentials(live_trading=False)
    adapter = AlpacaBrokerAdapter(credentials, BROKER_CONFIG)
    
    # Initialize strategy
    symbols = ['AAPL', 'TSLA', 'MSFT']  # Start with a few symbols
    strategy = LiveTradingStrategy(adapter, symbols)
    
    try:
        async with adapter:
            print("✅ Connected to Alpaca")
            
            # Get account info
            account_info = await adapter.get_account_info()
            print(f"💰 Account Value: ${account_info.portfolio_value:,.2f}")
            print(f"💵 Available Cash: ${account_info.cash:,.2f}")
            
            print("\n📊 Running backtests...")
            # Run backtests for each symbol
            for symbol in symbols:
                backtest_results = await strategy.run_backtest(symbol, days=60)
                if 'error' not in backtest_results:
                    print(f"\n{symbol} Backtest Results (60 days):")
                    print(f"  Total Return: {backtest_results.get('total_return', 0):.4f}")
                    print(f"  Sortino Ratio: {backtest_results.get('sortino_ratio', 0):.4f}")
                    print(f"  Max Drawdown: {backtest_results.get('max_drawdown', 0):.4f}")
                    print(f"  Win Rate: {backtest_results.get('win_rate', 0):.2%}")
                else:
                    print(f"\n{symbol} Backtest: {backtest_results['error']}")
            
            print("\n🎯 Starting live trading simulation...")
            print("Press Ctrl+C to stop")
            
            # Run strategy cycles
            cycle_count = 0
            while True:
                try:
                    cycle_count += 1
                    print(f"\n--- Strategy Cycle {cycle_count} ---")
                    
                    await strategy.run_strategy_cycle()
                    
                    # Show performance summary
                    summary = strategy.get_performance_summary()
                    if 'total_trades' in summary:
                        print(f"📈 Trades executed: {summary['total_trades']}")
                        print(f"📊 Symbols traded: {summary['symbols_traded']}")
                    
                    # Wait before next cycle (e.g., 5 minutes)
                    print("⏰ Waiting 5 minutes until next cycle...")
                    await asyncio.sleep(300)  # 5 minutes
                    
                except KeyboardInterrupt:
                    print("\n🛑 Stopping strategy...")
                    break
                except Exception as e:
                    logger.error(f"Error in strategy cycle: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
            
            # Final performance summary
            final_summary = strategy.get_performance_summary()
            print(f"\n📊 Final Performance Summary:")
            print(f"   Total Trades: {final_summary.get('total_trades', 0)}")
            if 'symbols_traded' in final_summary:
                print(f"   Symbols: {', '.join(final_summary['symbols_traded'])}")
            
    except Exception as e:
        logger.error(f"Strategy error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())