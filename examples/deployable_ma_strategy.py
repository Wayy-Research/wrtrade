#!/usr/bin/env python3
"""
Deployable Moving Average Crossover Strategy for WRTrade.
This strategy can be deployed locally using the WRTrade CLI with Alpaca broker.

Usage:
    # Deploy the strategy
    wrtrade strategy deploy examples/deployable_ma_strategy.py \
        --name ma_crossover \
        --broker alpaca \
        --symbols AAPL,TSLA,MSFT \
        --max-position 1000 \
        --cycle-minutes 5

    # Start the strategy
    wrtrade strategy start ma_crossover
    
    # Monitor the strategy
    wrtrade strategy status ma_crossover
    wrtrade strategy logs ma_crossover
"""

import asyncio
import sys
import os
import argparse
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# WRTrade imports
from wrtrade.brokers_real import AlpacaBrokerAdapter, BrokerConfig
from wrtrade.components import SignalComponent, CompositePortfolio
from wrtrade.kelly import KellyOptimizer, KellyConfig
from wrtrade.local_deploy import StrategyConfig
import polars as pl
import numpy as np


class DeployableMAStrategy:
    """
    Deployable Moving Average Crossover Strategy.
    This class implements a complete trading strategy that can be deployed
    and managed by the WRTrade CLI system.
    """
    
    def __init__(self, config: StrategyConfig, broker_adapter: AlpacaBrokerAdapter):
        """Initialize the strategy."""
        self.config = config
        self.broker = broker_adapter
        self.logger = self._setup_logging()
        
        # Strategy state
        self.is_running = False
        self.cycle_count = 0
        self.last_signals = {}
        self.positions = {}
        self.performance_stats = {
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_trades': 0,
            'loss_trades': 0,
            'start_time': datetime.now()
        }
        
        # Create trading components
        self._build_portfolio()
        
        # Kelly optimizer for position sizing
        if config.enable_kelly_optimization:
            self.kelly_optimizer = KellyOptimizer(KellyConfig(
                min_weight=0.0,
                max_weight=1.0,
                max_leverage=1.0,
                risk_free_rate=0.02
            ))
        else:
            self.kelly_optimizer = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup strategy-specific logging."""
        logger = logging.getLogger(f"Strategy.{self.config.name}")
        
        # Set log level from config
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler for when running standalone
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _build_portfolio(self):
        """Build the trading portfolio with signal components."""
        
        # Create signal functions
        fast_ma_signal = self._create_ma_crossover_signal(10, 30)
        slow_ma_signal = self._create_ma_crossover_signal(20, 50)
        momentum_signal = self._create_momentum_signal(14, 0.02)
        
        # Create signal components
        self.fast_ma_component = SignalComponent(
            "fast_ma_crossover", 
            fast_ma_signal, 
            weight=0.4
        )
        
        self.slow_ma_component = SignalComponent(
            "slow_ma_crossover", 
            slow_ma_signal, 
            weight=0.3
        )
        
        self.momentum_component = SignalComponent(
            "momentum_signal",
            momentum_signal,
            weight=0.3
        )
        
        # Create composite portfolio
        self.portfolio = CompositePortfolio(
            f"{self.config.name}_portfolio",
            [self.fast_ma_component, self.slow_ma_component, self.momentum_component],
            weight=1.0,
            kelly_optimization=self.config.enable_kelly_optimization
        )
        
        self.logger.info(f"Portfolio built with {len(self.portfolio.component_list)} components")
    
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
    
    def _create_momentum_signal(self, lookback: int = 14, threshold: float = 0.02):
        """Create momentum signal function."""
        def signal_func(prices: pl.Series) -> pl.Series:
            if len(prices) < lookback:
                return pl.Series([0] * len(prices))
            
            signals = []
            for i in range(len(prices)):
                if i < lookback:
                    signals.append(0)
                else:
                    current_price = prices[i]
                    past_price = prices[i - lookback]
                    momentum = (current_price - past_price) / past_price
                    
                    if momentum > threshold:
                        signals.append(1)
                    elif momentum < -threshold:
                        signals.append(-1)
                    else:
                        signals.append(0)
            
            return pl.Series(signals)
        
        return signal_func
    
    async def get_historical_data(self, symbol: str, days: int = 60) -> pl.DataFrame:
        """Get historical price data for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            hist_data = await self.broker.get_historical_data(symbol, start_date, end_date)
            return hist_data
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pl.DataFrame()
    
    async def calculate_signals(self, symbol: str) -> Dict[str, float]:
        """Calculate trading signals for a symbol."""
        hist_data = await self.get_historical_data(symbol)
        
        if hist_data.is_empty() or 'close' not in hist_data.columns:
            self.logger.warning(f"No valid historical data for {symbol}")
            return {"signal": 0.0, "strength": 0.0}
        
        prices = hist_data['close']
        
        # Generate portfolio signals
        portfolio_signals = self.portfolio.generate_signals(prices)
        latest_signal = float(portfolio_signals[-1]) if len(portfolio_signals) > 0 else 0.0
        
        # Calculate signal strength (based on component agreement)
        individual_signals = []
        for component in self.portfolio.component_list:
            comp_signals = component.generate_signals(prices)
            if len(comp_signals) > 0:
                individual_signals.append(float(comp_signals[-1]))
        
        # Signal strength is based on agreement between components
        signal_strength = abs(np.mean(individual_signals)) if individual_signals else 0.0
        
        return {
            "signal": latest_signal,
            "strength": signal_strength,
            "components": {
                "fast_ma": individual_signals[0] if len(individual_signals) > 0 else 0.0,
                "slow_ma": individual_signals[1] if len(individual_signals) > 1 else 0.0,
                "momentum": individual_signals[2] if len(individual_signals) > 2 else 0.0,
            }
        }
    
    async def calculate_position_size(self, symbol: str, signal_strength: float) -> int:
        """Calculate position size using risk management."""
        try:
            account_info = await self.broker.get_account_info()
            available_cash = account_info.cash
            
            # Get current market price
            market_data = await self.broker.get_market_data([symbol])
            if symbol not in market_data:
                return 0
            
            current_price = market_data[symbol].price
            
            # Base position size from risk management
            risk_amount = available_cash * self.config.risk_per_trade
            max_shares_by_risk = risk_amount / current_price
            max_shares_by_position = self.config.max_position_size / current_price
            
            base_shares = min(max_shares_by_risk, max_shares_by_position)
            
            # Apply signal strength scaling
            scaled_shares = base_shares * signal_strength
            
            # Minimum position of 1 share if signal is strong
            if signal_strength > 0.7 and scaled_shares < 1:
                scaled_shares = 1
            
            return int(scaled_shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    async def execute_trade(self, symbol: str, signal: float, position_size: int) -> bool:
        """Execute trade if conditions are met."""
        if position_size == 0 or abs(signal) < 0.1:
            return False
        
        # Check daily trade limit
        if self.performance_stats['total_trades'] >= self.config.max_daily_trades:
            self.logger.info(f"Daily trade limit reached ({self.config.max_daily_trades})")
            return False
        
        try:
            # Get current positions
            positions = await self.broker.get_positions()
            current_position = 0
            
            for pos in positions:
                if pos.symbol == symbol:
                    current_position = int(pos.quantity)
                    break
            
            # Determine target position and trade needed
            target_position = int(position_size * np.sign(signal))
            trade_quantity = target_position - current_position
            
            if abs(trade_quantity) < 1:
                return False
            
            # Create order
            from wrtrade.brokers_real import Order, OrderType, TimeInForce
            
            order = Order(
                symbol=symbol,
                quantity=abs(trade_quantity),
                side='buy' if trade_quantity > 0 else 'sell',
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            self.logger.info(f"Placing order: {order.side} {order.quantity} {symbol} (signal: {signal:.3f})")
            
            success = await self.broker.place_order(order)
            
            if success:
                self.performance_stats['total_trades'] += 1
                self.logger.info(f"Order executed: {order.order_id}")
                return True
            else:
                self.logger.error(f"Order failed for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    async def run_strategy_cycle(self):
        """Run one complete strategy cycle."""
        self.cycle_count += 1
        self.logger.info(f"=== Strategy Cycle {self.cycle_count} ===")
        
        successful_trades = 0
        
        for symbol in self.config.symbols:
            try:
                self.logger.info(f"Processing {symbol}...")
                
                # Calculate signals
                signal_data = await self.calculate_signals(symbol)
                signal = signal_data['signal']
                strength = signal_data['strength']
                
                self.logger.info(f"{symbol} - Signal: {signal:.3f}, Strength: {strength:.3f}")
                
                # Store latest signals for monitoring
                self.last_signals[symbol] = signal_data
                
                # Skip weak signals
                if abs(signal) < 0.3 or strength < 0.5:
                    self.logger.info(f"Signal too weak for {symbol}")
                    continue
                
                # Calculate position size
                position_size = await self.calculate_position_size(symbol, strength)
                
                if position_size > 0:
                    success = await self.execute_trade(symbol, signal, position_size)
                    if success:
                        successful_trades += 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
        
        self.logger.info(f"Cycle {self.cycle_count} complete. {successful_trades} trades executed.")
        
        # Update performance stats
        if successful_trades > 0:
            await self._update_performance_stats()
    
    async def _update_performance_stats(self):
        """Update performance statistics."""
        try:
            # Get current positions for PnL calculation
            positions = await self.broker.get_positions()
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            self.performance_stats['total_pnl'] = total_unrealized_pnl
            
            self.logger.info(f"Performance Update - Total PnL: ${total_unrealized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")
    
    async def start(self):
        """Start the strategy main loop."""
        self.is_running = True
        self.logger.info(f"Starting strategy '{self.config.name}'")
        
        # Connect to broker
        try:
            await self.broker.authenticate()
            account_info = await self.broker.get_account_info()
            self.logger.info(f"Connected to Alpaca - Account Value: ${account_info.portfolio_value:,.2f}")
        except Exception as e:
            self.logger.error(f"Failed to connect to broker: {e}")
            return
        
        # Main strategy loop
        while self.is_running:
            try:
                await self.run_strategy_cycle()
                
                # Wait for next cycle
                wait_time = self.config.cycle_interval_minutes * 60
                self.logger.info(f"Waiting {self.config.cycle_interval_minutes} minutes until next cycle...")
                await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                self.logger.info("Strategy stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        self.logger.info("Strategy stopped")
    
    def stop(self):
        """Stop the strategy."""
        self.is_running = False
        self.logger.info("Strategy stop requested")


async def main():
    """Main function for standalone execution or CLI deployment."""
    parser = argparse.ArgumentParser(description="Deployable MA Crossover Strategy")
    parser.add_argument('--deployed', action='store_true', help='Running as deployed strategy')
    parser.add_argument('--config', help='Path to strategy configuration file')
    parser.add_argument('--symbols', default='AAPL,TSLA,MSFT', help='Trading symbols (comma-separated)')
    parser.add_argument('--broker', default='alpaca', help='Broker name')
    parser.add_argument('--dry-run', action='store_true', help='Paper trading mode')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        # Load from YAML config (deployed mode)
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        strategy_config = StrategyConfig(**config_data)
    else:
        # Create basic config for standalone mode
        strategy_config = StrategyConfig(
            name="ma_crossover_standalone",
            description="Moving Average Crossover Strategy",
            strategy_file=__file__,
            broker_name=args.broker,
            symbols=args.symbols.split(','),
            cycle_interval_minutes=5,
            enable_kelly_optimization=True
        )
    
    # Setup broker configuration for paper trading
    broker_config = BrokerConfig(paper_trading=True)
    
    # Load broker credentials (you'll need to provide these)
    if args.deployed:
        # In deployed mode, credentials should be in environment or config
        credentials = {
            'api_key': os.getenv('ALPACA_API_KEY', 'PKERMRICJJBP7MAY61NN'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', 'PLEASE_SET_SECRET_KEY'),
            'base_url': 'https://paper-api.alpaca.markets'
        }
    else:
        # For standalone testing, use paper trading credentials
        credentials = {
            'api_key': 'PKERMRICJJBP7MAY61NN',  # Your paper trading key
            'secret_key': 'PLEASE_SET_YOUR_SECRET_KEY',  # You need to set this
            'base_url': 'https://paper-api.alpaca.markets'
        }
    
    # Create broker adapter
    from wrtrade.brokers_real import AlpacaBrokerAdapter
    broker_adapter = AlpacaBrokerAdapter(credentials, broker_config)
    
    # Create and start strategy
    strategy = DeployableMAStrategy(strategy_config, broker_adapter)
    
    try:
        async with broker_adapter:
            await strategy.start()
    except Exception as e:
        print(f"Strategy error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())