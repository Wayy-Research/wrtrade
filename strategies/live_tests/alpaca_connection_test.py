"""
Alpaca connection test and basic functionality verification.
This script tests the real Alpaca API connection and basic operations.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from wrtrade.brokers_real import AlpacaBrokerAdapter, BrokerConfig, Order, OrderType, TimeInForce
from strategies.configs.alpaca_config import get_alpaca_credentials, BROKER_CONFIG, validate_config


async def test_alpaca_connection():
    """Test basic Alpaca API connection and functionality."""
    
    print("🚀 Starting Alpaca API Connection Test")
    print("=" * 50)
    
    # Validate configuration first
    if not validate_config():
        print("❌ Configuration validation failed!")
        return False
    
    # Get credentials (paper trading)
    credentials = get_alpaca_credentials(live_trading=False)
    
    # Create adapter
    adapter = AlpacaBrokerAdapter(credentials, BROKER_CONFIG)
    
    try:
        # Connect and authenticate
        print("\n📡 Connecting to Alpaca...")
        async with adapter:
            print("✅ Connection successful!")
            
            # Test 1: Get account info
            print("\n💰 Getting account information...")
            account_info = await adapter.get_account_info()
            print(f"   Account ID: {account_info.account_id}")
            print(f"   Buying Power: ${account_info.buying_power:,.2f}")
            print(f"   Cash: ${account_info.cash:,.2f}")
            print(f"   Portfolio Value: ${account_info.portfolio_value:,.2f}")
            print(f"   Day Trader: {account_info.is_day_trader}")
            
            # Test 2: Get positions
            print("\n📊 Getting current positions...")
            positions = await adapter.get_positions()
            if positions:
                print(f"   Found {len(positions)} positions:")
                for pos in positions:
                    pnl_color = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                    print(f"   {pnl_color} {pos.symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}")
                    print(f"      Market Value: ${pos.market_value:,.2f}, P&L: ${pos.unrealized_pnl:,.2f}")
            else:
                print("   No positions found")
            
            # Test 3: Get market data
            print("\n📈 Getting market data...")
            test_symbols = ['AAPL', 'TSLA', 'SPY']
            market_data = await adapter.get_market_data(test_symbols)
            
            for symbol, data in market_data.items():
                print(f"   {symbol}: ${data.price:.2f} (Bid: ${data.bid:.2f}, Ask: ${data.ask:.2f})")
            
            # Test 4: Get historical data
            print("\n📚 Getting historical data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            hist_data = await adapter.get_historical_data('AAPL', start_date, end_date)
            if not hist_data.is_empty():
                print(f"   Retrieved {len(hist_data)} data points for AAPL")
                print(f"   Latest close: ${hist_data.tail(1)['close'][0]:.2f}")
            else:
                print("   No historical data retrieved")
            
            # Test 5: Get order history
            print("\n📋 Getting order history...")
            orders = await adapter.get_orders(limit=5)
            if orders:
                print(f"   Found {len(orders)} recent orders:")
                for order in orders[:3]:  # Show latest 3
                    status_emoji = {"filled": "✅", "cancelled": "❌", "pending": "⏳"}.get(order.status.value, "❓")
                    print(f"   {status_emoji} {order.symbol}: {order.side} {order.quantity} @ {order.order_type.value}")
                    print(f"      Status: {order.status.value}, Created: {order.created_at}")
            else:
                print("   No orders found")
            
            # Test 6: Paper trading order (small test)
            print("\n🧪 Testing paper trading order placement...")
            print("   ⚠️  This will place a REAL paper trading order!")
            
            response = input("   Continue with test order? (y/N): ").strip().lower()
            if response == 'y':
                # Place a small test order
                test_order = Order(
                    symbol='AAPL',
                    quantity=1,
                    side='buy',
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )
                
                success = await adapter.place_order(test_order)
                if success:
                    print(f"   ✅ Test order placed successfully!")
                    print(f"      Order ID: {test_order.order_id}")
                    print(f"      Status: {test_order.status.value}")
                    
                    # Wait a moment and check status
                    await asyncio.sleep(2)
                    updated_order = await adapter.get_order_status(test_order.order_id)
                    if updated_order:
                        print(f"   📊 Updated status: {updated_order.status.value}")
                        if updated_order.status.value == 'filled':
                            print(f"      Filled at: ${updated_order.avg_fill_price:.2f}")
                else:
                    print(f"   ❌ Test order failed: {test_order.status.value}")
            else:
                print("   Skipping test order")
            
            print("\n🎉 All tests completed successfully!")
            return True
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False


async def test_market_data_streaming():
    """Test real-time market data streaming capabilities."""
    print("\n🔄 Testing market data streaming...")
    
    credentials = get_alpaca_credentials(live_trading=False)
    adapter = AlpacaBrokerAdapter(credentials, BROKER_CONFIG)
    
    try:
        async with adapter:
            symbols = ['AAPL', 'TSLA', 'SPY']
            
            print(f"   Monitoring {symbols} for 30 seconds...")
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < 30:
                market_data = await adapter.get_market_data(symbols)
                
                print(f"   [{datetime.now().strftime('%H:%M:%S')}] ", end="")
                for symbol, data in market_data.items():
                    print(f"{symbol}: ${data.price:.2f} ", end="")
                print()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
    except KeyboardInterrupt:
        print("   Streaming stopped by user")
    except Exception as e:
        print(f"   Streaming error: {e}")


def main():
    """Main function to run all tests."""
    print("Alpaca API Live Testing Suite")
    print("============================")
    print("This will test real API connections with paper trading")
    print()
    
    # Run basic connection test
    success = asyncio.run(test_alpaca_connection())
    
    if success:
        print("\n" + "="*50)
        response = input("Run market data streaming test? (y/N): ").strip().lower()
        if response == 'y':
            asyncio.run(test_market_data_streaming())
    
    print("\n✨ Testing complete!")


if __name__ == "__main__":
    main()