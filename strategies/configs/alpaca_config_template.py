"""
Alpaca API configuration template.
Copy this file to alpaca_config.py and fill in your actual credentials.
Never commit the actual config file with real credentials to version control.
"""

from wrtrade.brokers_real import BrokerConfig
from typing import Dict

# Paper Trading Configuration (Safe for testing)
ALPACA_PAPER_CONFIG = {
    'api_key': 'PKERMRICJJBP7MAY61NN',  # Your paper trading key
    'secret_key': 'YOUR_SECRET_KEY_HERE',  # Fill in your secret key
    'base_url': 'https://paper-api.alpaca.markets'  # Paper trading URL
}

# Live Trading Configuration (Use with extreme caution)
ALPACA_LIVE_CONFIG = {
    'api_key': 'YOUR_LIVE_API_KEY_HERE',
    'secret_key': 'YOUR_LIVE_SECRET_KEY_HERE', 
    'base_url': 'https://api.alpaca.markets'  # Live trading URL - BE CAREFUL!
}

# Broker Configuration Settings
BROKER_CONFIG = BrokerConfig(
    paper_trading=True,  # Always start with paper trading!
    max_requests_per_minute=200,  # Alpaca rate limit
    retry_attempts=3,
    retry_delay=1.0,
    timeout=30.0,
    enable_logging=True,
    log_level="INFO"
)

# Trading Session Settings
TRADING_CONFIG = {
    'max_position_size': 1000.0,  # Maximum $ per position
    'max_daily_trades': 10,       # Max trades per day
    'risk_per_trade': 0.02,       # 2% risk per trade
    'stop_loss_percent': 0.05,    # 5% stop loss
    'take_profit_percent': 0.10,  # 10% take profit
    'start_time': '09:30',        # Market open
    'end_time': '15:30',          # Market close (EST)
    'symbols_watchlist': [        # Symbols to monitor
        'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 
        'NVDA', 'META', 'NFLX', 'SPY', 'QQQ'
    ]
}

def get_alpaca_credentials(live_trading: bool = False) -> Dict[str, str]:
    """
    Get Alpaca API credentials.
    
    Args:
        live_trading: If True, returns live trading credentials.
                     If False, returns paper trading credentials.
    
    Returns:
        Dictionary with API credentials
    """
    if live_trading:
        print("⚠️  WARNING: Using LIVE trading credentials! ⚠️")
        print("Make sure you understand the risks!")
        return ALPACA_LIVE_CONFIG
    else:
        print("✅ Using paper trading credentials (safe)")
        return ALPACA_PAPER_CONFIG

def validate_config() -> bool:
    """Validate that configuration is properly set up."""
    paper_config = ALPACA_PAPER_CONFIG
    
    if 'YOUR_SECRET_KEY_HERE' in paper_config['secret_key']:
        print("❌ Please update alpaca_config.py with your actual secret key")
        return False
    
    if not paper_config['api_key']:
        print("❌ API key is missing")
        return False
    
    print("✅ Configuration looks good!")
    return True