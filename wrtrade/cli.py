"""
WRTrade CLI - Command line interface for strategy deployment and management.
This is the main entry point for users to manage their trading strategies.
"""

import click
import asyncio
import sys
import json
import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from wrtrade.local_deploy import LocalDeployment, StrategyManager, StrategyConfig, StrategyStatus
from wrtrade.brokers_real import BrokerFactory


@click.group()
@click.version_option(version="1.0.0", prog_name="wrtrade")
def cli():
    """WRTrade - Advanced Portfolio Trading Framework"""
    pass


@cli.group()
def strategy():
    """Strategy management commands"""
    pass


@strategy.command()
@click.argument('strategy_file', type=click.Path(exists=True))
@click.option('--name', '-n', help='Strategy name', required=True)
@click.option('--description', '-d', help='Strategy description', default='')
@click.option('--broker', '-b', help='Broker name (alpaca, robinhood)', required=False)
@click.option('--symbols', '-s', help='Trading symbols (comma-separated)', required=False)
@click.option('--max-position', type=float, default=1000.0, help='Maximum position size ($)')
@click.option('--risk-per-trade', type=float, default=0.02, help='Risk per trade (fraction)')
@click.option('--max-trades', type=int, default=10, help='Maximum daily trades')
@click.option('--cycle-minutes', type=int, default=5, help='Strategy cycle interval (minutes)')
@click.option('--market-hours-only', is_flag=True, default=True, help='Trade during market hours only')
@click.option('--kelly-optimization', is_flag=True, default=True, help='Enable Kelly optimization')
@click.option('--permutation-testing', is_flag=True, default=False, help='Enable permutation testing')
@click.option('--auto-restart', is_flag=True, default=True, help='Auto-restart on failure')
@click.option('--config', type=click.Path(), help='Load config from .env or YAML file')
def deploy(strategy_file: str, name: str, description: str, broker: str, symbols: str,
          max_position: float, risk_per_trade: float, max_trades: int, cycle_minutes: int,
          market_hours_only: bool, kelly_optimization: bool, permutation_testing: bool,
          auto_restart: bool, config: Optional[str]):
    """Deploy a trading strategy for local execution"""
    
    try:
        # Load config from .env file if provided
        if config:
            # Load environment variables from .env file
            load_dotenv(config)
            
            # Helper function to get env var with type conversion
            def get_env(key: str, default=None, var_type=str):
                value = os.getenv(key, default)
                if value is None:
                    return default
                if var_type == bool:
                    return value.lower() in ('true', '1', 'yes', 'on')
                elif var_type == int:
                    return int(value)
                elif var_type == float:
                    return float(value)
                elif var_type == list:
                    return [s.strip() for s in value.split(',')]
                return value
            
            # Override with command line args where provided (use .env values as defaults)
            config_dict = {
                'name': name,
                'description': description or get_env('STRATEGY_DESCRIPTION', ''),
                'strategy_file': str(Path(strategy_file).absolute()),
                'broker_name': broker or get_env('BROKER_NAME', 'alpaca'),
                'symbols': symbols.split(',') if symbols else get_env('TRADING_SYMBOLS', 'AAPL,TSLA', list),
                'max_position_size': get_env('MAX_POSITION_SIZE', max_position, float),
                'risk_per_trade': get_env('RISK_PER_TRADE', risk_per_trade, float),
                'max_daily_trades': get_env('MAX_DAILY_TRADES', max_trades, int),
                'cycle_interval_minutes': get_env('CYCLE_INTERVAL_MINUTES', cycle_minutes, int),
                'market_hours_only': get_env('MARKET_HOURS_ONLY', market_hours_only, bool),
                'enable_kelly_optimization': get_env('ENABLE_KELLY_OPTIMIZATION', kelly_optimization, bool),
                'enable_permutation_testing': get_env('ENABLE_PERMUTATION_TESTING', permutation_testing, bool),
                'auto_restart': get_env('AUTO_RESTART', auto_restart, bool),
            }
        else:
            # Validate required fields when no config file
            if not broker:
                click.echo("❌ Error: --broker is required when not using config file")
                return
            if not symbols:
                click.echo("❌ Error: --symbols is required when not using config file")
                return
                
            config_dict = {
                'name': name,
                'description': description,
                'strategy_file': str(Path(strategy_file).absolute()),
                'broker_name': broker,
                'symbols': symbols.split(','),
                'max_position_size': max_position,
                'risk_per_trade': risk_per_trade,
                'max_daily_trades': max_trades,
                'cycle_interval_minutes': cycle_minutes,
                'market_hours_only': market_hours_only,
                'enable_kelly_optimization': kelly_optimization,
                'enable_permutation_testing': permutation_testing,
                'auto_restart': auto_restart,
            }
        
        # Validate broker
        available_brokers = BrokerFactory.list_available_brokers()
        if config_dict['broker_name'].lower() not in available_brokers:
            click.echo(f"❌ Error: Broker '{config_dict['broker_name']}' not available.")
            click.echo(f"Available brokers: {', '.join(available_brokers)}")
            return
        
        # Create strategy config
        config = StrategyConfig(**config_dict)
        
        # Deploy strategy
        manager = StrategyManager()
        success = manager.register_strategy(config)
        
        if success:
            click.echo(f"✅ Strategy '{name}' deployed successfully")
            click.echo(f"📁 Configuration saved to ~/.wrtrade/strategies/{name}/")
            click.echo(f"🚀 Start with: wrtrade strategy start {name}")
        else:
            click.echo(f"❌ Failed to deploy strategy '{name}'")
            
    except Exception as e:
        click.echo(f"❌ Error deploying strategy: {e}")


@strategy.command()
@click.argument('name')
def start(name: str):
    """Start a deployed strategy"""
    
    async def _start():
        try:
            manager = StrategyManager()
            success = await manager.start_strategy(name)
            
            if success:
                click.echo(f"✅ Strategy '{name}' started successfully")
                click.echo(f"📊 Monitor with: wrtrade strategy status {name}")
                click.echo(f"📋 View logs with: wrtrade strategy logs {name}")
            else:
                click.echo(f"❌ Failed to start strategy '{name}'")
                
        except Exception as e:
            click.echo(f"❌ Error starting strategy: {e}")
    
    asyncio.run(_start())


@strategy.command()
@click.argument('name')
@click.option('--force', is_flag=True, help='Force stop (kill process)')
def stop(name: str, force: bool):
    """Stop a running strategy"""
    
    async def _stop():
        try:
            manager = StrategyManager()
            success = await manager.stop_strategy(name, force=force)
            
            if success:
                click.echo(f"✅ Strategy '{name}' stopped successfully")
            else:
                click.echo(f"❌ Failed to stop strategy '{name}'")
                
        except Exception as e:
            click.echo(f"❌ Error stopping strategy: {e}")
    
    asyncio.run(_stop())


@strategy.command()
@click.argument('name')
def restart(name: str):
    """Restart a strategy"""
    
    async def _restart():
        try:
            manager = StrategyManager()
            success = await manager.restart_strategy(name)
            
            if success:
                click.echo(f"✅ Strategy '{name}' restarted successfully")
            else:
                click.echo(f"❌ Failed to restart strategy '{name}'")
                
        except Exception as e:
            click.echo(f"❌ Error restarting strategy: {e}")
    
    asyncio.run(_restart())


@strategy.command()
@click.argument('name', required=False)
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def status(name: Optional[str], output_json: bool):
    """Get strategy status"""
    
    try:
        manager = StrategyManager()
        
        if name:
            status_data = manager.get_strategy_status(name)
            if not status_data:
                click.echo(f"❌ Strategy '{name}' not found")
                return
            
            if output_json:
                click.echo(json.dumps(status_data, indent=2, default=str))
            else:
                _print_strategy_status(status_data)
        else:
            strategies = manager.list_strategies()
            if not strategies:
                click.echo("📭 No strategies deployed")
                return
            
            if output_json:
                click.echo(json.dumps(strategies, indent=2, default=str))
            else:
                click.echo("📊 Deployed Strategies:")
                click.echo("=" * 50)
                for strategy_data in strategies:
                    _print_strategy_summary(strategy_data)
                    click.echo()
                    
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")


def _print_strategy_status(data: Dict[str, Any]):
    """Print detailed strategy status"""
    status_emoji = {
        'running': '🟢',
        'stopped': '⚪',
        'error': '🔴',
        'starting': '🟡',
        'stopping': '🟠'
    }.get(data['status'], '❓')
    
    click.echo(f"{status_emoji} Strategy: {data['name']}")
    click.echo(f"   Status: {data['status'].upper()}")
    
    if data['pid']:
        click.echo(f"   PID: {data['pid']}")
    
    if data['uptime']:
        click.echo(f"   Uptime: {data['uptime']}")
    
    if data['total_trades'] > 0:
        click.echo(f"   Trades: {data['total_trades']} (P&L: ${data['total_pnl']:.2f})")
    
    if data['restart_count'] > 0:
        click.echo(f"   Restarts: {data['restart_count']}")
    
    if data['last_error']:
        click.echo(f"   Last Error: {data['last_error']}")
    
    click.echo(f"   Log Size: {data['log_size_mb']} MB")
    
    # Config summary
    config = data['config']
    click.echo(f"   Broker: {config['broker_name']}")
    click.echo(f"   Symbols: {', '.join(config['symbols'])}")
    click.echo(f"   Max Position: ${config['max_position_size']}")


def _print_strategy_summary(data: Dict[str, Any]):
    """Print strategy summary for list view"""
    status_emoji = {
        'running': '🟢',
        'stopped': '⚪',
        'error': '🔴',
        'starting': '🟡',
        'stopping': '🟠'
    }.get(data['status'], '❓')
    
    uptime = f" ({data['uptime']})" if data['uptime'] else ""
    trades = f" | {data['total_trades']} trades" if data['total_trades'] > 0 else ""
    
    click.echo(f"{status_emoji} {data['name']}: {data['status'].upper()}{uptime}{trades}")


@strategy.command()
@click.argument('name')
@click.option('--lines', '-n', type=int, default=50, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
def logs(name: str, lines: int, follow: bool):
    """View strategy logs"""
    
    try:
        manager = StrategyManager()
        log_file = manager.logs_dir / f"{name}.log"
        
        if not log_file.exists():
            click.echo(f"❌ No log file found for strategy '{name}'")
            return
        
        if follow:
            # Follow logs (like tail -f)
            import subprocess
            subprocess.run(['tail', '-f', str(log_file)])
        else:
            # Show last N lines
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    click.echo(line.rstrip())
                    
    except Exception as e:
        click.echo(f"❌ Error reading logs: {e}")


@strategy.command()
@click.argument('name')
@click.option('--remove-data', is_flag=True, help='Remove all data (logs, configs, etc.)')
def remove(name: str, remove_data: bool):
    """Remove a deployed strategy"""
    
    try:
        manager = StrategyManager()
        
        # Check if strategy exists
        if name not in manager.strategies:
            click.echo(f"❌ Strategy '{name}' not found")
            return
        
        # Confirm removal
        if remove_data:
            click.echo(f"⚠️  This will remove strategy '{name}' and ALL its data (logs, configs, etc.)")
        else:
            click.echo(f"⚠️  This will remove strategy '{name}' but keep its data")
        
        if not click.confirm('Are you sure?'):
            click.echo("Cancelled")
            return
        
        success = manager.unregister_strategy(name, remove_data=remove_data)
        
        if success:
            action = "removed completely" if remove_data else "unregistered"
            click.echo(f"✅ Strategy '{name}' {action}")
        else:
            click.echo(f"❌ Failed to remove strategy '{name}'")
            
    except Exception as e:
        click.echo(f"❌ Error removing strategy: {e}")


@cli.command()
def init():
    """Initialize WRTrade workspace"""
    
    try:
        manager = StrategyManager()
        workspace_dir = manager.workspace_dir
        
        click.echo(f"🚀 Initializing WRTrade workspace at {workspace_dir}")
        
        # Create example config
        example_config = {
            'name': 'my-strategy',
            'description': 'My trading strategy',
            'strategy_file': '/path/to/my_strategy.py',
            'broker_name': 'alpaca',
            'symbols': ['AAPL', 'TSLA', 'MSFT'],
            'max_position_size': 1000.0,
            'risk_per_trade': 0.02,
            'max_daily_trades': 10,
            'cycle_interval_minutes': 5,
            'market_hours_only': True,
            'enable_kelly_optimization': True,
            'enable_permutation_testing': False,
            'auto_restart': True
        }
        
        example_file = workspace_dir / "example_strategy_config.yaml"
        with open(example_file, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)
        
        click.echo(f"✅ Workspace initialized")
        click.echo(f"📁 Location: {workspace_dir}")
        click.echo(f"📄 Example config: {example_file}")
        click.echo(f"🚀 Deploy a strategy with: wrtrade strategy deploy <strategy.py> --name my-strategy")
        
    except Exception as e:
        click.echo(f"❌ Error initializing workspace: {e}")


@cli.group()
def broker():
    """Broker management commands"""
    pass


@broker.command()
def list():
    """List available brokers"""
    
    try:
        brokers = BrokerFactory.list_available_brokers()
        
        click.echo("📈 Available Brokers:")
        for broker_name in brokers:
            click.echo(f"  • {broker_name}")
        
        if not brokers:
            click.echo("❌ No brokers available. Install broker SDKs:")
            click.echo("  • Alpaca: pip install alpaca-trade-api")
            click.echo("  • Robinhood: pip install robin-stocks")
            
    except Exception as e:
        click.echo(f"❌ Error listing brokers: {e}")


@cli.command()
def version():
    """Show WRTrade version and info"""
    
    click.echo("WRTrade - Advanced Portfolio Trading Framework")
    click.echo("Version: 1.0.0")
    click.echo("Repository: https://github.com/wayy-research/wrtrade")
    click.echo()
    click.echo("Features:")
    click.echo("  • N-dimensional portfolios")
    click.echo("  • Kelly optimization")
    click.echo("  • Permutation testing")
    click.echo("  • Local deployment")
    click.echo("  • Multi-broker support")


if __name__ == '__main__':
    cli()