"""
Local deployment system for WRTrade strategies.
Manages strategy lifecycle: deployment, monitoring, logging, and shutdown.
"""

import os
import sys
import json
import yaml
import asyncio
import logging
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import pickle
from contextlib import asynccontextmanager

from wrtrade.brokers_real import BrokerAdapter, BrokerFactory, BrokerConfig
from wrtrade.components import CompositePortfolio
from wrtrade.metrics import calculate_all_metrics


class StrategyStatus(Enum):
    """Strategy deployment status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class StrategyConfig:
    """Configuration for a deployed strategy."""
    name: str
    description: str
    strategy_file: str
    broker_name: str
    symbols: List[str]
    
    # Trading parameters
    max_position_size: float = 1000.0
    risk_per_trade: float = 0.02
    max_daily_trades: int = 10
    
    # Execution parameters
    cycle_interval_minutes: int = 5
    market_hours_only: bool = True
    enable_kelly_optimization: bool = True
    enable_permutation_testing: bool = False
    
    # Risk management
    stop_loss_percent: float = 0.05
    take_profit_percent: float = 0.10
    max_drawdown_percent: float = 0.20
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_trades: bool = True
    save_performance: bool = True
    
    # Deployment settings
    auto_restart: bool = True
    max_restarts: int = 3
    health_check_interval: int = 60
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StrategyState:
    """Runtime state of a deployed strategy."""
    config: StrategyConfig
    status: StrategyStatus = StrategyStatus.STOPPED
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    restart_count: int = 0
    
    # Performance tracking
    total_trades: int = 0
    total_pnl: float = 0.0
    daily_trades: int = 0
    last_trade_time: Optional[datetime] = None
    
    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = {}
        
        # Convert each field manually to handle special cases
        for field_name, field_value in [
            ('config', self.config),
            ('status', self.status),
            ('pid', self.pid),
            ('start_time', self.start_time),
            ('last_heartbeat', self.last_heartbeat),
            ('restart_count', self.restart_count),
            ('total_trades', self.total_trades),
            ('total_pnl', self.total_pnl),
            ('daily_trades', self.daily_trades),
            ('last_trade_time', self.last_trade_time),
            ('last_error', self.last_error),
            ('error_count', self.error_count)
        ]:
            if isinstance(field_value, datetime):
                data[field_name] = field_value.isoformat()
            elif field_name == 'status':
                data[field_name] = field_value.value
            elif field_name == 'config':
                data[field_name] = field_value.to_dict() if hasattr(field_value, 'to_dict') else asdict(field_value)
            else:
                data[field_name] = field_value
        
        return data


class StrategyManager:
    """
    Manages local deployment of trading strategies.
    Handles strategy lifecycle, monitoring, and persistence.
    """
    
    def __init__(self, workspace_dir: str = "~/.wrtrade"):
        """Initialize strategy manager."""
        self.workspace_dir = Path(workspace_dir).expanduser()
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Directories
        self.strategies_dir = self.workspace_dir / "strategies"
        self.logs_dir = self.workspace_dir / "logs"
        self.data_dir = self.workspace_dir / "data"
        self.configs_dir = self.workspace_dir / "configs"
        
        for dir_path in [self.strategies_dir, self.logs_dir, self.data_dir, self.configs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # State management
        self.state_file = self.workspace_dir / "strategies_state.json"
        self.strategies: Dict[str, StrategyState] = {}
        self.load_state()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for strategy manager."""
        logger = logging.getLogger("StrategyManager")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.logs_dir / "strategy_manager.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_state(self):
        """Load strategies state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                for name, state_data in data.items():
                    # Reconstruct StrategyConfig
                    config_data = state_data['config']
                    config = StrategyConfig.from_dict(config_data)
                    
                    # Reconstruct StrategyState
                    state_data['config'] = config
                    state_data['status'] = StrategyStatus(state_data['status'])
                    
                    # Handle datetime fields
                    for field in ['start_time', 'last_heartbeat', 'last_trade_time']:
                        if state_data.get(field):
                            state_data[field] = datetime.fromisoformat(state_data[field])
                    
                    self.strategies[name] = StrategyState(**state_data)
                    
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                self.strategies = {}
    
    def save_state(self):
        """Save strategies state to disk."""
        try:
            data = {name: state.to_dict() for name, state in self.strategies.items()}
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def register_strategy(self, config: StrategyConfig) -> bool:
        """Register a new strategy for deployment."""
        try:
            # Validate configuration
            if not self._validate_config(config):
                return False
            
            # Create strategy state
            state = StrategyState(config=config)
            self.strategies[config.name] = state
            
            # Create strategy-specific directories
            strategy_dir = self.strategies_dir / config.name
            strategy_dir.mkdir(exist_ok=True)
            
            (strategy_dir / "logs").mkdir(exist_ok=True)
            (strategy_dir / "data").mkdir(exist_ok=True)
            
            # Save configuration
            config_file = strategy_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            
            self.save_state()
            self.logger.info(f"Strategy '{config.name}' registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register strategy '{config.name}': {e}")
            return False
    
    def _validate_config(self, config: StrategyConfig) -> bool:
        """Validate strategy configuration."""
        # Check if strategy file exists
        if not Path(config.strategy_file).exists():
            self.logger.error(f"Strategy file not found: {config.strategy_file}")
            return False
        
        # Check broker configuration
        try:
            available_brokers = BrokerFactory.list_available_brokers()
            if config.broker_name.lower() not in available_brokers:
                self.logger.error(f"Broker '{config.broker_name}' not available. Available: {available_brokers}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to validate broker: {e}")
            return False
        
        # Validate symbols
        if not config.symbols:
            self.logger.error("No symbols specified for strategy")
            return False
        
        return True
    
    async def start_strategy(self, name: str) -> bool:
        """Start a deployed strategy."""
        if name not in self.strategies:
            self.logger.error(f"Strategy '{name}' not found")
            return False
        
        state = self.strategies[name]
        
        if state.status in [StrategyStatus.RUNNING, StrategyStatus.STARTING]:
            self.logger.warning(f"Strategy '{name}' is already running")
            return True
        
        try:
            state.status = StrategyStatus.STARTING
            self.save_state()
            
            # Create process environment
            env = os.environ.copy()
            env['WRTRADE_STRATEGY_NAME'] = name
            env['WRTRADE_WORKSPACE'] = str(self.workspace_dir)
            
            # Start strategy process
            cmd = [
                sys.executable,
                state.config.strategy_file,
                "--deployed",
                "--config", str(self.strategies_dir / name / "config.yaml")
            ]
            
            # Setup logging for strategy process
            log_file = self.logs_dir / f"{name}.log"
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=open(log_file, 'a'),
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.workspace_dir)
            )
            
            state.pid = process.pid
            state.start_time = datetime.now()
            state.status = StrategyStatus.RUNNING
            state.last_heartbeat = datetime.now()
            
            self.save_state()
            self.logger.info(f"Strategy '{name}' started with PID {process.pid}")
            return True
            
        except Exception as e:
            state.status = StrategyStatus.ERROR
            state.last_error = str(e)
            self.save_state()
            self.logger.error(f"Failed to start strategy '{name}': {e}")
            return False
    
    async def stop_strategy(self, name: str, force: bool = False) -> bool:
        """Stop a running strategy."""
        if name not in self.strategies:
            self.logger.error(f"Strategy '{name}' not found")
            return False
        
        state = self.strategies[name]
        
        if state.status == StrategyStatus.STOPPED:
            self.logger.info(f"Strategy '{name}' is already stopped")
            return True
        
        try:
            state.status = StrategyStatus.STOPPING
            self.save_state()
            
            if state.pid:
                try:
                    process = psutil.Process(state.pid)
                    
                    if force:
                        process.kill()
                    else:
                        process.terminate()
                        # Wait for graceful shutdown
                        await asyncio.sleep(5)
                        if process.is_running():
                            process.kill()
                    
                    state.pid = None
                    
                except psutil.NoSuchProcess:
                    self.logger.warning(f"Process {state.pid} not found")
                    state.pid = None
            
            state.status = StrategyStatus.STOPPED
            self.save_state()
            self.logger.info(f"Strategy '{name}' stopped")
            return True
            
        except Exception as e:
            state.last_error = str(e)
            self.save_state()
            self.logger.error(f"Failed to stop strategy '{name}': {e}")
            return False
    
    async def restart_strategy(self, name: str) -> bool:
        """Restart a strategy."""
        if name not in self.strategies:
            return False
        
        state = self.strategies[name]
        state.restart_count += 1
        
        if state.restart_count > state.config.max_restarts:
            self.logger.error(f"Strategy '{name}' exceeded max restarts ({state.config.max_restarts})")
            state.status = StrategyStatus.ERROR
            self.save_state()
            return False
        
        # Stop first
        await self.stop_strategy(name)
        await asyncio.sleep(2)
        
        # Start again
        return await self.start_strategy(name)
    
    def get_strategy_status(self, name: str) -> Optional[Dict]:
        """Get detailed status of a strategy."""
        if name not in self.strategies:
            return None
        
        state = self.strategies[name]
        
        # Check if process is actually running
        if state.pid and state.status == StrategyStatus.RUNNING:
            try:
                process = psutil.Process(state.pid)
                if not process.is_running():
                    state.status = StrategyStatus.ERROR
                    state.last_error = "Process died unexpectedly"
                    state.pid = None
                    self.save_state()
            except psutil.NoSuchProcess:
                state.status = StrategyStatus.ERROR
                state.last_error = "Process not found"
                state.pid = None
                self.save_state()
        
        # Calculate uptime
        uptime = None
        if state.start_time and state.status == StrategyStatus.RUNNING:
            uptime = datetime.now() - state.start_time
        
        # Get log file size
        log_file = self.logs_dir / f"{name}.log"
        log_size = log_file.stat().st_size if log_file.exists() else 0
        
        return {
            'name': name,
            'status': state.status.value,
            'pid': state.pid,
            'uptime': str(uptime) if uptime else None,
            'restart_count': state.restart_count,
            'total_trades': state.total_trades,
            'total_pnl': state.total_pnl,
            'daily_trades': state.daily_trades,
            'last_trade_time': state.last_trade_time.isoformat() if state.last_trade_time else None,
            'last_error': state.last_error,
            'error_count': state.error_count,
            'log_size_mb': round(log_size / (1024 * 1024), 2),
            'config': state.config.to_dict()
        }
    
    def list_strategies(self) -> List[Dict]:
        """List all registered strategies with their status."""
        return [self.get_strategy_status(name) for name in self.strategies.keys()]
    
    async def monitor_strategies(self):
        """Monitor all running strategies and handle health checks."""
        self.logger.info("Starting strategy monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                for name, state in self.strategies.items():
                    if state.status == StrategyStatus.RUNNING:
                        # Health check
                        if state.pid:
                            try:
                                process = psutil.Process(state.pid)
                                if not process.is_running():
                                    self.logger.warning(f"Strategy '{name}' process died")
                                    
                                    if state.config.auto_restart:
                                        self.logger.info(f"Auto-restarting strategy '{name}'")
                                        await self.restart_strategy(name)
                                    else:
                                        state.status = StrategyStatus.ERROR
                                        state.last_error = "Process died"
                                        state.pid = None
                                        
                            except psutil.NoSuchProcess:
                                state.status = StrategyStatus.ERROR
                                state.last_error = "Process not found"
                                state.pid = None
                
                self.save_state()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(30)
    
    async def start_monitoring(self):
        """Start the monitoring task."""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self.monitor_strategies())
    
    async def stop_monitoring(self):
        """Stop the monitoring task."""
        self.shutdown_event.set()
        if self.monitoring_task:
            await self.monitoring_task
            self.monitoring_task = None
    
    async def shutdown_all(self):
        """Gracefully shutdown all strategies."""
        self.logger.info("Shutting down all strategies")
        
        # Stop monitoring first
        await self.stop_monitoring()
        
        # Stop all running strategies
        for name, state in self.strategies.items():
            if state.status in [StrategyStatus.RUNNING, StrategyStatus.STARTING]:
                await self.stop_strategy(name)
        
        self.logger.info("All strategies stopped")
    
    def unregister_strategy(self, name: str, remove_data: bool = False) -> bool:
        """Unregister a strategy."""
        if name not in self.strategies:
            return False
        
        state = self.strategies[name]
        
        # Make sure it's stopped
        if state.status in [StrategyStatus.RUNNING, StrategyStatus.STARTING]:
            self.logger.error(f"Cannot unregister running strategy '{name}'. Stop it first.")
            return False
        
        # Remove from memory
        del self.strategies[name]
        
        # Remove data if requested
        if remove_data:
            strategy_dir = self.strategies_dir / name
            if strategy_dir.exists():
                import shutil
                shutil.rmtree(strategy_dir)
        
        self.save_state()
        self.logger.info(f"Strategy '{name}' unregistered")
        return True


class LocalDeployment:
    """
    High-level interface for local strategy deployment.
    Provides context manager for easy deployment lifecycle management.
    """
    
    def __init__(self, workspace_dir: str = "~/.wrtrade"):
        """Initialize local deployment manager."""
        self.manager = StrategyManager(workspace_dir)
        self.started = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.manager.start_monitoring()
        self.started = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.started:
            await self.manager.shutdown_all()
            self.started = False
    
    def deploy(self, config: StrategyConfig) -> bool:
        """Deploy a strategy."""
        return self.manager.register_strategy(config)
    
    async def start(self, name: str) -> bool:
        """Start a strategy."""
        return await self.manager.start_strategy(name)
    
    async def stop(self, name: str) -> bool:
        """Stop a strategy."""
        return await self.manager.stop_strategy(name)
    
    def status(self, name: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """Get strategy status."""
        if name:
            return self.manager.get_strategy_status(name)
        else:
            return self.manager.list_strategies()
    
    def remove(self, name: str, remove_data: bool = False) -> bool:
        """Remove a strategy."""
        return self.manager.unregister_strategy(name, remove_data)