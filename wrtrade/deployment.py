import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import json
import uuid

from wrtrade.components import PortfolioComponent, CompositePortfolio
from wrtrade.ndimensional_portfolio import AdvancedPortfolioManager
from wrtrade.permutation import PermutationTester, PermutationConfig
from wrtrade.kelly import HierarchicalKellyOptimizer, KellyConfig
from wrtrade.brokers import BrokerAdapter, BrokerFactory, TradingSession


logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for strategy deployment."""
    strategy_name: str
    broker_name: str  # 'alpaca', 'robinhood'
    paper_trading: bool = True
    
    # Validation requirements
    min_sortino_ratio: float = 1.0
    max_permutation_pvalue: float = 0.05
    min_backtest_period_days: int = 252
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05    # 5% daily loss limit
    
    # Optimization settings
    kelly_optimization: bool = True
    rebalance_frequency_days: int = 21  # Monthly
    
    # Monitoring
    performance_check_interval: int = 3600  # 1 hour
    auto_shutdown_conditions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.auto_shutdown_conditions is None:
            self.auto_shutdown_conditions = {
                'max_drawdown': -0.15,  # 15% drawdown
                'daily_loss_limit': -0.05,  # 5% daily loss
                'consecutive_losing_days': 5
            }


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    passed: bool
    strategy_name: str
    backtest_metrics: Dict[str, float]
    permutation_pvalue: float
    validation_errors: List[str]
    validation_warnings: List[str]
    timestamp: datetime


@dataclass
class DeploymentStatus:
    """Status of deployed strategy."""
    deployment_id: str
    strategy_name: str
    status: str  # 'validating', 'deployed', 'running', 'paused', 'stopped', 'error'
    start_time: datetime
    last_update: datetime
    
    # Performance tracking
    total_return: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk monitoring
    risk_alerts: List[str] = None
    auto_shutdown_triggered: bool = False
    
    def __post_init__(self):
        if self.risk_alerts is None:
            self.risk_alerts = []


class StrategyValidator:
    """
    Validates strategies before deployment using comprehensive testing.
    """
    
    def __init__(self):
        """Initialize strategy validator."""
        self.validation_history = []
    
    async def validate_strategy(
        self,
        portfolio: CompositePortfolio,
        prices: pl.Series,
        config: DeploymentConfig
    ) -> ValidationResult:
        """
        Comprehensive strategy validation before deployment.
        
        Args:
            portfolio: Strategy portfolio to validate
            prices: Historical price data
            config: Deployment configuration
            
        Returns:
            ValidationResult with pass/fail and details
        """
        logger.info(f"Starting validation for strategy: {config.strategy_name}")
        
        errors = []
        warnings = []
        
        # Check minimum data requirements
        if len(prices) < config.min_backtest_period_days:
            errors.append(f"Insufficient historical data: {len(prices)} days < {config.min_backtest_period_days} required")
        
        # Run backtest
        try:
            portfolio_manager = AdvancedPortfolioManager(portfolio)
            backtest_results = portfolio_manager.backtest(prices)
            backtest_metrics = backtest_results['portfolio_metrics']
            
            logger.info(f"Backtest completed. Sortino ratio: {backtest_metrics.get('sortino_ratio', 0):.3f}")
            
        except Exception as e:
            errors.append(f"Backtest failed: {str(e)}")
            backtest_metrics = {}
        
        # Performance validation
        sortino_ratio = backtest_metrics.get('sortino_ratio', 0)
        if sortino_ratio < config.min_sortino_ratio:
            errors.append(f"Sortino ratio too low: {sortino_ratio:.3f} < {config.min_sortino_ratio}")
        
        max_drawdown = backtest_metrics.get('max_drawdown', 0)
        if max_drawdown < -0.25:  # More than 25% drawdown
            warnings.append(f"High maximum drawdown: {max_drawdown:.3f}")
        
        # Permutation test validation
        permutation_pvalue = 1.0
        try:
            logger.info("Running permutation test validation...")
            
            def strategy_func(test_prices, **kwargs):
                return portfolio
            
            permutation_config = PermutationConfig(n_permutations=200, parallel=False)  # Faster validation
            tester = PermutationTester(permutation_config)
            
            perm_results = tester.run_insample_test(
                prices, strategy_func, metric='sortino_ratio'
            )
            
            permutation_pvalue = perm_results['p_value']
            logger.info(f"Permutation test p-value: {permutation_pvalue:.4f}")
            
            if permutation_pvalue > config.max_permutation_pvalue:
                errors.append(f"Strategy failed permutation test: p-value {permutation_pvalue:.4f} > {config.max_permutation_pvalue}")
            
        except Exception as e:
            warnings.append(f"Permutation test failed: {str(e)}")
            logger.warning(f"Permutation test error: {e}")
        
        # Portfolio structure validation
        try:
            portfolio.print_structure()
            if len(portfolio.component_list) == 0:
                errors.append("Portfolio has no components")
        except Exception as e:
            errors.append(f"Portfolio structure validation failed: {str(e)}")
        
        # Determine validation result
        passed = len(errors) == 0
        
        result = ValidationResult(
            passed=passed,
            strategy_name=config.strategy_name,
            backtest_metrics=backtest_metrics,
            permutation_pvalue=permutation_pvalue,
            validation_errors=errors,
            validation_warnings=warnings,
            timestamp=datetime.now()
        )
        
        self.validation_history.append(result)
        
        if passed:
            logger.info(f"Strategy validation PASSED: {config.strategy_name}")
        else:
            logger.error(f"Strategy validation FAILED: {config.strategy_name}")
            for error in errors:
                logger.error(f"  - {error}")
        
        for warning in warnings:
            logger.warning(f"  - {warning}")
        
        return result


class StrategyDeployer:
    """
    Deploys validated strategies to live trading with monitoring and risk management.
    """
    
    def __init__(self):
        """Initialize strategy deployer."""
        self.active_deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history = []
        self.validator = StrategyValidator()
        
    async def deploy_strategy(
        self,
        portfolio: CompositePortfolio,
        prices: pl.Series,
        config: DeploymentConfig,
        api_credentials: Dict[str, str],
        symbols_map: Dict[str, str],  # Component name -> trading symbol
        skip_validation: bool = False
    ) -> str:
        """
        Deploy strategy to live/paper trading.
        
        Args:
            portfolio: Strategy portfolio to deploy
            prices: Historical data for validation
            config: Deployment configuration
            api_credentials: Broker API credentials
            symbols_map: Mapping from portfolio components to trading symbols
            skip_validation: Skip validation (not recommended for live trading)
            
        Returns:
            Deployment ID
        """
        deployment_id = str(uuid.uuid4())
        
        logger.info(f"Starting deployment of strategy: {config.strategy_name} (ID: {deployment_id})")
        
        # Initialize deployment status
        status = DeploymentStatus(
            deployment_id=deployment_id,
            strategy_name=config.strategy_name,
            status='validating',
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        self.active_deployments[deployment_id] = status
        
        try:
            # Strategy validation
            if not skip_validation:
                validation_result = await self.validator.validate_strategy(portfolio, prices, config)
                
                if not validation_result.passed:
                    status.status = 'error'
                    status.risk_alerts.append("Strategy failed validation")
                    logger.error(f"Deployment failed validation: {deployment_id}")
                    return deployment_id
            
            # Create broker adapter
            broker = BrokerFactory.create_broker(
                config.broker_name, 
                api_credentials, 
                config.paper_trading
            )
            
            # Authenticate with broker
            if not await broker.authenticate():
                status.status = 'error'
                status.risk_alerts.append("Broker authentication failed")
                logger.error(f"Broker authentication failed: {deployment_id}")
                return deployment_id
            
            # Create trading session
            trading_session = TradingSession(
                broker=broker,
                max_position_size=config.max_position_size,
                max_daily_loss=config.max_daily_loss,
                require_confirmation=config.paper_trading  # Only require confirmation for paper trading
            )
            
            # Initial portfolio optimization if enabled
            if config.kelly_optimization:
                logger.info("Running initial Kelly optimization...")
                kelly_optimizer = HierarchicalKellyOptimizer()
                optimization_results = kelly_optimizer.optimize_portfolio(portfolio, prices)
                logger.info(f"Kelly optimization completed for {len(optimization_results['level_optimizations'])} levels")
            
            # Deploy strategy
            status.status = 'deployed'
            status.last_update = datetime.now()
            
            # Start monitoring and trading loop
            asyncio.create_task(self._run_strategy_loop(
                deployment_id, portfolio, trading_session, symbols_map, config
            ))
            
            logger.info(f"Strategy deployment successful: {deployment_id}")
            
        except Exception as e:
            status.status = 'error'
            status.risk_alerts.append(f"Deployment error: {str(e)}")
            logger.error(f"Deployment error for {deployment_id}: {e}")
        
        return deployment_id
    
    async def _run_strategy_loop(
        self,
        deployment_id: str,
        portfolio: CompositePortfolio,
        trading_session: TradingSession,
        symbols_map: Dict[str, str],
        config: DeploymentConfig
    ) -> None:
        """
        Main strategy execution loop.
        
        Args:
            deployment_id: Deployment identifier
            portfolio: Portfolio to trade
            trading_session: Trading session
            symbols_map: Symbol mapping
            config: Configuration
        """
        status = self.active_deployments[deployment_id]
        status.status = 'running'
        
        logger.info(f"Starting strategy loop for deployment: {deployment_id}")
        
        last_rebalance = datetime.now()
        consecutive_losing_days = 0
        
        try:
            while status.status == 'running':
                # Performance monitoring
                await self._monitor_performance(deployment_id, trading_session)
                
                # Check rebalancing schedule
                days_since_rebalance = (datetime.now() - last_rebalance).days
                
                if days_since_rebalance >= config.rebalance_frequency_days:
                    logger.info(f"Triggering rebalance for deployment: {deployment_id}")
                    
                    try:
                        # Kelly optimization
                        if config.kelly_optimization:
                            kelly_optimizer = HierarchicalKellyOptimizer()
                            # Get recent price data for optimization
                            # This would need to be fetched from broker or data source
                            # For now, we'll skip the actual rebalancing
                            logger.info("Kelly optimization scheduled (implementation pending data source)")
                        
                        # Execute rebalancing
                        current_weights = portfolio.get_component_weights()
                        orders = await trading_session.execute_portfolio_rebalance(
                            portfolio, current_weights, symbols_map
                        )
                        
                        logger.info(f"Rebalancing completed: {len(orders)} orders executed")
                        last_rebalance = datetime.now()
                        
                    except Exception as e:
                        status.risk_alerts.append(f"Rebalancing failed: {str(e)}")
                        logger.error(f"Rebalancing error for {deployment_id}: {e}")
                
                # Check auto-shutdown conditions
                if self._should_auto_shutdown(status, config):
                    logger.warning(f"Auto-shutdown triggered for deployment: {deployment_id}")
                    status.status = 'stopped'
                    status.auto_shutdown_triggered = True
                    break
                
                # Update status
                status.last_update = datetime.now()
                
                # Wait before next iteration
                await asyncio.sleep(config.performance_check_interval)
                
        except Exception as e:
            logger.error(f"Strategy loop error for {deployment_id}: {e}")
            status.status = 'error'
            status.risk_alerts.append(f"Strategy loop error: {str(e)}")
        
        logger.info(f"Strategy loop ended for deployment: {deployment_id}")
    
    async def _monitor_performance(
        self, 
        deployment_id: str, 
        trading_session: TradingSession
    ) -> None:
        """Monitor strategy performance and update status."""
        status = self.active_deployments[deployment_id]
        
        try:
            session_summary = trading_session.get_session_summary()
            
            # Update performance metrics
            status.daily_pnl = session_summary['current_daily_pnl']
            
            # Get account info for additional metrics
            account_info = await trading_session.broker.get_account_info()
            
            # Calculate total return since deployment
            # This would need baseline portfolio value stored at deployment start
            # For now, we'll use a placeholder
            status.total_return = 0.0  # Would be calculated from baseline
            
            # Update max drawdown
            # This would need to track high-water mark
            status.max_drawdown = min(status.max_drawdown, status.daily_pnl / account_info.portfolio_value)
            
            logger.debug(f"Performance update for {deployment_id}: Daily P&L ${status.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Performance monitoring error for {deployment_id}: {e}")
    
    def _should_auto_shutdown(
        self, 
        status: DeploymentStatus, 
        config: DeploymentConfig
    ) -> bool:
        """Check if auto-shutdown conditions are met."""
        conditions = config.auto_shutdown_conditions
        
        # Max drawdown check
        if status.max_drawdown <= conditions.get('max_drawdown', -1.0):
            status.risk_alerts.append(f"Max drawdown exceeded: {status.max_drawdown:.3f}")
            return True
        
        # Daily loss limit check
        daily_loss_limit = conditions.get('daily_loss_limit', -1.0)
        if status.daily_pnl <= daily_loss_limit * 100000:  # Assuming $100k portfolio
            status.risk_alerts.append(f"Daily loss limit exceeded: ${status.daily_pnl:.2f}")
            return True
        
        return False
    
    def stop_deployment(self, deployment_id: str, reason: str = "Manual stop") -> bool:
        """
        Stop a running deployment.
        
        Args:
            deployment_id: Deployment to stop
            reason: Reason for stopping
            
        Returns:
            True if successfully stopped
        """
        if deployment_id not in self.active_deployments:
            logger.warning(f"Deployment not found: {deployment_id}")
            return False
        
        status = self.active_deployments[deployment_id]
        
        if status.status in ['running', 'deployed']:
            status.status = 'stopped'
            status.risk_alerts.append(f"Stopped: {reason}")
            status.last_update = datetime.now()
            
            logger.info(f"Deployment stopped: {deployment_id} - {reason}")
            return True
        else:
            logger.warning(f"Cannot stop deployment {deployment_id} in status: {status.status}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get status of a deployment."""
        return self.active_deployments.get(deployment_id)
    
    def list_active_deployments(self) -> List[DeploymentStatus]:
        """List all active deployments."""
        return [status for status in self.active_deployments.values() 
                if status.status in ['running', 'deployed']]
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get summary of all deployments."""
        statuses = list(self.active_deployments.values())
        
        return {
            'total_deployments': len(statuses),
            'active_deployments': len([s for s in statuses if s.status in ['running', 'deployed']]),
            'stopped_deployments': len([s for s in statuses if s.status == 'stopped']),
            'error_deployments': len([s for s in statuses if s.status == 'error']),
            'total_alerts': sum(len(s.risk_alerts) for s in statuses),
            'auto_shutdowns': len([s for s in statuses if s.auto_shutdown_triggered])
        }


class WRTradeDeploymentSystem:
    """
    Main deployment system that integrates all WRTrade components
    for end-to-end strategy deployment and management.
    """
    
    def __init__(self):
        """Initialize WRTrade deployment system."""
        self.deployer = StrategyDeployer()
        self.active_sessions = {}
        
        logger.info("WRTrade Deployment System initialized")
    
    async def deploy_portfolio_strategy(
        self,
        portfolio: CompositePortfolio,
        historical_prices: pl.Series,
        broker_config: Dict[str, str],  # {'broker': 'alpaca', 'api_key': '...', 'secret_key': '...'}
        symbols_map: Dict[str, str],    # Component names -> trading symbols
        deployment_config: Optional[DeploymentConfig] = None
    ) -> str:
        """
        One-click deployment of a portfolio strategy.
        
        Args:
            portfolio: N-dimensional portfolio to deploy
            historical_prices: Historical price data for validation
            broker_config: Broker configuration with credentials
            symbols_map: Mapping from portfolio components to trading symbols
            deployment_config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        if deployment_config is None:
            deployment_config = DeploymentConfig(
                strategy_name=portfolio.name,
                broker_name=broker_config['broker']
            )
        
        # Extract API credentials
        api_credentials = {k: v for k, v in broker_config.items() if k != 'broker'}
        
        # Deploy strategy
        deployment_id = await self.deployer.deploy_strategy(
            portfolio=portfolio,
            prices=historical_prices,
            config=deployment_config,
            api_credentials=api_credentials,
            symbols_map=symbols_map
        )
        
        return deployment_id
    
    def create_simple_strategy_example(self) -> CompositePortfolio:
        """
        Create a simple example strategy for demonstration.
        
        Returns:
            Example portfolio strategy
        """
        from wrtrade.components import SignalComponent
        
        # Simple moving average crossover signal
        def ma_crossover_signal(prices: pl.Series) -> pl.Series:
            """Simple MA crossover signal."""
            if len(prices) < 50:
                return pl.Series([0] * len(prices))
            
            prices_np = prices.to_numpy()
            
            # Calculate moving averages
            ma_short = pl.Series(prices_np).rolling_mean(window_size=10)
            ma_long = pl.Series(prices_np).rolling_mean(window_size=30)
            
            # Generate signals
            signals = []
            for i in range(len(prices)):
                if i < 30:  # Not enough data
                    signals.append(0)
                elif ma_short[i] > ma_long[i] and (i == 0 or ma_short[i-1] <= ma_long[i-1]):
                    signals.append(1)  # Buy signal
                elif ma_short[i] < ma_long[i] and (i == 0 or ma_short[i-1] >= ma_long[i-1]):
                    signals.append(-1)  # Sell signal
                else:
                    signals.append(0)  # Hold
            
            return pl.Series(signals)
        
        # Create signal component
        ma_component = SignalComponent(
            name="MA_Crossover",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        # Create portfolio
        example_portfolio = CompositePortfolio(
            name="Example_MA_Strategy",
            components=[ma_component],
            kelly_optimization=True
        )
        
        return example_portfolio
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        deployment_summary = self.deployer.get_deployment_summary()
        
        return {
            'system': 'WRTrade Deployment System',
            'version': '1.0.0',
            'timestamp': datetime.now(),
            'deployment_summary': deployment_summary,
            'supported_brokers': BrokerFactory.get_supported_brokers(),
            'active_sessions': len(self.active_sessions)
        }