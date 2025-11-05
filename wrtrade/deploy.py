"""
Simple deployment system for wrtrade.

Philosophy: Simple is better than complex.
- One function to deploy a portfolio
- Minimal configuration
- Clear defaults
"""

import polars as pl
import asyncio
import os
from typing import Dict, Optional
from dataclasses import dataclass, field

from wrtrade.components import CompositePortfolio
from wrtrade.brokers import BrokerAdapter, BrokerFactory


@dataclass
class DeployConfig:
    """Simple deployment configuration."""
    broker: str = "alpaca"
    paper: bool = True
    max_position_pct: float = 0.10  # 10% of portfolio
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit

    # Optional validation
    validate: bool = True
    min_sortino: float = 1.0

    # Get credentials from environment by default
    api_key: Optional[str] = None
    secret_key: Optional[str] = None

    def __post_init__(self):
        # Auto-load from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv(f"{self.broker.upper()}_API_KEY")
        if not self.secret_key:
            self.secret_key = os.getenv(f"{self.broker.upper()}_SECRET_KEY")


async def deploy(
    portfolio: CompositePortfolio,
    symbols: Dict[str, str],  # component_name -> symbol
    config: Optional[DeployConfig] = None
) -> str:
    """
    Deploy a portfolio to live/paper trading.

    Args:
        portfolio: Portfolio to deploy
        symbols: Mapping of component names to trading symbols
        config: Optional deployment configuration

    Returns:
        Deployment ID

    Example:
        >>> portfolio = builder.create_portfolio("MyStrategy", components)
        >>> deployment_id = await deploy(
        ...     portfolio,
        ...     symbols={"MA_Fast": "AAPL", "MA_Slow": "AAPL"},
        ...     config=DeployConfig(broker="alpaca", paper=True)
        ... )
    """
    if config is None:
        config = DeployConfig()

    # Create broker
    credentials = {
        'api_key': config.api_key,
        'secret_key': config.secret_key
    }

    broker = BrokerFactory.create_broker(
        config.broker,
        credentials,
        paper_trading=config.paper
    )

    # Authenticate
    if not await broker.authenticate():
        raise RuntimeError(f"Failed to authenticate with {config.broker}")

    # Get account info
    account = await broker.get_account_info()
    print(f"Connected to {config.broker}")
    print(f"Account value: ${account.portfolio_value:,.2f}")

    # TODO: Implement actual trading loop
    # For now, just return a deployment ID
    import uuid
    deployment_id = str(uuid.uuid4())[:8]

    print(f"Portfolio deployed: {deployment_id}")
    return deployment_id


def validate_strategy(portfolio: CompositePortfolio, prices: pl.Series, min_sortino: float = 1.0) -> bool:
    """
    Simple validation: backtest and check if Sortino ratio meets threshold.

    Args:
        portfolio: Portfolio to validate
        prices: Historical prices
        min_sortino: Minimum Sortino ratio required

    Returns:
        True if validation passes
    """
    from wrtrade.ndimensional_portfolio import AdvancedPortfolioManager

    manager = AdvancedPortfolioManager(portfolio)
    results = manager.backtest(prices)

    sortino = results['portfolio_metrics'].get('sortino_ratio', 0)
    passed = sortino >= min_sortino

    print(f"Validation: Sortino={sortino:.2f} (required: {min_sortino:.2f}) - {'PASS' if passed else 'FAIL'}")

    return passed
