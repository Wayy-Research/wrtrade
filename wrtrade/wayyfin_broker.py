"""
WayyFin Broker Adapter - Deploy strategies to WayyFin paper trading.

This is the easiest way to deploy and test strategies - no external broker
credentials needed. Strategies run in paper trading mode with live market data.
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if WayyFin is configured
WAYYFIN_URL = os.getenv("WAYYFIN_URL", "http://localhost:8000")
WAYYFIN_AVAILABLE = True  # Always available - it's our own service


@dataclass
class WayyFinConfig:
    """Configuration for WayyFin paper trading."""
    api_url: str = "http://localhost:8000"
    initial_cash: float = 10000.0
    fee_rate: float = 0.001  # 0.1%
    slippage_bps: float = 5  # 5 basis points


class WayyFinBroker:
    """
    WayyFin Paper Trading Broker.

    Deploys strategies to WayyFin's paper trading system with live market data.
    No external credentials required - perfect for testing and development.
    """

    def __init__(self, config: Optional[WayyFinConfig] = None):
        self.config = config or WayyFinConfig()
        self.api_url = self.config.api_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if WayyFin backend is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.api_url}/health") as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"WayyFin health check failed: {e}")
            return False

    async def create_strategy(
        self,
        name: Optional[str],
        symbol: str,
        signal_code: str,
        params: Optional[Dict] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new strategy in WayyFin.

        Args:
            name: Strategy name (optional - auto-generated if not provided)
            symbol: Trading symbol (e.g., "BTC-USD", "AAPL")
            signal_code: Python code that generates signals
            params: Strategy parameters
            description: Strategy description

        Returns:
            Created strategy data including ID
        """
        session = await self._get_session()

        payload = {
            "symbol": symbol,
            "signal_code": signal_code,
            "params": params or {},
        }

        if name:
            payload["name"] = name
        if description:
            payload["description"] = description

        async with session.post(
            f"{self.api_url}/api/strategy/create",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to create strategy: {error}")

            data = await resp.json()
            return data.get("strategy", data)

    async def run_backtest(
        self,
        strategy_id: str,
        validate: bool = True,
        optimize_kelly: bool = True,
    ) -> Dict[str, Any]:
        """
        Run backtest on a strategy.

        Args:
            strategy_id: Strategy ID
            validate: Run permutation validation
            optimize_kelly: Run Kelly optimization

        Returns:
            Backtest results
        """
        session = await self._get_session()

        async with session.post(
            f"{self.api_url}/api/strategy/{strategy_id}/backtest",
            json={"validate": validate, "optimize_kelly": optimize_kelly}
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to run backtest: {error}")

            return await resp.json()

    async def promote_to_paper(self, strategy_id: str) -> Dict[str, Any]:
        """
        Promote strategy to paper trading stage.

        Args:
            strategy_id: Strategy ID

        Returns:
            Updated strategy data
        """
        session = await self._get_session()

        async with session.post(
            f"{self.api_url}/api/strategy/{strategy_id}/promote/paper"
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to promote to paper: {error}")

            return await resp.json()

    async def start_paper_trading(
        self,
        strategy_id: str,
        initial_cash: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Start live paper trading with real market data.

        Args:
            strategy_id: Strategy ID
            initial_cash: Starting capital (default: $10,000)

        Returns:
            Paper trading status
        """
        session = await self._get_session()

        params = {}
        if initial_cash is not None:
            params["initial_cash"] = initial_cash

        async with session.post(
            f"{self.api_url}/api/strategy/{strategy_id}/paper/start",
            params=params
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to start paper trading: {error}")

            return await resp.json()

    async def stop_paper_trading(self, strategy_id: str) -> Dict[str, Any]:
        """Stop paper trading for a strategy."""
        session = await self._get_session()

        async with session.post(
            f"{self.api_url}/api/strategy/{strategy_id}/paper/stop"
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to stop paper trading: {error}")

            return await resp.json()

    async def get_paper_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get paper trading status for a strategy."""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/api/strategy/{strategy_id}/paper/status"
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to get paper status: {error}")

            return await resp.json()

    async def get_leaderboard(self, limit: int = 50) -> Dict[str, Any]:
        """Get public strategy leaderboard."""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/api/strategy/public/leaderboard",
            params={"limit": limit}
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to get leaderboard: {error}")

            return await resp.json()

    async def get_equity_curves(
        self,
        hours: int = 24,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Get equity curves for top strategies."""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/api/strategy/public/equity-curves",
            params={"hours": hours, "top_n": top_n}
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to get equity curves: {error}")

            return await resp.json()

    async def quick_promote_paper(self, strategy_id: str) -> Dict[str, Any]:
        """
        Quick promote to paper stage (bypasses backtest requirement).

        For development/testing - allows faster iteration.
        In production, strategies should go through proper backtest validation.
        """
        session = await self._get_session()

        async with session.post(
            f"{self.api_url}/api/strategy/{strategy_id}/quick-promote-paper"
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Failed to quick-promote: {error}")
            return await resp.json()

    async def deploy_strategy_file(
        self,
        strategy_file: str,
        name: Optional[str] = None,
        symbol: str = "BTC-USD",
        initial_cash: float = 10000.0,
        auto_start: bool = True,
    ) -> Dict[str, Any]:
        """
        Deploy a strategy file to WayyFin paper trading.

        This is the main entry point for the CLI. It:
        1. Reads the strategy file
        2. Extracts the signal code
        3. Creates the strategy in WayyFin
        4. Runs a quick backtest (optional)
        5. Promotes to paper stage
        6. Starts paper trading

        Args:
            strategy_file: Path to strategy Python file
            name: Strategy name (auto-generated if not provided)
            symbol: Trading symbol
            initial_cash: Starting capital
            auto_start: Start paper trading immediately

        Returns:
            Deployment result with strategy ID and status
        """
        # Read strategy file
        strategy_path = Path(strategy_file)
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_file}")

        signal_code = strategy_path.read_text()

        # Create strategy
        logger.info(f"Creating strategy from {strategy_file}...")
        strategy = await self.create_strategy(
            name=name,
            symbol=symbol,
            signal_code=signal_code,
            params={"max_position": 1.0},
        )

        strategy_id = strategy["id"]
        strategy_name = strategy["name"]

        logger.info(f"Created strategy: {strategy_name} (ID: {strategy_id[:8]}...)")

        result = {
            "strategy_id": strategy_id,
            "name": strategy_name,
            "symbol": symbol,
            "stage": "discovery",
            "status": "created",
        }

        # Try to run backtest first
        try:
            logger.info("Running backtest...")
            backtest_result = await self.run_backtest(strategy_id, validate=False, optimize_kelly=False)
            result["backtest"] = backtest_result
            logger.info("Backtest complete")
        except Exception as e:
            logger.debug(f"Backtest skipped: {e}")

        # Promote to paper stage (use quick-promote for dev)
        try:
            logger.info("Promoting to paper trading stage...")
            await self.quick_promote_paper(strategy_id)
            result["stage"] = "paper"
        except Exception as e:
            logger.debug(f"Quick promote failed: {e}")
            # Try standard promote
            try:
                await self.promote_to_paper(strategy_id)
                result["stage"] = "paper"
            except Exception:
                logger.warning("Could not promote to paper stage")

        # Start paper trading if requested
        if auto_start and result["stage"] == "paper":
            logger.info("Starting paper trading with live data...")
            try:
                status = await self.start_paper_trading(strategy_id, initial_cash)
                result["status"] = "running"
                result["paper_trading"] = status
                logger.info(f"Paper trading started for {strategy_name}")
            except Exception as e:
                result["status"] = "created"
                result["error"] = str(e)
                logger.warning(f"Could not start paper trading: {e}")
        elif auto_start:
            result["error"] = "Strategy not in paper stage - run backtest first"

        return result


# Synchronous wrapper for CLI usage
class WayyFinBrokerSync:
    """Synchronous wrapper for WayyFin broker."""

    def __init__(self, config: Optional[WayyFinConfig] = None):
        self.broker = WayyFinBroker(config)

    def _run(self, coro):
        """Run async coroutine synchronously."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def health_check(self) -> bool:
        return self._run(self.broker.health_check())

    def deploy_strategy_file(self, **kwargs) -> Dict[str, Any]:
        return self._run(self.broker.deploy_strategy_file(**kwargs))

    def start_paper_trading(self, strategy_id: str, **kwargs) -> Dict[str, Any]:
        return self._run(self.broker.start_paper_trading(strategy_id, **kwargs))

    def stop_paper_trading(self, strategy_id: str) -> Dict[str, Any]:
        return self._run(self.broker.stop_paper_trading(strategy_id))

    def get_paper_status(self, strategy_id: str) -> Dict[str, Any]:
        return self._run(self.broker.get_paper_status(strategy_id))

    def get_leaderboard(self, limit: int = 50) -> Dict[str, Any]:
        return self._run(self.broker.get_leaderboard(limit))

    def close(self):
        self._run(self.broker.close())
