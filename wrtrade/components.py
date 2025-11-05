import polars as pl
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass


@dataclass
class AllocationWeights:
    """Container for component allocation weights with metadata."""
    weights: Dict[str, float]
    timestamp: Optional[str] = None
    rebalance_reason: Optional[str] = None
    
    def normalize(self) -> 'AllocationWeights':
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total == 0:
            return self
        normalized_weights = {k: v / total for k, v in self.weights.items()}
        return AllocationWeights(
            weights=normalized_weights,
            timestamp=self.timestamp,
            rebalance_reason=self.rebalance_reason
        )


class PortfolioComponent(ABC):
    """
    Abstract base class for portfolio components.
    Supports both direct signals and nested portfolios for N-dimensional composition.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize portfolio component.
        
        Args:
            name: Unique identifier for this component
            weight: Initial allocation weight (can be dynamically adjusted)
        """
        self.name = name
        self.weight = weight
        self._returns = None
        self._positions = None
        self._metrics = None
        
    @abstractmethod
    def generate_signals(self, prices: pl.Series) -> pl.Series:
        """
        Generate trading signals for given price data.
        
        Args:
            prices: Price series
            
        Returns:
            Signal series (-1, 0, 1)
        """
        pass
    
    @abstractmethod
    def calculate_returns(self, prices: pl.Series) -> pl.Series:
        """
        Calculate component returns for given price data.
        
        Args:
            prices: Price series
            
        Returns:
            Return series
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for this component.
        
        Returns:
            Dictionary of performance metrics
        """
        pass
    
    def set_weight(self, weight: float) -> None:
        """Update allocation weight for this component."""
        self.weight = weight
        
    def get_weight(self) -> float:
        """Get current allocation weight."""
        return self.weight


class SignalComponent(PortfolioComponent):
    """
    Portfolio component wrapping a direct trading signal.
    """
    
    def __init__(
        self, 
        name: str, 
        signal_func: callable, 
        weight: float = 1.0,
        max_position: float = float('inf'),
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ):
        """
        Initialize signal component.
        
        Args:
            name: Component name
            signal_func: Function that takes prices and returns signals
            weight: Allocation weight
            max_position: Maximum position size
            take_profit: Take profit threshold
            stop_loss: Stop loss threshold
        """
        super().__init__(name, weight)
        self.signal_func = signal_func
        self.max_position = max_position
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
    def generate_signals(self, prices: pl.Series) -> pl.Series:
        """Generate signals using the signal function."""
        return self.signal_func(prices)
    
    def calculate_returns(self, prices: pl.Series) -> pl.Series:
        """Calculate returns from signal-based trading."""
        from wrtrade.trade import calculate_positions, calculate_returns, apply_take_profit_stop_loss
        
        signals = self.generate_signals(prices)
        positions = calculate_positions(signals, self.max_position)
        
        if self.take_profit is not None or self.stop_loss is not None:
            positions = apply_take_profit_stop_loss(
                prices, positions, signals, self.take_profit, self.stop_loss
            )
        
        self._positions = positions
        self._returns = calculate_returns(prices, positions) * self.weight
        return self._returns
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this signal."""
        from wrtrade.metrics import calculate_all_metrics
        
        if self._returns is None:
            return {}
        
        self._metrics = calculate_all_metrics(self._returns)
        return self._metrics


class CompositePortfolio(PortfolioComponent):
    """
    Portfolio component that contains other components (signals or portfolios).
    Implements the composite pattern for N-dimensional portfolio construction.
    """
    
    def __init__(
        self, 
        name: str, 
        components: List[PortfolioComponent],
        weight: float = 1.0,
        rebalance_frequency: Optional[int] = None,
        kelly_optimization: bool = False
    ):
        """
        Initialize composite portfolio.
        
        Args:
            name: Portfolio name
            components: List of child components
            weight: Allocation weight
            rebalance_frequency: How often to rebalance (in periods)
            kelly_optimization: Whether to use Kelly optimization for weights
        """
        super().__init__(name, weight)
        self.components = {comp.name: comp for comp in components}
        self.component_list = components
        self.rebalance_frequency = rebalance_frequency
        self.kelly_optimization = kelly_optimization
        self._allocation_history = []
        
    def add_component(self, component: PortfolioComponent) -> None:
        """Add a new component to the portfolio."""
        self.components[component.name] = component
        self.component_list.append(component)
        
    def remove_component(self, name: str) -> None:
        """Remove a component from the portfolio."""
        if name in self.components:
            comp = self.components.pop(name)
            self.component_list.remove(comp)
    
    def generate_signals(self, prices: pl.Series) -> pl.Series:
        """
        Generate aggregate signals from all components.
        This is a weighted combination of all component signals.
        """
        if not self.components:
            return pl.Series(values=[0] * len(prices))
            
        total_signal = pl.Series(values=[0.0] * len(prices))
        total_weight = 0
        
        for component in self.component_list:
            signals = component.generate_signals(prices)
            total_signal = total_signal + (signals * component.weight)
            total_weight += abs(component.weight)
            
        if total_weight > 0:
            total_signal = total_signal / total_weight
            
        return total_signal.clip(-1, 1)
    
    def calculate_returns(self, prices: pl.Series) -> pl.Series:
        """
        Calculate portfolio returns as weighted sum of component returns.
        Optionally applies Kelly optimization for dynamic rebalancing.
        """
        if not self.components:
            self._returns = pl.Series(values=[0.0] * len(prices))
            return self._returns
        
        # Calculate returns for each component
        component_returns = {}
        for component in self.component_list:
            component_returns[component.name] = component.calculate_returns(prices)
        
        # Apply dynamic rebalancing if enabled
        if self.kelly_optimization:
            self._returns = self._calculate_kelly_optimized_returns(
                prices, component_returns
            )
        else:
            # Simple weighted combination
            total_returns = pl.Series(values=[0.0] * len(prices))
            for component in self.component_list:
                total_returns = total_returns + component_returns[component.name]
            self._returns = total_returns * self.weight
            
        return self._returns
    
    def _calculate_kelly_optimized_returns(
        self, 
        prices: pl.Series, 
        component_returns: Dict[str, pl.Series]
    ) -> pl.Series:
        """
        Calculate returns with Kelly optimization for dynamic weight allocation.
        
        Args:
            prices: Price series
            component_returns: Dict mapping component names to return series
            
        Returns:
            Kelly-optimized return series
        """
        # This is a placeholder for the Kelly optimization logic
        # Will be implemented in the Kelly optimization system
        total_returns = pl.Series(values=[0.0] * len(prices))
        
        # For now, use equal weighting as default
        n_components = len(self.component_list)
        if n_components > 0:
            equal_weight = 1.0 / n_components
            for component_name, returns in component_returns.items():
                total_returns = total_returns + (returns * equal_weight)
                
        return total_returns * self.weight
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the composite portfolio."""
        from wrtrade.metrics import calculate_all_metrics
        
        if self._returns is None:
            return {}
        
        self._metrics = calculate_all_metrics(self._returns)
        
        # Add component-level metrics
        component_metrics = {}
        for component in self.component_list:
            comp_metrics = component.get_performance_metrics()
            for metric_name, value in comp_metrics.items():
                component_metrics[f"{component.name}_{metric_name}"] = value
                
        self._metrics.update(component_metrics)
        return self._metrics
    
    def get_component_weights(self) -> Dict[str, float]:
        """Get current weights for all components."""
        return {comp.name: comp.weight for comp in self.component_list}
    
    def rebalance_components(self, new_weights: Dict[str, float]) -> None:
        """
        Rebalance component weights.
        
        Args:
            new_weights: Dictionary mapping component names to new weights
        """
        allocation = AllocationWeights(
            weights=new_weights.copy(),
            rebalance_reason="manual_rebalance"
        ).normalize()
        
        for component in self.component_list:
            if component.name in allocation.weights:
                component.set_weight(allocation.weights[component.name])
                
        self._allocation_history.append(allocation)
    
    def get_allocation_history(self) -> List[AllocationWeights]:
        """Get history of allocation changes."""
        return self._allocation_history.copy()
    
    def print_structure(self, indent: int = 0) -> None:
        """Print the portfolio structure in a tree format."""
        prefix = "  " * indent
        print(f"{prefix}{self.name} (weight: {self.weight:.3f})")
        
        for component in self.component_list:
            if isinstance(component, CompositePortfolio):
                component.print_structure(indent + 1)
            else:
                print(f"{prefix}  {component.name} (weight: {component.weight:.3f})")