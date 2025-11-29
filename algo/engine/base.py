"""Base engine class defining the execution engine interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algo.broker.models import Position
    from algo.data.models import Candle
    from algo.strategy.base import BaseStrategy, Signal


class BaseEngine(ABC):
    """
    Abstract base class for execution engines.

    Defines the interface that strategy context uses to interact with
    the engine, and that concrete engines must implement.
    """

    @abstractmethod
    def add_strategy(self, strategy: BaseStrategy) -> str:
        """
        Add a strategy to the engine.

        Args:
            strategy: Strategy instance to add

        Returns:
            Strategy ID for identification
        """
        pass

    @abstractmethod
    def get_historical(
        self,
        instrument: str,
        periods: int,
        field: str = "close",
    ) -> list[float]:
        """
        Get historical data for indicator calculation.

        Args:
            instrument: Instrument identifier
            periods: Number of periods to retrieve
            field: Field to extract (open, high, low, close, volume)

        Returns:
            List of values
        """
        pass

    @abstractmethod
    def get_candle(self, instrument: str, offset: int = 0) -> Candle | None:
        """
        Get a specific historical candle.

        Args:
            instrument: Instrument identifier
            offset: Offset from current (0 = current, -1 = previous)

        Returns:
            Candle or None if not available
        """
        pass

    @abstractmethod
    def get_position(self, instrument: str) -> Position | None:
        """
        Get current position for instrument.

        Args:
            instrument: Instrument identifier

        Returns:
            Position or None if no position
        """
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Get available cash balance."""
        pass

    @abstractmethod
    def get_equity(self) -> float:
        """Get total portfolio equity."""
        pass

    @abstractmethod
    def submit_order(self, strategy_id: str, signal: Signal) -> str:
        """
        Submit an order based on a signal.

        Args:
            strategy_id: ID of the strategy submitting the order
            signal: Trading signal

        Returns:
            Order ID
        """
        pass
