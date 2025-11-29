"""Data feed protocol definitions."""

from typing import Callable, Iterator, Protocol, runtime_checkable

from algo.data.models import Candle, Tick


@runtime_checkable
class DataFeed(Protocol):
    """
    Protocol for real-time market data feeds (WebSocket-based).

    For historical data, use HistoricalDataFeed or core.services.HistoricalService.
    """

    async def connect(self) -> None:
        """Establish connection to data source."""
        ...

    async def disconnect(self) -> None:
        """Close connection to data source."""
        ...

    async def subscribe(self, instruments: list[str]) -> None:
        """
        Subscribe to market data for given instruments.

        Args:
            instruments: List of instrument identifiers
                         Format: "{exchange_segment}:{security_id}"
                         Example: ["NSE_EQ:1333", "NSE_FNO:43225"]
        """
        ...

    async def unsubscribe(self, instruments: list[str]) -> None:
        """Unsubscribe from market data for given instruments."""
        ...

    def on_tick(self, callback: Callable[[Tick], None]) -> None:
        """Register callback for tick data."""
        ...

    def on_candle(self, callback: Callable[[Candle], None]) -> None:
        """Register callback for candle data."""
        ...


@runtime_checkable
class HistoricalDataFeed(Protocol):
    """Protocol for historical data iteration (backtesting)."""

    def set_instruments(self, instruments: list[str]) -> None:
        """Set instruments to iterate over."""
        ...

    def __iter__(self) -> Iterator[Candle]:
        """Iterate through historical candles in chronological order."""
        ...

    def get_candles(
        self,
        instrument: str,
        count: int,
        end_offset: int = 0,
    ) -> list[Candle]:
        """Get historical candles relative to current position."""
        ...
