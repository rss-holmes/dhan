"""Data models for market data."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class Interval(Enum):
    """Candle interval/timeframe."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_25 = "25m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


class Tick(BaseModel):
    """Real-time tick data."""

    instrument: str
    timestamp: datetime
    ltp: float = Field(description="Last traded price")
    ltq: int = Field(ge=0, description="Last traded quantity")
    volume: int = Field(ge=0)
    bid_price: float | None = None
    ask_price: float | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None
    oi: int | None = Field(default=None, description="Open interest (derivatives)")

    model_config = {"frozen": True}


class Candle(BaseModel):
    """OHLCV candle data."""

    instrument: str
    timestamp: datetime
    interval: Interval
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    oi: int | None = None

    model_config = {"frozen": True}

    @computed_field
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @computed_field
    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @computed_field
    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open

    @computed_field
    @property
    def body_size(self) -> float:
        """Absolute difference between open and close."""
        return abs(self.close - self.open)

    @computed_field
    @property
    def candle_range(self) -> float:
        """High minus low."""
        return self.high - self.low
