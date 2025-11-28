"""Market data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LTPData(BaseModel):
    """Last traded price data."""

    security_id: str = Field(alias="securityId")
    last_price: float = Field(alias="lastPrice")

    model_config = {"populate_by_name": True}


class OHLCValues(BaseModel):
    """OHLC values."""

    open: float
    high: float
    low: float
    close: float


class OHLCData(BaseModel):
    """OHLC data with LTP."""

    security_id: str = Field(alias="securityId")
    last_price: float = Field(alias="lastPrice")
    ohlc: OHLCValues

    model_config = {"populate_by_name": True}


class DepthEntry(BaseModel):
    """Single entry in market depth."""

    quantity: int
    price: float
    orders: int


class MarketDepth(BaseModel):
    """Market depth data."""

    buy: list[DepthEntry]
    sell: list[DepthEntry]


class MarketQuote(BaseModel):
    """Full market quote with depth."""

    security_id: str = Field(alias="securityId")
    last_price: float = Field(alias="lastPrice")
    average_price: float = Field(default=0, alias="averagePrice")
    net_change: float = Field(default=0, alias="netChange")
    ohlc: OHLCValues
    depth: MarketDepth
    volume: int = 0
    oi: int = 0  # Open Interest
    oi_day_high: int = Field(default=0, alias="oiDayHigh")
    oi_day_low: int = Field(default=0, alias="oiDayLow")
    upper_circuit_limit: float = Field(default=0, alias="upperCircuitLimit")
    lower_circuit_limit: float = Field(default=0, alias="lowerCircuitLimit")
    last_trade_time: str | None = Field(default=None, alias="lastTradeTime")
    last_quantity: int = Field(default=0, alias="lastQuantity")

    model_config = {"populate_by_name": True}
