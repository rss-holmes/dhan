"""Historical data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.common import ExchangeSegment, ExpiryCode, Instrument, IntradayInterval


class DailyHistoricalRequest(BaseModel):
    """Request model for daily historical data."""

    security_id: str = Field(alias="securityId")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    instrument: Instrument
    expiry_code: ExpiryCode = Field(default=ExpiryCode.NOT_APPLICABLE, alias="expiryCode")
    from_date: str = Field(alias="fromDate")  # YYYY-MM-DD
    to_date: str = Field(alias="toDate")  # YYYY-MM-DD (non-inclusive)
    oi: bool = False  # Include Open Interest

    model_config = {"populate_by_name": True}


class IntradayHistoricalRequest(BaseModel):
    """Request model for intraday historical data."""

    security_id: str = Field(alias="securityId")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    instrument: Instrument
    interval: IntradayInterval
    from_date: str = Field(alias="fromDate")  # datetime string
    to_date: str = Field(alias="toDate")  # datetime string
    oi: bool = False  # Include Open Interest

    model_config = {"populate_by_name": True}


class HistoricalData(BaseModel):
    """Response model for historical data."""

    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[int]
    timestamp: list[float]  # Unix timestamps
    open_interest: list[int] = Field(default_factory=list, alias="open_interest")

    model_config = {"populate_by_name": True}
