"""Historical data service."""

from __future__ import annotations

from core.models.historical import (
    DailyHistoricalRequest,
    HistoricalData,
    IntradayHistoricalRequest,
)
from core.services.base import BaseService


class HistoricalService(BaseService):
    """Service for historical data operations."""

    def get_daily(self, request: DailyHistoricalRequest) -> HistoricalData:
        """Get daily OHLC historical data.

        Args:
            request: Request with security ID, dates, and other parameters.

        Returns:
            Historical OHLC data with timestamps.
        """
        body = request.model_dump(by_alias=True, exclude_none=True)
        response = self._request("POST", "/charts/historical", body)
        return HistoricalData.model_validate(response)

    def get_intraday(self, request: IntradayHistoricalRequest) -> HistoricalData:
        """Get intraday OHLC historical data.

        Args:
            request: Request with security ID, interval, dates, and other parameters.

        Returns:
            Historical OHLC data with timestamps.

        Note:
            Maximum 90 days of data can be fetched at once.
        """
        body = request.model_dump(by_alias=True, exclude_none=True)
        response = self._request("POST", "/charts/intraday", body)
        return HistoricalData.model_validate(response)
