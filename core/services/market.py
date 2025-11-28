"""Market data service."""

from __future__ import annotations

from core.models.common import ExchangeSegment
from core.models.market import (
    DepthEntry,
    LTPData,
    MarketDepth,
    MarketQuote,
    OHLCData,
    OHLCValues,
)
from core.services.base import BaseService


class MarketService(BaseService):
    """Service for market data operations."""

    def get_ltp(
        self, instruments: dict[ExchangeSegment, list[str]]
    ) -> dict[str, LTPData]:
        """Get last traded price for multiple instruments.

        Args:
            instruments: Dictionary mapping exchange segments to lists of security IDs.
                Example: {ExchangeSegment.NSE_EQ: ["1333", "2885"]}

        Returns:
            Dictionary mapping security IDs to LTP data.

        Note:
            Maximum 1000 instruments per request, 1 request per second rate limit.
        """
        body = {segment.value: ids for segment, ids in instruments.items()}
        response = self._request("POST", "/marketfeed/ltp", body)

        result = {}
        if isinstance(response, dict) and "data" in response:
            for segment_data in response["data"].values():
                if isinstance(segment_data, dict):
                    for sec_id, data in segment_data.items():
                        if isinstance(data, dict):
                            result[sec_id] = LTPData(
                                securityId=sec_id,
                                lastPrice=data.get("last_price", 0),
                            )
        return result

    def get_ohlc(
        self, instruments: dict[ExchangeSegment, list[str]]
    ) -> dict[str, OHLCData]:
        """Get OHLC data for multiple instruments.

        Args:
            instruments: Dictionary mapping exchange segments to lists of security IDs.

        Returns:
            Dictionary mapping security IDs to OHLC data.
        """
        body = {segment.value: ids for segment, ids in instruments.items()}
        response = self._request("POST", "/marketfeed/ohlc", body)

        result = {}
        if isinstance(response, dict) and "data" in response:
            for segment_data in response["data"].values():
                if isinstance(segment_data, dict):
                    for sec_id, data in segment_data.items():
                        if isinstance(data, dict):
                            ohlc = data.get("ohlc", {})
                            result[sec_id] = OHLCData(
                                securityId=sec_id,
                                lastPrice=data.get("last_price", 0),
                                ohlc=OHLCValues(
                                    open=ohlc.get("open", 0),
                                    high=ohlc.get("high", 0),
                                    low=ohlc.get("low", 0),
                                    close=ohlc.get("close", 0),
                                ),
                            )
        return result

    def get_quote(
        self, instruments: dict[ExchangeSegment, list[str]]
    ) -> dict[str, MarketQuote]:
        """Get full market quote with depth for multiple instruments.

        Args:
            instruments: Dictionary mapping exchange segments to lists of security IDs.

        Returns:
            Dictionary mapping security IDs to full market quotes.
        """
        body = {segment.value: ids for segment, ids in instruments.items()}
        response = self._request("POST", "/marketfeed/quote", body)

        result = {}
        if isinstance(response, dict) and "data" in response:
            for segment_data in response["data"].values():
                if isinstance(segment_data, dict):
                    for sec_id, data in segment_data.items():
                        if isinstance(data, dict):
                            ohlc = data.get("ohlc", {})
                            depth = data.get("depth", {"buy": [], "sell": []})
                            result[sec_id] = MarketQuote(
                                securityId=sec_id,
                                lastPrice=data.get("last_price", 0),
                                averagePrice=data.get("average_price", 0),
                                netChange=data.get("net_change", 0),
                                ohlc=OHLCValues(
                                    open=ohlc.get("open", 0),
                                    high=ohlc.get("high", 0),
                                    low=ohlc.get("low", 0),
                                    close=ohlc.get("close", 0),
                                ),
                                depth=MarketDepth(
                                    buy=[
                                        DepthEntry(
                                            quantity=d.get("quantity", 0),
                                            price=d.get("price", 0),
                                            orders=d.get("orders", 0),
                                        )
                                        for d in depth.get("buy", [])
                                    ],
                                    sell=[
                                        DepthEntry(
                                            quantity=d.get("quantity", 0),
                                            price=d.get("price", 0),
                                            orders=d.get("orders", 0),
                                        )
                                        for d in depth.get("sell", [])
                                    ],
                                ),
                                volume=data.get("volume", 0),
                                oi=data.get("oi", 0),
                                oiDayHigh=data.get("oi_day_high", 0),
                                oiDayLow=data.get("oi_day_low", 0),
                                upperCircuitLimit=data.get("upper_circuit_limit", 0),
                                lowerCircuitLimit=data.get("lower_circuit_limit", 0),
                                lastTradeTime=data.get("last_trade_time"),
                                lastQuantity=data.get("last_quantity", 0),
                            )
        return result
