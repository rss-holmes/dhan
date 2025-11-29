"""Dhan WebSocket market data feed."""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import datetime
from typing import Any, Callable

import websockets

from algo.data.models import Candle, Tick


class DhanDataFeed:
    """
    Dhan WebSocket market data feed implementation.

    Connects to wss://api-feed.dhan.co with authentication.
    Handles binary message parsing for tick data.
    """

    WS_URL = "wss://api-feed.dhan.co"

    # Request codes
    SUBSCRIBE = 15
    DISCONNECT = 12

    # Response codes
    TICKER_PACKET = 2
    QUOTE_PACKET = 4
    OI_PACKET = 5
    PREV_CLOSE_PACKET = 6
    FULL_PACKET = 8
    DISCONNECT_RESPONSE = 50

    # Segment mapping
    SEGMENT_MAP = {
        0: "NSE_EQ",
        1: "NSE_FNO",
        2: "NSE_CURRENCY",
        3: "BSE_EQ",
        4: "MCX_COMM",
        5: "BSE_CURRENCY",
        6: "BSE_FNO",
    }

    def __init__(
        self,
        client_id: str,
        access_token: str,
        on_tick: Callable[[Tick], None] | None = None,
        on_candle: Callable[[Candle], None] | None = None,
    ):
        self._client_id = client_id
        self._access_token = access_token
        self._ws: Any = None  # WebSocketClientProtocol
        self._subscribed: set[str] = set()
        self._tick_callbacks: list[Callable[[Tick], None]] = []
        self._candle_callbacks: list[Callable[[Candle], None]] = []
        self._running = False
        self._message_task: asyncio.Task[None] | None = None

        if on_tick:
            self._tick_callbacks.append(on_tick)
        if on_candle:
            self._candle_callbacks.append(on_candle)

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        url = (
            f"{self.WS_URL}?version=2"
            f"&token={self._access_token}"
            f"&clientId={self._client_id}"
            f"&authType=2"
        )
        self._ws = await websockets.connect(url)
        self._running = True
        self._message_task = asyncio.create_task(self._message_loop())

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        if self._ws:
            self._running = False
            try:
                await self._ws.send(json.dumps({"RequestCode": self.DISCONNECT}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            self._message_task = None

    async def subscribe(self, instruments: list[str]) -> None:
        """
        Subscribe to instruments.

        Args:
            instruments: List in format "EXCHANGE_SEGMENT:SECURITY_ID"
        """
        if not self._ws:
            raise RuntimeError("Not connected")

        # Convert to Dhan format
        instrument_list = []
        for inst in instruments:
            segment, security_id = inst.split(":")
            instrument_list.append(
                {"ExchangeSegment": segment, "SecurityId": security_id}
            )

        # Dhan allows max 100 instruments per message
        for i in range(0, len(instrument_list), 100):
            batch = instrument_list[i : i + 100]
            message = {
                "RequestCode": self.SUBSCRIBE,
                "InstrumentCount": len(batch),
                "InstrumentList": batch,
            }
            await self._ws.send(json.dumps(message))

        self._subscribed.update(instruments)

    async def unsubscribe(self, instruments: list[str]) -> None:
        """Unsubscribe from instruments."""
        # Dhan doesn't have explicit unsubscribe - reconnect with new list
        self._subscribed -= set(instruments)
        if self._ws:
            await self.disconnect()
            await self.connect()
            if self._subscribed:
                await self.subscribe(list(self._subscribed))

    def on_tick(self, callback: Callable[[Tick], None]) -> None:
        """Register tick callback."""
        self._tick_callbacks.append(callback)

    def on_candle(self, callback: Callable[[Candle], None]) -> None:
        """Register candle callback."""
        self._candle_callbacks.append(callback)

    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        while self._running and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=15,  # Heartbeat timeout
                )

                if isinstance(message, bytes):
                    self._parse_binary_message(message)

            except asyncio.TimeoutError:
                # Send pong for heartbeat
                if self._ws:
                    try:
                        await self._ws.pong()
                    except Exception:
                        pass
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                print(f"Error in message loop: {e}")

    def _parse_binary_message(self, data: bytes) -> None:
        """Parse binary market data message."""
        if len(data) < 2:
            return

        response_code = struct.unpack("<H", data[0:2])[0]

        if response_code == self.TICKER_PACKET:
            self._parse_ticker(data)
        elif response_code == self.QUOTE_PACKET:
            self._parse_quote(data)
        elif response_code == self.FULL_PACKET:
            self._parse_full(data)

    def _parse_ticker(self, data: bytes) -> None:
        """Parse ticker packet (LTP data)."""
        # Binary format: response_code(2) + segment(1) + security_id(4) + ltp(4) + ltt(4)
        if len(data) < 15:
            return

        segment = data[2]
        security_id = struct.unpack("<I", data[3:7])[0]
        ltp = struct.unpack("<f", data[7:11])[0]
        ltt = struct.unpack("<I", data[11:15])[0]

        instrument = f"{self._segment_to_str(segment)}:{security_id}"

        tick = Tick(
            instrument=instrument,
            timestamp=datetime.fromtimestamp(ltt),
            ltp=ltp,
            ltq=0,
            volume=0,
        )

        for callback in self._tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                print(f"Error in tick callback: {e}")

    def _parse_quote(self, data: bytes) -> None:
        """Parse quote packet (more detailed data)."""
        if len(data) < 30:
            return

        segment = data[2]
        security_id = struct.unpack("<I", data[3:7])[0]
        ltp = struct.unpack("<f", data[7:11])[0]
        ltq = struct.unpack("<I", data[11:15])[0]
        ltt = struct.unpack("<I", data[15:19])[0]
        volume = struct.unpack("<I", data[19:23])[0]

        instrument = f"{self._segment_to_str(segment)}:{security_id}"

        tick = Tick(
            instrument=instrument,
            timestamp=datetime.fromtimestamp(ltt),
            ltp=ltp,
            ltq=ltq,
            volume=volume,
        )

        for callback in self._tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                print(f"Error in tick callback: {e}")

    def _parse_full(self, data: bytes) -> None:
        """Parse full packet (includes market depth)."""
        # Similar to quote but with additional depth data
        self._parse_quote(data)

    def _segment_to_str(self, segment: int) -> str:
        """Convert segment code to string."""
        return self.SEGMENT_MAP.get(segment, f"UNKNOWN_{segment}")

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._running

    @property
    def subscribed_instruments(self) -> set[str]:
        """Get set of subscribed instruments."""
        return self._subscribed.copy()
