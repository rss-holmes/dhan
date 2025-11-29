# Technical Specification: Backtesting & Algorithmic Trading System

## 1. Executive Summary

This document specifies the design for a modular backtesting and algorithmic trading system built on top of the existing Dhan API client. The system enables:

- **Strategy Development**: Class-based strategies with pluggable architecture
- **Backtesting**: Historical simulation with configurable fill logic
- **Paper Trading**: Simulated live trading with real market data
- **Live Trading**: Production trading with real capital
- **Broker Agnosticism**: Swappable market data feeds and order execution providers

---

## 2. Requirements Summary

| Requirement | Decision |
|-------------|----------|
| Strategy API | Class-based (inherit from `BaseStrategy`) |
| Technical Indicators | TA-Lib integration |
| Order Types | Market + Limit |
| Risk Management | None built-in (strategy-controlled) |
| Concurrency | Multiple strategies in parallel |
| Paper Trading | Yes |
| Persistence | CSV files |
| Visualization | Interactive Plotly charts |
| Fill Logic | Configurable (open, close, VWAP, slippage) |
| State Recovery | Sync from broker on startup |
| Timeframes | Daily + Intraday (1m, 5m, 15m, 25m, 60m) |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Strategy   │  │  Strategy   │  │  Strategy   │  │   CLI / Scripts     │ │
│  │  (SMA Cross)│  │  (RSI Mean) │  │  (Custom)   │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENGINE LAYER                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Strategy Engine                                  │ │
│  │   • Strategy Registry & Lifecycle Management                           │ │
│  │   • Signal Generation & Routing                                        │ │
│  │   • Multi-Strategy Orchestration                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   Backtester    │  │   Paper Trader      │  │     Live Trader         │  │
│  │                 │  │                     │  │                         │  │
│  │ • CSV Data Feed │  │ • Real Market Feed  │  │ • Real Market Feed      │  │
│  │ • Simulated     │  │ • Simulated Orders  │  │ • Real Orders           │  │
│  │   Execution     │  │ • Virtual Portfolio │  │ • Real Portfolio        │  │
│  │ • Fill Models   │  │                     │  │                         │  │
│  └────────┬────────┘  └──────────┬──────────┘  └────────────┬────────────┘  │
└───────────┼──────────────────────┼──────────────────────────┼───────────────┘
            │                      │                          │
            ▼                      ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ABSTRACTION LAYER                                   │
│  ┌─────────────────────────┐          ┌─────────────────────────┐           │
│  │    DataFeed Protocol    │          │   Broker Protocol       │           │
│  │                         │          │                         │           │
│  │  • subscribe()          │          │  • place_order()        │           │
│  │  • unsubscribe()        │          │  • cancel_order()       │           │
│  │  • get_historical()     │          │  • get_positions()      │           │
│  │  • on_tick()            │          │  • get_orders()         │           │
│  │  • on_candle()          │          │  • get_balance()        │           │
│  └───────────┬─────────────┘          └───────────┬─────────────┘           │
└──────────────┼────────────────────────────────────┼─────────────────────────┘
               │                                    │
               ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROVIDER LAYER                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │  Dhan DataFeed   │  │ Upstox DataFeed  │  │  CSV DataFeed    │           │
│  │  (WebSocket)     │  │  (WebSocket)     │  │  (Historical)    │           │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │   Dhan Broker    │  │  Upstox Broker   │  │ Simulated Broker │           │
│  │   (core/)        │  │  (future/)       │  │  (Paper/Backtest)│           │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
dhan/
├── core/                           # Existing Dhan API client
│   ├── client.py
│   ├── config.py
│   ├── exceptions.py
│   ├── models/
│   └── services/
│
├── algo/                           # New algorithmic trading module
│   ├── __init__.py
│   │
│   ├── strategy/                   # Strategy Framework
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseStrategy abstract class
│   │   ├── registry.py             # Strategy registration & discovery
│   │   └── context.py              # StrategyContext (data access, order submission)
│   │
│   ├── data/                       # Data Feed Abstraction
│   │   ├── __init__.py
│   │   ├── protocols.py            # DataFeed protocol definition
│   │   ├── models.py               # Candle, Tick, OHLCV models
│   │   ├── aggregator.py           # Tick-to-candle aggregation
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── csv_feed.py         # CSV file data feed
│   │       ├── dhan_feed.py        # Dhan WebSocket feed
│   │       └── upstox_feed.py      # Upstox WebSocket feed (future)
│   │
│   ├── broker/                     # Broker Abstraction
│   │   ├── __init__.py
│   │   ├── protocols.py            # Broker protocol definition
│   │   ├── models.py               # Order, Position, Fill models
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── simulated.py        # Simulated broker (backtest/paper)
│   │       ├── dhan_broker.py      # Dhan broker (wraps core/)
│   │       └── upstox_broker.py    # Upstox broker (future)
│   │
│   ├── engine/                     # Execution Engines
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseEngine abstract class
│   │   ├── backtest.py             # BacktestEngine
│   │   ├── paper.py                # PaperTradingEngine
│   │   ├── live.py                 # LiveTradingEngine
│   │   └── fill_models.py          # Fill simulation models
│   │
│   ├── portfolio/                  # Portfolio Management
│   │   ├── __init__.py
│   │   ├── portfolio.py            # Portfolio state management
│   │   ├── position.py             # Position tracking
│   │   └── performance.py          # Performance metrics calculation
│   │
│   ├── analytics/                  # Results & Visualization
│   │   ├── __init__.py
│   │   ├── metrics.py              # Performance metrics (Sharpe, etc.)
│   │   ├── reports.py              # Report generation
│   │   └── charts.py               # Plotly visualizations
│   │
│   └── persistence/                # Data Persistence
│       ├── __init__.py
│       ├── trades_csv.py           # Trade log CSV writer
│       ├── results_csv.py          # Backtest results CSV
│       └── equity_csv.py           # Equity curve CSV
│
├── strategies/                     # User Strategy Implementations
│   ├── __init__.py
│   ├── sma_crossover.py
│   ├── rsi_mean_reversion.py
│   └── examples/
│
├── data/                           # Historical data storage
│   └── *.csv
│
├── scripts/                        # Utility scripts
│   ├── run_backtest.py
│   ├── run_paper.py
│   ├── run_live.py
│   └── download_data.py
│
└── docs/
    └── TECHNICAL_SPECIFICATION.md
```

---

## 4. Core Components

### 4.1 Strategy Framework

#### 4.1.1 BaseStrategy Abstract Class

```python
# algo/strategy/base.py

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
from pydantic import BaseModel, Field
from algo.data.models import Candle
from algo.strategy.context import StrategyContext

class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Signal(BaseModel):
    """Represents a trading signal generated by a strategy."""
    instrument: str
    action: SignalAction  # BUY, SELL, HOLD
    quantity: int = Field(gt=0)
    order_type: "OrderType"  # MARKET, LIMIT
    limit_price: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}  # Signals are immutable

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Lifecycle:
    1. __init__() - Set strategy parameters
    2. on_init(context) - Called once before first candle
    3. on_candle(candle) - Called for each new candle
    4. on_order_fill(fill) - Called when an order is filled
    5. on_stop() - Called when strategy is stopped
    """

    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.params = params
        self.context: StrategyContext | None = None

    def set_context(self, context: StrategyContext) -> None:
        """Inject the strategy context. Called by engine."""
        self.context = context

    def on_init(self) -> None:
        """
        Called once before the first candle is processed.
        Override to initialize indicators, state, etc.
        """
        pass

    @abstractmethod
    def on_candle(self, candle: Candle) -> Signal | list[Signal] | None:
        """
        Called for each new candle.

        Args:
            candle: The new OHLCV candle

        Returns:
            Signal, list of Signals, or None for no action
        """
        pass

    def on_order_fill(self, fill: "Fill") -> None:
        """
        Called when an order placed by this strategy is filled.
        Override to track fills, update state, etc.
        """
        pass

    def on_stop(self) -> None:
        """
        Called when the strategy is stopped.
        Override to cleanup resources.
        """
        pass

    @property
    def name(self) -> str:
        """Strategy name for logging and identification."""
        return self.__class__.__name__

    @property
    def instruments(self) -> list[str]:
        """List of instruments this strategy trades. Override in subclass."""
        return []
```

#### 4.1.2 StrategyContext

```python
# algo/strategy/context.py

from typing import TYPE_CHECKING
import talib
import numpy as np

if TYPE_CHECKING:
    from algo.engine.base import BaseEngine

class StrategyContext:
    """
    Provides strategies with access to market data, indicators,
    portfolio state, and order submission.
    """

    def __init__(self, engine: "BaseEngine", strategy_id: str):
        self._engine = engine
        self._strategy_id = strategy_id

    # === Data Access ===

    def get_historical(
        self,
        instrument: str,
        periods: int,
        field: str = "close"  # open, high, low, close, volume
    ) -> np.ndarray:
        """Get historical data for indicator calculation."""
        return self._engine.get_historical(instrument, periods, field)

    def get_candle(self, instrument: str, offset: int = 0) -> Candle | None:
        """Get a specific historical candle. offset=0 is current, -1 is previous."""
        return self._engine.get_candle(instrument, offset)

    # === TA-Lib Indicators (convenience wrappers) ===

    def sma(self, instrument: str, period: int) -> float:
        """Simple Moving Average of close prices."""
        closes = self.get_historical(instrument, period + 1)
        return talib.SMA(closes, timeperiod=period)[-1]

    def ema(self, instrument: str, period: int) -> float:
        """Exponential Moving Average of close prices."""
        closes = self.get_historical(instrument, period * 2)
        return talib.EMA(closes, timeperiod=period)[-1]

    def rsi(self, instrument: str, period: int = 14) -> float:
        """Relative Strength Index."""
        closes = self.get_historical(instrument, period + 10)
        return talib.RSI(closes, timeperiod=period)[-1]

    def macd(self, instrument: str) -> tuple[float, float, float]:
        """MACD (macd, signal, histogram)."""
        closes = self.get_historical(instrument, 50)
        macd, signal, hist = talib.MACD(closes)
        return macd[-1], signal[-1], hist[-1]

    def bollinger_bands(
        self, instrument: str, period: int = 20, std: float = 2.0
    ) -> tuple[float, float, float]:
        """Bollinger Bands (upper, middle, lower)."""
        closes = self.get_historical(instrument, period + 5)
        upper, middle, lower = talib.BBANDS(closes, timeperiod=period, nbdevup=std, nbdevdn=std)
        return upper[-1], middle[-1], lower[-1]

    def atr(self, instrument: str, period: int = 14) -> float:
        """Average True Range."""
        high = self.get_historical(instrument, period + 5, "high")
        low = self.get_historical(instrument, period + 5, "low")
        close = self.get_historical(instrument, period + 5, "close")
        return talib.ATR(high, low, close, timeperiod=period)[-1]

    # === Portfolio State ===

    def get_position(self, instrument: str) -> "Position | None":
        """Get current position for instrument."""
        return self._engine.portfolio.get_position(instrument)

    def get_balance(self) -> float:
        """Get available cash balance."""
        return self._engine.portfolio.cash

    def get_equity(self) -> float:
        """Get total portfolio equity (cash + positions value)."""
        return self._engine.portfolio.equity

    # === Order Submission ===

    def submit_order(self, signal: Signal) -> str:
        """
        Submit an order based on a signal.
        Returns order_id for tracking.
        """
        return self._engine.submit_order(self._strategy_id, signal)
```

#### 4.1.3 Example Strategy Implementation

```python
# strategies/sma_crossover.py

from algo.strategy.base import BaseStrategy, Signal, SignalAction
from algo.data.models import Candle
from core.models.common import OrderType

class SmaCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Buys when fast SMA crosses above slow SMA.
    Sells when fast SMA crosses below slow SMA.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 20, quantity: int = 1):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            quantity=quantity
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    @property
    def instruments(self) -> list[str]:
        return ["NSE_EQ:1333"]  # HDFC Bank

    def on_candle(self, candle: Candle) -> Signal | None:
        # Calculate current SMAs
        fast_sma = self.context.sma(candle.instrument, self.fast_period)
        slow_sma = self.context.sma(candle.instrument, self.slow_period)

        signal = None

        # Check for crossover
        if self._prev_fast is not None and self._prev_slow is not None:
            # Bullish crossover
            if self._prev_fast <= self._prev_slow and fast_sma > slow_sma:
                position = self.context.get_position(candle.instrument)
                if position is None or position.quantity <= 0:
                    signal = Signal(
                        instrument=candle.instrument,
                        action=SignalAction.BUY,
                        quantity=self.quantity,
                        order_type=OrderType.MARKET,
                    )

            # Bearish crossover
            elif self._prev_fast >= self._prev_slow and fast_sma < slow_sma:
                position = self.context.get_position(candle.instrument)
                if position is not None and position.quantity > 0:
                    signal = Signal(
                        instrument=candle.instrument,
                        action=SignalAction.SELL,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET,
                    )

        # Store for next comparison
        self._prev_fast = fast_sma
        self._prev_slow = slow_sma

        return signal
```

---

### 4.2 Data Feed Abstraction

#### 4.2.1 DataFeed Protocol

```python
# algo/data/protocols.py

from typing import Protocol, Callable, AsyncIterator, runtime_checkable
from algo.data.models import Candle, Tick

@runtime_checkable
class DataFeed(Protocol):
    """
    Protocol for market data feeds.
    Implementations can be WebSocket-based (live) or file-based (backtest).
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

    async def get_historical(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        interval: str  # "1m", "5m", "15m", "1h", "1d"
    ) -> list[Candle]:
        """Fetch historical candle data."""
        ...


class HistoricalDataFeed(Protocol):
    """
    Protocol for historical data iteration (backtesting).
    """

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
        end_offset: int = 0
    ) -> list[Candle]:
        """Get historical candles relative to current position."""
        ...
```

#### 4.2.2 Data Models

```python
# algo/data/models.py

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, computed_field

class Interval(Enum):
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

    model_config = {"frozen": True}  # Immutable

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

    model_config = {"frozen": True}  # Immutable

    @computed_field
    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @computed_field
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @computed_field
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @computed_field
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @computed_field
    @property
    def candle_range(self) -> float:  # renamed from 'range' to avoid builtin conflict
        return self.high - self.low
```

#### 4.2.3 Dhan WebSocket Feed Implementation

```python
# algo/data/providers/dhan_feed.py

import asyncio
import json
import struct
from datetime import datetime
from typing import Callable
import websockets

from algo.data.protocols import DataFeed
from algo.data.models import Tick, Candle
from algo.data.aggregator import CandleAggregator

class DhanDataFeed(DataFeed):
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

    def __init__(
        self,
        client_id: str,
        access_token: str,
        on_tick: Callable[[Tick], None] | None = None,
        on_candle: Callable[[Candle], None] | None = None,
    ):
        self._client_id = client_id
        self._access_token = access_token
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._subscribed: set[str] = set()
        self._tick_callbacks: list[Callable[[Tick], None]] = []
        self._candle_callbacks: list[Callable[[Candle], None]] = []
        self._aggregators: dict[str, CandleAggregator] = {}
        self._running = False

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
        asyncio.create_task(self._message_loop())

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        if self._ws:
            self._running = False
            await self._ws.send(json.dumps({"RequestCode": self.DISCONNECT}))
            await self._ws.close()
            self._ws = None

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
            instrument_list.append({
                "ExchangeSegment": segment,
                "SecurityId": security_id
            })

        # Dhan allows max 100 instruments per message
        for i in range(0, len(instrument_list), 100):
            batch = instrument_list[i:i+100]
            message = {
                "RequestCode": self.SUBSCRIBE,
                "InstrumentCount": len(batch),
                "InstrumentList": batch
            }
            await self._ws.send(json.dumps(message))

        self._subscribed.update(instruments)

    async def unsubscribe(self, instruments: list[str]) -> None:
        """Unsubscribe from instruments."""
        # Dhan doesn't have explicit unsubscribe - reconnect with new list
        self._subscribed -= set(instruments)
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
                    timeout=15  # Heartbeat timeout
                )

                if isinstance(message, bytes):
                    self._parse_binary_message(message)

            except asyncio.TimeoutError:
                # Send pong for heartbeat
                await self._ws.pong()
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                print(f"Error in message loop: {e}")

    def _parse_binary_message(self, data: bytes) -> None:
        """Parse binary market data message."""
        if len(data) < 2:
            return

        response_code = struct.unpack('<H', data[0:2])[0]

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
        security_id = struct.unpack('<I', data[3:7])[0]
        ltp = struct.unpack('<f', data[7:11])[0]
        ltt = struct.unpack('<I', data[11:15])[0]

        instrument = f"{self._segment_to_str(segment)}:{security_id}"

        tick = Tick(
            instrument=instrument,
            timestamp=datetime.fromtimestamp(ltt),
            ltp=ltp,
            ltq=0,
            volume=0,
        )

        for callback in self._tick_callbacks:
            callback(tick)

    def _segment_to_str(self, segment: int) -> str:
        """Convert segment code to string."""
        segments = {
            0: "NSE_EQ",
            1: "NSE_FNO",
            2: "NSE_CURRENCY",
            3: "BSE_EQ",
            4: "MCX_COMM",
        }
        return segments.get(segment, f"UNKNOWN_{segment}")
```

#### 4.2.4 CSV Data Feed (Backtesting)

```python
# algo/data/providers/csv_feed.py

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterator

from algo.data.protocols import HistoricalDataFeed
from algo.data.models import Candle, Interval

class CSVDataFeed(HistoricalDataFeed):
    """
    Historical data feed from CSV files for backtesting.

    Expected CSV format:
    timestamp,open,high,low,close,volume[,oi]

    timestamp format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    """

    def __init__(
        self,
        data_dir: Path | str,
        interval: Interval = Interval.DAY_1,
    ):
        self._data_dir = Path(data_dir)
        self._interval = interval
        self._instruments: list[str] = []
        self._data: dict[str, list[Candle]] = {}
        self._current_idx: int = 0
        self._merged_timeline: list[tuple[datetime, str, Candle]] = []

    def set_instruments(self, instruments: list[str]) -> None:
        """Load data for specified instruments."""
        self._instruments = instruments
        self._data.clear()

        for instrument in instruments:
            self._load_instrument_data(instrument)

        self._build_merged_timeline()

    def _load_instrument_data(self, instrument: str) -> None:
        """Load CSV data for a single instrument."""
        # Convert instrument ID to filename
        # e.g., "NSE_EQ:1333" -> "NSE_EQ_1333.csv"
        filename = instrument.replace(":", "_") + ".csv"
        filepath = self._data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        candles = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candle = Candle(
                    instrument=instrument,
                    timestamp=self._parse_timestamp(row['timestamp']),
                    interval=self._interval,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(float(row['volume'])),
                    oi=int(float(row.get('oi', 0))) if row.get('oi') else None,
                )
                candles.append(candle)

        # Sort by timestamp
        candles.sort(key=lambda c: c.timestamp)
        self._data[instrument] = candles

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse timestamp from various formats."""
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"]:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {ts}")

    def _build_merged_timeline(self) -> None:
        """Build chronological timeline of all candles."""
        self._merged_timeline = []

        for instrument, candles in self._data.items():
            for candle in candles:
                self._merged_timeline.append((candle.timestamp, instrument, candle))

        # Sort by timestamp
        self._merged_timeline.sort(key=lambda x: x[0])
        self._current_idx = 0

    def __iter__(self) -> Iterator[Candle]:
        """Iterate through candles in chronological order."""
        self._current_idx = 0
        for _, _, candle in self._merged_timeline:
            self._current_idx += 1
            yield candle

    def get_candles(
        self,
        instrument: str,
        count: int,
        end_offset: int = 0
    ) -> list[Candle]:
        """
        Get historical candles relative to current position.

        Args:
            instrument: Instrument identifier
            count: Number of candles to retrieve
            end_offset: Offset from current position (0 = current candle)
        """
        if instrument not in self._data:
            return []

        all_candles = self._data[instrument]

        # Find current position in instrument's timeline
        current_ts = self._merged_timeline[self._current_idx - 1][0] if self._current_idx > 0 else None

        if current_ts is None:
            return []

        # Find candles up to current timestamp
        valid_candles = [c for c in all_candles if c.timestamp <= current_ts]

        if not valid_candles:
            return []

        end_idx = len(valid_candles) - end_offset
        start_idx = max(0, end_idx - count)

        return valid_candles[start_idx:end_idx]
```

---

### 4.3 Broker Abstraction

#### 4.3.1 Broker Protocol

```python
# algo/broker/protocols.py

from typing import Protocol, runtime_checkable
from algo.broker.models import Order, OrderRequest, Position, Fill, OrderStatus

@runtime_checkable
class Broker(Protocol):
    """
    Protocol for order execution and account management.
    Implementations can be real brokers or simulated (paper/backtest).
    """

    async def connect(self) -> None:
        """Establish connection to broker."""
        ...

    async def disconnect(self) -> None:
        """Close connection to broker."""
        ...

    # === Order Management ===

    async def place_order(self, request: OrderRequest) -> Order:
        """
        Place a new order.

        Returns Order with order_id and initial status.
        """
        ...

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order."""
        ...

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> Order:
        """Modify a pending order."""
        ...

    async def get_order(self, order_id: str) -> Order:
        """Get order details by ID."""
        ...

    async def get_orders(self) -> list[Order]:
        """Get all orders for the day."""
        ...

    # === Position Management ===

    async def get_positions(self) -> list[Position]:
        """Get all current positions."""
        ...

    async def get_position(self, instrument: str) -> Position | None:
        """Get position for specific instrument."""
        ...

    # === Account ===

    async def get_balance(self) -> float:
        """Get available cash balance."""
        ...

    async def get_margin_used(self) -> float:
        """Get margin currently used."""
        ...

    # === Callbacks ===

    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        """Register callback for order status updates."""
        ...

    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        """Register callback for order fills."""
        ...
```

#### 4.3.2 Broker Models

```python
# algo/broker/models.py

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, computed_field

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderRequest(BaseModel):
    """Request to place a new order."""
    instrument: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    limit_price: float | None = Field(default=None, gt=0)
    strategy_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}  # Requests are immutable

class Order(BaseModel):
    """Represents an order in the system."""
    order_id: str
    instrument: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    status: OrderStatus
    limit_price: float | None = Field(default=None, gt=0)
    filled_quantity: int = Field(default=0, ge=0)
    average_price: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    strategy_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"validate_assignment": True}  # Validate on mutation

    @computed_field
    @property
    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        )

    @computed_field
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

class Fill(BaseModel):
    """Represents an order fill (execution)."""
    fill_id: str
    order_id: str
    instrument: str
    side: OrderSide
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime
    commission: float = Field(default=0.0, ge=0)
    strategy_id: str | None = None

    model_config = {"frozen": True}  # Fills are immutable

class Position(BaseModel):
    """Represents an open position."""
    instrument: str
    quantity: int  # Positive for long, negative for short
    average_price: float = Field(ge=0)
    current_price: float = Field(ge=0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    model_config = {"validate_assignment": True}

    @computed_field
    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @computed_field
    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @computed_field
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
```

#### 4.3.3 Simulated Broker (Backtest/Paper)

```python
# algo/broker/providers/simulated.py

import uuid
from datetime import datetime
from typing import Callable
from algo.broker.protocols import Broker
from algo.broker.models import (
    Order, OrderRequest, Position, Fill,
    OrderStatus, OrderSide, OrderType
)
from algo.engine.fill_models import FillModel, FillResult

class SimulatedBroker(Broker):
    """
    Simulated broker for backtesting and paper trading.

    Handles order matching, position tracking, and P&L calculation.
    """

    def __init__(
        self,
        initial_capital: float,
        fill_model: FillModel,
        commission_rate: float = 0.0,  # Per-trade commission rate
    ):
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._fill_model = fill_model
        self._commission_rate = commission_rate

        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []

        self._order_callbacks: list[Callable[[Order], None]] = []
        self._fill_callbacks: list[Callable[[Fill], None]] = []

        self._current_prices: dict[str, float] = {}

    async def connect(self) -> None:
        """No-op for simulated broker."""
        pass

    async def disconnect(self) -> None:
        """No-op for simulated broker."""
        pass

    def update_price(self, instrument: str, price: float) -> None:
        """Update current price for an instrument. Called by engine."""
        self._current_prices[instrument] = price
        self._update_position_pnl(instrument, price)
        self._check_pending_orders(instrument, price)

    async def place_order(self, request: OrderRequest) -> Order:
        """Place and potentially fill an order."""
        order = Order(
            order_id=str(uuid.uuid4()),
            instrument=request.instrument,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
            status=OrderStatus.PENDING,
            strategy_id=request.strategy_id,
            metadata=request.metadata,
        )

        self._orders[order.order_id] = order

        # Try to fill immediately for market orders
        if order.order_type == OrderType.MARKET:
            current_price = self._current_prices.get(request.instrument)
            if current_price:
                self._try_fill_order(order, current_price)
        else:
            # Limit order - add to pending
            order.status = OrderStatus.OPEN
            self._pending_orders.append(order)
            self._notify_order_update(order)

        return order

    def _try_fill_order(self, order: Order, reference_price: float) -> None:
        """Attempt to fill an order using the fill model."""
        fill_result = self._fill_model.calculate_fill(
            order=order,
            reference_price=reference_price,
        )

        if fill_result.filled:
            self._execute_fill(order, fill_result)

    def _execute_fill(self, order: Order, fill_result: FillResult) -> None:
        """Execute a fill and update positions."""
        commission = fill_result.fill_price * fill_result.fill_quantity * self._commission_rate

        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=fill_result.fill_quantity,
            price=fill_result.fill_price,
            timestamp=datetime.now(),
            commission=commission,
            strategy_id=order.strategy_id,
        )

        # Update order status
        order.filled_quantity += fill_result.fill_quantity
        order.average_price = (
            (order.average_price * (order.filled_quantity - fill_result.fill_quantity) +
             fill_result.fill_price * fill_result.fill_quantity) / order.filled_quantity
        )

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        order.updated_at = datetime.now()

        # Update position
        self._update_position(fill)

        # Update cash
        trade_value = fill.price * fill.quantity
        if fill.side == OrderSide.BUY:
            self._cash -= trade_value + commission
        else:
            self._cash += trade_value - commission

        # Notify callbacks
        self._notify_order_update(order)
        self._notify_fill(fill)

    def _update_position(self, fill: Fill) -> None:
        """Update position based on fill."""
        position = self._positions.get(fill.instrument)

        if position is None:
            # New position
            quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            self._positions[fill.instrument] = Position(
                instrument=fill.instrument,
                quantity=quantity,
                average_price=fill.price,
                current_price=fill.price,
            )
        else:
            # Update existing position
            if fill.side == OrderSide.BUY:
                if position.quantity >= 0:
                    # Adding to long position
                    total_cost = position.average_price * position.quantity + fill.price * fill.quantity
                    position.quantity += fill.quantity
                    position.average_price = total_cost / position.quantity if position.quantity > 0 else 0
                else:
                    # Closing short position
                    position.realized_pnl += (position.average_price - fill.price) * min(fill.quantity, abs(position.quantity))
                    position.quantity += fill.quantity
                    if position.quantity > 0:
                        position.average_price = fill.price
            else:  # SELL
                if position.quantity <= 0:
                    # Adding to short position
                    total_cost = abs(position.average_price * position.quantity) + fill.price * fill.quantity
                    position.quantity -= fill.quantity
                    position.average_price = total_cost / abs(position.quantity) if position.quantity != 0 else 0
                else:
                    # Closing long position
                    position.realized_pnl += (fill.price - position.average_price) * min(fill.quantity, position.quantity)
                    position.quantity -= fill.quantity
                    if position.quantity < 0:
                        position.average_price = fill.price

            # Remove closed positions
            if position.quantity == 0:
                del self._positions[fill.instrument]

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_position(self, instrument: str) -> Position | None:
        return self._positions.get(instrument)

    async def get_balance(self) -> float:
        return self._cash

    async def get_orders(self) -> list[Order]:
        return list(self._orders.values())

    async def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    async def cancel_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            if order in self._pending_orders:
                self._pending_orders.remove(order)
            self._notify_order_update(order)
        return order

    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        self._order_callbacks.append(callback)

    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        self._fill_callbacks.append(callback)

    def _notify_order_update(self, order: Order) -> None:
        for callback in self._order_callbacks:
            callback(order)

    def _notify_fill(self, fill: Fill) -> None:
        for callback in self._fill_callbacks:
            callback(fill)
```

#### 4.3.4 Dhan Broker (wraps existing core/)

```python
# algo/broker/providers/dhan_broker.py

from typing import Callable
from core import DhanClient, DhanContext
from core.models.common import TransactionType, ProductType, OrderType as DhanOrderType
from core.models.orders import OrderRequest as DhanOrderRequest
from algo.broker.protocols import Broker
from algo.broker.models import (
    Order, OrderRequest, Position, Fill,
    OrderStatus, OrderSide, OrderType
)

class DhanBroker(Broker):
    """
    Dhan broker implementation wrapping the existing core/ module.
    """

    def __init__(self, context: DhanContext):
        self._client = DhanClient(context)
        self._order_callbacks: list[Callable[[Order], None]] = []
        self._fill_callbacks: list[Callable[[Fill], None]] = []

    async def connect(self) -> None:
        """No persistent connection needed for REST API."""
        pass

    async def disconnect(self) -> None:
        """No cleanup needed."""
        pass

    async def place_order(self, request: OrderRequest) -> Order:
        """Place order via Dhan API."""
        # Map to Dhan order request
        segment, security_id = request.instrument.split(":")

        dhan_request = DhanOrderRequest(
            securityId=security_id,
            exchangeSegment=segment,
            transactionType=TransactionType.BUY if request.side == OrderSide.BUY else TransactionType.SELL,
            orderType=DhanOrderType.MARKET if request.order_type == OrderType.MARKET else DhanOrderType.LIMIT,
            productType=ProductType.INTRADAY,  # Configurable
            quantity=request.quantity,
            price=request.limit_price or 0,
            validity="DAY",
        )

        dhan_order = self._client.orders.place_order(dhan_request)

        return self._map_dhan_order(dhan_order, request.strategy_id)

    async def get_positions(self) -> list[Position]:
        """Get positions from Dhan API."""
        dhan_positions = self._client.portfolio.get_positions()
        return [self._map_dhan_position(p) for p in dhan_positions]

    async def get_balance(self) -> float:
        """Get available balance from Dhan API."""
        fund_limit = self._client.funds.get_fund_limit()
        return fund_limit.availabelBalance  # Note: Dhan API typo

    async def get_orders(self) -> list[Order]:
        """Get all orders for the day."""
        dhan_orders = self._client.orders.get_orders()
        return [self._map_dhan_order(o) for o in dhan_orders]

    def _map_dhan_order(self, dhan_order, strategy_id: str | None = None) -> Order:
        """Map Dhan order to internal Order model."""
        status_map = {
            "TRANSIT": OrderStatus.PENDING,
            "PENDING": OrderStatus.OPEN,
            "TRADED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }

        return Order(
            order_id=dhan_order.order_id,
            instrument=f"{dhan_order.exchange_segment}:{dhan_order.security_id}",
            side=OrderSide.BUY if dhan_order.transaction_type == "BUY" else OrderSide.SELL,
            quantity=dhan_order.quantity,
            order_type=OrderType.MARKET if dhan_order.order_type == "MARKET" else OrderType.LIMIT,
            status=status_map.get(dhan_order.order_status, OrderStatus.PENDING),
            filled_quantity=dhan_order.traded_quantity or 0,
            average_price=dhan_order.traded_price or 0.0,
            limit_price=dhan_order.price,
            strategy_id=strategy_id,
        )

    def _map_dhan_position(self, dhan_position) -> Position:
        """Map Dhan position to internal Position model."""
        return Position(
            instrument=f"{dhan_position.exchange_segment}:{dhan_position.security_id}",
            quantity=dhan_position.net_qty,
            average_price=dhan_position.cost_price,
            current_price=dhan_position.day_sell_value / dhan_position.day_sell_qty if dhan_position.day_sell_qty else dhan_position.cost_price,
            unrealized_pnl=dhan_position.unrealized_profit,
            realized_pnl=dhan_position.realized_profit,
        )

    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        self._order_callbacks.append(callback)

    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        self._fill_callbacks.append(callback)
```

---

### 4.4 Execution Engines

#### 4.4.1 Fill Models

```python
# algo/engine/fill_models.py

from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field
from algo.broker.models import Order, OrderType, OrderSide
from algo.data.models import Candle

class FillModelType(Enum):
    NEXT_OPEN = "next_open"
    SAME_CLOSE = "same_close"
    VWAP = "vwap"
    SLIPPAGE = "slippage"

class FillResult(BaseModel):
    """Result of fill calculation."""
    filled: bool
    fill_price: float = Field(ge=0)
    fill_quantity: int = Field(ge=0)
    slippage: float = Field(default=0.0, ge=0)

    model_config = {"frozen": True}

class FillModel(ABC):
    """Abstract base for fill simulation models."""

    @abstractmethod
    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        """Calculate fill price and quantity."""
        pass

class NextOpenFillModel(FillModel):
    """Fill at next candle's open price."""

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        if candle is None:
            return FillResult(filled=False, fill_price=0, fill_quantity=0)

        fill_price = candle.open

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
        )

class SameCloseFillModel(FillModel):
    """Fill at same candle's close price."""

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        fill_price = reference_price

        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
        )

class SlippageFillModel(FillModel):
    """Fill with configurable slippage."""

    def __init__(self, slippage_pct: float = 0.001):
        """
        Args:
            slippage_pct: Slippage as percentage (0.001 = 0.1%)
        """
        self._slippage_pct = slippage_pct

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        # Apply slippage (adverse to trader)
        if order.side == OrderSide.BUY:
            fill_price = reference_price * (1 + self._slippage_pct)
        else:
            fill_price = reference_price * (1 - self._slippage_pct)

        slippage = abs(fill_price - reference_price)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
            slippage=slippage,
        )
```

#### 4.4.2 Backtest Engine

```python
# algo/engine/backtest.py

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Iterator

from algo.engine.base import BaseEngine
from algo.engine.fill_models import FillModel, NextOpenFillModel
from algo.strategy.base import BaseStrategy, Signal
from algo.data.providers.csv_feed import CSVDataFeed
from algo.data.models import Candle, Interval
from algo.broker.providers.simulated import SimulatedBroker
from algo.portfolio.portfolio import Portfolio
from algo.portfolio.performance import PerformanceCalculator
from algo.analytics.metrics import BacktestResult

class BacktestEngine(BaseEngine):
    """
    Engine for running backtests on historical data.
    """

    def __init__(
        self,
        data_dir: Path | str,
        initial_capital: float = 100000.0,
        fill_model: FillModel | None = None,
        commission_rate: float = 0.0,
        interval: Interval = Interval.DAY_1,
    ):
        self._data_dir = Path(data_dir)
        self._initial_capital = initial_capital
        self._fill_model = fill_model or NextOpenFillModel()
        self._commission_rate = commission_rate
        self._interval = interval

        self._data_feed: CSVDataFeed | None = None
        self._broker: SimulatedBroker | None = None
        self._portfolio: Portfolio | None = None
        self._strategies: dict[str, BaseStrategy] = {}

        self._candle_history: dict[str, list[Candle]] = {}
        self._equity_curve: list[tuple[datetime, float]] = []
        self._current_candle: Candle | None = None
        self._next_candle: Candle | None = None

    def add_strategy(self, strategy: BaseStrategy) -> str:
        """Add a strategy to the backtest."""
        strategy_id = f"{strategy.name}_{len(self._strategies)}"
        self._strategies[strategy_id] = strategy
        return strategy_id

    async def run(self) -> BacktestResult:
        """Run the backtest and return results."""
        # Collect all instruments from strategies
        all_instruments = set()
        for strategy in self._strategies.values():
            all_instruments.update(strategy.instruments)

        # Initialize components
        self._data_feed = CSVDataFeed(self._data_dir, self._interval)
        self._data_feed.set_instruments(list(all_instruments))

        self._broker = SimulatedBroker(
            initial_capital=self._initial_capital,
            fill_model=self._fill_model,
            commission_rate=self._commission_rate,
        )

        self._portfolio = Portfolio(self._initial_capital)

        # Initialize strategy contexts
        for strategy_id, strategy in self._strategies.items():
            from algo.strategy.context import StrategyContext
            context = StrategyContext(self, strategy_id)
            strategy.set_context(context)
            strategy.on_init()

        # Setup fill callback
        self._broker.on_fill(self._on_fill)

        # Process candles
        candle_iterator = iter(self._data_feed)
        self._current_candle = None
        self._next_candle = next(candle_iterator, None)

        for candle in candle_iterator:
            self._current_candle = self._next_candle
            self._next_candle = candle

            if self._current_candle:
                await self._process_candle(self._current_candle)

        # Process last candle
        if self._next_candle:
            self._current_candle = self._next_candle
            self._next_candle = None
            await self._process_candle(self._current_candle)

        # Stop strategies
        for strategy in self._strategies.values():
            strategy.on_stop()

        # Calculate results
        return self._calculate_results()

    async def _process_candle(self, candle: Candle) -> None:
        """Process a single candle through all strategies."""
        # Update price in broker
        self._broker.update_price(candle.instrument, candle.close)

        # Store candle in history
        if candle.instrument not in self._candle_history:
            self._candle_history[candle.instrument] = []
        self._candle_history[candle.instrument].append(candle)

        # Record equity
        equity = await self._calculate_equity()
        self._equity_curve.append((candle.timestamp, equity))

        # Process through strategies
        for strategy_id, strategy in self._strategies.items():
            if candle.instrument in strategy.instruments:
                signals = strategy.on_candle(candle)

                if signals:
                    if isinstance(signals, Signal):
                        signals = [signals]

                    for signal in signals:
                        await self._process_signal(strategy_id, signal)

    async def _process_signal(self, strategy_id: str, signal: Signal) -> None:
        """Convert signal to order and submit."""
        from algo.broker.models import OrderRequest, OrderSide, OrderType

        request = OrderRequest(
            instrument=signal.instrument,
            side=OrderSide.BUY if signal.action.value == "BUY" else OrderSide.SELL,
            quantity=signal.quantity,
            order_type=OrderType.MARKET if signal.order_type.value == "MARKET" else OrderType.LIMIT,
            limit_price=signal.limit_price,
            strategy_id=strategy_id,
        )

        await self._broker.place_order(request)

    def _on_fill(self, fill) -> None:
        """Handle order fill."""
        # Notify strategy
        if fill.strategy_id and fill.strategy_id in self._strategies:
            self._strategies[fill.strategy_id].on_order_fill(fill)

    async def _calculate_equity(self) -> float:
        """Calculate current portfolio equity."""
        cash = await self._broker.get_balance()
        positions_value = sum(
            p.quantity * p.current_price
            for p in (await self._broker.get_positions())
        )
        return cash + positions_value

    def get_historical(self, instrument: str, periods: int, field: str = "close") -> list[float]:
        """Get historical data for indicator calculation."""
        if instrument not in self._candle_history:
            return []

        candles = self._candle_history[instrument][-periods:]
        return [getattr(c, field) for c in candles]

    def get_candle(self, instrument: str, offset: int = 0) -> Candle | None:
        """Get a specific historical candle."""
        if instrument not in self._candle_history:
            return None

        candles = self._candle_history[instrument]
        idx = -1 + offset
        if abs(idx) <= len(candles):
            return candles[idx]
        return None

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def _calculate_results(self) -> "BacktestResult":
        """Calculate backtest performance metrics."""
        from algo.analytics.metrics import BacktestResult, PerformanceMetrics

        calculator = PerformanceCalculator()
        metrics = calculator.calculate(
            equity_curve=self._equity_curve,
            trades=self._broker._orders,  # Access internal for now
            initial_capital=self._initial_capital,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=self._equity_curve,
            trades=list(self._broker._orders.values()),
            strategies={sid: s.name for sid, s in self._strategies.items()},
        )
```

#### 4.4.3 Live Trading Engine

```python
# algo/engine/live.py

import asyncio
from datetime import datetime
from typing import Callable

from algo.engine.base import BaseEngine
from algo.strategy.base import BaseStrategy, Signal
from algo.data.protocols import DataFeed
from algo.broker.protocols import Broker
from algo.data.models import Candle, Tick
from algo.portfolio.portfolio import Portfolio

class LiveTradingEngine(BaseEngine):
    """
    Engine for live trading with real market data and orders.

    Supports multiple strategies running in parallel.
    Syncs state from broker on startup.
    """

    def __init__(
        self,
        data_feed: DataFeed,
        broker: Broker,
        initial_capital: float | None = None,  # None = sync from broker
    ):
        self._data_feed = data_feed
        self._broker = broker
        self._initial_capital = initial_capital

        self._strategies: dict[str, BaseStrategy] = {}
        self._portfolio: Portfolio | None = None
        self._running = False

        self._candle_history: dict[str, list[Candle]] = {}

        # Aggregation settings
        self._candle_interval = 60  # seconds
        self._tick_aggregators: dict[str, list[Tick]] = {}

    def add_strategy(self, strategy: BaseStrategy) -> str:
        """Add a strategy to live trading."""
        strategy_id = f"{strategy.name}_{len(self._strategies)}"
        self._strategies[strategy_id] = strategy
        return strategy_id

    async def start(self) -> None:
        """Start live trading."""
        self._running = True

        # Connect to data feed and broker
        await self._data_feed.connect()
        await self._broker.connect()

        # Sync state from broker
        await self._sync_state()

        # Initialize strategies
        for strategy_id, strategy in self._strategies.items():
            from algo.strategy.context import StrategyContext
            context = StrategyContext(self, strategy_id)
            strategy.set_context(context)
            strategy.on_init()

        # Subscribe to instruments
        all_instruments = set()
        for strategy in self._strategies.values():
            all_instruments.update(strategy.instruments)

        await self._data_feed.subscribe(list(all_instruments))

        # Register callbacks
        self._data_feed.on_tick(self._on_tick)
        self._data_feed.on_candle(self._on_candle)
        self._broker.on_fill(self._on_fill)

        print(f"Live trading started with {len(self._strategies)} strategies")

    async def stop(self) -> None:
        """Stop live trading gracefully."""
        self._running = False

        for strategy in self._strategies.values():
            strategy.on_stop()

        await self._data_feed.disconnect()
        await self._broker.disconnect()

        print("Live trading stopped")

    async def _sync_state(self) -> None:
        """Sync portfolio state from broker."""
        if self._initial_capital is None:
            balance = await self._broker.get_balance()
            positions = await self._broker.get_positions()
            positions_value = sum(p.market_value for p in positions)
            self._initial_capital = balance + positions_value

        self._portfolio = Portfolio(self._initial_capital)

        # Sync positions
        positions = await self._broker.get_positions()
        for position in positions:
            self._portfolio.update_position(
                position.instrument,
                position.quantity,
                position.average_price,
            )

        print(f"Synced state: {len(positions)} positions, ${self._initial_capital:.2f} capital")

    def _on_tick(self, tick: Tick) -> None:
        """Handle incoming tick data."""
        # Aggregate ticks into candles if needed
        # For now, just update prices
        pass

    def _on_candle(self, candle: Candle) -> None:
        """Handle incoming candle data."""
        asyncio.create_task(self._process_candle(candle))

    async def _process_candle(self, candle: Candle) -> None:
        """Process candle through strategies."""
        # Store in history
        if candle.instrument not in self._candle_history:
            self._candle_history[candle.instrument] = []
        self._candle_history[candle.instrument].append(candle)

        # Limit history size
        max_history = 500
        if len(self._candle_history[candle.instrument]) > max_history:
            self._candle_history[candle.instrument] = self._candle_history[candle.instrument][-max_history:]

        # Process through strategies
        for strategy_id, strategy in self._strategies.items():
            if candle.instrument in strategy.instruments:
                try:
                    signals = strategy.on_candle(candle)

                    if signals:
                        if isinstance(signals, Signal):
                            signals = [signals]

                        for signal in signals:
                            await self._process_signal(strategy_id, signal)

                except Exception as e:
                    print(f"Error in strategy {strategy_id}: {e}")

    async def _process_signal(self, strategy_id: str, signal: Signal) -> None:
        """Convert signal to order and submit to broker."""
        from algo.broker.models import OrderRequest, OrderSide, OrderType

        request = OrderRequest(
            instrument=signal.instrument,
            side=OrderSide.BUY if signal.action.value == "BUY" else OrderSide.SELL,
            quantity=signal.quantity,
            order_type=OrderType.MARKET if signal.order_type.value == "MARKET" else OrderType.LIMIT,
            limit_price=signal.limit_price,
            strategy_id=strategy_id,
            metadata=signal.metadata or {},
        )

        try:
            order = await self._broker.place_order(request)
            print(f"Order placed: {order.order_id} - {signal.action.value} {signal.quantity} {signal.instrument}")
        except Exception as e:
            print(f"Order placement failed: {e}")

    def _on_fill(self, fill) -> None:
        """Handle order fill."""
        # Update portfolio
        if fill.side.value == "BUY":
            self._portfolio.update_position(
                fill.instrument,
                fill.quantity,
                fill.price,
            )
        else:
            self._portfolio.reduce_position(fill.instrument, fill.quantity, fill.price)

        # Notify strategy
        if fill.strategy_id and fill.strategy_id in self._strategies:
            self._strategies[fill.strategy_id].on_order_fill(fill)

        print(f"Fill: {fill.side.value} {fill.quantity} {fill.instrument} @ {fill.price}")

    def get_historical(self, instrument: str, periods: int, field: str = "close") -> list[float]:
        """Get historical data for indicator calculation."""
        if instrument not in self._candle_history:
            return []

        candles = self._candle_history[instrument][-periods:]
        return [getattr(c, field) for c in candles]

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio
```

---

### 4.5 Analytics & Visualization

#### 4.5.1 Performance Metrics

```python
# algo/analytics/metrics.py

from datetime import datetime, timedelta
from typing import Any
from pydantic import BaseModel, Field
import numpy as np

class PerformanceMetrics(BaseModel):
    """Complete backtest performance metrics."""

    # Time
    start_date: datetime
    end_date: datetime
    duration: timedelta

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    buy_and_hold_return_pct: float
    cagr_pct: float

    # Risk
    volatility_ann_pct: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration: timedelta
    avg_drawdown_duration: timedelta

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Alpha/Beta
    alpha_pct: float
    beta: float

    # Capital
    initial_capital: float = Field(gt=0)
    final_equity: float
    peak_equity: float
    exposure_time_pct: float = Field(ge=0, le=100)

    # Trades
    total_trades: int = Field(ge=0)
    win_rate_pct: float = Field(ge=0, le=100)
    best_trade_pct: float
    worst_trade_pct: float
    avg_trade_pct: float
    max_trade_duration: timedelta
    avg_trade_duration: timedelta
    profit_factor: float
    expectancy_pct: float
    sqn: float  # System Quality Number
    kelly_criterion: float

    model_config = {"frozen": True}  # Metrics are immutable once calculated

class BacktestResult(BaseModel):
    """Complete backtest result including metrics and data."""
    metrics: PerformanceMetrics
    equity_curve: list[tuple[datetime, float]]
    trades: list[Any]  # List of Order objects
    strategies: dict[str, str]

    model_config = {"frozen": True}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "metrics": self.metrics.model_dump(),
            "equity_curve": [(str(dt), eq) for dt, eq in self.equity_curve],
            "trades": [t.model_dump() if hasattr(t, 'model_dump') else t for t in self.trades],
            "strategies": self.strategies,
        }

    def summary(self) -> str:
        """Generate text summary similar to the example."""
        m = self.metrics
        return f"""
Start                     {m.start_date}
End                       {m.end_date}
Duration                  {m.duration}
Exposure Time [%]         {m.exposure_time_pct:.2f}
Equity Final [$]          {m.final_equity:.2f}
Equity Peak [$]           {m.peak_equity:.2f}
Return [%]                {m.total_return_pct:.2f}
Buy & Hold Return [%]     {m.buy_and_hold_return_pct:.2f}
Return (Ann.) [%]         {m.annualized_return_pct:.2f}
Volatility (Ann.) [%]     {m.volatility_ann_pct:.2f}
CAGR [%]                  {m.cagr_pct:.2f}
Sharpe Ratio              {m.sharpe_ratio:.2f}
Sortino Ratio             {m.sortino_ratio:.2f}
Calmar Ratio              {m.calmar_ratio:.2f}
Alpha [%]                 {m.alpha_pct:.2f}
Beta                      {m.beta:.2f}
Max. Drawdown [%]         {m.max_drawdown_pct:.2f}
Avg. Drawdown [%]         {m.avg_drawdown_pct:.2f}
Max. Drawdown Duration    {m.max_drawdown_duration}
Avg. Drawdown Duration    {m.avg_drawdown_duration}
# Trades                  {m.total_trades}
Win Rate [%]              {m.win_rate_pct:.2f}
Best Trade [%]            {m.best_trade_pct:.2f}
Worst Trade [%]           {m.worst_trade_pct:.2f}
Avg. Trade [%]            {m.avg_trade_pct:.2f}
Max. Trade Duration       {m.max_trade_duration}
Avg. Trade Duration       {m.avg_trade_duration}
Profit Factor             {m.profit_factor:.2f}
Expectancy [%]            {m.expectancy_pct:.2f}
SQN                       {m.sqn:.2f}
Kelly Criterion           {m.kelly_criterion:.4f}
"""
```

#### 4.5.2 Interactive Charts

```python
# algo/analytics/charts.py

from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from algo.analytics.metrics import BacktestResult

class BacktestCharts:
    """Interactive Plotly charts for backtest visualization."""

    def __init__(self, result: BacktestResult):
        self.result = result

    def equity_curve(self) -> go.Figure:
        """Generate interactive equity curve chart."""
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = [eq for _, eq in self.result.equity_curve]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#2E86AB', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)',
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            template='plotly_white',
        )

        return fig

    def drawdown(self) -> go.Figure:
        """Generate drawdown chart."""
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = [eq for _, eq in self.result.equity_curve]

        # Calculate drawdown
        peak = equity[0]
        drawdowns = []
        for eq in equity:
            if eq > peak:
                peak = eq
            drawdown_pct = (eq - peak) / peak * 100
            drawdowns.append(drawdown_pct)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            line=dict(color='#E63946', width=2),
            fill='tozeroy',
            fillcolor='rgba(230, 57, 70, 0.2)',
        ))

        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
        )

        return fig

    def trades_on_chart(self, ohlc_data: list, instrument: str) -> go.Figure:
        """Generate candlestick chart with trade markers."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Candlestick
        dates = [c.timestamp for c in ohlc_data]

        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=[c.open for c in ohlc_data],
                high=[c.high for c in ohlc_data],
                low=[c.low for c in ohlc_data],
                close=[c.close for c in ohlc_data],
                name='OHLC',
            ),
            row=1, col=1
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=dates,
                y=[c.volume for c in ohlc_data],
                name='Volume',
                marker_color='rgba(100, 100, 100, 0.5)',
            ),
            row=2, col=1
        )

        # Add trade markers
        for trade in self.result.trades:
            if trade.instrument == instrument:
                color = '#2E7D32' if trade.side.value == 'BUY' else '#C62828'
                symbol = 'triangle-up' if trade.side.value == 'BUY' else 'triangle-down'

                fig.add_trace(
                    go.Scatter(
                        x=[trade.created_at],
                        y=[trade.average_price],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=12,
                            color=color,
                        ),
                        name=f"{trade.side.value} @ {trade.average_price:.2f}",
                        hovertemplate=f"{trade.side.value}<br>Qty: {trade.quantity}<br>Price: {trade.average_price:.2f}",
                    ),
                    row=1, col=1
                )

        fig.update_layout(
            title=f'Trades - {instrument}',
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            height=600,
        )

        return fig

    def full_report(self) -> go.Figure:
        """Generate comprehensive multi-panel report."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Drawdown',
                'Monthly Returns',
                'Trade Distribution',
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
            ],
        )

        # Add equity curve
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = [eq for _, eq in self.result.equity_curve]

        fig.add_trace(
            go.Scatter(x=dates, y=equity, name='Equity', line=dict(color='#2E86AB')),
            row=1, col=1
        )

        # Add drawdown
        peak = equity[0]
        drawdowns = []
        for eq in equity:
            if eq > peak:
                peak = eq
            drawdowns.append((eq - peak) / peak * 100)

        fig.add_trace(
            go.Scatter(x=dates, y=drawdowns, name='Drawdown',
                      fill='tozeroy', line=dict(color='#E63946')),
            row=1, col=2
        )

        # Trade returns histogram
        trade_returns = []
        for trade in self.result.trades:
            if trade.status.value == 'FILLED':
                # Simplified P&L calculation
                trade_returns.append(trade.average_price)

        if trade_returns:
            fig.add_trace(
                go.Histogram(x=trade_returns, name='Trade Returns',
                            marker_color='#457B9D'),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            showlegend=False,
            template='plotly_white',
        )

        return fig

    def save_html(self, filepath: str) -> None:
        """Save full report as interactive HTML."""
        fig = self.full_report()
        fig.write_html(filepath)
```

---

### 4.6 Persistence

```python
# algo/persistence/trades_csv.py

import csv
from pathlib import Path
from datetime import datetime
from algo.broker.models import Order, Fill

class TradesCSVWriter:
    """Write trade data to CSV files."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_trades(self, trades: list[Order], filename: str = "trades.csv") -> Path:
        """Write trades to CSV."""
        filepath = self._output_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'order_id', 'timestamp', 'instrument', 'side',
                'quantity', 'order_type', 'status', 'filled_qty',
                'avg_price', 'strategy_id'
            ])

            for trade in trades:
                writer.writerow([
                    trade.order_id,
                    trade.created_at.isoformat(),
                    trade.instrument,
                    trade.side.value,
                    trade.quantity,
                    trade.order_type.value,
                    trade.status.value,
                    trade.filled_quantity,
                    trade.average_price,
                    trade.strategy_id,
                ])

        return filepath

    def write_equity_curve(
        self,
        equity_curve: list[tuple[datetime, float]],
        filename: str = "equity_curve.csv"
    ) -> Path:
        """Write equity curve to CSV."""
        filepath = self._output_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'equity'])

            for dt, equity in equity_curve:
                writer.writerow([dt.isoformat(), equity])

        return filepath

    def write_results_summary(
        self,
        result: "BacktestResult",
        filename: str = "results.csv"
    ) -> Path:
        """Write performance metrics to CSV."""
        filepath = self._output_dir / filename

        metrics = result.metrics

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])

            for key, value in metrics.__dict__.items():
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, (int, float)):
                    value = f"{value:.4f}" if isinstance(value, float) else value
                writer.writerow([key, value])

        return filepath
```

---

## 5. Usage Examples

### 5.1 Running a Backtest

```python
# scripts/run_backtest.py

import asyncio
from pathlib import Path
from algo.engine.backtest import BacktestEngine
from algo.engine.fill_models import NextOpenFillModel, SlippageFillModel
from algo.data.models import Interval
from strategies.sma_crossover import SmaCrossover
from algo.analytics.charts import BacktestCharts
from algo.persistence.trades_csv import TradesCSVWriter

async def main():
    # Configure backtest
    engine = BacktestEngine(
        data_dir=Path("data"),
        initial_capital=100000.0,
        fill_model=SlippageFillModel(slippage_pct=0.001),
        commission_rate=0.0001,
        interval=Interval.DAY_1,
    )

    # Add strategy
    strategy = SmaCrossover(fast_period=10, slow_period=20, quantity=100)
    engine.add_strategy(strategy)

    # Run backtest
    result = await engine.run()

    # Print summary
    print(result.summary())

    # Generate charts
    charts = BacktestCharts(result)
    charts.save_html("backtest_report.html")

    # Save to CSV
    writer = TradesCSVWriter("output")
    writer.write_trades(result.trades)
    writer.write_equity_curve(result.equity_curve)
    writer.write_results_summary(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 Running Live Trading

```python
# scripts/run_live.py

import asyncio
import os
from dotenv import load_dotenv
from core import DhanContext, Environment
from algo.engine.live import LiveTradingEngine
from algo.data.providers.dhan_feed import DhanDataFeed
from algo.broker.providers.dhan_broker import DhanBroker
from strategies.sma_crossover import SmaCrossover

load_dotenv()

async def main():
    # Setup context
    ctx = DhanContext(
        environment=Environment.LIVE,
        client_id=os.getenv("CLIENT_ID"),
        access_token=os.getenv("ACCESS_TOKEN"),
    )

    # Create data feed and broker
    data_feed = DhanDataFeed(
        client_id=ctx.client_id,
        access_token=ctx.access_token,
    )
    broker = DhanBroker(ctx)

    # Create engine
    engine = LiveTradingEngine(
        data_feed=data_feed,
        broker=broker,
    )

    # Add strategies
    engine.add_strategy(SmaCrossover(fast_period=10, slow_period=20, quantity=10))

    # Start trading
    await engine.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.3 Running Paper Trading

```python
# scripts/run_paper.py

import asyncio
import os
from dotenv import load_dotenv
from core import DhanContext, Environment
from algo.engine.paper import PaperTradingEngine
from algo.data.providers.dhan_feed import DhanDataFeed
from strategies.sma_crossover import SmaCrossover

load_dotenv()

async def main():
    ctx = DhanContext(
        environment=Environment.LIVE,  # Use live market data
        client_id=os.getenv("CLIENT_ID"),
        access_token=os.getenv("ACCESS_TOKEN"),
    )

    data_feed = DhanDataFeed(
        client_id=ctx.client_id,
        access_token=ctx.access_token,
    )

    # Paper engine uses simulated broker with real data
    engine = PaperTradingEngine(
        data_feed=data_feed,
        initial_capital=100000.0,
    )

    engine.add_strategy(SmaCrossover(fast_period=10, slow_period=20, quantity=10))

    await engine.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        result = await engine.stop()
        print(result.summary())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Swappable Provider Architecture

### 6.1 Adding a New Data Feed (e.g., Upstox)

```python
# algo/data/providers/upstox_feed.py

import asyncio
import json
import websockets
from algo.data.protocols import DataFeed
from algo.data.models import Tick, Candle

class UpstoxDataFeed(DataFeed):
    """
    Upstox WebSocket market data feed.

    Follows same protocol as DhanDataFeed.
    """

    WS_URL = "wss://api.upstox.com/v3/feed/market-data-feed"

    def __init__(self, api_key: str, access_token: str):
        self._api_key = api_key
        self._access_token = access_token
        # ... similar implementation

    async def connect(self) -> None:
        # Upstox-specific connection logic
        pass

    async def subscribe(self, instruments: list[str]) -> None:
        # Map instrument format and subscribe
        pass

    # ... implement rest of protocol
```

### 6.2 Adding a New Broker (e.g., Upstox)

```python
# algo/broker/providers/upstox_broker.py

from algo.broker.protocols import Broker
from algo.broker.models import Order, OrderRequest, Position

class UpstoxBroker(Broker):
    """
    Upstox broker implementation.

    Would wrap a hypothetical upstox_core/ module similar to core/.
    """

    def __init__(self, api_key: str, access_token: str):
        self._api_key = api_key
        self._access_token = access_token
        # Initialize Upstox client

    async def place_order(self, request: OrderRequest) -> Order:
        # Map to Upstox API and place order
        pass

    async def get_positions(self) -> list[Position]:
        # Fetch from Upstox API
        pass

    # ... implement rest of protocol
```

---

## 7. Dependencies

```toml
# pyproject.toml additions

[project]
dependencies = [
    "pydantic>=2.0",
    "python-dotenv",
    "numpy>=1.24",
    "pandas>=2.0",
    "TA-Lib",  # Requires system-level installation
    "websockets>=11.0",
    "plotly>=5.0",
    "aiohttp>=3.8",
]

[project.optional-dependencies]
dev = [
    "pyright",
    "pytest",
    "pytest-asyncio",
]
```

**Note on TA-Lib Installation:**
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

---

## 8. Future Extensibility

### 8.1 Planned Extensions
- **Multi-timeframe strategies**: Strategies that operate on multiple intervals
- **Walk-forward optimization**: Rolling parameter optimization
- **Monte Carlo simulation**: Statistical robustness testing
- **Portfolio optimization**: Kelly, mean-variance, risk-parity allocators
- **Machine learning integration**: sklearn/pytorch model signals

---

## 9. Summary

This specification defines a modular, extensible algorithmic trading system with:

1. **Pluggable Strategies**: Class-based strategies with TA-Lib integration
2. **Unified Data Protocol**: Swappable data feeds (CSV, Dhan, Upstox)
3. **Unified Broker Protocol**: Swappable brokers (Simulated, Dhan, Upstox)
4. **Three Execution Modes**: Backtest, Paper, Live
5. **Comprehensive Analytics**: Full metrics + interactive Plotly charts
6. **CSV Persistence**: Trades, equity curve, and results export

The architecture cleanly separates concerns, making it easy to add new data providers, brokers, or strategy types without modifying core engine logic.
