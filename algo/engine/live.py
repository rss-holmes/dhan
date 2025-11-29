"""Live trading engine for production trading."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from algo.broker.models import OrderRequest, OrderSide, OrderType
from algo.data.models import Candle, Tick
from algo.engine.base import BaseEngine
from algo.strategy.base import BaseStrategy, Signal, SignalAction
from algo.strategy.context import StrategyContext

if TYPE_CHECKING:
    from algo.broker.models import Position
    from algo.broker.protocols import Broker
    from algo.data.protocols import DataFeed


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
    ):
        """
        Initialize live trading engine.

        Args:
            data_feed: Real-time market data feed
            broker: Broker for order execution
        """
        self._data_feed = data_feed
        self._broker = broker

        self._strategies: dict[str, BaseStrategy] = {}
        self._candle_history: dict[str, list[Candle]] = {}
        self._current_prices: dict[str, float] = {}
        self._running = False

        self._initial_capital: float = 0.0
        self._equity_curve: list[tuple[datetime, float]] = []

        # Max history size
        self._max_history = 500

        # Callbacks
        self._on_signal_callbacks: list[Callable[[str, Signal], None]] = []
        self._on_error_callbacks: list[Callable[[str, Exception], None]] = []

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
            context = StrategyContext(self, strategy_id)
            strategy.set_context(context)
            strategy.on_init()

        # Subscribe to instruments
        all_instruments: set[str] = set()
        for strategy in self._strategies.values():
            all_instruments.update(strategy.instruments)

        await self._data_feed.subscribe(list(all_instruments))

        # Register callbacks
        self._data_feed.on_tick(self._on_tick)
        self._data_feed.on_candle(self._on_candle)
        self._broker.on_fill(self._on_fill)

        print(f"Live trading started with {len(self._strategies)} strategies")
        print(f"Capital: ${self._initial_capital:,.2f}")

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
        balance = await self._broker.get_balance()
        positions = await self._broker.get_positions()
        positions_value = sum(p.market_value for p in positions)
        self._initial_capital = balance + positions_value

        # Update current prices from positions
        for position in positions:
            self._current_prices[position.instrument] = position.current_price

        print(f"Synced state: {len(positions)} positions")

    def _on_tick(self, tick: Tick) -> None:
        """Handle incoming tick data."""
        self._current_prices[tick.instrument] = tick.ltp

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
        if len(self._candle_history[candle.instrument]) > self._max_history:
            self._candle_history[candle.instrument] = self._candle_history[
                candle.instrument
            ][-self._max_history :]

        # Update price
        self._current_prices[candle.instrument] = candle.close

        # Record equity
        equity = await self._calculate_equity()
        self._equity_curve.append((candle.timestamp, equity))

        # Process through strategies
        for strategy_id, strategy in self._strategies.items():
            if candle.instrument in strategy.instruments:
                try:
                    signals = strategy.on_candle(candle)

                    if signals:
                        if isinstance(signals, Signal):
                            signals = [signals]

                        for signal in signals:
                            # Notify callbacks
                            for callback in self._on_signal_callbacks:
                                try:
                                    callback(strategy_id, signal)
                                except Exception:
                                    pass

                            await self._process_signal(strategy_id, signal)

                except Exception as e:
                    print(f"Error in strategy {strategy_id}: {e}")
                    for callback in self._on_error_callbacks:
                        try:
                            callback(strategy_id, e)
                        except Exception:
                            pass

    async def _process_signal(self, strategy_id: str, signal: Signal) -> None:
        """Convert signal to order and submit to broker."""
        if signal.action == SignalAction.HOLD:
            return

        request = OrderRequest(
            instrument=signal.instrument,
            side=OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL,
            quantity=signal.quantity,
            order_type=(
                OrderType.MARKET
                if signal.order_type == OrderType.MARKET
                else OrderType.LIMIT
            ),
            limit_price=signal.limit_price,
            strategy_id=strategy_id,
            metadata=signal.metadata,
        )

        try:
            order = await self._broker.place_order(request)
            print(
                f"[LIVE] Order placed: {order.order_id} - "
                f"{signal.action.value} {signal.quantity} {signal.instrument}"
            )
        except Exception as e:
            print(f"[LIVE] Order failed: {e}")
            for callback in self._on_error_callbacks:
                try:
                    callback(strategy_id, e)
                except Exception:
                    pass

    def _on_fill(self, fill) -> None:
        """Handle order fill."""
        print(
            f"[LIVE] Fill: {fill.side.value} {fill.quantity} "
            f"{fill.instrument} @ {fill.price:.2f}"
        )

        # Notify strategy
        if fill.strategy_id and fill.strategy_id in self._strategies:
            self._strategies[fill.strategy_id].on_order_fill(fill)

    async def _calculate_equity(self) -> float:
        """Calculate current portfolio equity."""
        balance = await self._broker.get_balance()
        positions = await self._broker.get_positions()
        positions_value = sum(p.market_value for p in positions)
        return balance + positions_value

    # === BaseEngine interface implementation ===

    def get_historical(
        self,
        instrument: str,
        periods: int,
        field: str = "close",
    ) -> list[float]:
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

    def get_position(self, instrument: str) -> Position | None:
        """Get current position for instrument."""
        # This is synchronous but we need async broker call
        # For live trading, we cache positions and update on fills
        return asyncio.get_event_loop().run_until_complete(
            self._broker.get_position(instrument)
        )

    def get_balance(self) -> float:
        """Get available cash balance."""
        return asyncio.get_event_loop().run_until_complete(self._broker.get_balance())

    def get_equity(self) -> float:
        """Get total portfolio equity."""
        return asyncio.get_event_loop().run_until_complete(self._calculate_equity())

    def submit_order(self, strategy_id: str, signal: Signal) -> str:
        """Submit an order."""
        asyncio.create_task(self._process_signal(strategy_id, signal))
        return f"live_{datetime.now().timestamp()}"

    # === Callbacks ===

    def on_signal(self, callback: Callable[[str, Signal], None]) -> None:
        """Register callback for signals."""
        self._on_signal_callbacks.append(callback)

    def on_error(self, callback: Callable[[str, Exception], None]) -> None:
        """Register callback for errors."""
        self._on_error_callbacks.append(callback)

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    @property
    def broker(self) -> Broker:
        """Get the broker instance."""
        return self._broker

    @property
    def data_feed(self) -> DataFeed:
        """Get the data feed instance."""
        return self._data_feed
