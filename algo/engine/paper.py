"""Paper trading engine for simulated live trading."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from algo.broker.models import OrderRequest, OrderSide, OrderType
from algo.broker.providers.simulated import SimulatedBroker
from algo.data.models import Candle, Tick
from algo.engine.base import BaseEngine
from algo.engine.fill_models import FillModel, SlippageFillModel
from algo.strategy.base import BaseStrategy, Signal, SignalAction
from algo.strategy.context import StrategyContext

if TYPE_CHECKING:
    from algo.analytics.metrics import BacktestResult
    from algo.broker.models import Position
    from algo.data.protocols import DataFeed


class PaperTradingEngine(BaseEngine):
    """
    Engine for paper trading with real market data but simulated orders.

    Uses real-time market data from a data feed but executes orders
    through a simulated broker.
    """

    def __init__(
        self,
        data_feed: DataFeed,
        initial_capital: float = 100000.0,
        fill_model: FillModel | None = None,
        commission_rate: float = 0.0,
    ):
        """
        Initialize paper trading engine.

        Args:
            data_feed: Real-time market data feed
            initial_capital: Starting capital
            fill_model: Model for simulating fills
            commission_rate: Commission as percentage
        """
        self._data_feed = data_feed
        self._initial_capital = initial_capital
        self._fill_model = fill_model or SlippageFillModel(slippage_pct=0.0005)
        self._commission_rate = commission_rate

        self._broker = SimulatedBroker(
            initial_capital=initial_capital,
            fill_model=self._fill_model,
            commission_rate=commission_rate,
        )

        self._strategies: dict[str, BaseStrategy] = {}
        self._candle_history: dict[str, list[Candle]] = {}
        self._equity_curve: list[tuple[datetime, float]] = []
        self._running = False
        self._on_equity_callbacks: list[Callable[[float], None]] = []

        # Max history size
        self._max_history = 500

    def add_strategy(self, strategy: BaseStrategy) -> str:
        """Add a strategy to paper trading."""
        strategy_id = f"{strategy.name}_{len(self._strategies)}"
        self._strategies[strategy_id] = strategy
        return strategy_id

    async def start(self) -> None:
        """Start paper trading."""
        self._running = True

        # Connect to data feed
        await self._data_feed.connect()

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

        print(f"Paper trading started with {len(self._strategies)} strategies")
        print(f"Initial capital: ${self._initial_capital:,.2f}")

    async def stop(self) -> BacktestResult:
        """Stop paper trading and return results."""
        self._running = False

        for strategy in self._strategies.values():
            strategy.on_stop()

        await self._data_feed.disconnect()

        print("Paper trading stopped")

        return self._calculate_results()

    def _on_tick(self, tick: Tick) -> None:
        """Handle incoming tick data."""
        # Update broker prices
        self._broker.update_price(tick.instrument, tick.ltp)

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

        # Update broker price
        self._broker.update_price(candle.instrument, candle.close)

        # Record equity
        equity = self._broker.equity
        self._equity_curve.append((candle.timestamp, equity))

        # Notify equity callbacks
        for callback in self._on_equity_callbacks:
            try:
                callback(equity)
            except Exception as e:
                print(f"Error in equity callback: {e}")

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
        """Convert signal to order and submit."""
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
                f"[PAPER] Order placed: {order.order_id[:8]}... - "
                f"{signal.action.value} {signal.quantity} {signal.instrument}"
            )
        except Exception as e:
            print(f"[PAPER] Order failed: {e}")

    def _on_fill(self, fill) -> None:
        """Handle order fill."""
        print(
            f"[PAPER] Fill: {fill.side.value} {fill.quantity} "
            f"{fill.instrument} @ {fill.price:.2f}"
        )

        # Notify strategy
        if fill.strategy_id and fill.strategy_id in self._strategies:
            self._strategies[fill.strategy_id].on_order_fill(fill)

    def _calculate_results(self) -> BacktestResult:
        """Calculate paper trading results."""
        from algo.analytics.metrics import BacktestResult
        from algo.portfolio.performance import PerformanceCalculator

        calculator = PerformanceCalculator()
        metrics = calculator.calculate(
            equity_curve=self._equity_curve,
            trades=list(self._broker.orders.values()),
            fills=self._broker.fills,
            initial_capital=self._initial_capital,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=self._equity_curve,
            trades=list(self._broker.orders.values()),
            fills=self._broker.fills,
            strategies={sid: s.name for sid, s in self._strategies.items()},
        )

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
        return self._broker._positions.get(instrument)

    def get_balance(self) -> float:
        """Get available cash balance."""
        return self._broker.cash

    def get_equity(self) -> float:
        """Get total portfolio equity."""
        return self._broker.equity

    def submit_order(self, strategy_id: str, signal: Signal) -> str:
        """Submit an order."""
        asyncio.create_task(self._process_signal(strategy_id, signal))
        return f"paper_{len(self._broker.orders)}"

    def on_equity_update(self, callback: Callable[[float], None]) -> None:
        """Register callback for equity updates."""
        self._on_equity_callbacks.append(callback)

    @property
    def broker(self) -> SimulatedBroker:
        """Get the broker instance."""
        return self._broker

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running
