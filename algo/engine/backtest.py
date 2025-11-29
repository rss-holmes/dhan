"""Backtesting engine for historical simulation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from algo.broker.models import OrderRequest, OrderSide, OrderType
from algo.broker.providers.simulated import SimulatedBroker
from algo.data.models import Candle, Interval
from algo.data.providers.csv_feed import CSVDataFeed
from algo.engine.base import BaseEngine
from algo.engine.fill_models import FillModel, NextOpenFillModel
from algo.strategy.base import BaseStrategy, Signal, SignalAction
from algo.strategy.context import StrategyContext

if TYPE_CHECKING:
    from algo.analytics.metrics import BacktestResult
    from algo.broker.models import Position


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
        """
        Initialize backtest engine.

        Args:
            data_dir: Directory containing CSV data files
            initial_capital: Starting capital
            fill_model: Model for simulating fills (default: NextOpenFillModel)
            commission_rate: Commission as percentage (0.001 = 0.1%)
            interval: Data interval/timeframe
        """
        self._data_dir = Path(data_dir)
        self._initial_capital = initial_capital
        self._fill_model = fill_model or NextOpenFillModel()
        self._commission_rate = commission_rate
        self._interval = interval

        self._data_feed: CSVDataFeed | None = None
        self._broker: SimulatedBroker | None = None
        self._strategies: dict[str, BaseStrategy] = {}

        self._candle_history: dict[str, list[Candle]] = {}
        self._equity_curve: list[tuple[datetime, float]] = []
        self._current_candle: Candle | None = None
        self._next_candle: Candle | None = None
        self._pending_signals: list[tuple[str, Signal]] = []

    def add_strategy(self, strategy: BaseStrategy) -> str:
        """Add a strategy to the backtest."""
        strategy_id = f"{strategy.name}_{len(self._strategies)}"
        self._strategies[strategy_id] = strategy
        return strategy_id

    def run(self) -> BacktestResult:
        """Run the backtest synchronously and return results."""
        return asyncio.get_event_loop().run_until_complete(self.run_async())

    async def run_async(self) -> BacktestResult:
        """Run the backtest asynchronously and return results."""
        # Collect all instruments from strategies
        all_instruments: set[str] = set()
        for strategy in self._strategies.values():
            all_instruments.update(strategy.instruments)

        if not all_instruments:
            raise ValueError("No instruments to trade. Add strategies with instruments.")

        # Initialize components
        self._data_feed = CSVDataFeed(self._data_dir, self._interval)
        self._data_feed.set_instruments(list(all_instruments))

        self._broker = SimulatedBroker(
            initial_capital=self._initial_capital,
            fill_model=self._fill_model,
            commission_rate=self._commission_rate,
        )

        # Initialize strategy contexts
        for strategy_id, strategy in self._strategies.items():
            context = StrategyContext(self, strategy_id)
            strategy.set_context(context)
            strategy.on_init()

        # Setup fill callback
        self._broker.on_fill(self._on_fill)

        # Reset state
        self._candle_history.clear()
        self._equity_curve.clear()
        self._pending_signals.clear()

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
        assert self._broker is not None, "Broker not initialized"

        # Process any pending signals from previous candle (fill at open)
        await self._process_pending_signals(candle)

        # Update price in broker
        self._broker.update_price(candle.instrument, candle.close)

        # Store candle in history
        if candle.instrument not in self._candle_history:
            self._candle_history[candle.instrument] = []
        self._candle_history[candle.instrument].append(candle)

        # Record equity
        equity = self._broker.equity
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
                            # Queue signal for next candle open
                            self._pending_signals.append((strategy_id, signal))
                except Exception as e:
                    print(f"Error in strategy {strategy_id}: {e}")

    async def _process_pending_signals(self, candle: Candle) -> None:
        """Process pending signals at candle open."""
        signals_to_process = [
            (sid, sig)
            for sid, sig in self._pending_signals
            if sig.instrument == candle.instrument
        ]

        # Remove processed signals
        self._pending_signals = [
            (sid, sig)
            for sid, sig in self._pending_signals
            if sig.instrument != candle.instrument
        ]

        for strategy_id, signal in signals_to_process:
            await self._execute_signal(strategy_id, signal, candle)

    async def _execute_signal(
        self, strategy_id: str, signal: Signal, candle: Candle
    ) -> None:
        """Execute a signal as an order."""
        assert self._broker is not None, "Broker not initialized"

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

        # For backtest, we fill at candle open using fill model
        order = await self._broker.place_order(request)

        # If market order and not filled yet, fill at open
        if order.order_type == OrderType.MARKET and order.filled_quantity == 0:
            fill_result = self._fill_model.calculate_fill(
                order=order,
                reference_price=candle.open,
                candle=candle,
            )
            if fill_result.filled:
                self._broker._execute_fill(order, fill_result)

    def _on_fill(self, fill) -> None:
        """Handle order fill."""
        # Notify strategy
        if fill.strategy_id and fill.strategy_id in self._strategies:
            self._strategies[fill.strategy_id].on_order_fill(fill)

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics."""
        from algo.analytics.metrics import BacktestResult
        from algo.portfolio.performance import PerformanceCalculator

        assert self._broker is not None, "Broker not initialized"

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
        if self._broker is None:
            return None
        return self._broker._positions.get(instrument)

    def get_balance(self) -> float:
        """Get available cash balance."""
        if self._broker is None:
            return 0.0
        return self._broker.cash

    def get_equity(self) -> float:
        """Get total portfolio equity."""
        if self._broker is None:
            return 0.0
        return self._broker.equity

    def submit_order(self, strategy_id: str, signal: Signal) -> str:
        """Submit an order (queues for next candle in backtest)."""
        self._pending_signals.append((strategy_id, signal))
        return f"pending_{len(self._pending_signals)}"

    @property
    def broker(self) -> SimulatedBroker | None:
        """Get the broker instance."""
        return self._broker

    @property
    def data_feed(self) -> CSVDataFeed | None:
        """Get the data feed instance."""
        return self._data_feed
