"""Simulated broker for backtesting and paper trading."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from algo.broker.models import (
    Fill,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

if TYPE_CHECKING:
    from algo.engine.fill_models import FillModel, FillResult


class SimulatedBroker:
    """
    Simulated broker for backtesting and paper trading.

    Handles order matching, position tracking, and P&L calculation.
    """

    def __init__(
        self,
        initial_capital: float,
        fill_model: FillModel,
        commission_rate: float = 0.0,
    ):
        """
        Initialize simulated broker.

        Args:
            initial_capital: Starting cash balance
            fill_model: Model for simulating order fills
            commission_rate: Commission as percentage of trade value (0.001 = 0.1%)
        """
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._fill_model = fill_model
        self._commission_rate = commission_rate

        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._fills: list[Fill] = []

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
        """
        Update current price for an instrument.

        Called by engine to update prices for position valuation
        and pending order checks.
        """
        self._current_prices[instrument] = price
        self._update_position_pnl(instrument, price)
        self._check_pending_orders(instrument, price)

    def _update_position_pnl(self, instrument: str, price: float) -> None:
        """Update unrealized P&L for a position."""
        position = self._positions.get(instrument)
        if position:
            if position.quantity > 0:
                pnl = (price - position.average_price) * position.quantity
            else:
                pnl = (position.average_price - price) * abs(position.quantity)
            # Create new position with updated values
            self._positions[instrument] = Position(
                instrument=position.instrument,
                quantity=position.quantity,
                average_price=position.average_price,
                current_price=price,
                unrealized_pnl=pnl,
                realized_pnl=position.realized_pnl,
            )

    def _check_pending_orders(self, instrument: str, price: float) -> None:
        """Check if any pending limit orders should be filled."""
        from algo.data.models import Candle, Interval

        orders_to_remove = []
        for order in self._pending_orders:
            if order.instrument != instrument:
                continue

            # Check if limit order can be filled
            if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                should_fill = False
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    should_fill = True

                if should_fill:
                    # Create a dummy candle for fill model
                    candle = Candle(
                        instrument=instrument,
                        timestamp=datetime.now(),
                        interval=Interval.DAY_1,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0,
                    )
                    fill_result = self._fill_model.calculate_fill(
                        order=order,
                        reference_price=order.limit_price,
                        candle=candle,
                    )
                    if fill_result.filled:
                        self._execute_fill(order, fill_result)
                        orders_to_remove.append(order)

        for order in orders_to_remove:
            self._pending_orders.remove(order)

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
            order.updated_at = datetime.now()
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
        commission = (
            fill_result.fill_price * fill_result.fill_quantity * self._commission_rate
        )

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
        self._fills.append(fill)

        # Update order status
        old_filled = order.filled_quantity
        new_filled = old_filled + fill_result.fill_quantity

        # Calculate new average price
        if old_filled > 0:
            new_avg = (
                order.average_price * old_filled
                + fill_result.fill_price * fill_result.fill_quantity
            ) / new_filled
        else:
            new_avg = fill_result.fill_price

        order.filled_quantity = new_filled
        order.average_price = new_avg

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
            new_realized_pnl = position.realized_pnl

            if fill.side == OrderSide.BUY:
                if position.quantity >= 0:
                    # Adding to long position
                    total_cost = (
                        position.average_price * position.quantity
                        + fill.price * fill.quantity
                    )
                    new_qty = position.quantity + fill.quantity
                    new_avg = total_cost / new_qty if new_qty > 0 else 0
                else:
                    # Closing short position
                    close_qty = min(fill.quantity, abs(position.quantity))
                    new_realized_pnl += (position.average_price - fill.price) * close_qty
                    new_qty = position.quantity + fill.quantity
                    new_avg = fill.price if new_qty > 0 else position.average_price
            else:  # SELL
                if position.quantity <= 0:
                    # Adding to short position
                    total_cost = (
                        abs(position.average_price * position.quantity)
                        + fill.price * fill.quantity
                    )
                    new_qty = position.quantity - fill.quantity
                    new_avg = (
                        total_cost / abs(new_qty) if new_qty != 0 else position.average_price
                    )
                else:
                    # Closing long position
                    close_qty = min(fill.quantity, position.quantity)
                    new_realized_pnl += (fill.price - position.average_price) * close_qty
                    new_qty = position.quantity - fill.quantity
                    new_avg = fill.price if new_qty < 0 else position.average_price

            if new_qty == 0:
                # Position closed
                del self._positions[fill.instrument]
            else:
                self._positions[fill.instrument] = Position(
                    instrument=fill.instrument,
                    quantity=new_qty,
                    average_price=new_avg,
                    current_price=self._current_prices.get(fill.instrument, fill.price),
                    realized_pnl=new_realized_pnl,
                )

    async def cancel_order(self, order_id: str) -> Order | None:
        """Cancel a pending order."""
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            if order in self._pending_orders:
                self._pending_orders.remove(order)
            self._notify_order_update(order)
        return order

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> Order | None:
        """Modify a pending order."""
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            if quantity is not None:
                order.quantity = quantity
            if price is not None:
                order.limit_price = price
            order.updated_at = datetime.now()
            self._notify_order_update(order)
        return order

    async def get_order(self, order_id: str) -> Order:
        """Get order by ID."""
        return self._orders[order_id]

    async def get_orders(self) -> list[Order]:
        """Get all orders."""
        return list(self._orders.values())

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        return list(self._positions.values())

    async def get_position(self, instrument: str) -> Position | None:
        """Get position for specific instrument."""
        return self._positions.get(instrument)

    async def get_balance(self) -> float:
        """Get current cash balance."""
        return self._cash

    async def get_margin_used(self) -> float:
        """Get margin used (not applicable for simulated)."""
        return 0.0

    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        """Register order update callback."""
        self._order_callbacks.append(callback)

    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        """Register fill callback."""
        self._fill_callbacks.append(callback)

    def _notify_order_update(self, order: Order) -> None:
        """Notify all order callbacks."""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                print(f"Error in order callback: {e}")

    def _notify_fill(self, fill: Fill) -> None:
        """Notify all fill callbacks."""
        for callback in self._fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                print(f"Error in fill callback: {e}")

    @property
    def initial_capital(self) -> float:
        """Get initial capital."""
        return self._initial_capital

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash

    @property
    def equity(self) -> float:
        """Get total equity (cash + positions value)."""
        positions_value = sum(p.market_value for p in self._positions.values())
        return self._cash + positions_value

    @property
    def fills(self) -> list[Fill]:
        """Get all fills."""
        return self._fills.copy()

    @property
    def orders(self) -> dict[str, Order]:
        """Get all orders."""
        return self._orders.copy()

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._cash = self._initial_capital
        self._orders.clear()
        self._positions.clear()
        self._pending_orders.clear()
        self._fills.clear()
        self._current_prices.clear()
