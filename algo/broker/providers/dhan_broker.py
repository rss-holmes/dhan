"""Dhan broker implementation wrapping the existing core/ module."""

from __future__ import annotations

from typing import Callable

from algo.broker.models import (
    Fill,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from core import DhanClient, DhanContext
from core.models.common import ExchangeSegment, Validity
from core.models.common import OrderType as DhanOrderType
from core.models.common import ProductType, TransactionType
from core.models.orders import OrderRequest as DhanOrderRequest


class DhanBroker:
    """
    Dhan broker implementation wrapping the existing core/ module.
    """

    def __init__(
        self,
        context: DhanContext,
        product_type: ProductType = ProductType.INTRADAY,
    ):
        """
        Initialize Dhan broker.

        Args:
            context: Dhan API context with credentials
            product_type: Default product type for orders
        """
        self._client = DhanClient(context)
        self._product_type = product_type
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

        dhan_order_type = (
            DhanOrderType.MARKET
            if request.order_type == OrderType.MARKET
            else DhanOrderType.LIMIT
        )

        dhan_request = DhanOrderRequest(
            securityId=security_id,
            exchangeSegment=ExchangeSegment(segment),
            transactionType=(
                TransactionType.BUY
                if request.side == OrderSide.BUY
                else TransactionType.SELL
            ),
            orderType=dhan_order_type,
            productType=self._product_type,
            quantity=request.quantity,
            price=request.limit_price or 0,
            validity=Validity.DAY,
        )

        dhan_order = self._client.orders.place_order(dhan_request)

        return self._map_dhan_order(dhan_order, request.strategy_id)

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel order via Dhan API."""
        dhan_order = self._client.orders.cancel_order(order_id)
        return self._map_dhan_order(dhan_order)

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> Order:
        """Modify order via Dhan API."""
        kwargs = {}
        if quantity is not None:
            kwargs["quantity"] = quantity
        if price is not None:
            kwargs["price"] = price

        dhan_order = self._client.orders.modify_order(order_id, **kwargs)
        return self._map_dhan_order(dhan_order)

    async def get_order(self, order_id: str) -> Order:
        """Get order by ID from Dhan API."""
        dhan_order = self._client.orders.get_order(order_id)
        return self._map_dhan_order(dhan_order)

    async def get_orders(self) -> list[Order]:
        """Get all orders for the day."""
        dhan_orders = self._client.orders.get_orders()
        return [self._map_dhan_order(o) for o in dhan_orders]

    async def get_positions(self) -> list[Position]:
        """Get positions from Dhan API."""
        dhan_positions = self._client.portfolio.get_positions()
        return [self._map_dhan_position(p) for p in dhan_positions]

    async def get_position(self, instrument: str) -> Position | None:
        """Get position for specific instrument."""
        positions = await self.get_positions()
        for p in positions:
            if p.instrument == instrument:
                return p
        return None

    async def get_balance(self) -> float:
        """Get available balance from Dhan API."""
        fund_limit = self._client.funds.get_fund_limit()
        return fund_limit.available_balance

    async def get_margin_used(self) -> float:
        """Get margin used from Dhan API."""
        fund_limit = self._client.funds.get_fund_limit()
        return fund_limit.utilized_amount

    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        """Register order update callback."""
        self._order_callbacks.append(callback)

    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        """Register fill callback."""
        self._fill_callbacks.append(callback)

    def _map_dhan_order(
        self, dhan_order, strategy_id: str | None = None
    ) -> Order:
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
            side=(
                OrderSide.BUY
                if dhan_order.transaction_type == "BUY"
                else OrderSide.SELL
            ),
            quantity=dhan_order.quantity,
            order_type=(
                OrderType.MARKET
                if dhan_order.order_type == "MARKET"
                else OrderType.LIMIT
            ),
            status=status_map.get(dhan_order.order_status, OrderStatus.PENDING),
            filled_quantity=dhan_order.traded_quantity or 0,
            average_price=dhan_order.traded_price or 0.0,
            limit_price=dhan_order.price if dhan_order.price > 0 else None,
            strategy_id=strategy_id,
        )

    def _map_dhan_position(self, dhan_position) -> Position:
        """Map Dhan position to internal Position model."""
        # Calculate current price from sell data or cost price
        if dhan_position.day_sell_qty and dhan_position.day_sell_qty > 0:
            current_price = dhan_position.day_sell_value / dhan_position.day_sell_qty
        else:
            current_price = dhan_position.cost_price

        return Position(
            instrument=f"{dhan_position.exchange_segment}:{dhan_position.security_id}",
            quantity=dhan_position.net_qty,
            average_price=dhan_position.cost_price,
            current_price=current_price,
            unrealized_pnl=dhan_position.unrealized_profit,
            realized_pnl=dhan_position.realized_profit,
        )

    def switch_context(self, context: DhanContext) -> None:
        """Switch to a different Dhan context."""
        self._client.switch_context(context)
