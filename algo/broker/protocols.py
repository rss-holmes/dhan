"""Broker protocol definitions."""

from typing import Callable, Protocol, runtime_checkable

from algo.broker.models import Fill, Order, OrderRequest, Position


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
