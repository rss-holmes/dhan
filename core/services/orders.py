"""Orders service for order management."""

from __future__ import annotations

from core.models.common import OrderType, Validity
from core.models.orders import Order, OrderRequest, SliceOrderRequest, Trade
from core.services.base import BaseService


class OrdersService(BaseService):
    """Service for order management operations."""

    def place_order(self, request: OrderRequest) -> Order:
        """Place a new order.

        Args:
            request: Order request with all required parameters.

        Returns:
            Order response with order ID and status.
        """
        body = request.model_dump(by_alias=True, exclude_none=True)
        body["dhanClientId"] = self.client.context.client_id
        response = self._request("POST", "/orders", body)
        return Order.model_validate(response)

    def place_slice_order(self, request: SliceOrderRequest) -> list[Order]:
        """Place a slice order (for large quantities exceeding freeze limit).

        Args:
            request: Slice order request.

        Returns:
            List of orders created from slicing.
        """
        body = request.model_dump(by_alias=True, exclude_none=True)
        body["dhanClientId"] = self.client.context.client_id
        response = self._request("POST", "/orders/slicing", body)
        if isinstance(response, list):
            return [Order.model_validate(o) for o in response]
        return [Order.model_validate(response)]

    def modify_order(
        self,
        order_id: str,
        *,
        order_type: OrderType | None = None,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        validity: Validity | None = None,
        disclosed_quantity: int | None = None,
        leg_name: str | None = None,
    ) -> Order:
        """Modify a pending order.

        Args:
            order_id: The order ID to modify.
            order_type: New order type (optional).
            quantity: New quantity (optional).
            price: New price (optional).
            trigger_price: New trigger price (optional).
            validity: New validity (optional).
            disclosed_quantity: New disclosed quantity (optional).
            leg_name: Leg name for BO/CO orders (optional).

        Returns:
            Updated order details.
        """
        body: dict = {"dhanClientId": self.client.context.client_id, "orderId": order_id}

        if order_type is not None:
            body["orderType"] = order_type.value
        if quantity is not None:
            body["quantity"] = quantity
        if price is not None:
            body["price"] = price
        if trigger_price is not None:
            body["triggerPrice"] = trigger_price
        if validity is not None:
            body["validity"] = validity.value
        if disclosed_quantity is not None:
            body["disclosedQuantity"] = disclosed_quantity
        if leg_name is not None:
            body["legName"] = leg_name

        response = self._request("PUT", f"/orders/{order_id}", body)
        return Order.model_validate(response)

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            Cancelled order details.
        """
        response = self._request("DELETE", f"/orders/{order_id}")
        return Order.model_validate(response)

    def get_orders(self) -> list[Order]:
        """Get all orders for the day.

        Returns:
            List of all orders placed today.
        """
        response = self._request("GET", "/orders")
        if isinstance(response, list):
            return [Order.model_validate(o) for o in response]
        return []

    def get_order(self, order_id: str) -> Order:
        """Get order details by order ID.

        Args:
            order_id: The order ID to fetch.

        Returns:
            Order details.
        """
        response = self._request("GET", f"/orders/{order_id}")
        return Order.model_validate(response)

    def get_order_by_correlation_id(self, correlation_id: str) -> Order:
        """Get order details by correlation ID.

        Args:
            correlation_id: The correlation ID to fetch.

        Returns:
            Order details.
        """
        response = self._request("GET", f"/orders/external/{correlation_id}")
        return Order.model_validate(response)

    def get_trades(self) -> list[Trade]:
        """Get all trades for the day.

        Returns:
            List of all trades executed today.
        """
        response = self._request("GET", "/trades")
        if isinstance(response, list):
            return [Trade.model_validate(t) for t in response]
        return []

    def get_trades_by_order(self, order_id: str) -> list[Trade]:
        """Get trades for a specific order.

        Args:
            order_id: The order ID to fetch trades for.

        Returns:
            List of trades for the order.
        """
        response = self._request("GET", f"/trades/{order_id}")
        if isinstance(response, list):
            return [Trade.model_validate(t) for t in response]
        return [Trade.model_validate(response)]
