"""Funds service."""

from __future__ import annotations

from core.models.funds import FundLimit, MarginCalculatorRequest, MarginDetails
from core.services.base import BaseService


class FundsService(BaseService):
    """Service for funds and margin operations."""

    def get_fund_limit(self) -> FundLimit:
        """Get fund limit and balance information.

        Returns:
            Fund limit details including available balance.
        """
        response = self._request("GET", "/fundlimit")
        return FundLimit.model_validate(response)

    def calculate_margin(self, request: MarginCalculatorRequest) -> MarginDetails:
        """Calculate margin requirements for an order.

        Args:
            request: Margin calculator request with order details.

        Returns:
            Margin details including required margin and brokerage.
        """
        body = request.model_dump(by_alias=True, exclude_none=True)
        body["dhanClientId"] = self.client.context.client_id
        response = self._request("POST", "/margincalculator", body)
        return MarginDetails.model_validate(response)
