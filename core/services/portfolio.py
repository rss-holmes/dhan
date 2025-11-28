"""Portfolio service."""

from __future__ import annotations

from core.models.portfolio import ConvertPositionRequest, Holding, Position
from core.services.base import BaseService


class PortfolioService(BaseService):
    """Service for portfolio operations."""

    def get_holdings(self) -> list[Holding]:
        """Get all holdings.

        Returns:
            List of all holdings in the account.
        """
        response = self._request("GET", "/holdings")
        if isinstance(response, list):
            return [Holding.model_validate(h) for h in response]
        return []

    def get_positions(self) -> list[Position]:
        """Get all positions.

        Returns:
            List of all open positions.
        """
        response = self._request("GET", "/positions")
        if isinstance(response, list):
            return [Position.model_validate(p) for p in response]
        return []

    def convert_position(self, request: ConvertPositionRequest) -> bool:
        """Convert a position from one product type to another.

        Args:
            request: Position conversion request.

        Returns:
            True if conversion was accepted (HTTP 202).
        """
        body = request.model_dump(by_alias=True, exclude_none=True)
        body["dhanClientId"] = self.client.context.client_id
        self._request("POST", "/positions/convert", body)
        return True
