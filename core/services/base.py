"""Base service class with HTTP request handling."""

from __future__ import annotations

import http.client
import json
from typing import TYPE_CHECKING, Any

from core.exceptions import (
    DhanAPIError,
    DhanAuthenticationError,
    DhanRateLimitError,
    DhanValidationError,
)

if TYPE_CHECKING:
    from core.client import DhanClient


class BaseService:
    """Base class for API services with shared HTTP functionality."""

    def __init__(self, client: DhanClient) -> None:
        self.client = client

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            path: API endpoint path (without /v2 prefix).
            body: Optional request body.

        Returns:
            Parsed JSON response.

        Raises:
            DhanAPIError: If the API returns an error response.
        """
        conn = http.client.HTTPSConnection(self.client.context.base_url)

        headers = {
            "client-id": self.client.context.client_id,
            "access-token": self.client.context.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = json.dumps(body) if body else None
        full_path = f"/v2{path}"

        try:
            conn.request(method, full_path, payload, headers)
            response = conn.getresponse()
            data = response.read().decode("utf-8")

            # Handle empty responses (like 202 Accepted)
            if not data:
                return {}

            parsed = json.loads(data)

            if response.status >= 400:
                self._raise_for_status(response.status, parsed)

            return parsed
        finally:
            conn.close()

    def _raise_for_status(
        self, status_code: int, response: dict[str, Any]
    ) -> None:
        """Raise appropriate exception based on status code."""
        if status_code == 400:
            raise DhanValidationError(status_code, response)
        elif status_code in (401, 403):
            raise DhanAuthenticationError(status_code, response)
        elif status_code == 429:
            raise DhanRateLimitError(status_code, response)
        else:
            raise DhanAPIError(status_code, response)
