"""Custom exceptions for Dhan API client."""

from __future__ import annotations

from typing import Any


class DhanAPIError(Exception):
    """Exception raised when Dhan API returns an error response."""

    def __init__(self, status_code: int, response: dict[str, Any]) -> None:
        self.status_code = status_code
        self.response = response
        self.message = response.get("message", response.get("errorMessage", str(response)))
        super().__init__(f"API Error ({status_code}): {self.message}")


class DhanAuthenticationError(DhanAPIError):
    """Exception raised for authentication failures (401/403)."""

    pass


class DhanValidationError(DhanAPIError):
    """Exception raised for validation errors (400)."""

    pass


class DhanRateLimitError(DhanAPIError):
    """Exception raised when rate limit is exceeded (429)."""

    pass
