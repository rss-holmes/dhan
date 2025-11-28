"""Configuration module for Dhan API client."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class Environment(Enum):
    """API environment."""

    LIVE = "live"
    SANDBOX = "sandbox"


class DhanContext(BaseModel):
    """Context configuration for Dhan API authentication."""

    environment: Environment
    client_id: str
    access_token: str

    model_config = {"frozen": True}

    @computed_field
    @property
    def base_url(self) -> str:
        """Get the base URL for the current environment."""
        return {
            Environment.LIVE: "api.dhan.co",
            Environment.SANDBOX: "sandbox.dhan.co",
        }[self.environment]
