"""Broker models for orders, fills, and positions."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderRequest(BaseModel):
    """Request to place a new order."""

    instrument: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    limit_price: float | None = Field(default=None, gt=0)
    strategy_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class Order(BaseModel):
    """Represents an order in the system."""

    order_id: str
    instrument: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    status: OrderStatus
    limit_price: float | None = Field(default=None, gt=0)
    filled_quantity: int = Field(default=0, ge=0)
    average_price: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    strategy_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"validate_assignment": True}

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @computed_field
    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity


class Fill(BaseModel):
    """Represents an order fill (execution)."""

    fill_id: str
    order_id: str
    instrument: str
    side: OrderSide
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime
    commission: float = Field(default=0.0, ge=0)
    strategy_id: str | None = None

    model_config = {"frozen": True}


class Position(BaseModel):
    """Represents an open position."""

    instrument: str
    quantity: int  # Positive for long, negative for short
    average_price: float = Field(ge=0)
    current_price: float = Field(ge=0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    model_config = {"validate_assignment": True}

    @computed_field
    @property
    def is_long(self) -> bool:
        """True if position is long."""
        return self.quantity > 0

    @computed_field
    @property
    def is_short(self) -> bool:
        """True if position is short."""
        return self.quantity < 0

    @computed_field
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return abs(self.quantity) * self.current_price
