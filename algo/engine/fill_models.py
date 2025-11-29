"""Fill models for simulating order execution."""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field

from algo.broker.models import Order, OrderSide, OrderType
from algo.data.models import Candle


class FillModelType(Enum):
    """Fill model types."""

    NEXT_OPEN = "next_open"
    SAME_CLOSE = "same_close"
    VWAP = "vwap"
    SLIPPAGE = "slippage"


class FillResult(BaseModel):
    """Result of fill calculation."""

    filled: bool
    fill_price: float = Field(ge=0)
    fill_quantity: int = Field(ge=0)
    slippage: float = Field(default=0.0, ge=0)

    model_config = {"frozen": True}


class FillModel(ABC):
    """Abstract base for fill simulation models."""

    @abstractmethod
    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        """
        Calculate fill price and quantity.

        Args:
            order: The order to fill
            reference_price: Current market price
            candle: Optional candle data for more accurate fills

        Returns:
            FillResult with fill details
        """
        pass


class NextOpenFillModel(FillModel):
    """
    Fill at next candle's open price.

    Assumes signals generated on candle close, orders fill at next open.
    """

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        if candle is None:
            return FillResult(filled=False, fill_price=0, fill_quantity=0)

        fill_price = candle.open

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
        )


class SameCloseFillModel(FillModel):
    """
    Fill at same candle's close price.

    Assumes orders fill at the close of the signal candle.
    """

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        fill_price = reference_price

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
        )


class SlippageFillModel(FillModel):
    """
    Fill with configurable slippage.

    Applies slippage adverse to the trader (higher for buys, lower for sells).
    """

    def __init__(self, slippage_pct: float = 0.001):
        """
        Initialize with slippage percentage.

        Args:
            slippage_pct: Slippage as percentage (0.001 = 0.1%)
        """
        self._slippage_pct = slippage_pct

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        # Apply slippage (adverse to trader)
        if order.side == OrderSide.BUY:
            fill_price = reference_price * (1 + self._slippage_pct)
        else:
            fill_price = reference_price * (1 - self._slippage_pct)

        slippage = abs(fill_price - reference_price)

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
            slippage=slippage,
        )

    @property
    def slippage_pct(self) -> float:
        """Get slippage percentage."""
        return self._slippage_pct


class VWAPFillModel(FillModel):
    """
    Fill at Volume Weighted Average Price.

    Uses candle's typical price as VWAP approximation.
    """

    def __init__(self, slippage_pct: float = 0.0):
        """
        Initialize with optional slippage.

        Args:
            slippage_pct: Additional slippage percentage
        """
        self._slippage_pct = slippage_pct

    def calculate_fill(
        self,
        order: Order,
        reference_price: float,
        candle: Candle | None = None,
    ) -> FillResult:
        if candle is None:
            fill_price = reference_price
        else:
            # Use typical price as VWAP approximation
            fill_price = candle.typical_price

        # Apply slippage
        if self._slippage_pct > 0:
            if order.side == OrderSide.BUY:
                fill_price *= 1 + self._slippage_pct
            else:
                fill_price *= 1 - self._slippage_pct

        slippage = abs(fill_price - reference_price)

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return FillResult(filled=False, fill_price=0, fill_quantity=0)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.remaining_quantity,
            slippage=slippage,
        )
