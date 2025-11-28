"""Portfolio data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.common import ExchangeSegment, PositionType, ProductType


class Holding(BaseModel):
    """Holdings data model."""

    exchange: str
    trading_symbol: str = Field(alias="tradingSymbol")
    security_id: str = Field(alias="securityId")
    isin: str
    total_qty: int = Field(alias="totalQty")
    dp_qty: int = Field(alias="dpQty")
    t1_qty: int = Field(alias="t1Qty")
    available_qty: int = Field(alias="availableQty")
    collateral_qty: int = Field(default=0, alias="collateralQty")
    avg_cost_price: float = Field(alias="avgCostPrice")

    model_config = {"populate_by_name": True}


class Position(BaseModel):
    """Position data model."""

    dhan_client_id: str | None = Field(default=None, alias="dhanClientId")
    trading_symbol: str = Field(alias="tradingSymbol")
    security_id: str = Field(alias="securityId")
    position_type: PositionType = Field(alias="positionType")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    product_type: ProductType = Field(alias="productType")
    buy_avg: float = Field(alias="buyAvg")
    buy_qty: int = Field(alias="buyQty")
    sell_avg: float = Field(alias="sellAvg")
    sell_qty: int = Field(alias="sellQty")
    net_qty: int = Field(alias="netQty")
    cost_price: float = Field(alias="costPrice")
    realized_profit: float = Field(alias="realizedProfit")
    unrealized_profit: float = Field(alias="unrealizedProfit")
    carry_forward_buy_qty: int = Field(default=0, alias="carryForwardBuyQty")
    carry_forward_sell_qty: int = Field(default=0, alias="carryForwardSellQty")
    day_buy_qty: int = Field(default=0, alias="dayBuyQty")
    day_sell_qty: int = Field(default=0, alias="daySellQty")
    drv_expiry_date: str | None = Field(default=None, alias="drvExpiryDate")
    drv_option_type: str | None = Field(default=None, alias="drvOptionType")
    drv_strike_price: float | None = Field(default=None, alias="drvStrikePrice")
    cross_currency: bool = Field(default=False, alias="crossCurrency")

    model_config = {"populate_by_name": True}


class ConvertPositionRequest(BaseModel):
    """Request model for converting a position."""

    from_product_type: ProductType = Field(alias="fromProductType")
    to_product_type: ProductType = Field(alias="toProductType")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    position_type: PositionType = Field(alias="positionType")
    security_id: str = Field(alias="securityId")
    trading_symbol: str = Field(alias="tradingSymbol")
    convert_qty: int = Field(alias="convertQty")

    model_config = {"populate_by_name": True}
