"""Order-related data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.common import (
    ExchangeSegment,
    OrderStatus,
    OrderType,
    ProductType,
    TransactionType,
    Validity,
)


class OrderRequest(BaseModel):
    """Request model for placing an order."""

    security_id: str = Field(alias="securityId")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    transaction_type: TransactionType = Field(alias="transactionType")
    quantity: int
    order_type: OrderType = Field(alias="orderType")
    product_type: ProductType = Field(alias="productType")
    validity: Validity = Validity.DAY
    price: float = 0
    trigger_price: float | None = Field(default=None, alias="triggerPrice")
    disclosed_quantity: int | None = Field(default=None, alias="disclosedQuantity")
    after_market_order: bool = Field(default=False, alias="afterMarketOrder")
    amo_time: str | None = Field(default=None, alias="amoTime")
    bo_profit_value: float | None = Field(default=None, alias="boProfitValue")
    bo_stop_loss_value: float | None = Field(default=None, alias="boStopLossValue")
    correlation_id: str | None = Field(default=None, alias="correlationId")

    model_config = {"populate_by_name": True}


class SliceOrderRequest(BaseModel):
    """Request model for placing a slice order (for large quantities)."""

    security_id: str = Field(alias="securityId")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    transaction_type: TransactionType = Field(alias="transactionType")
    quantity: int
    order_type: OrderType = Field(alias="orderType")
    product_type: ProductType = Field(alias="productType")
    validity: Validity = Validity.DAY
    price: float = 0
    trigger_price: float | None = Field(default=None, alias="triggerPrice")
    correlation_id: str | None = Field(default=None, alias="correlationId")

    model_config = {"populate_by_name": True}


class Order(BaseModel):
    """Response model for an order."""

    dhan_client_id: str | None = Field(default=None, alias="dhanClientId")
    order_id: str = Field(alias="orderId")
    correlation_id: str | None = Field(default=None, alias="correlationId")
    order_status: OrderStatus = Field(alias="orderStatus")
    transaction_type: TransactionType = Field(alias="transactionType")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    product_type: ProductType = Field(alias="productType")
    order_type: OrderType = Field(alias="orderType")
    validity: Validity
    trading_symbol: str | None = Field(default=None, alias="tradingSymbol")
    security_id: str = Field(alias="securityId")
    quantity: int
    disclosed_quantity: int = Field(default=0, alias="disclosedQuantity")
    price: float
    trigger_price: float = Field(default=0, alias="triggerPrice")
    after_market_order: bool = Field(default=False, alias="afterMarketOrder")
    bo_profit_value: float | None = Field(default=None, alias="boProfitValue")
    bo_stop_loss_value: float | None = Field(default=None, alias="boStopLossValue")
    leg_name: str | None = Field(default=None, alias="legName")
    create_time: str | None = Field(default=None, alias="createTime")
    update_time: str | None = Field(default=None, alias="updateTime")
    exchange_time: str | None = Field(default=None, alias="exchangeTime")
    drv_expiry_date: str | None = Field(default=None, alias="drvExpiryDate")
    drv_option_type: str | None = Field(default=None, alias="drvOptionType")
    drv_strike_price: float | None = Field(default=None, alias="drvStrikePrice")
    oms_error_code: str | None = Field(default=None, alias="omsErrorCode")
    oms_error_description: str | None = Field(default=None, alias="omsErrorDescription")
    filled_qty: int = Field(default=0, alias="filledQty")
    pending_qty: int = Field(default=0, alias="pendingQty")
    cancelled_qty: int = Field(default=0, alias="cancelledQty")
    average_price: float = Field(default=0, alias="averagePrice")
    average_traded_price: float = Field(default=0, alias="averageTradedPrice")

    model_config = {"populate_by_name": True}


class Trade(BaseModel):
    """Response model for a trade."""

    dhan_client_id: str | None = Field(default=None, alias="dhanClientId")
    order_id: str = Field(alias="orderId")
    exchange_order_id: str | None = Field(default=None, alias="exchangeOrderId")
    exchange_trade_id: str | None = Field(default=None, alias="exchangeTradeId")
    transaction_type: TransactionType = Field(alias="transactionType")
    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    product_type: ProductType = Field(alias="productType")
    order_type: OrderType = Field(alias="orderType")
    trading_symbol: str | None = Field(default=None, alias="tradingSymbol")
    security_id: str = Field(alias="securityId")
    traded_quantity: int = Field(alias="tradedQuantity")
    traded_price: float = Field(alias="tradedPrice")
    trade_time: str | None = Field(default=None, alias="tradeTime")
    drv_expiry_date: str | None = Field(default=None, alias="drvExpiryDate")
    drv_option_type: str | None = Field(default=None, alias="drvOptionType")
    drv_strike_price: float | None = Field(default=None, alias="drvStrikePrice")

    model_config = {"populate_by_name": True}
