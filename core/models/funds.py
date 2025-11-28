"""Funds and margin data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.common import ExchangeSegment, ProductType, TransactionType


class FundLimit(BaseModel):
    """Fund limit response model."""

    dhan_client_id: str | None = Field(default=None, alias="dhanClientId")
    available_balance: float = Field(alias="availabelBalance")  # Note: API typo preserved
    sod_limit: float = Field(alias="sodLimit")
    collateral_amount: float = Field(alias="collateralAmount")
    receivable_amount: float = Field(alias="receiveableAmount")  # Note: API typo preserved
    utilized_amount: float = Field(alias="utilizedAmount")
    blocked_payout_amount: float = Field(alias="blockedPayoutAmount")
    withdrawable_balance: float = Field(alias="withdrawableBalance")

    model_config = {"populate_by_name": True}


class MarginCalculatorRequest(BaseModel):
    """Request model for margin calculator."""

    exchange_segment: ExchangeSegment = Field(alias="exchangeSegment")
    transaction_type: TransactionType = Field(alias="transactionType")
    quantity: int
    product_type: ProductType = Field(alias="productType")
    security_id: str = Field(alias="securityId")
    price: float
    trigger_price: float = Field(default=0, alias="triggerPrice")

    model_config = {"populate_by_name": True}


class MarginDetails(BaseModel):
    """Margin calculator response model."""

    total_margin: float = Field(alias="totalMargin")
    span_margin: float = Field(alias="spanMargin")
    exposure_margin: float = Field(alias="exposureMargin")
    available_balance: float = Field(alias="availableBalance")
    variable_margin: float = Field(alias="variableMargin")
    insufficient_balance: float = Field(alias="insufficientBalance")
    brokerage: float
    leverage: float

    model_config = {"populate_by_name": True}
