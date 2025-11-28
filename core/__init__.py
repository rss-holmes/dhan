"""Dhan API client library.

This module provides a Python client for the Dhan trading API with support
for both live and sandbox environments.

Example:
    ```python
    from core import DhanClient, DhanContext, Environment, load_context_from_env

    # Load context from environment variables
    ctx = load_context_from_env(Environment.SANDBOX)

    # Or create context manually
    ctx = DhanContext(
        environment=Environment.SANDBOX,
        client_id="your_client_id",
        access_token="your_access_token",
    )

    # Create client
    client = DhanClient(ctx)

    # Use services
    orders = client.orders.get_orders()
    holdings = client.portfolio.get_holdings()
    ltp = client.market.get_ltp({ExchangeSegment.NSE_EQ: ["1333"]})
    ```
"""

from core.client import DhanClient
from core.config import DhanContext, Environment
from core.exceptions import (
    DhanAPIError,
    DhanAuthenticationError,
    DhanRateLimitError,
    DhanValidationError,
)
from core.models import (
    ConvertPositionRequest,
    DailyHistoricalRequest,
    ExchangeSegment,
    ExpiryCode,
    FundLimit,
    HistoricalData,
    Holding,
    Instrument,
    IntradayHistoricalRequest,
    IntradayInterval,
    LTPData,
    MarginCalculatorRequest,
    MarginDetails,
    MarketDepth,
    MarketQuote,
    OHLCData,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    PositionType,
    ProductType,
    SliceOrderRequest,
    Trade,
    TransactionType,
    Validity,
)

__all__ = [
    # Client
    "DhanClient",
    # Config
    "DhanContext",
    "Environment",
    # Exceptions
    "DhanAPIError",
    "DhanAuthenticationError",
    "DhanRateLimitError",
    "DhanValidationError",
    # Enums
    "ExchangeSegment",
    "ExpiryCode",
    "Instrument",
    "IntradayInterval",
    "OrderStatus",
    "OrderType",
    "PositionType",
    "ProductType",
    "TransactionType",
    "Validity",
    # Models
    "ConvertPositionRequest",
    "DailyHistoricalRequest",
    "FundLimit",
    "HistoricalData",
    "Holding",
    "IntradayHistoricalRequest",
    "LTPData",
    "MarginCalculatorRequest",
    "MarginDetails",
    "MarketDepth",
    "MarketQuote",
    "OHLCData",
    "Order",
    "OrderRequest",
    "Position",
    "SliceOrderRequest",
    "Trade",
]
