"""Main Dhan API client."""

from __future__ import annotations

from core.config import DhanContext
from core.services.funds import FundsService
from core.services.historical import HistoricalService
from core.services.market import MarketService
from core.services.orders import OrdersService
from core.services.portfolio import PortfolioService


class DhanClient:
    """Main client for interacting with the Dhan API.

    The client supports both live and sandbox environments, and allows
    switching between contexts at runtime.

    Example:
        ```python
        from core import DhanClient, DhanContext, Environment

        # Create a sandbox context for testing
        sandbox_ctx = DhanContext(
            environment=Environment.SANDBOX,
            client_id="your_sandbox_client_id",
            access_token="your_sandbox_token",
        )

        # Initialize client with sandbox context
        client = DhanClient(sandbox_ctx)

        # Use the client
        orders = client.orders.get_orders()
        holdings = client.portfolio.get_holdings()

        # Switch to live when ready
        live_ctx = DhanContext(
            environment=Environment.LIVE,
            client_id="your_live_client_id",
            access_token="your_live_token",
        )
        client.switch_context(live_ctx)
        ```
    """

    def __init__(self, context: DhanContext) -> None:
        """Initialize the Dhan client.

        Args:
            context: The context configuration with credentials and environment.
        """
        self._context = context

        # Lazy-loaded service instances
        self._orders: OrdersService | None = None
        self._market: MarketService | None = None
        self._historical: HistoricalService | None = None
        self._portfolio: PortfolioService | None = None
        self._funds: FundsService | None = None

    @property
    def context(self) -> DhanContext:
        """Get the current context."""
        return self._context

    @property
    def orders(self) -> OrdersService:
        """Order management service.

        Provides methods for:
        - Placing orders (place_order, place_slice_order)
        - Modifying orders (modify_order)
        - Cancelling orders (cancel_order)
        - Fetching orders (get_orders, get_order, get_order_by_correlation_id)
        - Fetching trades (get_trades, get_trades_by_order)
        """
        if self._orders is None:
            self._orders = OrdersService(self)
        return self._orders

    @property
    def market(self) -> MarketService:
        """Market data service.

        Provides methods for:
        - Last traded price (get_ltp)
        - OHLC data (get_ohlc)
        - Full market quote with depth (get_quote)
        """
        if self._market is None:
            self._market = MarketService(self)
        return self._market

    @property
    def historical(self) -> HistoricalService:
        """Historical data service.

        Provides methods for:
        - Daily OHLC data (get_daily)
        - Intraday OHLC data (get_intraday)
        """
        if self._historical is None:
            self._historical = HistoricalService(self)
        return self._historical

    @property
    def portfolio(self) -> PortfolioService:
        """Portfolio service.

        Provides methods for:
        - Holdings (get_holdings)
        - Positions (get_positions)
        - Position conversion (convert_position)
        """
        if self._portfolio is None:
            self._portfolio = PortfolioService(self)
        return self._portfolio

    @property
    def funds(self) -> FundsService:
        """Funds and margin service.

        Provides methods for:
        - Fund limit (get_fund_limit)
        - Margin calculator (calculate_margin)
        """
        if self._funds is None:
            self._funds = FundsService(self)
        return self._funds
