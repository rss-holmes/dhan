
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for interacting with the Dhan API (https://api.dhan.co), which provides algorithmic trading and market data services. The project uses Python 3.13 and uv as the package manager.

## Environment Setup

The project requires environment variables for authentication:
- `CLIENT_ID`: Dhan API client ID (live)
- `ACCESS_TOKEN`: Dhan API access token (live)
- `SANDBOX_CLIENT_ID`: Dhan API client ID (sandbox)
- `SANDBOX_ACCESS_TOKEN`: Dhan API access token (sandbox)

These should be stored in a `.env` file (already gitignored).

## Development Commands

### Package Management
- Install dependencies: `uv sync`
- Add a new dependency: `uv add <package-name>`
- Run Python scripts: `uv run python <script-path>`

### Running Code
- Execute the example script: `uv run python dhan.py`

### Type Checking
- Run type checker: `uv run pyright core/`

## Code Architecture

### Project Structure
```
dhan/
├── core/                    # Main library code
│   ├── __init__.py          # Package exports
│   ├── client.py            # DhanClient - main entry point
│   ├── config.py            # DhanContext, Environment enum
│   ├── exceptions.py        # Custom exceptions
│   ├── models/              # Pydantic data models
│   │   ├── common.py        # Shared enums (ExchangeSegment, OrderType, etc.)
│   │   ├── orders.py        # Order, OrderRequest, Trade models
│   │   ├── market.py        # LTPData, OHLCData, MarketQuote models
│   │   ├── historical.py    # HistoricalData, DailyHistoricalRequest models
│   │   ├── portfolio.py     # Holding, Position models
│   │   └── funds.py         # FundLimit, MarginDetails models
│   └── services/            # API service implementations
│       ├── base.py          # BaseService with HTTP handling
│       ├── orders.py        # OrdersService
│       ├── market.py        # MarketService
│       ├── historical.py    # HistoricalService
│       ├── portfolio.py     # PortfolioService
│       └── funds.py         # FundsService
├── dhan.py                  # Example usage script
└── pyproject.toml
```

### Core Components

#### DhanClient (`core/client.py`)
Main client class with lazy-loaded service properties:
- `client.orders` - Order management (place, modify, cancel, get orders/trades)
- `client.market` - Market data (LTP, OHLC, full market depth)
- `client.historical` - Historical data (daily, intraday OHLC)
- `client.portfolio` - Portfolio (holdings, positions)
- `client.funds` - Funds (fund limit, margin calculator)

#### DhanContext (`core/config.py`)
Immutable configuration holding environment, client_id, and access_token. Provides `base_url` property based on environment.

#### Environment Support
| Environment | Base URL | Purpose |
|-------------|----------|---------|
| `Environment.LIVE` | `api.dhan.co` | Production trading |
| `Environment.SANDBOX` | `sandbox.dhan.co` | Testing (orders fill at 100, capital resets daily) |

### Usage Pattern
```python
from core import DhanClient, DhanContext, Environment, ExchangeSegment

# Create context
ctx = DhanContext(
    environment=Environment.LIVE,
    client_id="your_client_id",
    access_token="your_access_token",
)

# Create client
client = DhanClient(ctx)

# Use services
orders = client.orders.get_orders()
ltp = client.market.get_ltp({ExchangeSegment.NSE_EQ: ["1333"]})

# Switch context at runtime
sandbox_ctx = DhanContext(environment=Environment.SANDBOX, ...)
client.switch_context(sandbox_ctx)
```

### API Services

#### OrdersService
- `place_order(request)` - Place new order
- `modify_order(order_id, ...)` - Modify pending order
- `cancel_order(order_id)` - Cancel pending order
- `get_orders()` - Get all orders for the day
- `get_order(order_id)` - Get order by ID
- `get_trades()` - Get all trades for the day

#### MarketService
- `get_ltp(instruments)` - Last traded price (up to 1000 instruments)
- `get_ohlc(instruments)` - OHLC with LTP
- `get_quote(instruments)` - Full market depth

#### HistoricalService
- `get_daily(request)` - Daily OHLC data
- `get_intraday(request)` - Intraday OHLC (1, 5, 15, 25, 60 min intervals)

#### PortfolioService
- `get_holdings()` - All holdings
- `get_positions()` - All open positions
- `convert_position(request)` - Convert position product type

#### FundsService
- `get_fund_limit()` - Account balance and limits
- `calculate_margin(request)` - Margin requirements for an order

### Key Enums (`core/models/common.py`)
- `ExchangeSegment`: NSE_EQ, NSE_FNO, BSE_EQ, MCX_COMM, etc.
- `TransactionType`: BUY, SELL
- `ProductType`: CNC, INTRADAY, MARGIN, MTF, CO, BO
- `OrderType`: LIMIT, MARKET, STOP_LOSS, STOP_LOSS_MARKET
- `Validity`: DAY, IOC
- `OrderStatus`: TRANSIT, PENDING, REJECTED, CANCELLED, TRADED, EXPIRED

### Error Handling
Custom exceptions in `core/exceptions.py`:
- `DhanAPIError` - Base exception for API errors
- `DhanAuthenticationError` - 401/403 errors
- `DhanValidationError` - 400 errors
- `DhanRateLimitError` - 429 errors

## Dependencies
- `pydantic>=2.0` - Data validation and serialization
- `python-dotenv` - Environment variable loading
- `pyright` (dev) - Type checking
