"""Example usage of the Dhan API client.

Run with: PYTHONPATH=. uv run python dhan.py
"""

import os

from dotenv import load_dotenv

from core import (
    DailyHistoricalRequest,
    DhanClient,
    DhanContext,
    Environment,
    ExchangeSegment,
    ExpiryCode,
    Instrument,
)

load_dotenv()


def main() -> None:
    ctx = DhanContext(
        environment=Environment.LIVE,
        client_id=os.getenv("CLIENT_ID", ""),
        access_token=os.getenv("ACCESS_TOKEN", ""),
    )

    client = DhanClient(ctx)

    print(f"Using environment: {ctx.environment.value}")
    print(f"Base URL: {ctx.base_url}")
    print()

    request = DailyHistoricalRequest(
        securityId="1333",
        exchangeSegment=ExchangeSegment.NSE_EQ,
        instrument=Instrument.INDEX,
        expiryCode=ExpiryCode.NOT_APPLICABLE,
        fromDate="2025-08-24",
        toDate="2025-10-24",
    )

    print("Fetching historical data...")
    data = client.historical.get_daily(request)

    print(f"Received {len(data.timestamp)} data points")
    if data.timestamp:
        print(f"Date range: {data.timestamp[0]} to {data.timestamp[-1]}")
        print("Sample OHLC (first candle):")
        print(f"  Open:   {data.open[0]}")
        print(f"  High:   {data.high[0]}")
        print(f"  Low:    {data.low[0]}")
        print(f"  Close:  {data.close[0]}")
        print(f"  Volume: {data.volume[0]}")


if __name__ == "__main__":
    main()
