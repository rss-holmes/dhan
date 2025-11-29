import csv
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

from core import (
    IntradayHistoricalRequest,
    DhanClient,
    DhanContext,
    Environment,
    ExchangeSegment,
    Instrument,
    IntradayInterval,
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

    # Use dates within last 90 days (API limit)
    # Intraday requires datetime format: YYYY-MM-DD HH:MM:SS
    to_date = datetime.now()
    from_date = to_date - timedelta(days=5)

    request = IntradayHistoricalRequest(
        securityId="1333",
        exchangeSegment=ExchangeSegment.NSE_EQ,
        instrument=Instrument.EQUITY,
        interval=IntradayInterval.FIFTEEN_MINUTES,
        fromDate=from_date.strftime("%Y-%m-%d"),
        toDate=to_date.strftime("%Y-%m-%d"),
        oi=False,
    )

    print("Fetching intraday historical data...")
    data = client.historical.get_intraday(request)

    print(f"Received {len(data.timestamp)} data points")
    if data.timestamp:
        print(f"Date range: {data.timestamp[0]} to {data.timestamp[-1]}")
        print("Sample OHLC (first candle):")
        print(f"  Open:   {data.open[0]}")
        print(f"  High:   {data.high[0]}")
        print(f"  Low:    {data.low[0]}")
        print(f"  Close:  {data.close[0]}")
        print(f"  Volume: {data.volume[0]}")

        # Save to CSV
        filename = f"intraday_{request.security_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for i in range(len(data.timestamp)):
                writer.writerow([
                    data.timestamp[i],
                    data.open[i],
                    data.high[i],
                    data.low[i],
                    data.close[i],
                    data.volume[i],
                ])
        print(f"\nSaved to {filename}")


if __name__ == "__main__":
    main()
