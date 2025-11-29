#!/usr/bin/env python3
"""Run paper trading with live market data."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()


async def main():
    """Run paper trading example."""
    from algo.data.providers.dhan_feed import DhanDataFeed
    from algo.engine.paper import PaperTradingEngine
    from strategies.sma_crossover import SmaCrossover

    # Get credentials from environment
    client_id = os.getenv("CLIENT_ID")
    access_token = os.getenv("ACCESS_TOKEN")

    if not client_id or not access_token:
        print("Error: Missing credentials. Set CLIENT_ID and ACCESS_TOKEN in .env")
        return 1

    # Instruments to trade
    instruments = ["NSE_EQ:1333"]  # HDFC Bank

    # Create data feed
    data_feed = DhanDataFeed(
        client_id=client_id,
        access_token=access_token,
    )

    # Create engine
    engine = PaperTradingEngine(
        data_feed=data_feed,
        initial_capital=100000.0,
    )

    # Add strategy
    strategy = SmaCrossover(
        fast_period=10,
        slow_period=20,
        quantity=10,
        instruments=instruments,
    )
    engine.add_strategy(strategy)

    # Start paper trading
    print(f"Starting paper trading with {strategy}...")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    await engine.start()

    try:
        # Run until interrupted
        while engine.is_running:
            await asyncio.sleep(1)

            # Print periodic updates
            equity = engine.get_equity()
            print(f"Equity: ${equity:,.2f}", end="\r")

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")

    # Stop and get results
    result = await engine.stop()

    # Print summary
    print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
