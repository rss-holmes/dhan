#!/usr/bin/env python3
"""Run live trading with real orders."""

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
    """Run live trading example."""
    from core import DhanContext, Environment

    from algo.broker.providers.dhan_broker import DhanBroker
    from algo.data.providers.dhan_feed import DhanDataFeed
    from algo.engine.live import LiveTradingEngine
    from strategies.sma_crossover import SmaCrossover

    # Get credentials from environment
    client_id = os.getenv("CLIENT_ID")
    access_token = os.getenv("ACCESS_TOKEN")

    if not client_id or not access_token:
        print("Error: Missing credentials. Set CLIENT_ID and ACCESS_TOKEN in .env")
        return 1

    # Confirm live trading
    print("=" * 60)
    print("  WARNING: LIVE TRADING MODE")
    print("  This will place REAL orders with REAL money!")
    print("=" * 60)

    confirm = input("\nType 'CONFIRM' to proceed: ")
    if confirm != "CONFIRM":
        print("Aborted.")
        return 1

    # Instruments to trade
    instruments = ["NSE_EQ:1333"]  # HDFC Bank

    # Create context
    ctx = DhanContext(
        environment=Environment.LIVE,
        client_id=client_id,
        access_token=access_token,
    )

    # Create data feed and broker
    data_feed = DhanDataFeed(
        client_id=client_id,
        access_token=access_token,
    )
    broker = DhanBroker(ctx)

    # Create engine
    engine = LiveTradingEngine(
        data_feed=data_feed,
        broker=broker,
    )

    # Add strategy (use small quantity for safety)
    strategy = SmaCrossover(
        fast_period=10,
        slow_period=20,
        quantity=1,  # Small quantity for live testing
        instruments=instruments,
    )
    engine.add_strategy(strategy)

    # Register error handler
    def on_error(strategy_id: str, error: Exception):
        print(f"[ERROR] Strategy {strategy_id}: {error}")

    engine.on_error(on_error)

    # Start live trading
    print(f"\nStarting LIVE trading with {strategy}...")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    await engine.start()

    try:
        # Run until interrupted
        while engine.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping live trading...")

    # Stop
    await engine.stop()

    print("Live trading stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
