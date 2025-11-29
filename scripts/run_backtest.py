#!/usr/bin/env python3
"""Run a backtest with SMA Crossover strategy."""

import sys
from pathlib import Path

from algo.analytics.charts import BacktestCharts
from algo.data.models import Interval
from algo.engine.backtest import BacktestEngine
from algo.engine.fill_models import SlippageFillModel
from algo.persistence.csv_writer import CSVWriter
from strategies.sma_crossover import SmaCrossover

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run backtest example."""
    # Configuration
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # Check for data files
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please create CSV files in the data/ directory.")
        print(
            "Expected format: NSE_EQ_1333.csv with columns: timestamp,open,high,low,close,volume"
        )
        return 1

    # Find available data files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        print("Please add historical data files.")
        return 1

    # Extract instruments from filenames
    instruments = []
    for csv_file in csv_files:
        # Convert filename back to instrument format
        # e.g., "NSE_EQ_1333.csv" -> "NSE_EQ:1333"
        name = csv_file.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            instrument = f"{parts[0]}:{parts[1]}"
            instruments.append(instrument)

    print(f"Found {len(instruments)} instruments: {instruments}")

    # Create engine
    engine = BacktestEngine(
        data_dir=data_dir,
        initial_capital=100000.0,
        fill_model=SlippageFillModel(slippage_pct=0.001),
        commission_rate=0.0001,  # 0.01%
        interval=Interval.DAY_1,
    )

    # Create and add strategy
    strategy = SmaCrossover(
        fast_period=10,
        slow_period=20,
        quantity=100,
        instruments=instruments,
    )
    engine.add_strategy(strategy)

    print(f"\nRunning backtest with {strategy}...")
    print("Initial capital: $100,000")
    print("-" * 60)

    # Run backtest
    result = engine.run()

    # Print summary
    print(result.summary())

    # Save results to CSV
    writer = CSVWriter(output_dir)
    writer.write_all(result, prefix="backtest")

    # Generate charts (if plotly available)
    try:
        charts = BacktestCharts(result)
        charts.save_html(str(output_dir / "backtest_report.html"))
    except ImportError:
        print("\nNote: Install plotly for interactive charts: pip install plotly")

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
