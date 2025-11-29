"""CSV persistence utilities."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algo.analytics.metrics import BacktestResult
    from algo.broker.models import Fill, Order


class CSVWriter:
    """Write backtest data to CSV files."""

    def __init__(self, output_dir: Path | str):
        """
        Initialize CSV writer.

        Args:
            output_dir: Directory for output files
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_trades(
        self,
        trades: list[Order],
        filename: str = "trades.csv",
    ) -> Path:
        """
        Write trades to CSV.

        Args:
            trades: List of orders/trades
            filename: Output filename

        Returns:
            Path to written file
        """
        filepath = self._output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "order_id",
                    "timestamp",
                    "instrument",
                    "side",
                    "quantity",
                    "order_type",
                    "status",
                    "filled_qty",
                    "avg_price",
                    "limit_price",
                    "strategy_id",
                ]
            )

            for trade in trades:
                writer.writerow(
                    [
                        trade.order_id,
                        trade.created_at.isoformat(),
                        trade.instrument,
                        trade.side.value,
                        trade.quantity,
                        trade.order_type.value,
                        trade.status.value,
                        trade.filled_quantity,
                        trade.average_price,
                        trade.limit_price or "",
                        trade.strategy_id or "",
                    ]
                )

        print(f"Trades written to: {filepath}")
        return filepath

    def write_fills(
        self,
        fills: list[Fill],
        filename: str = "fills.csv",
    ) -> Path:
        """
        Write fills to CSV.

        Args:
            fills: List of fills
            filename: Output filename

        Returns:
            Path to written file
        """
        filepath = self._output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fill_id",
                    "order_id",
                    "timestamp",
                    "instrument",
                    "side",
                    "quantity",
                    "price",
                    "commission",
                    "strategy_id",
                ]
            )

            for fill in fills:
                writer.writerow(
                    [
                        fill.fill_id,
                        fill.order_id,
                        fill.timestamp.isoformat(),
                        fill.instrument,
                        fill.side.value,
                        fill.quantity,
                        fill.price,
                        fill.commission,
                        fill.strategy_id or "",
                    ]
                )

        print(f"Fills written to: {filepath}")
        return filepath

    def write_equity_curve(
        self,
        equity_curve: list[tuple[datetime, float]],
        filename: str = "equity_curve.csv",
    ) -> Path:
        """
        Write equity curve to CSV.

        Args:
            equity_curve: List of (timestamp, equity) tuples
            filename: Output filename

        Returns:
            Path to written file
        """
        filepath = self._output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity"])

            for dt, equity in equity_curve:
                writer.writerow([dt.isoformat(), equity])

        print(f"Equity curve written to: {filepath}")
        return filepath

    def write_results_summary(
        self,
        result: BacktestResult,
        filename: str = "results.csv",
    ) -> Path:
        """
        Write performance metrics to CSV.

        Args:
            result: Backtest result
            filename: Output filename

        Returns:
            Path to written file
        """
        filepath = self._output_dir / filename

        metrics = result.metrics

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])

            # Convert metrics to dict and write
            metrics_dict = metrics.model_dump()
            for key, value in metrics_dict.items():
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif hasattr(value, "total_seconds"):  # timedelta
                    value = f"{value.days} days"
                elif isinstance(value, float):
                    value = f"{value:.4f}"
                writer.writerow([key, value])

        print(f"Results summary written to: {filepath}")
        return filepath

    def write_all(
        self,
        result: BacktestResult,
        prefix: str = "",
    ) -> dict[str, Path]:
        """
        Write all data to CSV files.

        Args:
            result: Backtest result
            prefix: Optional prefix for filenames

        Returns:
            Dict of filename -> path
        """
        paths = {}

        prefix_str = f"{prefix}_" if prefix else ""

        paths["trades"] = self.write_trades(
            result.trades, f"{prefix_str}trades.csv"
        )

        if result.fills:
            paths["fills"] = self.write_fills(
                result.fills, f"{prefix_str}fills.csv"
            )

        paths["equity_curve"] = self.write_equity_curve(
            result.equity_curve, f"{prefix_str}equity_curve.csv"
        )

        paths["results"] = self.write_results_summary(
            result, f"{prefix_str}results.csv"
        )

        return paths
