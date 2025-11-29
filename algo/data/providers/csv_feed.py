"""CSV file data feed for backtesting."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterator

from algo.data.models import Candle, Interval


class CSVDataFeed:
    """
    Historical data feed from CSV files for backtesting.

    Expected CSV format:
    timestamp,open,high,low,close,volume[,oi]

    timestamp format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    """

    def __init__(
        self,
        data_dir: Path | str,
        interval: Interval = Interval.DAY_1,
    ):
        self._data_dir = Path(data_dir)
        self._interval = interval
        self._instruments: list[str] = []
        self._data: dict[str, list[Candle]] = {}
        self._current_idx: int = 0
        self._merged_timeline: list[tuple[datetime, str, Candle]] = []

    def set_instruments(self, instruments: list[str]) -> None:
        """Load data for specified instruments."""
        self._instruments = instruments
        self._data.clear()

        for instrument in instruments:
            self._load_instrument_data(instrument)

        self._build_merged_timeline()

    def _load_instrument_data(self, instrument: str) -> None:
        """Load CSV data for a single instrument."""
        # Convert instrument ID to filename
        # e.g., "NSE_EQ:1333" -> "NSE_EQ_1333.csv"
        filename = instrument.replace(":", "_") + ".csv"
        filepath = self._data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        candles = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candle = Candle(
                    instrument=instrument,
                    timestamp=self._parse_timestamp(row["timestamp"]),
                    interval=self._interval,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(float(row["volume"])),
                    oi=int(float(row.get("oi", 0))) if row.get("oi") else None,
                )
                candles.append(candle)

        # Sort by timestamp
        candles.sort(key=lambda c: c.timestamp)
        self._data[instrument] = candles

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse timestamp from various formats."""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d-%m-%Y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {ts}")

    def _build_merged_timeline(self) -> None:
        """Build chronological timeline of all candles."""
        self._merged_timeline = []

        for instrument, candles in self._data.items():
            for candle in candles:
                self._merged_timeline.append((candle.timestamp, instrument, candle))

        # Sort by timestamp
        self._merged_timeline.sort(key=lambda x: x[0])
        self._current_idx = 0

    def __iter__(self) -> Iterator[Candle]:
        """Iterate through candles in chronological order."""
        self._current_idx = 0
        for _, _, candle in self._merged_timeline:
            self._current_idx += 1
            yield candle

    def __len__(self) -> int:
        """Return total number of candles."""
        return len(self._merged_timeline)

    def get_candles(
        self,
        instrument: str,
        count: int,
        end_offset: int = 0,
    ) -> list[Candle]:
        """
        Get historical candles relative to current position.

        Args:
            instrument: Instrument identifier
            count: Number of candles to retrieve
            end_offset: Offset from current position (0 = current candle)
        """
        if instrument not in self._data:
            return []

        all_candles = self._data[instrument]

        # Find current position in instrument's timeline
        if self._current_idx <= 0:
            return []

        current_ts = self._merged_timeline[self._current_idx - 1][0]

        # Find candles up to current timestamp
        valid_candles = [c for c in all_candles if c.timestamp <= current_ts]

        if not valid_candles:
            return []

        end_idx = len(valid_candles) - end_offset
        start_idx = max(0, end_idx - count)

        return valid_candles[start_idx:end_idx]

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_idx = 0

    @property
    def instruments(self) -> list[str]:
        """Get list of loaded instruments."""
        return self._instruments

    @property
    def current_index(self) -> int:
        """Get current position in timeline."""
        return self._current_idx
