"""Strategy context providing access to data, indicators, and order submission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from algo.broker.models import Position
    from algo.data.models import Candle
    from algo.engine.base import BaseEngine
    from algo.strategy.base import Signal


class StrategyContext:
    """
    Provides strategies with access to market data, indicators,
    portfolio state, and order submission.
    """

    def __init__(self, engine: BaseEngine, strategy_id: str):
        """
        Initialize strategy context.

        Args:
            engine: The execution engine
            strategy_id: Unique identifier for this strategy instance
        """
        self._engine = engine
        self._strategy_id = strategy_id

    # === Data Access ===

    def get_historical(
        self,
        instrument: str,
        periods: int,
        field: str = "close",
    ) -> np.ndarray:
        """
        Get historical data for indicator calculation.

        Args:
            instrument: Instrument identifier
            periods: Number of periods to retrieve
            field: Field to extract (open, high, low, close, volume)

        Returns:
            NumPy array of values
        """
        values = self._engine.get_historical(instrument, periods, field)
        return np.array(values, dtype=np.float64)

    def get_candle(self, instrument: str, offset: int = 0) -> Candle | None:
        """
        Get a specific historical candle.

        Args:
            instrument: Instrument identifier
            offset: Offset from current (0 = current, -1 = previous)

        Returns:
            Candle or None if not available
        """
        return self._engine.get_candle(instrument, offset)

    # === TA-Lib Indicators ===

    def sma(self, instrument: str, period: int) -> float:
        """
        Simple Moving Average of close prices.

        Args:
            instrument: Instrument identifier
            period: SMA period

        Returns:
            Current SMA value
        """
        try:
            import talib  # type: ignore[import-not-found]

            closes = self.get_historical(instrument, period + 1)
            if len(closes) < period:
                return float("nan")
            result = talib.SMA(closes, timeperiod=period)
            return float(result[-1])
        except ImportError:
            # Fallback to numpy implementation
            closes = self.get_historical(instrument, period)
            if len(closes) < period:
                return float("nan")
            return float(np.mean(closes[-period:]))

    def ema(self, instrument: str, period: int) -> float:
        """
        Exponential Moving Average of close prices.

        Args:
            instrument: Instrument identifier
            period: EMA period

        Returns:
            Current EMA value
        """
        try:
            import talib  # type: ignore[import-not-found]

            closes = self.get_historical(instrument, period * 2)
            if len(closes) < period:
                return float("nan")
            result = talib.EMA(closes, timeperiod=period)
            return float(result[-1])
        except ImportError:
            # Fallback to pandas implementation
            closes = self.get_historical(instrument, period * 2)
            if len(closes) < period:
                return float("nan")
            alpha = 2 / (period + 1)
            ema = closes[0]
            for price in closes[1:]:
                ema = alpha * price + (1 - alpha) * ema
            return float(ema)

    def rsi(self, instrument: str, period: int = 14) -> float:
        """
        Relative Strength Index.

        Args:
            instrument: Instrument identifier
            period: RSI period (default 14)

        Returns:
            Current RSI value (0-100)
        """
        try:
            import talib  # type: ignore[import-not-found]

            closes = self.get_historical(instrument, period + 10)
            if len(closes) < period + 1:
                return float("nan")
            result = talib.RSI(closes, timeperiod=period)
            return float(result[-1])
        except ImportError:
            # Fallback implementation
            closes = self.get_historical(instrument, period + 10)
            if len(closes) < period + 1:
                return float("nan")
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return float(100 - (100 / (1 + rs)))

    def macd(
        self,
        instrument: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        """
        MACD indicator.

        Args:
            instrument: Instrument identifier
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Tuple of (macd, signal, histogram)
        """
        try:
            import talib  # type: ignore[import-not-found]

            closes = self.get_historical(instrument, slow + signal + 10)
            if len(closes) < slow:
                return float("nan"), float("nan"), float("nan")
            macd_line, signal_line, hist = talib.MACD(
                closes, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return float(macd_line[-1]), float(signal_line[-1]), float(hist[-1])
        except ImportError:
            # Simplified fallback
            fast_ema = self.ema(instrument, fast)
            slow_ema = self.ema(instrument, slow)
            macd_line = fast_ema - slow_ema
            return macd_line, float("nan"), float("nan")

    def bollinger_bands(
        self,
        instrument: str,
        period: int = 20,
        std: float = 2.0,
    ) -> tuple[float, float, float]:
        """
        Bollinger Bands.

        Args:
            instrument: Instrument identifier
            period: Moving average period (default 20)
            std: Standard deviation multiplier (default 2.0)

        Returns:
            Tuple of (upper, middle, lower) band values
        """
        try:
            import talib  # type: ignore[import-not-found]

            closes = self.get_historical(instrument, period + 5)
            if len(closes) < period:
                return float("nan"), float("nan"), float("nan")
            upper, middle, lower = talib.BBANDS(
                closes, timeperiod=period, nbdevup=std, nbdevdn=std
            )
            return float(upper[-1]), float(middle[-1]), float(lower[-1])
        except ImportError:
            # Fallback implementation
            closes = self.get_historical(instrument, period)
            if len(closes) < period:
                return float("nan"), float("nan"), float("nan")
            middle = float(np.mean(closes))
            std_dev = float(np.std(closes))
            upper = middle + std * std_dev
            lower = middle - std * std_dev
            return upper, middle, lower

    def atr(self, instrument: str, period: int = 14) -> float:
        """
        Average True Range.

        Args:
            instrument: Instrument identifier
            period: ATR period (default 14)

        Returns:
            Current ATR value
        """
        try:
            import talib  # type: ignore[import-not-found]

            high = self.get_historical(instrument, period + 5, "high")
            low = self.get_historical(instrument, period + 5, "low")
            close = self.get_historical(instrument, period + 5, "close")
            if len(high) < period + 1:
                return float("nan")
            result = talib.ATR(high, low, close, timeperiod=period)
            return float(result[-1])
        except ImportError:
            # Fallback implementation
            high = self.get_historical(instrument, period + 1, "high")
            low = self.get_historical(instrument, period + 1, "low")
            close = self.get_historical(instrument, period + 1, "close")
            if len(high) < period + 1:
                return float("nan")
            tr_list = []
            for i in range(1, len(high)):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )
                tr_list.append(tr)
            return float(np.mean(tr_list[-period:]))

    # === Portfolio State ===

    def get_position(self, instrument: str) -> Position | None:
        """
        Get current position for instrument.

        Args:
            instrument: Instrument identifier

        Returns:
            Position or None if no position
        """
        return self._engine.get_position(instrument)

    def get_balance(self) -> float:
        """Get available cash balance."""
        return self._engine.get_balance()

    def get_equity(self) -> float:
        """Get total portfolio equity (cash + positions value)."""
        return self._engine.get_equity()

    # === Order Submission ===

    def submit_order(self, signal: Signal) -> str:
        """
        Submit an order based on a signal.

        Args:
            signal: Trading signal

        Returns:
            Order ID for tracking
        """
        return self._engine.submit_order(self._strategy_id, signal)

    @property
    def strategy_id(self) -> str:
        """Get strategy ID."""
        return self._strategy_id
