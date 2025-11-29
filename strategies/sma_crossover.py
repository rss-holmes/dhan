"""Simple Moving Average Crossover Strategy."""

from algo.broker.models import OrderType
from algo.data.models import Candle
from algo.strategy.base import BaseStrategy, Signal, SignalAction


class SmaCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Buys when fast SMA crosses above slow SMA.
    Sells when fast SMA crosses below slow SMA.

    Parameters:
        fast_period: Period for fast SMA (default: 10)
        slow_period: Period for slow SMA (default: 20)
        quantity: Order quantity (default: 1)
        instruments: List of instruments to trade
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        quantity: int = 1,
        instruments: list[str] | None = None,
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            quantity=quantity,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity
        self._instruments = instruments or []
        self._prev_fast: dict[str, float | None] = {}
        self._prev_slow: dict[str, float | None] = {}

    @property
    def instruments(self) -> list[str]:
        """List of instruments this strategy trades."""
        return self._instruments

    def on_init(self) -> None:
        """Initialize state for each instrument."""
        for instrument in self._instruments:
            self._prev_fast[instrument] = None
            self._prev_slow[instrument] = None

    def on_candle(self, candle: Candle) -> Signal | None:
        """Process candle and generate signals."""
        instrument = candle.instrument

        # Calculate current SMAs
        fast_sma = self.context.sma(instrument, self.fast_period)
        slow_sma = self.context.sma(instrument, self.slow_period)

        # Skip if we don't have enough data
        if fast_sma != fast_sma or slow_sma != slow_sma:  # NaN check
            return None

        signal = None
        prev_fast = self._prev_fast.get(instrument)
        prev_slow = self._prev_slow.get(instrument)

        # Check for crossover
        if prev_fast is not None and prev_slow is not None:
            # Bullish crossover (fast crosses above slow)
            if prev_fast <= prev_slow and fast_sma > slow_sma:
                position = self.context.get_position(instrument)
                if position is None or position.quantity <= 0:
                    signal = Signal(
                        instrument=instrument,
                        action=SignalAction.BUY,
                        quantity=self.quantity,
                        order_type=OrderType.MARKET,
                    )

            # Bearish crossover (fast crosses below slow)
            elif prev_fast >= prev_slow and fast_sma < slow_sma:
                position = self.context.get_position(instrument)
                if position is not None and position.quantity > 0:
                    signal = Signal(
                        instrument=instrument,
                        action=SignalAction.SELL,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET,
                    )

        # Store for next comparison
        self._prev_fast[instrument] = fast_sma
        self._prev_slow[instrument] = slow_sma

        return signal

    def __repr__(self) -> str:
        return f"SmaCrossover(fast={self.fast_period}, slow={self.slow_period})"
