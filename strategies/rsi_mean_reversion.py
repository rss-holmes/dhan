"""RSI Mean Reversion Strategy."""

from algo.broker.models import OrderType
from algo.data.models import Candle
from algo.strategy.base import BaseStrategy, Signal, SignalAction


class RsiMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Buys when RSI is oversold (below lower threshold).
    Sells when RSI is overbought (above upper threshold).

    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
        quantity: Order quantity (default: 1)
        instruments: List of instruments to trade
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        quantity: int = 1,
        instruments: list[str] | None = None,
    ):
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            quantity=quantity,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.quantity = quantity
        self._instruments = instruments or []

    @property
    def instruments(self) -> list[str]:
        """List of instruments this strategy trades."""
        return self._instruments

    def on_candle(self, candle: Candle) -> Signal | None:
        """Process candle and generate signals."""
        instrument = candle.instrument

        # Calculate RSI
        rsi = self.context.rsi(instrument, self.rsi_period)

        # Skip if we don't have enough data
        if rsi != rsi:  # NaN check
            return None

        position = self.context.get_position(instrument)

        # Oversold - buy signal
        if rsi < self.oversold:
            if position is None or position.quantity <= 0:
                return Signal(
                    instrument=instrument,
                    action=SignalAction.BUY,
                    quantity=self.quantity,
                    order_type=OrderType.MARKET,
                    metadata={"rsi": rsi},
                )

        # Overbought - sell signal
        elif rsi > self.overbought:
            if position is not None and position.quantity > 0:
                return Signal(
                    instrument=instrument,
                    action=SignalAction.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    metadata={"rsi": rsi},
                )

        return None

    def __repr__(self) -> str:
        return (
            f"RsiMeanReversion(period={self.rsi_period}, "
            f"oversold={self.oversold}, overbought={self.overbought})"
        )
