"""Portfolio management for tracking positions and cash."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from algo.broker.models import Fill, OrderSide, Position


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    cash: float
    positions_value: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)


class Portfolio:
    """
    Portfolio state management.

    Tracks cash, positions, and calculates equity.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash balance
        """
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._realized_pnl = 0.0
        self._snapshots: list[PortfolioSnapshot] = []

    def update_position(
        self,
        instrument: str,
        quantity: int,
        price: float,
    ) -> None:
        """
        Update or create a position.

        Args:
            instrument: Instrument identifier
            quantity: Position quantity (positive for long)
            price: Average price
        """
        self._positions[instrument] = Position(
            instrument=instrument,
            quantity=quantity,
            average_price=price,
            current_price=price,
        )

    def reduce_position(
        self,
        instrument: str,
        quantity: int,
        price: float,
    ) -> float:
        """
        Reduce a position and calculate realized P&L.

        Args:
            instrument: Instrument identifier
            quantity: Quantity to reduce
            price: Sale price

        Returns:
            Realized P&L from the reduction
        """
        if instrument not in self._positions:
            return 0.0

        position = self._positions[instrument]
        realized_pnl = 0.0

        if position.quantity > 0:
            # Long position - selling
            close_qty = min(quantity, position.quantity)
            realized_pnl = (price - position.average_price) * close_qty
            new_qty = position.quantity - quantity
        else:
            # Short position - buying to cover
            close_qty = min(quantity, abs(position.quantity))
            realized_pnl = (position.average_price - price) * close_qty
            new_qty = position.quantity + quantity

        self._realized_pnl += realized_pnl

        if new_qty == 0:
            del self._positions[instrument]
        else:
            self._positions[instrument] = Position(
                instrument=instrument,
                quantity=new_qty,
                average_price=position.average_price,
                current_price=price,
                realized_pnl=position.realized_pnl + realized_pnl,
            )

        return realized_pnl

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices for all positions.

        Args:
            prices: Dict of instrument -> price
        """
        for instrument, price in prices.items():
            if instrument in self._positions:
                pos = self._positions[instrument]
                unrealized = (price - pos.average_price) * pos.quantity
                self._positions[instrument] = Position(
                    instrument=pos.instrument,
                    quantity=pos.quantity,
                    average_price=pos.average_price,
                    current_price=price,
                    unrealized_pnl=unrealized,
                    realized_pnl=pos.realized_pnl,
                )

    def apply_fill(self, fill: Fill) -> None:
        """
        Apply a fill to the portfolio.

        Args:
            fill: Order fill to apply
        """
        instrument = fill.instrument
        position = self._positions.get(instrument)

        if position is None:
            # New position
            qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            self._positions[instrument] = Position(
                instrument=instrument,
                quantity=qty,
                average_price=fill.price,
                current_price=fill.price,
            )
            trade_value = fill.price * fill.quantity
            if fill.side == OrderSide.BUY:
                self._cash -= trade_value + fill.commission
            else:
                self._cash += trade_value - fill.commission
        else:
            # Update existing position
            if fill.side == OrderSide.BUY:
                if position.quantity >= 0:
                    # Adding to long
                    total_cost = (
                        position.average_price * position.quantity
                        + fill.price * fill.quantity
                    )
                    new_qty = position.quantity + fill.quantity
                    new_avg = total_cost / new_qty
                    self._positions[instrument] = Position(
                        instrument=instrument,
                        quantity=new_qty,
                        average_price=new_avg,
                        current_price=fill.price,
                        realized_pnl=position.realized_pnl,
                    )
                else:
                    # Closing short
                    self.reduce_position(instrument, fill.quantity, fill.price)
                self._cash -= fill.price * fill.quantity + fill.commission
            else:
                if position.quantity <= 0:
                    # Adding to short
                    total_cost = (
                        abs(position.average_price * position.quantity)
                        + fill.price * fill.quantity
                    )
                    new_qty = position.quantity - fill.quantity
                    new_avg = total_cost / abs(new_qty)
                    self._positions[instrument] = Position(
                        instrument=instrument,
                        quantity=new_qty,
                        average_price=new_avg,
                        current_price=fill.price,
                        realized_pnl=position.realized_pnl,
                    )
                else:
                    # Closing long
                    self.reduce_position(instrument, fill.quantity, fill.price)
                self._cash += fill.price * fill.quantity - fill.commission

    def take_snapshot(self, timestamp: datetime | None = None) -> PortfolioSnapshot:
        """
        Take a snapshot of current portfolio state.

        Args:
            timestamp: Snapshot timestamp (default: now)

        Returns:
            Portfolio snapshot
        """
        if timestamp is None:
            timestamp = datetime.now()

        positions_value = sum(p.market_value for p in self._positions.values())
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions_value=positions_value,
            equity=self._cash + positions_value,
            positions=self._positions.copy(),
        )
        self._snapshots.append(snapshot)
        return snapshot

    def get_position(self, instrument: str) -> Position | None:
        """Get position for an instrument."""
        return self._positions.get(instrument)

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash

    @property
    def positions_value(self) -> float:
        """Get total value of all positions."""
        return sum(p.market_value for p in self._positions.values())

    @property
    def equity(self) -> float:
        """Get total portfolio equity."""
        return self._cash + self.positions_value

    @property
    def positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self._realized_pnl + self.unrealized_pnl

    @property
    def initial_capital(self) -> float:
        """Get initial capital."""
        return self._initial_capital

    @property
    def snapshots(self) -> list[PortfolioSnapshot]:
        """Get all snapshots."""
        return self._snapshots.copy()

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._cash = self._initial_capital
        self._positions.clear()
        self._realized_pnl = 0.0
        self._snapshots.clear()
