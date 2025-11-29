"""Performance calculation utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from algo.analytics.metrics import PerformanceMetrics
from algo.broker.models import Fill, Order, OrderSide, OrderStatus


class PerformanceCalculator:
    """
    Calculate comprehensive performance metrics from backtest results.
    """

    def calculate(
        self,
        equity_curve: list[tuple[datetime, float]],
        trades: list[Order],
        fills: list[Fill],
        initial_capital: float,
        benchmark_returns: list[float] | None = None,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.

        Args:
            equity_curve: List of (timestamp, equity) tuples
            trades: List of orders
            fills: List of fills
            initial_capital: Starting capital
            benchmark_returns: Optional benchmark returns for alpha/beta
            risk_free_rate: Risk-free rate for Sharpe calculation

        Returns:
            PerformanceMetrics object
        """
        if not equity_curve:
            return self._empty_metrics(initial_capital)

        # Extract data
        timestamps = [t for t, _ in equity_curve]
        equity_values = np.array([e for _, e in equity_curve])

        start_date = timestamps[0]
        end_date = timestamps[-1]
        duration = end_date - start_date

        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Basic metrics
        final_equity = equity_values[-1]
        peak_equity = np.max(equity_values)
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

        # Annualized returns
        days = max(1, duration.days)
        years = days / 365.25
        if years > 0:
            annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
            cagr = annualized_return
        else:
            annualized_return = total_return_pct
            cagr = total_return_pct

        # Volatility
        if len(returns) > 1:
            daily_vol = np.std(returns)
            volatility_ann = daily_vol * np.sqrt(252) * 100
        else:
            volatility_ann = 0.0

        # Drawdown analysis
        drawdowns = self._calculate_drawdowns(equity_values)
        max_drawdown = float(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0.0
        avg_drawdown = float(np.mean(drawdowns[drawdowns < 0]) * 100) if np.any(drawdowns < 0) else 0.0

        max_dd_duration, avg_dd_duration = self._calculate_drawdown_durations(
            drawdowns, timestamps
        )

        # Risk-adjusted metrics
        if volatility_ann > 0:
            sharpe_ratio = (annualized_return - risk_free_rate * 100) / volatility_ann
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_dev = np.std(negative_returns) * np.sqrt(252)
            sortino_ratio = (annualized_return / 100 - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
        else:
            sortino_ratio = 0.0

        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Alpha and Beta (against benchmark or simple market proxy)
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            alpha_val, beta_val = self._calculate_alpha_beta(returns, benchmark_returns)
            alpha = float(alpha_val)
            beta = float(beta_val)
        else:
            alpha = total_return_pct
            beta = 0.0

        # Trade statistics
        completed_trades = [t for t in trades if t.status == OrderStatus.FILLED]
        trade_stats = self._calculate_trade_stats(completed_trades, fills)

        # Exposure time
        exposure_time = self._calculate_exposure_time(equity_values, initial_capital)

        # Buy and hold return (using first and last equity)
        buy_hold_return = total_return_pct

        return PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            duration=duration,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return,
            buy_and_hold_return_pct=buy_hold_return,
            cagr_pct=cagr,
            volatility_ann_pct=volatility_ann,
            max_drawdown_pct=max_drawdown,
            avg_drawdown_pct=avg_drawdown,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown_duration=avg_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            alpha_pct=alpha,
            beta=beta,
            initial_capital=initial_capital,
            final_equity=final_equity,
            peak_equity=peak_equity,
            exposure_time_pct=exposure_time,
            **trade_stats,
        )

    def _calculate_drawdowns(self, equity: np.ndarray) -> np.ndarray:
        """Calculate drawdown series."""
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        return drawdowns

    def _calculate_drawdown_durations(
        self,
        drawdowns: np.ndarray,
        timestamps: list[datetime],
    ) -> tuple[timedelta, timedelta]:
        """Calculate max and average drawdown durations."""
        if len(drawdowns) == 0 or not np.any(drawdowns < 0):
            return timedelta(0), timedelta(0)

        in_drawdown = drawdowns < 0
        durations = []
        start_idx = None

        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                duration = timestamps[i - 1] - timestamps[start_idx]
                durations.append(duration)
                start_idx = None

        # Handle ongoing drawdown
        if start_idx is not None:
            duration = timestamps[-1] - timestamps[start_idx]
            durations.append(duration)

        if not durations:
            return timedelta(0), timedelta(0)

        max_duration = max(durations)
        avg_duration = sum(durations, timedelta(0)) / len(durations)

        return max_duration, avg_duration

    def _calculate_alpha_beta(
        self,
        returns: np.ndarray,
        benchmark_returns: list[float],
    ) -> tuple[float, float]:
        """Calculate alpha and beta against benchmark."""
        bench = np.array(benchmark_returns[: len(returns)])
        if len(bench) != len(returns) or len(bench) < 2:
            return 0.0, 0.0

        # Beta = Cov(portfolio, benchmark) / Var(benchmark)
        cov = np.cov(returns, bench)[0, 1]
        var = np.var(bench)
        beta = float(cov / var) if var > 0 else 0.0

        # Alpha = portfolio return - beta * benchmark return
        alpha = float((np.mean(returns) - beta * np.mean(bench)) * 252 * 100)

        return alpha, beta

    def _calculate_trade_stats(
        self,
        trades: list[Order],
        fills: list[Fill],
    ) -> dict[str, Any]:
        """Calculate trade statistics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0,
                "avg_trade_pct": 0.0,
                "max_trade_duration": timedelta(0),
                "avg_trade_duration": timedelta(0),
                "profit_factor": 0.0,
                "expectancy_pct": 0.0,
                "sqn": 0.0,
                "kelly_criterion": 0.0,
            }

        # Calculate P&L per round trip
        trade_pnls = self._calculate_trade_pnls(trades, fills)

        if not trade_pnls:
            return {
                "total_trades": len(trades),
                "win_rate_pct": 0.0,
                "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0,
                "avg_trade_pct": 0.0,
                "max_trade_duration": timedelta(0),
                "avg_trade_duration": timedelta(0),
                "profit_factor": 0.0,
                "expectancy_pct": 0.0,
                "sqn": 0.0,
                "kelly_criterion": 0.0,
            }

        pnls = np.array([p["pnl_pct"] for p in trade_pnls])
        durations = [p["duration"] for p in trade_pnls]

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        win_rate = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = np.mean(pnls) if len(pnls) > 0 else 0.0

        # SQN (System Quality Number)
        if len(pnls) > 0 and np.std(pnls) > 0:
            sqn = np.sqrt(len(pnls)) * np.mean(pnls) / np.std(pnls)
        else:
            sqn = 0.0

        # Kelly Criterion
        if avg_loss != 0 and len(wins) > 0:
            win_prob = len(wins) / len(pnls)
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            kelly = win_prob - ((1 - win_prob) / win_loss_ratio) if win_loss_ratio > 0 else 0.0
        else:
            kelly = 0.0

        return {
            "total_trades": len(trade_pnls),
            "win_rate_pct": win_rate,
            "best_trade_pct": float(np.max(pnls)) if len(pnls) > 0 else 0.0,
            "worst_trade_pct": float(np.min(pnls)) if len(pnls) > 0 else 0.0,
            "avg_trade_pct": float(np.mean(pnls)) if len(pnls) > 0 else 0.0,
            "max_trade_duration": max(durations) if durations else timedelta(0),
            "avg_trade_duration": (
                sum(durations, timedelta(0)) / len(durations)
                if durations
                else timedelta(0)
            ),
            "profit_factor": profit_factor,
            "expectancy_pct": expectancy,
            "sqn": sqn,
            "kelly_criterion": kelly,
        }

    def _calculate_trade_pnls(
        self,
        trades: list[Order],
        fills: list[Fill],
    ) -> list[dict]:
        """Calculate P&L for each round trip trade."""
        # Match buys with sells
        trade_pnls = []

        # Group fills by instrument
        fills_by_instrument: dict[str, list[Fill]] = {}
        for fill in fills:
            if fill.instrument not in fills_by_instrument:
                fills_by_instrument[fill.instrument] = []
            fills_by_instrument[fill.instrument].append(fill)

        for instrument, inst_fills in fills_by_instrument.items():
            # Sort by timestamp
            inst_fills.sort(key=lambda f: f.timestamp)

            # Match buys with sells (FIFO)
            buys: list[tuple[Fill, int]] = []  # (fill, remaining_qty)

            for fill in inst_fills:
                if fill.side == OrderSide.BUY:
                    buys.append((fill, fill.quantity))
                else:
                    # Match with buys
                    remaining = fill.quantity
                    while remaining > 0 and buys:
                        buy_fill, buy_remaining = buys[0]
                        match_qty = min(remaining, buy_remaining)

                        pnl = (fill.price - buy_fill.price) * match_qty
                        pnl_pct = ((fill.price - buy_fill.price) / buy_fill.price) * 100
                        duration = fill.timestamp - buy_fill.timestamp

                        trade_pnls.append({
                            "instrument": instrument,
                            "entry_price": buy_fill.price,
                            "exit_price": fill.price,
                            "quantity": match_qty,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "duration": duration,
                        })

                        remaining -= match_qty
                        buy_remaining -= match_qty

                        if buy_remaining == 0:
                            buys.pop(0)
                        else:
                            buys[0] = (buy_fill, buy_remaining)

        return trade_pnls

    def _calculate_exposure_time(
        self,
        equity: np.ndarray,
        initial_capital: float,
    ) -> float:
        """Calculate percentage of time with open positions."""
        # Simple approximation: if equity != cash, we have positions
        # This is a simplification - actual implementation would track position changes
        changes = np.diff(equity)
        non_zero_changes = np.sum(changes != 0)
        return (non_zero_changes / len(equity)) * 100 if len(equity) > 0 else 0.0

    def _empty_metrics(self, initial_capital: float) -> PerformanceMetrics:
        """Return empty metrics when no data available."""
        now = datetime.now()
        return PerformanceMetrics(
            start_date=now,
            end_date=now,
            duration=timedelta(0),
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            buy_and_hold_return_pct=0.0,
            cagr_pct=0.0,
            volatility_ann_pct=0.0,
            max_drawdown_pct=0.0,
            avg_drawdown_pct=0.0,
            max_drawdown_duration=timedelta(0),
            avg_drawdown_duration=timedelta(0),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            alpha_pct=0.0,
            beta=0.0,
            initial_capital=initial_capital,
            final_equity=initial_capital,
            peak_equity=initial_capital,
            exposure_time_pct=0.0,
            total_trades=0,
            win_rate_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            avg_trade_pct=0.0,
            max_trade_duration=timedelta(0),
            avg_trade_duration=timedelta(0),
            profit_factor=0.0,
            expectancy_pct=0.0,
            sqn=0.0,
            kelly_criterion=0.0,
        )
