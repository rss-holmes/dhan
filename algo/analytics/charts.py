"""Interactive Plotly charts for backtest visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from algo.analytics.metrics import BacktestResult
    from algo.data.models import Candle

PLOTLY_AVAILABLE = False

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    make_subplots = None


class BacktestCharts:
    """Interactive Plotly charts for backtest visualization."""

    def __init__(self, result: BacktestResult):
        """
        Initialize charts with backtest result.

        Args:
            result: Backtest result containing equity curve and trades
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for charts. Install with: pip install plotly"
            )
        self.result = result

    def equity_curve(self) -> Any:
        """Generate interactive equity curve chart."""
        assert go is not None, "Plotly not available"
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = [eq for _, eq in self.result.equity_curve]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(color="#2E86AB", width=2),
                fill="tozeroy",
                fillcolor="rgba(46, 134, 171, 0.1)",
            )
        )

        # Add horizontal line for initial capital
        fig.add_hline(
            y=self.result.metrics.initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital",
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def drawdown(self) -> Any:
        """Generate drawdown chart."""
        assert go is not None, "Plotly not available"
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = np.array([eq for _, eq in self.result.equity_curve])

        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdowns,
                mode="lines",
                name="Drawdown",
                line=dict(color="#E63946", width=2),
                fill="tozeroy",
                fillcolor="rgba(230, 57, 70, 0.2)",
            )
        )

        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def returns_distribution(self) -> Any:
        """Generate returns distribution histogram."""
        assert go is not None, "Plotly not available"
        equity = np.array([eq for _, eq in self.result.equity_curve])
        returns = np.diff(equity) / equity[:-1] * 100

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name="Daily Returns",
                marker_color="#457B9D",
            )
        )

        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        # Add mean line
        mean_return = np.mean(returns)
        fig.add_vline(
            x=mean_return,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Mean: {mean_return:.2f}%",
        )

        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
        )

        return fig

    def monthly_returns_heatmap(self) -> Any:
        """Generate monthly returns heatmap."""
        assert go is not None, "Plotly not available"
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = [eq for _, eq in self.result.equity_curve]

        # Calculate monthly returns
        monthly_data: dict[tuple[int, int], list[float]] = {}
        for i in range(1, len(dates)):
            year = dates[i].year
            month = dates[i].month
            daily_return = (equity[i] - equity[i - 1]) / equity[i - 1]

            key = (year, month)
            if key not in monthly_data:
                monthly_data[key] = []
            monthly_data[key].append(daily_return)

        # Aggregate to monthly returns
        monthly_returns: dict[tuple[int, int], float] = {}
        for key, daily_returns in monthly_data.items():
            # Compound daily returns
            monthly_return = (np.prod(1 + np.array(daily_returns)) - 1) * 100
            monthly_returns[key] = float(monthly_return)

        if not monthly_returns:
            return go.Figure()

        # Create matrix
        years = sorted(set(k[0] for k in monthly_returns.keys()))
        months = list(range(1, 13))
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        z = []
        for year in years:
            row = []
            for month in months:
                ret = monthly_returns.get((year, month), np.nan)
                row.append(ret)
            z.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=month_names,
                y=[str(y) for y in years],
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
        )

        return fig

    def trades_chart(self, ohlc_data: list[Candle], instrument: str) -> Any:
        """
        Generate candlestick chart with trade markers.

        Args:
            ohlc_data: List of candles for the instrument
            instrument: Instrument identifier
        """
        assert go is not None and make_subplots is not None, "Plotly not available"
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Candlestick
        dates = [c.timestamp for c in ohlc_data]

        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=[c.open for c in ohlc_data],
                high=[c.high for c in ohlc_data],
                low=[c.low for c in ohlc_data],
                close=[c.close for c in ohlc_data],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=dates,
                y=[c.volume for c in ohlc_data],
                name="Volume",
                marker_color="rgba(100, 100, 100, 0.5)",
            ),
            row=2,
            col=1,
        )

        # Add trade markers
        for trade in self.result.trades:
            if trade.instrument == instrument and hasattr(trade, "average_price"):
                color = "#2E7D32" if trade.side.value == "BUY" else "#C62828"
                symbol = "triangle-up" if trade.side.value == "BUY" else "triangle-down"

                fig.add_trace(
                    go.Scatter(
                        x=[trade.created_at],
                        y=[trade.average_price],
                        mode="markers",
                        marker=dict(
                            symbol=symbol,
                            size=12,
                            color=color,
                        ),
                        name=f"{trade.side.value} @ {trade.average_price:.2f}",
                        hovertemplate=(
                            f"{trade.side.value}<br>"
                            f"Qty: {trade.quantity}<br>"
                            f"Price: {trade.average_price:.2f}"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            title=f"Trades - {instrument}",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=600,
        )

        return fig

    def full_report(self) -> Any:
        """Generate comprehensive multi-panel report."""
        assert go is not None and make_subplots is not None, "Plotly not available"
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Equity Curve",
                "Drawdown",
                "Monthly Returns",
                "Trade Returns Distribution",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
            ],
        )

        # Equity curve
        dates = [dt for dt, _ in self.result.equity_curve]
        equity = np.array([eq for _, eq in self.result.equity_curve])

        fig.add_trace(
            go.Scatter(
                x=dates, y=equity, name="Equity", line=dict(color="#2E86AB")
            ),
            row=1,
            col=1,
        )

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak * 100

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdowns,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#E63946"),
            ),
            row=1,
            col=2,
        )

        # Returns histogram
        returns = np.diff(equity) / equity[:-1] * 100

        fig.add_trace(
            go.Histogram(x=returns, name="Returns", marker_color="#457B9D"),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            template="plotly_white",
            title_text="Backtest Report",
        )

        return fig

    def save_html(self, filepath: str) -> None:
        """
        Save full report as interactive HTML.

        Args:
            filepath: Path to save HTML file
        """
        fig = self.full_report()
        fig.write_html(filepath)
        print(f"Report saved to: {filepath}")

    def show(self) -> None:
        """Display full report in browser."""
        fig = self.full_report()
        fig.show()
