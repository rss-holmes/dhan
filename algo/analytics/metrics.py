"""Performance metrics models."""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Complete backtest performance metrics."""

    # Time
    start_date: datetime
    end_date: datetime
    duration: timedelta

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    buy_and_hold_return_pct: float
    cagr_pct: float

    # Risk
    volatility_ann_pct: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration: timedelta
    avg_drawdown_duration: timedelta

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Alpha/Beta
    alpha_pct: float
    beta: float

    # Capital
    initial_capital: float = Field(gt=0)
    final_equity: float
    peak_equity: float
    exposure_time_pct: float = Field(ge=0, le=100)

    # Trades
    total_trades: int = Field(ge=0)
    win_rate_pct: float = Field(ge=0, le=100)
    best_trade_pct: float
    worst_trade_pct: float
    avg_trade_pct: float
    max_trade_duration: timedelta
    avg_trade_duration: timedelta
    profit_factor: float
    expectancy_pct: float
    sqn: float  # System Quality Number
    kelly_criterion: float

    model_config = {"frozen": True}


class BacktestResult(BaseModel):
    """Complete backtest result including metrics and data."""

    metrics: PerformanceMetrics
    equity_curve: list[tuple[datetime, float]]
    trades: list[Any]  # List of Order objects
    fills: list[Any] = Field(default_factory=list)  # List of Fill objects
    strategies: dict[str, str]

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "metrics": self.metrics.model_dump(),
            "equity_curve": [(str(dt), eq) for dt, eq in self.equity_curve],
            "trades": [
                t.model_dump() if hasattr(t, "model_dump") else t for t in self.trades
            ],
            "fills": [
                f.model_dump() if hasattr(f, "model_dump") else f for f in self.fills
            ],
            "strategies": self.strategies,
        }

    def summary(self) -> str:
        """Generate text summary similar to the example."""
        m = self.metrics
        return f"""
================================================================================
                         BACKTEST RESULTS
================================================================================

Strategy: {', '.join(self.strategies.values())}

--- TIME ---
Start                     {m.start_date.strftime('%Y-%m-%d %H:%M:%S')}
End                       {m.end_date.strftime('%Y-%m-%d %H:%M:%S')}
Duration                  {m.duration.days} days

--- RETURNS ---
Total Return [%]          {m.total_return_pct:>10.2f}
Buy & Hold Return [%]     {m.buy_and_hold_return_pct:>10.2f}
Return (Ann.) [%]         {m.annualized_return_pct:>10.2f}
CAGR [%]                  {m.cagr_pct:>10.2f}

--- RISK ---
Volatility (Ann.) [%]     {m.volatility_ann_pct:>10.2f}
Max. Drawdown [%]         {m.max_drawdown_pct:>10.2f}
Avg. Drawdown [%]         {m.avg_drawdown_pct:>10.2f}
Max. Drawdown Duration    {m.max_drawdown_duration.days} days
Avg. Drawdown Duration    {m.avg_drawdown_duration.days} days

--- RISK-ADJUSTED ---
Sharpe Ratio              {m.sharpe_ratio:>10.2f}
Sortino Ratio             {m.sortino_ratio:>10.2f}
Calmar Ratio              {m.calmar_ratio:>10.2f}

--- ALPHA/BETA ---
Alpha [%]                 {m.alpha_pct:>10.2f}
Beta                      {m.beta:>10.2f}

--- CAPITAL ---
Initial Capital [$]       {m.initial_capital:>10,.2f}
Final Equity [$]          {m.final_equity:>10,.2f}
Peak Equity [$]           {m.peak_equity:>10,.2f}
Exposure Time [%]         {m.exposure_time_pct:>10.2f}

--- TRADES ---
# Trades                  {m.total_trades:>10}
Win Rate [%]              {m.win_rate_pct:>10.2f}
Best Trade [%]            {m.best_trade_pct:>10.2f}
Worst Trade [%]           {m.worst_trade_pct:>10.2f}
Avg. Trade [%]            {m.avg_trade_pct:>10.2f}
Max. Trade Duration       {m.max_trade_duration.days} days
Avg. Trade Duration       {m.avg_trade_duration.days} days
Profit Factor             {m.profit_factor:>10.2f}
Expectancy [%]            {m.expectancy_pct:>10.2f}
SQN                       {m.sqn:>10.2f}
Kelly Criterion           {m.kelly_criterion:>10.4f}

================================================================================
"""
