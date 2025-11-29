"""
Algorithmic Trading System

A modular backtesting and live trading framework.
"""

from algo.data.models import Candle, Tick, Interval
from algo.broker.models import (
    Order,
    OrderRequest,
    OrderStatus,
    OrderSide,
    OrderType,
    Fill,
    Position,
)
from algo.strategy.base import BaseStrategy, Signal, SignalAction
from algo.engine.backtest import BacktestEngine
from algo.engine.paper import PaperTradingEngine
from algo.engine.live import LiveTradingEngine
from algo.engine.fill_models import (
    FillModel,
    FillModelType,
    FillResult,
    NextOpenFillModel,
    SameCloseFillModel,
    SlippageFillModel,
)
from algo.analytics.metrics import PerformanceMetrics, BacktestResult

__all__ = [
    # Data models
    "Candle",
    "Tick",
    "Interval",
    # Broker models
    "Order",
    "OrderRequest",
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "Fill",
    "Position",
    # Strategy
    "BaseStrategy",
    "Signal",
    "SignalAction",
    # Engines
    "BacktestEngine",
    "PaperTradingEngine",
    "LiveTradingEngine",
    # Fill models
    "FillModel",
    "FillModelType",
    "FillResult",
    "NextOpenFillModel",
    "SameCloseFillModel",
    "SlippageFillModel",
    # Analytics
    "PerformanceMetrics",
    "BacktestResult",
]
