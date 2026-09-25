"""live: execution-side plumbing for turning a trade feed into bars.

Deliberately small and dependency-free. The heavy live path (broker adapters,
IPC transport) lives in the C++ engine; what is here is the pure-Python glue that
a backtest and a paper-trading demo can share with the live loop.
"""
from __future__ import annotations

from .bar_aggregator import Bar, BarAggregator

__all__ = ["Bar", "BarAggregator"]
