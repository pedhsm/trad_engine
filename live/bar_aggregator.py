"""Bar aggregator: executed trades -> fixed-interval OHLCV bars.

Why this module exists
----------------------
A strategy is usually validated over BARS, while a live feed delivers TRADES
(tick by tick). Someone has to turn one into the other, and that someone is this
file.

It is deliberately dumb and dependency-free: a trade goes in, a closed bar comes
out. No network, no pandas, no wall clock. That is what lets a test suite exercise
every closing rule with no broker and no compiler — and those rules CHANGE THE
SIGNAL, so they need a test, not trust.

A divergence this module does NOT solve
---------------------------------------
If your live feed and your backtest bars come from DIFFERENT SOURCES for the same
instrument, they will diverge in detail: timestamps, trade-condition filters
(auction, cross, correction), exchange consolidation. Two bars of the same minute
may not share the same close or the same volume. There is no fix inside this
module — only measurement: compare, at end of day, the bar built here with the bar
of the same window in your store, and watch the size of the difference. Until that
measurement exists, it is a known, accepted risk, not an eliminated one.

Sampling caveat
---------------
If your feed delivers trades in periodic samples rather than one-by-one, individual
trades in a very liquid asset are lost in the sampling: the bar volume is a FLOOR,
not the true volume of the interval. Features that depend on the LEVEL of volume
carry this bias; those that depend on the RATIO between volumes are far less
affected, because the bias appears on both sides of the division.

The three rules that change the signal
--------------------------------------
1. BOOT PARTIAL BAR — the strategy starts in the middle of an interval. That first
   bar covers only part of it and would enter the feature window with the same
   weight as a full bar. It is discarded.
2. INTERVAL WITH NO TRADE — emits a synthetic bar (OHLC = previous close, volume 0).
   Feature windows count BARS, not time: skipping the empty interval would make a
   window of 20 mean 20 minutes in a liquid asset and several hours in an illiquid
   one — two different things with the same name.
3. LATE TRADE — arrived after its bar had already closed. Discarded and COUNTED.
   Reopening an already-emitted bar would re-emit a signal for an instant that has
   passed, and the corresponding order would already have been sent.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, List, Optional

log = logging.getLogger("bar_aggregator")

# Default bar interval. 1 minute is short enough to react within the session, long
# enough that feed sampling does not dominate the bar content.
DEFAULT_INTERVAL_S = 60

# Cap on synthetic bars emitted at once. An interval with no trade can be a dead
# minute (filling is right) or an entire weekend (filling would invent thousands of
# fake bars and push the whole feature window out of the real market). Above the cap
# the aggregator RESTARTS instead of filling, and counts the jump so it shows up
# instead of going unnoticed.
MAX_SYNTHETIC_GAP = 120

# History kept in memory. It must cover the largest window a strategy uses, with
# room to spare. The buffer is bounded on purpose: a process running for weeks with
# a growing list ends up in swap.
DEFAULT_MAX_BARS = 5_000


@dataclass(frozen=True)
class Bar:
    """A closed bar.

    `timestamp` is the START of the bucket (convention: the 10:05 bar covers
    [10:05, 10:06)). Switching to the end would shift the whole series by one bar
    against a backtest that uses the start convention.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # An interval-with-no-trade bar, reconstructed from the previous close. Marked
    # because whoever diagnoses needs to distinguish "market stalled" from "collector
    # crashed": a long run of synthetics is a symptom, not data.
    synthetic: bool = False


class BarAggregator:
    """Builds fixed-interval bars from the trades of a single asset.

    The closing clock comes from the TRADE TIMESTAMP, never the local clock. Using
    machine time would close the bar early or late depending on the network latency
    of the moment, and a backtest — which reads recorded stamps — has no such jitter.
    Two bars built from the same data would give different results just because of
    where the process happened to run.
    """

    def __init__(self, ticker: str = "", interval_s: int = DEFAULT_INTERVAL_S,
                 fill_gaps: bool = True,
                 max_synthetic_gap: int = MAX_SYNTHETIC_GAP,
                 max_bars: int = DEFAULT_MAX_BARS,
                 discard_boot_bar: bool = True):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")

        self.ticker = ticker
        self.interval_s = int(interval_s)
        self.fill_gaps = bool(fill_gaps)
        self.max_synthetic_gap = int(max_synthetic_gap)
        self.discard_boot_bar = bool(discard_boot_bar)

        self.bars: Deque[Bar] = deque(maxlen=max_bars)

        # Bucket being formed (epoch in seconds, aligned to the interval).
        self._bucket: Optional[int] = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0.0
        self._last_close: Optional[float] = None
        self._is_boot_bar = True

        # Exposed counters. They all exist to show up in the log/diagnostics: a number
        # nobody can see protects nobody.
        self.trades_received = 0
        self.trades_late = 0
        self.trades_invalid = 0
        self.bars_emitted = 0
        self.bars_synthetic = 0
        self.bars_discarded_boot = 0
        self.large_gaps = 0

    # -- input ---------------------------------------------------------------
    def add_trade(self, timestamp_us: int, price: float, size: float) -> List[Bar]:
        """Register a trade. Returns the bars it CLOSED (0, 1 or more).

        Returns a list, not a bar, because a single trade can close the current bar
        AND the synthetics of every empty interval between it and the trade's bucket.
        """
        if price <= 0 or size <= 0:
            # A non-positive price is a common "no trade" sentinel, and a zero size
            # moves no bar at all.
            self.trades_invalid += 1
            return []

        self.trades_received += 1
        bucket = self._bucket_of(timestamp_us)

        # --- first trade in the aggregator's life ---
        if self._bucket is None:
            self._open_bucket(bucket, price, size)
            self._is_boot_bar = self.discard_boot_bar
            return []

        # --- late trade: its bar has already closed ---
        if bucket < self._bucket:
            self.trades_late += 1
            log.warning(
                "[%s] late trade discarded (bucket %s < current bar %s). "
                "Total late: %d. Reopening the bar would emit a signal for an "
                "instant that has passed.",
                self.ticker, self._iso(bucket), self._iso(self._bucket),
                self.trades_late,
            )
            return []

        # --- same bucket: just update what is being formed ---
        if bucket == self._bucket:
            self._accumulate(price, size)
            return []

        # --- new bucket: close the current, fill the gap, open the new one ---
        closed: List[Bar] = []
        current = self._close_bucket()
        if current is not None:
            closed.append(current)

        closed.extend(self._fill_until(bucket))

        self._open_bucket(bucket, price, size)
        return closed

    # -- query ---------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "ticker": self.ticker,
            "trades_received": self.trades_received,
            "trades_late": self.trades_late,
            "trades_invalid": self.trades_invalid,
            "bars_emitted": self.bars_emitted,
            "bars_synthetic": self.bars_synthetic,
            "bars_discarded_boot": self.bars_discarded_boot,
            "large_gaps": self.large_gaps,
            "bars_in_memory": len(self.bars),
            "forming_bar": self._iso(self._bucket) if self._bucket else None,
        }

    # -- internals -----------------------------------------------------------
    def _bucket_of(self, timestamp_us: int) -> int:
        """Epoch in seconds truncated to the start of the interval."""
        seconds = int(timestamp_us) // 1_000_000
        return seconds - (seconds % self.interval_s)

    @staticmethod
    def _dt(bucket: int) -> datetime:
        # Explicit UTC: the stamp is epoch. Leaving it naive would invite pandas to
        # read it as local time on one machine and not on another.
        return datetime.fromtimestamp(bucket, tz=timezone.utc)

    def _iso(self, bucket: Optional[int]) -> Optional[str]:
        return None if bucket is None else self._dt(bucket).isoformat()

    def _open_bucket(self, bucket: int, price: float, size: float) -> None:
        self._bucket = bucket
        self._open = self._high = self._low = self._close = float(price)
        self._volume = float(size)

    def _accumulate(self, price: float, size: float) -> None:
        p = float(price)
        if p > self._high:
            self._high = p
        if p < self._low:
            self._low = p
        self._close = p
        self._volume += float(size)

    def _close_bucket(self) -> Optional[Bar]:
        """Close the current bucket. Returns None if the bar is discarded."""
        if self._bucket is None:
            return None

        bar = Bar(
            timestamp=self._dt(self._bucket),
            open=self._open, high=self._high, low=self._low,
            close=self._close, volume=self._volume, synthetic=False,
        )

        if self._is_boot_bar:
            # The boot bar is not a bar: it is a fragment of one. It would enter the
            # feature window with a full-interval weight without being one.
            self._is_boot_bar = False
            self.bars_discarded_boot += 1
            self._last_close = self._close   # the close is real and serves as anchor
            log.info(
                "[%s] boot partial bar (%s) discarded — the strategy started in the "
                "middle of the interval and this bar is incomplete.",
                self.ticker, bar.timestamp.isoformat(),
            )
            return None

        self.bars.append(bar)
        self.bars_emitted += 1
        self._last_close = self._close
        return bar

    def _fill_until(self, next_bucket: int) -> List[Bar]:
        """Synthetics for the intervals between the closed bar and the next trade."""
        if self._bucket is None or self._last_close is None:
            return []

        missing = (next_bucket - self._bucket) // self.interval_s - 1
        if missing <= 0:
            return []

        if not self.fill_gaps:
            return []

        if missing > self.max_synthetic_gap:
            # Weekend, holiday, closed session, or the collector crashed. Filling would
            # invent hundreds of bars and push the whole feature window out of the real
            # market. Restart and leave the jump in plain sight.
            self.large_gaps += 1
            log.warning(
                "[%s] jump of %d bars (%s -> %s) above the cap of %d. NOT filled: the "
                "series has a hole, and inventing bars here would be worse than "
                "admitting it. Feature windows take %d bars to cover only real data "
                "again.",
                self.ticker, missing, self._iso(self._bucket),
                self._iso(next_bucket), self.max_synthetic_gap, missing,
            )
            return []

        close = float(self._last_close)
        synthetics: List[Bar] = []
        for i in range(1, missing + 1):
            bucket = self._bucket + i * self.interval_s
            bar = Bar(timestamp=self._dt(bucket), open=close, high=close,
                      low=close, close=close, volume=0.0, synthetic=True)
            self.bars.append(bar)
            self.bars_emitted += 1
            self.bars_synthetic += 1
            synthetics.append(bar)
        return synthetics
