"""BarAggregator: the three closing rules change the signal, so each has a test."""
from datetime import datetime, timezone

from live.bar_aggregator import BarAggregator

T0 = int(datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc).timestamp()) * 1_000_000
MIN = 60 * 1_000_000


def _feed(agg, trades):
    out = []
    for ts, px, sz in trades:
        out.extend(agg.add_trade(ts, px, sz))
    return out


def test_ohlcv_and_bucket_start_convention():
    agg = BarAggregator(interval_s=60, discard_boot_bar=False)
    bars = _feed(agg, [(T0, 10.0, 1), (T0 + 10_000_000, 12.0, 2), (T0 + 20_000_000, 9.0, 3),
                       (T0 + 59_000_000, 11.0, 4), (T0 + MIN, 11.5, 1)])
    assert len(bars) == 1
    b = bars[0]
    assert (b.open, b.high, b.low, b.close, b.volume) == (10.0, 12.0, 9.0, 11.0, 10.0)
    assert b.timestamp == datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)  # START of bucket
    assert not b.synthetic


def test_boot_partial_bar_is_discarded():
    agg = BarAggregator(interval_s=60)
    bars = _feed(agg, [(T0 + 30_000_000, 10.0, 1), (T0 + MIN, 10.5, 1), (T0 + 2 * MIN, 11.0, 1)])
    assert [b.timestamp.minute for b in bars] == [1]  # the 10:00 fragment never comes out
    assert agg.bars_discarded_boot == 1


def test_empty_interval_emits_synthetic_bar_at_previous_close():
    agg = BarAggregator(interval_s=60, discard_boot_bar=False)
    bars = _feed(agg, [(T0, 10.0, 1), (T0 + 3 * MIN, 12.0, 1)])
    assert [b.timestamp.minute for b in bars] == [0, 1, 2]
    for b in bars[1:]:
        assert b.synthetic and b.volume == 0.0
        assert b.open == b.high == b.low == b.close == 10.0


def test_gap_above_cap_is_not_filled():
    agg = BarAggregator(interval_s=60, discard_boot_bar=False, max_synthetic_gap=5)
    bars = _feed(agg, [(T0, 10.0, 1), (T0 + 60 * MIN, 12.0, 1)])
    assert len(bars) == 1 and not bars[0].synthetic
    assert agg.large_gaps == 1


def test_late_trade_is_discarded_and_counted():
    agg = BarAggregator(interval_s=60, discard_boot_bar=False)
    _feed(agg, [(T0, 10.0, 1), (T0 + MIN, 11.0, 1)])
    assert agg.add_trade(T0 + 5_000_000, 99.0, 1) == []  # belongs to the closed 10:00 bar
    assert agg.trades_late == 1
    closed = agg.add_trade(T0 + 2 * MIN, 11.0, 1)
    assert closed[0].high == 11.0  # the late 99.0 never leaked into any bar


def test_invalid_trades_are_ignored():
    agg = BarAggregator(interval_s=60)
    assert agg.add_trade(T0, -1.0, 5) == []
    assert agg.add_trade(T0, 10.0, 0) == []
    assert agg.trades_invalid == 2 and agg.trades_received == 0
