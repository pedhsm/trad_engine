"""Triple-barrier labeling: barriers, the tie rule, and the vertical-barrier horizon."""
import pandas as pd
import pytest

from backtest.validation.pit_invariants import assert_entry_not_before_event
from core_math.labeling import apply_triple_barrier

T0 = pd.Timestamp("2024-01-01 10:00")


def _m(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def _label(prices_at, direction=1, profit_bps=100, stop_bps=100, max_holding_min=5, event_ts=T0):
    mid = pd.Series([p for _, p in prices_at], index=[_m(m) for m, _ in prices_at])
    ev = pd.DataFrame({"bet_direction": [direction]}, index=[event_ts])
    return apply_triple_barrier(ev, mid, profit_bps, stop_bps, max_holding_min)


def test_profit_and_stop_long_and_short():
    up = [(0, 100.0), (1, 100.5), (2, 101.5), (3, 98.0)]
    assert _label(up, direction=1).label.iloc[0] == 1
    assert _label(up, direction=-1).label.iloc[0] == -1


def test_quote_after_vertical_barrier_is_ignored():
    # The only move happens at 10:07, after the 5-minute horizon (10:05). It must not
    # be credited to the trade: the result is a timeout at the price in force at 10:05.
    out = _label([(0, 100.0), (7, 102.0)])
    assert out.label.iloc[0] == 0
    assert out.t_end.iloc[0] == _m(0)
    assert out.ret_realizado.iloc[0] == 0.0


def test_hit_exactly_at_vertical_barrier_counts():
    out = _label([(0, 100.0), (5, 101.0), (6, 90.0)])
    assert out.label.iloc[0] == 1 and out.t_end.iloc[0] == _m(5)


def test_timeout_inside_horizon():
    out = _label([(0, 100.0), (2, 100.3), (4, 100.2), (9, 100.0)])
    assert out.label.iloc[0] == 0
    assert out.t_end.iloc[0] == _m(4)
    assert out.ret_realizado.iloc[0] == pytest.approx(0.001998, rel=1e-3)


def test_same_timestamp_tie_is_labeled_stop():
    idx = [T0, _m(1), _m(1)]
    mid = pd.Series([100.0, 102.0, 98.0], index=idx)  # duplicated tick crosses both
    ev = pd.DataFrame({"bet_direction": [1]}, index=[T0])
    assert apply_triple_barrier(ev, mid, 100, 100, 5).label.iloc[0] == -1


def test_entry_snaps_forward_never_backward():
    out = _label([(0, 100.0), (2, 100.0), (3, 102.0)], event_ts=_m(1))
    assert out.t_start.iloc[0] == _m(2)
    assert assert_entry_not_before_event(out)[0]


def test_events_without_forward_data_are_skipped():
    assert _label([(0, 100.0)]).empty                       # no quote after the entry
    assert _label([(0, 100.0), (9, 101.0)], event_ts=_m(1)).empty  # first quote is past the horizon
