"""Point-in-time: every core_math primitive must be causal, and the PIT checkers
themselves must actually catch a leak.

The second half matters as much as the first: a leak detector that has never been
seen failing proves nothing. Each checker is run against a pipeline with a planted
lookahead and must return ok=False.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.validation import pit_invariants as pit
from core_math import bars_math, micro_math


@pytest.fixture(scope="module")
def bars():
    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) * 1.001,
         "low": np.minimum(open_, close) * 0.999,
         "close": close,
         "volume": rng.integers(100, 1000, n).astype(float)},
        index=pd.date_range("2024-01-01", periods=n, freq="min"),
    )


CUTOFF_POS = 250

BAR_PRIMITIVES = {
    "returns": lambda d: bars_math.returns(d["close"]),
    "log_returns": lambda d: bars_math.log_returns(d["close"]),
    "sma": lambda d: bars_math.sma(d["close"], 20),
    "ema": lambda d: bars_math.ema(d["close"], 20),
    "rolling_std": lambda d: bars_math.rolling_std(d["close"], 20),
    "realized_vol": lambda d: bars_math.realized_vol(d["close"], 20),
    "zscore": lambda d: bars_math.zscore(d["close"], 20),
    "rolling_rank": lambda d: bars_math.rolling_rank(d["close"], 20),
    "true_range": lambda d: bars_math.true_range(d["high"], d["low"], d["close"]),
    "atr": lambda d: bars_math.atr(d["high"], d["low"], d["close"], 14),
    "rolling_min": lambda d: bars_math.rolling_min(d["close"], 20),
    "rolling_max": lambda d: bars_math.rolling_max(d["close"], 20),
    "rsi": lambda d: bars_math.rsi(d["close"], 14),
    "bvc": lambda d: micro_math.bulk_volume_classification(d["close"].to_numpy(), vol_window=30)[0],
}


@pytest.mark.parametrize("name", sorted(BAR_PRIMITIVES))
def test_bar_primitive_is_causal(bars, name):
    fn = BAR_PRIMITIVES[name]
    pipeline = lambda d: pd.DataFrame({"v": fn(d)}, index=d.index)
    ok, detail = pit.assert_feature_no_future_dependence(
        pipeline, bars, cutoff=bars.index[CUTOFF_POS], numeric_cols=["open", "high", "low", "close"])
    assert ok, detail


def test_vpin_pipeline_is_causal(bars):
    def pipeline(d):
        out = micro_math.compute_vpin_pipeline(
            d.reset_index(names="ts"), bucket_volume=2000.0, vpin_window_buckets=10)
        return out.set_index("bucket_end_ts")[["vpin", "tib"]]

    ok, detail = pit.assert_feature_no_future_dependence(
        pipeline, bars, cutoff=bars.index[CUTOFF_POS], numeric_cols=["close", "volume"])
    assert ok, detail


# --- the checkers must catch planted leaks -----------------------------------

def test_future_dependence_checker_catches_centered_window(bars):
    # A centered rolling mean uses bars AFTER t: a textbook lookahead.
    leaky = lambda d: pd.DataFrame({"v": d["close"].rolling(21, center=True).mean()}, index=d.index)
    ok, _ = pit.assert_feature_no_future_dependence(leaky, bars, cutoff=bars.index[CUTOFF_POS])
    assert not ok


def test_future_dependence_checker_catches_negative_shift(bars):
    leaky = lambda d: pd.DataFrame({"v": d["close"].shift(-1)}, index=d.index)
    ok, _ = pit.assert_feature_no_future_dependence(leaky, bars, cutoff=bars.index[CUTOFF_POS])
    assert not ok


def test_event_support_checker_catches_left_edge_stamp():
    # Observations burst at 10:00-10:04 after a quiet hour. A detector stamping the
    # event at the LEFT edge of its 5-min window (09:55) has no observations behind it.
    obs = pd.DatetimeIndex(pd.date_range("2024-01-01 10:00", periods=5, freq="min"))
    right_edge = pd.DatetimeIndex([pd.Timestamp("2024-01-01 10:04")])
    left_edge = pd.DatetimeIndex([pd.Timestamp("2024-01-01 09:59")])
    assert pit.assert_events_have_causal_support(right_edge, obs, window_min=5, min_obs=5)[0]
    assert not pit.assert_events_have_causal_support(left_edge, obs, window_min=5, min_obs=5)[0]


def test_entry_before_event_checker():
    t = pd.Timestamp("2024-01-01 10:00")
    good = pd.DataFrame({"event_ts": [t], "t_start": [t + pd.Timedelta(seconds=1)]})
    bad = pd.DataFrame({"event_ts": [t], "t_start": [t - pd.Timedelta(seconds=1)]})
    assert pit.assert_entry_not_before_event(good)[0]
    assert not pit.assert_entry_not_before_event(bad)[0]


def test_trade_sign_checker_catches_inverted_convention():
    idx = pd.date_range("2024-01-01", periods=200, freq="s")
    rng = np.random.default_rng(4)
    side = rng.choice([-1.0, 1.0], idx.size)
    mid = pd.Series(100.0, index=idx)
    trades = pd.DataFrame({"price": 100.0 + 0.01 * side, "signed_vol": side * 5}, index=idx)
    assert pit.assert_trade_sign_matches_price(trades, mid)[0]
    trades["signed_vol"] *= -1  # vendor aggressor flag mapped backwards
    assert not pit.assert_trade_sign_matches_price(trades, mid)[0]


def test_no_event_before_first_data():
    data = pd.date_range("2024-01-01 10:00", periods=10, freq="min")
    assert pit.assert_no_event_before_first_data(data[3:5], data)[0]
    assert not pit.assert_no_event_before_first_data(
        pd.DatetimeIndex([pd.Timestamp("2024-01-01 09:59")]), data)[0]
