"""Sanity checks: known-answer tests for the validation harness.

The other test files check that each piece WORKS. These check that the harness
reaches the RIGHT CONCLUSION on cases where the answer is known in advance:

- accounting identities (buy-and-hold must earn exactly log(P_end / P_start));
- MCPT must find an edge that was planted, and must NOT find one in pure noise;
- MCPT is blind to lookahead (an oracle wins on permuted data too), so the PIT
  checks must be the ones that catch it — both halves are asserted here, because
  "MCPT passed" is exactly the sentence that makes people skip the PIT checks.

Everything is seeded, so these are deterministic, not flaky statistics.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.validation import pit_invariants as pit
from backtest.validation.strategy_base import (
    MCPTTester, StrategyResult, TradingStrategy, _insample_signal_func,
)
from core_math import micro_math
from core_math.labeling import apply_triple_barrier


# --- strategies with a KNOWN nature (module level: MCPT pickles them to workers) ---

class ConstantStrategy(TradingStrategy):
    def __init__(self, position: float = 1.0):
        super().__init__(f"constant {position:+}")
        self.position = position

    def generate_signal(self, ohlc, **params):
        return StrategyResult(pd.Series(self.position, index=ohlc.index), {})

    def get_parameter_space(self):
        return {}


class AlternatingStrategy(TradingStrategy):
    """+1, -1, +1, ... : flips every bar, to make costs easy to predict."""
    def __init__(self):
        super().__init__("alternating")

    def generate_signal(self, ohlc, **params):
        sig = np.where(np.arange(len(ohlc)) % 2 == 0, 1.0, -1.0)
        return StrategyResult(pd.Series(sig, index=ohlc.index), {})

    def get_parameter_space(self):
        return {}


class MomentumStrategy(TradingStrategy):
    """Causal: hold the sign of the LAST bar's return."""
    def __init__(self):
        super().__init__("momentum")

    def generate_signal(self, ohlc, **params):
        r = np.log(ohlc["close"]).diff()
        return StrategyResult(np.sign(r).fillna(0.0), {})

    def get_parameter_space(self):
        return {}


class OracleStrategy(TradingStrategy):
    """LOOKAHEAD on purpose: holds the sign of the NEXT bar's return."""
    def __init__(self):
        super().__init__("oracle")

    def generate_signal(self, ohlc, **params):
        r_next = np.log(ohlc["close"]).diff().shift(-1)
        return StrategyResult(np.sign(r_next).fillna(0.0), {})

    def get_parameter_space(self):
        return {}


def make_ohlc(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    """Bars whose open is the previous close, so each bar's move IS its return."""
    close = start * np.exp(np.cumsum(returns))
    open_ = np.r_[start, close[:-1]]
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) * 1.0005,
         "low": np.minimum(open_, close) * 0.9995,
         "close": close},
        index=pd.date_range("2020-01-01", periods=len(returns), freq="D"),
    )


def random_walk(n: int, seed: int) -> pd.DataFrame:
    return make_ohlc(np.random.default_rng(seed).normal(0.0, 0.01, n))


def momentum_market(n: int, phi: float, seed: int) -> pd.DataFrame:
    """AR(1) returns: r_t = phi * r_{t-1} + noise. phi > 0 plants a real, causal edge."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.01, n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return make_ohlc(r)


def strategy_returns(strategy, df, **params) -> pd.Series:
    return _insample_signal_func((df, strategy, params))


# --- 1. accounting identities ------------------------------------------------

def test_buy_and_hold_earns_exactly_the_market():
    df = random_walk(500, seed=1)
    market = np.log(df["close"].iloc[-1] / df["close"].iloc[0])
    assert strategy_returns(ConstantStrategy(+1), df).sum() == pytest.approx(market, abs=1e-12)
    assert strategy_returns(ConstantStrategy(-1), df).sum() == pytest.approx(-market, abs=1e-12)
    assert strategy_returns(ConstantStrategy(0), df).sum() == 0.0


def test_returns_do_not_depend_on_the_price_level():
    df = random_walk(300, seed=2)
    scaled = df * 7.0
    a = strategy_returns(MomentumStrategy(), df)
    b = strategy_returns(MomentumStrategy(), scaled)
    np.testing.assert_allclose(a.fillna(0), b.fillna(0), rtol=0, atol=1e-12)


def test_costs_are_charged_per_unit_of_turnover():
    df = random_walk(200, seed=3)
    gross = strategy_returns(AlternatingStrategy(), df)
    net = strategy_returns(AlternatingStrategy(), df, spread_bps=1.0, commission_bps=0.5)
    # The aligned position goes NaN, +1, -1, +1, ...: the first defined flip is from
    # NaN (not charged), every later bar turns over 2 units, at 1.5 bps each.
    n_flips = len(df) - 2
    assert (gross - net).sum() == pytest.approx(n_flips * 2 * 1.5e-4, rel=1e-12)


# --- 2. MCPT known answers ------------------------------------------------------
# p-value = share of permuted series (bars shuffled, which destroys any time
# structure) on which the strategy did at least as well as on the real one, the real
# run included. A real edge -> rarely matched by chance -> small p. No edge -> p is
# uniformly distributed between 0 and 1.

def test_mcpt_finds_a_planted_edge():
    df = momentum_market(1500, phi=0.3, seed=4)
    res = MCPTTester(MomentumStrategy()).run_insample_test(df, n_permutations=100)
    assert res["p_value"] == pytest.approx(0.01)  # the minimum possible with 100 draws


def test_mcpt_does_not_find_an_edge_in_noise():
    # Same strategy, 8 independent random walks. Under "no edge", P(p <= 0.05) = 5%,
    # so 3 or more hits out of 8 would happen by chance only ~0.6% of the time.
    ps = []
    for seed in range(8):
        res = MCPTTester(MomentumStrategy()).run_insample_test(random_walk(800, seed=100 + seed),
                                                               n_permutations=40)
        ps.append(res["p_value"])
    assert sum(p <= 0.05 for p in ps) <= 2, ps
    assert 0.25 <= float(np.mean(ps)) <= 0.75, ps


def test_mcpt_is_blind_to_lookahead_so_pit_must_catch_it():
    df = random_walk(800, seed=5)
    oracle = OracleStrategy()

    # The oracle makes a fortune...
    assert strategy_returns(oracle, df).sum() > 1.0
    # ...but it cheats on the permuted series just as well, so MCPT sees nothing special.
    res = MCPTTester(oracle).run_insample_test(df, n_permutations=40)
    assert res["p_value"] > 0.5

    # The point-in-time check is what flags it.
    pipeline = lambda d: pd.DataFrame({"signal": oracle.generate_signal(d).signal})
    ok, _ = pit.assert_feature_no_future_dependence(pipeline, df, cutoff=df.index[400])
    assert not ok
    ok, detail = pit.assert_feature_no_future_dependence(
        lambda d: pd.DataFrame({"signal": MomentumStrategy().generate_signal(d).signal}),
        df, cutoff=df.index[400])
    assert ok, detail


# --- 3. research -> live parity -----------------------------------------------------

def test_live_example_decides_exactly_like_the_backtest_primitive():
    # The live SMA-cross computes only the last value of each mean (for speed); the
    # backtest would use core_math.bars_math.sma over the whole series. Bar by bar,
    # both must reach the same decision.
    from datetime import datetime, timezone
    from core_math.bars_math import sma
    from examples.live_client import SmaCross
    from live.bar_aggregator import Bar

    close = random_walk(3000, seed=8)["close"].to_numpy()
    live = SmaCross(fast=10, slow=40)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    live_decisions = [live.on_bar(Bar(ts, c, c, c, c, 1.0)) for c in close]
    fast, slow = sma(close, 10), sma(close, 40)
    for i, d in enumerate(live_decisions):
        expected = None if np.isnan(slow[i]) else (1 if fast[i] > slow[i] else -1)
        assert d == expected, f"bar {i}: live {d} vs backtest {expected}"


# --- 4. primitives on inputs with an obvious answer ------------------------------

def test_triple_barrier_is_unbiased_on_a_driftless_walk():
    rng = np.random.default_rng(6)
    n = 200_000
    mid = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 2e-4, n))),
                    index=pd.date_range("2024-01-01", periods=n, freq="s"))
    event_ts = mid.index[rng.choice(n - 5000, 2000, replace=False)]
    events = pd.DataFrame({"bet_direction": rng.choice([-1, 1], 2000)}, index=event_ts).sort_index()
    out = apply_triple_barrier(events, mid, profit_bps=20, stop_bps=20, max_holding_min=60)
    decided = out[out.label != 0]
    assert len(decided) > 1500
    assert 0.45 <= (decided.label == 1).mean() <= 0.55


def test_bvc_reads_the_direction_of_the_move():
    up = 100 * np.exp(np.cumsum(np.full(200, 1e-3) + np.random.default_rng(7).normal(0, 1e-4, 200)))
    buy, sell = micro_math.bulk_volume_classification(up, vol_window=30)
    assert buy[50:].mean() > 0.9 and np.allclose(buy + sell, 1.0)
    buy_down, _ = micro_math.bulk_volume_classification(up[::-1], vol_window=30)
    assert buy_down[50:].mean() < 0.1
