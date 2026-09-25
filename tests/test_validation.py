"""Validation harness: permutation mechanics, the MCPT p-value, and the frozen model."""
import numpy as np
import pandas as pd
import pytest

from backtest.validation.bar_permute import get_permutation
from core_math.meta_model import apply_meta_model


def _ohlc(n=500, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.001, n))
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) * 1.002,
         "low": np.minimum(open_, close) * 0.998,
         "close": close},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


def test_bar_permutation_is_seeded_and_changes_the_path():
    df = _ohlc()
    a = get_permutation(df, seed=1)
    b = get_permutation(df, seed=1)
    c = get_permutation(df, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["close"], c["close"])
    assert not np.allclose(a["close"], df["close"])


def test_bar_permutation_preserves_endpoints_and_bar_shape():
    # Shuffling the relative moves keeps their SUM, so the path starts and ends at
    # the same price: the permuted market has the same drift, only the ORDER of
    # moves is destroyed — which is exactly the null hypothesis being tested.
    df = _ohlc()
    p = get_permutation(df, seed=3)
    assert p["close"].iloc[0] == pytest.approx(df["close"].iloc[0])
    assert p["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1], rel=1e-9)
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()


def test_bar_permutation_keeps_prefix_before_start_index():
    df = _ohlc()
    p = get_permutation(df, start_index=100, seed=4)
    np.testing.assert_allclose(p.iloc[:101][["open", "high", "low", "close"]],
                               df.iloc[:101][["open", "high", "low", "close"]])


def test_mcpt_p_value_counts_the_real_run():
    # p = (1 + #perms >= real) / n_permutations: the real series counts as one of
    # the n draws, so p can never be 0 (the smallest possible is 1/n).
    from backtest.validation.strategy_base import MCPTTester
    from backtest.strategies._legacy.donchian_strategy import DonchianStrategy
    res = MCPTTester(DonchianStrategy()).run_insample_test(_ohlc(800), n_permutations=20)
    assert 1 / 20 <= res["p_value"] <= 1.0
    assert len(res["permuted_cumulative_returns"]) == 19


def test_frozen_meta_model_matches_sklearn():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(6)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.lognormal(size=400), "c": rng.normal(size=400)})
    y = (X["a"] + 0.5 * np.log1p(X["b"]) + rng.normal(0, 0.5, 400) > 0.3).astype(int)

    Xt = X.copy()
    Xt["b"] = np.log1p(Xt["b"])
    scaler = StandardScaler().fit(Xt)
    clf = LogisticRegression().fit(scaler.transform(Xt), y)
    model = {
        "feature_order": ["a", "b", "c"], "log1p_cols": ["b"],
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "decision_threshold": 0.5,
    }
    ours = apply_meta_model(X[["c", "b", "a"]], model)  # column order must not matter
    theirs = clf.predict_proba(scaler.transform(Xt))[:, 1]
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=0, atol=1e-12)


def test_frozen_meta_model_propagates_nan():
    model = {"feature_order": ["a"], "log1p_cols": [], "scaler_mean": [0.0], "scaler_scale": [1.0],
             "coef": [1.0], "intercept": 0.0, "decision_threshold": 0.5}
    out = apply_meta_model(pd.DataFrame({"a": [0.0, np.nan]}), model)
    assert out.iloc[0] == pytest.approx(0.5) and np.isnan(out.iloc[1])
