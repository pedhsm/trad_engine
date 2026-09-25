"""Basic bar-level feature primitives for 1-min+ strategies (Python reference).

Building blocks, NOT strategies: a strategy is composed FROM these. Every
function is pure (array in, array out), stateless, and causal — the value at t
uses only data up to t, never the future. This is the Python half of core_math;
the C++ half mirrors the microstructure side and is kept in lockstep with it by
parity tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _s(x) -> pd.Series:
    """Coerce ndarray/Series into a float Series without carrying an index we would
    then have to reconcile — position is what the primitives care about."""
    if isinstance(x, pd.Series):
        return x.astype(float).reset_index(drop=True)
    return pd.Series(np.asarray(x, dtype=float))


def _out(s: pd.Series, like) -> np.ndarray:
    return s.to_numpy()


def returns(close) -> np.ndarray:
    """Simple returns C_t/C_{t-1} - 1; first element NaN."""
    return _out(_s(close).pct_change(), close)


def log_returns(close) -> np.ndarray:
    """Log returns log(C_t/C_{t-1}); first element NaN."""
    c = _s(close)
    return _out(np.log(c / c.shift(1)), close)


def sma(x, w: int) -> np.ndarray:
    """Simple moving average over the trailing window of length w."""
    return _out(_s(x).rolling(int(w)).mean(), x)


def ema(x, span: int) -> np.ndarray:
    """Exponential moving average (pandas ewm, adjust=False), span = smoothing."""
    return _out(_s(x).ewm(span=int(span), adjust=False).mean(), x)


def rolling_std(x, w: int, ddof: int = 1) -> np.ndarray:
    """Trailing rolling standard deviation (sample, ddof=1 by default)."""
    return _out(_s(x).rolling(int(w)).std(ddof=ddof), x)


def realized_vol(close, w: int) -> np.ndarray:
    """Rolling std of log returns over w bars — a causal volatility estimate."""
    lr = _s(close)
    lr = np.log(lr / lr.shift(1))
    return _out(lr.rolling(int(w)).std(ddof=1), close)


def zscore(x, w: int) -> np.ndarray:
    """Trailing z-score: (x_t - rolling_mean) / rolling_std(ddof=1)."""
    s = _s(x)
    m = s.rolling(int(w)).mean()
    sd = s.rolling(int(w)).std(ddof=1)
    return _out((s - m) / sd, x)


def rolling_rank(x, w: int) -> np.ndarray:
    """Percentile rank (0..1) of the LAST point within its trailing window of w."""
    def _rank(a: np.ndarray) -> float:
        last = a[-1]
        return float((a <= last).sum() - 1) / float(len(a) - 1) if len(a) > 1 else 0.5
    return _out(_s(x).rolling(int(w)).apply(_rank, raw=True), x)


def true_range(high, low, close) -> np.ndarray:
    """True range: max(H-L, |H-C_{t-1}|, |L-C_{t-1}|); first element = H-L."""
    h, l, c = _s(high), _s(low), _s(close)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = (h.iloc[0] - l.iloc[0])
    return _out(tr, close)


def atr(high, low, close, w: int) -> np.ndarray:
    """Average true range: rolling mean of true range over w bars."""
    tr = pd.Series(true_range(high, low, close))
    return _out(tr.rolling(int(w)).mean(), close)


def rolling_min(x, w: int) -> np.ndarray:
    """Trailing rolling minimum over window w."""
    return _out(_s(x).rolling(int(w)).min(), x)


def rolling_max(x, w: int) -> np.ndarray:
    """Trailing rolling maximum over window w."""
    return _out(_s(x).rolling(int(w)).max(), x)


def rsi(close, w: int = 14) -> np.ndarray:
    """Wilder-style RSI over w bars (SMA of gains/losses; 0..100)."""
    c = _s(close)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(int(w)).mean()
    avg_loss = loss.rolling(int(w)).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)   # all-gains window -> RSI 100
    out[avg_gain.isna()] = np.nan              # keep warmup NaN
    return _out(out, close)
