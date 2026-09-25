"""
Microstructure primitives: BVC-style signed volume, constant-volume buckets, VPIN, TIB.

Pure compute (arrays/DataFrames in, DataFrame/Series out) — no data loading, no I/O.
Feed it bars from whatever source you use; it never reaches for a database.

References (informal): Easley, Lopez de Prado & O'Hara (VPIN); bulk-volume
classification via price pressure mapped through a Gaussian CDF (a research-grade
proxy for order-flow sign when you only have L1/OHLCV bars, not signed trades).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


def bulk_volume_classification(
    close: np.ndarray,
    open_: np.ndarray | None = None,
    *,
    mode: Literal["close_return", "close_minus_open", "close_close_lag1"] = "close_return",
    vol_window: int = 30,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate buy / sell volume per bar using Phi(z) on a standardized price move.

    - close_return: z_t = log_ret_t / rolling_std(log_ret), log_ret_t = log(C_t/C_{t-1})
    - close_minus_open: z_t = (C_t - O_t) / rolling_std(C - O)
    - close_close_lag1: z_t = (C_t - C_{t-1}) / rolling_std(first-differenced close)
    """
    close = np.asarray(close, dtype=float)

    if mode == "close_return":
        lr = np.diff(np.log(np.clip(close, eps, None)), prepend=np.nan)
        lr[0] = 0.0
        sig = (
            pd.Series(lr)
            .rolling(vol_window, min_periods=max(3, vol_window // 3))
            .std()
            .to_numpy()
        )
        sig = np.where(np.isfinite(sig) & (sig > eps), sig, eps)
        z = lr / sig
    elif mode == "close_minus_open":
        if open_ is None:
            raise ValueError("close_minus_open mode requires open_")
        open_ = np.asarray(open_, dtype=float)
        co = close - open_
        sig = (
            pd.Series(co)
            .rolling(vol_window, min_periods=max(3, vol_window // 3))
            .std()
            .to_numpy()
        )
        sig = np.where(np.isfinite(sig) & (sig > eps), sig, eps)
        z = co / sig
    elif mode == "close_close_lag1":
        d = close - np.roll(close, 1)
        d[0] = 0.0
        sig = (
            pd.Series(d)
            .rolling(vol_window, min_periods=max(3, vol_window // 3))
            .std()
            .to_numpy()
        )
        sig = np.where(np.isfinite(sig) & (sig > eps), sig, eps)
        z = d / sig
    else:
        raise ValueError(mode)

    p_buy = norm.cdf(np.clip(z, -8.0, 8.0))
    v_frac_buy = p_buy
    v_frac_sell = 1.0 - p_buy
    return v_frac_buy, v_frac_sell


def volume_buckets_from_bars(
    volume: np.ndarray,
    v_buy_frac: np.ndarray,
    v_sell_frac: np.ndarray,
    bucket_volume: float,
    bar_ts: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Aggregate bars into constant-volume buckets (fractional splits when a bar spans buckets).

    Each completed bucket has total volume ~= ``bucket_volume``.
    If ``bar_ts`` is given (same length as ``volume``), column ``bucket_end_ts`` is the bar
    timestamp when each bucket closed (convention: end of the active bar).
    """
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be positive")
    vol = np.asarray(volume, dtype=float)
    fb = np.asarray(v_buy_frac, dtype=float)
    fs = np.asarray(v_sell_frac, dtype=float)
    tot = fb + fs
    ok = tot > 1e-15
    fb = np.where(ok, fb / tot, 0.5)
    fs = np.where(ok, fs / tot, 0.5)

    rows: list[tuple[float, float, float]] = []
    bucket_ts: list[pd.Timestamp] = []
    buf_b = buf_s = 0.0
    target = float(bucket_volume)
    use_ts = bar_ts is not None
    if use_ts and len(bar_ts) != len(vol):
        raise ValueError("bar_ts must have same length as volume")

    for i in range(vol.shape[0]):
        t_end = pd.Timestamp(bar_ts[i]) if use_ts else None
        vb = vol[i] * fb[i]
        vs = vol[i] * fs[i]
        rem_b, rem_s = vb, vs
        while rem_b + rem_s > 1e-12:
            cur = buf_b + buf_s
            space = target - cur
            if space <= 1e-12:
                rows.append((buf_b, buf_s, buf_b + buf_s))
                if use_ts:
                    bucket_ts.append(t_end)
                buf_b = buf_s = 0.0
                continue
            take = min(space, rem_b + rem_s)
            share = take / (rem_b + rem_s) if (rem_b + rem_s) > 0 else 0.0
            d_b = rem_b * share
            d_s = rem_s * share
            buf_b += d_b
            buf_s += d_s
            rem_b -= d_b
            rem_s -= d_s
            if buf_b + buf_s >= target - 1e-9:
                rows.append((buf_b, buf_s, buf_b + buf_s))
                if use_ts:
                    bucket_ts.append(t_end)
                buf_b = buf_s = 0.0

    if not rows:
        cols = ["v_buy", "v_sell", "v_total", "abs_imbalance", "dir_imbalance"]
        if use_ts:
            cols.append("bucket_end_ts")
        return pd.DataFrame(columns=cols)

    arr = np.asarray(rows, dtype=float)
    buy_, sell_, tot_ = arr[:, 0], arr[:, 1], arr[:, 2]
    imb = np.abs(buy_ - sell_) / np.clip(tot_, 1e-12, None)
    dir_imb = (buy_ - sell_) / np.clip(tot_, 1e-12, None)
    out = pd.DataFrame(
        {"v_buy": buy_, "v_sell": sell_, "v_total": tot_, "abs_imbalance": imb, "dir_imbalance": dir_imb}
    )
    if use_ts:
        out["bucket_end_ts"] = bucket_ts
    return out


def vpin_from_buckets(
    bucket_df: pd.DataFrame,
    n_buckets: int,
) -> pd.Series:
    """
    VPIN: rolling mean of |V_buy - V_sell| / V_total over the last ``n_buckets`` buckets.
    """
    if bucket_df.empty or n_buckets < 1:
        return pd.Series(dtype=float)
    x = bucket_df["abs_imbalance"].astype(float)
    return x.rolling(n_buckets, min_periods=1).mean()


def tib_from_buckets(
    bucket_df: pd.DataFrame,
    n_buckets: int,
) -> pd.Series:
    """
    TIB (Trade Imbalance): rolling mean of (V_buy - V_sell) / V_total over the last ``n_buckets`` buckets.
    """
    if bucket_df.empty or n_buckets < 1:
        return pd.Series(dtype=float)
    x = bucket_df["dir_imbalance"].astype(float)
    return x.rolling(n_buckets, min_periods=1).mean()


def compute_vpin_pipeline(
    df: pd.DataFrame,
    *,
    bucket_volume: float = 50_000.0,
    vpin_window_buckets: int = 50,
    bvc_mode: Literal["close_return", "close_minus_open", "close_close_lag1"] = "close_return",
    bvc_vol_window: int = 30,
) -> pd.DataFrame:
    """
    End-to-end: per-bar BVC fractions -> volume buckets -> VPIN / TIB series (aligned to bucket index).

    ``df`` needs columns ``volume`` and ``close`` (``open`` and ``ts`` optional).
    """
    if df.empty:
        return pd.DataFrame()
    vol = df["volume"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float) if "open" in df.columns else None
    fb, fs = bulk_volume_classification(
        close,
        open_,
        mode=bvc_mode,
        vol_window=bvc_vol_window,
    )
    ts_series = df["ts"] if "ts" in df.columns else None
    buckets = volume_buckets_from_bars(
        vol, fb, fs, bucket_volume, bar_ts=ts_series,
    )
    if buckets.empty:
        cols = ["v_buy", "v_sell", "v_total", "abs_imbalance", "dir_imbalance", "vpin", "tib"]
        if ts_series is not None:
            cols.append("bucket_end_ts")
        return pd.DataFrame(columns=cols)
    vpin = vpin_from_buckets(buckets, vpin_window_buckets)
    tib = tib_from_buckets(buckets, vpin_window_buckets)
    out = buckets.copy()
    out["vpin"] = vpin.to_numpy()
    out["tib"] = tib.to_numpy()
    return out


# ---------------------------------------------------------------------------
# Public API aliases
# ---------------------------------------------------------------------------
compute_vpin = vpin_from_buckets
compute_tib = tib_from_buckets
