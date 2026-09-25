"""Point-in-time (anti-lookahead) invariants — reusable, strategy-agnostic.

Why these exist, separate from a parity test
--------------------------------------------
A parity test proves that a port CLONES its reference — it says nothing about whether
the reference is CORRECT. A causal bug (a feature that peeks at the future, an event
stamped before the data that generated it) survives every parity test, because the
clone reproduces the bug faithfully. These checks test CORRECTNESS instead of equality.

Each checker is a pure function returning ``(ok: bool, detail: str)`` — easy to use in
pytest (``assert ok, detail``) and as a programmatic gate (do not promote a strategy to
production if any PIT invariant fails).

Vocabulary: ``t`` is the decision instant. The mother rule is that every quantity used
to decide at ``t`` may depend only on data timestamped ``<= t``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


def assert_no_event_before_first_data(
    events_index: pd.DatetimeIndex, data_index: pd.DatetimeIndex
) -> tuple[bool, str]:
    """PIT-0 — coarse sanity: no decision may exist before the first data point.

    Cheap, and it catches the most obvious pathological case (an off-by-a-window
    stamp that lands events before the sample even begins).
    """
    if len(events_index) == 0:
        return True, "ok: no events"
    first = data_index.min()
    early = events_index[events_index < first]
    if len(early) > 0:
        return False, f"{len(early)} event(s) before first data point ({first}); e.g. {early[0]}"
    return True, "ok"


def assert_events_have_causal_support(
    events_index: pd.DatetimeIndex,
    obs_index: pd.DatetimeIndex,
    window_min: float,
    min_obs: int,
) -> tuple[bool, str]:
    """PIT-1 — an event cannot be stamped BEFORE the observations that produced it.

    A detector that fires because ``>= min_obs`` observations occurred inside a window
    should, if the event is stamped at the END of that window, have those observations
    fall in ``(t - window_min, t]``. If the event is stamped at the window's LEFT edge
    (a classic resample lookahead), the generating observations fall in the FUTURE and
    the retrospective window comes up short. This discriminates best on data that goes
    quiet and then bursts; in dense, continuous flow both stampings have observations
    behind them and the test cannot separate them.
    """
    oi = obs_index.sort_values()
    w = pd.Timedelta(minutes=window_min)
    for ts in events_index:
        lo = oi.searchsorted(ts - w, side="right")
        hi = oi.searchsorted(ts, side="right")
        if (hi - lo) < min_obs:
            return False, (
                f"event {ts} has only {hi - lo} observation(s) in (t-{window_min}min, t]; "
                f"expected >= min_obs={min_obs}. Symptom of a stamp placed BEFORE the "
                f"generating observations (left-edge resample lookahead)."
            )
    return True, f"ok: all {len(events_index)} events have observation support <= t"


def assert_entry_not_before_event(
    barrier_df: pd.DataFrame,
    event_col: str = "event_ts",
    entry_col: str = "t_start",
) -> tuple[bool, str]:
    """PIT-2 — the entry timestamp must not precede the event that triggered it.

    Pairs with the triple-barrier output (``core_math.labeling.apply_triple_barrier``),
    where the fill uses the first quote ``>= event_ts``. An entry stamped before its
    event is fill-side lookahead.
    """
    if barrier_df.empty:
        return True, "ok: no barriers"
    if event_col not in barrier_df or entry_col not in barrier_df:
        return False, f"barrier_df missing columns '{event_col}'/'{entry_col}'"
    viol = barrier_df[barrier_df[entry_col] < barrier_df[event_col]]
    if len(viol) > 0:
        return False, f"{len(viol)} entr(y/ies) with {entry_col} < {event_col} (fill lookahead)"
    return True, "ok"


def assert_feature_no_future_dependence(
    pipeline_fn: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    numeric_cols: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> tuple[bool, str]:
    """PIT-3 — end-to-end causality, implementation-agnostic and the strongest check.

    Perturbing the FUTURE (rows timestamped ``> cutoff``) must not change any output row
    at ``<= cutoff``. ``pipeline_fn(data) -> DataFrame`` runs your whole pipeline (event
    detection + features, or signal generation) and returns one row per decision, indexed
    by timestamp. If a row at ``t <= cutoff`` changes when only the future was altered,
    something in that row depends on the future — a leak. This catches lookahead without
    knowing anything about how the pipeline is implemented.

    ``numeric_cols`` selects which columns of ``data`` to perturb; default: all numeric.
    The perturbation both shuffles and rescales the future values, so it disturbs level
    and ordering at once.
    """
    rng = np.random.default_rng(seed)
    base = pipeline_fn(data)
    base = base[base.index <= cutoff]
    if base.empty:
        return True, "ok: no decisions <= cutoff to check (weak fixture)"

    cols = list(numeric_cols) if numeric_cols is not None else list(
        data.select_dtypes(include=[np.number]).columns
    )
    if not cols:
        return False, "no numeric columns in `data` to perturb"

    perturbed = data.copy()
    fut = perturbed.index > cutoff
    if fut.sum() == 0:
        return True, "ok: no future rows beyond cutoff to perturb (weak fixture)"
    for c in cols:
        vals = perturbed.loc[fut, c].to_numpy(dtype=float)
        perturbed.loc[fut, c] = rng.permutation(vals) * 1.5 + 1.0

    pert = pipeline_fn(perturbed).reindex(base.index)

    common = [c for c in base.columns if c in pert.columns]
    diff = ~np.isclose(
        base[common].to_numpy(dtype=float),
        pert[common].to_numpy(dtype=float),
        equal_nan=True,
    )
    if diff.any():
        i, j = np.argwhere(diff)[0]
        return False, (
            f"output '{common[j]}' at decision {base.index[i]} (<= cutoff) changed when "
            f"ONLY the future was perturbed: {base[common].iloc[i, j]} -> "
            f"{pert[common].iloc[i, j]}. Future dependence = lookahead leak."
        )
    return True, f"ok: {len(base)} decisions <= cutoff are immune to future perturbation"


def assert_trade_sign_matches_price(
    trades_df: pd.DataFrame,
    mid: pd.Series,
    price_col: str = "price",
    sign_col: str = "signed_vol",
    tol: float = 0.6,
) -> tuple[bool, str]:
    """PIT-4 — the sign of your signed volume must agree with the real aggressor side.

    Via ASOF (mid quote in force ``<= trade timestamp``): a trade executed ABOVE the mid
    is buyer-initiated (sign should be > 0); below, seller-initiated (< 0). If the
    majority disagree, the data-vendor's aggressor convention was mapped inverted — a
    silent sign flip that quietly reverses every microstructure feature built on it.
    """
    if price_col not in trades_df:
        return False, f"trades_df has no '{price_col}' column for the ASOF"
    if sign_col not in trades_df:
        return False, f"trades_df has no '{sign_col}' column"
    tr = trades_df.sort_index()
    mid_at = mid.sort_index().reindex(tr.index, method="ffill")
    aggressive = tr[price_col].to_numpy() != mid_at.to_numpy()
    price_side = np.sign(tr[price_col].to_numpy() - mid_at.to_numpy())
    vol_side = np.sign(tr[sign_col].to_numpy())
    mask = aggressive & (price_side != 0) & (vol_side != 0)
    if mask.sum() == 0:
        return True, "ok: no identifiable aggressive trades (nothing to check)"
    agree = float((price_side[mask] == vol_side[mask]).mean())
    if agree < tol:
        return False, (
            f"aggressor sign agrees with price on only {agree:.1%} of trades "
            f"(< {tol:.0%}). Convention likely inverted."
        )
    return True, f"ok: aggressor sign agrees on {agree:.1%} of trades"
