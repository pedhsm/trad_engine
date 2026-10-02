"""Event labeling primitives (the triple-barrier method).

Generic, strategy-agnostic labeling: given a set of event timestamps each carrying a
bet direction (+1 long / -1 short), walk price forward and label whether the profit
barrier, the stop barrier, or the time (vertical) barrier was hit first. This is the
Lopez de Prado triple-barrier method; it assumes nothing about how the events were
generated -- plug in your own signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_triple_barrier(
    events_df: pd.DataFrame,
    mid: pd.Series,
    profit_bps: float,
    stop_bps: float,
    max_holding_min: float,
) -> pd.DataFrame:
    """Label each event with the triple-barrier method.

    ``events_df`` needs index=timestamp and a ``bet_direction`` column (+1/-1),
    already resolved by the caller. This function assumes neither direction.

    Entry price = first ``mid`` quote timestamped >= the event ts (searchsorted,
    never an earlier quote -- avoids entry lookahead). Walks forward to the vertical
    barrier (max_holding_min) looking for which profit/stop barrier is touched
    first, using only quotes timestamped <= the vertical barrier. If none is touched,
    label=0 (timeout) and t_end = the last quote at or before the vertical barrier
    (or the last available quote, if the data ends first).

    Returns: event_ts (the raw event timestamp, before it is snapped to the entry
    quote), t_start (real entry timestamp), bet_direction, label (+1 profit /
    -1 stop / 0 timeout), t_end, realized_ret.
    """
    mid = mid.sort_index()
    rows = []
    for ts, row in events_df.iterrows():
        d = row["bet_direction"]
        idx_entry = mid.index.searchsorted(ts)
        if idx_entry >= len(mid):
            continue
        entry_ts = mid.index[idx_entry]
        entry_price = mid.iloc[idx_entry]

        if idx_entry == len(mid) - 1:
            continue  # no quote after the entry: nothing to label
        vertical_ts = ts + pd.Timedelta(minutes=max_holding_min)
        # Last quote AT OR BEFORE the vertical barrier. A quote after it is outside
        # the holding period: letting it hit a barrier would label the event with a
        # move the position was never allowed to wait for.
        idx_end = mid.index.searchsorted(vertical_ts, side="right") - 1
        if idx_end < idx_entry:
            continue  # the first quote after the event is already past the horizon
        # idx_end == idx_entry (no new quote inside the horizon) is a timeout at the
        # entry price, not a skip: dropping it would bias the sample against quiet
        # periods.
        window = mid.iloc[idx_entry:idx_end + 1]

        profit_price = entry_price * (1 + d * profit_bps / 10000.0)
        stop_price = entry_price * (1 - d * stop_bps / 10000.0)
        if d > 0:
            hit_profit = window[window >= profit_price]
            hit_stop = window[window <= stop_price]
        else:
            hit_profit = window[window <= profit_price]
            hit_stop = window[window >= stop_price]

        t_profit = hit_profit.index[0] if len(hit_profit) else None
        t_stop = hit_stop.index[0] if len(hit_stop) else None
        # Capture the price alongside the timestamp (re-looking up via mid.loc[t_end]
        # later would return an ambiguous Series on a duplicated index, not a scalar).
        p_profit = hit_profit.iloc[0] if len(hit_profit) else None
        p_stop = hit_stop.iloc[0] if len(hit_stop) else None

        if t_profit is not None and t_stop is not None and t_profit == t_stop:
            # Exact timestamp tie (e.g. a large price gap, or a duplicated tick,
            # crossing both barriers at the same instant) -- the data's granularity
            # cannot say which was hit first. Assume the WORST case (stop), not the
            # best -- avoids a systematic optimistic labeling bias.
            t_end, label, exit_price = t_stop, -1, p_stop
        elif t_profit is not None and (t_stop is None or t_profit < t_stop):
            t_end, label, exit_price = t_profit, 1, p_profit
        elif t_stop is not None:
            t_end, label, exit_price = t_stop, -1, p_stop
        else:
            t_end, label, exit_price = window.index[-1], 0, window.iloc[-1]

        realized_ret = float(np.log(exit_price / entry_price))
        rows.append({
            "event_ts": ts, "t_start": entry_ts, "bet_direction": d, "label": label,
            "t_end": t_end, "realized_ret": realized_ret,
        })

    return pd.DataFrame(rows, columns=["event_ts", "t_start", "bet_direction", "label",
                                       "t_end", "realized_ret"])
