"""End-to-end paper-trading demo — no broker, no market data account, no network.

Pipeline (the whole point of this file is that it is end-to-end):

    synthetic trade feed  ->  BarAggregator  ->  SMA-cross strategy  ->  simulated
    market fills          ->  sqlite journal ->  printed summary

Everything is deterministic (fixed RNG seed), so two runs print the same numbers.
Run it from the repo root:

    python -m examples.paper_demo

It exists to prove the plumbing works together, NOT to show an edge — the SMA-cross
strategy is a textbook placeholder. Bring your own signal.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np

from core_math.bars_math import sma
from live.bar_aggregator import BarAggregator

# --- knobs ------------------------------------------------------------------
SEED = 7
N_MINUTES = 2_000          # how many 1-min bars to synthesize
TRADES_PER_MINUTE = 5      # trades generated inside each minute
FAST_WINDOW = 10           # fast SMA (bars)
SLOW_WINDOW = 40           # slow SMA (bars)
START_PRICE = 100.0
COST_BPS = 1.0             # round-trip-ish cost charged whenever the position flips
JOURNAL_PATH = "paper_demo_journal.db"


def synth_trades(rng: np.random.Generator):
    """Yield (timestamp_us, price, size) for a Gaussian random walk.

    A random walk has, by construction, no predictable edge — that is deliberate:
    it keeps the demo honest. The price wanders; the strategy has nothing real to
    catch. We only want to see the machinery run end to end.
    """
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    price = START_PRICE
    for minute in range(N_MINUTES):
        for k in range(TRADES_PER_MINUTE):
            price = max(0.01, price + rng.normal(0.0, 0.02))
            # spread trades across the minute so buckets fill realistically
            ts_us = (base_ts + minute * 60 + int(k * 60 / TRADES_PER_MINUTE)) * 1_000_000
            size = float(rng.integers(1, 10))
            yield ts_us, price, size


def build_bars(rng: np.random.Generator):
    """Feed synthetic trades through the BarAggregator and collect closed bars."""
    agg = BarAggregator(ticker="DEMO", interval_s=60)
    bars = []
    for ts_us, price, size in synth_trades(rng):
        bars.extend(agg.add_trade(ts_us, price, size))
    return bars, agg


def sma_cross_positions(closes: np.ndarray) -> np.ndarray:
    """+1 when fast SMA > slow SMA, -1 when below, 0 during warmup (NaN).

    The position is then shifted forward by one bar in the caller, so the decision
    taken at the close of bar t only takes effect on the move from t to t+1 — never
    on the move that produced it (that would be lookahead).
    """
    fast = sma(closes, FAST_WINDOW)
    slow = sma(closes, SLOW_WINDOW)
    pos = np.where(fast > slow, 1.0, -1.0)
    pos[np.isnan(fast) | np.isnan(slow)] = 0.0
    return pos


def simulate(bars, journal: sqlite3.Connection):
    """Mark-to-market simulation with fills at the bar close. Returns per-bar
    strategy returns (numpy array) and the trade count (position flips)."""
    closes = np.array([b.close for b in bars], dtype=float)
    raw_pos = sma_cross_positions(closes)

    # Causal: hold, on the move t-1 -> t, the position decided at t-1.
    held = np.concatenate([[0.0], raw_pos[:-1]])

    price_ret = np.concatenate([[0.0], np.diff(closes)])   # absolute price move per bar
    gross = held * price_ret

    # A "trade" is a change in target position; charge a cost when it flips.
    flips = np.abs(np.diff(np.concatenate([[0.0], held])))
    costs = flips * (COST_BPS / 10_000.0) * closes
    net = gross - costs

    cur = journal.cursor()
    for i, b in enumerate(bars):
        if flips[i] > 0:
            side = "LONG" if held[i] > 0 else ("SHORT" if held[i] < 0 else "FLAT")
            cur.execute(
                "INSERT INTO fills (ts, side, price, target_position) VALUES (?,?,?,?)",
                (b.timestamp.isoformat(), side, float(b.close), float(held[i])),
            )
    journal.commit()

    n_trades = int((flips > 0).sum())
    return net, gross, n_trades


def open_journal(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS fills")
    conn.execute(
        "CREATE TABLE fills (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, "
        "side TEXT, price REAL, target_position REAL)"
    )
    return conn


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 66)
    print("trad_engine - end-to-end paper-trading demo")
    print("=" * 66)
    print(
        "What you are about to see: a synthetic random-walk feed is turned into\n"
        "1-min bars, an SMA-cross strategy takes causal positions on them, fills\n"
        "are simulated at the bar close, and each fill is written to a sqlite\n"
        "journal. The summary at the end reports:\n"
        "  - bars / trades : size of the run and how often the position flipped\n"
        "  - gross vs net PnL : price PnL before and after a per-flip cost\n"
        "  - win rate : share of bars with positive strategy return\n"
        "  - per-bar Sharpe : mean/std of per-bar returns (NOT annualized)\n"
        "The feed is a random walk on purpose: there is no edge to find here, so\n"
        "a PnL near zero (minus costs) is the EXPECTED, honest result.\n"
    )

    bars, agg = build_bars(rng)
    print(f"Aggregator stats: {agg.stats()}")

    journal = open_journal(JOURNAL_PATH)
    net, gross, n_trades = simulate(bars, journal)
    n_fills = journal.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    journal.close()

    wins = int((net > 0).sum())
    active = int((net != 0).sum())
    win_rate = (wins / active * 100.0) if active else 0.0
    sharpe = (net.mean() / net.std()) if net.std() > 0 else 0.0

    print("-" * 66)
    print(f"Bars produced        : {len(bars)}")
    print(f"Position flips (trades): {n_trades}   (fills journaled: {n_fills})")
    print(f"Gross PnL (price pts): {gross.sum():+.4f}")
    print(f"Net PnL   (price pts): {net.sum():+.4f}")
    print(f"Win rate (active bars): {win_rate:.1f}%")
    print(f"Per-bar Sharpe        : {sharpe:+.4f}  (not annualized)")
    print(f"Journal written to    : {JOURNAL_PATH}")
    print("-" * 66)
    print("Plumbing OK. Swap the SMA-cross for your own signal to make it interesting.")


if __name__ == "__main__":
    main()
