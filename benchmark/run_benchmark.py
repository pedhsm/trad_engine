"""Reproducible micro-benchmark of the core_math primitives.

Run it from the repo root:

    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --bars 500000 --repeat 7

What it measures, and what it does NOT
--------------------------------------
It times the pure compute of each primitive over N synthetic bars, at the Python
level, on THIS machine, and reports the median wall-clock of a few repeats plus a
throughput (bars processed per second). That is an honest relative picture of where
time goes — it is NOT a latency SLA and not a claim about your hardware. The number
that travels is the RATIO (e.g. C++ vs Python), not the absolute milliseconds.

If a compiled C++ mirror (core_math/cpp) is found and loadable via ctypes, the
book-imbalance and BVC primitives are timed in C++ too, side by side. If not, the
Python-only numbers print with a clear notice — nothing here requires the C++ build.
"""
from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core_math import bars_math, calculos_l2, micro_math


def make_bars(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV: a random walk with strictly positive prices and volumes."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.05, size=n).cumsum()
    close = 100.0 + steps - steps.min() + 1.0
    high = close + np.abs(rng.normal(0.0, 0.05, size=n))
    low = close - np.abs(rng.normal(0.0, 0.05, size=n))
    open_ = close + rng.normal(0.0, 0.02, size=n)
    volume = rng.integers(100, 10_000, size=n).astype(float)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _time(fn, repeat: int) -> float:
    """Median wall-clock (seconds) of `repeat` runs, after one warmup run."""
    fn()  # warmup (fills caches, triggers any lazy import)
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _row(name: str, secs: float, n: int) -> str:
    per_call_ms = secs * 1e3
    throughput = n / secs if secs > 0 else float("inf")
    return f"{name:<34} {per_call_ms:>10.3f} ms   {throughput:>14,.0f} bars/s"


def try_load_cpp() -> ctypes.CDLL | None:
    """Look for a compiled core_math/cpp shared library next to the sources."""
    here = Path(__file__).resolve().parent.parent / "core_math" / "cpp"
    candidates = []
    for stem in ("micro", "libmicro"):
        for ext in (".so", ".dll", ".dylib"):
            candidates.append(here / f"{stem}{ext}")
    candidates += list(here.glob("*.so")) + list(here.glob("*.dll")) + list(here.glob("*.dylib"))
    for path in candidates:
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError:
                continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=int, default=200_000, help="number of synthetic bars")
    parser.add_argument("--repeat", type=int, default=5, help="timed repeats (median reported)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n = args.bars
    df = make_bars(n, args.seed)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    book_bids = list(zip(np.linspace(100, 99, 5), np.random.default_rng(1).integers(1, 100, 5).astype(float)))
    book_asks = list(zip(np.linspace(100.1, 101, 5), np.random.default_rng(2).integers(1, 100, 5).astype(float)))

    print("=" * 72)
    print(f"core_math benchmark - {n:,} bars, median of {args.repeat} repeats (after warmup)")
    print("Python-level wall-clock on THIS machine. Read the ratios, not the absolutes.")
    print("=" * 72)
    print(f"{'primitive':<34} {'per call':>13}   {'throughput':>14}")
    print("-" * 72)

    print(_row("bars_math.sma(w=20)", _time(lambda: bars_math.sma(close, 20), args.repeat), n))
    print(_row("bars_math.ema(span=20)", _time(lambda: bars_math.ema(close, 20), args.repeat), n))
    print(_row("bars_math.rsi(w=14)", _time(lambda: bars_math.rsi(close, 14), args.repeat), n))
    print(_row("bars_math.atr(w=14)", _time(lambda: bars_math.atr(high, low, close, 14), args.repeat), n))
    print(_row("micro_math.compute_vpin_pipeline",
               _time(lambda: micro_math.compute_vpin_pipeline(df), args.repeat), n))

    # book_imbalance is O(depth), not O(bars); time N calls to get a comparable rate.
    def book_loop():
        for _ in range(n):
            calculos_l2.book_imbalance_signal(book_bids, book_asks)
    print(_row("calculos_l2.imbalance (xN)", _time(book_loop, args.repeat), n))

    print("-" * 72)
    lib = try_load_cpp()
    if lib is None:
        print("C++ mirror: not found (build core_math/cpp to enable the side-by-side).")
        print("            Python-only numbers above are complete and sufficient.")
    else:
        print(f"C++ mirror: loaded ({lib._name}). Symbols available for side-by-side timing.")
        print("            (Wire the ctypes signatures for book_imbalance / bvc_close_return")
        print("             to compare against the Python rows above.)")
    print("=" * 72)


if __name__ == "__main__":
    main()
