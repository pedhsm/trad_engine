"""Tick-to-order latency of the live path, measured component by component.

    python -m benchmark.latency
    python -m benchmark.latency --n 20000

    broker callback --(A)--> ZMQ publish --(C: ZMQ + (B) + ZMQ)--> engine receives the target
                             `------ strategy process: decode, bar, signal, encode ------'

(A) ENGINE HOT PATH (C++): trade callback -> record in the arena -> publish on ZMQ.
    Needs a C++ compiler, lib/client (IB API) and ZeroMQ; skipped otherwise.
(B1) STRATEGY FRAMEWORK (Python, in process): decode the TradeUpdate, close a bar,
    encode the target. No sockets, trivial signal.
(B2) EXAMPLE SIGNAL: the SMA-cross's on_bar alone — the part YOUR signal replaces.
(C) ROUND TRIP (Python, two processes): a fake engine publishes a trade; the real
    examples/live_client.py, in ANOTHER process, turns it into a target position and
    sends it back. Timed from "trade sent" to "target received" — i.e. (B1) plus two
    localhost ZMQ hops plus the process wake-ups.

Every trade here closes a bar and flips the target, so each one produces an order:
this is the WORST case per tick, not the average.

What it does NOT measure: the broker (IB Gateway <-> exchange) and the network,
which are milliseconds and dominate everything above. Read these numbers as "the
engine's own contribution", on THIS machine.

Percentiles: p50 is the median tick; p99 is the latency only 1 tick in 100 exceeds.
The tail is what matters in trading: bursts are when latency spikes and when it
costs the most.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from examples.live_client import Endpoints, LiveClient
from live import ipc
from live.bar_aggregator import BarAggregator

ROOT = Path(__file__).resolve().parents[1]
MIN_US = 60_000_000
T0_US = 1_704_103_200 * 1_000_000  # 2024-01-01 10:00 UTC


def summarize(samples_ns) -> dict:
    a = np.asarray(samples_ns, dtype=float) / 1e3  # -> microseconds
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)),
            "p999": float(np.percentile(a, 99.9)), "max": float(a.max())}


def _row(name: str, s: dict) -> str:
    return (f"{name:<34} {s['p50']:>9.1f} {s['p99']:>9.1f} {s['p999']:>9.1f} {s['max']:>10.1f}"
            f"   (n={s['n']:,})")


class FlipStrategy:
    """Flips the target on every bar: every bar becomes an order (worst case)."""
    def __init__(self):
        self.target = -1

    def on_bar(self, bar):
        self.target = -self.target
        return self.target


# --- (B) strategy compute, in process -------------------------------------------

def bench_strategy_compute(n: int) -> tuple[dict, dict]:
    """(B1) framework per tick: decode + close a bar + encode, with the trivial
    FlipStrategy (the same one the round trip uses); (B2) the example SMA-cross
    signal alone, on the same bars. Your own signal replaces (B2)."""
    from examples.live_client import SmaCross

    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.05, n + 200))
    payloads = [ipc.encode_trade(T0_US + i * MIN_US, float(p), 1) for i, p in enumerate(prices)]

    agg = BarAggregator(interval_s=60, discard_boot_bar=False)
    flip = FlipStrategy()
    framework = []
    bars = []
    for i, payload in enumerate(payloads):
        t0 = time.perf_counter_ns()
        t = ipc.decode_trade(payload)
        for bar in agg.add_trade(t.timestamp_us, t.price, t.size):
            ipc.encode_target_position(1, flip.on_bar(bar))
            bars.append(bar)
        dt = time.perf_counter_ns() - t0
        if i >= 200:
            framework.append(dt)

    sma = SmaCross(fast=10, slow=40)
    signal = []
    for i, bar in enumerate(bars):
        t0 = time.perf_counter_ns()
        sma.on_bar(bar)
        dt = time.perf_counter_ns() - t0
        if i >= 200:  # warmup: fill the SMA window first
            signal.append(dt)
    return summarize(framework), summarize(signal)


# --- (C) round trip across processes -----------------------------------------------

def _client_process(endpoints: Endpoints, ready: mp.Event, stop: mp.Event) -> None:
    client = LiveClient("BENCH", ticker_id=1, endpoints=endpoints, strategy=FlipStrategy(),
                        send_orders=True)
    import logging
    logging.disable(logging.CRITICAL)  # printing per tick would dominate the measurement
    ready.set()
    while not stop.is_set():
        client.step(10)
    client.close()


def bench_round_trip(n: int) -> dict:
    import zmq

    ctx = zmq.Context()
    pub, pull, exec_pub, hb_pub, hb_sub = (ctx.socket(t) for t in
                                           (zmq.PUB, zmq.PULL, zmq.PUB, zmq.PUB, zmq.SUB))
    hb_sub.setsockopt(zmq.SUBSCRIBE, b"")
    eps = []
    for s in (pub, pull, exec_pub, hb_pub, hb_sub):
        s.setsockopt(zmq.LINGER, 0)
        s.bind("tcp://127.0.0.1:*")
        eps.append(s.getsockopt_string(zmq.LAST_ENDPOINT))

    mp_ctx = mp.get_context("spawn")
    ready, stop = mp_ctx.Event(), mp_ctx.Event()
    proc = mp_ctx.Process(target=_client_process, args=(Endpoints(*eps), ready, stop), daemon=True)
    proc.start()
    try:
        if not ready.wait(30):
            raise RuntimeError("client process did not come up")
        # Subscription handshake: keep sending minute-0 trades (same bucket, harmless)
        # for a moment, then minute 1 closes the boot bar (discarded, no order).
        for _ in range(20):
            pub.send_multipart([b"BENCH_TRADE", ipc.encode_trade(T0_US, 100.0, 1)])
            time.sleep(0.02)
        pub.send_multipart([b"BENCH_TRADE", ipc.encode_trade(T0_US + MIN_US, 100.0, 1)])
        time.sleep(0.2)

        samples = []
        for i in range(n + 200):
            minute = i + 2
            payload = ipc.encode_trade(T0_US + minute * MIN_US, 100.0 + (i % 10) * 0.01, 1)
            t0 = time.perf_counter_ns()
            pub.send_multipart([b"BENCH_TRADE", payload])
            if not pull.poll(5000):
                raise RuntimeError(f"no target position for trade {i}")
            pull.recv()
            dt = time.perf_counter_ns() - t0
            if i >= 200:
                samples.append(dt)
            # Fill it, so the client's one-order-in-flight lock opens for the next tick.
            exec_pub.send(ipc.encode_execution(100.0, 1, 1000 + i, ipc.STATUS_FILLED, 1, 0))
            time.sleep(0.0005)  # let the fill land before the next trade
        return summarize(samples)
    finally:
        stop.set()
        proc.join(10)
        for s in (pub, pull, exec_pub, hb_pub, hb_sub):
            s.close()
        ctx.term()


# --- (A) C++ engine hot path ---------------------------------------------------------

def bench_engine_hot_path(n: int) -> dict | None:
    """Returns None (with a reason printed) when the C++ side cannot be built here."""
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    client = ROOT / "lib" / "client"
    if cxx is None or not (client / "TwsApiL0.cpp").exists():
        print("  (A) skipped: needs a C++ compiler and lib/client (IB API).")
        return None
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / ("hot_path_bench" + (".exe" if sys.platform == "win32" else ""))
        cmd = [cxx, "-O2", "-std=c++17", "-DIB_USE_STD_STRING",
               "-I", str(ROOT / "cpp_engine" / "include"), "-I", str(client),
               str(ROOT / "benchmark" / "cpp" / "hot_path_bench.cpp"),
               str(ROOT / "cpp_engine" / "src" / "AllocatorArena.cpp"), str(client / "TwsApiL0.cpp"),
               "-o", str(exe), "-lzmq"]
        cmd += ["-DWIN32", "-lws2_32"] if sys.platform == "win32" else ["-pthread"]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            print("  (A) skipped: build failed (is ZeroMQ / cppzmq installed?).")
            return None
        run = subprocess.run([str(exe), str(n)], capture_output=True, text=True, cwd=d, timeout=600)
        stats = dict(line.split() for line in run.stdout.splitlines()
                     if line.split() and line.split()[0] in {"n", "p50", "p99", "p999", "max"})
        return {"n": int(stats["n"]), **{k: int(stats[k]) / 1e3 for k in ("p50", "p99", "p999", "max")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=5000, help="measured ticks per component")
    ap.add_argument("--skip-cpp", action="store_true", help="skip (A), which compiles the engine")
    args = ap.parse_args()

    print("=" * 84)
    print(f"tick-to-order latency, component by component - {args.n:,} ticks, microseconds")
    print("excludes the broker and the network; worst case: every tick becomes an order")
    print("=" * 84)
    print(f"{'component':<34} {'p50':>9} {'p99':>9} {'p99.9':>9} {'max':>10}")
    print("-" * 84)
    a = None if args.skip_cpp else bench_engine_hot_path(max(args.n, 100_000))
    if a:
        print(_row("(A) engine hot path (C++)", a))
    framework, signal = bench_strategy_compute(args.n)
    print(_row("(B1) strategy framework (Python)", framework))
    print(_row("(B2) example signal: SMA-cross", signal))
    print(_row("(C) round trip, 2 processes", bench_round_trip(args.n)))
    print("-" * 84)
    print("tick-to-order inside the machine ~= (A) + (C) + your signal's own cost (B2 here).")
    print("(C) runs the trivial flip strategy, so it contains (B1) but not (B2).")
    print("=" * 84)


if __name__ == "__main__":
    main()
