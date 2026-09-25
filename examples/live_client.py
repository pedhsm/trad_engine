"""Minimal live strategy for the C++ engine — the other end of the ZeroMQ protocol.

    engine  --<TICKER>_TRADE-->  BarAggregator -> SMA cross -> target position
            <--TargetPositionRequest--  (only when the target changes, one at a time)
            --ExecutionReport-->        (unlocks the next order)
            <--StrategyHeartbeat-- every second (silence -> engine flattens the book)

Run the engine first (see README), with the same config file:

    trad_engine --live --mode listen_only --config examples/engine_config.example.json
    python -m examples.live_client --config examples/engine_config.example.json --ticker SPY

DRY RUN BY DEFAULT: decisions are printed, no order is sent. Add --send-orders to
actually send target positions (the engine's risk gate still applies). The SMA cross is
a placeholder with no edge — the point is the plumbing and its safety rules:

- Target positions, not orders: the strategy says "I want +1", the engine computes the
  delta against the broker's position. Re-sending the same target is harmless.
- One order in flight: after sending, wait for a terminal ExecutionReport for this
  ticker before sending another target. Without it, a target sent before the broker
  confirms the first fill is computed against a stale position and doubles the trade.
- Heartbeat on a dedicated socket: the engine's watchdog measures silence there, not
  on the orders socket, so "no signal to send" never looks like "strategy dead".
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from live import ipc
from live.bar_aggregator import Bar, BarAggregator

log = logging.getLogger("live_client")


@dataclass
class Endpoints:
    data: str = "tcp://127.0.0.1:5555"
    orders: str = "tcp://127.0.0.1:5556"
    executions: str = "tcp://127.0.0.1:5557"
    engine_heartbeat: str = "tcp://127.0.0.1:5558"
    strategy_heartbeat: str = "tcp://127.0.0.1:5559"


@dataclass
class SmaCross:
    """Placeholder signal: +1 when the fast SMA is above the slow one, else -1."""
    fast: int = 10
    slow: int = 40
    closes: Deque[float] = field(default_factory=deque)

    def __post_init__(self):
        self.closes = deque(self.closes, maxlen=self.slow)  # keep only what the means use

    def on_bar(self, bar: Bar) -> Optional[int]:
        self.closes.append(bar.close)
        if len(self.closes) < self.slow:
            return None  # warmup: no opinion yet
        # Only the LAST value of each SMA is needed: the mean of the last `fast` and
        # `slow` closes, the same formula as core_math.bars_math.sma. Calling sma()
        # here (a whole-series pandas primitive, built for backtests) cost ~200 us
        # per bar in pandas overhead alone — see benchmark/latency.py (B2).
        c = np.fromiter(self.closes, dtype=float, count=len(self.closes))
        return 1 if c[-self.fast:].mean() > c[-self.slow:].mean() else -1


def ticker_id_from_config(config: dict, ticker: str) -> int:
    """The engine numbers assets 1..N in the order of config["tickers"]."""
    return config["tickers"].index(ticker) + 1


class LiveClient:
    def __init__(self, ticker: str, ticker_id: int, endpoints: Endpoints = Endpoints(),
                 strategy: Optional[SmaCross] = None, send_orders: bool = False,
                 bar_interval_s: int = 60, context=None):
        import zmq  # pyzmq is only needed for the live path

        self.zmq = zmq
        self.ticker, self.ticker_id = ticker, ticker_id
        self.strategy = strategy or SmaCross()
        self.send_orders = send_orders
        self.aggregator = BarAggregator(ticker=ticker, interval_s=bar_interval_s)

        self.position: Optional[int] = None      # from the broker (POSITIONS), None = unknown
        self.last_target: Optional[int] = None   # last target actually sent
        self.order_in_flight = False
        self.in_flight_since = 0.0
        self.last_engine_hb = time.monotonic()
        self.targets_sent: list[int] = []

        self.ctx = context or zmq.Context.instance()
        self.sub_data = self.ctx.socket(zmq.SUB)
        self.sub_data.connect(endpoints.data)
        for topic in (f"{ticker}_TRADE", "POSITIONS"):
            self.sub_data.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sub_exec = self.ctx.socket(zmq.SUB)
        self.sub_exec.connect(endpoints.executions)
        self.sub_exec.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_hb = self.ctx.socket(zmq.SUB)
        self.sub_hb.connect(endpoints.engine_heartbeat)
        self.sub_hb.setsockopt(zmq.SUBSCRIBE, b"")
        self.push_orders = self.ctx.socket(zmq.PUSH)
        self.push_orders.connect(endpoints.orders)
        self.pub_hb = self.ctx.socket(zmq.PUB)
        self.pub_hb.connect(endpoints.strategy_heartbeat)
        for s in (self.sub_data, self.sub_exec, self.sub_hb, self.push_orders, self.pub_hb):
            s.setsockopt(zmq.LINGER, 0)

        self.poller = zmq.Poller()
        for s in (self.sub_data, self.sub_exec, self.sub_hb):
            self.poller.register(s, zmq.POLLIN)

        self._stop = threading.Event()
        self._hb_seq = 0
        self._hb_lock = threading.Lock()

    # -- heartbeat ------------------------------------------------------------
    def send_heartbeat(self) -> None:
        with self._hb_lock:  # zmq sockets are not thread-safe
            self._hb_seq += 1
            self.pub_hb.send(ipc.encode_heartbeat(self._hb_seq, int(time.time() * 1000)))

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(1.0):
            self.send_heartbeat()

    # -- event handling -------------------------------------------------------
    def step(self, timeout_ms: int = 100) -> None:
        """Process whatever arrived within ``timeout_ms``."""
        events = dict(self.poller.poll(timeout_ms))
        if self.sub_hb in events:
            self.sub_hb.recv()
            self.last_engine_hb = time.monotonic()
        if self.sub_exec in events:
            self._on_execution(ipc.decode_execution(self.sub_exec.recv()))
        if self.sub_data in events:
            topic, payload = self.sub_data.recv_multipart()
            if topic == b"POSITIONS":
                self._on_position(ipc.decode_position(payload))
            else:
                t = ipc.decode_trade(payload)
                for bar in self.aggregator.add_trade(t.timestamp_us, t.price, t.size):
                    self._on_bar(bar)

    def _on_position(self, rep: ipc.PositionReport) -> None:
        if rep.ticker_id == self.ticker_id:
            self.position = rep.position
            log.info("[POSITION] broker says %s = %d", self.ticker, rep.position)

    def _on_execution(self, rep: ipc.ExecutionReport) -> None:
        if rep.status == ipc.STATUS_BROKER_DISCONNECT:
            log.error("[EXEC] engine lost the broker connection")
            return
        if rep.ticker_id != self.ticker_id:
            return
        log.info("[EXEC] order %d status %d filled %d remaining %d @ %.4f",
                 rep.order_id, rep.status, rep.filled, rep.remaining, rep.price)
        if rep.status in ipc.TERMINAL_STATUSES:
            self.order_in_flight = False
            if rep.status != ipc.STATUS_FILLED:
                # The target was NOT reached: forget it, so the next bar re-sends it
                # (safe — the engine turns the same target into the remaining delta).
                self.last_target = None

    def _on_bar(self, bar: Bar) -> None:
        target = self.strategy.on_bar(bar)
        log.info("[BAR] %s close=%.4f vol=%.0f%s -> target %s", bar.timestamp.isoformat(),
                 bar.close, bar.volume, " (synthetic)" if bar.synthetic else "", target)
        if target is None or target == self.last_target:
            return
        if self.order_in_flight:
            # Deliberately NOT cleared on a timer: guessing that an order died and
            # sending another is how a position gets doubled. A stuck order is for
            # a human (or the engine's watchdog) to resolve.
            log.warning("[ORDER] target %d waits: previous order still in flight for %.0fs",
                        target, time.monotonic() - self.in_flight_since)
            return
        if not self.send_orders:
            log.info("[DRY RUN] would send target position %d for %s", target, self.ticker)
            self.last_target = target
            return
        self.push_orders.send(ipc.encode_target_position(self.ticker_id, target))
        self.targets_sent.append(target)
        self.last_target = target
        self.order_in_flight = True
        self.in_flight_since = time.monotonic()
        log.info("[ORDER] sent target position %d for %s", target, self.ticker)

    # -- lifecycle ------------------------------------------------------------
    def run(self) -> None:
        hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb.start()
        log.info("live client up for %s (tickerId %d), %s", self.ticker, self.ticker_id,
                 "SENDING ORDERS" if self.send_orders else "dry run")
        try:
            while not self._stop.is_set():
                self.step()
                if time.monotonic() - self.last_engine_hb > 5.0:
                    log.warning("[ENGINE] no heartbeat for %.0fs", time.monotonic() - self.last_engine_hb)
                    self.last_engine_hb = time.monotonic()  # warn at most every 5s
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        for s in (self.sub_data, self.sub_exec, self.sub_hb, self.push_orders, self.pub_hb):
            s.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="the SAME json the engine was started with")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--send-orders", action="store_true", help="actually send target positions")
    ap.add_argument("--fast", type=int, default=10)
    ap.add_argument("--slow", type=int, default=40)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    client = LiveClient(args.ticker, ticker_id_from_config(config, args.ticker),
                        strategy=SmaCross(args.fast, args.slow), send_orders=args.send_orders)
    client.run()


if __name__ == "__main__":
    main()
