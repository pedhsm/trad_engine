"""IPC: the Python wire formats must match the C++ structs, and the example live
client must speak the protocol correctly against a fake engine."""
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from live import ipc

INCLUDE = Path(__file__).resolve().parents[1] / "cpp_engine" / "include"

# C++ struct -> (Python Struct, field names in declaration order, skipping reserved/padding
# arrays that the Python format spells as "Nx").
LAYOUTS = {
    "TargetPositionRequest": (ipc.TARGET_POSITION, ["version", "price", "tickerId", "target_position", "orderType", "padding"]),
    "ExecutionReport": (ipc.EXECUTION, ["version", "price", "tickerId", "orderId", "status", "filled", "remaining"]),
    "PositionReport": (ipc.POSITION, ["version", "tickerId", "position", "avgCost"]),
    "StrategyHeartbeat": (ipc.HEARTBEAT, ["version", "seq", "timestamp_unix_ms"]),
    "TradeUpdate": (ipc.TRADE, ["version", "timestamp_us", "price", "size", "padding"]),
    "L2Update": (ipc.L2, ["timestamp", "price", "position", "operation", "side", "size"]),
}


def _python_offsets(st: struct.Struct) -> list[int]:
    """Offset of each non-pad field of a '<'-format struct."""
    offsets, pos = [], 0
    for count, code in re.findall(r"(\d*)([a-zA-Z?])", st.format.lstrip("<=!>@")):
        n = int(count) if count else 1
        if code == "x":
            pos += n
            continue
        for _ in range(n):
            offsets.append(pos)
            pos += struct.calcsize("<" + code)
    return offsets


@pytest.fixture(scope="module")
def cpp_layout(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    d = tmp_path_factory.mktemp("ipc")
    lines = ['#include <cstdio>', '#include <cstddef>', '#include "ipc_messages.h"', "int main() {"]
    for name, (_, fields) in LAYOUTS.items():
        lines.append(f'  std::printf("{name} sizeof %zu\\n", sizeof({name}));')
        for f in fields:
            lines.append(f'  std::printf("{name} {f} %zu\\n", offsetof({name}, {f}));')
    lines += ["  return 0;", "}"]
    src = d / "layout.cpp"
    src.write_text("\n".join(lines))
    exe = d / ("layout.exe" if sys.platform == "win32" else "layout")
    subprocess.run([cxx, "-std=c++17", "-I", str(INCLUDE), "-o", str(exe), str(src)]
                   + (["-static"] if sys.platform == "win32" else []), check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    table = {}
    for line in out.splitlines():
        name, field, value = line.split()
        table[(name, field)] = int(value)
    return table


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_python_layout_matches_cpp_struct(cpp_layout, name):
    st, fields = LAYOUTS[name]
    assert st.size == cpp_layout[(name, "sizeof")], f"{name}: size differs"
    cpp_offsets = [cpp_layout[(name, f)] for f in fields]
    assert _python_offsets(st) == cpp_offsets, f"{name}: field offsets differ"


def test_roundtrips_and_version_check():
    t = ipc.decode_trade(ipc.encode_trade(1_700_000_000_000_000, 101.25, 7))
    assert (t.timestamp_us, t.price, t.size) == (1_700_000_000_000_000, 101.25, 7)
    p = ipc.decode_position(ipc.encode_position(3, -2, 99.5))
    assert (p.ticker_id, p.position, p.avg_cost) == (3, -2, 99.5)
    e = ipc.decode_execution(ipc.encode_execution(10.0, 1, 1001, ipc.STATUS_FILLED, 1, 0))
    assert (e.ticker_id, e.order_id, e.status) == (1, 1001, ipc.STATUS_FILLED)

    stale = bytearray(ipc.encode_trade(1, 1.0, 1))
    stale[0] = 2  # a v2 engine
    with pytest.raises(ipc.ProtocolError):
        ipc.decode_trade(bytes(stale))
    with pytest.raises(ipc.ProtocolError):
        ipc.decode_execution(b"\x03" * 31)
    with pytest.raises(ValueError):
        ipc.encode_target_position(1, 1, ipc.ORDER_LMT, price=0.0)


# --- end to end against a fake engine ----------------------------------------

zmq = pytest.importorskip("zmq")

MIN_US = 60_000_000
T0_US = 1_704_103_200 * 1_000_000  # 2024-01-01 10:00 UTC, minute-aligned


class FakeEngine:
    """Binds the engine's five sockets on random ports, like the C++ engine does."""

    def __init__(self, ctx):
        self.pub = ctx.socket(zmq.PUB)
        self.pull = ctx.socket(zmq.PULL)
        self.exec_pub = ctx.socket(zmq.PUB)
        self.hb_pub = ctx.socket(zmq.PUB)
        self.hb_sub = ctx.socket(zmq.SUB)
        self.hb_sub.setsockopt(zmq.SUBSCRIBE, b"")
        eps = []
        for s in (self.pub, self.pull, self.exec_pub, self.hb_pub, self.hb_sub):
            s.setsockopt(zmq.LINGER, 0)
            s.bind("tcp://127.0.0.1:*")
            eps.append(s.getsockopt_string(zmq.LAST_ENDPOINT))
        from examples.live_client import Endpoints
        self.endpoints = Endpoints(*eps)

    def trade(self, minute, price, ticker="SPY"):
        self.pub.send_multipart([f"{ticker}_TRADE".encode(),
                                 ipc.encode_trade(T0_US + minute * MIN_US, price, 1)])

    def close(self):
        for s in (self.pub, self.pull, self.exec_pub, self.hb_pub, self.hb_sub):
            s.close()


def _pump(client, cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        client.step(20)
        if cond():
            return True
    return False


def test_live_client_against_fake_engine():
    from examples.live_client import LiveClient, SmaCross

    ctx = zmq.Context()
    eng = FakeEngine(ctx)
    client = LiveClient("SPY", ticker_id=1, endpoints=eng.endpoints,
                        strategy=SmaCross(fast=2, slow=3), send_orders=True, context=ctx)
    try:
        time.sleep(0.3)  # PUB/SUB slow joiner

        # Rising prices: minute 0 is the discarded boot bar, bars 1..3 fill the slow SMA.
        for m, px in enumerate([100, 101, 102, 103, 104]):
            eng.trade(m, px)
        assert _pump(client, lambda: client.targets_sent == [1])
        assert eng.pull.poll(2000), "target position never reached the engine"
        req = eng.pull.recv()
        version, price, ticker_id, target, order_type, _ = ipc.TARGET_POSITION.unpack(req)
        assert (version, ticker_id, target, order_type) == (ipc.IPC_PROTOCOL_VERSION, 1, 1, ipc.ORDER_MKT)

        # Falling prices flip the signal, but the first order has no fill yet: no second order.
        for m, px in enumerate([90, 80, 70], start=5):
            eng.trade(m, px)
        _pump(client, lambda: False, timeout=0.5)
        assert client.targets_sent == [1] and client.order_in_flight

        # Fill for ANOTHER ticker must not unlock this one.
        eng.exec_pub.send(ipc.encode_execution(104.0, 2, 1000, ipc.STATUS_FILLED, 1, 0))
        _pump(client, lambda: False, timeout=0.3)
        assert client.order_in_flight

        eng.exec_pub.send(ipc.encode_execution(104.0, 1, 1000, ipc.STATUS_FILLED, 1, 0))
        assert _pump(client, lambda: not client.order_in_flight)
        eng.trade(8, 60)  # closes the minute-7 bar -> target -1 goes out now
        assert _pump(client, lambda: client.targets_sent == [1, -1])

        # Broker reconciliation and the strategy heartbeat.
        eng.pub.send_multipart([b"POSITIONS", ipc.encode_position(1, 1, 104.0)])
        assert _pump(client, lambda: client.position == 1)
        # PUB->SUB drops messages until the subscription handshake completes (ZMQ
        # "slow joiner"), so beat repeatedly, as the real 1s loop does.
        hb = b""
        for _ in range(30):
            client.send_heartbeat()
            if eng.hb_sub.poll(100):
                hb = eng.hb_sub.recv()
                break
        version, seq, _ts = ipc.HEARTBEAT.unpack(hb)
        assert version == ipc.IPC_PROTOCOL_VERSION and seq >= 1
    finally:
        client.close()
        eng.close()
        ctx.term()
