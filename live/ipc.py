"""Wire format of the C++ engine's ZeroMQ protocol (IPC v3), Python side.

This module is the ONLY place the Python side spells out the binary layouts. Each
``struct.Struct`` mirrors a C++ struct in ``cpp_engine/include`` byte for byte, and
``tests/test_ipc.py`` compiles those headers and checks sizeof + every field offset
against the formats below — a layout change on one side without the other fails CI
instead of silently reading a padding byte as a field.

Socket map (the engine BINDS everything; a strategy CONNECTS):

    5555  PUB   engine -> strategy   multipart [topic, payload]
                  "<TICKER>_TRADE"  TradeUpdate     (executed trade)
                  "<TICKER>_L2"     L2Update        (book level update)
                  "POSITIONS"       PositionReport  (broker reconciliation)
    5556  PULL  strategy -> engine   TargetPositionRequest (desired NET position)
    5557  PUB   engine -> strategy   ExecutionReport (single frame, no topic)
    5558  PUB   engine -> strategy   b"HB" every second (engine liveness)
    5559  SUB   strategy -> engine   StrategyHeartbeat (strategy liveness; silence
                                     beyond the watchdog timeout flattens the book)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

IPC_PROTOCOL_VERSION = 3

# "<" = little-endian, no implicit padding: explicit "3x" / trailing ints mirror the
# explicit reserved/padding fields of the #pragma pack(1) structs.
TARGET_POSITION = struct.Struct("<B3xdiiii")   # TargetPositionRequest, 28 bytes
EXECUTION = struct.Struct("<B3xdiiiii")        # ExecutionReport,       32 bytes
POSITION = struct.Struct("<B3xiid")            # PositionReport,        20 bytes
HEARTBEAT = struct.Struct("<B3xIq")            # StrategyHeartbeat,     16 bytes
TRADE = struct.Struct("<B3xqdii")              # TradeUpdate,           28 bytes
L2 = struct.Struct("<qdiiii")                  # L2Update (not packed, no padding), 32 bytes

ORDER_MKT = 1
ORDER_LMT = 2

# ExecutionReport.status
STATUS_OTHER = 0
STATUS_SUBMITTED = 1
STATUS_FILLED = 2
STATUS_CANCELLED = 3
STATUS_PRESUBMITTED = 4
STATUS_INACTIVE = 5
STATUS_BROKER_DISCONNECT = 9
TERMINAL_STATUSES = frozenset({STATUS_FILLED, STATUS_CANCELLED, STATUS_INACTIVE})


class ProtocolError(ValueError):
    """Payload of the wrong size or of another protocol version."""


def _check(payload: bytes, st: struct.Struct, name: str) -> tuple:
    if len(payload) != st.size:
        raise ProtocolError(f"{name}: {len(payload)} bytes, expected {st.size}")
    fields = st.unpack(payload)
    return fields


@dataclass(frozen=True)
class TradeUpdate:
    timestamp_us: int
    price: float
    size: int


@dataclass(frozen=True)
class L2Update:
    timestamp_us: int
    price: float
    position: int
    operation: int
    side: int      # 1 = bid, 0 = ask
    size: int


@dataclass(frozen=True)
class PositionReport:
    ticker_id: int
    position: int
    avg_cost: float


@dataclass(frozen=True)
class ExecutionReport:
    price: float
    ticker_id: int   # -1: order the engine did not originate, or a broker disconnect
    order_id: int    # 0: synthetic "already at target" fill
    status: int
    filled: int
    remaining: int


def _versioned(fields: tuple, name: str) -> tuple:
    if fields[0] != IPC_PROTOCOL_VERSION:
        raise ProtocolError(f"{name}: protocol version {fields[0]}, expected {IPC_PROTOCOL_VERSION}")
    return fields[1:]


def decode_trade(payload: bytes) -> TradeUpdate:
    ts, price, size, _pad = _versioned(_check(payload, TRADE, "TradeUpdate"), "TradeUpdate")
    return TradeUpdate(ts, price, size)


def decode_l2(payload: bytes) -> L2Update:
    return L2Update(*_check(payload, L2, "L2Update"))


def decode_position(payload: bytes) -> PositionReport:
    return PositionReport(*_versioned(_check(payload, POSITION, "PositionReport"), "PositionReport"))


def decode_execution(payload: bytes) -> ExecutionReport:
    return ExecutionReport(*_versioned(_check(payload, EXECUTION, "ExecutionReport"), "ExecutionReport"))


def encode_target_position(ticker_id: int, target_position: int,
                           order_type: int = ORDER_MKT, price: float = 0.0) -> bytes:
    if order_type == ORDER_LMT and price <= 0:
        raise ValueError("a limit order needs a positive price")
    return TARGET_POSITION.pack(IPC_PROTOCOL_VERSION, float(price), int(ticker_id),
                                int(target_position), int(order_type), 0)


def encode_heartbeat(seq: int, timestamp_unix_ms: int) -> bytes:
    return HEARTBEAT.pack(IPC_PROTOCOL_VERSION, seq & 0xFFFFFFFF, int(timestamp_unix_ms))


# Encoders for the engine -> strategy messages: used by tests and by anyone writing a
# fake engine to exercise a strategy without a broker.
def encode_trade(timestamp_us: int, price: float, size: int) -> bytes:
    return TRADE.pack(IPC_PROTOCOL_VERSION, int(timestamp_us), float(price), int(size), 0)


def encode_position(ticker_id: int, position: int, avg_cost: float) -> bytes:
    return POSITION.pack(IPC_PROTOCOL_VERSION, ticker_id, position, avg_cost)


def encode_execution(price: float, ticker_id: int, order_id: int, status: int,
                     filled: int, remaining: int) -> bytes:
    return EXECUTION.pack(IPC_PROTOCOL_VERSION, price, ticker_id, order_id, status, filled, remaining)
