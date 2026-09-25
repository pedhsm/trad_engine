#pragma once
#include <cstdint>
#include "DataStructures.h"

// ── IPC PROTOCOL v3 ─────────────────────────────────────────────────────
// v1 -> v2: ExecutionReport gained the tickerId field.
// In v1 the execution report did not say which asset it belonged to, and the
// Python side ended up reading a padding byte in place of the id — which made
// EVERY exec report be discarded and the order mutex never unlock.
//
// v2 -> v3: TradeUpdate was born (executed trade, topic `<ticker>_TRADE`),
// which feeds the 1-minute OHLCV bars of contract mode. No previous struct
// changed layout — v3 is additive. Even so the version was bumped, because it
// is what tells the operator that engine and strategy must be deployed
// together: a v3 strategy asking a v2 engine for `_TRADE` would get silence,
// and silence here looks like "market stopped", not "wrong version".
//
// Engine and strategy MUST be deployed together. A divergent version is
// rejected explicitly in the check, instead of failing silently.
#define IPC_PROTOCOL_VERSION 3

// ExecutionReport.status. A strategy that keeps one order in flight per asset
// unlocks on the TERMINAL ones (FILLED, CANCELLED, INACTIVE, REJECTED).
enum ExecStatus : int {
    EXEC_OTHER = 0,
    EXEC_SUBMITTED = 1,
    EXEC_FILLED = 2,          // also sent with orderId 0 when already at the target
    EXEC_CANCELLED = 3,
    EXEC_PRESUBMITTED = 4,
    EXEC_INACTIVE = 5,
    // The engine refused the target (risk limit, kill switch, order already in
    // flight). orderId 0, remaining = the refused quantity. Without it a refused
    // target is silence, and a strategy waiting for its order would wait forever.
    EXEC_REJECTED = 6,
    EXEC_BROKER_DISCONNECT = 9,
};

#pragma pack(push, 1)
struct ExecutionReport {
    uint8_t version;      // IPC_PROTOCOL_VERSION
    uint8_t reserved[3];  // Explicit alignment
    double price;
    int tickerId;         // v2: which asset this report belongs to
    int orderId;
    int status;           // ExecStatus
    int filled;
    int remaining;
};

struct PositionReport {
    uint8_t version;      // IPC_PROTOCOL_VERSION
    uint8_t reserved[3];  // Explicit alignment
    int tickerId;
    int position;
    double avgCost;
};

// Liveness signal from the Python strategy -> engine (port 5559).
//
// A DEDICATED channel, not the orders one (5556), on purpose: measuring silence
// on the command channel would confuse "dead strategy" with "live strategy with
// no signal to send", and the engine would flatten the book in a merely stalled
// market.
struct StrategyHeartbeat {
    uint8_t version;          // IPC_PROTOCOL_VERSION
    uint8_t reserved[3];      // Explicit alignment
    uint32_t seq;             // sequential; a jump indicates a missed beat
    int64_t timestamp_unix_ms;// strategy clock, for diagnostics only
};
#pragma pack(pop)

static_assert(sizeof(ExecutionReport) == 32, "ExecutionReport must be 32 bytes");
static_assert(sizeof(PositionReport) == 20, "PositionReport must be 20 bytes");
static_assert(sizeof(TargetPositionRequest) == 28, "TargetPositionRequest must be 28 bytes");
static_assert(sizeof(StrategyHeartbeat) == 16, "StrategyHeartbeat must be 16 bytes");
static_assert(sizeof(TradeUpdate) == 28, "TradeUpdate must be 28 bytes");
static_assert(sizeof(L2Update) == 32, "L2Update must be 32 bytes");
