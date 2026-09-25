#pragma once
#include <cstdint>
// from biggest to smallest always (no padding!)

struct L2Update {
    long long timestamp; 
    double price;        
    int position;        
    int operation;       
    int side;            
    int size;            
};

// EXECUTED TRADE (LAST + LAST_SIZE) — basis for the 1-minute OHLCV bars.
//
// Replaces the old `L1Update`, which was declared with ZERO uses in the engine:
// it was never populated nor published. This struct takes that place, now with a
// version header — an unversioned payload inside a versioned protocol is the
// exception that comes back as a bug.
//
// Goes on its OWN topic (`<ticker>_TRADE`), never on `_L2`. Mixing the two on the
// same topic would force the consumer to guess the payload from its size.
//
// WHY TRADE AND NOT MID-PRICE: build the live bar from executed trades, not from
// (bid+ask)/2. A mid-price bar carries no volume, so any volume-based feature
// silently breaks; and it is a different series than the trade bars a backtest is
// usually validated on. Validating on one kind of series and trading on another
// reintroduces, through another door, the very mismatch you were trying to close.
#pragma pack(push, 1)
struct TradeUpdate {
    uint8_t version;        // 1 byte  — IPC_PROTOCOL_VERSION
    uint8_t reserved[3];    // 3 bytes — explicit alignment
    int64_t timestamp_us;   // 8 bytes — epoch in MICROSECONDS (same as L2)
    double price;           // 8 bytes — trade price (LAST)
    int32_t size;           // 4 bytes — traded quantity (LAST_SIZE)
    int32_t padding;        // 4 bytes — closes at exactly 28 bytes
};
#pragma pack(pop)

struct OHLC{
    long long timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

#pragma pack(push, 1)
struct TargetPositionRequest {
    uint8_t version;      // 1 byte  — IPC protocol version
    uint8_t reserved[3];  // 3 bytes — Explicit padding (alignment)
    double price;         // 8 bytes — Price for LMT orders
    int tickerId;         // 4 bytes — Asset ID
    int target_position;  // 4 bytes — Desired net position (e.g.: +1 = Long 1)
    int orderType;        // 4 bytes — 1 = MKT, 2 = LMT
    int padding;          // 4 bytes — Keeps exactly 28 bytes
};
#pragma pack(pop)