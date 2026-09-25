#pragma once
#include <cstdint>

// ==========================================
// L1 CACHE STRUCTURES (Stateful Order Book)
// ==========================================
struct BookSide {
    double price = 0.0;
    int size = 0;
};

struct L1Cache {
    BookSide bid;
    BookSide ask;
};

// Last trade seen, per asset.
//
// IBKR delivers price and size of the SAME trade in two separate callbacks:
// tickPrice(LAST) and then tickSize(LAST_SIZE). This cache joins the two.
struct TradeCache {
    double last_price = 0.0;
    int64_t last_ts_us = 0;
};
