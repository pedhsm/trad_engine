// Unit test for the pre-trade risk gate (header-only, no broker SDK needed).
// Built and run by tests/test_cpp_engine.py; exits non-zero on the first failure.
#include "risk_manager.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); std::exit(1); } } while (0)

static const int64_t DAY = 86400;
static const int64_t T0 = 1704067200;  // 2024-01-01 00:00:00 UTC

int main() {
    // --- sizing, direction, lot limit, limit price ---
    {
        ExecutionRiskManager rm;
        PreTradeRiskLimits lim;
        lim.max_lot_size = 3;
        rm.applyLimits(lim);
        int qty = 0; std::string action, reason;

        CHECK(rm.approveTargetOrder(2, -1, 0.0, 1, qty, action, reason));
        CHECK(qty == 3 && action == "BUY");
        CHECK(rm.approveTargetOrder(-1, 1, 0.0, 1, qty, action, reason));
        CHECK(qty == 2 && action == "SELL");

        CHECK(!rm.approveTargetOrder(4, 0, 0.0, 1, qty, action, reason));
        CHECK(reason == "quantity above max_lot_size");
        CHECK(!rm.approveTargetOrder(1, 0, 0.0, 2, qty, action, reason));   // LMT without price
        CHECK(reason == "invalid limit price");
        CHECK(!rm.approveTargetOrder(1, 1, 0.0, 1, qty, action, reason));   // already there
        CHECK(reason.empty());
    }

    // --- per-day order counter resets at the configured UTC hour ---
    {
        ExecutionRiskManager rm;
        PreTradeRiskLimits lim;
        lim.max_orders_per_day = 2;
        lim.day_reset_utc_hour = 22;
        rm.applyLimits(lim);
        int qty = 0; std::string action, reason;

        rm.rollDayIfNeeded(T0 + 12 * 3600);              // Jan 1, 12:00 UTC
        rm.registerOrderStart(1); rm.registerOrderClosed(1);
        rm.registerOrderStart(2); rm.registerOrderClosed(2);
        CHECK(rm.ordersToday() == 2);

        rm.rollDayIfNeeded(T0 + 21 * 3600 + 3599);       // 21:59:59: same trading day
        CHECK(rm.ordersToday() == 2);
        rm.rollDayIfNeeded(T0 + 22 * 3600);              // 22:00: new trading day
        CHECK(rm.ordersToday() == 0);
        rm.rollDayIfNeeded(T0 + DAY + 12 * 3600);        // still that same trading day
        CHECK(rm.ordersToday() == 0);
        (void)qty; (void)action; (void)reason;
    }

    // --- order limit refuses, concurrency limit refuses ---
    {
        ExecutionRiskManager rm;
        PreTradeRiskLimits lim;
        lim.max_orders_per_day = 1;
        lim.max_concurrent_orders = 5;
        rm.applyLimits(lim);
        int qty = 0; std::string action, reason;
        rm.registerOrderStart(7);
        CHECK(!rm.approveTargetOrder(1, 0, 0.0, 1, qty, action, reason));
        CHECK(reason == "daily order limit");

        ExecutionRiskManager rm2;
        lim.max_orders_per_day = 100;
        lim.max_concurrent_orders = 1;
        rm2.applyLimits(lim);
        rm2.registerOrderStart(8);
        CHECK(!rm2.approveTargetOrder(1, 0, 0.0, 1, qty, action, reason));
        CHECK(reason == "concurrent order limit");
        rm2.registerOrderClosed(8);
        CHECK(rm2.approveTargetOrder(1, 0, 0.0, 1, qty, action, reason));
    }

    // --- daily loss engages the kill switch, which a new day does NOT release ---
    {
        ExecutionRiskManager rm;
        PreTradeRiskLimits lim;
        lim.max_daily_loss_usd = 1000.0;
        rm.applyLimits(lim);
        int qty = 0; std::string action, reason;

        rm.updatePnL(-400.0, -599.0);
        CHECK(!rm.dailyLimitBreached());
        rm.updatePnL(-400.0, -600.0);
        CHECK(rm.dailyLimitBreached());
        CHECK(!rm.approveTargetOrder(1, 0, 0.0, 1, qty, action, reason));
        CHECK(reason == "max daily loss hit" && rm.killSwitchActive());

        rm.updatePnL(0.0, 0.0);
        rm.rollDayIfNeeded(T0 + 10 * DAY);
        CHECK(!rm.approveTargetOrder(1, 0, 0.0, 1, qty, action, reason));
        CHECK(reason == "kill switch engaged");
    }

    std::printf("risk_manager: all checks passed\n");
    return 0;
}
