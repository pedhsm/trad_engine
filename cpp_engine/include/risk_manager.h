#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>

struct PreTradeRiskLimits {
    int max_lot_size = 50;           // Max lots per order
    int max_orders_per_day = 200;    // Frequency circuit breaker
    int max_concurrent_orders = 10;  // Prevents runaway loops
    double max_daily_loss_usd = 2000.0; // Financial hard limit
    // Strategy silence tolerated before flattening. Five missed beats at 1s.
    // Too short flattens on a network hiccup; too long leaves an orphan
    // position in the market.
    int watchdog_timeout_ms = 5000;
    // UTC hour at which the trading "day" rolls over and the per-day order
    // counter resets (0 = midnight UTC; e.g. 21 or 22 for US futures, whose
    // session rolls at 17:00 New York time).
    int day_reset_utc_hour = 0;
};

class ExecutionRiskManager {
    private:
        PreTradeRiskLimits limits;
        std::atomic<bool> kill_switch_active{false};
        std::atomic<int> orders_today{0};
        std::atomic<int> open_orders{0};
        std::atomic<int64_t> current_day{-1};
        std::unordered_set<int> active_orders;
        std::mutex risk_mutex;

        std::atomic<double> total_realized_pnl{0.0};
        std::atomic<double> total_unrealized_pnl{0.0};

    public:
        // Applies the limits coming from the config JSON.
        //
        // Before this the engine IGNORED `risk_params` entirely and used the
        // struct defaults. The operator would write max_daily_loss_usd: 1500 in
        // the config and the real cut happened at 2000 — the configured number
        // was not the number protecting the account, and nothing said so.
        //
        // Called ONCE at boot, before any thread starts.
        void applyLimits(const PreTradeRiskLimits& updated) {
            limits = updated;
            std::cout << "[RISK] Limits applied: max daily loss USD "
                      << limits.max_daily_loss_usd
                      << " | max lot " << limits.max_lot_size
                      << " | orders/day " << limits.max_orders_per_day
                      << " | concurrent " << limits.max_concurrent_orders
                      << " | watchdog " << limits.watchdog_timeout_ms << "ms"
                      << " | day resets at " << limits.day_reset_utc_hour << ":00 UTC"
                      << std::endl;
        }

        // Resets the per-day order counter when the trading day changes. Called by
        // both the orders thread and the watchdog; the compare-exchange makes the
        // reset happen exactly once per day.
        //
        // The KILL SWITCH is deliberately NOT reset here: once the engine halted,
        // a human decides when trading resumes (restart), not the calendar.
        void rollDayIfNeeded(int64_t now_unix_s) {
            const int64_t day = (now_unix_s - int64_t(limits.day_reset_utc_hour) * 3600) / 86400;
            int64_t prev = current_day.load();
            if (day != prev && current_day.compare_exchange_strong(prev, day)) {
                // The first call only anchors the current day: resetting there
                // would wipe orders counted before it.
                if (prev != -1) {
                    std::cout << "[RISK] New trading day: order counter reset (was "
                              << orders_today.load() << ")." << std::endl;
                    orders_today = 0;
                }
            }
        }

        void rollDayIfNeeded() {
            rollDayIfNeeded(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        }

        void activateKillSwitch(){
            kill_switch_active = true;
            std::cout << "[RISK] KILL SWITCH ENGAGED! No new orders will be sent." << std::endl;
        }

        bool killSwitchActive() const { return kill_switch_active.load(); }

        int watchdogTimeoutMs() const { return limits.watchdog_timeout_ms; }

        int ordersToday() const { return orders_today.load(); }

        // Daily limit breached? Consulted by the watchdog thread, which is the
        // one that decides to flatten. Before, this check only ran when an order
        // arrived — meaning a portfolio sinking without new orders was never
        // noticed. The PnL is whatever the broker reports (updatePortfolio).
        bool dailyLimitBreached() const {
            return pnlTotal() <= -limits.max_daily_loss_usd;
        }

        double pnlTotal() const {
            return total_realized_pnl.load(std::memory_order_relaxed)
                 + total_unrealized_pnl.load(std::memory_order_relaxed);
        }

        // Copy of the live orders, so Halt can cancel them one by one without
        // holding the lock while talking to the broker.
        std::unordered_set<int> activeOrders() {
            std::lock_guard<std::mutex> lock(risk_mutex);
            return active_orders;
        }

        void registerOrderStart(int orderId) {
            std::lock_guard<std::mutex> lock(risk_mutex);
            active_orders.insert(orderId);
            open_orders = static_cast<int>(active_orders.size());
            orders_today++;
        }

        // An order the broker had reported as closed is working again (an Inactive
        // order that came back). Counts toward the open orders, not toward orders_today:
        // it is the same order, not a new one.
        void registerOrderReopened(int orderId) {
            std::lock_guard<std::mutex> lock(risk_mutex);
            active_orders.insert(orderId);
            open_orders = static_cast<int>(active_orders.size());
        }

        void registerOrderClosed(int orderId) {
            std::lock_guard<std::mutex> lock(risk_mutex);
            active_orders.erase(orderId);
            open_orders = static_cast<int>(active_orders.size());
        }

        void updatePnL(double realized, double unrealized) {
            total_realized_pnl.store(realized, std::memory_order_relaxed);
            total_unrealized_pnl.store(unrealized, std::memory_order_relaxed);
        }

        // Pre-trade gate. Returns true and fills out_quantity/out_action when the
        // move from current_position to target_position may be sent. On false,
        // out_reason says why (empty when already at the target: not a refusal).
        bool approveTargetOrder(int target_position, int current_position, double price, int orderType,
                                int& out_quantity, std::string& out_action, std::string& out_reason){
            out_reason.clear();
            rollDayIfNeeded();

            if (kill_switch_active){
                out_reason = "kill switch engaged";
                std::cout << "[RISK REJECT] System is in Kill Switch mode." << std::endl;
                return false;
            }

            double pnl_total = pnlTotal();
            if (pnl_total <= -limits.max_daily_loss_usd) {
                out_reason = "max daily loss hit";
                std::cout << "[RISK REJECT] Max Daily Loss hit! Current PnL: " << pnl_total << " (Limit: " << -limits.max_daily_loss_usd << ")" << std::endl;
                activateKillSwitch();
                return false;
            }

            int delta = target_position - current_position;
            if (delta == 0) {
                // Already at the target position. Not a refusal (no spam).
                return false;
            }

            int quantity = std::abs(delta);
            std::string action = (delta > 0) ? "BUY" : "SELL";

            if (quantity > limits.max_lot_size){
                out_reason = "quantity above max_lot_size";
                std::cout << "[RISK REJECT] Invalid/absurd quantity: " << quantity
                          << " (Max: " << limits.max_lot_size << ")" << std::endl;
                return false;
            }

            if (orderType == 2 && price <= 0.0){
                out_reason = "invalid limit price";
                std::cout << "[RISK REJECT] Invalid limit price: " << price << std::endl;
                return false;
            }

            if (orders_today >= limits.max_orders_per_day) {
                out_reason = "daily order limit";
                std::cout << "[RISK REJECT] Daily order limit exceeded: " << orders_today << std::endl;
                return false;
            }

            if (open_orders >= limits.max_concurrent_orders) {
                out_reason = "concurrent order limit";
                std::cout << "[RISK REJECT] Concurrent order limit exceeded: " << open_orders << std::endl;
                return false;
            }

            out_quantity = quantity;
            out_action = action;
            return true;
        }
};
