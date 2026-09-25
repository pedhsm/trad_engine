#pragma once
#include <atomic>
#include <string>
#include <cstdint>
#include <iostream>

struct PreTradeRiskLimits {
    int max_lot_size = 50;           // Max lots per order
    int max_orders_per_day = 200;    // Frequency circuit breaker
    int max_concurrent_orders = 10;  // Prevents runaway loops
    double max_daily_loss_usd = 2000.0; // Financial hard limit
    // Strategy silence tolerated before flattening. Five missed beats at 1s.
    // Too short flattens on a network hiccup; too long leaves an orphan
    // position in the market.
    int watchdog_timeout_ms = 5000;
};

class ExecutionRiskManager{
    private:
        PreTradeRiskLimits limits;
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
                      << std::endl;
        }
    private:
        std::atomic<bool> kill_switch_active;
        std::atomic<int> orders_today{0};
        std::atomic<int> open_orders{0};
        std::unordered_set<int> active_orders;
        std::mutex risk_mutex;

        std::atomic<double> total_realized_pnl{0.0};
        std::atomic<double> total_unrealized_pnl{0.0};

    public:
        ExecutionRiskManager() : kill_switch_active(false) {}

        void activateKillSwitch(){
            kill_switch_active = true;
            std::cout << "[RISK] KILL SWITCH ENGAGED! No new orders will be sent." << std::endl;
        }

        bool killSwitchActive() const { return kill_switch_active.load(); }

        int watchdogTimeoutMs() const { return limits.watchdog_timeout_ms; }

        // Daily limit breached? Consulted by the watchdog thread, which is the
        // one that decides to flatten. Before, this check only ran when an order
        // arrived — meaning a portfolio sinking without new orders was never
        // noticed.
        bool dailyLimitBreached() const {
            double pnl = total_realized_pnl.load(std::memory_order_relaxed)
                       + total_unrealized_pnl.load(std::memory_order_relaxed);
            return pnl <= -limits.max_daily_loss_usd;
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
            open_orders = active_orders.size();
            orders_today++;
        }

        void registerOrderClosed(int orderId) {
            std::lock_guard<std::mutex> lock(risk_mutex);
            active_orders.erase(orderId);
            open_orders = active_orders.size();
        }

        void updatePnL(double realized, double unrealized) {
            total_realized_pnl.store(realized, std::memory_order_relaxed);
            total_unrealized_pnl.store(unrealized, std::memory_order_relaxed);
        }

        bool approveTargetOrder(int tickerId, int target_position, int current_position, double price, int orderType, int& out_quantity, std::string& out_action){
            if (kill_switch_active){
                std::cout << "[RISK REJECT] System is in Kill Switch mode." << std::endl;
                return false;
            }

            double pnl_total = total_realized_pnl.load(std::memory_order_relaxed) + total_unrealized_pnl.load(std::memory_order_relaxed);
            if (pnl_total <= -limits.max_daily_loss_usd) {
                std::cout << "[RISK REJECT] Max Daily Loss hit! Current PnL: " << pnl_total << " (Limit: " << -limits.max_daily_loss_usd << ")" << std::endl;
                activateKillSwitch();
                return false;
            }

            int delta = target_position - current_position;
            if (delta == 0) {
                // Already at the target position. Silently ignored (no spam).
                return false;
            }

            int quantity = std::abs(delta);
            std::string action = (delta > 0) ? "BUY" : "SELL";

            if (quantity > limits.max_lot_size){
                std::cout << "[RISK REJECT] Invalid/absurd quantity: " << quantity
                          << " (Max: " << limits.max_lot_size << ")" << std::endl;
                return false;
            }

            if (orderType == 2 && price <= 0.0){
                std::cout << "[RISK REJECT] Invalid limit price: " << price << std::endl;
                return false;
            }

            if (orders_today >= limits.max_orders_per_day) {
                std::cout << "[RISK REJECT] Daily order limit exceeded: " << orders_today << std::endl;
                return false;
            }

            if (open_orders >= limits.max_concurrent_orders) {
                std::cout << "[RISK REJECT] Concurrent order limit exceeded: " << open_orders << std::endl;
                return false;
            }

            out_quantity = quantity;
            out_action = action;
            return true;
        }
};
