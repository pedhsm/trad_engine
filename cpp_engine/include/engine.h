#pragma once

#ifdef _WIN32
#ifndef WIN32
#define WIN32
#endif
#endif
#ifndef IB_USE_STD_STRING
#define IB_USE_STD_STRING
#endif

#include "TwsApiL0.h"
#include "DataStructures.h"
#include "AllocatorArena.h"
#include "Contracts.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <zmq.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdint>

#include "caches.h"
#include "ipc_messages.h"
#include "risk_manager.h"

using json = nlohmann::json;

// ==========================================
// RECORDING FRAME (arena -> disk)
// ==========================================
// The arena holds records of every asset and kind back to back. Each payload is
// preceded by this header so the drain thread can route it to its own per-ticker
// file (`data/<T>/<T>_L1.bin`, `_L2.bin`, `_TRADE.bin`). The files themselves
// contain only the raw payloads, exactly as before.
enum RecordKind : int32_t {
    REC_L1 = 1,      // payload: L2Update (top of book, position 0)
    REC_L2 = 2,      // payload: L2Update (depth level)
    REC_TRADE = 3,   // payload: TradeUpdate
};

struct RecordHeader {
    int32_t tickerId;
    int32_t kind;    // RecordKind
};

static size_t recordPayloadSize(int32_t kind) {
    return kind == REC_TRADE ? sizeof(TradeUpdate) : sizeof(L2Update);
}

// ==========================================
// PER-ASSET DATA MODE (Phase 6)
// ==========================================
// Generalizes the old boolean flag `use_l1`. Each asset declares which data it
// needs, because subscribing to depth without using it costs bandwidth and —
// what matters more — consumes an IBKR market-data line, which is a counted
// resource.
//
// TICK_L2 is the historical default: no existing startup changes behavior.
enum class DataMode {
    TICK_L2,   // reqMktDepth  — deep book. Feeds VPIN (legacy mode).
    TICK_L1,   // reqMktData   — top of book. Was the --L1 flag.
    TRADE,     // reqMktData   — executed trades. Feeds the OHLCV bars.
    BOTH       // reqMktDepth + reqMktData — book AND trades on the same asset.
};

static bool wantsBook(DataMode m) {
    return m == DataMode::TICK_L2 || m == DataMode::BOTH;
}

static bool wantsTrades(DataMode m) {
    return m == DataMode::TRADE || m == DataMode::BOTH;
}

// ==========================================
// OPERATION MODE (Phase 1 — Parameterization)
// ==========================================
enum class OperationMode {
    LISTEN_ONLY,    // No disk I/O — ZMQ only
    RECORD_LOCAL,   // Current behavior — writes .bin into data/
    RECORD_CLOUD    // Same as record_local + future upload flag
};

class TradingEngine: public EWrapperL0 {
    private:
        OperationMode op_mode;  // Recording mode
        EClientL0* ptr_begin;

        // ===== PING-PONG BUFFER (Double Arena) =====
        // The market-data callbacks never touch the disk. They copy each record
        // (header + payload) into the ACTIVE arena; when it passes SWAP_THRESHOLD
        // of its capacity, or SWAP_INTERVAL has elapsed, the callback thread flips
        // `active_arena` and the drain thread writes the other arena to the
        // per-ticker files and resets it. Publishing on ZMQ does not depend on
        // any of this: a full arena drops the RECORD (and counts it), never the
        // live message.
        ArenaAllocator* arenas[2];
        std::atomic<int> active_arena;    // 0 or 1 — index of the write arena

        std::mutex              drain_mutex;
        std::condition_variable drain_cv;
        std::atomic<bool>       drain_pending;  // is there an arena to drain?
        int                     arena_to_drain; // index of the arena to be drained
        std::thread             thread_drain;
        std::chrono::steady_clock::time_point last_swap;

        // Shutdown quiescence: record() counts itself in and out, so the final
        // flush can wait until no callback is mid-copy before reading the arenas.
        std::atomic<bool> recording_open{true};
        std::atomic<int>  writers_in_record{0};
        std::atomic<uint64_t> dropped_records{0};
        std::atomic<bool> backpressure_warned{false};

        // File and name routers. The FILE* maps are written ONLY by the drain
        // thread (and by the final flush, after that thread has been joined).
        std::unordered_map<int, FILE*> file_map;       // <name>_L2.bin (depth)
        std::unordered_map<int, std::string> ticker_names;
        std::unordered_map<int, Contract> contract_map;
        std::unordered_map<int, FILE*> file_map_l1;    // <name>_L1.bin (top of book)
        // Trade trail (`<name>_TRADE.bin`). Without it there is no way to answer
        // later "which bar produced this order?" — a question any risk
        // desk asks, and that the order log alone does not answer.
        std::unordered_map<int, FILE*> file_map_trade;

        // State cache for L1 (unified Top of Book)
        std::unordered_map<int, L1Cache> active_book_state;

        // ── DATA AND TRADE MODE (Phase 6) ─────────────────────────────
        // Both maps are POPULATED IN registerAsset (boot, single thread) and
        // afterwards only read/updated by the IBKR callback thread. No insertion
        // happens on the hot-path — it was exactly a hot-path insertion that
        // caused the current_positions data race. Reads use find(), never
        // operator[], which inserts on a missing key.
        std::unordered_map<int, DataMode> ticker_modes;
        std::unordered_map<int, TradeCache> trade_cache;

        // Position cache (Target Position Routing). Seeded by the broker's
        // position() callbacks and updated from our own fills (orderStatus), so
        // a target arriving right after a fill is not computed against the
        // pre-fill position.
        std::unordered_map<int, std::atomic<int>> current_positions;

        // Per-ticker PnL tracking via IBKR
        std::unordered_map<int, double> realized_pnl;
        std::unordered_map<int, double> unrealized_pnl;
        std::mutex pnl_mutex;

        // ── STRATEGY WATCHDOG (Dead Man's Switch) ─────────────────────
        // STEADY clock, not wall clock: a system time adjustment must not make
        // the engine think the strategy went silent.
        std::atomic<int64_t> last_hb_ms{0};
        std::atomic<uint32_t> last_hb_seq{0};
        // The watchdog only arms AFTER the first beat. Without this the engine
        // would flatten during boot, before the strategy even starts.
        std::atomic<bool> watchdog_armed{false};
        // Ensures Halt & Liquidate runs ONCE, even if the watchdog thread and
        // the callback thread fire together.
        std::atomic<bool> liquidation_triggered{false};
        std::thread watchdog_thread;

        // Live orders sent by the engine: orderId -> {asset, position when sent,
        // direction}.
        //
        // The IBKR orderStatus callback delivers only the orderId; this map is
        // the bridge to the asset (for the ExecutionReport) and to the position
        // the order leads to: pos_at_send + direction * filled. That value is
        // ABSOLUTE, not a delta, so applying it is idempotent even when the
        // broker's position() update for the same fill arrives first.
        //
        // CONCURRENCY: written by the orders-loop thread (ZMQ) and read by the
        // IBKR callback thread. Every access goes under order_map_mutex, and the
        // lookup uses find() — never operator[], which INSERTS when the key does
        // not exist and would cause a concurrent write to the map structure.
        struct OrderInfo {
            int tickerId;
            int pos_at_send;
            int direction;   // +1 buy, -1 sell
        };
        std::unordered_map<int, OrderInfo> order_info;
        std::mutex order_map_mutex;

        // Orders that already reached a terminal status (same mutex), with the last
        // status seen. IB repeats terminal statuses (on a paper account every fill
        // arrived twice): an EXACT repeat is dropped. A CHANGE after a terminal status
        // is real and must not be lost: a fill racing a cancel (Cancelled, then
        // Filled), or an Inactive order coming back to life. FIFO-bounded.
        struct ClosedOrder {
            OrderInfo info;
            int status;
            int filled;
        };
        std::unordered_map<int, ClosedOrder> closed_orders;
        std::deque<int> closed_orders_fifo;
        static constexpr size_t CLOSED_ORDERS_KEPT = 1024;

        zmq::context_t zmq_ctx;
        zmq::socket_t zmq_pub;
        zmq::socket_t zmq_pull;
        zmq::socket_t zmq_exec_pub;
        zmq::socket_t zmq_hb_pub;
        zmq::socket_t zmq_hb_sub;   // strategy liveness signal (5559)
        std::mutex zmq_pub_mutex;   // zmq_pub is shared by the callback thread paths
        std::mutex zmq_exec_mutex;  // zmq_exec_pub: orders thread AND callback thread

        std::thread execution_thread;
        std::thread thread_heartbeat;
        std::atomic<bool> engine_running;
        std::atomic<int> next_order_id;
        ExecutionRiskManager risk_manager;

        // Swap when the active arena is 90% full, or at least once per interval
        // while it holds data: a quiet market must not leave hours of records in
        // RAM, where a crash would lose them.
        static constexpr double SWAP_THRESHOLD = 0.90;
        static constexpr std::chrono::milliseconds SWAP_INTERVAL{1000};

    public:
        void requestPortfolioUpdates() {
            if (ptr_begin) {
                ptr_begin->reqAccountUpdates(true, "");
            }
        }

        TradingEngine(ArenaAllocator* arena_a, ArenaAllocator* arena_b, OperationMode mode)
            : op_mode(mode), active_arena(0), drain_pending(false), arena_to_drain(-1),
              last_swap(std::chrono::steady_clock::now()),
              zmq_ctx(1), zmq_pub(zmq_ctx, zmq::socket_type::pub),
              zmq_pull(zmq_ctx, zmq::socket_type::pull),
              zmq_exec_pub(zmq_ctx, zmq::socket_type::pub),
              zmq_hb_pub(zmq_ctx, zmq::socket_type::pub),
              zmq_hb_sub(zmq_ctx, zmq::socket_type::sub),
              engine_running(true), next_order_id(1000)
        {
            ptr_begin = EClientL0::New(this);
            arenas[0] = arena_a;
            arenas[1] = arena_b;

            // ── ZMQ HWM: Avoids silent drop under back-pressure (default=1000) ──
            zmq_pub.set(zmq::sockopt::sndhwm, 10000);
            zmq_pull.set(zmq::sockopt::rcvhwm, 10000);
            zmq_exec_pub.set(zmq::sockopt::sndhwm, 10000);
            zmq_hb_pub.set(zmq::sockopt::sndhwm, 10000);
            zmq_hb_sub.set(zmq::sockopt::rcvhwm, 100);   // liveness signal does not accumulate
            zmq_hb_sub.set(zmq::sockopt::subscribe, "");
            zmq_hb_sub.set(zmq::sockopt::rcvtimeo, 500); // wakes up to check silence
        }

        ~TradingEngine(){
            engine_running = false;

            // 1. Stop the broker callbacks first: after this nothing new enters
            //    the arenas.
            if (ptr_begin) ptr_begin->eDisconnect();

            // 2. Stop the worker threads.
            drain_cv.notify_one();
            if(thread_drain.joinable()) thread_drain.join();
            if(execution_thread.joinable()) execution_thread.join();
            if(thread_heartbeat.joinable()) thread_heartbeat.join();
            if(watchdog_thread.joinable()) watchdog_thread.join();

            // 3. Flush what is still in RAM. Before, the active arena (up to 45 MB
            //    of records) was simply discarded on shutdown.
            flushRecordingOnShutdown();

            delete ptr_begin;
            for (auto* m : {&file_map, &file_map_l1, &file_map_trade}) {
                for (auto& kv : *m) {
                    if (kv.second) fclose(kv.second);
                }
            }
        }

        bool connectTws(const char* host, int port, int clientId) {
            return ptr_begin->eConnect(host, port, clientId);
        }
        
        void setMarketDataType(int type) {
            if (ptr_begin) {
                ptr_begin->reqMarketDataType(type);
                std::cout << "[INFO] Market data type set to: " << type << " (1=Realtime, 3=Delayed)" << std::endl;
            }
        }
        
        void startInfrastructure(const std::string& pubAddress, const std::string& pullAddress){
            int hwm = 10000;
            int linger = 0;

            zmq_pub.set(zmq::sockopt::sndhwm, hwm);
            zmq_pub.set(zmq::sockopt::linger, linger);
            zmq_pub.bind(pubAddress);
            std::cout << "[ZMQ-PUB] Publishing data on " << pubAddress << std::endl;
            
            zmq_pull.set(zmq::sockopt::rcvhwm, hwm);
            zmq_pull.set(zmq::sockopt::linger, linger);
            zmq_pull.bind(pullAddress);
            std::cout << "[ZMQ-PULL] Listening for orders on " << pullAddress << std::endl;
            
            zmq_exec_pub.set(zmq::sockopt::sndhwm, hwm);
            zmq_exec_pub.set(zmq::sockopt::linger, linger);
            zmq_exec_pub.bind("tcp://127.0.0.1:5557");
            std::cout << "[ZMQ-PUB-EXEC] Publishing executions on tcp://127.0.0.1:5557" << std::endl;
            
            zmq_hb_pub.set(zmq::sockopt::sndhwm, hwm);
            zmq_hb_pub.set(zmq::sockopt::linger, linger);
            zmq_hb_pub.bind("tcp://127.0.0.1:5558");
            std::cout << "[ZMQ-PUB-HB] Publishing heartbeats on tcp://127.0.0.1:5558" << std::endl;

            zmq_hb_sub.set(zmq::sockopt::rcvhwm, hwm);
            zmq_hb_sub.set(zmq::sockopt::linger, linger);
            zmq_hb_sub.bind("tcp://127.0.0.1:5559");
            std::cout << "[ZMQ-SUB-HB] Watching strategy liveness on tcp://127.0.0.1:5559"
                      << " (silence limit: " << risk_manager.watchdogTimeoutMs() << "ms)" << std::endl;

            execution_thread = std::thread(&TradingEngine::executionLoop, this);
            thread_drain    = std::thread(&TradingEngine::drainLoop, this);
            thread_heartbeat = std::thread(&TradingEngine::heartbeatLoop, this);
            watchdog_thread = std::thread(&TradingEngine::watchdogLoop, this);

            std::cout << "[PING-PONG] Double buffer active. Swap threshold: "
                      << (SWAP_THRESHOLD * 100) << "% of "
                      << (arenas[0]->getCapacity() / (1024*1024)) << " MB" << std::endl;
        }

        void registerAsset(int tickerId, const std::string& asset_name, const Contract& contract,
                            DataMode mode = DataMode::TICK_L2){
            ticker_names[tickerId] = asset_name;
            contract_map[tickerId] = contract;
            // Mode and trade cache are born HERE, at boot, so the callback thread
            // only does lookups of existing keys (see the maps comment).
            ticker_modes[tickerId] = mode;
            trade_cache[tickerId] = TradeCache{};
            active_book_state[tickerId] = L1Cache{};
            // Pre-populates the position in the registry (boot, single thread).
            //
            // current_positions[id] with a missing key INSERTS, and inserting into
            // an unordered_map can trigger a rehash — rebuilding the entire bucket
            // table. Without this line, the first read of a ticker happened on the
            // orders-loop thread, and could collide with the IBKR callback thread
            // writing to the same map: a data race, with a crash or infinite loop.
            // The more tickers, the more insertions and rehashes — hence the more
            // likely.
            //
            // After this the threads only do lookups of existing keys, and each
            // atomic protects its own value.
            current_positions[tickerId] = 0;

            if (op_mode != OperationMode::LISTEN_ONLY) {
                // ── Multi-Ticker: Partitions data by ticker folder ──
                // data/EURUSD/EURUSD_L2.bin, data/GC/GC_L2.bin, etc. One file per
                // kind of data the asset actually receives: an empty .bin for data
                // that will never arrive just creates junk on disk.
                std::string ticker_dir = "data/" + asset_name;
#ifdef _WIN32
                CreateDirectoryA("data", NULL);
                CreateDirectoryA(ticker_dir.c_str(), NULL);
#else
                mkdir("data", 0755);
                mkdir(ticker_dir.c_str(), 0755);
#endif
                auto openLog = [&](const char* suffix, std::unordered_map<int, FILE*>& target) {
                    std::string filename = ticker_dir + "/" + asset_name + suffix;
                    FILE* f = fopen(filename.c_str(), "ab");
                    if (f) {
                        target[tickerId] = f;
                    } else {
                        std::cout << "[ERROR] Failed to open " << filename << std::endl;
                    }
                };
                if (wantsBook(mode))                                 openLog("_L2.bin", file_map);
                if (mode == DataMode::TICK_L1 || mode == DataMode::BOTH) openLog("_L1.bin", file_map_l1);
                if (wantsTrades(mode))                               openLog("_TRADE.bin", file_map_trade);
            }

            // ── Market subscriptions — ALWAYS active (regardless of the
            //    recording mode). One call per data type the mode requests.
            //
            // reqMktData covers BOTH the top of book (TICK_L1) and the trades
            // (TRADE): they are the same ticks, filtered later by `field` in
            // tickPrice/tickSize. That is why there is a single call.
            if (wantsBook(mode)) {
                std::cout << "[INFO] " << asset_name << ": Level 2 (Deep Book) via reqMktDepth" << std::endl;
                ptr_begin->reqMktDepth(tickerId, contract, 3, TagValueListSPtr());
            }
            if (mode == DataMode::TICK_L1 || wantsTrades(mode)) {
                const char* label = wantsTrades(mode)
                    ? "trades (LAST/LAST_SIZE) for OHLCV bars"
                    : "Level 1 (Top of Book)";
                std::cout << "[INFO] " << asset_name << ": " << label << " via reqMktData" << std::endl;
                ptr_begin->reqMktData(tickerId, contract, "233", false, TagValueListSPtr());
            }
        }

        // Declared mode for the asset. TICK_L2 (the historical default) for an
        // unknown id — find() instead of operator[], which would insert on the hot-path.
        DataMode tickerMode(int tickerId) const {
            auto it = ticker_modes.find(tickerId);
            return (it == ticker_modes.end()) ? DataMode::TICK_L2 : it->second;
        }

        // Forwards the limits read from the JSON to the risk manager. Called at
        // boot, before any thread — after that `limits` is read-only.
        void applyRiskLimits(const PreTradeRiskLimits& updated) {
            risk_manager.applyLimits(updated);
        }

        void requestPositions() {
            if (ptr_begin) {
                ptr_begin->reqPositions();
                std::cout << "[SYSTEM] Position request sent to IBKR." << std::endl;
            }
        }

        // One ExecutionReport on 5557. Called from the orders thread AND the IBKR
        // callback thread, hence the mutex: a zmq socket is not thread-safe.
        void publishExecution(double price, int tickerId, int orderId, int status,
                              int filled, int remaining) {
            ExecutionReport report;
            report.version = IPC_PROTOCOL_VERSION;
            memset(report.reserved, 0, sizeof(report.reserved));
            report.price = price;
            report.tickerId = tickerId;
            report.orderId = orderId;
            report.status = status;
            report.filled = filled;
            report.remaining = remaining;
            zmq::message_t msg(sizeof(report));
            memcpy(msg.data(), &report, sizeof(report));
            std::lock_guard<std::mutex> lock(zmq_exec_mutex);
            zmq_exec_pub.send(msg, zmq::send_flags::dontwait);
        }

        bool orderInFlight(int tickerId) {
            std::lock_guard<std::mutex> lock(order_map_mutex);
            for (const auto& kv : order_info) {
                if (kv.second.tickerId == tickerId) return true;
            }
            return false;
        }

        // --- EXECUTION THREAD (Receives target positions from the strategy) ---
        void executionLoop(){
            int timeout = 1000;
            zmq_pull.set(zmq::sockopt::rcvtimeo, timeout);

            while (engine_running){
                zmq::message_t msg;
                auto recv_ok = zmq_pull.recv(msg, zmq::recv_flags::none);
                if (!recv_ok) continue;

                if (msg.size() != sizeof(TargetPositionRequest)) {
                    std::cout << "\n[IPC ERROR] TargetPositionRequest of " << msg.size()
                              << " bytes, expected " << sizeof(TargetPositionRequest)
                              << ". The strategy's struct layout does not match the engine's "
                              << "(see live/ipc.py and tests/test_ipc.py)." << std::endl;
                    continue;
                }

                // Copy out of the message buffer: the packed struct is read as a
                // value, never through a possibly misaligned pointer.
                TargetPositionRequest req;
                memcpy(&req, msg.data(), sizeof(req));

                if (req.version != IPC_PROTOCOL_VERSION) {
                    std::cout << "[IPC ERROR] Unknown version in TargetPositionRequest: "
                              << (int)req.version << " (expected: " << IPC_PROTOCOL_VERSION
                              << ")" << std::endl;
                    continue;
                }

                // find() and not operator[]: the tickerId comes from the
                // strategy's payload and operator[] INSERTS when the key does not
                // exist — a write to the map structure, on the orders thread,
                // while the watchdog thread may be iterating the same map to
                // flatten. Besides, an id outside the config should never
                // become an order.
                auto it_pos = current_positions.find(req.tickerId);
                auto it_contract = contract_map.find(req.tickerId);
                if (it_pos == current_positions.end() || it_contract == contract_map.end()) {
                    std::cerr << "[IPC ERROR] tickerId " << req.tickerId
                              << " is not registered in the engine. Order DISCARDED."
                              << std::endl;
                    publishExecution(req.price, req.tickerId, 0, EXEC_REJECTED, 0, 0);
                    continue;
                }
                const int current_pos = it_pos->second.load();

                // One order per asset at a time. The position only reflects a fill
                // once the broker reports it; a second target computed before that
                // would size its delta on the stale position and double the trade.
                // Checked BEFORE "already there": with a buy 0 -> +1 working, a target
                // of 0 equals the stale position, and confirming it would tell the
                // strategy it is flat while the buy can still fill.
                if (orderInFlight(req.tickerId)) {
                    std::cout << "[RISK REJECT] Ticker " << req.tickerId
                              << ": an order is still in flight. Target " << req.target_position
                              << " refused; resend after its ExecutionReport." << std::endl;
                    publishExecution(req.price, req.tickerId, 0, EXEC_REJECTED, 0,
                                     std::abs(req.target_position - current_pos));
                    continue;
                }

                if (req.target_position == current_pos) {
                    // Already there: confirm, so a strategy waiting on this target unlocks.
                    publishExecution(req.price, req.tickerId, 0, EXEC_FILLED, 0, 0);
                    std::cout << "[SYSTEM] Target position already reached. Published sync fill for ticker "
                              << req.tickerId << std::endl;
                    continue;
                }

                int quantity = 0;
                std::string order_action;
                std::string reason;
                if (!risk_manager.approveTargetOrder(req.target_position, current_pos, req.price,
                                                     req.orderType, quantity, order_action, reason)) {
                    publishExecution(req.price, req.tickerId, 0, EXEC_REJECTED, 0,
                                     std::abs(req.target_position - current_pos));
                    continue;
                }

                const std::string order_type = (req.orderType == 1) ? "MKT" : "LMT";
                std::cout << "\n >>> [TARGET POSITION] Ticker: " << req.tickerId
                          << " | Current: " << current_pos << " -> Target: " << req.target_position
                          << " | Sending: " << order_action << " " << quantity << " " << order_type << std::endl;

                Order ib_order;
                ib_order.action = order_action;
                ib_order.totalQuantity = quantity;
                ib_order.orderType = order_type;
                if (req.orderType == 2) ib_order.lmtPrice = req.price;
                ib_order.eTradeOnly = false;
                ib_order.firmQuoteOnly = false;

                const int curr_id = next_order_id++;
                risk_manager.registerOrderStart(curr_id);
                {
                    std::lock_guard<std::mutex> lock(order_map_mutex);
                    order_info[curr_id] = OrderInfo{req.tickerId, current_pos, (order_action == "BUY") ? 1 : -1};
                }
                ptr_begin->placeOrder(curr_id, it_contract->second, ib_order);
                std::cout << ">>> Order sent to IBKR!" << std::endl;
            }
        }

        // -- DEAD MAN SWITCH --------------------------------------------
        //
        // Watches two conditions and, on either, flattens the book:
        //   1. the strategy stopped signaling liveness (silence > limit)
        //   2. the day's loss breached the configured limit
        //
        // The second condition used to be evaluated only when AN ORDER ARRIVED,
        // inside approveTargetOrder. A portfolio sinking without new orders was
        // never noticed. Now it is checked on a timer.
        void watchdogLoop() {
            const int timeout_ms = risk_manager.watchdogTimeoutMs();

            while (engine_running.load(std::memory_order_relaxed)) {
                // The 500ms rcvtimeo makes this call return even without a
                // message, allowing silence to be checked periodically.
                zmq::message_t msg;
                auto recv_ok = zmq_hb_sub.recv(msg, zmq::recv_flags::none);

                if (recv_ok && msg.size() == sizeof(StrategyHeartbeat)) {
                    StrategyHeartbeat* hb = static_cast<StrategyHeartbeat*>(msg.data());
                    if (hb->version == IPC_PROTOCOL_VERSION) {
                        auto now = std::chrono::steady_clock::now().time_since_epoch();
                        int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

                        uint32_t prev_seq = last_hb_seq.exchange(hb->seq);
                        if (prev_seq != 0 && hb->seq > prev_seq + 1) {
                            std::cout << "[WATCHDOG] Missed beats: seq jumped from "
                                      << prev_seq << " to " << hb->seq << std::endl;
                        }
                        last_hb_ms.store(now_ms);

                        if (!watchdog_armed.exchange(true)) {
                            std::cout << "[WATCHDOG] First beat received. Watchdog ARMED ("
                                      << timeout_ms << "ms tolerance)." << std::endl;
                        }
                    }
                }

                if (!engine_running.load(std::memory_order_relaxed)) break;

                // Resets the per-day order counter at the configured day boundary,
                // even on a day with no orders at all.
                risk_manager.rollDayIfNeeded();

                // Condition 2 holds even before the strategy starts: if there is
                // an open position from a previous session and the PnL breaches,
                // the engine must act on its own.
                if (risk_manager.dailyLimitBreached()) {
                    std::ostringstream reason;
                    reason << "daily limit breached (PnL " << risk_manager.pnlTotal() << ")";
                    haltAndLiquidate(reason.str());
                    continue;
                }

                // Condition 1 only holds after the first beat — otherwise the
                // engine would flatten during boot, before the strategy exists.
                if (watchdog_armed.load()) {
                    auto now = std::chrono::steady_clock::now().time_since_epoch();
                    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
                    int64_t silence_ms = now_ms - last_hb_ms.load();

                    if (silence_ms > timeout_ms) {
                        std::ostringstream reason;
                        reason << "strategy silent for " << silence_ms << "ms (limit "
                               << timeout_ms << "ms)";
                        haltAndLiquidate(reason.str());
                    }
                }
            }
        }

        // Flattens the book and starts refusing everything. Runs ONCE.
        //
        // Does not terminate the process on purpose: dying would leave TWS alone
        // with positions and no one watching. The engine stays alive, refusing
        // orders and receiving position() callbacks to confirm it flattened.
        void haltAndLiquidate(const std::string& reason) {
            bool expected = false;
            if (!liquidation_triggered.compare_exchange_strong(expected, true)) {
                return;   // another thread is already handling this
            }

            std::cout << std::endl;
            std::cout << "==============================================================" << std::endl;
            std::cout << "[HALT & LIQUIDATE] REASON: " << reason << std::endl;
            std::cout << "==============================================================" << std::endl;

            // (a) block any new order coming from the strategy
            risk_manager.activateKillSwitch();

            // (b) cancel pending orders
            auto pending = risk_manager.activeOrders();
            std::cout << "[HALT] Cancelling " << pending.size() << " pending order(s)." << std::endl;
            for (int orderId : pending) {
                if (ptr_begin) ptr_begin->cancelOrder(orderId);
            }

            // (c) flatten each open position
            //
            // Iterates over a SNAPSHOT, never over the live map: sending an order
            // takes time, and during that time another thread may touch the
            // unordered_map structure. Iterating a map that changes underneath is
            // undefined behavior — and this is the worst possible moment for the
            // engine to crash, because the book would be left exposed with no one
            // watching.
            std::vector<std::pair<int,int>> positions_to_flatten;
            positions_to_flatten.reserve(current_positions.size());
            for (auto& kv : current_positions) {
                positions_to_flatten.emplace_back(kv.first, kv.second.load());
            }

            int flattened = 0;
            for (const auto& kv : positions_to_flatten) {
                int tickerId = kv.first;
                int position = kv.second;
                if (position == 0) continue;

                std::string order_action = (position > 0) ? "SELL" : "BUY";
                int qty = std::abs(position);
                std::cout << "[HALT] Ticker " << tickerId << ": position " << position
                          << " -> sending " << order_action << " " << qty << " at market." << std::endl;
                sendEmergencyOrder(tickerId, order_action, qty);
                flattened++;
            }

            if (flattened == 0) {
                std::cout << "[HALT] No open position to flatten." << std::endl;
            }
            std::cout << "[HALT] Engine stays ALIVE, refusing new orders. "
                      << "Manual restart required to resume trading." << std::endl;
            std::cout << "==============================================================" << std::endl;
        }

        // ATTENTION: the only path in the system authorized to send an order
        // without going through approveTargetOrder.
        //
        // The normal gate refuses everything when the kill switch is on — and
        // Halt turns the kill switch on before flattening. If liquidation used
        // the normal path, the system would prevent itself from closing its own
        // positions: it would have the protection on paper and exposure in the
        // market.
        void sendEmergencyOrder(int tickerId, const std::string& order_action, int qty) {
            auto it = contract_map.find(tickerId);
            if (it == contract_map.end()) {
                std::cerr << "[HALT] ERROR: contract for ticker " << tickerId
                          << " not found. POSITION WAS NOT FLATTENED." << std::endl;
                return;
            }

            Order order;
            order.action = order_action;
            order.totalQuantity = qty;
            order.orderType = "MKT";
            order.eTradeOnly = false;
            order.firmQuoteOnly = false;

            int orderId = next_order_id++;
            risk_manager.registerOrderStart(orderId);
            {
                std::lock_guard<std::mutex> lock(order_map_mutex);
                auto it_pos = current_positions.find(tickerId);
                const int pos_now = (it_pos == current_positions.end()) ? 0 : it_pos->second.load();
                order_info[orderId] = OrderInfo{tickerId, pos_now, (order_action == "BUY") ? 1 : -1};
            }
            if (ptr_begin) ptr_begin->placeOrder(orderId, it->second, order);
        }

        void heartbeatLoop() {
            while (engine_running.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                if (!engine_running.load(std::memory_order_relaxed)) break;

                const char* hb_msg = "HB";
                zmq::message_t msg(2);
                memcpy(msg.data(), hb_msg, 2);
                zmq_hb_pub.send(msg, zmq::send_flags::none);
            }
        }

        // IBKR sends these routine status notices through the same callback as
        // real errors, and it localises their text to whatever Locale the
        // Gateway was installed with. Relying on that text would make our
        // output change language depending on the operator's machine, so the
        // known housekeeping codes get our own English one-liner instead.
        // Anything not listed still prints IBKR's own words verbatim: an
        // unrecognised message is exactly the one nobody should be paraphrasing.
        static const char* noticeText(int code) {
            switch (code) {
                case 2103: return "Market data farm connection is BROKEN";
                case 2104: return "Market data farm connection is OK";
                case 2105: return "Historical data farm connection is BROKEN";
                case 2106: return "Historical data farm connection is OK";
                case 2107: return "Historical data farm is inactive (available on demand)";
                case 2108: return "Market data farm is inactive (available on demand)";
                case 2119: return "Market data farm is connecting";
                case 2158: return "Security definition data farm connection is OK";
                case 2100: return "API client unsubscribed from account data";
                default:   return nullptr;
            }
        }

        virtual void error(const int id, const int errorCode, const IBString errorString) override {
            if (const char* note = noticeText(errorCode)) {
                std::cerr << "[IBKR " << errorCode << "] " << note << std::endl;
                return;
            }

            std::string severity = "[ERROR IBKR]";
            if (id == -1) severity = "[INFO/SYSTEM IBKR]";

            std::cerr << "\n" << std::string(60, '=') << "\n"
                      << severity << "\n"
                      << "Request ID : " << id << "\n"
                      << "Error Code : " << errorCode << "\n"
                      << "Message    : " << (const char*)errorString << "\n"
                      << std::string(60, '=') << "\n" << std::endl;
        }

        virtual void winError(const IBString& str, int lastError) override {
            std::cerr << "\n[WIN/SOCKET ERROR] OS Socket Error: " << (const char*)str 
                      << " | Code: " << lastError << "\n" << std::endl;
        }

        virtual void connectionClosed() override {
            std::cerr << "\n[DISCONNECT] CONNECTION TO THE IB GATEWAY DROPPED ABRUPTLY!\n" << std::endl;
            publishExecution(0.0, -1, -1, EXEC_BROKER_DISCONNECT, 0, 0);
        }

        virtual void OnCatch(const char* MethodName, const long Id) override {
            std::cerr << "\n[FATAL CRASH] TwsApiL0 SWALLOWED AN EXCEPTION!\n"
                      << "Method: " << MethodName << " | Request ID: " << Id << "\n" << std::endl;
        }

        virtual void marketDataType(TickerId reqId, int marketDataType) override {
            std::cerr << "\n[MARKET DATA TYPE] Feed type changed by the broker!\n"
                      << "Request ID: " << reqId << " | New type: " << marketDataType << "\n" << std::endl;
        }

        virtual void nextValidId(OrderId orderId) override {
            // The broker tells us the correct ID and we sync our variable
            next_order_id = orderId;
            std::cout << "[SYSTEM] Synced with IBKR. Next Order ID: " << next_order_id << std::endl;
        }

        virtual void orderStatus(OrderId orderId, const IBString &status, int filled,
                                 int remaining, double avgFillPrice, int /*permId*/, int /*parentId*/,
                                 double /*lastFillPrice*/, int /*clientId*/, const IBString& /*whyHeld*/) override {
            int statusInt = EXEC_OTHER;
            if (status == "Submitted") statusInt = EXEC_SUBMITTED;
            else if (status == "Filled") statusInt = EXEC_FILLED;
            else if (status == "Cancelled") statusInt = EXEC_CANCELLED;
            else if (status == "PreSubmitted") statusInt = EXEC_PRESUBMITTED;
            else if (status == "Inactive") statusInt = EXEC_INACTIVE;
            const bool terminal = statusInt == EXEC_FILLED || statusInt == EXEC_CANCELLED
                               || statusInt == EXEC_INACTIVE;

            // Resolve the asset and the position this order leads to. find() and
            // not operator[], which inserts and would race with the orders thread.
            const int id = static_cast<int>(orderId);
            int orderTickerId = -1;
            int prev_status = EXEC_OTHER;
            bool after_terminal = false;   // a change to an order already reported closed
            bool reopened = false;         // ...that is working again (not terminal)
            {
                std::lock_guard<std::mutex> lock(order_map_mutex);
                OrderInfo info{};
                bool known = false;

                auto it_closed = closed_orders.find(id);
                if (it_closed != closed_orders.end()) {
                    ClosedOrder& c = it_closed->second;
                    // Exact repeat: the strategy already has this report.
                    if (c.status == statusInt && c.filled == filled) return;
                    after_terminal = true;
                    prev_status = c.status;
                    info = c.info;
                    known = true;
                    if (terminal) {
                        c.status = statusInt;   // e.g. Cancelled -> Filled: still closed
                        c.filled = filled;
                    } else {
                        // Working again: back in order_info, so the asset is locked
                        // while it works. Its id stays in the FIFO; eviction tolerates
                        // ids that are no longer (or again) in closed_orders.
                        order_info[id] = info;
                        closed_orders.erase(it_closed);
                        reopened = true;
                    }
                } else {
                    auto it = order_info.find(orderId);
                    if (it != order_info.end()) {
                        info = it->second;
                        known = true;
                        // Terminal state: frees the entry (unblocks the asset for the
                        // next target) and remembers it, to recognize repeats.
                        if (terminal) {
                            order_info.erase(it);
                            closed_orders[id] = ClosedOrder{info, statusInt, filled};
                            closed_orders_fifo.push_back(id);
                            while (closed_orders_fifo.size() > CLOSED_ORDERS_KEPT) {
                                closed_orders.erase(closed_orders_fifo.front());
                                closed_orders_fifo.pop_front();
                            }
                        }
                    }
                }

                if (known) {
                    orderTickerId = info.tickerId;
                    if (filled > 0) {
                        // `filled` is cumulative, and the result is absolute: the
                        // same value whether or not the broker's position() update
                        // for this fill already arrived.
                        auto it_pos = current_positions.find(info.tickerId);
                        if (it_pos != current_positions.end()) {
                            it_pos->second.store(info.pos_at_send + info.direction * filled);
                        }
                    }
                }
            }

            if (after_terminal) {
                std::cerr << "[EXECUTION] WARNING: orderId " << orderId
                          << " changed AFTER a terminal status (" << prev_status << " -> "
                          << statusInt << ", filled " << filled << ")"
                          << (reopened ? ": working again, asset locked until it closes." : ".")
                          << std::endl;
            }
            if (reopened) risk_manager.registerOrderReopened(id);

            if (orderTickerId == -1) {
                // Order the engine did not originate (sent by hand via TWS, or a
                // survivor of a restart). Reported anyway, with id -1, so the
                // strategy can log it instead of being blind.
                std::cerr << "[EXECUTION] WARNING: orderId " << orderId
                          << " has no associated ticker (external order, or predates the restart)."
                          << std::endl;
            }
            if (terminal) risk_manager.registerOrderClosed(orderId);

            publishExecution(avgFillPrice, orderTickerId, static_cast<int>(orderId), statusInt,
                             filled, remaining);

            std::cout << "[EXECUTION] OrderID: " << orderId << " Status: " << status
                      << " (" << statusInt << ") Filled: " << filled << std::endl;
        }

        static bool sameContract(const Contract& a, const Contract& b) {
            if (a.symbol != b.symbol) return false;
            if (!a.secType.empty() && !b.secType.empty() && a.secType != b.secType) return false;
            if (!a.expiry.empty() && !b.expiry.empty() && a.expiry != b.expiry) return false;
            if (!a.currency.empty() && !b.currency.empty() && a.currency != b.currency) return false;
            return true;
        }

        virtual void position(const IBString& account, const Contract& contract, int position, double avgCost) override {
            std::cout << "[IBKR POSITION] Account: " << (const char*)account 
                      << " | Contract: " << (const char*)contract.symbol 
                      << " " << (const char*)contract.secType 
                      << " " << (const char*)contract.currency 
                      << " | Pos: " << position << " | AvgCost: " << avgCost << std::endl;

            int tickerId = -1;
            // Look up the tickerId associated with this contract
            for (auto& kv : contract_map) {
                if (sameContract(kv.second, contract)) {
                    tickerId = kv.first;
                    break;
                }
            }

            if (tickerId != -1) {
                // find(): same reason as the orders loop — do not insert at
                // runtime into a map another thread may be iterating.
                auto it_cp = current_positions.find(tickerId);
                if (it_cp == current_positions.end()) {
                    std::cerr << "[RECON] Position received for tickerId " << tickerId
                              << " not registered. Ignored." << std::endl;
                    return;
                }
                it_cp->second.store(position);
                
                PositionReport report;
                report.version = IPC_PROTOCOL_VERSION;
                memset(report.reserved, 0, sizeof(report.reserved));
                report.tickerId = tickerId;
                report.position = position;
                report.avgCost = avgCost;

                publish("POSITIONS", &report, sizeof(report));

                std::cout << "[RECON] Position received from IBKR: " << contract.symbol 
                          << " | Qtd: " << position << " | Price: " << avgCost << std::endl;
            }
        }

        virtual void updatePortfolio(const Contract& contract, int /*position*/,
                                     double /*marketPrice*/, double /*marketValue*/, double /*averageCost*/,
                                     double unrealizedPNL, double realizedPNL, const IBString& /*accountName*/) override {
            int tickerId = -1;
            for (auto& kv : contract_map) {
                if (sameContract(kv.second, contract)) {
                    tickerId = kv.first;
                    break;
                }
            }

            if (tickerId != -1) {
                std::lock_guard<std::mutex> lock(pnl_mutex);
                realized_pnl[tickerId] = realizedPNL;
                unrealized_pnl[tickerId] = unrealizedPNL;

                double total_realized = 0.0;
                double total_unrealized = 0.0;
                for (auto& kv : realized_pnl) total_realized += kv.second;
                for (auto& kv : unrealized_pnl) total_unrealized += kv.second;
                
                risk_manager.updatePnL(total_realized, total_unrealized);
            }
        }

        virtual void positionEnd() override {
            std::cout << "[RECON] Position sync with IBKR complete." << std::endl;
            for (auto& kv : current_positions) {
                int tickerId = kv.first;
                int pos = kv.second.load();

                PositionReport report;
                report.version = IPC_PROTOCOL_VERSION;
                memset(report.reserved, 0, sizeof(report.reserved));
                report.tickerId = tickerId;
                report.position = pos;
                report.avgCost = 0.0;

                publish("POSITIONS", &report, sizeof(report));
            }
        }

        // ===== RECORDING (hot path side) =====
        // Copies one record into the ACTIVE arena. No disk I/O, no allocation, no
        // lock: the only cost is a memcpy and a few atomics. Only the IBKR callback
        // thread calls this, and only that thread flips `active_arena`, so the
        // arena it writes to cannot be handed to the drain thread mid-copy.
        void record(int tickerId, int32_t kind, const void* payload) {
            if (op_mode == OperationMode::LISTEN_ONLY) return;
            writers_in_record.fetch_add(1);  // seq_cst: pairs with the shutdown flush (Dekker-style)
            if (recording_open.load()) {
                const size_t n = recordPayloadSize(kind);
                const int idx = active_arena.load(std::memory_order_relaxed);
                char* p = static_cast<char*>(arenas[idx]->allocate(sizeof(RecordHeader) + n));
                if (p) {
                    const RecordHeader h{tickerId, kind};
                    memcpy(p, &h, sizeof(h));
                    memcpy(p + sizeof(h), payload, n);
                } else {
                    // Both arenas full: the disk cannot keep up. Drop the RECORD,
                    // never the live message, and make the loss visible.
                    if (dropped_records.fetch_add(1) % 10000 == 0) {
                        std::cout << "[RECORDING WARN] Arena full, records dropped so far: "
                                  << dropped_records.load() << std::endl;
                    }
                }
                trySwapArena();
            }
            writers_in_record.fetch_sub(1);
        }

        // Called after each record. Flips the arenas when the active one is 90%
        // full or SWAP_INTERVAL has passed with data in it — and only if the drain
        // thread finished the previous cycle.
        void trySwapArena() {
            const int idx = active_arena.load(std::memory_order_relaxed);
            ArenaAllocator* active = arenas[idx];
            const size_t used = active->getUsedBytes();
            if (used == 0) return;

            const auto now = std::chrono::steady_clock::now();
            const bool full = used >= static_cast<size_t>(active->getCapacity() * SWAP_THRESHOLD);
            if (!full && now - last_swap < SWAP_INTERVAL) return;

            if (drain_pending.load(std::memory_order_acquire)) {
                // The other arena is still being written to disk. Keep filling
                // this one; warn once per episode, not once per tick.
                if (full && !backpressure_warned.exchange(true)) {
                    std::cout << "[PING-PONG WARN] Back-pressure: previous drain did not finish." << std::endl;
                }
                return;
            }
            backpressure_warned = false;

            active_arena.store(1 - idx, std::memory_order_release);
            last_swap = now;
            {
                std::lock_guard<std::mutex> lk(drain_mutex);
                arena_to_drain = idx;
                drain_pending.store(true, std::memory_order_release);
            }
            drain_cv.notify_one();
        }

        // ===== RECORDING (drain side) =====
        // Walks one arena record by record and appends each payload to its asset's
        // file. Runs on the drain thread, or on the destructor's thread after the
        // drain thread has been joined — never on two threads at once.
        void writeArenaToDisk(ArenaAllocator* arena) {
            const char* p = arena->getBuffer();
            const char* end = p + arena->getUsedBytes();
            while (p + sizeof(RecordHeader) <= end) {
                RecordHeader h;
                memcpy(&h, p, sizeof(h));
                p += sizeof(h);
                const size_t n = recordPayloadSize(h.kind);
                if (p + n > end) break;   // cannot happen: allocate() is all-or-nothing

                std::unordered_map<int, FILE*>* target =
                    h.kind == REC_TRADE ? &file_map_trade :
                    h.kind == REC_L2    ? &file_map : &file_map_l1;
                auto it = target->find(h.tickerId);
                if (it != target->end() && it->second) {
                    fwrite(p, 1, n, it->second);
                }
                p += n;
            }
            for (auto* m : {&file_map, &file_map_l1, &file_map_trade}) {
                for (auto& kv : *m) {
                    if (kv.second) fflush(kv.second);
                }
            }
        }

        // ===== DRAIN THREAD (Background) =====
        void drainLoop() {
            while (true) {
                std::unique_lock<std::mutex> lk(drain_mutex);
                drain_cv.wait(lk, [this]{
                    return drain_pending.load(std::memory_order_acquire)
                           || !engine_running.load(std::memory_order_relaxed);
                });
                if (!drain_pending.load(std::memory_order_acquire)) break;  // shutdown, nothing to do
                const int idx = arena_to_drain;
                lk.unlock(); // Release the mutex before the heavy I/O

                ArenaAllocator* arena = arenas[idx];
                writeArenaToDisk(arena);
                arena->reset();
                drain_pending.store(false, std::memory_order_release);

                if (!engine_running.load(std::memory_order_relaxed)) break;
            }
        }

        // Shutdown: stop new records, wait for any callback still mid-copy, then
        // write whatever is left — the arena waiting for the drain thread first
        // (it is older), then the active one.
        void flushRecordingOnShutdown() {
            if (op_mode == OperationMode::LISTEN_ONLY) return;
            recording_open.store(false);   // seq_cst, see record()
            while (writers_in_record.load() != 0) {
                std::this_thread::yield();
            }
            const int active = active_arena.load();
            if (drain_pending.load()) {
                writeArenaToDisk(arenas[1 - active]);
                arenas[1 - active]->reset();
                drain_pending = false;
            }
            writeArenaToDisk(arenas[active]);
            arenas[active]->reset();
            if (dropped_records.load() > 0) {
                std::cout << "[RECORDING] " << dropped_records.load()
                          << " record(s) were dropped because both arenas were full." << std::endl;
            }
        }

        // ===== PUBLISHING =====
        // Multipart [topic, payload] on 5555. The mutex serializes the callback
        // paths that share zmq_pub; it is uncontended in practice (they all run on
        // the IBKR callback thread) and is the only lock on the market-data path.
        void publish(const std::string& topic, const void* data, size_t size) {
            zmq::message_t msg_topic(topic.data(), topic.size());
            zmq::message_t msg_data(data, size);
            std::lock_guard<std::mutex> lock(zmq_pub_mutex);
            zmq_pub.send(msg_topic, zmq::send_flags::sndmore);
            zmq_pub.send(msg_data, zmq::send_flags::none);
        }

        static int64_t nowMicros() {
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }

        // ===== CALLBACKS — HOT PATH =====
        void onDepthUpdate(TickerId id, int position, int operation, int side, double price, int size) {
            auto it_name = ticker_names.find(id);
            if (it_name == ticker_names.end()) return;

            L2Update update;
            update.timestamp = nowMicros();
            update.price = price;
            update.position = position;
            update.operation = operation;
            update.side = side;
            update.size = size;

            record(static_cast<int>(id), REC_L2, &update);
            publish(it_name->second + "_L2", &update, sizeof(update));
        }

        virtual void updateMktDepth(TickerId id, int position, int operation, int side, double price, int size) override {
            onDepthUpdate(id, position, operation, side, price, size);
        }

        virtual void updateMktDepthL2(TickerId id, int position, IBString /*marketMaker*/, int operation, int side, double price, int size) override {
            onDepthUpdate(id, position, operation, side, price, size);
        }

        // Top of book changed (bid/ask price or size): record + publish the side
        // as a level-0 L2Update, but only once both its price and size are known.
        void onTopOfBook(TickerId tickerId, bool is_bid) {
            auto it_name = ticker_names.find(tickerId);
            auto it_book = active_book_state.find(tickerId);
            if (it_name == ticker_names.end() || it_book == active_book_state.end()) return;
            const BookSide& side = is_bid ? it_book->second.bid : it_book->second.ask;
            if (side.price <= 0.0 || side.size <= 0) return;

            L2Update update;
            update.timestamp = nowMicros();
            update.price = side.price;
            update.position = 0;              // Top of Book is level 0
            update.operation = 1;             // 1 = Update
            update.side = is_bid ? 1 : 0;     // 1 for Bid, 0 for Ask
            update.size = side.size;

            record(static_cast<int>(tickerId), REC_L1, &update);
            publish(it_name->second + "_L2", &update, sizeof(update));
        }

        virtual void tickPrice(TickerId tickerId, TickType field, double price, int /*canAutoExecute*/) override {
            const DataMode mode = tickerMode(tickerId);

            // ── TRADE: LAST (4) and Delayed Last (68) ────────────────────
            // Only CACHES. The publisher is tickSize(LAST_SIZE), because price and
            // size are two callbacks of the SAME trade: publishing on both would
            // emit the trade twice and DOUBLE the bar's volume.
            //
            // This is the inverse of what the book code does just below (there
            // both sides publish, because bid and ask are independent states and
            // republishing is just redundancy). The asymmetry is deliberate.
            if (field == 4 || field == 68) {
                if (!wantsTrades(mode)) return;
                if (price <= 0.0) return;   // -1 = no trade in the session
                auto it = trade_cache.find(tickerId);
                if (it == trade_cache.end()) return;
                it->second.last_price = price;
                it->second.last_ts_us = nowMicros();
                return;
            }

            // Pure TRADE mode does not publish book: the reqMktData that brings
            // the trades also brings bid/ask, and emitting `_L2` that no one
            // subscribed to would be just traffic. TICK_L1 keeps publishing,
            // which is the --L1 flag's behavior from the start.
            if (mode == DataMode::TRADE) return;

            // Bid (1), Ask (2) and the Delayed versions (66, 67)
            if (field == 1 || field == 2 || field == 66 || field == 67) {
                const bool is_bid = (field == 1 || field == 66);
                auto it_book = active_book_state.find(tickerId);
                if (it_book == active_book_state.end()) return;
                (is_bid ? it_book->second.bid : it_book->second.ask).price = price;
                onTopOfBook(tickerId, is_bid);
            }
        }

        virtual void tickSize(TickerId tickerId, TickType field, int size) override {
            const DataMode mode = tickerMode(tickerId);

            // ── TRADE: LAST_SIZE (5) and Delayed Last Size (71) ──────────
            // 71 only arrives if the IB client maps it (see lib/client/README.md,
            // "Delayed market data needs a 5-line patch").
            // Closes the pair with the price cached in tickPrice(LAST) and PUBLISHES.
            // The engine aggregates nothing: it collects, stamps and sends. The
            // 1-minute bar is built by Python (live/bar_aggregator.py), which
            // can thus be tested without a compiler, without a network and without
            // a broker.
            if (field == 5 || field == 71) {
                if (!wantsTrades(mode)) return;
                if (size <= 0) return;

                auto it_cache = trade_cache.find(tickerId);
                if (it_cache == trade_cache.end()) return;
                const double px = it_cache->second.last_price;
                if (px <= 0.0) return;   // the matching LAST has not arrived yet

                TradeUpdate ev{};
                ev.version = IPC_PROTOCOL_VERSION;
                // Stamp of the LAST (when the PRICE arrived), not the LAST_SIZE.
                // The two callbacks are from the same trade and are microseconds
                // apart, but on a minute boundary that decides the bar: LAST at
                // 10:00:59.9998 with LAST_SIZE at 10:01:00.0002 would put the trade
                // in the wrong bar. The aggregator closes the bar by the timestamp
                // that comes from here, so the error would look like its fault when
                // it isn't.
                ev.timestamp_us = it_cache->second.last_ts_us > 0 ? it_cache->second.last_ts_us : nowMicros();
                ev.price = px;
                ev.size = size;
                ev.padding = 0;

                auto it_name = ticker_names.find(tickerId);
                if (it_name == ticker_names.end()) return;
                record(static_cast<int>(tickerId), REC_TRADE, &ev);
                publish(it_name->second + "_TRADE", &ev, sizeof(ev));
                return;
            }

            // See the note in tickPrice: pure TRADE mode does not publish book.
            if (mode == DataMode::TRADE) return;

            // Bid Size (0), Ask Size (3) and the Delayed versions (69, 70)
            if (field == 0 || field == 3 || field == 69 || field == 70) {
                const bool is_bid = (field == 0 || field == 69);
                auto it_book = active_book_state.find(tickerId);
                if (it_book == active_book_state.end()) return;
                (is_bid ? it_book->second.bid : it_book->second.ask).size = size;
                onTopOfBook(tickerId, is_bid);
            }
        }
};
