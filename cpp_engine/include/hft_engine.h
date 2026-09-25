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

class HFTEngine: public EWrapperL0 {
    private:
        OperationMode op_mode;  // Recording mode
        EClientL0* ptr_begin;

        // ===== PING-PONG BUFFER (Double Arena) =====
        // Two identical arenas. The ingestion thread writes ONLY to the active arena.
        // When the active arena fills up (>= THRESHOLD), we do an atomic swap and the
        // drain thread writes the inactive arena to disk and resets it.
        ArenaAllocator* arenas[2];
        std::atomic<int> active_arena;    // 0 or 1 — index of the write arena

        // Lock-free signaling to the drain thread
        std::mutex              drain_mutex;
        std::condition_variable drain_cv;
        std::atomic<bool>       drain_pendente;  // flag: is there an arena to drain?
        int                     arena_to_drain; // index of the arena to be drained
        std::thread             thread_drain;

        // File and name routers
        std::unordered_map<int, FILE*> file_map;
        std::unordered_map<int, std::string> ticker_names;
        std::unordered_map<int, Contract> contract_map;
        std::unordered_map<int, FILE*> file_map_l1;
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

        // Position cache (Target Position Routing)
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

        // orderId -> tickerId (IPC v2)
        //
        // The IBKR orderStatus callback delivers only the orderId, but the Python
        // side needs to know the asset. This map is the only bridge between the two.
        //
        // CONCURRENCY: written by the orders-loop thread (ZMQ) and read by the
        // IBKR callback thread. Every access goes under order_map_mutex, and the
        // lookup uses find() — never operator[], which INSERTS when the key does
        // not exist and would cause a concurrent write to the map structure.
        std::unordered_map<int, int> order_to_ticker;
        std::mutex order_map_mutex;

        zmq::context_t zmq_ctx;
        zmq::socket_t zmq_pub;
        zmq::socket_t zmq_pull;
        zmq::socket_t zmq_exec_pub;
        zmq::socket_t zmq_hb_pub;
        zmq::socket_t zmq_hb_sub;   // strategy liveness signal (5559)
        std::mutex zmq_pub_mutex;

        std::thread execution_thread;
        std::thread thread_heartbeat;
        std::atomic<bool> engine_running;
        std::atomic<int> next_order_id;
        ExecutionRiskManager risk_manager;

        // Swap threshold: 90% of the arena capacity
        static constexpr double SWAP_THRESHOLD = 0.90;

    public:
        void requestPortfolioUpdates() {
            if (ptr_begin) {
                ptr_begin->reqAccountUpdates(true, "");
            }
        }

        HFTEngine(ArenaAllocator* arena_a, ArenaAllocator* arena_b, OperationMode mode) 
            : op_mode(mode), active_arena(0), drain_pendente(false), arena_to_drain(-1),
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

        ~HFTEngine(){
            engine_running = false;

            // Wake the drain thread so it can shut down
            drain_cv.notify_one();
            if(thread_drain.joinable()) thread_drain.join();

            if(execution_thread.joinable()) execution_thread.join();
            if(thread_heartbeat.joinable()) thread_heartbeat.join();
            if(watchdog_thread.joinable()) watchdog_thread.join();

            if (ptr_begin) {
                ptr_begin->eDisconnect();
                delete ptr_begin;
            }
            if (op_mode != OperationMode::LISTEN_ONLY) {
                for (auto& kv : file_map) {
                    if (kv.second) fclose(kv.second);
                }
                for (auto& kv : file_map_l1) {
                    if (kv.second) fclose(kv.second);
                }
                for (auto& kv : file_map_trade) {
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

            execution_thread = std::thread(&HFTEngine::executionLoop, this);
            thread_drain    = std::thread(&HFTEngine::drainLoop, this);
            thread_heartbeat = std::thread(&HFTEngine::heartbeatLoop, this);
            watchdog_thread = std::thread(&HFTEngine::watchdogLoop, this);

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
                // data/EURUSD/EURUSD_L2.bin, data/GC/GC_L2.bin, etc.
                std::string ticker_dir = "data/" + asset_name;
#ifdef _WIN32
                CreateDirectoryA(ticker_dir.c_str(), NULL);
#else
                mkdir(ticker_dir.c_str(), 0755);
#endif

                // Setup L2 — disk write (partitioned)
                std::string l2_filename = ticker_dir + "/" + asset_name + "_L2.bin";
                FILE* f_l2 = fopen(l2_filename.c_str(),"ab");
                if (f_l2) {
                    file_map[tickerId] = f_l2;
                } else {
                    std::cout << "[ERROR] Failed to open " << l2_filename << std::endl;
                }

                // Setup L1 — disk write (partitioned)
                std::string l1_filename = ticker_dir + "/" + asset_name + "_L1.bin";
                FILE* f_l1 = fopen(l1_filename.c_str(),"ab");
                if (f_l1) {
                    file_map_l1[tickerId] = f_l1;
                } else {
                    std::cout << "[ERROR] Failed to open " << l1_filename << std::endl;
                }

                // Setup TRADE — only for those subscribed to trades. Opening an
                // empty .bin for an asset that will never receive a trade just
                // creates junk on disk.
                if (wantsTrades(mode)) {
                    std::string trade_filename = ticker_dir + "/" + asset_name + "_TRADE.bin";
                    FILE* f_trade = fopen(trade_filename.c_str(),"ab");
                    if (f_trade) {
                        file_map_trade[tickerId] = f_trade;
                    } else {
                        std::cout << "[ERROR] Failed to open " << trade_filename << std::endl;
                    }
                }
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
                const char* rotulo = wantsTrades(mode)
                    ? "trades (LAST/LAST_SIZE) for OHLCV bars"
                    : "Level 1 (Top of Book)";
                std::cout << "[INFO] " << asset_name << ": " << rotulo << " via reqMktData" << std::endl;
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

        // --- EXECUTION THREAD (Receives orders from Python) ---
        void executionLoop(){
            int timeout = 1000;
            zmq_pull.set(zmq::sockopt::rcvtimeo, timeout);

            while (engine_running){
                zmq::message_t msg;
                auto recv_ok = zmq_pull.recv(msg, zmq::recv_flags::none);

                if (recv_ok) {
                    // --- MEMORY DEBUG MODE ENABLED ---
                    if (msg.size() != sizeof(TargetPositionRequest)) {
                        std::cout << "\n[ZMQ MEMORY ERROR] PADDING ALERT!" << std::endl;
                        std::cout << "-> Python sent    : " << msg.size() << " bytes." << std::endl;
                        std::cout << "-> C++ struct has : " << sizeof(TargetPositionRequest) << " bytes." << std::endl;
                        std::cout << "Fix: adjust the struct.pack mask in Python to match the C++ layout.\n" << std::endl;
                        continue;
                    }

                    TargetPositionRequest* req = static_cast<TargetPositionRequest*>(msg.data());

                    // ── IPC v1: Protocol version validation ──
                    if (req->version != IPC_PROTOCOL_VERSION) {
                        std::cout << "[IPC ERROR] Unknown version in TargetPositionRequest: "
                                  << (int)req->version << " (expected: " << IPC_PROTOCOL_VERSION
                                  << ")" << std::endl;
                        continue;
                    }
                    
                    // find() and not operator[]: the tickerId comes from the
                    // Python payload and operator[] INSERTS when the key does not
                    // exist — a write to the map structure, on the orders thread,
                    // while the watchdog thread may be iterating the same map to
                    // flatten. Besides, an id outside the config should never
                    // become an order.
                    auto it_pos = current_positions.find(req->tickerId);
                    if (it_pos == current_positions.end()) {
                        std::cerr << "[IPC ERROR] tickerId " << req->tickerId
                                  << " is not registered in the engine. Order DISCARDED."
                                  << std::endl;
                        continue;
                    }
                    int current_pos = it_pos->second.load();
                    int quantity = 0;
                    std::string order_action = "";

                    if (!risk_manager.approveTargetOrder(req->tickerId, req->target_position, current_pos, req->price, req->orderType, quantity, order_action)){
                        if (req->target_position == current_pos) {
                            ExecutionReport report;
                            report.version = IPC_PROTOCOL_VERSION;
                            memset(report.reserved, 0, sizeof(report.reserved));
                            report.price = req->price;
                            report.tickerId = req->tickerId;
                            report.orderId = 0;
                            report.status = 2; // FILLED
                            report.filled = 0;
                            report.remaining = 0;
                            zmq::message_t msg_exec(&report, sizeof(report));
                            zmq_exec_pub.send(msg_exec, zmq::send_flags::dontwait);
                            std::cout << "[SYSTEM] Target position already reached. Published sync fill for ticker " << req->tickerId << std::endl;
                        }
                        continue;
                    }

                    std::string tipo = (req->orderType == 1) ? "MKT":"LMT";

                    std::cout << "\n >>> [TARGET POSITION] Ticker: " << req->tickerId
                              << " | Current: " << current_pos << " -> Target: " << req->target_position
                              << " | Sending: " << order_action << " " << quantity << " " << tipo << std::endl; 

                    Order ib_order;
                    ib_order.action = order_action;
                    ib_order.totalQuantity = quantity;
                    ib_order.orderType = tipo;
                    if (req->orderType == 2) ib_order.lmtPrice = req->price;
                    ib_order.eTradeOnly = false;
                    ib_order.firmQuoteOnly = false;

                    if (contract_map.find(req->tickerId) != contract_map.end()){
                        int curr_id = next_order_id++;
                        risk_manager.registerOrderStart(curr_id);

                        // IPC v2: stores which asset this order belongs to, so
                        // orderStatus (which only receives orderId) can inform Python.
                        {
                            std::lock_guard<std::mutex> lock(order_map_mutex);
                            order_to_ticker[curr_id] = req->tickerId;
                        }
                        ptr_begin->placeOrder(curr_id, contract_map[req->tickerId], ib_order);
                        std::cout << ">>> Order sent to IBKR!" << std::endl;
                    }
                }
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
                    int64_t silencio = now_ms - last_hb_ms.load();

                    if (silencio > timeout_ms) {
                        std::ostringstream reason;
                        reason << "strategy silent for " << silencio << "ms (limit "
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
            auto pendentes = risk_manager.activeOrders();
            std::cout << "[HALT] Cancelling " << pendentes.size() << " pending order(s)." << std::endl;
            for (int orderId : pendentes) {
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

            int zeradas = 0;
            for (const auto& kv : positions_to_flatten) {
                int tickerId = kv.first;
                int position = kv.second;
                if (position == 0) continue;

                std::string order_action = (position > 0) ? "SELL" : "BUY";
                int qty = std::abs(position);
                std::cout << "[HALT] Ticker " << tickerId << ": position " << position
                          << " -> sending " << order_action << " " << qty << " at market." << std::endl;
                sendEmergencyOrder(tickerId, order_action, qty);
                zeradas++;
            }

            if (zeradas == 0) {
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
                order_to_ticker[orderId] = tickerId;
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
            if (const char* nota = noticeText(errorCode)) {
                std::cerr << "[IBKR " << errorCode << "] " << nota << std::endl;
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
            ExecutionReport report;
            report.version = IPC_PROTOCOL_VERSION;
            memset(report.reserved, 0, sizeof(report.reserved));
            report.price = 0.0;
            report.tickerId = -1;
            report.orderId = -1;
            report.filled = 0;
            report.remaining = 0;
            report.status = 9; // Status 9: Broker Disconnect Alert
            zmq::message_t msg(&report, sizeof(report));
            zmq_exec_pub.send(msg, zmq::send_flags::dontwait);
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
                                 int remaining, double avgFillPrice, int permId, int parentId,
                                 double lastFillPrice, int clientId, const IBString& whyHeld) override {
            ExecutionReport report;
            report.version = IPC_PROTOCOL_VERSION;
            memset(report.reserved, 0, sizeof(report.reserved));
            report.price = avgFillPrice;
            report.orderId = orderId;

            // IPC v2: resolves this order's asset. find() and not operator[],
            // which inserts and would cause a concurrent write with the orders thread.
            int orderTickerId = -1;
            {
                std::lock_guard<std::mutex> lock(order_map_mutex);
                auto it = order_to_ticker.find(orderId);
                if (it != order_to_ticker.end()) orderTickerId = it->second;
            }
            report.tickerId = orderTickerId;

            if (orderTickerId == -1) {
                // Order the engine did not originate (sent by hand via TWS, or a
                // survivor of a restart). Reported anyway, with id -1, so Python
                // can log it instead of being blind.
                std::cerr << "[EXECUTION] WARNING: orderId " << orderId
                          << " has no associated ticker (external order, or predates the restart)."
                          << std::endl;
            }

            int statusInt = 0;
            if (status == "Submitted") statusInt = 1;
            else if (status == "Filled") statusInt = 2;
            else if (status == "Cancelled") statusInt = 3;
            else if (status == "PreSubmitted") statusInt = 4;
            else if (status == "Inactive") statusInt = 5;
            report.status = statusInt;

            report.filled = filled;
            report.remaining = remaining;

            if (statusInt == 2 || statusInt == 3 || statusInt == 5) { // Filled, Cancelled, Inactive
                risk_manager.registerOrderClosed(orderId);
                // Terminal state: frees the entry so the map does not grow unbounded.
                std::lock_guard<std::mutex> lock(order_map_mutex);
                order_to_ticker.erase(orderId);
            }

            zmq::message_t msg(sizeof(ExecutionReport));
            memcpy(msg.data(), &report, sizeof(ExecutionReport));
            zmq_exec_pub.send(msg, zmq::send_flags::none);
            
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

                std::string topico = "POSITIONS";
                zmq::message_t msg_topico(topico.size());
                memcpy(msg_topico.data(), topico.c_str(), topico.size());

                zmq::message_t msg_data(sizeof(PositionReport));
                memcpy(msg_data.data(), &report, sizeof(PositionReport));

                {
                    std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                    zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                    zmq_pub.send(msg_data, zmq::send_flags::none);
                }

                std::cout << "[RECON] Position received from IBKR: " << contract.symbol 
                          << " | Qtd: " << position << " | Price: " << avgCost << std::endl;
            }
        }

        virtual void updatePortfolio(const Contract& contract, int position,
                                     double marketPrice, double marketValue, double averageCost,
                                     double unrealizedPNL, double realizedPNL, const IBString& accountName) override {
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

                std::string topico = "POSITIONS";
                zmq::message_t msg_topico(topico.size());
                memcpy(msg_topico.data(), topico.c_str(), topico.size());

                zmq::message_t msg_data(sizeof(PositionReport));
                memcpy(msg_data.data(), &report, sizeof(PositionReport));

                {
                    std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                    zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                    zmq_pub.send(msg_data, zmq::send_flags::none);
                }
            }
        }

        // ===== ATOMIC SWAP LOGIC =====
        // Called after each allocation on the hot path.
        // If the active arena crossed the threshold, swap and wake the drain thread.
        // GUARANTEE: the swap operation is a single atomic store — zero locks on the hot path.
        void trySwapArena() {
            int idx = active_arena.load(std::memory_order_relaxed);
            ArenaAllocator* active = arenas[idx];

            size_t threshold_bytes = static_cast<size_t>(active->getCapacity() * SWAP_THRESHOLD);
            if (active->getUsedBytes() < threshold_bytes) return;

            // Only swap if the drain thread already finished the previous cycle
            if (drain_pendente.load(std::memory_order_acquire)) {
                // The reserve arena is still being drained — we cannot swap.
                // In production this indicates back-pressure; log for diagnostics.
                std::cout << "[PING-PONG WARN] Back-pressure: previous drain did not finish." << std::endl;
                return;
            }

            int next_idx = 1 - idx; // 0 -> 1,  1 -> 0
            active_arena.store(next_idx, std::memory_order_release); // <<< ATOMIC SWAP

            // Signal the drain thread to process the arena that became inactive
            {
                std::lock_guard<std::mutex> lk(drain_mutex);
                arena_to_drain = idx;
                drain_pendente.store(true, std::memory_order_release);
            }
            drain_cv.notify_one();

            std::cout << "[PING-PONG] Swap done: arena " << idx
                      << " -> drain | arena " << next_idx << " -> active ("
                      << active->getUsedBytes() << " bytes acumulados)" << std::endl;
        }

        // ===== DRAIN THREAD (Background) =====
        // Waits for a signal, writes the inactive arena's contents to disk (.bin)
        // and calls resetar() to leave it clean for the next cycle.
        void drainLoop() {
            while (engine_running.load(std::memory_order_relaxed)) {
                std::unique_lock<std::mutex> lk(drain_mutex);
                drain_cv.wait(lk, [this]{
                    return drain_pendente.load(std::memory_order_acquire) 
                           || !engine_running.load(std::memory_order_relaxed);
                });

                if (!engine_running.load(std::memory_order_relaxed)) break;

                int idx = arena_to_drain;
                lk.unlock(); // Release the mutex before the heavy I/O

                ArenaAllocator* arena = arenas[idx];
                size_t bytes = arena->getUsedBytes();

                if (bytes > 0 && op_mode != OperationMode::LISTEN_ONLY) {
                    // Write the arena's entire contiguous block into a dump file
                    auto now = std::chrono::system_clock::now();
                    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()
                    ).count();

                    std::string dump_filename = "data/arena_" + std::to_string(idx)
                                          + "_" + std::to_string(epoch_ms) + ".bin";

                    FILE* f = fopen(dump_filename.c_str(), "wb");
                    if (f) {
                        fwrite(arena->getBuffer(), 1, bytes, f);
                        fclose(f);
                        std::cout << "[DRAIN] Arena " << idx << " -> " << dump_filename
                                  << " (" << bytes << " bytes)" << std::endl;
                    } else {
                        std::cout << "[DRAIN ERROR] Failed to open " << dump_filename << std::endl;
                    }
                }

                // Clear the arena for reuse in the next cycle
                arena->resetar();
                drain_pendente.store(false, std::memory_order_release);
            }
        }

        // ===== CALLBACKS — HOT PATH (lock-free) =====
        virtual void updateMktDepth(TickerId id, int position, int operation, int side, double price, int size) override {
            if (ticker_names.find(id) == ticker_names.end()) return;

            L2Update update;
            update.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count(); 
            update.price = price;
            update.position = position;
            update.operation = operation;
            update.side = side;
            update.size = size;

            std::string topico = ticker_names[id] + "_L2";
            
            zmq::message_t msg_topico(topico.size());
            memcpy(msg_topico.data(), topico.c_str(), topico.size());

            zmq::message_t msg_data(sizeof(L2Update));
            memcpy(msg_data.data(), &update, sizeof(L2Update));

            {
                std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                zmq_pub.send(msg_data, zmq::send_flags::none);
            }
        }

        virtual void updateMktDepthL2(TickerId id, int position, IBString marketMaker, int operation, int side, double price, int size) override {
            if (ticker_names.find(id) == ticker_names.end()) return;

            L2Update update;
            update.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count(); 
            update.price = price;
            update.position = position;
            update.operation = operation;
            update.side = side;
            update.size = size;

            std::string topico = ticker_names[id] + "_L2";
            
            zmq::message_t msg_topico(topico.size());
            memcpy(msg_topico.data(), topico.c_str(), topico.size());

            zmq::message_t msg_data(sizeof(L2Update));
            memcpy(msg_data.data(), &update, sizeof(L2Update));

            {
                std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                zmq_pub.send(msg_data, zmq::send_flags::none);
            }
        }

        virtual void tickPrice(TickerId tickerId, TickType field, double price, int canAutoExecute) override {
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
                it->second.last_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                return;
            }

            // Filter Bid (1), Ask (2) and the Delayed versions (66, 67)
            //
            // Pure TRADE mode does not publish book: the reqMktData that brings
            // the trades also brings bid/ask, and emitting `_L2` that no one
            // subscribed to would be just traffic. TICK_L1 keeps publishing,
            // which is the --L1 flag's behavior from the start.
            if (mode == DataMode::TRADE) return;

            if (field == 1 || field == 2 || field == 66 || field == 67) {
                bool is_bid = (field == 1 || field == 66);
                
                // Update the in-memory state (cache)
                if (is_bid) {
                    active_book_state[tickerId].bid.price = price;
                } else {
                    active_book_state[tickerId].ask.price = price;
                }

                // Fetch the current size from cache. If zero, wait for the matching tickSize
                int current_size = is_bid ? active_book_state[tickerId].bid.size : active_book_state[tickerId].ask.size;
                if (current_size <= 0) return;

                // Allocate on the ACTIVE arena (atomic read — lock-free)
                int idx = active_arena.load(std::memory_order_relaxed);
                L2Update* ptr_l2 = (L2Update*) arenas[idx]->alocar(sizeof(L2Update));
                if (ptr_l2 == nullptr) return;

                ptr_l2->timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                
                ptr_l2->price = price;
                ptr_l2->position = 0; // Top of Book is level 0
                ptr_l2->operation = 1; // 1 = Update
                ptr_l2->side = is_bid ? 1 : 0; // 1 for Bid, 0 for Ask
                ptr_l2->size = current_size; // Intact size from cache

                // Conditional write (L1) -> uses file_map_l1 without depending on the L2 file_map
                if (op_mode != OperationMode::LISTEN_ONLY) {
                    auto it_l1 = file_map_l1.find(tickerId);
                    if (it_l1 != file_map_l1.end() && it_l1->second) {
                        fwrite(ptr_l2, sizeof(L2Update), 1, it_l1->second);
                    }
                }

                std::string topico = ticker_names[tickerId] + "_L2";
                
                zmq::message_t msg_topico(topico.size());
                memcpy(msg_topico.data(), topico.c_str(), topico.size());

                zmq::message_t msg_data(sizeof(L2Update));
                memcpy(msg_data.data(), ptr_l2, sizeof(L2Update));

                {
                    std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                    zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                    zmq_pub.send(msg_data, zmq::send_flags::none);
                }

                // Check whether to swap arenas
                trySwapArena();
            }
        }

        virtual void tickSize(TickerId tickerId, TickType field, int size) override {
            const DataMode mode = tickerMode(tickerId);

            // ── TRADE: LAST_SIZE (5) and Delayed Last Size (71) ──────────
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
                double px = it_cache->second.last_price;
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
                ev.timestamp_us = it_cache->second.last_ts_us;
                if (ev.timestamp_us <= 0) {
                    ev.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();
                }
                ev.price = px;
                ev.size = size;
                ev.padding = 0;

                if (op_mode != OperationMode::LISTEN_ONLY) {
                    auto it_f = file_map_trade.find(tickerId);
                    if (it_f != file_map_trade.end() && it_f->second) {
                        fwrite(&ev, sizeof(TradeUpdate), 1, it_f->second);
                    }
                }

                auto it_nome = ticker_names.find(tickerId);
                if (it_nome == ticker_names.end()) return;
                std::string topico = it_nome->second + "_TRADE";

                zmq::message_t msg_topico(topico.size());
                memcpy(msg_topico.data(), topico.c_str(), topico.size());
                zmq::message_t msg_data(sizeof(TradeUpdate));
                memcpy(msg_data.data(), &ev, sizeof(TradeUpdate));

                {
                    std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                    zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                    zmq_pub.send(msg_data, zmq::send_flags::none);
                }
                return;
            }

            // See the note in tickPrice: pure TRADE mode does not publish book.
            if (mode == DataMode::TRADE) return;

            // Filter Bid Size (0), Ask Size (3) and the Delayed versions (69, 70)
            if (field == 0 || field == 3 || field == 69 || field == 70) {
                bool is_bid = (field == 0 || field == 69);
                
                // Update the in-memory state (cache)
                if (is_bid) {
                    active_book_state[tickerId].bid.size = size;
                } else {
                    active_book_state[tickerId].ask.size = size;
                }

                // Fetch the current price from cache. If <= 0, wait for the matching tickPrice
                double current_price = is_bid ? active_book_state[tickerId].bid.price : active_book_state[tickerId].ask.price;
                if (current_price <= 0.0) return;

                // Allocate on the ACTIVE arena (atomic read — lock-free)
                int idx = active_arena.load(std::memory_order_relaxed);
                L2Update* ptr_l2 = (L2Update*) arenas[idx]->alocar(sizeof(L2Update));
                if (ptr_l2 == nullptr) return;

                ptr_l2->timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                
                ptr_l2->price = current_price; // Intact price from cache
                ptr_l2->position = 0; // Top of Book is level 0
                ptr_l2->operation = 1; // 1 = Update
                ptr_l2->side = is_bid ? 1 : 0; // 1 for Bid, 0 for Ask
                ptr_l2->size = size; // The size received in the tick (also in cache)

                // Conditional write (L1) -> uses file_map_l1 without depending on the L2 file_map
                if (op_mode != OperationMode::LISTEN_ONLY) {
                    auto it_l1 = file_map_l1.find(tickerId);
                    if (it_l1 != file_map_l1.end() && it_l1->second) {
                        fwrite(ptr_l2, sizeof(L2Update), 1, it_l1->second);
                    }
                }

                std::string topico = ticker_names[tickerId] + "_L2";
                
                zmq::message_t msg_topico(topico.size());
                memcpy(msg_topico.data(), topico.c_str(), topico.size());

                zmq::message_t msg_data(sizeof(L2Update));
                memcpy(msg_data.data(), ptr_l2, sizeof(L2Update));

                {
                    std::lock_guard<std::mutex> lock(zmq_pub_mutex);
                    zmq_pub.send(msg_topico, zmq::send_flags::sndmore);
                    zmq_pub.send(msg_data, zmq::send_flags::none);
                }

                // Check whether to swap arenas
                trySwapArena();
            }
        }
};
