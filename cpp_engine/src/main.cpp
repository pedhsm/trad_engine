#include "hft_engine.h"

std::atomic<bool> global_running{true};

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        std::cout << "\n[SHUTDOWN] Ctrl+C received. Shutting the system down..." << std::endl;
        global_running = false;
        return TRUE;
    }
    return FALSE;
}
#else
void consoleHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n[SHUTDOWN] Signal received. Shutting the system down..." << std::endl;
        global_running = false;
    }
}
#endif

// ==========================================
// ENTRY POINT AND CONTROL (CLI)
// ==========================================
int main(int argc, char* argv[]) {
#ifdef _WIN32
    if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
        std::cout << "Failed to register the console control handler." << std::endl;
        return 1;
    }
#else
    signal(SIGINT, consoleHandler);
    signal(SIGTERM, consoleHandler);
#endif

    // ==========================================
    // CLI ARGUMENT PARSING (Phase 1)
    // ==========================================
    // Usage: .\bin\hft_engine.exe --live --mode <listen_only|record_local|record_cloud>
    //        .\bin\hft_engine.exe --historical --mode <listen_only|record_local|record_cloud>
    // --mode is REQUIRED.

    if (argc < 4) {
        std::cout << "Usage: .\\bin\\hft_engine.exe [--live | --historical] --mode <listen_only|record_local|record_cloud>" << std::endl;
    std::cout << "     [--config <path.json>] [--data-mode <l2|l1|trade|both>]" << std::endl;
    std::cout << "     --data-mode default: l2 (book). Use 'trade' for OHLCV bars." << std::endl;
        return 1;
    }

    std::string exec_mode;       // --live or --historical
    std::string record_mode;   // listen_only, record_local, record_cloud
    std::string config_path = "config.json";  // default; see examples/engine_config.example.json

    // DEFAULT data mode for the run. TICK_L2 keeps the historical behavior:
    // whoever passes nothing keeps receiving exactly what they received before.
    // --L1 still works, now as an alias for --data-mode l1.
    DataMode data_mode = DataMode::TICK_L2;
    std::string data_mode_txt = "l2";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--live" || arg == "--historical") {
            exec_mode = arg;
        } else if (arg == "--mode" && (i + 1) < argc) {
            record_mode = argv[++i];
        } else if (arg == "--config" && (i + 1) < argc) {
            config_path = argv[++i];
        } else if (arg == "--L1") {
            data_mode = DataMode::TICK_L1;
            data_mode_txt = "l1";
        } else if (arg == "--data-mode" && (i + 1) < argc) {
            data_mode_txt = argv[++i];
        }
    }

    // Translation of the data-mode text. An unknown value KILLS the startup
    // instead of falling into a silent default: starting up subscribed to the
    // wrong data would only be discovered when the strategy went mute waiting
    // for bars.
    auto translateMode = [](const std::string& txt, DataMode& out) -> bool {
        if (txt == "l2")         { out = DataMode::TICK_L2; return true; }
        if (txt == "l1")         { out = DataMode::TICK_L1; return true; }
        if (txt == "trade")      { out = DataMode::TRADE;   return true; }
        if (txt == "both")       { out = DataMode::BOTH;    return true; }
        return false;
    };
    if (!translateMode(data_mode_txt, data_mode)) {
        std::cout << "[FATAL] Invalid --data-mode: '" << data_mode_txt << "'" << std::endl;
        std::cout << "Valid options: l2 (book, default) | l1 (top of book) | "
                  << "trade (trades/OHLCV) | both" << std::endl;
        return 1;
    }

    // Validation: --live or --historical is required
    if (exec_mode.empty()) {
        std::cout << "[FATAL] Neither --live nor --historical was given." << std::endl;
        std::cout << "Usage: .\\bin\\hft_engine.exe [--live | --historical] --mode <listen_only|record_local|record_cloud>" << std::endl;
    std::cout << "     [--config <path.json>] [--data-mode <l2|l1|trade|both>]" << std::endl;
    std::cout << "     --data-mode default: l2 (book). Use 'trade' for OHLCV bars." << std::endl;
        return 1;
    }

    // Validation: --mode is required and must have a valid value
    OperationMode op_mode;
    if (record_mode == "listen_only") {
        op_mode = OperationMode::LISTEN_ONLY;
    } else if (record_mode == "record_local") {
        op_mode = OperationMode::RECORD_LOCAL;
    } else if (record_mode == "record_cloud") {
        op_mode = OperationMode::RECORD_CLOUD;
    } else {
        std::cout << "[FATAL] Invalid --mode: '" << record_mode << "'" << std::endl;
        std::cout << "Valid options: listen_only | record_local | record_cloud" << std::endl;
        return 1;
    }

    // ==========================================
    // STARTUP BANNER
    // ==========================================
    std::cout << "========================================" << std::endl;
    // ASCII hyphen on purpose: the Windows console renders a UTF-8 em dash as
    // mojibake ("HFT Engine v1.2 <garbage> IPC v3"), and this banner is the
    // first thing anyone watching a recording sees.
    std::cout << "  HFT Engine v1.2 - IPC v" << IPC_PROTOCOL_VERSION << std::endl;
    std::cout << "  Execution : " << exec_mode << std::endl;
    std::cout << "  Recording : " << record_mode << std::endl;
    std::cout << "  Data      : " << data_mode_txt << " (default; the JSON may override per asset)" << std::endl;
    std::cout << "========================================" << std::endl;

    if (op_mode == OperationMode::LISTEN_ONLY) {
        std::cout << "[Listen Only] No .bin file will be opened or written." << std::endl;
    } else if (op_mode == OperationMode::RECORD_CLOUD) {
        std::cout << "[Cloud Mode] Files will be flagged for async upload." << std::endl;
    }

    // PING-PONG: Two identical arenas of 50 MB each
    ArenaAllocator arena_A(1024 * 1024 * 50); // 50 MB
    ArenaAllocator arena_B(1024 * 1024 * 50); // 50 MB

    HFTEngine engine(&arena_A, &arena_B, op_mode);

    // ══════════════════════════════════════════════════════════════
    // CONFIGURATION LOADING (JSON Config-Driven)
    // ══════════════════════════════════════════════════════════════
    std::cout << "[CONFIG] Loading configuration from: " << config_path << std::endl;
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cout << "[FATAL] Configuration file not found: " << config_path << std::endl;
        return 1;
    }

    json config;
    try {
        config_file >> config;
    } catch (const json::parse_error& e) {
        std::cout << "[FATAL] Invalid JSON in " << config_path << ": " << e.what() << std::endl;
        return 1;
    }
    config_file.close();

    // Minimal validation
    if (!config.contains("tickers") || !config["tickers"].is_array() || config["tickers"].empty()) {
        std::cout << "[FATAL] Key 'tickers' missing or empty in the JSON." << std::endl;
        return 1;
    }
    if (!config.contains("contracts") || !config["contracts"].is_object()) {
        std::cout << "[FATAL] Key 'contracts' missing in the JSON." << std::endl;
        return 1;
    }

    // ══════════════════════════════════════════════════════════════
    // RISK LIMITS — from the JSON, no longer hard-coded
    // ══════════════════════════════════════════════════════════════
    // The engine ignored `risk_params` entirely: the operator would write
    // max_daily_loss_usd: 1500 and the real cut happened at the struct's 2000.
    // Each key is OPTIONAL and falls back to the previous default, so an old
    // config keeps behaving exactly as before — but now what is written in the
    // file is what applies.
    PreTradeRiskLimits risk_limits_cfg;   // born with the historical defaults
    if (config.contains("risk_params") && config["risk_params"].is_object()) {
        const auto& rp = config["risk_params"];
        risk_limits_cfg.max_daily_loss_usd   = rp.value("max_daily_loss_usd",
                                                    risk_limits_cfg.max_daily_loss_usd);
        risk_limits_cfg.max_lot_size         = rp.value("max_lot_size",
                                                    risk_limits_cfg.max_lot_size);
        risk_limits_cfg.max_orders_per_day   = rp.value("max_orders_per_day",
                                                    risk_limits_cfg.max_orders_per_day);
        risk_limits_cfg.max_concurrent_orders= rp.value("max_concurrent_orders",
                                                    risk_limits_cfg.max_concurrent_orders);
        risk_limits_cfg.watchdog_timeout_ms  = rp.value("watchdog_timeout_ms",
                                                    risk_limits_cfg.watchdog_timeout_ms);
        risk_limits_cfg.day_reset_utc_hour   = rp.value("day_reset_utc_hour",
                                                    risk_limits_cfg.day_reset_utc_hour);
        if (risk_limits_cfg.day_reset_utc_hour < 0 || risk_limits_cfg.day_reset_utc_hour > 23) {
            std::cout << "[FATAL] risk_params.day_reset_utc_hour must be 0..23." << std::endl;
            return 1;
        }

        if (risk_limits_cfg.max_daily_loss_usd <= 0 || risk_limits_cfg.max_lot_size <= 0
            || risk_limits_cfg.max_orders_per_day <= 0
            || risk_limits_cfg.max_concurrent_orders <= 0) {
            std::cout << "[FATAL] risk_params holds a value <= 0. A null limit "
                      << "disables protection instead of tightening it." << std::endl;
            return 1;
        }
        // Too short flattens on a network hiccup; the strategy beats every 1s.
        if (risk_limits_cfg.watchdog_timeout_ms < 2000) {
            std::cout << "[FATAL] watchdog_timeout_ms=" << risk_limits_cfg.watchdog_timeout_ms
                      << " is too short. The strategy beats every 1000ms; below "
                      << "2000ms a network hiccup flattens the book." << std::endl;
            return 1;
        }
    } else {
        std::cout << "[WARN] JSON has no 'risk_params'. Using the engine's default limits."
                  << std::endl;
    }
    engine.applyRiskLimits(risk_limits_cfg);

    // Builds the contract list from the JSON
    struct TickerConfig {
        std::string name;       // Friendly name (e.g.: "EURUSD", "GC")
        Contract contract;      // IBKR contract
        DataMode mode;          // Which data this asset needs (Phase 6)
    };
    std::vector<TickerConfig> active_tickers;

    for (const auto& ticker_name : config["tickers"]) {
        std::string tk = ticker_name.get<std::string>();

        if (!config["contracts"].contains(tk)) {
            std::cout << "[FATAL] Ticker '" << tk << "' listed in 'tickers' but missing from 'contracts'." << std::endl;
            return 1;
        }

        const auto& ct = config["contracts"][tk];
        std::string symbol   = ct.value("symbol",   tk);
        std::string secType  = ct.value("secType",  "STK");
        std::string exchange = ct.value("exchange", "SMART");
        std::string currency = ct.value("currency", "USD");
        std::string expiry   = ct.value("expiry",   "");

        Contract c = Contracts::FromConfig(symbol, secType, exchange, currency, expiry);

        // Per-asset data mode: the JSON can override the command-line default.
        // This is the real case of a mixed portfolio — an asset traded by
        // contract (bars) alongside another traded by VPIN (book).
        DataMode tick_mode = data_mode;
        std::string tick_mode_txt = ct.value("data_mode", data_mode_txt);
        if (!translateMode(tick_mode_txt, tick_mode)) {
            std::cout << "[FATAL] contracts." << tk << ".data_mode invalid: '"
                      << tick_mode_txt << "' (use l2 | l1 | trade | both)" << std::endl;
            return 1;
        }

        active_tickers.push_back({tk, c, tick_mode});
        std::cout << "[CONFIG] Ticker " << active_tickers.size() << ": " << tk
                  << " (" << secType << " @ " << exchange;
        if (!expiry.empty()) std::cout << " exp=" << expiry;
        std::cout << ", data=" << tick_mode_txt << ")" << std::endl;
    }

    std::cout << "[CONFIG] Total active tickers: " << active_tickers.size() << std::endl;

    if (exec_mode == "--live") {
        std::cout << "--- STARTING ENGINE: LIVE MODE ---" << std::endl;
        
        engine.startInfrastructure("tcp://127.0.0.1:5555", "tcp://127.0.0.1:5556");

        if (engine.connectTws("127.0.0.1", 4002, 99)) {
            // Enable delayed data fallback (Type 3 = Delayed, Type 4 = Delayed Frozen)
            engine.setMarketDataType(3);

            // Registers all tickers from the JSON dynamically
            for (size_t i = 0; i < active_tickers.size(); ++i) {
                int tickerId = static_cast<int>(i + 1);
                engine.registerAsset(tickerId, active_tickers[i].name,
                                     active_tickers[i].contract, active_tickers[i].mode);
            }
            
            // Reconciliation (N-06)
            engine.requestPositions();

            // Daily PnL subscription
            std::cout << "[SYSTEM] Requesting continuous portfolio updates (PnL)..." << std::endl;
            engine.requestPortfolioUpdates();
            
            std::cout << "Press Ctrl+C to stop." << std::endl;
            while(global_running) { 
#ifdef _WIN32
                Sleep(100); 
#else
                usleep(100000); 
#endif
            }
        } else {
            std::cout << "ERROR: Failed to connect to the IB Gateway." << std::endl;
        }
    } 
    else if (exec_mode == "--historical") {
        std::cout << "--- STARTING ENGINE: HISTORICAL MODE ---" << std::endl;
        std::cout << "Not implemented yet..." << std::endl;
    }

    return 0;
}
