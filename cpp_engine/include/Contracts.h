#pragma once
#include "../../lib/client/Contract.h"
#include <string>

class Contracts {
public:
    // Creates contracts for Brazilian stocks automatically
    static Contract B3Stock(const std::string& ticker) {
        Contract c;
        c.symbol = ticker;
        c.secType = "STK";
        c.currency = "BRL";
        c.exchange = "SMART"; // Or BVMF, depending on your account configuration
        return c;
    }

    // Creates contracts for US stocks
    static Contract USStock(const std::string& ticker) {
        Contract c;
        c.symbol = ticker;
        c.secType = "STK";
        c.currency = "USD";
        c.exchange = "SMART";
        return c;
    }

    // ── Factory for Commodity / Macro Futures ──
    static Contract Future(const std::string& symbol,
                           const std::string& exchange,
                           const std::string& currency,
                           const std::string& expiry) {
        Contract c;
        c.symbol   = symbol;
        c.secType  = "FUT";
        c.exchange = exchange;
        c.currency = currency;
        c.expiry   = expiry;  // TWS API: "expiry" field (Contract.h L82)
        return c;
    }

    // ── Generic factory built from JSON fields ──
    // Supports CASH (forex), STK (stocks), FUT (futures) and any future secType.
    static Contract FromConfig(const std::string& symbol,
                               const std::string& secType,
                               const std::string& exchange,
                               const std::string& currency,
                               const std::string& expiry = "") {
        Contract c;
        c.symbol   = symbol;
        c.secType  = secType;
        c.exchange = exchange;
        c.currency = currency;
        if (!expiry.empty()) {
            c.expiry = expiry;
        }
        return c;
    }
};