// Integration harness for HFTEngine, with NO broker connection.
//
// Needs the IB client in lib/client and ZeroMQ, so tests/test_cpp_engine.py only
// runs it when both are present (it is skipped in CI). It drives the engine the
// way the broker and a strategy would:
//
//   1. recording: feeds market-data callbacks straight into the engine, destroys
//      it, and checks the per-ticker .bin files byte for byte;
//   2. orders: sends TargetPositionRequests over the real ZMQ sockets and fakes
//      the broker's orderStatus callbacks, checking the in-flight lock, the
//      REJECTED report and that a fill updates the position without waiting for
//      the broker's position() callback.
//
// Must run from an empty working directory (it writes data/ there). Uses the
// engine's fixed localhost ports 5555-5559.
#include "hft_engine.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); std::fflush(stdout); std::_Exit(1); } } while (0)

static long fileSize(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fclose(f);
    return n;
}

template <typename T>
static T readRecord(const char* path, long index) {
    T out{};
    FILE* f = std::fopen(path, "rb");
    CHECK(f != nullptr);
    std::fseek(f, index * static_cast<long>(sizeof(T)), SEEK_SET);
    CHECK(std::fread(&out, sizeof(T), 1, f) == 1);
    std::fclose(f);
    return out;
}

static Contract stock(const char* sym) {
    return Contracts::FromConfig(sym, "STK", "SMART", "USD");
}

static void testRecording() {
    // 256 KB arenas hold the whole burst (~4k records of 36-40 bytes) without
    // hitting the drop path; the pause makes the next record trigger a TIME-based
    // swap, so records reach the files through both the drain thread and the
    // final flush.
    ArenaAllocator a(1 << 18), b(1 << 18);
    {
        HFTEngine engine(&a, &b, OperationMode::RECORD_LOCAL);
        engine.registerAsset(1, "BOOK", stock("BOOK"), DataMode::TICK_L2);
        engine.registerAsset(2, "TOP", stock("TOP"), DataMode::TICK_L1);
        engine.registerAsset(3, "TRD", stock("TRD"), DataMode::TRADE);
        engine.startInfrastructure("tcp://127.0.0.1:5555", "tcp://127.0.0.1:5556");  // starts the drain thread

        // 3000 depth updates for BOOK (the old engine never wrote these).
        for (int i = 0; i < 3000; ++i) {
            engine.updateMktDepth(1, i % 3, 1, i % 2, 100.0 + i * 0.01, 10 + i);
            if (i == 1499) std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        }
        // Top of book for TOP: price then size on each side -> 1 record per side
        // once both halves are known, then 1 per update.
        engine.tickPrice(2, (TickType)1, 50.0, 0);    // bid px (size unknown: no record)
        engine.tickSize(2, (TickType)0, 7);           // bid size -> record
        engine.tickPrice(2, (TickType)2, 50.5, 0);    // ask px
        engine.tickSize(2, (TickType)3, 9);           // ask size -> record
        engine.tickPrice(2, (TickType)1, 50.1, 0);    // bid px update -> record
        // Trades for TRD: LAST then LAST_SIZE = one trade.
        for (int i = 0; i < 500; ++i) {
            engine.tickPrice(3, (TickType)4, 20.0 + i, 0);
            engine.tickSize(3, (TickType)5, 1 + i);
        }
        // TRADE mode ignores book ticks; TICK_L2 mode ignores trades.
        engine.tickPrice(3, (TickType)1, 19.0, 0);
        engine.tickSize(3, (TickType)0, 1);
        engine.tickSize(1, (TickType)5, 3);
    }   // destructor: final flush of whatever is still in the arenas

    CHECK(fileSize("data/BOOK/BOOK_L2.bin") == 3000L * (long)sizeof(L2Update));
    CHECK(fileSize("data/BOOK/BOOK_L1.bin") == -1);         // not opened for a pure-depth asset
    CHECK(fileSize("data/TOP/TOP_L1.bin") == 3L * (long)sizeof(L2Update));
    CHECK(fileSize("data/TOP/TOP_L2.bin") == -1);
    CHECK(fileSize("data/TRD/TRD_TRADE.bin") == 500L * (long)sizeof(TradeUpdate));
    CHECK(fileSize("data/TRD/TRD_L1.bin") == -1);

    L2Update last_depth = readRecord<L2Update>("data/BOOK/BOOK_L2.bin", 2999);
    CHECK(last_depth.size == 10 + 2999 && last_depth.position == 2999 % 3);
    L2Update top_ask = readRecord<L2Update>("data/TOP/TOP_L1.bin", 1);
    CHECK(top_ask.side == 0 && top_ask.price == 50.5 && top_ask.size == 9);
    L2Update top_bid = readRecord<L2Update>("data/TOP/TOP_L1.bin", 2);
    CHECK(top_bid.side == 1 && top_bid.price == 50.1 && top_bid.size == 7);
    TradeUpdate t = readRecord<TradeUpdate>("data/TRD/TRD_TRADE.bin", 499);
    CHECK(t.version == IPC_PROTOCOL_VERSION && t.price == 20.0 + 499 && t.size == 500);
    std::printf("recording: ok\n");
}

// --- orders ------------------------------------------------------------------

static zmq::message_t targetMsg(int tickerId, int target) {
    TargetPositionRequest r{};
    r.version = IPC_PROTOCOL_VERSION;
    r.tickerId = tickerId;
    r.target_position = target;
    r.orderType = 1;
    zmq::message_t m(sizeof(r));
    memcpy(m.data(), &r, sizeof(r));
    return m;
}

static ExecutionReport nextReport(zmq::socket_t& sub) {
    zmq::message_t m;
    auto ok = sub.recv(m, zmq::recv_flags::none);   // rcvtimeo set by the caller
    CHECK(ok && m.size() == sizeof(ExecutionReport));
    ExecutionReport r;
    memcpy(&r, m.data(), sizeof(r));
    return r;
}

static void testOrders() {
    ArenaAllocator a(1 << 16), b(1 << 16);
    HFTEngine engine(&a, &b, OperationMode::LISTEN_ONLY);
    PreTradeRiskLimits lim;
    lim.max_lot_size = 1;
    engine.applyRiskLimits(lim);
    engine.registerAsset(1, "ES", stock("ES"), DataMode::TRADE);
    engine.startInfrastructure("tcp://127.0.0.1:5555", "tcp://127.0.0.1:5556");

    zmq::context_t ctx(1);
    zmq::socket_t push(ctx, zmq::socket_type::push);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    push.set(zmq::sockopt::linger, 0);
    sub.set(zmq::sockopt::linger, 0);
    sub.set(zmq::sockopt::subscribe, "");
    sub.set(zmq::sockopt::rcvtimeo, 3000);
    push.connect("tcp://127.0.0.1:5556");
    sub.connect("tcp://127.0.0.1:5557");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // PUB/SUB join

    // 0 -> +1: an order goes to the (disconnected) broker, id 1000.
    push.send(targetMsg(1, 1), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Before any fill, a second target is refused: an order is in flight.
    push.send(targetMsg(1, -1), zmq::send_flags::none);
    ExecutionReport r = nextReport(sub);
    CHECK(r.status == EXEC_REJECTED && r.tickerId == 1 && r.orderId == 0 && r.remaining == 1);

    // The broker fills order 1000. No position() callback follows.
    engine.orderStatus(1000, "Filled", 1, 0, 101.0, 0, 0, 101.0, 0, "");
    r = nextReport(sub);
    CHECK(r.status == EXEC_FILLED && r.tickerId == 1 && r.orderId == 1000 && r.filled == 1);

    // Target +1 again: the engine already knows it is at +1 -> sync fill, no order.
    push.send(targetMsg(1, 1), zmq::send_flags::none);
    r = nextReport(sub);
    CHECK(r.status == EXEC_FILLED && r.orderId == 0 && r.tickerId == 1);

    // +1 -> -1 is 2 lots with max_lot_size 1: refused by the risk gate, and said so.
    push.send(targetMsg(1, -1), zmq::send_flags::none);
    r = nextReport(sub);
    CHECK(r.status == EXEC_REJECTED && r.remaining == 2);

    // Unknown ticker: refused, not silently dropped.
    push.send(targetMsg(42, 1), zmq::send_flags::none);
    r = nextReport(sub);
    CHECK(r.status == EXEC_REJECTED && r.tickerId == 42);

    // A cancelled order with a partial fill leaves the position where the fill put it.
    push.send(targetMsg(1, 0), zmq::send_flags::none);          // +1 -> 0, id 1001
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    engine.orderStatus(1001, "Cancelled", 0, 1, 0.0, 0, 0, 0.0, 0, "");
    r = nextReport(sub);
    CHECK(r.status == EXEC_CANCELLED && r.orderId == 1001);
    push.send(targetMsg(1, 1), zmq::send_flags::none);          // still +1 -> sync fill
    r = nextReport(sub);
    CHECK(r.status == EXEC_FILLED && r.orderId == 0);

    push.close();
    sub.close();
    std::printf("orders: ok\n");
}

int main() {
    testRecording();
    testOrders();
    std::printf("engine harness: all checks passed\n");
    std::fflush(stdout);
    return 0;
}
