// Engine hot path: time from a broker trade callback (LAST + LAST_SIZE) to the
// TradeUpdate being recorded in the arena and published on ZMQ, per tick.
//
// Built and run by benchmark/latency.py when lib/client and ZeroMQ are available.
// Prints one line per percentile; runs in RECORD_LOCAL mode (so the arena path is
// included) from whatever working directory the caller chose.
#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100000;
    ArenaAllocator a(64 << 20), b(64 << 20);
    std::vector<long long> samples;
    samples.reserve(n);
    {
        TradingEngine engine(&a, &b, OperationMode::RECORD_LOCAL);
        engine.registerAsset(1, "BENCH", Contracts::FromConfig("BENCH", "STK", "SMART", "USD"),
                             DataMode::TRADE);
        engine.startInfrastructure("tcp://127.0.0.1:5555", "tcp://127.0.0.1:5556");

        // A live subscriber draining 5555, so publish() does real work instead of
        // dropping messages that nobody listens to.
        zmq::context_t ctx(1);
        zmq::socket_t sub(ctx, zmq::socket_type::sub);
        sub.set(zmq::sockopt::subscribe, "");
        sub.set(zmq::sockopt::rcvtimeo, 200);
        sub.set(zmq::sockopt::linger, 0);
        sub.connect("tcp://127.0.0.1:5555");
        std::atomic<bool> stop{false};
        std::thread drain([&] {
            while (!stop) {
                zmq::message_t m;
                (void)sub.recv(m, zmq::recv_flags::none);
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        for (int i = 0; i < n + 1000; ++i) {   // first 1000 = warmup, discarded
            const auto t0 = std::chrono::steady_clock::now();
            engine.tickPrice(1, (TickType)4, 100.0 + (i % 100) * 0.01, 0);   // LAST
            engine.tickSize(1, (TickType)5, 1 + i % 7);                       // LAST_SIZE -> record + publish
            const auto t1 = std::chrono::steady_clock::now();
            if (i >= 1000) samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        stop = true;
        drain.join();
        sub.close();
    }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double q) { return samples[std::min(samples.size() - 1, size_t(q * samples.size()))]; };
    std::printf("n %zu\np50 %lld\np99 %lld\np999 %lld\nmax %lld\n",
                samples.size(), pct(0.50), pct(0.99), pct(0.999), samples.back());
    return 0;
}
