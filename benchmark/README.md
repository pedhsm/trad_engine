# benchmark

A reproducible micro-benchmark of the `core_math` primitives.

## Run

From the repo root:

```bash
python -m benchmark.run_benchmark
python -m benchmark.run_benchmark --bars 500000 --repeat 7
```

It synthesizes N random-walk bars and times each primitive (SMA/EMA/RSI/ATR, the
VPIN pipeline, and the depth-weighted book imbalance), reporting the **median**
wall-clock of a few repeats (after a warmup run) plus a throughput in bars/second.

## How to read it

- These are **Python-level** timings on **your** machine. They are an honest
  *relative* picture of where compute goes — not a latency SLA, not a hardware claim.
- The number worth quoting is a **ratio** (e.g. C++ mirror vs Python), never the
  absolute milliseconds, which depend entirely on the box you ran it on.

## Optional: the C++ mirror

`core_math/cpp/` is a C++ mirror of the microstructure primitives, exposed with
C linkage so it loads via `ctypes`. If you compile it into a shared library
(`micro.so` / `micro.dll` / `micro.dylib`) next to the sources, the benchmark
detects and loads it, ready for a side-by-side comparison. Nothing here requires
it — the Python-only run is complete on its own.

## Latency of the live path

`benchmark/latency.py` is the other half: not throughput over a batch of bars, but
the time from one tick to one order, split into the C++ engine hot path, the Python
strategy framework, the example signal, and the full round trip between the engine
and a strategy in another process. See the "Latency" section of the main README for
what each row covers and what it leaves out (the broker and the network).

```bash
python -m benchmark.latency            # (A) needs lib/client + ZeroMQ, else skipped
python -m benchmark.latency --skip-cpp
```

