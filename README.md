# trad_engine

A self-hosted, end-to-end algorithmic-trading engine you can take as a **base** and
build a desk on top of — not a toy, not a bloated framework. A low-latency C++ market
data / execution core, a pure-Python `core_math` layer that mirrors it under parity
tests, a backtest engine, and — the part most open engines skip — a **serious
validation harness** (Monte Carlo permutation tests, walk-forward, point-in-time
invariants) so the numbers you get out are ones you can actually trust.

> **What this is not.** This is engine + method. It ships with *example* strategies
> (textbook moving-average / RSI / Donchian, and a microstructure template). It does
> **not** ship anyone's alpha — bring your own signal; the engine is agnostic to it.

---

## Why it might be worth your time

Three design decisions do most of the work here, and they are the reason the code is
worth reading even if you never run it:

1. **A Python reference and a C++ mirror, locked by parity tests.**
   `core_math` exists twice: a readable Python implementation (the ground truth) and a
   fast C++ implementation (the hot path). A parity test asserts they produce the same
   numbers, bit for bit. You get C++ speed in production without giving up a
   researchable, debuggable reference — and you can never silently drift the two apart.

2. **Models frozen as data, not as pickles.**
   A trained linear/logistic classifier is frozen into ~30 plain floats (coefficients,
   scaler stats, feature order, threshold) and applied with a sigmoid and no sklearn
   dependency (`core_math/meta_model.py`). That makes a decision **reproducible** across
   processes and across library versions: a separate verifier can reproduce the exact
   same probability — the same decision fingerprint — as the trainer. A pickled model
   cannot promise that.

3. **Causality is a first-class invariant, not a hope.**
   Every primitive documents that the value at time *t* uses only data at or before *t*.
   The validation layer includes point-in-time invariant checks so lookahead leakage
   fails loudly in CI instead of quietly inflating a backtest.

---

## Layout

```
trad_engine/
├── cpp_engine/          # low-latency C++ core: market-data ingest, IPC, execution
│   ├── include/         #   arena allocator, IPC structs, contract factory
│   └── src/
├── lib/client/          # (you populate this — see lib/client/README.md)
├── core_math/           # pure, causal compute primitives
│   ├── bars_math.py     #   returns, SMA/EMA, ATR, RSI, z-score, realized vol …
│   ├── micro_math.py    #   BVC signed volume, volume buckets, VPIN, TIB
│   ├── labeling.py      #   triple-barrier labeling (López de Prado)
│   ├── meta_model.py    #   apply a frozen linear model as data
│   └── cpp/             #   the C++ mirror of the microstructure math
├── backtest/
│   ├── engine/          # the backtest loop + trade metrics
│   ├── validation/      # MCPT, walk-forward, bar-permutation, PIT invariants
│   └── strategies/      # example strategies (bring your own for real use)
├── examples/            # runnable end-to-end paper-trading demo
└── benchmark/           # reproducible latency / throughput numbers
```

## Requirements

- **Python** ≥ 3.11 — `pip install -r requirements.txt` (numpy, pandas, scipy).
- **C++17** toolchain + CMake, only if you build `cpp_engine`.
- **Interactive Brokers TWS API** — *not vendored here* (see below).

### The IBKR API is not included, on purpose

`cpp_engine` talks to Interactive Brokers through the TWS API. Interactive Brokers'
API code is **copyrighted and cannot be redistributed** ("All rights reserved", IB API
License), so this repo does not ship it. Populate `lib/client/` yourself — instructions
and the credit to the community C++ wrapper this integration was built against are in
[`lib/client/README.md`](lib/client/README.md).

## Quickstart

```bash
pip install -r requirements.txt

# run the end-to-end paper-trading demo (replay feed → bars → example strategy
# → simulated fills → journal), no broker or market data account needed:
python -m examples.paper_demo

# validate a strategy the honest way (Monte Carlo permutation test):
python -m backtest.validation.mcpt_runner --help
```

## License

MIT — see [LICENSE](LICENSE). Third-party components keep their own licenses (the IBKR
TWS API you supply is **not** MIT and is not redistributed by this repo).
