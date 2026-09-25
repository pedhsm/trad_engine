"""Latency guards: catastrophe detectors, not performance targets.

Latency depends on the machine (and CI runners are shared and noisy), so these
ceilings sit ~10x or more above what a laptop measures (benchmark/latency.py). They
only catch a gross regression — a pandas call creeping back into the per-bar path, a
sleep or a blocking call in the loop — never a few percent. For real numbers, run
`python -m benchmark.latency`.
"""
import pytest

from benchmark.latency import bench_strategy_compute


def test_python_strategy_path_has_no_gross_regression():
    framework, signal = bench_strategy_compute(1000)
    assert framework["p50"] < 100.0, framework   # measured ~4 us
    assert signal["p50"] < 100.0, signal         # measured ~10-30 us; was ~200 us with pandas per bar


def test_round_trip_has_no_gross_regression():
    pytest.importorskip("zmq")
    from benchmark.latency import bench_round_trip
    rt = bench_round_trip(300)
    assert rt["p50"] < 20_000.0, rt              # measured ~0.3 ms; 20 ms means something blocks
