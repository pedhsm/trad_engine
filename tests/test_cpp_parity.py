"""Parity: the C++ mirror (core_math/cpp) must reproduce the Python reference.

The tolerance is 1e-12, not bitwise equality: pandas and the C++ loops accumulate
sums in a different order, so the last bit legitimately differs (measured max
|diff| ~ 4e-15). 1e-12 is ~1000x that rounding noise — any real formula drift
(a wrong window, a different warmup rule, ddof) is many orders of magnitude larger.

Skipped when no C++ compiler is on PATH.
"""
import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core_math import l2_math, micro_math

TOL = 1e-12
SRC = Path(__file__).resolve().parents[1] / "core_math" / "cpp" / "micro.cpp"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    ext = ".dll" if sys.platform == "win32" else ".so"
    out = tmp_path_factory.mktemp("cpp") / f"micro{ext}"
    cmd = [cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-o", str(out), str(SRC)]
    if sys.platform == "win32":
        cmd[1:1] = ["-static"]  # do not depend on the MinGW runtime DLLs being on PATH
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(out))
    lib.book_imbalance.restype = ctypes.c_double
    return lib


def _p(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.mark.parametrize("vol_window", [5, 30, 100])
def test_bvc_close_return(lib, vol_window):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 5000)))
    ref_buy, ref_sell = micro_math.bulk_volume_classification(close, vol_window=vol_window)
    buy, sell = np.zeros(close.size), np.zeros(close.size)
    lib.bvc_close_return(_p(close), close.size, vol_window, ctypes.c_double(1e-12), _p(buy), _p(sell))
    np.testing.assert_allclose(buy, ref_buy, rtol=0, atol=TOL)
    np.testing.assert_allclose(sell, ref_sell, rtol=0, atol=TOL)


@pytest.mark.parametrize("n_buckets", [1, 10, 50])
def test_vpin_and_tib(lib, n_buckets):
    rng = np.random.default_rng(1)
    n = 3000
    abs_imb = rng.uniform(0, 1, n)
    dir_imb = rng.uniform(-1, 1, n)

    out = np.zeros(n)
    lib.vpin_from_abs_imbalance(_p(abs_imb), n, n_buckets, _p(out))
    ref = micro_math.vpin_from_buckets(pd.DataFrame({"abs_imbalance": abs_imb}), n_buckets)
    np.testing.assert_allclose(out, ref.to_numpy(), rtol=0, atol=TOL)

    out = np.zeros(n)
    lib.tib_from_dir_imbalance(_p(dir_imb), n, n_buckets, _p(out))
    ref = micro_math.tib_from_buckets(pd.DataFrame({"dir_imbalance": dir_imb}), n_buckets)
    np.testing.assert_allclose(out, ref.to_numpy(), rtol=0, atol=TOL)


def test_book_imbalance_matches_signal_thresholds(lib):
    rng = np.random.default_rng(2)
    for _ in range(200):
        bids = rng.uniform(1, 100, rng.integers(1, 6))
        asks = rng.uniform(1, 100, rng.integers(1, 6))
        imb = lib.book_imbalance(_p(bids), bids.size, _p(asks), asks.size)
        wb = sum(v / (i + 1) for i, v in enumerate(bids))
        wa = sum(v / (i + 1) for i, v in enumerate(asks))
        assert abs(imb - wb / (wb + wa)) <= TOL
        # The Python signal is the thresholded version of the same number.
        sig = l2_math.book_imbalance_signal(
            [(0.0, v) for v in bids], [(0.0, v) for v in asks], trigger_threshold=0.6)
        expected = "BUY" if imb >= 0.6 else ("SELL" if imb <= 0.4 else None)
        assert sig == expected


def test_book_imbalance_empty_book(lib):
    zeros = np.zeros(3)
    assert lib.book_imbalance(_p(zeros), 3, _p(zeros), 3) == -1.0
