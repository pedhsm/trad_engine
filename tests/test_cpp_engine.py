"""C++ engine tests.

- risk_manager: header-only, needs just a compiler -> runs in CI.
- engine harness: needs the IB client (lib/client, not redistributable) and ZeroMQ,
  so it runs only on a machine that has both, and is skipped elsewhere.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "cpp_engine" / "include"
CLIENT = ROOT / "lib" / "client"
HERE = Path(__file__).resolve().parent / "cpp"


def _cxx():
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    return cxx


def _exe(d: Path, name: str) -> Path:
    return d / (name + (".exe" if sys.platform == "win32" else ""))


def test_risk_manager(tmp_path):
    exe = _exe(tmp_path, "risk_manager_test")
    subprocess.run([_cxx(), "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE),
                    str(HERE / "risk_manager_test.cpp"), "-o", str(exe)]
                   + (["-static"] if sys.platform == "win32" else ["-pthread"]), check=True)
    res = subprocess.run([str(exe)], capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "all checks passed" in res.stdout


def test_engine_harness(tmp_path):
    cxx = _cxx()
    if not (CLIENT / "TwsApiL0.cpp").exists():
        pytest.skip("lib/client is not populated (IB TWS API is not redistributable)")
    probe = subprocess.run([cxx, "-x", "c++", "-E", "-"], input="#include <zmq.hpp>\n",
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("zmq.hpp (cppzmq) not found")

    exe = _exe(tmp_path, "engine_harness")
    defines = ["-DIB_USE_STD_STRING"] + (["-DWIN32"] if sys.platform == "win32" else [])
    libs = ["-lzmq"] + (["-lws2_32"] if sys.platform == "win32" else ["-pthread"])
    subprocess.run([cxx, "-O1", "-std=c++17", *defines, "-I", str(INCLUDE), "-I", str(CLIENT),
                    str(HERE / "engine_harness.cpp"), str(ROOT / "cpp_engine" / "src" / "AllocatorArena.cpp"),
                    str(CLIENT / "TwsApiL0.cpp"), "-o", str(exe), *libs], check=True,
                   capture_output=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    res = subprocess.run([str(exe)], capture_output=True, text=True, cwd=run_dir, timeout=120)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]
    assert "engine harness: all checks passed" in res.stdout
