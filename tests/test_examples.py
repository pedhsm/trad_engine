"""End-to-end: the paper demo runs and is deterministic."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_demo(cwd: Path) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, "-m", "examples.paper_demo"], cwd=cwd, env=env,
                         capture_output=True, text=True, check=True)
    return res.stdout


def test_paper_demo_is_deterministic(tmp_path):
    a = _run_demo(tmp_path)
    b = _run_demo(tmp_path)
    assert a == b
    assert "Plumbing OK" in a
    assert (tmp_path / "paper_demo_journal.db").exists()
