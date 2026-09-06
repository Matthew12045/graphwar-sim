#!/usr/bin/env python3
"""Time each test file with a hard per-file kill (watchdog).

Usage: python3 tools/time_tests.py [max_seconds_per_file]
"""
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
ROOT = Path(__file__).resolve().parents[1]

for f in ["test_parser", "test_golden", "test_solver", "test_agents", "test_corridor", "test_eval"]:
    t0 = time.time()
    proc = subprocess.Popen(
        ["python3", "-m", "pytest", f"tests/{f}.py", "-q", "--tb=line"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    def kill(signum, frame):
        proc.kill()

    signal.signal(signal.SIGALRM, kill)
    signal.alarm(int(MAX) + 1)
    out, _ = proc.communicate()
    signal.alarm(0)
    dt = time.time() - t0
    status = "TIMEOUT-KILLED" if proc.returncode < 0 else f"exit={proc.returncode}"
    tail = "\n".join(out.splitlines()[-4:])
    print(f"== {f}: {dt:6.1f}s {status}")
    print(tail)