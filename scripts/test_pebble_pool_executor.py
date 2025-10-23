#!/usr/bin/env python3
"""
Repro with Pebble: demonstrate that a hanging task can be hard-timed-out
(and its worker process terminated), unlike stdlib ProcessPoolExecutor.

Requires:
    pip install pebble

Tested on macOS/Linux (Unix-like). On Windows, replace is_pid_alive accordingly.
"""

import os
import sys
import time
from concurrent.futures import TimeoutError

from pebble import ProcessPool, ProcessExpired

HEARTBEAT_SECS = 2
HANG_TIMEOUT = 5  # seconds


def is_pid_alive(pid: int) -> bool:
    """Best-effort liveness check for Unix-like systems."""
    try:
        os.kill(pid, 0)  # signal 0: check existence/permission only
        return True
    except OSError:
        return False


def worker_hang(pidfile: str) -> None:
    """
    A worker that never finishes. It writes its PID to `pidfile`, then loops,
    printing a heartbeat so you can see it's alive.
    """
    pid = os.getpid()
    print(f"[hang] started in PID {pid}", flush=True)
    with open(pidfile, "w", encoding="utf-8") as f:
        f.write(str(pid))
    i = 0
    while True:
        time.sleep(HEARTBEAT_SECS)
        i += 1
        print(f"[hang] PID {pid} heartbeat #{i}", flush=True)


def worker_quick(x: int) -> int:
    """A quick worker to prove the new pool works fine."""
    pid = os.getpid()
    print(f"[quick] started in PID {pid}", flush=True)
    return x * x


def main() -> int:
    pidfile = "hung_worker.pid"

    print("=== 1) Start Pebble pool and schedule a hanging task WITH timeout ===")
    # max_tasks=1 mimics recycling after each task (like max_tasks_per_child)
    with ProcessPool(max_workers=1, max_tasks=1) as pool1:
        # Pebble lets you set a per-task timeout here. When it fires, Pebble
        # terminates the worker process hosting this task.
        fut_hang = pool1.schedule(worker_hang, args=(pidfile,), timeout=HANG_TIMEOUT)

        # Give it a moment to start and write its pid
        time.sleep(3)

        # Read the hung worker PID
        if not os.path.exists(pidfile):
            print("ERROR: pidfile not created; the hang worker didn't start?")
            return 2
        with open(pidfile, "r", encoding="utf-8") as f:
            hung_pid = int(f.read().strip())
        print(f"[main] Hung worker PID is {hung_pid}. Alive? {is_pid_alive(hung_pid)}")

        print(f"\n=== 2) Wait for task timeout (~{HANG_TIMEOUT}s) ===")
        try:
            # This will raise concurrent.futures.TimeoutError after HANG_TIMEOUT.
            fut_hang.result()  # no extra timeout here; the per-task timeout applies
            print("[main] (unexpected) hang task finished without timing out")
        except TimeoutError:
            print("[main] Task hit its timeout; Pebble should have terminated the worker.")
        except ProcessExpired as e:
            print(f"[main] Worker process died unexpectedly: {e} (exit={e.exitcode})")

        # Give the OS a moment to reap the process, then check liveness
        time.sleep(1.0)
        print(f"[main] After timeout: hung PID {hung_pid} alive? {is_pid_alive(hung_pid)}")

    print("\n=== 3) Start a fresh Pebble pool and run a quick task ===")
    with ProcessPool(max_workers=1, max_tasks=1) as pool2:
        fut = pool2.schedule(worker_quick, args=(42,), timeout=10)
        try:
            result = fut.result()
            print(f"[main] quick result: {result}")
        except TimeoutError:
            print("[main] quick task unexpectedly timed out")
        except ProcessExpired as e:
            print(f"[main] quick worker died unexpectedly: {e} (exit={e.exitcode})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
