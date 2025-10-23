#!/usr/bin/env python3
"""
Repro: ProcessPoolExecutor does NOT kill a running child task when you
shutdown(wait=False). The hung child keeps running in the background.

Tested on macOS/Linux (Unix-like). On Windows, the concept is the same,
but the os.kill(pid, 0) liveness probe differs.
"""

import os
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError

HEARTBEAT_SECS = 2


def is_pid_alive(pid: int) -> bool:
    """Best-effort liveness check for Unix-like systems."""
    try:
        # signal 0 doesn't kill; it just errors if the pid doesn't exist or no perms
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def worker_hang(pidfile: str) -> None:
    """
    A worker that never finishes. It writes its PID to `pidfile`, then loops,
    printing a heartbeat so you can see it's still alive even after pool shutdown.
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

    print("=== 1) Start first pool and submit a hanging task ===")
    pool1 = ProcessPoolExecutor(max_workers=1)
    fut_hang = pool1.submit(worker_hang, pidfile)

    # Give it a moment to start and write its pid
    time.sleep(3)

    # Read the hung worker PID
    if not os.path.exists(pidfile):
        print("ERROR: pidfile not created; the hang worker didn't start?")
        return 2
    with open(pidfile, "r", encoding="utf-8") as f:
        hung_pid = int(f.read().strip())
    print(f"[main] Hung worker PID is {hung_pid}. Alive? {is_pid_alive(hung_pid)}")

    print("\n=== 2) 'Restart' the executor (shutdown(wait=False)) ===")
    # This does NOT kill the running child task.
    pool1.shutdown(wait=False, cancel_futures=True)
    print("[main] pool1.shutdown(wait=False, cancel_futures=True) returned.")

    # Prove the hung child is still alive and printing heartbeats
    time.sleep(5)
    print(f"[main] After shutdown: hung PID {hung_pid} alive? {is_pid_alive(hung_pid)}")

    print("\n=== 3) Start a new pool and run a quick task ===")
    pool2 = ProcessPoolExecutor(max_workers=1)
    fut = pool2.submit(worker_quick, 42)
    try:
        result = fut.result(timeout=10)
        print(f"[main] quick result: {result}")
    except TimeoutError:
        print("[main] quick task unexpectedly timed out")

    # Keep watching a bit so you can see the hang task still printing
    print("\n=== 4) Observe that the hung worker keeps running ===")
    for s in range(6, 0, -1):
        print(f"[main] waiting… ({s})  hung PID alive? {is_pid_alive(hung_pid)}")
        time.sleep(1)

    print("\n=== 5) (Optional) Manually kill the hung worker ===")
    if is_pid_alive(hung_pid):
        try:
            # SIGTERM first…
            os.kill(hung_pid, signal.SIGTERM)
            time.sleep(1)
            if is_pid_alive(hung_pid):
                # …then SIGKILL if needed.
                os.kill(hung_pid, signal.SIGKILL)
            print(f"[main] killed hung PID {hung_pid}. Alive? {is_pid_alive(hung_pid)}")
        except Exception as e:
            print(f"[main] failed to kill hung PID {hung_pid}: {e}")

    pool2.shutdown(wait=True)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
