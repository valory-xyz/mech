#!/usr/bin/env python3
"""
Two stdlib-only scenarios with ProcessPoolExecutor:

A) NO manual kill (pidfile): show that shutdown(wait=False) does NOT kill a hung child.
B) WITH manual kill (Manager().Queue()): get child PID, send SIGTERM/SIGKILL, then recover.

Tested on macOS/Linux. For Windows, replace os.kill checks accordingly.
"""

import os
import sys
import time
import signal
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError

HEARTBEAT_SECS = 2
WATCHDOG_TIMEOUT = 6  # seconds


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 only checks existence/permission
        return True
    except OSError:
        return False


# ---------------- Scenario A uses a pidfile (no Queue needed) ----------------
def worker_hang_pidfile(pidfile: str) -> None:
    pid = os.getpid()
    print(f"[hang:A] started in PID {pid}", flush=True)
    with open(pidfile, "w", encoding="utf-8") as f:
        f.write(str(pid))
    i = 0
    while True:
        time.sleep(HEARTBEAT_SECS)
        i += 1
        print(f"[hang:A] PID {pid} heartbeat #{i}", flush=True)


# ------------- Scenario B uses a Manager().Queue() (picklable) ---------------
def worker_hang_queue(pid_queue) -> None:
    pid = os.getpid()
    # Manager().Queue() is a proxy, so put() is fine under spawn.
    pid_queue.put(pid)
    print(f"[hang:B] started in PID {pid}", flush=True)
    i = 0
    while True:
        time.sleep(HEARTBEAT_SECS)
        i += 1
        print(f"[hang:B] PID {pid} heartbeat #{i}", flush=True)


def worker_quick(x: int) -> int:
    pid = os.getpid()
    print(f"[quick] started in PID {pid}", flush=True)
    return x * x


def scenario_a_no_manual_kill() -> None:
    print("\n=== Scenario A) NO manual kill (pidfile) ===")
    # Use a temp pidfile so spawn/fork differences don't matter
    with tempfile.TemporaryDirectory() as td:
        pidfile = os.path.join(td, "hung_worker.pid")

        pool = ProcessPoolExecutor(max_workers=1)
        fut_hang = pool.submit(worker_hang_pidfile, pidfile)

        # Wait for child to write its pid
        for _ in range(30):  # up to ~3s
            if os.path.exists(pidfile):
                break
            time.sleep(0.1)

        if not os.path.exists(pidfile):
            print("[A] ERROR: pidfile not created; hang worker may not have started.")
            pool.shutdown(wait=False, cancel_futures=True)
            return

        with open(pidfile, "r", encoding="utf-8") as f:
            hung_pid = int(f.read().strip())
        print(f"[A] Hung worker PID is {hung_pid}. Alive? {is_pid_alive(hung_pid)}")

        print("[A] Calling pool.shutdown(wait=False, cancel_futures=True)")
        pool.shutdown(wait=False, cancel_futures=True)
        print("[A] shutdown returned.")

        # Prove the hung child is still alive and printing heartbeats
        time.sleep(5)
        print(f"[A] After shutdown: hung PID {hung_pid} alive? {is_pid_alive(hung_pid)}")
        print("[A] (Leaving it alive to demonstrate the issue.)")


def scenario_b_with_manual_kill() -> None:
    print("\n=== Scenario B) WITH manual kill (Manager().Queue()) ===")

    # Use the same mp context for both the Manager and the pool (good hygiene)
    ctx = mp.get_context("spawn")  # works on macOS/Linux; adjust if needed
    with ctx.Manager() as manager:
        pid_queue = manager.Queue()

        # Create pool bound to same context
        pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)

        fut_hang = pool.submit(worker_hang_queue, pid_queue)

        # Get PID from queue (proxy)
        try:
            hung_pid = pid_queue.get(timeout=3)
        except Exception:
            print("[B] ERROR: Did not get PID from worker (queue timeout).")
            pool.shutdown(wait=False, cancel_futures=True)
            return

        print(f"[B] Hung worker PID is {hung_pid}. Alive? {is_pid_alive(hung_pid)}")

        # Watchdog: give it some time, then kill
        print(f"[B] Waiting ~{WATCHDOG_TIMEOUT}s before manual termination …")
        time.sleep(WATCHDOG_TIMEOUT)

        # Try graceful first
        if is_pid_alive(hung_pid):
            print("[B] Sending SIGTERM …")
            try:
                os.kill(hung_pid, signal.SIGTERM)
            except Exception as e:
                print(f"[B] SIGTERM failed: {e}")
            time.sleep(1.0)

        # If still alive, force kill
        if is_pid_alive(hung_pid):
            print("[B] Still alive; sending SIGKILL …")
            try:
                os.kill(hung_pid, signal.SIGKILL)
            except Exception as e:
                print(f"[B] SIGKILL failed: {e}")

        time.sleep(0.5)
        print(f"[B] After kill attempts: PID {hung_pid} alive? {is_pid_alive(hung_pid)}")

        # Cleanly shut down the pool (child is gone)
        pool.shutdown(wait=True, cancel_futures=True)
        print("[B] pool.shutdown(wait=True) completed.")

    # Prove we can run a fresh quick task after cleaning up
    print("\n[B] Spinning up a fresh pool for a quick task …")
    pool2 = ProcessPoolExecutor(max_workers=1)
    fut = pool2.submit(worker_quick, 42)
    try:
        res = fut.result(timeout=5)
        print(f"[B] quick result: {res}")
    except TimeoutError:
        print("[B] quick task unexpectedly timed out")
    finally:
        pool2.shutdown(wait=True)


def main() -> int:
    scenario_a_no_manual_kill()
    scenario_b_with_manual_kill()
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
