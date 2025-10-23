#!/usr/bin/env python3
import os
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError, wait, FIRST_COMPLETED

HEARTBEAT_SECS = 2
HANG_TIMEOUT = 5  # seconds to wait before declaring the hang task timed out

def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def worker_hang(pidfile: str) -> None:
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
    pid = os.getpid()
    print(f"[quick] started in PID {pid}", flush=True)
    return x * x

def main() -> int:
    pidfile = "hung_worker.pid"

    print("=== 1) Start first pool and submit a hanging task ===")
    pool1 = ProcessPoolExecutor(max_workers=1)
    fut_hang = pool1.submit(worker_hang, pidfile)

    time.sleep(3)

    if not os.path.exists(pidfile):
        print("ERROR: pidfile not created; the hang worker didn't start?")
        return 2
    with open(pidfile, "r", encoding="utf-8") as f:
        hung_pid = int(f.read().strip())
    print(f"[main] Hung worker PID is {hung_pid}. Alive? {is_pid_alive(hung_pid)}")

    print("\n=== 2) Wait with timeout; kill if still running ===")
    # Option A: wait() with timeout (non-raising)
    done, not_done = wait({fut_hang}, timeout=HANG_TIMEOUT, return_when=FIRST_COMPLETED)
    if fut_hang in not_done:
        print(f"[main] hang task exceeded {HANG_TIMEOUT}s; terminating PID {hung_pid}...")
        try:
            os.kill(hung_pid, signal.SIGTERM)
            time.sleep(1)
            if is_pid_alive(hung_pid):
                os.kill(hung_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        print(f"[main] Hung PID alive after kill? {is_pid_alive(hung_pid)}")
    else:
        print("[main] (unexpected) hang task finished within timeout")

    print("\n=== 3) Shutdown the first pool ===")
    pool1.shutdown(wait=False, cancel_futures=True)
    print("[main] pool1.shutdown(wait=False, cancel_futures=True) returned.")

    print("\n=== 4) Start a new pool and run a quick task ===")
    pool2 = ProcessPoolExecutor(max_workers=1)
    fut = pool2.submit(worker_quick, 42)
    try:
        result = fut.result(timeout=10)
        print(f"[main] quick result: {result}")
    except TimeoutError:
        print("[main] quick task unexpectedly timed out")

    pool2.shutdown(wait=True)
    print("\nDone.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
