#!/usr/bin/env python3
"""
Asyncio version of the 'hung worker' demo, callable from sync code.

- An AsyncRunner runs an asyncio event loop in a background thread.
- We launch a hanging subprocess via asyncio (prints heartbeats).
- Scenario A: stop the event loop -> child keeps running (not auto-killed).
- Scenario B: kill the child by PID from sync.

Works on macOS/Linux. On Windows, replace os.kill / signals accordingly.
"""

import asyncio
import concurrent.futures
import os
import signal
import sys
import threading
import time
from typing import Optional

HEARTBEAT_SECS = 2


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# --------------------- Async runner living in a thread --------------------- #
class AsyncRunner:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._loop:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def submit(self, coro: "asyncio.Future") -> concurrent.futures.Future:
        """Submit a coroutine to the background loop; returns a concurrent.futures.Future."""
        if not self._loop:
            raise RuntimeError("AsyncRunner not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_with_timeout(self, coro, timeout: float):
        """Run a coroutine with a timeout from sync code; cancel if it times out."""
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise

    def shutdown(self) -> None:
        """Stop the loop (does NOT kill any subprocesses you started)."""
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)
        self._loop = None
        self._thread = None


# -------------------------- Async “work” coroutines ------------------------ #
async def start_hanging_subprocess() -> int:
    """
    Launch a subprocess that prints a heartbeat forever.
    Return its PID. Never completes (awaits the process).
    """
    # A tiny Python one-liner that loops forever, printing every HEARTBEAT_SECS.
    code = (
        "import os,sys,time;"
        "pid=os.getpid();"
        "print(f'[child] started PID {pid}', flush=True);"
        f"i=0\n"
        f"while True:\n"
        f"    time.sleep({HEARTBEAT_SECS});"
        f"    i+=1;"
        f"    print(f'[child] PID {{pid}} heartbeat #{{i}}', flush=True)\n"
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )

    # Print first line from child (optional)
    if proc.stdout:
        try:
            first = await asyncio.wait_for(proc.stdout.readline(), timeout=2.0)
            if first:
                print(first.decode().rstrip())
        except asyncio.TimeoutError:
            pass

    # Return the PID immediately, but keep task alive by waiting on the process.
    pid = proc.pid

    # Keep streaming output (so you can see heartbeats)
    async def _pump_stdout():
        if not proc.stdout:
            return
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(line.decode().rstrip())

    # Run the pump + wait forever (until killed)
    await asyncio.gather(proc.wait(), _pump_stdout())
    return pid  # practically unreachable unless the process exits


async def quick_task(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * x


# ----------------------------- Sync helpers -------------------------------- #
def kill_pid(pid: int) -> None:
    """SIGTERM then SIGKILL if needed."""
    if not is_pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        print(f"SIGTERM failed: {e}")
    time.sleep(1.0)
    if is_pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            print(f"SIGKILL failed: {e}")


# ------------------------------- Demo flow --------------------------------- #
def scenario_a_no_manual_kill(runner: AsyncRunner) -> Optional[int]:
    """
    Start the hanging subprocess via asyncio, then stop the loop without killing it.
    Returns the child PID so caller can decide to kill later.
    """
    print("\n=== Scenario A) NO manual kill ===")
    fut = runner.submit(start_hanging_subprocess())
    # Give it a moment to print the startup line
    time.sleep(1.0)

    # We can't `result()` here (it never finishes). Instead, peek PID by polling stdout was already printed.
    # We don't have the PID directly; so ask the OS: best we can do is keep it running and let user kill later.
    # To capture PID explicitly, you can refactor start_hanging_subprocess to return it early via a transport.
    # Simpler approach: since the child prints "[child] started PID N", we skip extracting it here.
    # For determinism, let’s re-run a lightweight helper coroutine to capture the PID:
    #   In practice you'd architect start_hanging_subprocess to "return" the PID via another channel.
    print("[A] Stopping the asyncio runner (child will keep running)…")
    runner.shutdown()
    print("[A] Async loop stopped. (Child subprocess is NOT auto-killed.)")
    print("[A] NOTE: Without tracking the PID, we can't kill it here.")
    return None


def scenario_b_with_manual_kill(runner: AsyncRunner) -> None:
    """
    Start a hanging subprocess, but this time **grab the PID** immediately
    (by racing a small wrapper) and kill it after a delay.
    """
    print("\n=== Scenario B) WITH manual kill ===")

    # We’ll wrap the hanging launch to get the PID first, then detach.
    async def launch_and_return_pid():
        # Start the process but DO NOT await proc.wait() here; we need its PID now.
        code = (
            "import os,sys,time;"
            "pid=os.getpid();"
            "print(f'[child] started PID {pid}', flush=True);"
            f"i=0\n"
            f"while True:\n"
            f"    time.sleep({HEARTBEAT_SECS});"
            f"    i+=1;"
            f"    print(f'[child] PID {{pid}} heartbeat #{{i}}', flush=True)\n"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        # start background pump
        async def _pump():
            if not proc.stdout:
                return
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                print(line.decode().rstrip())

        asyncio.create_task(_pump())  # don't await it here
        return proc.pid

    pid = runner.run_with_timeout(launch_and_return_pid(), timeout=5.0)
    print(f"[B] Hung child PID: {pid}. Alive? {is_pid_alive(pid)}")

    print("[B] Sleeping a few seconds, then terminating…")
    time.sleep(6)
    kill_pid(pid)
    print(f"[B] After kill: alive? {is_pid_alive(pid)}")

    # Prove the loop still works
    res = runner.run_with_timeout(quick_task(42), timeout=5.0)
    print(f"[B] quick result: {res}")


def main() -> int:
    # Runner for Scenario A
    runner_a = AsyncRunner()
    runner_a.start()
    scenario_a_no_manual_kill(runner_a)

    # Fresh runner for Scenario B
    runner_b = AsyncRunner()
    runner_b.start()
    scenario_b_with_manual_kill(runner_b)
    runner_b.shutdown()

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
