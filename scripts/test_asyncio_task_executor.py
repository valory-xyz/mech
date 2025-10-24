#!/usr/bin/env python3
"""
AsyncRunner demo: hard-timeout a hanging task (subprocess) with asyncio.

What it shows:
- Launch a child process that prints heartbeats forever.
- Enforce a per-task timeout: on timeout -> SIGTERM, then SIGKILL if needed.
- Verify the child is gone.
- Run a quick task to prove the runner is fine.

Works on macOS/Linux. On Windows, replace signal handling accordingly.
"""

import asyncio
import concurrent.futures
import os
import sys
import threading
import time
from typing import Optional, Tuple

HEARTBEAT_SECS = 2
HANG_TIMEOUT = 5  # seconds


def is_pid_alive(pid: int) -> bool:
    """Best-effort liveness check for Unix-like systems."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class AsyncRunner:
    """Run an asyncio loop in a background thread; submit coroutines from sync code."""

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
        if not self._loop:
            raise RuntimeError("AsyncRunner not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run(self, coro):
        """Run and block until completion (no extra timeout)."""
        return self.submit(coro).result()

    def run_with_timeout(self, coro, timeout: float):
        """Run with a sync-side timeout (cancels the future if it times out)."""
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise

    def shutdown(self) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)
        self._loop = None
        self._thread = None


async def _pump_stdout(proc: asyncio.subprocess.Process) -> None:
    """Continuously print child's stdout."""
    if not proc.stdout:
        return
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(line.decode().rstrip())
    except Exception:
        # If proc is killed abruptly, read may error—ok to ignore in demo.
        pass


async def run_hanging_with_hard_timeout(pidfile: str, timeout: float) -> Tuple[int, bool]:
    """
    Start a subprocess that hangs forever; enforce a hard timeout.
    Returns (pid, finished_normally).
    """
    code = (
        "import os,time,sys;"
        "pid=os.getpid();"
        "print(f'[hang] started in PID {pid}', flush=True);"
        f"i=0\n"
        f"while True:\n"
        f"    time.sleep({HEARTBEAT_SECS});"
        f"    i+=1;"
        f"    print(f'[hang] PID {{pid}} heartbeat #{{i}}', flush=True)\n"
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    pid = proc.pid
    # write PID so we can check liveness from sync side
    with open(pidfile, "w", encoding="utf-8") as f:
        f.write(str(pid))

    # start stdout pump (don't await it; it ends when proc exits)
    asyncio.create_task(_pump_stdout(proc))

    try:
        # Wait for process with a per-task timeout
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        # finished before timeout
        return pid, True
    except asyncio.TimeoutError:
        # Hard timeout path: try SIGTERM, then SIGKILL
        # (if running on Windows, replace with appropriate termination)
        try:
            proc.terminate()
        except ProcessLookupError:
            pass  # already gone

        try:
            await asyncio.wait_for(proc.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # ensure it is reaped
            try:
                await proc.wait()
            except Exception:
                pass

        return pid, False


async def quick_task(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * x


def main() -> int:
    runner = AsyncRunner()
    runner.start()

    pidfile = "hung_worker.pid"

    print("=== 1) Start hanging subprocess with hard timeout ===")
    pid, finished = runner.run(run_hanging_with_hard_timeout(pidfile, HANG_TIMEOUT))
    print(f"[main] Child PID: {pid}, finished_normally: {finished}")
    time.sleep(1.0)  # let the OS reap

    print(f"[main] After timeout: PID {pid} alive? {is_pid_alive(pid)}")

    print("\n=== 2) Run a quick task to prove runner health ===")
    res = runner.run_with_timeout(quick_task(42), timeout=5.0)
    print(f"[main] quick result: {res}")

    runner.shutdown()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
