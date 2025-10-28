# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Wraps the execute method of ta tool to a killable process."""

import multiprocessing as mp
import traceback
from typing import Any, Dict, Optional, Tuple


# POSIX-only helpers for killing a whole process group
try:
    import os
    import signal

    _HAS_POSIX = True
except Exception:  # pragma: no cover
    _HAS_POSIX = False


def _runner(
    q_: "mp.Queue",
    src: str,
    meth: str,
    a: tuple,
    k: dict,
    make_new_pgrp: bool,
) -> None:
    """Top-level target for the child process."""
    if make_new_pgrp and _HAS_POSIX:
        try:
            os.setpgrp()  # become leader of a new process group
        except Exception:
            pass

    local_ns: Dict[str, Any] = {}
    try:
        exec(src, local_ns)  # nosec: intentional dynamic tool exec
        fn = local_ns[meth]
        out = fn(*a, **k)
        q_.put(("ok", out, None))
    except Exception:
        q_.put(("error", None, traceback.format_exc()))


def run_tool_with_timeout(
    tool_src: str,
    method_name: str,
    *,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0,
) -> Tuple[str, Optional[Any], Optional[str]]:
    """Execute `method_name` from `tool_src` in a fresh process with a hard timeout."""
    kwargs = kwargs or {}
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    # Ask child to create a new process group (POSIX) so we can kill its children too.
    make_new_pgrp = True

    p = ctx.Process(
        target=_runner,
        args=(q, tool_src, method_name, args, kwargs, make_new_pgrp),
        daemon=False,  # allow tool to spawn children if it wants; we control lifetime
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        try:
            if _HAS_POSIX:
                os.killpg(p.pid, signal.SIGKILL)  # type: ignore[arg-type]
            else:
                raise RuntimeError("no posix killpg")
        except Exception:
            p.terminate()
        p.join()

        # Clean queue to avoid leaks
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass

        return ("timeout", None, None)

    # Normal path
    try:
        result = q.get_nowait()
    except Exception:
        result = ("error", None, "Child exited without posting a result.")
    finally:
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass

    return result  # (status, result, err)
