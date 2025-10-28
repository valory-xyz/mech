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

    def _runner(q_: mp.Queue, src: str, meth: str, a: tuple, k: dict) -> None:
        local_ns: Dict[str, Any] = {}
        try:
            exec(src, local_ns)  # nosec
            fn = local_ns[meth]
            out = fn(*a, **k)
            q_.put(("ok", out, None))
        except Exception:
            q_.put(("error", None, traceback.format_exc()))

    p = ctx.Process(
        target=_runner, args=(q, tool_src, method_name, args, kwargs), daemon=True
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        # Best-effort drain
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass
        return ("timeout", None, None)

    try:
        return q.get_nowait()
    except Exception:
        return ("error", None, "Child exited without posting a result.")
