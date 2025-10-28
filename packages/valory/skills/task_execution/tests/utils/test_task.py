# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""
Unit tests for AnyToolAsTask + run_tool_with_timeout.

Covers:
- success path (tool returns a value)
- timeout path (child process is killed at deadline)
- error path (exception in tool -> error tuple)
"""

import re
import time

import pytest

from packages.valory.skills.task_execution.utils.task import AnyToolAsTask
from packages.valory.skills.task_execution.utils.timeout_exec import (
    run_tool_with_timeout,
)


def _tool_src_success() -> str:
    # Returns a simple 5-tuple with plain types
    return """
def run(*args, **kwargs):
    # Emulate a small amount of work
    x = kwargs.get("x", 0)
    return ("DELIVERED", "PROMPT", {"tx": x}, "CB", "KEYS")
"""


def _tool_src_timeout() -> str:
    # Sleeps for a long time to trigger timeout
    return """
import time
def run(*args, **kwargs):
    time.sleep(5.0)  # intentionally longer than the test deadline
    return ("SHOULD_NOT_HAPPEN", "", None, None, None)
"""


def _tool_src_error() -> str:
    # Raises an exception
    return """
def run(*args, **kwargs):
    raise RuntimeError("boom")
"""


def _tool_src_kwargs_roundtrip() -> str:
    return """
def run(*args, **kwargs):
    return ("OK", "PROMPT", None, None, kwargs.get("echo"))
"""


@pytest.mark.timeout(10)
def test_run_tool_with_timeout_success():
    """run_tool_with_timeout returns ('ok', result, None) when the tool finishes."""
    status, result, err = run_tool_with_timeout(
        tool_src=_tool_src_success(),
        method_name="run",
        args=(),
        kwargs={"x": 42},
        timeout=1.0,
    )

    assert status == "ok"
    assert err is None
    assert isinstance(result, tuple) and len(result) == 5
    assert result[0] == "DELIVERED"
    assert result[2] == {"tx": 42}


@pytest.mark.timeout(10)
def test_run_tool_with_timeout_timeout_is_bounded():
    """run_tool_with_timeout returns ('timeout', None, None) and finishes within the bound."""
    deadline = 0.3
    t0 = time.monotonic()
    status, result, err = run_tool_with_timeout(
        tool_src=_tool_src_timeout(),
        method_name="run",
        args=(),
        kwargs={},
        timeout=deadline,
    )
    dt = time.monotonic() - t0

    assert status == "timeout"
    assert result is None
    assert err is None
    # Allow a small scheduling overhead margin
    assert dt < deadline + 0.5, f"Timeout path took too long: {dt:.3f}s"


@pytest.mark.timeout(10)
def test_run_tool_with_timeout_error_surface():
    """run_tool_with_timeout returns ('error', None, <traceback>) on exception."""
    status, result, err = run_tool_with_timeout(
        tool_src=_tool_src_error(),
        method_name="run",
        args=(),
        kwargs={},
        timeout=1.0,
    )

    assert status == "error"
    assert result is None
    assert isinstance(err, str)
    # We should see the exception type or message somewhere in the traceback text
    assert "RuntimeError" in err or "boom" in err


@pytest.mark.timeout(10)
def test_any_tool_as_task_success_path_returns_tool_result():
    """AnyToolAsTask.execute returns whatever the tool returns on success."""
    # Set a generous timeout to ensure success runs
    task = AnyToolAsTask(timeout=2.0)
    res = task.execute(
        tool_py=_tool_src_success(),
        callable_method="run",
        x=7,  # forwarded to tool
        counter_callback="IGNORED",  # tool doesn't use these
        api_keys="IGNORED",
    )
    assert isinstance(res, tuple) and len(res) == 5
    assert res[0] == "DELIVERED"
    assert res[2] == {"tx": 7}  # confirm kwargs made it into the child


@pytest.mark.timeout(10)
def test_any_tool_as_task_timeout_returns_5tuple_and_message_contains_seconds():
    """On timeout, AnyToolAsTask returns a 5-tuple with a helpful message and preserves cb/api_keys in parent tuple."""
    task = AnyToolAsTask(timeout=0.2)
    cb = "CB_PARENT"
    keys = "KEYS_PARENT"
    res = task.execute(
        tool_py=_tool_src_timeout(),
        callable_method="run",
        counter_callback=cb,
        api_keys=keys,
    )

    assert isinstance(res, tuple) and len(res) == 5
    msg, prompt, tx, out_cb, out_keys = res
    assert isinstance(msg, str) and "Task timed out" in msg
    # seconds value should be in the message; don't assert exact float, just presence of digits
    assert re.search(r"\d+(\.\d+)?", msg) is not None
    assert prompt == ""
    assert tx is None
    # cb and keys come from the parent (not from child), so they should round-trip by identity
    assert out_cb == cb
    assert out_keys == keys


@pytest.mark.timeout(10)
def test_any_tool_as_task_error_returns_5tuple_with_error_text():
    """On error, AnyToolAsTask returns a 5-tuple with error info."""
    task = AnyToolAsTask(timeout=1.0)
    res = task.execute(
        tool_py=_tool_src_error(),
        callable_method="run",
        counter_callback="CB",
        api_keys="KEYS",
    )
    assert isinstance(res, tuple) and len(res) == 5
    msg, prompt, tx, out_cb, out_keys = res
    assert "Task failed with error" in msg
    assert prompt == ""
    assert tx is None
    assert out_cb == "CB"
    assert out_keys == "KEYS"


@pytest.mark.timeout(10)
def test_kwargs_are_forwarded_into_child():
    """Prove that kwargs make it into the child invocation by echoing."""
    task = AnyToolAsTask(timeout=1.0)
    res = task.execute(
        tool_py=_tool_src_kwargs_roundtrip(),
        callable_method="run",
        echo="ECHO_ME",
    )
    assert isinstance(res, tuple) and len(res) == 5
    assert res[-1] == "ECHO_ME"
