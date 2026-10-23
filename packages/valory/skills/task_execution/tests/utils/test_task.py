# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
"""Tests for task_execution.utils.task."""

import pickle  # nosec

import pytest
from pebble import ProcessPool

from packages.valory.skills.task_execution.utils.task import AnyToolAsTask, ToolError

SIMPLE_TOOL = """\
def run(*args, **kwargs):
    return list(args), kwargs
"""

RETURN_VALUE_TOOL = """\
def compute(x, y):
    return x + y
"""

CLASS_BASED_TOOL = """\
class MyTool:
    def run(self):
        return "class-result"

run = MyTool().run
"""

SYNTAX_ERROR_TOOL = "def bad(: pass"


class TestAnyToolAsTask:
    """Tests for AnyToolAsTask.execute."""

    def test_execute_simple_function_with_args_and_kwargs(self) -> None:
        """Positional args and kwargs are forwarded to the callable."""
        task = AnyToolAsTask()
        args_out, kwargs_out = task.execute(
            1, 2, tool_py=SIMPLE_TOOL, callable_method="run", extra="val"
        )
        assert args_out == [1, 2]
        assert kwargs_out == {"extra": "val"}

    def test_execute_returns_method_result(self) -> None:
        """Return value from the executed method is returned correctly."""
        task = AnyToolAsTask()
        result = task.execute(
            tool_py=RETURN_VALUE_TOOL, callable_method="compute", x=3, y=4
        )
        assert result == 7

    def test_execute_class_based_callable(self) -> None:
        """Callable defined via a class instance in the tool code."""
        task = AnyToolAsTask()
        result = task.execute(tool_py=CLASS_BASED_TOOL, callable_method="run")
        assert result == "class-result"

    def test_missing_tool_py_raises_key_error(self) -> None:
        """Raise KeyError when tool_py kwarg is absent."""
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute(callable_method="run")

    def test_missing_callable_method_raises_key_error(self) -> None:
        """Raise KeyError when callable_method kwarg is absent."""
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute(tool_py=SIMPLE_TOOL)

    def test_undefined_method_in_code_raises_key_error(self) -> None:
        """Raise KeyError when the named method is not defined by the tool code."""
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute(tool_py=SIMPLE_TOOL, callable_method="nonexistent")

    def test_syntax_error_in_tool_code_propagates(self) -> None:
        """Raise SyntaxError in tool_py propagates out of execute."""
        task = AnyToolAsTask()
        with pytest.raises(SyntaxError):
            task.execute(tool_py=SYNTAX_ERROR_TOOL, callable_method="bad")

    def test_tool_py_and_callable_method_are_popped(self) -> None:
        """tool_py and callable_method are not forwarded as kwargs to the method."""
        tool_py = """\
def spy(**kwargs):
    return kwargs
"""
        task = AnyToolAsTask()
        result = task.execute(tool_py=tool_py, callable_method="spy", my_arg="hello")
        # tool_py and callable_method should NOT appear in the kwargs received by spy
        assert "tool_py" not in result
        assert "callable_method" not in result
        assert result == {"my_arg": "hello"}


CLIENT_FAULT_TOOL = """\
class InvalidRequestError(ValueError):
    pass

def run(**kwargs):
    raise InvalidRequestError("prompt is empty")
"""

UNPRINTABLE_ERROR_TOOL = """\
class Unprintable(Exception):
    def __str__(self):
        raise RuntimeError("no text")

def run(**kwargs):
    raise Unprintable()
"""

# Raises an exception class the agent process can import but cannot rebuild
# from its pickled args, the shape of the OpenAI and Anthropic SDK errors.
SDK_STYLE_ERROR_TOOL = """\
from packages.valory.skills.task_execution.tests.utils.test_task import (
    SdkStyleRateLimitError,
)

def run(**kwargs):
    raise SdkStyleRateLimitError("Rate limit reached", response=object(), body=None)
"""


class SdkStyleRateLimitError(Exception):
    """Needs keyword-only arguments, so ``pickle.loads`` cannot rebuild it."""

    def __init__(self, message: str, *, response: object, body: object) -> None:
        """Initialize the error.

        :param message: the error text.
        :param response: the provider HTTP response.
        :param body: the decoded response body.
        """
        super().__init__(message)
        self.response = response
        self.body = body


class TestToolError:
    """A tool's exception reaches the agent as a picklable ToolError."""

    def test_tool_exception_is_raised_as_tool_error_with_its_class_name(
        self,
    ) -> None:
        """The tool's class name and message survive, the class itself does not."""
        with pytest.raises(ToolError) as caught:
            AnyToolAsTask().execute(tool_py=CLIENT_FAULT_TOOL, callable_method="run")
        assert caught.value.type_name == "InvalidRequestError"
        assert caught.value.message == "prompt is empty"
        assert str(caught.value) == "prompt is empty"

    def test_tool_exception_without_text_keeps_its_class_name(self) -> None:
        """A message that cannot be rendered leaves an empty message, not a crash."""
        with pytest.raises(ToolError) as caught:
            AnyToolAsTask().execute(
                tool_py=UNPRINTABLE_ERROR_TOOL, callable_method="run"
            )
        assert (caught.value.type_name, caught.value.message) == ("Unprintable", "")

    def test_tool_error_survives_a_pickle_round_trip(self) -> None:
        """The worker pickles the exception; the agent must get the same fields."""
        restored = pickle.loads(  # nosec
            pickle.dumps(ToolError("InvalidRequestError", "prompt is empty"))
        )
        assert isinstance(restored, ToolError)
        assert (restored.type_name, restored.message) == (
            "InvalidRequestError",
            "prompt is empty",
        )

    def test_tool_exception_crosses_the_worker_process_boundary(self) -> None:
        """Run in a real worker: the agent sees the tool's class name, not a PicklingError."""
        with ProcessPool(max_workers=1) as pool:
            future = pool.schedule(
                AnyToolAsTask().execute,
                kwargs={"tool_py": CLIENT_FAULT_TOOL, "callable_method": "run"},
                timeout=60,
            )
            with pytest.raises(ToolError) as caught:
                future.result()
        assert caught.value.type_name == "InvalidRequestError"

    def test_sdk_style_exception_crosses_the_worker_process_boundary(self) -> None:
        """An exception that cannot be unpickled arrives as a ToolError, not a hang."""
        raw = SdkStyleRateLimitError("Rate limit reached", response=None, body=None)
        with pytest.raises(TypeError):
            pickle.loads(pickle.dumps(raw))  # nosec
        with ProcessPool(max_workers=1) as pool:
            future = pool.schedule(
                AnyToolAsTask().execute,
                kwargs={"tool_py": SDK_STYLE_ERROR_TOOL, "callable_method": "run"},
                timeout=60,
            )
            with pytest.raises(ToolError) as caught:
                future.result(timeout=60)
        assert (caught.value.type_name, caught.value.message) == (
            "SdkStyleRateLimitError",
            "Rate limit reached",
        )
