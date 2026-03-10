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

import pytest

from packages.valory.skills.task_execution.utils.task import AnyToolAsTask


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

    def test_execute_simple_function_with_args_and_kwargs(self):
        """Positional args and kwargs are forwarded to the callable."""
        task = AnyToolAsTask()
        args_out, kwargs_out = task.execute(
            1, 2, tool_py=SIMPLE_TOOL, callable_method="run", extra="val"
        )
        assert args_out == [1, 2]
        assert kwargs_out == {"extra": "val"}

    def test_execute_returns_method_result(self):
        """Return value from the executed method is returned correctly."""
        task = AnyToolAsTask()
        result = task.execute(tool_py=RETURN_VALUE_TOOL, callable_method="compute", x=3, y=4)
        assert result == 7

    def test_execute_class_based_callable(self):
        """Callable defined via a class instance in the tool code."""
        task = AnyToolAsTask()
        result = task.execute(tool_py=CLASS_BASED_TOOL, callable_method="run")
        assert result == "class-result"

    def test_missing_tool_py_raises_key_error(self):
        """KeyError when tool_py kwarg is absent."""
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute(callable_method="run")

    def test_missing_callable_method_raises_key_error(self):
        """KeyError when callable_method kwarg is absent."""
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute(tool_py=SIMPLE_TOOL)

    def test_undefined_method_in_code_raises_key_error(self):
        """KeyError when the named method is not defined by the tool code."""
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute(tool_py=SIMPLE_TOOL, callable_method="nonexistent")

    def test_syntax_error_in_tool_code_propagates(self):
        """SyntaxError in tool_py propagates out of execute."""
        task = AnyToolAsTask()
        with pytest.raises(SyntaxError):
            task.execute(tool_py=SYNTAX_ERROR_TOOL, callable_method="bad")

    def test_tool_py_and_callable_method_are_popped(self):
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
