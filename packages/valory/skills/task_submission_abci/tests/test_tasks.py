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
"""Tests for task_submission_abci.tasks."""

import pytest
from aea.skills.tasks import Task

from packages.valory.skills.task_submission_abci.tasks import AnyToolAsTask


class TestAnyToolAsTask:
    """Tests for AnyToolAsTask."""

    def test_inherits_from_aea_task(self):
        assert issubclass(AnyToolAsTask, Task)

    def test_execute_calls_method_with_args(self):
        """method kwarg is called with remaining args and kwargs."""
        results = []

        def my_method(*args, **kwargs):
            results.append((args, kwargs))
            return "done"

        task = AnyToolAsTask()
        result = task.execute(1, 2, method=my_method, extra="val")
        assert result == "done"
        assert results == [((1, 2), {"extra": "val"})]

    def test_execute_method_kwarg_is_popped(self):
        """method should NOT be forwarded as a kwarg to the called method."""

        def spy(**kwargs):
            return kwargs

        task = AnyToolAsTask()
        result = task.execute(method=spy, my_arg="hello")
        assert "method" not in result
        assert result == {"my_arg": "hello"}

    def test_execute_missing_method_raises_key_error(self):
        task = AnyToolAsTask()
        with pytest.raises(KeyError):
            task.execute()

    def test_execute_returns_method_return_value(self):
        task = AnyToolAsTask()
        result = task.execute(method=lambda: 42)
        assert result == 42
