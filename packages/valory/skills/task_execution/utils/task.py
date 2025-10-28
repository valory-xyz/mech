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

"""This package contains a custom Loader for the ipfs connection."""

from typing import Any

from packages.valory.skills.task_execution.utils.timeout_exec import (
    run_tool_with_timeout,
)


class AnyToolAsTask:
    """AnyToolAsTask with hard timeout using a killable child process."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool function."""
        tool_py: str = kwargs.pop("tool_py")
        callable_method: str = kwargs.pop("callable_method")
        timeout: float = float(kwargs.pop("task_deadline", 300.0))

        # For constructing a meaningful fallback tuple on timeout/error:
        counter_callback = kwargs.get("counter_callback")
        keychain = kwargs.get("api_keys")

        status, result, err = run_tool_with_timeout(
            tool_src=tool_py,
            method_name=callable_method,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
        )

        if status == "ok":
            return result

        if status == "timeout":
            return (
                f"Task timed out after {timeout} seconds.",
                "",
                None,
                counter_callback,
                keychain,
            )

        return (f"Task failed with error:\n{err}", "", None, counter_callback, keychain)
