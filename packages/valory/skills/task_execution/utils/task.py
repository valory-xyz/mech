# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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


class ToolError(Exception):
    """An exception a tool raised, carried by its type name and message.

    Tool code runs from an ``exec`` namespace, so its own exception classes
    cannot be unpickled outside the worker process. This class can, and keeps
    what the agent reads from a failed run.
    """

    def __init__(self, type_name: str, message: str) -> None:
        """Initialize the error.

        :param type_name: the class name of the exception the tool raised.
        :param message: the text of that exception.
        """
        super().__init__(type_name, message)
        self.type_name = type_name
        self.message = message

    def __str__(self) -> str:
        """Return the tool's own message."""
        return self.message


class AnyToolAsTask:
    """AnyToolAsTask"""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task.

        :param args: positional arguments for the tool callable.
        :param kwargs: ``tool_py``, ``callable_method`` and the tool kwargs.
        :return: whatever the tool callable returns.
        :raises ToolError: when the tool callable raises.
        """
        tool_py = kwargs.pop("tool_py")
        callable_method = kwargs.pop("callable_method")
        local_namespace: Any = {}
        exec(tool_py, local_namespace)  # pylint: disable=W0122  # nosec
        method = local_namespace[callable_method]
        try:
            return method(*args, **kwargs)
        except Exception as exc:  # pylint: disable=W0718
            try:
                message = str(exc)
            except Exception:  # pylint: disable=W0718
                message = ""
            raise ToolError(type(exc).__name__, message) from None
