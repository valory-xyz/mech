# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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

"""Contains the background tasks of the APY estimation skill."""

from typing import Any

from aea.skills.tasks import Task


def do_llm_request():
    """do_llm_request"""


class LLMTask(Task):
    """LLMTask"""

    def execute(self, *args: Any, **kwargs: Any):
        """Execute the task."""
        return do_llm_request(*args, **kwargs)
