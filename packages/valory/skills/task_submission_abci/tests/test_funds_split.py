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

"""Tests for funds splitting behaviour."""

from typing import Any

import pytest


@pytest.mark.usefixtures("fs_behaviour")
def test_should_split_profits_happy_path(
    fs_behaviour: Any, run_to_completion, patch_num_requests
) -> None:
    """Return True when total is exactly a multiple (e.g., 10)."""
    patch_num_requests([10])
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


@pytest.mark.xfail(
    reason="Modulo-only check misses split when counts jump over the boundary (e.g., 9→11).",
    strict=True,
)
def test_should_split_profits_missed_on_jump(
    fs_behaviour: Any, run_to_completion, patch_num_requests
) -> None:
    """
    Document flakiness: if last observed total was 9 and batch delivery bumps to 11,
    the current logic sees 11 % 10 != 0 and skips splitting — even though the 10th
    request happened within the batch.
    """
    # Simulate we 'observe' after the batch (current total=11).
    patch_num_requests([9]); assert run_to_completion(fs_behaviour._should_split_profits()) is False
    patch_num_requests([11]); assert run_to_completion(fs_behaviour._should_split_profits()) is True # <-- desired, but will be False