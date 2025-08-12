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


def test_should_split_profits_happy_path(
    fs_behaviour: Any,
    run_to_completion,
    patch_num_requests,
) -> None:
    """Return True when total requests is an exact multiple of the split frequency."""
    patch_num_requests(10)  # 10 % 5 == 0
    assert run_to_completion(fs_behaviour._should_split_profits()) is True


def test_should_split_profits_flaky_boundary(
    fs_behaviour: Any,
    run_to_completion,
    patch_num_requests,
) -> None:
    """
    Show flakiness at the modulo boundary: a +/-1 jitter flips the decision.

    First call returns 9 (False), second returns 10 (True). If upstream counting
    is eventually-consistent or races, the exact multiple-only check is brittle.
    """
    patch_num_requests([9, 10])
    assert run_to_completion(fs_behaviour._should_split_profits()) is False
    assert run_to_completion(fs_behaviour._should_split_profits()) is True
