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
"""Tests for task_execution.utils.apis."""

from typing import Any

import pytest

from packages.valory.skills.task_execution.utils.apis import KeyChain

SERVICES = {
    "openai": ["key-a", "key-b", "key-c"],
    "anthropic": ["anth-1"],
}


class TestKeyChainInit:
    """Tests for KeyChain.__init__."""

    def test_valid_dict_accepted(self) -> None:
        """Test KeyChain accepts a valid dict."""
        kc = KeyChain(SERVICES)
        assert kc.services == SERVICES

    def test_current_index_starts_at_zero(self) -> None:
        """Test all current_index values start at zero."""
        kc = KeyChain(SERVICES)
        assert all(idx == 0 for idx in kc.current_index.values())

    @pytest.mark.parametrize("bad_services", [["openai"], "openai", 42, None])
    def test_non_dict_raises_value_error(self, bad_services: Any) -> None:
        """Test non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="dictionary"):
            KeyChain(bad_services)

    def test_empty_dict_accepted(self) -> None:
        """Test empty dict is accepted."""
        kc = KeyChain({})
        assert kc.services == {}
        assert kc.current_index == {}


class TestKeyChainMaxRetries:
    """Tests for KeyChain.max_retries."""

    def test_returns_key_counts(self) -> None:
        """Test max_retries returns count of keys per service."""
        kc = KeyChain(SERVICES)
        retries = kc.max_retries()
        assert retries == {"openai": 3, "anthropic": 1}

    def test_empty_service_returns_zero(self) -> None:
        """Test empty service list returns zero retries."""
        kc = KeyChain({"svc": []})
        assert kc.max_retries() == {"svc": 0}


class TestKeyChainRotate:
    """Tests for KeyChain.rotate."""

    def test_advances_index(self) -> None:
        """Test rotate advances the current index."""
        kc = KeyChain(SERVICES)
        kc.rotate("openai")
        assert kc.current_index["openai"] == 1

    def test_wraps_to_zero_at_end(self) -> None:
        """Test rotate wraps to zero at end of key list."""
        kc = KeyChain(SERVICES)
        # advance to last key (index 2 for "openai" with 3 keys)
        kc.rotate("openai")  # → 1
        kc.rotate("openai")  # → 2
        kc.rotate("openai")  # → wraps to 0
        assert kc.current_index["openai"] == 0

    def test_single_key_service_stays_at_zero(self) -> None:
        """Test rotate on single-key service stays at zero."""
        kc = KeyChain(SERVICES)
        kc.rotate("anthropic")
        assert kc.current_index["anthropic"] == 0

    def test_unknown_service_raises_key_error(self) -> None:
        """Test rotate on unknown service raises KeyError."""
        kc = KeyChain(SERVICES)
        with pytest.raises(KeyError, match="not found"):
            kc.rotate("unknown")

    def test_roundrobin_cycle(self) -> None:
        """Test round-robin cycling through keys."""
        kc = KeyChain({"svc": ["a", "b", "c"]})
        keys_seen = []
        for _ in range(6):
            keys_seen.append(kc["svc"])
            kc.rotate("svc")
        assert keys_seen == ["a", "b", "c", "a", "b", "c"]


class TestKeyChainGet:
    """Tests for KeyChain.get."""

    def test_returns_current_key(self) -> None:
        """Test get returns the current key."""
        kc = KeyChain(SERVICES)
        assert kc.get("openai", "default") == "key-a"

    def test_returns_default_for_unknown_service(self) -> None:
        """Test get returns default for unknown service."""
        kc = KeyChain(SERVICES)
        assert kc.get("unknown", "fallback") == "fallback"

    def test_returns_updated_key_after_rotate(self) -> None:
        """Test get returns updated key after rotate."""
        kc = KeyChain(SERVICES)
        kc.rotate("openai")
        assert kc.get("openai", "x") == "key-b"


class TestKeyChainGetItem:
    """Tests for KeyChain.__getitem__."""

    def test_returns_first_key(self) -> None:
        """Test __getitem__ returns the first key."""
        kc = KeyChain(SERVICES)
        assert kc["openai"] == "key-a"

    def test_returns_key_after_rotate(self) -> None:
        """Test __getitem__ returns updated key after rotate."""
        kc = KeyChain(SERVICES)
        kc.rotate("openai")
        assert kc["openai"] == "key-b"

    def test_unknown_service_raises_key_error(self) -> None:
        """Test __getitem__ raises KeyError for unknown service."""
        kc = KeyChain(SERVICES)
        with pytest.raises(KeyError, match="not found"):
            _ = kc["no-such-service"]
