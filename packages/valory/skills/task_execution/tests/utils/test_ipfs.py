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
"""Tests for task_execution.utils.ipfs."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from packages.valory.skills.task_execution.utils.ipfs import (
    CID_PREFIX,
    ComponentPackageLoader,
    get_ipfs_file_hash,
    to_multihash,
)

# ---------------------------------------------------------------------------
# get_ipfs_file_hash
# ---------------------------------------------------------------------------


class TestGetIpfsFileHash:
    """Tests for get_ipfs_file_hash."""

    def test_valid_cid_string_bytes_returns_cid(self) -> None:
        """When the bytes decode to a valid CID string, return that CID's str()."""
        fake_cid = MagicMock()
        fake_cid.__str__ = MagicMock(return_value="bafytest123")  # type: ignore[method-assign]
        with patch(
            "packages.valory.skills.task_execution.utils.ipfs.CID"
        ) as mock_cid_cls:
            mock_cid_cls.from_string.return_value = fake_cid
            result = get_ipfs_file_hash(
                b"bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            )
        assert result == "bafytest123"

    def test_fallback_to_sha256_on_invalid_bytes(self) -> None:
        """When CID.from_string raises, fall back to CID_PREFIX + hex of bytes."""
        raw_bytes = b"\x00\x01\x02"
        expected_hex = raw_bytes.hex()
        expected_cid_input = CID_PREFIX + expected_hex

        fake_fallback_cid = MagicMock()
        fake_fallback_cid.__str__ = MagicMock(return_value="f_fallback_cid")  # type: ignore[method-assign]
        with patch(
            "packages.valory.skills.task_execution.utils.ipfs.CID"
        ) as mock_cid_cls:
            mock_cid_cls.from_string.side_effect = [
                Exception("not a cid"),  # first call (try block) fails
                fake_fallback_cid,  # second call (fallback) succeeds
            ]
            result = get_ipfs_file_hash(raw_bytes)

        assert result == "f_fallback_cid"
        # Verify fallback CID.from_string was called with the prefixed hex
        assert mock_cid_cls.from_string.call_args_list[1][0][0] == expected_cid_input

    def test_cid_prefix_constant(self) -> None:
        """Test CID_PREFIX has the expected value."""
        assert CID_PREFIX == "f01701220"


# ---------------------------------------------------------------------------
# to_multihash
# ---------------------------------------------------------------------------


class TestToMultihash:
    """Tests for to_multihash."""

    def test_strips_first_six_hex_chars(self) -> None:
        """Verify the function returns hex of multihash bytes with first 6 chars stripped."""
        multihash_bytes = bytes.fromhex(
            "1220" + "ab" * 32
        )  # sha2-256 multihash prefix + 32 bytes
        decoded_with_prefix = b"\x01\x70" + multihash_bytes  # CID v1 + dag-pb codec

        mock_cid_str = "bafytesthash"
        with (
            patch(
                "packages.valory.skills.task_execution.utils.ipfs.multibase"
            ) as mock_mb,
            patch(
                "packages.valory.skills.task_execution.utils.ipfs.multicodec"
            ) as mock_mc,
        ):
            mock_mb.decode.return_value = decoded_with_prefix
            mock_mc.remove_prefix.return_value = multihash_bytes

            result = to_multihash(mock_cid_str)

        expected = multihash_bytes.hex()[6:]
        assert result == expected

    def test_calls_multibase_decode(self) -> None:
        """multibase.decode is called with the input CID string."""
        with (
            patch(
                "packages.valory.skills.task_execution.utils.ipfs.multibase"
            ) as mock_mb,
            patch(
                "packages.valory.skills.task_execution.utils.ipfs.multicodec"
            ) as mock_mc,
        ):
            mock_mb.decode.return_value = b"\x00" * 10
            mock_mc.remove_prefix.return_value = b"\x00" * 8
            to_multihash("some-cid")
        mock_mb.decode.assert_called_once_with("some-cid")

    def test_empty_decode_returns_empty_string(self) -> None:
        """Return empty string when multibase.decode yields empty bytes."""
        with patch(
            "packages.valory.skills.task_execution.utils.ipfs.multibase"
        ) as mock_mb:
            mock_mb.decode.return_value = b""
            assert to_multihash("empty-cid") == ""


# ---------------------------------------------------------------------------
# ComponentPackageLoader
# ---------------------------------------------------------------------------

VALID_YAML = yaml.dump({"entry_point": "tool.py", "callable": "run", "version": "1.0"})
VALID_ENTRY_PY = "def run(*args, **kwargs): return 42"

VALID_PACKAGE = {
    "component.yaml": VALID_YAML,
    "tool.py": VALID_ENTRY_PY,
}


class TestComponentPackageLoader:
    """Tests for ComponentPackageLoader.load."""

    def test_valid_package_returns_tuple(self) -> None:
        """Valid package returns (component_yaml_dict, entry_point_source, callable_name)."""
        component_yaml, entry_point, callable_method = ComponentPackageLoader.load(
            VALID_PACKAGE
        )
        assert component_yaml["entry_point"] == "tool.py"
        assert component_yaml["callable"] == "run"
        assert entry_point == VALID_ENTRY_PY
        assert callable_method == "run"

    def test_missing_component_yaml_raises(self) -> None:
        """Test missing component.yaml raises ValueError."""
        with pytest.raises(ValueError, match="component.yaml"):
            ComponentPackageLoader.load({"tool.py": VALID_ENTRY_PY})

    def test_missing_entry_point_key_raises(self) -> None:
        """Test missing entry_point key raises ValueError."""
        bad_yaml = yaml.dump({"callable": "run"})  # no entry_point
        with pytest.raises(ValueError, match="entry_point"):
            ComponentPackageLoader.load(
                {"component.yaml": bad_yaml, "tool.py": VALID_ENTRY_PY}
            )

    def test_missing_callable_key_raises(self) -> None:
        """Test missing callable key raises ValueError."""
        bad_yaml = yaml.dump({"entry_point": "tool.py"})  # no callable
        with pytest.raises(ValueError, match="callable"):
            ComponentPackageLoader.load(
                {"component.yaml": bad_yaml, "tool.py": VALID_ENTRY_PY}
            )

    def test_missing_entry_point_file_raises(self) -> None:
        """entry_point file named in YAML but not in serialized_objects."""
        with pytest.raises(ValueError, match="tool.py"):
            ComponentPackageLoader.load(
                {"component.yaml": VALID_YAML}
            )  # tool.py absent

    def test_extra_files_in_package_ignored(self) -> None:
        """Extra files beyond component.yaml + entry_point are fine."""
        package = {**VALID_PACKAGE, "README.md": "hello"}
        component_yaml, entry_point, callable_method = ComponentPackageLoader.load(
            package
        )
        assert callable_method == "run"
