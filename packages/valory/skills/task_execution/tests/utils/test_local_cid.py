# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Tests for the local CID helper.

The fixtures pair each input byte-string with the CID a real ``ipfs add
--cid-version=1 --raw-leaves=false`` produced on the same input. Keeping those
pairs as a hardcoded table makes the contract explicit: if a future refactor
changes the encoding in a way that breaks parity with go-ipfs, these tests
fail loudly rather than silently producing CIDs the on-chain commitment side
no longer accepts.
"""

import pytest

from packages.valory.skills.task_execution.utils.local_cid import compute_cidv1


# Tuples of (label, content, expected_cid_from_real_ipfs_add).
_FIXTURES = [
    (
        "empty",
        b"",
        "bafybeif7ztnhq65lumvvtr4ekcwd2ifwgm3awq4zfr3srh462rwyinlb4y",
    ),
    (
        "small_text",
        b"hello",
        "bafybeid3weurg3gvyoi7nisadzolomlvoxoppe2sesktnpvdve3256n5tq",
    ),
    (
        "small_json",
        b'{"requestId":"42","result":"ok"}',
        "bafybeihjg4w34tu2zmlriemthl24wdynhfx2rpsiflv7wxmb4lsk34etlu",
    ),
    (
        "100k_repeated",
        b"x" * 100000,
        "bafybeiay23kics7rguz4kaxxmyz7d6bciozygbi27wllerri6dpqenuifi",
    ),
]


@pytest.mark.parametrize(
    "label,content,expected",
    _FIXTURES,
    ids=[fx[0] for fx in _FIXTURES],
)
def test_compute_cidv1_matches_real_ipfs_output(
    label: str, content: bytes, expected: str
) -> None:
    """Local CIDv1 matches ``ipfs add --cid-version=1 --raw-leaves=false`` byte-for-byte."""
    assert compute_cidv1(content) == expected


def test_compute_cidv1_oversize_raises() -> None:
    """Content beyond the single-block bound fails loudly rather than silently."""
    oversized = b"a" * (1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds single-block bound"):
        compute_cidv1(oversized)


def test_compute_cidv1_is_deterministic() -> None:
    """Identical input always produces the same CID — no hidden state."""
    content = b'{"different":"payload"}'
    assert compute_cidv1(content) == compute_cidv1(content)
