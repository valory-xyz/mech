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
"""This module contains helpers for IPFS interaction."""
from aea.helpers.cid import CID
from multibase import multibase
from multicodec import multicodec

CID_PREFIX = "f01701220"


def get_ipfs_file_hash(data: bytes) -> str:
    """Get hash from bytes"""
    try:
        return str(CID.from_string(data.decode()))
    except Exception:  # noqa
        # if something goes wrong, fallback to sha256
        file_hash = data.hex()
        file_hash = CID_PREFIX + file_hash
        file_hash = str(CID.from_string(file_hash))
        return file_hash


def to_multihash(hash_string: str) -> bytes:
    """To multihash string."""
    # Decode the Base32 CID to bytes
    cid_bytes = multibase.decode(hash_string)
    # Remove the multicodec prefix (0x01) from the bytes
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    # Convert the multihash bytes to a hexadecimal string
    hex_multihash = multihash_bytes.hex()
    return hex_multihash[6:]
