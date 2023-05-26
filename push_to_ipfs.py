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

"""
This script allows pushing a file directly to IPFS.

Usage:

python push_to_ipfs.py <file_path>
"""

import sys
from typing import Tuple

import multibase
import multicodec
from aea.helpers.cid import to_v1
from aea_cli_ipfs.ipfs_utils import IPFSTool


def push_to_ipfs(file_path: str) -> Tuple[str, str]:
    response = IPFSTool().client.add(
        file_path, pin=True, recursive=True, wrap_with_directory=False
    )
    v1_file_hash = to_v1(response["Hash"])
    cid_bytes = multibase.decode(v1_file_hash)
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    v1_file_hash_hex = "f01" + multihash_bytes.hex()
    return v1_file_hash, v1_file_hash_hex


def main(file_path: str) -> None:
    v1_file_hash, v1_file_hash_hex = push_to_ipfs(file_path)
    print("IPFS file hash v1: {}".format(v1_file_hash))
    print("IPFS file hash v1 hex: {}".format(v1_file_hash_hex))


if __name__ == "__main__":
    _, file_path = sys.argv
    main(file_path)
