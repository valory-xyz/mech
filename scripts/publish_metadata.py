# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
"""The script allows the user to publish the metadata of the tools on ipfs"""
import argparse
import sys

from aea.helpers.cid import to_v1
from aea_cli_ipfs.ipfs_utils import IPFSTool
from multibase import multibase
from multicodec import multicodec

from scripts.generate_metadata import METADATA_FILE_PATH


PREFIX = "f01701220"
IPFS_PREFIX_LENGTH = 6
RESPONSE_KEY = "Hash"
DEFAULT_IPFS_NODE = "/dns/registry.autonolas.tech/tcp/443/https"


def push_metadata_to_ipfs() -> None:
    """Pushes the metadata to ipfs"""
    parser = argparse.ArgumentParser(description="Pushes metadata.json to ipfs")
    parser.add_argument("--ipfs-node", type=str, default=DEFAULT_IPFS_NODE)
    args = parser.parse_args()
    addr = args.ipfs_node
    try:
        response = IPFSTool(addr=addr).client.add(
            METADATA_FILE_PATH, pin=True, recursive=True, wrap_with_directory=False
        )
    except Exception as e:
        print(f"Error pushing metadata to ipfs: {e}")
        sys.exit(1)

    if RESPONSE_KEY not in response:
        print(f"Key '{RESPONSE_KEY!r}' not found in ipfs response")
        sys.exit(1)

    cid_bytes = multibase.decode(to_v1(response[RESPONSE_KEY]))
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    hex_multihash = multihash_bytes.hex()
    ipfs_hash = PREFIX + hex_multihash[IPFS_PREFIX_LENGTH:]
    print(f"Metadata successfully pushed to ipfs. The metadata hash is: {ipfs_hash}")


def main() -> None:
    """Run the publish_metadata script."""
    push_metadata_to_ipfs()


if __name__ == "__main__":
    main()
