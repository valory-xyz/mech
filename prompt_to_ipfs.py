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
This script allows pushing a prompt compatible with the on-chain mechs directly to IPFS.

Usage:

python push_to_ipfs.py <prompt> <tool>
"""

import json
import shutil
import sys
import tempfile
import uuid
from typing import Tuple

from push_to_ipfs import push_to_ipfs


def push_metadata_to_ipfs(prompt: str, tool: str) -> Tuple[str, str]:
    metadata = {"prompt": prompt, "tool": tool, "nonce": str(uuid.uuid4())}
    dirpath = tempfile.mkdtemp()
    file_name = dirpath + "metadata.json"
    with open(file_name, "w") as f:
        json.dump(metadata, f)
    _, v1_file_hash_hex = push_to_ipfs(file_name)
    shutil.rmtree(dirpath)
    return "0x" + v1_file_hash_hex[6:], v1_file_hash_hex


def main(prompt: str, tool: str) -> None:
    v1_file_hash_hex_truncated, v1_file_hash_hex = push_metadata_to_ipfs(prompt, tool)
    print("Visit url: https://gateway.autonolas.tech/ipfs/{}".format(v1_file_hash_hex))
    print("Hash for Request method: {}".format(v1_file_hash_hex_truncated))


if __name__ == "__main__":
    _, prompt, tool = sys.argv
    main(prompt, tool)
