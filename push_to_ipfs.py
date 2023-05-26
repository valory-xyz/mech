import sys
from typing import Tuple

import multibase
import multicodec
from aea.helpers.cid import to_v1
from aea_cli_ipfs.ipfs_utils import IPFSTool


def push_to_ipfs(file) -> Tuple[str, str]:
    response = IPFSTool().client.add(file, pin=True, recursive=True, wrap_with_directory=False)
    v1_file_hash = to_v1(response["Hash"])
    cid_bytes = multibase.decode(v1_file_hash)
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    v1_file_hash_hex = "f01" + multihash_bytes.hex()
    return v1_file_hash, v1_file_hash_hex

def main(file):
    v1_file_hash, v1_file_hash_hex = push_to_ipfs(file)
    print("IPFS file hash v1: {}".format(v1_file_hash))
    print("IPFS file hash v1 hex: {}".format(v1_file_hash_hex))


if __name__ == '__main__':
    _, file_path = sys.argv
    main(file_path)

