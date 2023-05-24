import multibase
import multicodec
import sys

from aea.helpers.cid import to_v1
from aea_cli_ipfs.ipfs_utils import IPFSTool


def main(file):
    response = IPFSTool().client.add(file, pin=True, recursive=True, wrap_with_directory=False)
    file_hash = to_v1(response["Hash"])
    print("IPFS file hash v1: {}".format(file_hash))
    cid_bytes = multibase.decode(file_hash)
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    result = "f01" + multihash_bytes.hex()
    print("IPFS file hash v1 hex: {}".format(result))

if __name__ == '__main__':
    _, file_path = sys.argv
    main(file_path)

