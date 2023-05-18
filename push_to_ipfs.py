import sys
from aea_cli_ipfs.ipfs_utils import IPFSTool
from aea.helpers.cid import to_v1

def main(file):
    response = IPFSTool().client.add(file, pin=True, recursive=True, wrap_with_directory=False)
    file_hash = to_v1(response["Hash"])
    print("IPFS file hash: {}".format(file_hash))

if __name__ == '__main__':
    _, file_path = sys.argv
    main(file_path)
