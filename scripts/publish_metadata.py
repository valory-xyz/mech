from multibase import multibase
from multicodec import multicodec
from aea.helpers.cid import to_v1
from aea_cli_ipfs.ipfs_utils import IPFSTool
from scripts.generate_metadata import METADATA_FILE_PATH

PREFIX = "f01701220"


def push_metadata_to_ipfs() -> None:
    response = IPFSTool().client.add(
        METADATA_FILE_PATH, pin=True, recursive=True, wrap_with_directory=False
    )
    cid_bytes = multibase.decode(to_v1(response["Hash"]))
    multihash_bytes = multicodec.remove_prefix(cid_bytes)
    hex_multihash = multihash_bytes.hex()
    ipfs_hash = PREFIX + hex_multihash[6:]
    print(f"Metadata successfully pushed to ipfs. The metadata hash is: {ipfs_hash}")


def main() -> None:
    push_metadata_to_ipfs()


if __name__ == "__main__":
    main()
