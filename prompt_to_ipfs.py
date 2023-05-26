import json
import shutil
import sys
import tempfile
import uuid
from push_to_ipfs import push_to_ipfs


def push_metadata_to_ipfs(prompt, tool):
    metadata = {"prompt": prompt, "tool": tool, "nonce": str(uuid.uuid4())}
    dirpath = tempfile.mkdtemp()
    file_name = dirpath + 'metadata.json'
    with open(file_name, "w") as f:
        json.dump(metadata, f)
    _, v1_file_hash_hex = push_to_ipfs(file_name)
    shutil.rmtree(dirpath)
    return "0x" + v1_file_hash_hex[6:], v1_file_hash_hex


def main(prompt, tool):
    v1_file_hash_hex_truncated, v1_file_hash_hex = push_metadata_to_ipfs(prompt, tool)
    print("Visit url: https://gateway.autonolas.tech/ipfs/{}".format(v1_file_hash_hex))
    print("Hash for Request method: {}".format(v1_file_hash_hex_truncated))


if __name__ == '__main__':
    _, prompt, tool = sys.argv
    main(prompt, tool)
