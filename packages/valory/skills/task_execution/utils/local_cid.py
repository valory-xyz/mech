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
"""Pure-Python local CIDv1 computation matching IPFS UnixFS + DAG-PB single-block output.

Produces the same CIDv1 string that ``ipfs add --cid-version=1 --raw-leaves=false``
would for the same bytes. Intended for the offchain delivery path where we put the
content commitment on chain without publishing the content to public IPFS.

Single-block only: content must fit in one IPFS block. IPFS chunks at 256 KiB by
default; payloads above that bound need a chunked / linked DAG and are not handled
here. Hash sanity asserts the bound at runtime so a future oversized response fails
loudly rather than silently producing a CID a real IPFS upload would not match.
"""

import base64
import hashlib

_MAX_BLOCK_BYTES = 1024 * 1024  # IPFS default chunker is 256 KiB; one MiB is plenty.

# Multicodec / multihash / multibase prefix bytes.
_CIDV1_VERSION = 0x01
_DAG_PB_CODEC = 0x70
_SHA256_MULTIHASH_CODE = 0x12
_SHA256_DIGEST_LEN = 0x20

# UnixFS data type enum: File = 2.
_UNIXFS_TYPE_FILE = 2


def _varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint.

    :param value: the non-negative integer to encode.
    :return: the encoded varint bytes.
    :raises ValueError: if ``value`` is negative.
    """
    if value < 0:
        raise ValueError("varint requires non-negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _length_delimited(field_number: int, payload: bytes) -> bytes:
    """Encode a protobuf length-delimited field (wire type 2).

    :param field_number: the protobuf field number.
    :param payload: the field payload bytes.
    :return: the encoded tag + length + payload.
    """
    tag = (field_number << 3) | 2
    return _varint(tag) + _varint(len(payload)) + payload


def _varint_field(field_number: int, value: int) -> bytes:
    """Encode a protobuf varint field (wire type 0).

    :param field_number: the protobuf field number.
    :param value: the field value.
    :return: the encoded tag + value.
    """
    tag = (field_number << 3) | 0
    return _varint(tag) + _varint(value)


def _encode_unixfs_file(content: bytes) -> bytes:
    """Encode a UnixFS ``Type=File`` Data message for the given content.

    Mirrors the canonical encoding go-ipfs produces for a single-block file: a
    Type varint (field 1), optional Data bytes (field 2), and the filesize
    varint (field 3). The Data field is omitted entirely for empty content,
    matching go-ipfs' behaviour.

    :param content: the raw file content.
    :return: the serialized UnixFS message bytes.
    """
    out = bytearray()
    out += _varint_field(1, _UNIXFS_TYPE_FILE)
    if content:
        out += _length_delimited(2, content)
    out += _varint_field(3, len(content))
    return bytes(out)


def _encode_dag_pb_node(data: bytes) -> bytes:
    """Wrap ``data`` in a DAG-PB ``PBNode`` with no links.

    The DAG-PB ``PBNode`` schema has ``Data`` as field 1 and ``Links`` as field 2.
    For a single-block file we have no links, only the Data field carrying the
    serialized UnixFS message.

    :param data: the serialized UnixFS bytes to wrap.
    :return: the serialized DAG-PB node bytes.
    """
    return _length_delimited(1, data)


def _multibase_base32_lower(payload: bytes) -> str:
    """Encode bytes as RFC 4648 base32 lowercase without padding, prefixed with 'b'.

    Multibase prefix ``b`` selects base32-lower; this is the encoding ``ipfs add
    --cid-version=1`` emits by default (CIDs starting with ``bafy...`` for
    DAG-PB).

    :param payload: the bytes to multibase-encode.
    :return: the multibase string with the ``b`` prefix.
    """
    encoded = base64.b32encode(payload).decode("ascii").rstrip("=").lower()
    return "b" + encoded


def compute_cidv1(content: bytes) -> str:
    """Compute the CIDv1 string a real ``ipfs add`` would produce for ``content``.

    The output is byte-for-byte equivalent to running
    ``ipfs add --cid-version=1 --raw-leaves=false`` on the same content. The
    returned string is a multibase base32-lower (``bafy...``) DAG-PB CIDv1 over
    a SHA-256 multihash of the UnixFS-wrapped content.

    :param content: the content bytes to commit to.
    :return: the multibase-encoded CIDv1 string.
    :raises ValueError: if ``content`` exceeds the single-block bound; chunked
        DAGs are intentionally unsupported (offchain payloads in the mech are
        well under 256 KiB today and the unbounded path needs a different DAG
        shape).
    """
    if len(content) > _MAX_BLOCK_BYTES:
        raise ValueError(
            f"content size {len(content)} exceeds single-block bound "
            f"{_MAX_BLOCK_BYTES}; chunked DAG encoding is not supported"
        )

    unixfs_bytes = _encode_unixfs_file(content)
    dag_pb_bytes = _encode_dag_pb_node(unixfs_bytes)

    digest = hashlib.sha256(dag_pb_bytes).digest()
    multihash = bytes([_SHA256_MULTIHASH_CODE, _SHA256_DIGEST_LEN]) + digest
    cid_bytes = bytes([_CIDV1_VERSION, _DAG_PB_CODEC]) + multihash

    return _multibase_base32_lower(cid_bytes)
