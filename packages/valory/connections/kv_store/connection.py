#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""Key-value connection and channel."""

from pathlib import Path
from typing import Any, Callable, Optional, cast

from aea.configurations.base import PublicId
from aea.connections.base import BaseSyncConnection
from aea.mail.base import Envelope
from aea.protocols.base import Address, Message
from aea.protocols.dialogue.base import Dialogue
from peewee import CharField, Model, SqliteDatabase, TextField  # type: ignore

from packages.valory.protocols.kv_store.dialogues import (
    KvStoreDialogue,
)
from packages.valory.protocols.kv_store.dialogues import (
    KvStoreDialogues as BaseKvStoreDialogues,
)
from packages.valory.protocols.kv_store.message import KvStoreMessage

PUBLIC_ID = PublicId.from_str("valory/kv_store:0.1.0")

_GENERIC_ERROR_MESSAGE = "Internal handler error"

# LIST_REQUEST page-size policy. Clients that pass limit=0 (the protobuf
# default, i.e. unset) get _LIST_DEFAULT_PAGE_SIZE. Any positive client value
# is clamped to _LIST_MAX_PAGE_SIZE so a single LIST_REQUEST cannot force the
# server to materialize an unbounded response. The default sits well below the
# clamp so the common case stays predictable for sweepers without forcing
# every caller to think about paging.
_LIST_DEFAULT_PAGE_SIZE = 100
_LIST_MAX_PAGE_SIZE = 1000


db = SqliteDatabase(
    None,
    pragmas={
        "journal_mode": "wal",
        "foreign_keys": 1,
        "busy_timeout": 5000,
    },
)


class BaseModel(Model):
    """Database base model"""

    class Meta:  # noqa pylint: disable=too-few-public-methods
        """Database meta model, as required per peewee"""

        database = db  # noqa: F841


class Store(BaseModel):
    """Database Store table"""

    key = CharField(unique=True)
    value = TextField()


class KvStoreDialogues(BaseKvStoreDialogues):
    """A class to keep track of KvStore dialogues."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize dialogues.

        :param kwargs: keyword arguments
        """

        def role_from_first_message(  # pylint: disable=unused-argument
            message: Message, receiver_address: Address
        ) -> Dialogue.Role:
            """Infer the role of the agent from an incoming/outgoing first message

            :param message: an incoming/outgoing first message
            :param receiver_address: the address of the receiving agent
            :return: The role of the agent
            """
            return KvStoreDialogue.Role.CONNECTION

        BaseKvStoreDialogues.__init__(
            self,
            self_address=str(kwargs.pop("connection_id")),
            role_from_first_message=role_from_first_message,
            **kwargs,
        )


class KvStoreConnection(BaseSyncConnection):
    """Proxy to the functionality of the SDK or API."""

    MAX_WORKER_THREADS = 5

    connection_id = PUBLIC_ID

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """
        Initialize the connection.

        The configuration must be specified if and only if the following
        parameters are None: connection_id, excluded_protocols or restricted_to_protocols.

        Possible arguments:
        - configuration: the connection configuration.
        - data_dir: directory where to put local files.
        - identity: the identity object held by the agent.
        - crypto_store: the crypto store for encrypted communication.
        - restricted_to_protocols: the set of protocols ids of the only supported protocols for this connection.
        - excluded_protocols: the set of protocols ids that we want to exclude for this connection.

        :param args: arguments passed to component base
        :param kwargs: keyword arguments passed to component base
        """
        super().__init__(*args, **kwargs)
        self.dialogues = KvStoreDialogues(connection_id=PUBLIC_ID)

        store_path = Path(self.configuration.config.get("store_path", "/data"))  # nosec

        if not store_path.exists():
            store_path.mkdir(parents=True, exist_ok=True)

        self.db_path = str(store_path / "memeooorr.db")  # nosec

    def main(self) -> None:
        """
        Run synchronous code in background.

        SyncConnection `main()` usage:
        The idea of the `main` method in the sync connection
        is to provide for a way to actively generate messages by the connection via the `put_envelope` method.

        A simple example is the generation of a message every second:
        ```
        while self.is_connected:
            envelope = make_envelope_for_current_time()
            self.put_enevelope(envelope)
            time.sleep(1)
        ```
        In this case, the connection will generate a message every second
        regardless of envelopes sent to the connection by the agent.
        For instance, this way one can implement periodically polling some internet resources
        and generate envelopes for the agent if some updates are available.
        Another example is the case where there is some framework that runs blocking
        code and provides a callback on some internal event.
        This blocking code can be executed in the main function and new envelops
        can be created in the event callback.
        """

    def _error_reply(
        self, dialogue: KvStoreDialogue, target_message: KvStoreMessage
    ) -> Optional[KvStoreMessage]:
        """Build an ERROR reply, returning None if the reply itself fails."""
        try:
            return cast(
                KvStoreMessage,
                dialogue.reply(
                    performative=KvStoreMessage.Performative.ERROR,
                    target_message=target_message,
                    message=_GENERIC_ERROR_MESSAGE,
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("Failed to build ERROR reply; dropping message.")
            return None

    def on_send(self, envelope: Envelope) -> None:
        """
        Send an envelope.

        :param envelope: the envelope to send.
        """
        kv_store_message = cast(KvStoreMessage, envelope.message)
        dialogue: Optional[KvStoreDialogue] = self.dialogues.update(kv_store_message)

        if dialogue is None:
            self.logger.error(
                "Could not associate dialogue with message "
                f"(performative={kv_store_message.performative.value}, "
                f"dialogue_reference={kv_store_message.dialogue_reference})."
            )
            return

        response: Optional[KvStoreMessage]
        try:
            handler: Callable[
                [KvStoreMessage, KvStoreDialogue], Optional[KvStoreMessage]
            ] = getattr(self, kv_store_message.performative.value)
            response = handler(kv_store_message, dialogue)
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception(
                f"Handler `{kv_store_message.performative.value}` raised."
            )
            response = self._error_reply(dialogue, kv_store_message)

        if response is None:
            return

        response_envelope = Envelope(
            to=envelope.sender,
            sender=envelope.to,
            message=response,
            context=envelope.context,
        )
        self.put_envelope(response_envelope)

    def read_request(
        self,
        message: KvStoreMessage,
        dialogue: KvStoreDialogue,
    ) -> Optional[KvStoreMessage]:
        """Read several keys."""
        keys = message.keys
        self.logger.info(f"DB read: {len(keys)} keys")
        self.logger.debug(f"DB read keys: {keys}")
        try:
            query = Store.select().where(Store.key.in_(keys))
            response_data = {entry.key: entry.value for entry in query}
            return cast(
                KvStoreMessage,
                dialogue.reply(
                    performative=KvStoreMessage.Performative.READ_RESPONSE,
                    target_message=message,
                    data=response_data,
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("DB read failed.")
            return self._error_reply(dialogue, message)

    def create_or_update_request(
        self,
        message: KvStoreMessage,
        dialogue: KvStoreDialogue,
    ) -> Optional[KvStoreMessage]:
        """Write several key-value pairs."""

        self.logger.info(f"DB write: {len(message.data)} keys")
        self.logger.debug(f"DB write keys: {list(message.data)}")

        try:
            with db.atomic(lock_type="IMMEDIATE"):
                for k, v in message.data.items():
                    entry = Store.get_or_none(Store.key == k)

                    if not entry:
                        Store.create(key=k, value=v)
                    else:
                        entry.value = v
                        entry.save()

            return cast(
                KvStoreMessage,
                dialogue.reply(
                    performative=KvStoreMessage.Performative.SUCCESS,
                    target_message=message,
                    message="OK",
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("DB write failed.")
            return self._error_reply(dialogue, message)

    def delete_request(
        self,
        message: KvStoreMessage,
        dialogue: KvStoreDialogue,
    ) -> Optional[KvStoreMessage]:
        """Delete several keys.

        Missing keys are not an error: the operation is set-difference
        semantics, so the caller can re-run the same delete idempotently.
        An empty key list is a no-op that still returns SUCCESS.
        """
        keys = message.keys
        self.logger.info(f"DB delete: {len(keys)} keys")
        self.logger.debug(f"DB delete keys: {keys}")

        try:
            with db.atomic(lock_type="IMMEDIATE"):
                if keys:
                    Store.delete().where(Store.key.in_(keys)).execute()

            return cast(
                KvStoreMessage,
                dialogue.reply(
                    performative=KvStoreMessage.Performative.SUCCESS,
                    target_message=message,
                    message="OK",
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("DB delete failed.")
            return self._error_reply(dialogue, message)

    def list_request(
        self,
        message: KvStoreMessage,
        dialogue: KvStoreDialogue,
    ) -> Optional[KvStoreMessage]:
        """List key-value pairs whose key starts with the given prefix, paged.

        Prefix matching is case-sensitive (BINARY collation), so a sweeper that
        lists-then-deletes by prefix can't over-match a different-cased
        namespace. An empty ``key_prefix`` matches every row.

        Paging keeps the response size bounded regardless of namespace size:
        the client passes ``limit`` (clamped to ``_LIST_MAX_PAGE_SIZE``, with
        ``_LIST_DEFAULT_PAGE_SIZE`` used when ``limit == 0``) and an opaque
        ``cursor`` returned by the prior page. The server orders rows by key
        and starts each page strictly after ``cursor``, so concurrent writes
        between pages don't shift the iteration. ``next_cursor`` is the last
        returned key when there is more to read, or empty when the page is the
        last one.
        """
        prefix = message.key_prefix
        cursor = message.cursor
        requested_limit = message.limit
        page_size = self._resolve_page_size(requested_limit)
        self.logger.info(
            f"DB list: prefix=({prefix!r}) len={len(prefix)} "
            f"limit={requested_limit} cursor=({cursor!r}) "
            f"effective_page_size={page_size}"
        )

        try:
            if prefix:
                # Case-sensitive prefix match. SQLite LIKE (what peewee's
                # startswith emits) is case-insensitive for ASCII, which would
                # over-match — dangerous for the delete-driving sweeper. TEXT
                # uses BINARY collation, so a half-open range
                # [prefix, prefix + max-code-point) is case-sensitive and
                # index-friendly, and avoids LIKE wildcard semantics entirely.
                upper_sentinel = prefix + chr(0x10FFFF)
                query = Store.select().where(
                    (Store.key >= prefix) & (Store.key < upper_sentinel)
                )
            else:
                # Empty prefix = full-table scan. Surface it so an operator can
                # distinguish an intentional list-all from a sender that forgot
                # to set key_prefix (the protobuf string default is "").
                self.logger.warning(
                    "DB list called with empty key_prefix; "
                    "returning a page of the full table."
                )
                query = Store.select()

            if cursor:
                # Strict > so the row whose key matched the previous page's
                # next_cursor is not re-emitted. Cursor pointing at a now-deleted
                # key is still well-defined: the next-greater key wins.
                query = query.where(Store.key > cursor)

            # Order by key and pull one extra row to know whether another page
            # follows. This avoids a second COUNT query and keeps the cursor
            # scheme stateless.
            query = query.order_by(Store.key).limit(page_size + 1)

            rows = list(query)
            has_more = len(rows) > page_size
            if has_more:
                rows = rows[:page_size]
            response_data = {row.key: row.value for row in rows}
            # `has_more` is only True when len(rows) > page_size and page_size
            # is always >= 1, so rows is guaranteed non-empty under has_more.
            next_cursor = rows[-1].key if has_more else ""
            return cast(
                KvStoreMessage,
                dialogue.reply(
                    performative=KvStoreMessage.Performative.LIST_RESPONSE,
                    target_message=message,
                    data=response_data,
                    next_cursor=next_cursor,
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("DB list failed.")
            return self._error_reply(dialogue, message)

    @staticmethod
    def _resolve_page_size(requested_limit: int) -> int:
        """Pick the effective page size, defaulting and clamping the client value.

        ``requested_limit == 0`` means "let the server choose"; any positive
        value is clamped to ``_LIST_MAX_PAGE_SIZE`` so a buggy or hostile caller
        can't force the server to materialize an unbounded response.
        """
        if requested_limit <= 0:
            return _LIST_DEFAULT_PAGE_SIZE
        return min(requested_limit, _LIST_MAX_PAGE_SIZE)

    def on_connect(self) -> None:
        """Set up the connection"""
        db.init(self.db_path)
        self.logger.info(f"KV database initialized in {self.db_path}")
        db.connect()
        self.logger.info("KV database connection established")
        db.create_tables([Store])

    def on_disconnect(self) -> None:
        """
        Tear down the connection.

        Connection status set automatically.
        """
        if db.is_closed():
            return
        try:
            db.close()
            self.logger.info("KV database connection closed")
        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("KV database close failed.")
