import threading
from types import SimpleNamespace
from concurrent.futures import Future
from collections import defaultdict
import pytest

from packages.valory.skills.task_execution.behaviours import (
    TaskExecutionBehaviour,
    PENDING_TASKS,
    DONE_TASKS,
    IPFS_TASKS,
    DONE_TASKS_LOCK,
    REQUEST_ID_TO_DELIVERY_RATE_INFO,
)

@pytest.fixture
def shared_state():
    """
    The behaviour reads/writes these keys on context.shared_state.
    Start with empty lists and a lock.
    """
    return {
        PENDING_TASKS: [],
        DONE_TASKS: [],
        IPFS_TASKS: [],
        DONE_TASKS_LOCK: threading.Lock(),
        REQUEST_ID_TO_DELIVERY_RATE_INFO: {},
    }

@pytest.fixture
def params_stub():
    """
    Minimal Params-like object with only the attributes the behaviour touches.
    We use SimpleNamespace to avoid pulling in the real Params class.
    """
    return SimpleNamespace(
        tools_to_package_hash={},     # e.g. {"sum": "bafy..."}
        tools_to_pricing={},          # e.g. {"sum": 100}
        api_keys={},

        req_params=SimpleNamespace(from_block={}, last_polling={}),
        polling_interval=10.0,
        in_flight_req=False,
        req_to_callback={},
        req_to_deadline={},
        req_type=None,

        default_chain_id=100,
        agent_mech_contract_addresses=["0x0000000000000000000000000000000000000000"],
        mech_to_config={
            "0xmech": SimpleNamespace(is_marketplace_mech=False)
        },
        use_mech_marketplace=False,
        mech_marketplace_address=None,
        max_block_window=10_000,

        task_deadline=10.0,
        timeout_limit=2,
        request_id_to_num_timeouts=defaultdict(int),
        is_cold_start=True,
    )

@pytest.fixture
def context_stub(shared_state, params_stub):
    """
    Bare-bones AEA context with just enough for this behaviour.
    We stub the logger and outbox to no-ops so tests don't need real connections.
    """
    class Outbox:
        def put_message(self, *_, **__):
            pass

    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    return SimpleNamespace(
        logger=logger,
        shared_state=shared_state,
        params=params_stub,
        agent_address="0xagent",
        default_ledger_id="ethereum",
        outbox=Outbox(),

        ipfs_dialogues=None,
        contract_dialogues=None,
        ledger_dialogues=None,
        acn_data_share_dialogues=None,
    )

@pytest.fixture
def behaviour(context_stub):
    """
    An instance of TaskExecutionBehaviour wired up with our stub context.
    We call setup() so it pulls params/api keys like it would at runtime.
    """
    b = TaskExecutionBehaviour(name="task_execution", skill_context=context_stub)
    b.setup()
    return b

class FakeDialogue:
    """Mimics a dialogue label with a stable nonce (the behaviour stores callbacks by nonce)."""
    class Label:
        dialogue_reference = ("nonce-1", "x")
    dialogue_label = Label()

class FakeIpfsMsg:
    """Minimal shape of an IpfsMessage used by the behaviour in callbacks."""
    def __init__(self, files=None, ipfs_hash=None):
        self.files = files or {}
        self.ipfs_hash = ipfs_hash

@pytest.fixture
def fake_dialogue():
    return FakeDialogue()

@pytest.fixture
def done_future():
    """Factory: return a Future already completed with the given value."""
    def _make(value):
        f = Future()
        f.set_result(value)
        return f
    return _make
