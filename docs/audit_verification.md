# Audit Verification Report

**Source:** `AUDIT.md` (FSM Audit Report dated 2026-03-05)
**Verified on:** 2026-03-09

---

## C-1: None appended to transaction list in delivery rate behaviour

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **File** | `packages/valory/skills/delivery_rate_abci/behaviours.py:183` |
| **Verdict** | **TRUE** |

### Claim
When `_get_delivery_rate_update_tx()` returns `None`, the code logs a warning but still appends `None` to `txs`. This list is later passed to `_to_multisend()`, which accesses `transaction["to"]` at line 209, causing `TypeError: 'NoneType' object is not subscriptable`.

### Reasoning
The code at lines 174-183 is:
```python
tx = yield from self._get_delivery_rate_update_tx(
    mech_address, delivery_rate
)
if tx is None:
    self.context.logger.warning(
        f"Could not get delivery_rate update tx for {mech_address}."
    )

txs.append(tx)  # BUG: appends None unconditionally
```

There is no `continue` or `return` after the `if tx is None` block, so execution falls through to `txs.append(tx)`. The similar guard patterns at lines 163-166 and 167-172 in the same loop do use `continue`, confirming this is an oversight rather than intentional design.

### Recommended Fix
Add `continue` after the warning log to skip the append:
```python
if tx is None:
    self.context.logger.warning(
        f"Could not get delivery_rate update tx for {mech_address}."
    )
    continue

txs.append(tx)
```

---

## H-1: ThreadPoolExecutor never shut down on disconnect

| Field | Value |
|-------|-------|
| **Severity** | High |
| **File** | `packages/valory/connections/websocket_client/connection.py:306,318-327` |
| **Verdict** | **TRUE** |

### Claim
`connect()` creates a `ThreadPoolExecutor` at line 306, but `disconnect()` never calls `shutdown()` on it, causing thread resource leaks.

### Reasoning
The `connect()` method creates the executor:
```python
self._executor = ThreadPoolExecutor()  # line 306
```

The `disconnect()` method only does:
```python
async def disconnect(self) -> None:
    await self._manager.remove_all_subscriptions()
    self._outbox.empty()
    self.state = ConnectionStates.disconnected
```

There is no `self._executor.shutdown()` call anywhere in `disconnect()`. Worker threads from the pool will persist after disconnection, leaking OS-level resources.

### Recommended Fix
Add executor shutdown before setting state to disconnected:
```python
async def disconnect(self) -> None:
    await self._manager.remove_all_subscriptions()
    self._outbox.empty()
    self._executor.shutdown(wait=True)
    self.state = ConnectionStates.disconnected
```

---

## H-2: ProcessPool.join() without timeout

| Field | Value |
|-------|-------|
| **Severity** | High |
| **File** | `packages/valory/skills/task_execution/behaviours.py:830-831` |
| **Verdict** | **TRUE** |

### Claim
`self._executor.join()` is called without a timeout. If a child process is stuck, this blocks the agent indefinitely.

### Reasoning
The `_restart_executor` method at lines 828-833:
```python
def _restart_executor(self) -> None:
    """Restarts the executor."""
    self._executor.stop()
    self._executor.join()  # No timeout — blocks indefinitely if worker is stuck
    self._executor = ProcessPool(max_workers=1)
```

Pebble's `ProcessPool.join()` without a timeout will wait forever for worker processes to finish. A stuck worker (e.g. infinite loop in a tool, deadlock) will hang the entire agent.

### Recommended Fix
Add a reasonable timeout:
```python
self._executor.join(timeout=10.0)
```

---

## M-1: Offchain task filtering uses unsynchronized shared state

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **File** | `packages/valory/skills/task_submission_abci/behaviours.py:1597` |
| **Verdict** | **TRUE** |

### Claim
`_get_offchain_tasks_deliver_data()` reads tasks from `self.done_tasks` (local shared state, not consensus-synchronized), while sibling filters at lines 1376 and 1389 use `self.synchronized_data.done_tasks`.

### Reasoning
The `done_tasks` property (lines 167-176) explicitly warns in its own docstring:
```python
@property
def done_tasks(self) -> List[Dict[str, Any]]:
    """
    Return the done (ready) tasks from shared state.

    Use with care, the returned data here is NOT synchronized with the rest of the agents.
    """
    done_tasks = deepcopy(self.context.shared_state.get(DONE_TASKS, []))
    return cast(List[Dict[str, Any]], done_tasks)
```

At line 1597:
```python
done_tasks_list = self.done_tasks  # local, NOT consensus-synchronized
```

While lines 1376 and 1389 use:
```python
for done_task in self.synchronized_data.done_tasks  # consensus-synchronized
```

This inconsistency means offchain task lists may differ across agents, which could lead to agents disagreeing on which tasks to deliver.

### Recommended Fix
Use `self.synchronized_data.done_tasks` consistently, or document explicitly why offchain tasks are intentionally agent-local (if that is the design intent).

---

## M-2: `final_tx_hash` in db_post_conditions but never set by this FSM

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **File** | `packages/valory/skills/task_submission_abci/rounds.py:255` |
| **Verdict** | **TRUE** |

### Claim
`FinishedTaskPoolingRound` post-conditions declare `{"most_voted_tx_hash", "final_tx_hash"}`, but `final_tx_hash` is never written within `TaskSubmissionAbciApp`.

### Reasoning
The post-conditions at line 255:
```python
db_post_conditions: Dict[AppState, Set[str]] = {
    FinishedTaskPoolingRound: {"most_voted_tx_hash", "final_tx_hash"},
    ...
}
```

A search for `final_tx_hash` within `task_submission_abci` shows only a READ at line 335:
```python
final_tx_hash = self.synchronized_data.final_tx_hash
```

No code in this FSM writes `final_tx_hash` to the synchronized data. It is set by the downstream `TransactionSettlementAbciApp`. The post-condition cannot be satisfied by this FSM alone. It is already listed in `cross_period_persisted_keys` (line 248), which is the appropriate place for it.

### Recommended Fix
Remove `"final_tx_hash"` from the post-conditions:
```python
FinishedTaskPoolingRound: {"most_voted_tx_hash"},
```

---

## L-1: Docstring drift in composition.py

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **File** | `packages/valory/skills/mech_abci/composition.py:20` |
| **Verdict** | **TRUE** |

### Claim
Docstring says `"round behaviours"` but the file defines FSM composition.

### Reasoning
Line 20:
```python
"""This package contains round behaviours of MechAbciApp."""
```

The file actually defines `abci_app_transition_mapping` — the FSM composition that chains multiple ABCI apps together. It contains no round behaviours.

### Recommended Fix
```python
"""This package contains the FSM composition for MechAbciApp."""
```

---

## L-2: Docstring drift in task_submission_abci

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Files** | `packages/valory/skills/task_submission_abci/behaviours.py:20`, `packages/valory/skills/task_submission_abci/models.py:20` |
| **Verdict** | **TRUE** |

### Claim
Docstrings reference `TaskExecutionAbciApp` instead of `TaskSubmissionAbciApp`.

### Reasoning
`behaviours.py` line 20:
```python
"""This package contains round behaviours of TaskExecutionAbciApp."""
```

`models.py` line 20:
```python
"""This module contains the shared state for the abci skill of TaskExecutionAbciApp."""
```

The actual FSM class used in `models.py` (line 44) is `TaskSubmissionAbciApp`, not `TaskExecutionAbciApp`. These are copy-paste errors from the similarly-named `task_execution` skill.

### Recommended Fix
```python
# behaviours.py
"""This package contains round behaviours of TaskSubmissionAbciApp."""

# models.py
"""This module contains the shared state for the abci skill of TaskSubmissionAbciApp."""
```

---

## Summary

| ID  | Severity | Verdict | Action Needed |
|-----|----------|---------|---------------|
| C-1 | Critical | TRUE | Add `continue` after None warning |
| H-1 | High     | TRUE | Add `executor.shutdown(wait=True)` in disconnect |
| H-2 | High     | TRUE | Add `timeout=10.0` to `join()` |
| M-1 | Medium   | TRUE | Use `synchronized_data.done_tasks` or document intent |
| M-2 | Medium   | TRUE | Remove `final_tx_hash` from post-conditions |
| L-1 | Low      | TRUE | Fix docstring to say "FSM composition" |
| L-2 | Low      | TRUE | Fix docstring to say "TaskSubmissionAbciApp" |

**Result: All 7 findings confirmed as TRUE.**
