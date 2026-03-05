# FSM Audit Report

**Scope:** `packages/valory/skills/task_submission_abci`, `packages/valory/skills/delivery_rate_abci`, `packages/valory/skills/mech_abci`, `packages/valory/skills/task_execution`, `packages/valory/skills/contract_subscription`, `packages/valory/skills/websocket_client`, `packages/valory/connections/websocket_client`

**Date:** 2026-03-05

## CLI Tool Results

```
$ autonomy analyse fsm-specs --package packages/valory/skills/mech_abci
Check successful

$ autonomy analyse fsm-specs --package packages/valory/skills/task_submission_abci
Check successful

$ autonomy analyse fsm-specs --package packages/valory/skills/delivery_rate_abci
Check successful

$ autonomy analyse handlers -h abci -h http -h contract_api -h ledger_api -h signing \
    -i abstract_abci -i contract_subscription -i task_execution -i websocket_client
Check successful

$ autonomy analyse docstrings
No update needed.

$ autonomy analyse dialogues
Error: Common dialogue 'abci_dialogues' is not defined in
  packages/valory/skills/contract_subscription/skill.yaml
```

The dialogues error is expected: `contract_subscription` is a background event-listener skill that does not participate in ABCI consensus, so it correctly omits `abci_dialogues`.

## Critical Findings

### C-1: None appended to transaction list in delivery rate behaviour

- **File:** `packages/valory/skills/delivery_rate_abci/behaviours.py:183`
- **Issue:** When `_get_delivery_rate_update_tx()` returns `None`, the code logs a warning but still appends `None` to `txs`. This list is passed to `_to_multisend()` which accesses `transaction["to"]` at line 209, causing `TypeError: 'NoneType' object is not subscriptable`.
- **Code:**
  ```python
  if tx is None:
      self.context.logger.warning(
          f"Could not get delivery_rate update tx for {mech_address}."
      )

  txs.append(tx)  # BUG: appends None
  ```
- **Fix:**
  ```python
  if tx is None:
      self.context.logger.warning(
          f"Could not get delivery_rate update tx for {mech_address}."
      )
      continue

  txs.append(tx)
  ```

## High Findings

### H-1: ThreadPoolExecutor never shut down on disconnect

- **File:** `packages/valory/connections/websocket_client/connection.py:306,318-327`
- **Issue:** `connect()` creates a `ThreadPoolExecutor` at line 306, but `disconnect()` never calls `shutdown()` on it. Threads persist after disconnection, causing resource leaks.
- **Code:**
  ```python
  async def disconnect(self) -> None:
      await self._manager.remove_all_subscriptions()
      self._outbox.empty()
      self.state = ConnectionStates.disconnected
      # Missing: self._executor.shutdown(wait=True)
  ```
- **Fix:** Add `self._executor.shutdown(wait=True)` before setting state to disconnected.

### H-2: ProcessPool.join() without timeout

- **File:** `packages/valory/skills/task_execution/behaviours.py:830-831`
- **Issue:** `self._executor.join()` is called without a timeout. If a child process is stuck, this blocks the agent indefinitely.
- **Code:**
  ```python
  def _restart_executor(self) -> None:
      self._executor.stop()
      self._executor.join()  # can block forever
  ```
- **Fix:**
  ```python
  self._executor.join(timeout=10.0)
  ```

## Medium Findings

### M-1: Offchain task filtering uses unsynchronized shared state

- **File:** `packages/valory/skills/task_submission_abci/behaviours.py:1597`
- **Issue:** `_get_offchain_tasks_deliver_data()` reads tasks from `self.done_tasks` (local shared state, not consensus-synchronized), while sibling filters at lines 1376 and 1389 use `self.synchronized_data.done_tasks` (consensus data). This inconsistency means offchain tasks may differ across agents.
- **Code:**
  ```python
  # Line 1597 - local shared state (NOT synchronized)
  done_tasks_list = self.done_tasks

  # Lines 1376, 1389 - consensus synchronized data
  for done_task in self.synchronized_data.done_tasks
  ```
- **Fix:** Use `self.synchronized_data.done_tasks` consistently, or document the intentional divergence if offchain tasks are expected to be agent-local.

### M-2: `final_tx_hash` in db_post_conditions but never set by this FSM

- **File:** `packages/valory/skills/task_submission_abci/rounds.py:255`
- **Issue:** `FinishedTaskPoolingRound` post-conditions declare `{"most_voted_tx_hash", "final_tx_hash"}`, but `final_tx_hash` is never written within `TaskSubmissionAbciApp` -- it is set by the downstream `TransactionSettlementAbciApp`. This post-condition cannot be satisfied by this FSM alone.
- **Code:**
  ```python
  db_post_conditions: Dict[AppState, Set[str]] = {
      FinishedTaskPoolingRound: {"most_voted_tx_hash", "final_tx_hash"},
  ```
- **Fix:** Remove `"final_tx_hash"` (it is already in `cross_period_persisted_keys` at line 248):
  ```python
  FinishedTaskPoolingRound: {"most_voted_tx_hash"},
  ```

## Low Findings

### L-1: Docstring drift in composition.py

- **File:** `packages/valory/skills/mech_abci/composition.py:20`
- **Issue:** Docstring says `"round behaviours"` but the file defines FSM composition.
- **Fix:** `"""This package contains the FSM composition for MechAbciApp."""`

### L-2: Docstring drift in task_submission_abci

- **File:** `packages/valory/skills/task_submission_abci/behaviours.py:20`
- **Issue:** Docstring says `"TaskExecutionAbciApp"` but this is the `task_submission_abci` skill.
- **Fix:** `"""This package contains round behaviours of TaskSubmissionAbciApp."""`

- **File:** `packages/valory/skills/task_submission_abci/models.py:20`
- **Issue:** Same naming mismatch -- references `TaskExecutionAbciApp` instead of `TaskSubmissionAbciApp`.

## Test Findings

No test-specific findings (T1-T6). The test files use correct base classes, payload registries are properly managed, and no `@classmethod @pytest.fixture` anti-patterns were found.

## Summary

| Severity | Count |
|----------|-------|
| Critical | 1     |
| High     | 2     |
| Medium   | 2     |
| Low      | 2     |

## Notes

- The `done_tasks_lock()` call at `task_submission_abci/behaviours.py:196` was investigated and confirmed correct -- it is a regular method (not a property) returning a `threading.Lock`, which is a valid context manager.
- `ROUND_TIMEOUT` event enum members were not flagged per false-positive guidance for library skills.
- The termination background app config at `composition.py:53-57` is correctly configured as TERMINATING type (has `start_event`, no `end_event`, has `abci_app`).
- The composition chain at `composition.py:40-51` covers all 10 final states with proper loop-back to valid initial states.
