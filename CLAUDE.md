# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonolas AI Mechs — a decentralized task execution service built on the Open Autonomy (AEA) framework. Agents receive AI task requests on-chain, execute them, reach consensus via Tendermint BFT, and deliver results back on-chain through Gnosis Safe multi-sig transactions. Runs on Gnosis chain (and Polygon, Base, Optimism).

## Build & Development Commands

```bash
# Install dependencies
poetry install && poetry shell

# Sync packages from remote IPFS registry
autonomy packages sync --update-packages

# Run tests (requires autonomy init + packages sync first)
pytest packages/valory/skills/mech_abci/tests packages/valory/skills/task_submission_abci/tests packages/valory/skills/task_execution/tests

# Run a single test file
pytest packages/valory/skills/mech_abci/tests/test_rounds.py -rfE

# Format code
make format          # or: tomte format-code

# Lint & type checks
make code-checks     # or: tomte check-code

# Security checks
make security

# All CI checks
make all-checks

# Regenerate FSM specs, docstrings, hashes
make generators

# Fix ABCI app specs only
make fix-abci-app-specs
```

Individual tox environments: `tox -e black-check`, `tox -e isort-check`, `tox -e flake8`, `tox -e mypy`, `tox -e pylint`, `tox -e darglint`.

## Architecture

### Chained FSM Composition

The service uses a chained Finite State Machine defined in `packages/valory/skills/mech_abci/composition.py`:

```
Registration → DeliveryRateUpdate → TaskSubmission → TransactionSettlement → ResetAndPause → (loop back)
```

Background skills run concurrently outside the FSM consensus loop:
- **contract_subscription**: Monitors on-chain events via WebSocket (eth_subscribe) for new task requests
- **task_execution**: Executes tasks from a pending queue using a thread pool, stores results in shared state

### Key Skills (under `packages/valory/skills/`)

| Skill | Role |
|---|---|
| `mech_abci` | Top-level composition of all ABCI apps into `MechAbciApp` |
| `task_submission_abci` | `TaskPoolingRound` collects done tasks, `TransactionPreparationRound` prepares delivery TX |
| `task_execution` | Background task executor with thread pool, deadline tracking, payment model support |
| `delivery_rate_abci` | Periodically updates delivery rate metrics on-chain |
| `contract_subscription` | WebSocket subscription to mech contract events |
| `websocket_client` | Base WebSocket connection management |

### Data Flow

1. `contract_subscription` detects on-chain request events → adds to `PENDING_TASKS`
2. `task_execution` (background) picks up tasks, executes them, writes to `DONE_TASKS` (protected by `DONE_TASKS_LOCK`)
3. `TaskPoolingRound` (FSM) collects done tasks from all agents, deduplicates by `request_id`
4. `TransactionPreparationRound` prepares multi-sig delivery transaction
5. `TransactionSettlementAbciApp` finalises on-chain delivery via Gnosis Safe

### Contracts (`packages/valory/contracts/`)

- **olas_mech**: Core mech contract — Request/Deliver events, delivery rate
- **balance_tracker**: Agent wallet balance monitoring
- **hash_checkpoint**: State validation
- **complementary_service_metadata**: Service metadata

### Agent & Service

- Agent definition: `packages/valory/agents/mech/aea-config.yaml`
- Service definition: `packages/valory/services/mech/service.yaml` (4 agents by default)

## Code Style

- Black formatting, isort with `profile=black`, line length 88
- Type checking: mypy with `--disallow-untyped-defs`
- Docstrings: sphinx style (checked by darglint)
- Linting scope is defined by `SERVICE_SPECIFIC_PACKAGES` in `tox.ini` — only service-specific packages are checked, not third-party ones under `packages/valory/`

## Important Conventions

- After modifying skill code, run `make fix-abci-app-specs` to update FSM specs
- After modifying packages, run `autonomy packages lock` to update hashes
- Package hashes in `packages/packages.json` must stay in sync — CI checks this via `tox -e check-hash`
- The `packages/` directory is an Open Autonomy local registry; third-party packages are synced from IPFS
