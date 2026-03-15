# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **Autonolas AI Mechs** repository for the Predict Agent Economy. It contains autonomous agents (mechs) that execute AI tasks, particularly predictions, using the Open Autonomy framework. The mechs operate as decentralized services that listen for requests on-chain, execute AI tools, and deliver results back on-chain.

**Important:** The [mech-quickstart](https://github.com/valory-xyz/mech-quickstart) repo is now the recommended way to run and extend mechs. This repository is the production mech with all available tools.

## Architecture

### Core Components

**Agent Service Architecture:**
- Built on Open Autonomy (v0.21.13) framework which uses ABCI (Application Blockchain Interface)
- Agents run as autonomous services with consensus via Tendermint
- Services are composed of multiple **skills** that define behaviors and state machines
- Skills communicate through FSM (Finite State Machine) rounds

**Key Skills (fetched via `autonomy packages sync`, not checked in):**
- `mech_abci`: Main FSM orchestration for the mech service
- `task_execution`: Handles receiving, executing, and delivering AI tool tasks
- `task_submission_abci`: Manages task submission rounds
- `transaction_settlement_abci`: Handles on-chain transaction delivery
- `contract_subscription`: Listens for on-chain events (Request events from marketplace)
- `registration_abci`, `reset_pause_abci`, `termination_abci`: Service lifecycle management

**Mech Tools (Custom Packages):**
- Tools live in `packages/{author}/customs/{tool_name}/`
- Each tool implements a `run()` function that takes a prompt and API keys, returns results
- Tools are registered via `TOOLS_TO_PACKAGE_HASH` environment variable mapping tool names to IPFS hashes
- Common tool pattern: use `@with_key_rotation` decorator for automatic API key rotation on rate limits

**Operating Modes:**
- **Polling mode**: Periodically reads blockchain for Request events (default: every 30s)
- **Websocket mode**: Subscribes to events via websocket for real-time notifications
- Mode controlled by `USE_POLLING` environment variable

### Agent vs Service

- **Agent** (`packages/valory/agents/mech_predict/`): Single autonomous agent instance
- **Service** (`packages/valory/services/mech_predict/`): Multi-agent service with consensus (typically 4 agents)
- Services require Tendermint for consensus, agents run with embedded Tendermint

### Tool Architecture

Tools follow a standard structure:
```
packages/{author}/customs/{tool_name}/
├── __init__.py
├── {tool_name}.py       # Main implementation with run() function
└── component.yaml       # Tool metadata (description, callable, dependencies)
```

The `run()` function signature:
```python
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    # Returns: (result, prompt_used, transaction_data, additional_info)
```

## Common Commands

### Setup and Installation

```bash
# Install dependencies with Poetry
poetry install && poetry shell

# Sync packages from Open Autonomy registry
autonomy packages sync --update-packages
```

### Development

```bash
# Format code
make format
# Or: tomte format-code

# Run all code checks (black, isort, flake8, mypy, pylint, darglint)
make code-checks
# Or: tomte check-code

# Security checks (safety, bandit, gitleaks)
make security
# Or: tomte check-security

# Clean build artifacts
make clean
```

### Testing

```bash
# Run all tests
tox -e check-tools
# Or: pytest tests

# Test a specific tool
python scripts/test_tool.py

# Test multiple tools
python scripts/test_tools.py
```

### Package Management

```bash
# Update package hashes after changes
autonomy packages lock

# Check package hashes are correct
tox -e check-hash

# Check package dependencies
tox -e check-packages
# Or: python scripts/check_dependencies.py

# Generate/update IPFS hashes in documentation
tox -e fix-doc-hashes
```

### ABCI/FSM Development

```bash
# Generate ABCI docstrings
tox -e abci-docstrings

# Check ABCI docstrings
tox -e check-abci-docstrings

# Update FSM specifications
make fix-abci-app-specs
# Or: autonomy analyse fsm-specs --update --app-class MechAbciApp --package packages/valory/skills/mech_abci

# Check FSM specifications
tox -e check-abciapp-specs

# Check handler implementations
tox -e check-handlers
```

### Running the Mech

**As a standalone agent:**
```bash
# Setup keys
autonomy generate-key ethereum

# Configure environment
cp .example.env .1env
# Edit .1env with your API keys
source .1env

# Run agent (starts Tendermint automatically)
bash run_agent.sh
```

**As a service:**
```bash
# Generate keys for service
autonomy generate-key ethereum -n 1

# Configure environment
cp .example.env .1env
# Edit .1env with ALL_PARTICIPANTS addresses from keys.json
source .1env

# Run service (builds Docker images)
bash run_service.sh
```

### Before Creating a PR

Run checks in this order:
```bash
make clean
make format         # or: tomte format-code
make code-checks    # or: tomte check-code
make security       # or: tomte check-security

# If you modified AbciApp definitions:
make abci-docstrings

# If you modified packages/:
make generators
make common-checks-1

# Otherwise:
tomte format-copyright --author valory [with exclusions from Makefile]

# After committing:
make common-checks-2
```

## Key Environment Variables

Configure in `.1env` or `.agentenv`:

- `API_KEYS`: JSON dict mapping service names to API key lists (e.g., `{"openai": ["key1", "key2"], "google_api_key": ["key"]}`)
- `TOOLS_TO_PACKAGE_HASH`: Maps tool names to IPFS package hashes
- `TOOLS_TO_PRICING`: Dynamic pricing configuration per tool
- `MECH_TO_CONFIG`: Mech-specific configuration (dynamic pricing, marketplace settings)
- `MECH_MARKETPLACE_ADDRESS`: Smart contract address for the marketplace
- `SERVICE_REGISTRY_ADDRESS`: Service registry contract
- `COMPLEMENTARY_SERVICE_METADATA_ADDRESS`: Metadata contract for mech registration
- `USE_POLLING`: Boolean to enable polling mode (vs websocket)
- `POLLING_INTERVAL`: Seconds between polls (default: 30.0)
- `TASK_DEADLINE`: Max seconds to execute a task (default: 240.0)
- `DEFAULT_CHAIN_ID`: Default blockchain (e.g., "gnosis")
- `{CHAIN}_LEDGER_RPC_{N}`: RPC endpoints per agent per chain

## Important Patterns

### Adding a New Tool

1. Create package structure in `packages/{author}/customs/{tool_name}/`
2. Implement `run(**kwargs)` function with proper signature
3. Use `@with_key_rotation` decorator if calling rate-limited APIs
4. Add tool metadata to `component.yaml`
5. Add IPFS hash to `TOOLS_TO_PACKAGE_HASH` mapping
6. Update package hashes: `autonomy packages lock`
7. Test with `scripts/test_tool.py`

### API Key Management

The `KeyChain` class (in `packages/valory/skills/task_execution/utils/apis.py`) handles:
- Multiple API keys per service
- Automatic rotation on rate limit errors
- Round-robin distribution across agents

### FSM Development

- FSM apps define state transitions through Round classes
- Each Round has an `end_block()` method determining next state
- Consensus is reached when threshold of agents agree (typically 2/3)
- See: [Open Autonomy FSM documentation](https://docs.autonolas.network/open-autonomy/key_concepts/fsm_app_introduction/)

### Package Versioning

- Packages use semantic versioning
- IPFS hashes ensure immutable package references
- Use `scripts/bump.py` for version bumps
- Package fingerprints track file changes

## Testing Notes

- Tests use pytest with asyncio mode
- Integration tests marked with `@pytest.mark.integration`
- E2E tests marked with `@pytest.mark.e2e`
- Mock API responses to avoid rate limits during testing
- Use `KeyChain` class for API key management in tests

## Dependencies

- Python >=3.10, <3.15
- Poetry for dependency management
- Docker and Docker Compose for service deployment
- Tendermint 0.34.19 for consensus
- Open Autonomy framework (all packages in `packages/` directory)

## Mints Directory

Contains NFT metadata (JSON and PNG files) for mech tokens - not relevant for development.
