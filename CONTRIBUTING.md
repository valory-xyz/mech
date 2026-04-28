# Contributing

For the contribution workflow — pre-commit routine, PR checklist, coding-style rules, `tomte format-code` → `tomte check-code` order — see the canonical AEA-agent stack guide:

→ **<https://github.com/valory-xyz/open-autonomy/blob/main/CONTRIBUTING.md>**

## Notes for `mech`

- Tests require `autonomy packages sync --update-packages` against IPFS first; see [`CLAUDE.md`](CLAUDE.md) for the full development sequence.
- After modifying skill code, run `make fix-abci-app-specs` to update FSM specs.
- After modifying packages, run `autonomy packages lock` to update package fingerprints (`packages/packages.json`).
- Linting scope is defined by `SERVICE_SPECIFIC_PACKAGES` in `tox.ini` — only mech's service-specific packages are checked, not third-party ones synced under `packages/valory/`.
