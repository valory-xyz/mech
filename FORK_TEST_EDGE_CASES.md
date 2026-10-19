# Fork-test edge cases — dynamic (cap-and-actual) pricing

Running list of things to work through after the initial fork-test proves the
new 5-arg `deliverMarketplaceWithSignatures` overload (autonolas-marketplace
PR #204) is callable end-to-end.

Nothing here is a blocker for the fork test. Each item is a follow-up change,
mostly in this repo, a few in sister repos.

## Pricing

- `delivery_rates = [max_rate - 1 for max_rate in max_rates]` in
  `_get_offchain_tasks_deliver_data` is a fork-test value only. It charges
  every requester 1 wei under their signed cap. Replace with the real
  per-tool cost — the number already lives in `TokenCounterCallback` /
  `get_cost_for_done_task` on the task-execution side, just not threaded
  down to the settlement batch today.
- Decide the thread. Simplest shape: carry the per-request actual cost on
  the done-task dict next to `delivery_rate` and read it out in
  `_get_offchain_tasks_deliver_data`.
- Floor + markup policy: operator-configurable, probably `models.py` /
  `skill.yaml`. Open question whether a floor is per-tool, per-mech, or
  flat.
- Guard against `actual > max` client-side as well. The contract reverts
  with `RateOverCap`, which is correct, but the mech should never submit
  that batch in the first place — add a `clip to cap` or `drop + log`
  policy.

## Client-side

- mech-client, mech-interact, trader/optimus do not change for the first
  test: the signed hash shape is identical, the requester still signs over
  what we now call `maxRate`. Confirm by driving one real signed request
  through the fork.
- Once the mech starts charging less than the cap, downstream consumers
  that read the `Deliver` event's `deliveryRate` field will see per-request
  variation instead of a flat rate. Subgraph consumers and
  mech-analytics / predict-api event schemas should be sanity-checked.

## Nonces (GS026)

- The 6-arg overload (autonolas-marketplace PR #205) decouples pre-signed
  requests from the shared sequential `mapNonces` counter. Fixes the
  GS026 nonce race we had flagged as not-fixable mech-side.
- Needs client coordination: mech-client/mech-interact must pick the
  nonce per request (and sign over it) instead of the contract reading it.
- Scope separately from this issue; land after the 5-arg path is stable
  on the fork.

## Mech plumbing

- `autonomy packages lock` + `autonomy push-all` once the shape is final.
  CI `check-hash` fails otherwise. See
  `~/.claude/rules/autonomy-packages-lock-and-push.md`.
- ABI refresh — DONE locally. `packages/valory/contracts/olas_mech/build/AgentMech.json`
  was swapped with the `MechFixedPriceNative` artifact compiled from
  autonolas-marketplace `origin/main` (commit a939810), picking up all three
  `deliverMarketplaceWithSignatures` overloads (4-arg legacy, 5-arg cap,
  6-arg cap+nonces). Original kept at `.bak` for rollback. This swap is
  temporary — the shipped flow is to refresh the ABI from the artifact the
  autonolas-marketplace team publishes when the marketplace redeploys.
- The on-chain marketplace path in `_get_marketplace_tasks_deliver_data`
  (same file) is untouched. If/when on-chain requests also move to
  cap-and-actual, that path needs the same treatment.

## FSM / retry / replay

- Settlement retry currently rebuilds the batch; confirm it rebuilds with
  the same `maxRates` the signatures covered, not a recomputed value.
- No code today reads the preimage buffer back into the deliver batch
  after a restart. Out of scope here — tracked in the separate kv-store
  drainer note (ref: `reference_kv_store_durability_note.md`).

## Testing surface

- Fork setup: Anvil fork of Gnosis, deploy the updated `MechMarketplace`
  + `OlasMech` from autonolas-marketplace HEAD, point one dev-mech agent
  at the fork RPC.
- First assertion: Deliver event emits with the actual rate (one wei
  below the cap), BalanceTracker debit equals the actual.
- Second assertion: force a `delivery_rate > max_rate` payload (temporary
  branch) and confirm the contract reverts with `RateOverCap`.
- Unit tests for `get_offchain_deliver_data_with_cap` + the behaviour's
  max_rates array construction before merging anywhere beyond the fork.
