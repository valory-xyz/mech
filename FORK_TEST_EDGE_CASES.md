# Fork-test edge cases — dynamic (cap-and-actual) pricing

Running list of things to work through after the initial fork-test proves the
new 5-arg `deliverMarketplaceWithSignatures` overload (autonolas-marketplace
PR #204) is callable end-to-end.

Nothing here is a blocker for the fork test. Each item is a follow-up change,
mostly in this repo, a few in sister repos.

## Pricing

- Resolved: on the flag-on path `deliveryRates[i]` is the measured tool cost
  (`cost_dict.total_cost`, USD cents) scaled by `payment_token_decimals`,
  marked up by `operator_markup_bp`, floored by the tool's `tools_to_pricing`
  entry (not marked up), and clamped to the signed cap with a warning. See
  `TransactionPreparationBehaviour._compute_actual_delivery_rate`.
- A tool that reports `total_cost == 0` and has no `tools_to_pricing` floor
  is delivered for free. Operators who want a minimum must set the floor.
- A tool that reports no cost and has no floor is charged the full cap, with
  a warning. Worth a Prometheus counter once the metrics follow-up lands.
- `payment_token_decimals` assumes the payment token trades at 1 USD. A
  native mech on a non-stable token (POL) needs a price source before the
  flag is flipped there.

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

- `autonomy packages lock` + `autonomy push-all` after any package change
  so CI `check-hash` stays green.
- ABI refresh — the shipped `AgentMech.json` carries the three overloads
  (4-arg legacy, 5-arg cap, 6-arg cap + nonces) from the compiled
  `MechFixedPriceNative` artifact; refresh again when the marketplace
  redeploys and publishes new artifacts.
- The on-chain marketplace path in `_get_marketplace_tasks_deliver_data`
  (same file) is untouched. If/when on-chain requests also move to
  cap-and-actual, that path needs the same treatment.

## FSM / retry / replay

- Settlement retry currently rebuilds the batch; confirm it rebuilds with
  the same `maxRates` the signatures covered, not a recomputed value.
- No code today reads the preimage buffer back into the deliver batch
  after a restart. Out of scope here — tracked separately as the kv-store
  drainer follow-up.

## Testing surface

- Fork setup: Hardhat deploy scripts (in `autonolas-marketplace`) run
  against a local Anvil node forking Polygon mainnet, point one dev-mech
  agent at the fork RPC. Details and gotchas: see the comment thread on
  valory-xyz/mech#501.
- First assertion: Deliver event emits with the actual rate (one wei
  below the cap), BalanceTracker debit equals the actual.
- Second assertion: force a `delivery_rate > max_rate` payload (temporary
  branch) and confirm the contract reverts with `RateOverCap`.
- Unit tests for `get_offchain_deliver_data_with_cap` + the behaviour's
  max_rates array construction before merging anywhere beyond the fork.
