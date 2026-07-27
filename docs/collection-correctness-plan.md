# Collection Diff Correctness and Authoritative Replay

## Status

Proposed implementation plan for issue 4: fix known collection-state divergence before refactoring collection transport.

Canonical landing order:

1. [`world-local-cleanup-plan.md`](world-local-cleanup-plan.md)
2. This collection correctness plan.
3. [`registration-lease-plan.md`](registration-lease-plan.md)
4. [`move-first-signal-delivery-plan.md`](move-first-signal-delivery-plan.md)

This semantic baseline lands before registration leases rewrite replay and dynamic processor ownership.

## Problem statement

Several collection combinators update internal state correctly while emitting incomplete or incorrect diffs to downstream consumers:

1. `sort_by` and `sort_by_key` emit an insertion index one position too early when an updated item moves toward a higher sorted index.
2. `map_value_signal` suppresses `Replace { entries: [] }`, so downstream maps retain stale entries.
3. Vector and map replay suppress empty snapshots, so switching from a populated source to an empty source leaves the old collection visible.
4. Closely related: source `Move` operations can change stable ordering among equal sort keys while sort combinators emit no downstream diff.

These are semantic bugs, not transport or lifecycle bugs. They should be fixed with focused production changes and oracle-based tests before move-first values alter the same code paths.

## Decision summary

1. Treat replay as an authoritative snapshot operation.
2. A replay snapshot emits `Replace` even when the collection is empty and advances a per-subscriber revision cursor so already-snapshotted pending diffs are not replayed again.
3. `map_value_signal` always forwards upstream `Replace`, including empty replacements.
4. Sort update insertion positions are emitted exactly as computed after removing the old sorted entry.
5. Stable equal-key ordering follows source order; source `Move` emits a downstream `Move` when that stable order changes.
6. Tests reconstruct downstream collections solely from emitted diffs and compare them with plain collection oracles after every operation.
7. No graph transport, registration ownership, scheduling, or handle changes are included.

## Goals

- Make emitted diffs sufficient to reconstruct exact downstream state.
- Cover transitions through empty collections explicitly.
- Preserve stable sorting semantics for duplicate keys.
- Add deterministic regressions for every known bug.
- Add model-based tests that detect future diff/state divergence.
- Establish semantic guardrails for later replay and move-first refactors.
- Keep production changes focused; replay versioning is an internal source/replay protocol change, not a graph transport redesign.

## Non-goals

This plan does not:

- Change `AnyClone` or erased graph transport.
- Add output delivery policies or `.broadcast()`.
- Change signal-description cloneability.
- Redesign `SignalHandle` ownership.
- Remove temporary polling edges.
- Refactor dynamic subscription lifecycle.
- Add schedule-aware replay.
- Audit every collection combinator.
- Add a public `MapDiff::apply_to_btree_map` API.
- Change malformed public diff handling.
- Modify unrelated duplicate map-insert work already present in the working tree.

## Semantic invariants

### General diff invariant

After every delivered batch:

> Applying all emitted diffs in order to the previous downstream state yields exactly the combinator's current logical state.

### Replay invariant

A replay request communicates a complete current snapshot:

- Non-empty vector → `Replace { values }`.
- Empty vector → `Replace { values: vec![] }`.
- Non-empty map → `Replace { entries }`.
- Empty map → `Replace { entries: vec![] }`.

Silence is not an authoritative empty snapshot because it cannot distinguish “the selected source is empty” from “retain the previous selected source.”

### Sort invariant

For `sort_by` and `sort_by_key`:

- Internal `sorted_indices` and downstream materialized order always agree.
- Update movement toward either lower or higher positions is represented correctly.
- Equal comparator/key values are ordered by current source position.
- Moving a source item within an equal-key group updates stable downstream order.

### Dynamic map invariant

For `map_value_signal`:

- Every successfully processed upstream `Replace` produces exactly one downstream `Replace` after processor reconstruction, including an explicit empty source replacement.
- Empty output is still meaningful output.
- After a replacement commits, previously mapped inner processors produce no downstream output.
- Exact handle disconnection, registration balancing, and replacement ordering belong to `registration-lease-plan.md`.

### Switch invariant

For currently supported mutable-backed switch chains with one replay root, after every source mutation or selector change, downstream materialized state equals the selected source's current complete state.

Multi-root replay graphs and sources explicitly documented as unsupported by switch combinators remain out of scope.

## Root-cause analysis

### `sort_by` forward updates

Current order:

1. Find the old sorted position.
2. Remove that entry from `sorted_indices`.
3. Update the source value.
4. Search the now-shorter sorted list.
5. Receive `new_pos`, already expressed in post-removal coordinates.
6. Reinsert internal tracking at `new_pos`.
7. Emit `RemoveAt(old_pos)`.
8. Incorrectly subtract one when `new_pos > old_pos` before emitting `InsertAt`.

Example:

```text
[1, 2, 3]
update 1 -> 4
```

After removing `1`, search correctly returns `new_pos = 2` in `[2, 3]`. Emitting `InsertAt(1, 4)` creates `[2, 4, 3]`, while internal state believes it is `[2, 3, 4]`.

Affected paths:

- `SignalVecExt::sort_by`
- `SignalVecExt::sort_by_key`

### `map_value_signal` empty replace

The combinator:

1. Cleans old processors.
2. Builds new processors and mapped entries.
3. Replaces internal manager state.
4. Emits `MapDiff::Replace` only when mapped entries are non-empty.

If the source replaces with an empty map, internal state becomes empty but downstream receives no mutation and retains stale entries.

This issue freezes behavior only for an explicit upstream empty `Replace`. Partial replacement semantics when factory systems fail require a separate error-contract decision and are not expanded here.

### Empty replay suppression

Mutable vec/map replay currently returns `None` when the current snapshot is empty. Switch combinators rely on replay to establish the newly selected source state. Therefore:

```text
selected A = non-empty
switch to B = empty
replay B = no output
downstream remains A
```

The source of truth should emit an empty `Replace`; switch managers should not need collection-specific reset logic.

### Stable equal-key source moves

Sort implementations tie equal comparator/key values by source index. A source `Move` changes those indices and therefore can change required stable output order.

Current code remaps source indices inside `sorted_indices` but emits no diff and does not reposition the moved item according to the updated tie-break. Internal binary-search ordering can then become inconsistent.

Example:

```text
[(key=1, id=A), (key=1, id=B)]
move source index 0 -> 1
```

Stable output should change from `[A, B]` to `[B, A]`.

## Production fixes

## 1. Correct sort update insertion indices

For both `sort_by` and `sort_by_key`:

- Keep removing the old sorted entry before search.
- Keep computing `new_pos` against the shortened list.
- Reinsert internal tracking at `new_pos`.
- Emit `InsertAt { index: new_pos, value }` directly.
- Remove the extra `new_pos - 1` adjustment and its comment.

No data-structure redesign is required.

## 2. Repair stable sorting after source `Move`

For both sort implementations:

1. Record the moved item's old sorted position before remapping.
2. Apply the source vec/key move.
3. Remap all tracked source indices as currently required.
4. Remove `sorted_indices[old_sorted_pos]` directly after remapping; do not binary-search the temporarily invalid ordering to rediscover the moved item.
5. Search its correct new sorted position in the shortened list using the moved item's new source index and updated tie-break.
6. Reinsert it.
7. Emit `VecDiff::Move { old_index: old_sorted_pos, new_index: new_sorted_pos }` only when the stable sorted position changed.

`new_sorted_pos` must use the same coordinate convention as `VecDiff::Move::apply_to_vec`: remove from old position, then insert at the supplied new position.

Add targeted tests before settling the exact calculation for adjacent forward/backward moves.

## 3. Always forward `map_value_signal` replacement

After rebuilding new processors and mapped entries, unconditionally append:

```rust
MapDiff::Replace {
    entries: new_entries_for_diff,
}
```

Do not guard with `is_empty()`.

Change only committed output semantics. Preserve whichever processor-ownership protocol is installed when this fix lands; the later registration lease plan replaces ownership mechanics without changing the empty-replacement tests.

## 4. Make replay snapshots authoritative with revisions

Current-state inspection alone cannot distinguish diffs already represented by a snapshot from later diffs. This affects initially empty and initially non-empty sources, mutations before replay registration, and several subscribers around one pending batch.

Add a monotonically increasing source revision and version every pending diff:

```rust
struct VersionedVecDiff<T> {
    revision: u64,
    diff: VecDiff<T>,
}

struct VersionedMapDiff<K, V> {
    revision: u64,
    diff: MapDiff<K, V>,
}
```

Mutable source data stores:

```rust
revision: u64,
pending_diffs: Vec<Versioned...>,
```

Every committed mutation increments the revision with checked overflow handling and records its resulting diff at that revision.

The internal broadcaster emits versioned batches. Each replay node stores its own cursor:

```rust
last_applied_revision: u64
```

On replay request:

1. Read an atomic snapshot of current collection state and current source revision `R` from the same world access.
2. Emit `Replace` for that snapshot, including empty.
3. Set the replay node cursor to `R`.
4. For later incoming broadcaster batches, discard entries with `revision <= last_applied_revision` and forward only the ordered suffix with newer revisions.
5. Advance the cursor to every forwarded revision.

For ordinary incoming diffs before any explicit replay, the node's initial replay still establishes a snapshot/cursor before incremental forwarding. Existing subscribers with older cursors receive pending diffs; newly registered or manually replayed subscribers skip only diffs included in their snapshot.

This supports multiple replay subscribers without globally clearing pending diffs and prevents historical positional diffs from being applied to the wrong baseline.

Define revision overflow as a checked invariant failure rather than wrapping into an ambiguous cursor ordering.

### Why fix replay rather than switch managers

Do not add unconditional `Clear` or materialized-state tracking to switch managers:

- Non-empty → non-empty should remain one authoritative replacement rather than `Clear + Replace`.
- Switch should select a source, not interpret collection state.
- Replay already claims responsibility for initial/current snapshots.
- One source-level fix covers supported `switch_signal_vec` and `switch_signal_map` mutable-backed single-replay-root paths consistently.

## Source touchpoints

### `src/signal_vec.rs`

- `sort_by` update handling.
- `sort_by_key` update handling.
- Both sort implementations' source `Move` handling.
- Add source revision and versioned pending vec diffs.
- Change the internal broadcaster to emit versioned batches.
- Add per-replay cursor and authoritative snapshot behavior.
- Sort and replay tests.
- Existing `VecDiff::apply_to_vec` test helper usage.

### `src/signal_map.rs`

- `map_value_signal` replace handling.
- Add source revision and versioned pending map diffs.
- Change the internal broadcaster to emit versioned batches.
- Add per-replay cursor and authoritative snapshot behavior.
- Map diff application test helper.
- Dynamic map regression tests.

### `src/signal.rs`

- Switch vec/map tests and expected initial-empty behavior.
- Production switch managers should remain unchanged unless tests prove replay cannot express the required semantics.

## Implementation phases

### Phase 0: Add deterministic failing regressions

Add tests for:

- `sort_by` update moving first to last.
- `sort_by_key` update moving first to last.
- Adjacent forward movement.
- Empty `map_value_signal` replacement.
- Non-empty → empty vector switch.
- Non-empty → empty map switch.
- Stable equal-key source move.
- Non-empty source mutated after replay registration but before first replay.
- Source mutated before replay registration.
- Two replay subscribers on opposite sides of one pending batch.

Acceptance: every new test fails for the expected materialized-state mismatch.

### Phase 1: Apply minimal semantic fixes

- Remove incorrect sort insertion adjustment.
- Reposition moved equal-key items and emit downstream move.
- Always forward map replacements.
- Add versioned pending diffs and per-replay cursors.
- Emit authoritative snapshots, including empty, and filter already-snapshotted revisions.

Acceptance:

- Exact regression tests pass.
- No graph transport or registration ownership code changes.

### Phase 2: Expand transition coverage

Add deterministic cases for:

- Forward/backward/adjacent/final sort movement.
- Duplicate keys and stable ordering.
- Empty ↔ non-empty replacements.
- Source cleared while inactive then selected.
- Empty → empty switch.
- Same target identity.
- Mutation before first update across all registration-time states.
- Source mutated before replay registration.
- Non-empty source replaced/cleared before first replay.
- Historical positional diffs generated from a non-empty state before a new subscriber.
- Two subscribers registering around one pending batch.
- Batched diffs in one frame.
- Old mapped inner signal produces no output after empty replace.

### Phase 3: Add model-based oracle tests

Use the existing `rand` dev dependency with fixed seeds before adding another property-test dependency.

For every generated operation:

1. Update a plain source model.
2. Apply the corresponding public mutable collection operation.
3. Run the app.
4. Apply emitted diffs to a downstream model.
5. Compute the expected combinator/selected-source state from the plain source model.
6. Assert exact equality and include seed/operation trace on failure.

### Phase 4: Establish handoff baseline

Run focused and full tests with normal parallelism supplied by world-local cleanup.

Mark these tests as required semantic regressions for all later replay, dynamic-combinator, registration-lease, and collection transport changes.

## Regression test details

## Sort tests

For both `sort_by` and `sort_by_key`:

1. Forward move: `[1, 2, 3]`, update `1 -> 4`.
2. Backward move: update highest to lowest.
3. Adjacent forward move.
4. Move to final sorted index.
5. Unchanged sorted position emits `UpdateAt`.
6. Update into an equal-key group.
7. Update within an equal-key group.
8. Update out of an equal-key group.
9. Source `Move` within an equal-key group in both directions.
10. Source move across several equal-key peers.
11. Source move across unrelated keys while crossing only some equal-key peers.
12. Unequal-key source move produces no downstream diff.
13. Remove/update/insert immediately after a source move to validate later binary searches.
14. Batched source move plus update before one app update.

Use tagged items:

```rust
struct TaggedItem {
    key: i32,
    id: u32,
}
```

Sort by `key`; `id` exposes stable-order mistakes.

Reconstruct downstream state only by applying emitted diffs.

## `map_value_signal` tests

1. Start populated and materialize output.
2. Replace source with empty map.
3. Assert exact `Replace { entries: vec![] }`.
4. Apply to downstream oracle and assert empty.
5. Mutate an old inner signal and assert no output.
6. Replace with non-empty entries again and verify recovery.
7. Keep separate `Clear` coverage.
8. Do not freeze partial factory-failure semantics in this issue; cover only explicit upstream empty replacement.

## Switch tests

For both vec and map:

1. Non-empty → initially empty target.
2. Non-empty → target emptied while inactive → switch back.
3. Empty → non-empty.
4. Empty → empty.
5. Same target identity produces no replay/reset; drive repeated outer emissions that return the same inner identity rather than relying on a deduped selector.
6. Inactive mutations remain silent until selection, then current snapshot is authoritative.
7. Empty and non-empty sources mutated after replay registration but before first replay do not double-apply pending diffs.
8. Sources mutated before replay registration establish a correct snapshot cursor.
9. Non-empty source replaced or cleared before first replay remains correct.
10. Historical positional diffs from a non-empty baseline are skipped by a new snapshot subscriber.
11. Two replay subscribers around one pending batch each receive exactly the state changes required by their own cursor.

Update current tests that expect no initial diff from an empty replay; authoritative replay should expect an empty `Replace` when no pending first-frame mutation must be deferred.

## Vector oracle

Maintain:

- Plain source `Vec<TaggedItem>`.
- Expected stable-sorted vector recomputed from source.
- Downstream vector reconstructed solely from emitted `VecDiff`s.

Generate valid operations, including no-ops:

- Replace, including empty to empty.
- InsertAt.
- UpdateAt.
- RemoveAt.
- Move, including `move_item(i, i)`.
- Push.
- Pop, including empty pop.
- Clear, including empty clear.

Use:

- Small exhaustive vectors of length 0–4 for boundaries.
- Fixed-seed longer operation streams for interactions.

After every operation or deliberate batch, assert downstream equals stable-sorted source.

## Map oracle

Maintain:

- Plain source `BTreeMap<K, V>`.
- Expected mapped `BTreeMap<K, O>`.
- Downstream map reconstructed from emitted `MapDiff`s.

Add a private test helper:

```rust
fn apply_map_diffs<K: Ord, V>(
    map: &mut BTreeMap<K, V>,
    diffs: impl IntoIterator<Item = MapDiff<K, V>>,
);
```

Generate:

- Insert new key.
- Update existing key.
- Remove present and missing keys.
- Clear, including empty clear.
- Replace empty, including empty to empty.
- Replace non-empty.

Compare after every update.

## Switch oracle

Maintain two or three plain collections and a selected-source ID. Reconstruct one downstream model from emitted diffs. After every source mutation or selector change, assert downstream equals the selected model collection.

Weight transitions through empty state heavily.

## Validation commands

World-local cleanup is a prerequisite, so run normally in parallel:

```text
cargo test --locked --lib test_sort_by
cargo test --locked --lib test_map_value_signal
cargo test --locked --lib test_switch_signal
cargo test --locked --lib
cargo test --locked
```

Also run supported feature checks. The known optional-time test compilation problem remains separate.

## Risks and mitigations

### Replay initial-state behavior changes

Risk: tests or downstream code rely on initially empty replay producing no diff.

Mitigation: document replay as authoritative snapshot semantics; update tests; distinguish replay requests from ordinary no-change frames.

### Replay cursor/version bugs

Risk: snapshots and pending batches overlap, revisions are filtered incorrectly, or revision overflow wraps ordering.

Mitigation: version every diff, capture snapshot state and revision together, use per-subscriber cursors, use checked revision increments, and test subscribers around historical/pending batches.

### Sort move coordinate errors

Risk: downstream `VecDiff::Move` coordinate convention differs from internal search coordinates.

Mitigation: add adjacent forward/backward and equal-key move tests before finalizing production calculation; always validate through `apply_to_vec`.

### Scope creep into dynamic ownership

Risk: changing `map_value_signal` cleanup while fixing empty output overlaps issue 2.

Mitigation: change only committed replacement output; preserve whichever ownership mechanics currently exist. Issue 2 later rewrites ownership under the same tests.

### Move-first regressions

Risk: later transport work reintroduces silent empty replay or drops move-only diffs.

Mitigation: make all oracle and empty-transition tests mandatory acceptance criteria in the move-first plan.

## Boundary with world-local cleanup

Issue 1 lands first and establishes normal parallel test reliability. All tests in this plan run under that baseline.

## Boundary with registration leases

Issue 2 may rewrite dynamic processor ownership and replay registration. It must preserve this plan's semantic outputs. This plan must not alter registration counts, handle cloneability, or active-subscription storage.

## Handoff to move-first transport

Before collection transport changes begin, require:

- All deterministic regressions pass.
- Vector/map/switch oracle tests pass.
- Empty replay is authoritative.
- `map_value_signal` empty replacement is preserved.
- Sort internal and emitted order remain synchronized.

Move-first collection phases may change:

- Erased value ownership.
- Diff clone bounds.
- Replay trigger transport.
- Description cloneability.
- Dynamic processor registration.

They may not change the semantic invariant that emitted diffs reconstruct exact state.

## Acceptance criteria

The issue is complete when:

- World-local cleanup has already established reliable parallel tests.
- Forward sorted updates emit correct indices in both sort implementations.
- Stable equal-key source moves update downstream order.
- Internal sort tracking and downstream materialized order never diverge in oracle tests.
- `map_value_signal` forwards empty replacements.
- Previously mapped inner signals produce no downstream output after committed replacement; exact lease cleanup is verified by issue 2.
- Vec/map replay emits authoritative snapshots, including empty, with per-subscriber revision cursors.
- Non-empty → empty switches clear downstream state through empty `Replace`.
- Pending or historical mutations already included in a snapshot are not applied again for that subscriber.
- Same-target switch behavior remains unchanged.
- Deterministic and fixed-seed oracle tests pass.
- No transport, registration ownership, or schedule changes are included.
- The registration-lease and move-first plans reference these tests as mandatory prerequisites.
