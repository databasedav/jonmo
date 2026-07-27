## Bottom line

`jonmo` has a strong core idea: represent reactive nodes as Bevy systems, materialize dependencies as an ECS graph, and expose the graph through typed signal combinators. The topological processing and fluent API are thoughtful.

The weak point is **lifecycle ownership across Rust handles and Bevy `World`s**. Several types look clonable or globally manageable even though their semantics are actually linear and world-local. That leads to genuine leaks, premature cleanup, cross-world interference, and some collection correctness bugs.

## Highest-priority issues

### 1. Critical: stale cleanup queues are process-global, but entities are world-local

The cleanup queues store bare `Entity` IDs in process-global statics:

- `jonmo/src/graph.rs:1228-1278`
- `jonmo/src/signal_vec.rs:4178-4211`
- `jonmo/src/signal_vec.rs:4310-4319`
- `jonmo/src/signal_map.rs:1059-1069`
- `jonmo/src/signal_map.rs:1184-1188`

Any app/world that runs its cleanup system drains the shared queue. Because Bevy entity IDs are only meaningful within their originating world, world B can consume cleanup work from world A. The vec/map cleanup paths even call `world.despawn(entity)` without verifying the entity belongs to the expected source type.

The current working-tree changes also clear all global queues whenever a plugin is built:

- `jonmo/src/lib.rs:175-180`

That can discard cleanup belonging to another live app.

This is not just theoretical: the normal parallel test run aborted inside `signal_vec::tests::test_enumerate`, while the same 131 tests pass when serialized. The project currently forces serialization in `jonmo/justfile:10-11`, masking the architectural problem.

**Recommendation:** make each queue world-local. A practical design is:

- Insert a cleanup queue resource into each world.
- Store an `Arc` to that specific queue inside `MutableVec`, `MutableBTreeMap`, and lazy signal ownership state.
- On `Drop`, enqueue into that world’s queue.
- Never globally clear cleanup state during plugin construction.

Alternatively, key global queues by a stable `WorldId`, but an `Arc` to a per-world queue is harder to misuse.

---

### 2. High: registration ownership and `SignalHandle` semantics are inconsistent

`SignalHandle` and `SignalHandles` derive `Clone`:

- `jonmo/src/graph.rs:1080-1120`

But cloning a handle does not increment `SignalRegistrationCount`. Each clone can independently call `cleanup`, which decrements the same count:

- `jonmo/src/graph.rs:1367-1428`

This can prematurely tear down a registration, drive the count negative, or leave a signal permanently ineligible for stale cleanup.

There is a second leak when the same cloned chain is registered multiple times. For a `Map`, every registration increments both the terminal and upstream nodes:

- `jonmo/src/signal.rs:183-195`
- `jonmo/src/graph.rs:1175-1191`

But cleanup only recurses upstream when the terminal reaches zero. Registering twice and cleaning both handles decrements the terminal twice but the upstream only once, leaving the upstream registered.

Dynamic combinators have related balancing problems:

- `flatten`: `jonmo/src/signal.rs:1916-1951`
- `switch_signal_vec`: `jonmo/src/signal.rs:2090-2117`
- `switch_signal_map`: `jonmo/src/signal.rs:2219-2247`
- `map_signal`: `jonmo/src/signal_vec.rs:1261-1289`
- `map_value_signal`: `jonmo/src/signal_map.rs:661-687`

They register an inner signal to discover its identity, retain only its copied `SignalSystem`, and discard the corresponding ownership token. Current active forwarder handles are also stored in system `Local`s, where dropping them does not call cleanup.

**Recommendation:** model each registration as a linear lease:

- Remove `Clone` from `SignalHandle` and `SignalHandles`.
- Have each handle retain the exact nodes whose counts were incremented by that registration.
- On cleanup, decrement every node in that lease exactly once, rather than recursing only when the terminal reaches zero.
- Store dynamic subscription handles in an entity-owned cleanup component, not solely in `Local<Option<SignalHandle>>`.

This would simplify several combinators and eliminate a large class of bookkeeping errors.

---

### 3. High: `Builder` implements a deliberately invalid `Clone`

`Builder::clone` panics in debug and returns an empty builder in release:

- `jonmo/src/builder.rs:31-89`

However, this is not merely satisfying an unused generic bound. Signal outputs are required to be `Clone`, and cloning occurs during normal execution:

- `signal::always` and `signal::once`: `jonmo/src/signal.rs:1070-1089`
- Graph fan-out: `jonmo/src/graph.rs:870-887`
- `VecDiff<Builder>` cloning: `jonmo/src/signal_vec.rs:40-75`

Consequences include:

- `signal::once(Some(Builder::new()...))` cloning the builder before emitting.
- Fan-out of a builder-producing signal panicking in debug.
- Release builds silently creating an empty entity on one branch.
- Behavior depending on downstream ordering.

A type implementing `Clone` must tolerate being cloned by generic code. Documentation cannot make an invalid implementation safe.

**Recommendation:** keep `Builder` move-only and introduce a reusable factory type for reactive children, for example a boxed/`Arc` factory that creates a fresh `Builder`. Reactive APIs could accept `BuilderFactory` rather than pretending the one-shot recipe itself is clonable.

---

### 4. High: several collection combinators can leave downstream state incorrect

#### Sorting updates moving toward a higher index

Both sorting implementations calculate `new_pos` after removing the old element, then incorrectly subtract one again:

- `sort_by`: `jonmo/src/signal_vec.rs:3120-3148`
- `sort_by_key`: `jonmo/src/signal_vec.rs:3332-3361`

For `[1, 2, 3]`, updating `1` to `4` should emit `RemoveAt(0), InsertAt(2, 4)`. The current implementation emits `InsertAt(1, 4)`, producing `[2, 4, 3]`.

#### Empty map replacement is suppressed

`map_value_signal` only forwards `Replace` when the new entries are non-empty:

- `jonmo/src/signal_map.rs:699-725`

`Replace { entries: [] }` clears internal processor state but emits nothing, leaving downstream consumers populated.

#### Switching from a populated collection to an empty one does not clear downstream state

Replay sources emit nothing when their current collection is empty:

- `jonmo/src/signal_vec.rs:4267-4287`
- `jonmo/src/signal_map.rs:1324-1344`

The switch combinators rely on replay to establish the new state:

- `jonmo/src/signal.rs:2119-2129`
- `jonmo/src/signal.rs:2249-2259`

Therefore, switching non-empty → empty produces no `Clear` or empty `Replace`; consumers retain the previous collection.

**Recommendation:** fix these directly and add oracle/property tests that apply every emitted diff to a plain `Vec`/`BTreeMap` and compare it against expected materialized state after every operation. This is particularly valuable for sort, filter, switch, and nested-signal combinators.

---

### 5. Medium-high: multi-schedule semantics are under-specified and inconsistently enforced

There are several distinct issues:

- Signals registered dynamically without an explicit `ScheduleTag` are excluded from same-frame fixpoint processing, even though they should use the default schedule: `jonmo/src/graph.rs:1017-1045`.
- Applying `.schedule()` to an already-cached cloned signal changes ECS tags without invalidating `by_schedule` or `signal_schedules`: `jonmo/src/graph.rs:139-156`, `jonmo/src/graph.rs:735-744`.
- A later-schedule source feeding an earlier-schedule target loses its buffered input when all buffers are cleared in `Last`: `jonmo/src/lib.rs:218-220`, `jonmo/src/graph.rs:1058-1067`.
- `trigger_signal_subgraph` immediately executes all descendants without respecting their configured schedules: `jonmo/src/graph.rs:892-925`.

**Recommendation:** define one explicit schedule contract:

1. Assign every registered node an effective schedule immediately.
2. Treat schedule changes as graph-cache invalidations.
3. Either reject unsupported backward schedule edges or retain their values until the target schedule consumes them.
4. Make synchronous internal triggers schedule-aware, or clearly model them as a separate immediate-execution graph that cannot cross schedule boundaries.

---

### 6. Medium: repeated builder injection can lose cleanup ownership

`spawn_on_entity` unconditionally replaces lifecycle bookkeeping:

- `jonmo/src/builder.rs:503-510`

This replaces `SignalHandles` and resets `ChildBlockPopulations`. Since cleanup is registered as an `on_remove` hook, replacing the component does not provide a clear, intentional cleanup path for the previous registrations. Existing children also remain while their block-offset bookkeeping is reset.

This conflicts with builder injection being presented as a supported integration mechanism.

**Recommendation:** either:

- Explicitly reject a second injection with a descriptive error, or
- Merge into existing `SignalHandles` and child-block state rather than replacing them.

## Secondary improvements

- The “constant-time regardless of collection size” claims are too strong. Positional `Vec` operations are O(n), `BTreeMap` is O(log n), sorting is O(n log n), and some combinators materialize and clone entire collections. “Incremental diff propagation that often avoids full replacement” would be more accurate.
- `map_signal`, `map_value_signal`, and `filter_signal` panic unless each inner signal emits synchronously during polling, but this is not documented as a public precondition.
- The README’s primary example uses the non-unit struct as `JonmoPlugin` instead of `JonmoPlugin::default()`: `jonmo/README.md:36-48`. The sync script generates the invalid expression at `jonmo/sync_readme_example.py:44-45`.
- Minimal-feature tests do not compile because the test module imports `bevy_time` and tests `.throttle()` unconditionally: `jonmo/src/signal.rs:3468-3472`, `jonmo/src/signal.rs:4816-4833`.
- Full CI runs only on `push`, while pull requests run example previews rather than the regular test/clippy/docs workflow: `jonmo/.github/workflows/ci.yaml:148-149`.

## What is already strong

- The signal graph maps naturally onto Bevy’s system model.
- Topological levels are deterministic and incrementally maintained.
- The runtime avoids holding `SignalGraphState` while user systems execute, which permits dynamic graph registration.
- Signal, vector, and map APIs are coherent and expressive.
- The builder’s child-block approach gives predictable ordering under its intended exclusive-ownership model.
- The project denies `unsafe`, supports feature-conscious dependencies, and has broad normal-path unit coverage.

## Suggested order of work

1. **Replace process-global cleanup queues with world-local ownership.**
2. **Redesign registration handles as linear per-registration leases.**
3. **Remove the invalid `Builder: Clone`; introduce a builder factory abstraction.**
4. **Fix empty collection propagation and sort indices; add property-based diff tests.**
5. **Formalize cross-schedule behavior and enforce it centrally.**
6. Harden repeated builder injection, feature tests, CI triggers, and documentation.

## Validation

I reviewed the current working tree as-is; it already had uncommitted changes in `src/graph.rs`, `src/lib.rs`, `src/signal_map.rs`, and `src/signal_vec.rs`. I did not modify anything.

Commands run:

- `cargo test --locked --lib -- --test-threads=1` — **131 passed**
- `cargo test --locked --lib` — **failed/aborted under parallel execution**, consistent with global-state interference
- `cargo test --locked --no-default-features --lib` — **failed** due to unconditional `bevy_time`/`throttle` test code
