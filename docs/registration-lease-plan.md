# Linear Registration Leases and Graph Ownership

## Status

Proposed implementation plan for issue 2: replace terminal-only, cloneable signal handles and topology-derived recursive cleanup with exact, linear, world-bound registration leases.

Canonical prerequisites:

1. [`world-local-cleanup-plan.md`](world-local-cleanup-plan.md)
2. [`collection-correctness-plan.md`](collection-correctness-plan.md)

Consumer:

- [`move-first-signal-delivery-plan.md`](move-first-signal-delivery-plan.md)

Collection semantic tests land before this plan rewrites replay and dynamic registration. This plan then lands before move-first transport or explicit broadcast descriptions.

## Problem statement

`SignalHandle` currently contains only a terminal `SignalSystem`, while registering one description can increment an entire chain or graph. Cleanup attempts to reconstruct ownership by recursively traversing upstream topology only when the terminal count reaches zero.

That representation cannot distinguish:

- Two independent registrations of the same chain.
- One registration that reaches the same source through two branches of a diamond.
- Duplicate acquisitions versus deduplicated graph edges.
- Dynamic subscriptions replaced independently from their outer graph.
- A reusable lazy definition from an active registration.

Additional problems:

- `SignalHandle` and `SignalHandles` implement `Clone`, but cloning does not acquire another registration.
- Counts are signed and can become negative.
- Dynamic combinators discard identity-only handles and keep final active handles only in `Local`s.
- A failed edge connection can leak handles acquired before `pipe_signal`.
- Bare `SignalSystem(Entity)` identities can collide across worlds.

## Decision summary

Model every successful registration as a linear lease over the exact multiset of node acquisitions performed by that call.

Core rule:

> If registration increments node N k times, the returned lease owns exactly those k increments and cleanup decrements them exactly once, independently of current graph topology.

Adopt these decisions:

1. `SignalSystem` carries both `WorldId` and `Entity`.
2. Registration counts are checked `usize` values initialized at zero.
3. Every successful registration returns one non-cloneable `SignalHandle`.
4. A handle stores its terminal identity and an immutable exact node/edge registration blueprint.
5. Blueprints preserve duplicate node and edge acquisitions.
6. A registration blueprint contains both node acquisitions and active-edge acquisitions.
7. Composite registrations merge blueprints explicitly; set-valued topology never deduplicates ownership multiplicity.
8. `LazySignal` caches the complete blueprint produced by first materialization.
9. Later registrations reactivate every cached node and edge acquisition.
10. Cleanup releases the stored blueprint rather than traversing topology to discover ownership.
11. Node/edge transitions between active and dormant state are distinct from final entity reaping.
12. Reusable lazy-definition liveness is tracked explicitly and separately from active registration counts.
13. Edge activation is preflighted and transactional.
14. Active dynamic subscriptions live in entity-owned cleanup components and deactivate with their owner.
15. `SignalHandle` and `SignalHandles` do not implement `Clone`.
16. Graph operations reject foreign-world identities before entity access.
17. Final definition drops without world access send one batched root through issue 1's world-local inbox.
18. Lease release, owner deactivation, and definition reaping performed with `&mut World` append causal candidates to issue 1's local `CleanupWorklist`, never back into the concurrent inbox.
19. Output transport and description cloneability remain unchanged until the move-first plan.

## Goals

- Balance every registration increment exactly once.
- Support duplicate registration, diamonds, shared upstreams, and dynamic replacement.
- Make cleanup ownership explicit and linear.
- Make failed connection attempts rollback-safe.
- Preserve reusable lazy descriptions after a temporary registration is cleaned.
- Eliminate identity-only registration leaks.
- Eliminate active dynamic handles stored solely in `Local`s.
- Provide the ownership substrate required by explicit broadcast descriptions.
- Keep issue 1's world-local inbox as the only deferred lifecycle queue and reuse its explicit local causal worklist.

## Non-goals

This plan does not:

- Replace `AnyClone` output transport.
- Add `OutputDelivery` or `.broadcast()`.
- Remove ordinary signal-description `Clone` globally.
- Remove `Builder: Clone`.
- Fix collection diff semantics.
- Redesign schedules.
- Make all public registration APIs return `Result`.
- Automatically clean arbitrary dropped `SignalHandle`s through `Drop`.
- Support one lazy identity materialized independently in multiple worlds.

## Required invariants

### Registration

1. Every successful public registration returns exactly one live lease.
2. Every count increment belongs to exactly one lease-blueprint entry.
3. Every lease-blueprint entry causes exactly one decrement when cleaned.
4. Blueprints preserve node and edge acquisition multiplicity.
5. All systems and edges in one blueprint belong to one world.
6. Cached lazy registration reacquires exactly its original blueprint.
7. Node and edge counts never underflow or overflow.
8. Cleanup never discovers ownership by traversing current topology.
9. A node with registration count zero is inactive: it is absent from active schedule buckets, does not run, receives no forwarded input, and has its pending input cleared.
10. A `0 -> 1` transition reactivates the node and a `1 -> 0` transition deactivates it without necessarily reaping its entity.

### Topology

11. `Upstream` and `Downstream` remain symmetric for active edges.
12. Active topology is set-valued while edge acquisition counts preserve multiplicity.
13. Failed edge batches install no partial edge mutation, leak no acquisition, and leave pre-existing active branches unchanged.
14. Dormant definitions retain cached edge blueprints, not active topology.
15. Reacquisition transactionally reconnects/reactivates cached edges.
16. Despawning a node removes incoming and outgoing active relationships.
17. A live downstream registration owns active node and edge acquisitions for every dependency in its blueprint.

### Handles and components

18. `SignalHandle` and `SignalHandles` do not implement `Clone`.
19. A cleanup component drains each stored handle at most once.
20. Active dynamic handles never live solely in a system `Local`.
21. Dynamic replacement keeps the old subscription active until the new connection succeeds.
22. Owner deactivation drains active dynamic subscriptions; owner reactivation rebuilds them from subsequent outer emissions/replay.
23. Owner despawn cleans final active subscriptions.
24. `SignalSystem` is copyable identity only, never ownership.

### Worlds

25. Every identity and lease is bound to a `WorldId`.
26. Cleanup, connection, polling, triggering, scheduling, and registration reject foreign worlds before mutation.
27. Failed cleanup preflight returns the still-armed lease to the caller.
28. Entity index collisions across worlds cannot compare equal as signal identities.
29. A final description drop without world access sends one move-only `LazyDefinitionCandidates` obligation root through issue 1's world-local inbox.
30. Candidate work discovered while cleanup already has `&mut World` appends to the invocation-local `CleanupWorklist`.
31. Causal lease/definition follow-ups never route back through the concurrent inbox.
32. Every local candidate batch is finite, move-only after construction, and backed by consumed description/lease ownership, preserving issue 1's monotonic termination invariant.

### Future broadcast

33. A broadcast description clone will acquire a new lease; it will not clone a handle.
34. A broadcast cached blueprint includes its upstream chain, active edges, and one broadcast-node acquisition.
35. Registration multiplicity is independent of set-valued topology.
36. Failed move-only fan-out can rollback newly acquired node/edge acquisitions without affecting existing branches.

## World-aware signal identity

Redefine `SignalSystem`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SignalSystem {
    world: WorldId,
    entity: Entity,
}
```

Provide explicit accessors:

```rust
impl SignalSystem {
    pub fn entity(self) -> Entity;
    pub fn world(self) -> WorldId;
}
```

All internal graph entry points validate:

```rust
signal.world() == world.id()
```

Apply this to:

- Registration reacquisition.
- Edge connection.
- Cleanup.
- Polling.
- Synchronous triggering.
- Schedule application.
- Graph-cache updates.

Remove or restrict conversions from bare `Entity` and `SystemId`; they lack world identity. Construct `SignalSystem` only where both world and entity are available.

## Checked registration counts

Replace signed counts:

```rust
#[derive(Component, Deref)]
struct SignalRegistrationCount(usize);
```

New nodes start at zero. All ownership changes use centralized helpers:

```rust
fn acquire_plan(
    world: &mut World,
    plan: &[SignalSystem],
) -> Result<(), RegistrationInvariantError>;

fn release_plan(
    world: &mut World,
    plan: &[SignalSystem],
) -> Result<(), RegistrationInvariantError>;
```

Before mutation:

1. Verify every system belongs to the current world.
2. Aggregate repeated systems into multiplicities.
3. Verify all entities and count components exist.
4. Check additions for overflow or releases for underflow.
5. Only then mutate counts.

Example plan:

```text
[source, left, source, right, join]
```

represents:

```text
source × 2
left   × 1
right  × 1
join   × 1
```

Do not deduplicate the lease plan.

## Registration blueprint and linear `SignalHandle`

Cache both node and edge acquisitions:

```rust
#[derive(Clone)]
struct RegistrationBlueprint {
    nodes: Arc<[SignalSystem]>,
    edges: Arc<[(SignalSystem, SignalSystem)]>,
}

#[must_use = "dropping a SignalHandle without cleanup leaves its registration active"]
pub struct SignalHandle {
    terminal: SignalSystem,
    blueprint: Arc<RegistrationBlueprint>,
}
```

Properties:

- No `Clone` implementation.
- Private fields.
- No `From<SignalSystem>`.
- No public tuple construction or destructuring.
- Exact node and edge multiplicity retained in the blueprint.
- Blueprint metadata may be shared with a lazy cache through `Arc`; active registration ownership itself is not shared.

Public API:

```rust
impl SignalHandle {
    pub fn system(&self) -> SignalSystem;
    pub fn try_cleanup(
            self,
            world: &mut World,
        ) -> Result<(), (RegistrationCleanupError, SignalHandle)>;

        pub fn cleanup(self, world: &mut World);
}
```

A temporary `Deref<Target = SignalSystem>` may reduce migration churn, but new code should use `.system()`.

### Explicit cleanup

For this plan, cleanup remains explicit and consuming. Preflight failure must not destroy the only lease token:

```rust
match handle.try_cleanup(world) {
    Ok(()) => {}
    Err((error, handle)) => {
        // `handle` remains armed and can be cleaned in its originating world.
    }
}
```

`cleanup` may be a convenience wrapper only if it cannot lose the armed handle on failure; for example, it may require an already validated same-world path or enqueue the returned handle through a future explicit mechanism. Do not panic after consuming an armed lease and then silently drop it.

Issue 1 makes future deferred cleanup from `Drop` possible, but adding that behavior changes cleanup timing and remains a separate decision. `#[must_use]` makes accidental unmanaged drops visible.

## Pending registration composition

Introduce an internal builder that consumes child leases:

```rust
struct PendingRegistration {
    terminal: SignalSystem,
    handles: Vec<SignalHandle>,
    proposed_edges: Vec<(SignalSystem, SignalSystem)>,
}
```

Suggested operations:

```rust
impl PendingRegistration {
    fn from_handle(handle: SignalHandle) -> Self;
    fn append(&mut self, handle: SignalHandle);
    fn propose_edge(&mut self, source: SignalSystem, target: SignalSystem);
    fn set_terminal(&mut self, terminal: SignalSystem);
    fn connect_and_finish(
        self,
        world: &mut World,
    ) -> Result<SignalHandle, RegistrationError>;
}
```

Retain original handles until every fallible edge preflight succeeds. Only after commit may the helper flatten/concatenate their blueprints. A raw vector of identities after consuming handles would have no safe rollback owner. Node and edge acquisitions both preserve duplicates.

Examples:

```text
Map registration
= upstream blueprint + map-node blueprint + connecting edge

Zip registration
= left blueprint + right blueprint + join-node blueprint + both edges

Chain registration
= all input blueprints + chain-node blueprint + connecting edges
```

If both zip inputs share one source, that source appears twice in the returned plan.

## Lazy registration blueprint

Change lazy registration so its initializer returns a complete handle rather than a bare terminal:

```rust
enum LazySystem {
    Unregistered(
        Option<
            Box<
                dyn FnOnce(&mut World)
                    -> Result<SignalHandle, RegistrationError>
                    + Send
                    + Sync
            >
        >,
    ),
    Registered {
        world: WorldId,
        terminal: SignalSystem,
        blueprint: Arc<RegistrationBlueprint>,
        cleanup: CleanupSender,
    },
    Poisoned,
}
```

### First registration

1. Run the initializer.
2. Receive one complete, already-acquired lease.
3. Validate the blueprint's world.
4. Cache terminal, node/edge blueprint metadata, and world cleanup sender.
5. Return the original handle.

### Later registration

1. Reject a different world before entity lookup.
2. Preflight cached node activation and edge reconnection as one transaction.
3. Increment every node/edge acquisition, including duplicates.
4. Apply `0 -> 1` node activation and reconnect edges whose acquisition count becomes active.
5. Return a new non-cloneable handle sharing immutable blueprint metadata.

Any current initializer that extracts `*handle` must instead return the full handle. No compatibility adapter may discard the plan.

## Lazy-definition ownership

Registration counts answer whether a node is actively leased. They do not answer whether a materialized graph must remain available for a reusable lazy description.

Example:

1. Clone a description under the current pre-move-first API.
2. Register one clone.
3. Clean its handle.
4. Register the retained clone again.

The materialized nodes and topology must remain available between steps 3 and 4.

### Definition owner component

Attach weak definition metadata to every unique node in the lazy blueprint:

```rust
#[derive(Component, Default)]
struct LazyDefinitionOwners {
    owners: Vec<Weak<LazySignalState>>,
}
```

`LazySignalState` also contains an explicit `live_descriptions: AtomicUsize`. Only actual `LazySignal` description construction/cloning increments it and only description drop decrements it. Internal temporary strong references, cleanup payloads, and bookkeeping must not affect definition liveness.

On first materialization:

1. Add a weak state reference to each unique blueprint node.
2. Cache the unique candidate set in `LazySignalState`.
3. Store issue 1's cleanup sender.

A node is reaping-eligible only when:

- Its registration count is zero and it is inactive.
- Every attached definition owner either no longer upgrades or reports `live_descriptions == 0`.
- It has not already been removed.

### Definition drop

Represent the zero-description transition as a private move-only obligation token:

```rust
pub(crate) struct ReleasedDefinitionCandidates(Arc<[SignalSystem]>); // deliberately non-Clone
```

The cached candidate identities may use shared immutable storage, but only the `1 -> 0` liveness transition constructs the wrapper and safe cleanup APIs consume it exactly once. This supports issue 1's local termination accounting even though `SignalSystem` identities themselves remain copyable.

When `live_descriptions` transitions to zero:

1. If release occurs without world access, send its unique node candidates as one issue-1 `LazyDefinitionCandidates` root.
2. If release occurs inside an active world cleanup transaction, append that same finite batch to issue 1's local `CleanupWorklist`.
3. The worklist handler rechecks counts and all remaining weak definition owners.
4. Eligible nodes are reaped downstream-to-upstream within the same local causal transaction.

Remove the current `LazySignalHolder` strong-reference workaround and reference-threshold drop logic. Registration/cleanup bookkeeping may hold internal strong references only if they do not increment `live_descriptions`; nodes should otherwise retain weak definition metadata.

## Active nodes, active edges, and dormant definitions

Materialized entity lifetime and graph activation are separate states.

### Node activation

- `0 -> 1` registration count inserts/reactivates the node in effective schedule buckets and makes it executable.
- `1 -> 0` invokes a deactivation hook, removes the node from active schedule buckets, clears pending input, and drains entity-owned dynamic subscriptions without necessarily despawning the entity.
- Scheduled processing, synchronous triggering, polling propagation, and downstream selection skip inactive count-zero nodes.
- Reaping occurs only when the node is inactive and no live definition owns it.

### Edge activation counts

Cache edge acquisitions in every blueprint and track active multiplicity separately from set-valued `Upstream`/`Downstream` topology, for example in graph state:

```rust
edge_registration_counts: HashMap<(SignalSystem, SignalSystem), usize>
```

- Edge acquisition `0 -> 1` installs the active relationship components.
- Additional acquisitions increment count without duplicating topology.
- Edge release `1 -> 0` removes the active relationship and invalidates affected levels.
- Dormant lazy definitions retain edge blueprints only; they do not leave active relationships behind.
- Cached reacquisition transactionally reactivates both nodes and edges.

This model lets future move-first fan-out count active edges. An inactive cached branch does not falsely consume fan-out capacity, and reactivation revalidates fan-out through the normal edge transaction.

## Transactional edge connection

Replace one-edge mutation with preflighted batch activation:

```rust
fn try_connect_edges(
    world: &mut World,
    edges: &[(SignalSystem, SignalSystem)],
) -> Result<EdgeCommit, SignalGraphError>;
```

Validation before mutation:

1. Verify all systems belong to `world.id()`.
2. Verify entities and required graph components exist.
3. Preserve proposed edge acquisition multiplicity while normalizing the set of topology transitions.
4. Remove already-active edges from the topology-install subset while still incrementing their acquisition counts; return early only if the entire remaining batch has no validation or activation work.
5. Detect cycles against existing and all newly activated topology edges.
6. Calculate schedule inheritance changes.
7. Leave a hook for issue 3's output-delivery/fan-out validation.

Only after complete success mutate:

- `Downstream`.
- `Upstream`.
- `ScheduleTag`/`ScheduleHint` inheritance.
- `edge_change_seeds`.

### Registration transaction helper

```rust
fn connect_registration(
    world: &mut World,
    terminal: SignalHandle,
    upstreams: impl IntoIterator<Item = SignalHandle>,
    edges: &[(SignalSystem, SignalSystem)],
) -> Result<SignalHandle, SignalGraphError>;
```

Behavior:

- Own every newly acquired handle.
- Preflight/connect all edges atomically.
- Retain original handles until preflight succeeds.
- On success, commit edge-count/topology transitions, then merge blueprints and return one terminal lease.
- On structured failure, release every newly acquired lease in reverse construction order.
- Reap newly spawned zero-count nodes where eligible.
- Install no partial mutation from the rejected edge batch, leak no acquisition, and leave pre-existing active branches unchanged.

Nested first-time lazy materialization may already have consumed `FnOnce` initializers and cached a dormant definition before the final outer edge batch fails. If a live description owns that definition, rollback may legitimately leave inactive materialized entities/blueprints rather than restoring byte-for-byte pre-attempt structure.

Convert the structured error to current public panic behavior only after rollback. Arbitrary panics inside custom/user registration code remain outside this guarantee until registration broadly becomes fallible.

## Cleanup algorithm

Remove `cleanup_recursive`.

Extend issue 1's local worklist with a move-only `ReleaseSignalLease(SignalHandle)` action and typed `push_signal_lease`. Public `SignalHandle::try_cleanup` creates an empty local worklist seeded with its own lease action and drives the phased transaction to exhaustion before returning; callers already inside cleanup move leases into their existing worklist.

The worklist must exhaust all lease-release actions, including dynamically discovered inner leases, before reaping accumulated node candidates. If reaping explicitly drains another lease-owning component, return to the release phase before continuing. This prevents an outer candidate from being tested while an inner lease still owns an edge into it.

The internal lease-release action:

1. Verify the handle, every node, and every edge belong to the supplied world; on failure return `(error, handle)` unchanged.
2. Aggregate node and edge multiplicities.
3. Preflight every decrement.
4. Compute deterministic reverse topological order from the released blueprint's edge list while active topology is still available.
5. Apply edge releases, removing active topology on `1 -> 0`.
6. Apply node releases. On each `1 -> 0` transition, invoke the node-deactivation hook, remove schedule membership, clear pending input, and drain any entity-owned dynamic subscription slots exactly once into the same cleanup transaction.
7. Build a unique finite node candidate set from the released blueprint.
8. Append/transfer that candidate batch to the local worklist or reap it immediately through the same worklist handler in precomputed order.

For each reaped node:

- Explicitly drain/disarm cleanup-producing components into the current local worklist before despawn.
- Remove it from upstream `Downstream` sets.
- Remove it from downstream `Upstream` sets.
- Mark affected descendants for level recomputation.
- Remove graph-cache entries through the existing deferred-removal mechanism.
- Despawn the system entity fallibly.

No reaping or component-deactivation path may use ordinary `Drop` to republish causal work through the world inbox. Each appended batch must be backed by a consumed lease, description reference, or entity-owned slot so issue 1's finite-potential termination argument remains valid.

If cleanup occurs while `SignalGraphState::is_processing`, defer ECS despawn/topology finalization as required so active iteration cannot observe invalid state. Acquisition order is not a valid reaping order for diamonds; always compute graph order.

Reaching node count zero deactivates execution and active edges. It does not necessarily reap the materialized entity when a reusable definition remains alive.

Do not use downstream presence as an ownership substitute. Correct blueprints account for dependency ownership explicitly.

## Dynamic subscription ownership

### Component-owned leases

Introduce a reusable internal component:

```rust
#[derive(Component, Default)]
#[component(on_remove = cleanup_dynamic_signal_leases)]
struct DynamicSignalLeases {
    slots: Vec<Option<SignalHandle>>,
    free: Vec<usize>,
}
```

A slot key lets vector indices and map keys reference active leases without moving or cloning handles. Draining a slot moves its lease into `CleanupWorklist::push_signal_lease`; it never recursively invokes public cleanup and never sends through the concurrent root inbox.

Identity is stored separately:

```rust
struct ActiveSubscription {
    system: SignalSystem,
    lease_slot: LeaseSlot,
}
```

### Replacement protocol

1. Register the new inner description once.
2. Obtain identity from `handle.system()`.
3. Poll that registered terminal directly if needed.
4. Register the typed processor/forwarder separately.
5. Connect and merge the inner/processor leases transactionally.
6. Keep the old subscription active until success.
7. Swap the new lease into the owner component.
8. Update the stored copyable identity.
9. Move the old lease into the active local cleanup worklist after the successful swap.
10. On owner removal/deactivation, drain every remaining slot into that same worklist.

If the structured new connection fails, release only the new acquisitions and retain the old subscription.

When the static outer manager/output lease transitions inactive, move its active dynamic slots into the current worklist before adding the outer blueprint's node candidates, even if the retained definition keeps the owner entity materialized. Count-zero dormant entities do not trigger `on_remove`. Later reactivation starts with empty dynamic slots and rebuilds inner subscriptions from subsequent outer emissions or replay.

### Static manager acquisitions

Current dynamic lazy initializers often register a static manager chain and store its handle in `SignalHandles` instead of returning it as part of the outer registration. Under exact leases, the outer cached blueprint must include:

- The output-node acquisition.
- The static manager-chain acquisition.
- Every connecting edge acquisition.

Do not duplicate that ownership in `SignalHandles`. Dynamic per-selection/per-item inner leases remain in `DynamicSignalLeases`; static manager ownership belongs to the outer cached blueprint.

### No identity-only registration

Dynamic paths must not register an inner description only to discover its identity. One registration provides both:

- `handle.system()` for identity.
- The owned registration blueprint.

Remove temporary `.first()` registrations used only for initial polling. Poll the registered inner terminal directly.

### Paths to migrate

- `SignalExt::flatten`
- `SignalExt::switch`
- `SignalExt::switch_signal_vec`
- `SignalExt::switch_signal_map`
- `SignalVecExt::map_signal`
- `SignalVecExt::filter_signal`
- `SignalMapExt::map_value_signal`

Remove inner-description `S: Clone` bounds only where cloning exists solely to register the same description repeatedly. Retain item clone bounds required by current output transport until issue 3.

## `SignalHandles` component

Redefine as a non-cloneable owner:

```rust
#[derive(Component, Default, Deref)]
#[component(on_remove = cleanup_signal_handles)]
pub struct SignalHandles(Vec<SignalHandle>);
```

Requirements:

- No `Clone`.
- Append/move handles through `add`.
- Drain exactly once in the lifecycle hook.
- Share hook cleanup machinery with dynamic lease storage.
- Support entity despawn and explicit component removal.
- Define replacement behavior explicitly; prefer mutation over replacement.

## Builder integration

`Builder::spawn_on_entity` must not overwrite an existing `SignalHandles` component with an empty one. For issue 2:

- Insert `SignalHandles::default()` only if absent.
- Preserve existing registration ownership during builder injection.
- Keep `add_handles` move-only.

Resetting `ChildBlockPopulations` remains part of the broader repeated-injection issue, but registration ownership must not be discarded.

## External custom signal implementations

`Signal`, `SignalVec`, and `SignalMap` are publicly implementable. Removing tuple-handle construction without replacement would make correct external implementations impossible.

Choose and document one supported boundary:

### Preferred: safe advanced registrar

Expose a narrow world-aware API, potentially through a `SignalRegistrar<'w>`:

```rust
pub struct SignalRegistrar<'w> {
    world: &'w mut World,
}

impl<'w> SignalRegistrar<'w> {
    pub fn register_system<I, O, IOO, F, M>(...) -> SignalHandle;
    pub fn connect(
        &mut self,
        terminal: SignalHandle,
        upstreams: impl IntoIterator<Item = SignalHandle>,
        edges: &[(SignalSystem, SignalSystem)],
    ) -> Result<SignalHandle, SignalGraphError>;
}
```

It must construct world-aware identities, exact blueprints, and transactional edges without exposing raw handle fields.

### Alternative: seal registration traits

If safe external composition cannot be supported within scope, explicitly seal the traits and treat that as a breaking API decision. Do not leave public implementers with no valid handle-construction path.

Delegation to built-in descriptions remains valid because delegated registration returns a complete lease.

## Source touchpoints

### `src/cleanup.rs`

- Replace temporary `StaleSignal` threshold/holder requests with batched `LazyDefinitionCandidates` work during lazy migration.
- Provide both a typed root-sender method for no-`World` final description drops and a typed `CleanupWorklist` append method for causal in-world releases.
- Dispatch candidate batches to the lease eligibility/reaping handler within the current local cleanup transaction.
- Ensure one ownership release cannot emit both old/new variants or both inbox/local copies.

### `src/graph.rs`

- `SignalSystem`: add world identity and checked accessors.
- `SignalRegistrationCount`: use checked `usize`, initialized at zero.
- `register_signal`: return a one-node handle.
- `pipe_signal`: replace with batch preflight/commit.
- `SignalHandle`/`SignalHandles`: linear lease ownership.
- `spawn_signal`: create zero-count node then acquire through helpers.
- `LazySignalState`/`LazySystem`: cache world, terminal, exact node/edge blueprint, and explicit live-description count.
- `LazySignal` ordinary drop: send one definition-candidate root through issue 1 only when no world cleanup context owns the release.
- Explicit in-world description/lease release: append candidates to issue 1's local worklist.
- Remove `LazySignalHolder` ownership semantics.
- Replace `cleanup_recursive` with aggregate release/reaping that drains causal component ownership explicitly before despawn.
- Add wrong-world checks to poll/trigger/schedule/graph helpers.

### `src/signal.rs`

- Make all registration implementations preserve complete handles.
- Convert `Map`, `Zip`, schedules, and lazy initializers.
- Convert flatten/switch dynamic ownership.
- Never destructure a handle into only its terminal.

### `src/signal_vec.rs`

- Convert `ForEach`, `Chain`, schedules, replay, and all lazy initializers.
- Convert `map_signal`/`filter_signal` to component-owned leases.
- Preserve duplicate acquisition multiplicity.

### `src/signal_map.rs`

- Convert `ForEach`, map processors, schedules, replay, and lazy initializers.
- Convert `map_value_signal` to keyed component-owned leases.

### `src/builder.rs`

- Move handles into owner components.
- Preserve existing `SignalHandles` during injection.
- Keep task adapters returning one linear handle.

### `src/lib.rs`

- Reuse issue 1's lifecycle resource.
- Do not restore global cleanup queues.

## Implementation phases

### Phase 0: Freeze semantics and add introspection

Add test-only graph snapshots containing:

- Registration counts.
- Upstream/downstream sets.
- Schedule tags/hints.
- Graph-cache membership.
- Lazy definition owners.
- System entity existence.

Add failing regressions for:

- Cloned-handle cleanup.
- Duplicate chain registration cleanup.
- Diamond acquisition multiplicity.
- Dynamic final-subscription leak.

### Phase 1: Confirm cleanup and collection prerequisites

Require:

- Per-world lifecycle inbox and boundary-delimited root snapshot.
- Invocation-local `CleanupWorklist` with monotonic termination invariants.
- No process-global entity queues.
- Lazy definition-drop notifications can use a world-bound sender while in-world releases append locally.
- Normal parallel test isolation.
- Collection authoritative replay, empty replacement, switch, sort, and oracle tests are green.

Do not proceed by adding a temporary second queue.

### Phase 2: Atomically introduce world-aware identities and blueprints

In one compile-complete migration:

- Make `SignalSystem` world-aware.
- Add checked node/edge count helpers.
- Introduce private-field, non-cloneable `SignalHandle`.
- Remove `Clone` from `SignalHandles`.
- Add pending-registration helpers that retain original handles through fallible work.
- Convert foundational static paths (`Source`, `Map`, `Zip`, scalar/vec/map `ForEach`, `Chain`, replay, and schedule wrappers) so no call site destructures/discards ownership.
- Provide the safe external registrar or explicitly seal public registration traits.
- Keep output transport and description clone behavior unchanged.

Acceptance:

- Wrong-world operations fail before mutation.
- Handle cloning fails to compile.
- One-node acquisition/release balances.
- Foundational static combinators return complete blueprints.
- External custom implementations retain a supported safe path or are explicitly sealed.

### Phase 3: Convert lazy registration

- Initializers return complete handles.
- Cache exact immutable node/edge blueprints.
- Reacquire/reactivate cached blueprints on later registration.
- Attach weak definition metadata plus explicit live-description counts.
- Replace issue 1's temporary stale-signal path with batched definition candidates in `src/cleanup.rs`.
- Send batches as roots only from ordinary no-`World` final description drop; append batches locally for releases already inside cleanup.
- Remove holder/reference-threshold semantics without reintroducing implicit queue-producing component drops.

Acceptance:

- Duplicate chain registrations balance every node.
- Dormant retained descriptions can register again.
- Dropping a final description outside world access sends one root and eventually reaps zero-count definitions.
- In-world final releases reap causally in the same local cleanup transaction without another inbox round trip.

### Phase 4: Transactional connections and cleanup

- Add atomic node/edge activation preflight and edge acquisition counts.
- Convert remaining registration paths to transactional edge activation.
- Replace recursive cleanup with aggregate blueprint release.
- Deactivate count-zero nodes and edges while retaining dormant definition blueprints.
- Add deterministic reverse-topological candidate reaping and processing-safe deferred removal.
- Thread the active `CleanupWorklist` through node deactivation, dynamic slot draining, and reaping so every causal candidate remains local.

Acceptance:

- Cycle/batch failures install no partial batch mutation, leak no acquisitions, and leave pre-existing active branches unchanged.
- Diamond blueprints preserve duplicates.
- Shared branches clean independently.

### Phase 5: Dynamic combinators

Migrate dynamic scalar, vec, and map paths to entity-owned lease slots and transactional replacement.

Acceptance:

- Same-identity emissions balance transient acquisitions.
- Replacements clean old leases only after success.
- Owner deactivation drains active inner leases; owner reactivation rebuilds from subsequent emissions/replay.
- Owner despawn cleans any final active subscriptions.
- No temporary polling edges remain.
- Repeated inner identity across indices/keys remains independently owned.
- The full authoritative replay, empty replacement, switch, and collection oracle suite remains green.

### Phase 6: Public ownership hardening

- Finalize `SignalHandle::system` and `#[must_use]`.
- Remove tuple/from-system construction surfaces.
- Finalize component hooks.
- Preserve builder-owned handles during injection.
- Update documentation and changelog.

### Phase 7: Handoff to move-first transport

Only after lease tests pass:

- Add output-delivery preflight to the existing edge transaction.
- Add explicit broadcast wrappers using cached node/edge blueprints.
- Remove ordinary description `Clone`.
- Remove `Builder: Clone`.

## Test plan

### Core lease tests

- Simple chain acquisition/release.
- Two registrations of one chain.
- Cleanup in either order.
- Handle and handle-component compile-fail clone tests.
- Diamond multiplicity.
- Shared upstream independent branches.
- Duplicate edge with duplicate acquisition.
- Dormant reusable definition.
- `1 -> 0` deactivation removes schedule execution, active edges, and pending input without reaping a live definition.
- `0 -> 1` cached reactivation restores nodes/edges transactionally.
- Count-zero nodes are skipped by scheduled processing, polling propagation, and synchronous triggers.
- Wrong-world `try_cleanup` returns the still-armed handle unchanged.
- Wrong-world registration.
- Checked underflow/overflow preflight.

### Transaction tests

- Cycle rollback.
- One-valid/one-invalid batch atomicity.
- Schedule inheritance rollback.
- Existing-edge topology idempotence with edge acquisition count increments.
- Duplicate edge acquisitions release independently.
- Missing entity/component rollback.
- Future move-only fan-out rollback hook.

### Component lifecycle

- Entity despawn drains several handles once.
- Explicit `SignalHandles` removal.
- Defined replacement behavior.
- Builder injection preserves existing handles.
- Dynamic owner deactivation drains active slots even while its definition entity remains materialized.
- Dynamic owner removal drains active slots.

### Dynamic scalar

- Flatten A → B → A.
- Same identity repeated.
- Failed replacement preserves old forwarding.
- Owner despawn cleans current inner subscription.
- Optional/no-inner transitions where supported.

### Dynamic collections

- `map_signal` replace/insert/update/remove/move/pop/clear.
- Same inner at multiple indices.
- `map_value_signal` same inner at multiple keys.
- Switch vec/map repeatedly.
- Output owner despawn drains all leases.
- Initial polling creates no temporary edge.

### World and parallel

After issue 1:

- Full library suite with normal parallelism.
- Two apps with colliding entity indices and independent registrations.
- Descriptions and handles dropped in opposite orders.
- No world consumes or mutates another's lifecycle state.

## Data model alternatives

| Option | Advantages | Problems | Decision |
|---|---|---|---|
| Flat node/edge `RegistrationBlueprint` | Simple, exact node/edge multiplicity, cheap lazy reactivation | Cleanup O(blueprint length) | Recommended |
| Tree of child leases | Avoids flattening | Harder rollback, caching, diagnostics, more allocations | Reject initially |
| ECS lease entity | Strong introspection | One entity per registration, dynamic churn | Reconsider only if introspection is a priority |
| World registry with lease IDs | Small handles | Additional registry lifecycle and stale IDs | Not needed |
| Cloneable `Arc<LeaseInner>` handle | Familiar | Clone still does not acquire counts; obscures linearity | Reject |
| Clone acquires registration | Intuitive | `Clone` has no `&mut World` | Impossible with current API |
| Recursive terminal cleanup | Minimal change | Cannot represent duplicates/diamonds | Reject |

## Migration impact

Public lifecycle changes:

- `SignalHandle: Clone` removed.
- `SignalHandles: Clone` removed.
- `SignalHandle` fields become private.
- `.system()` replaces tuple access.
- `SignalSystem` becomes world-aware.
- Bare entity conversions are removed/restricted.
- Cross-world misuse becomes an explicit failure.
- `cleanup` remains consuming.

Custom signal implementations must obey a stronger contract:

> Registration must return ownership for every node acquisition performed while constructing the returned terminal.

Delegating to built-in descriptions naturally returns a complete handle. Extracting a terminal and discarding a delegated handle is invalid.

## Boundary with world-local cleanup

Issue 1 owns:

- Per-world root inbox, bound/deferred senders, and one-shot command binder.
- Boundary-delimited external root snapshots.
- Invocation-local causal `CleanupWorklist` and its monotonic termination contract.
- Explicit release/disarm substrate.
- Removal of global queues.
- Mutable collection world ownership and aggregate broadcaster release.
- Safe dead-world send behavior.

This plan owns:

- World-aware graph identities.
- Registration lease payloads.
- Lazy-definition candidate roots and local worklist batches composed through issue 1.
- Exact count balancing.
- Transactional graph connection.
- Dynamic active subscriptions and explicit causal draining.

No second lifecycle queue may be added. No cleanup path already holding `&mut World` may round-trip causal candidate work through the concurrent inbox; it must reuse the current local worklist.

## Boundary with collection correctness

[`collection-correctness-plan.md`](collection-correctness-plan.md) owns diff semantics and lands before this lease work. Registration phases that rewrite replay or dynamic processors must rerun its authoritative replay, empty replacement, switch, sort, and oracle tests. This plan changes ownership beneath those semantics and must not preserve known incorrect behavior.

## Handoff to move-first transport

After this plan lands, the move-first plan should treat the following as supplied infrastructure:

- Linear `SignalHandle` and `SignalHandles`.
- World-aware `SignalSystem`.
- Exact cached lazy node/edge blueprints and activation transitions.
- Transactional edge connection and rollback.
- Entity-owned dynamic subscription slots.
- Direct polling of registered inner terminals.

Move-first then adds:

- Move-only erased values.
- Output delivery policies.
- Fan-out validation inside the existing edge preflight.
- Explicit cloneable broadcast description wrappers.
- Move-only ordinary descriptions.
- Move-only `Builder`.

A broadcast description clone acquires/reactivates a new lease from one cached broadcast blueprint; it never clones a handle or recursively rebuilds topology.

## Acceptance criteria

The issue is complete when:

- `SignalSystem` is world-aware.
- Counts are checked unsigned values.
- Every registration returns one non-cloneable exact node/edge lease.
- Duplicate node and edge acquisitions are preserved.
- Count-zero nodes and edges are inactive and cannot execute or forward.
- Cached definitions reactivate transactionally.
- Duplicate registrations clean independently.
- Diamonds balance exactly.
- Lazy cached blueprints reactivate correctly.
- Reusable dormant definitions retain blueprints while their nodes and edges remain inactive.
- Final definition drop without world access uses one world-local root; in-world releases append locally.
- Causal lease/definition cleanup reaches a local fixed point without a retry constant or inbox recursion.
- Recursive cleanup is removed.
- Edge batches are transactional.
- Structured connection failures install no partial batch mutation, leak no acquisitions, and preserve pre-existing active branches.
- Active dynamic subscriptions are entity-owned.
- Same-identity dynamic emissions do not leak.
- Owner deactivation drains active subscriptions and owner despawn cleans any final active subscriptions.
- Builder injection cannot discard existing handles.
- Cross-world graph operations fail before mutation.
- Parallel lifecycle tests pass.
- Existing output transport behavior remains unchanged pending move-first work.
