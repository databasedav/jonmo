# Move-First Signal Delivery and Explicit Broadcasting

## Status

Proposed implementation plan for issue 3.

## Prerequisites and landing order

This plan assumes the following plans have already landed:

1. [`world-local-cleanup-plan.md`](world-local-cleanup-plan.md)
2. [`collection-correctness-plan.md`](collection-correctness-plan.md)
3. [`registration-lease-plan.md`](registration-lease-plan.md)

Required supplied infrastructure:

- One root lifecycle inbox per world, explicit local causal cleanup worklists, and reliable parallel tests.
- World-aware `SignalSystem` identities.
- Non-cloneable exact `SignalHandle` leases.
- Cached lazy node/edge registration blueprints with activation/deactivation.
- Transactional edge-batch preflight and rollback.
- Entity-owned dynamic subscription leases.
- Dynamic combinators that register inner descriptions once and poll registered terminals directly.
- Authoritative empty replay, versioned per-subscriber replay cursors, and collection oracle tests.

This plan extends those foundations; it does not reimplement them.

Recommended repository-level order:

```text
world-local cleanup
    -> collection semantic fixes
    -> registration leases
    -> move-first transport and explicit broadcast
```

This document describes a refactor of jonmo's signal graph from blanket clone-based transport to move-first transport with explicit broadcast boundaries. The primary motivating case is allowing `Builder` to remain a one-shot, move-only entity recipe while preserving the current declarative APIs:

```rust
Builder::new().child_signal(
    state.map_in(|state| state.visible.then(|| build_panel(state))),
)
```

and:

```rust
Builder::new().children_signal_vec(
    items.signal_vec().map_in(item_view),
)
```

The design is intentionally broader than a builder-specific workaround. It makes ownership and fan-out explicit throughout the graph and restricts `Clone` bounds to combinators whose semantics actually require cloning.

## Decision summary

Adopt these rules:

1. Signal outputs are move-only by default.
2. Ordinary public signal descriptions are move-only and do not implement `Clone`.
3. A move-only signal node may have at most one distinct downstream edge.
4. `.broadcast()` consumes a linear description and returns the only generally cloneable public signal-description capability.
5. The broadcast wrapper inserts an identity node whose output may be cloned to multiple downstreams.
6. `.broadcast()` requires the signal item to implement `Clone`.
7. Every combinator consumes a description and produces a new linear description unless the combinator is itself `.broadcast()`.
8. Fan-out violations are detected while creating the second downstream edge, not when a value is emitted.
9. All graph execution paths use one centralized forwarding implementation.
10. `Builder` no longer implements `Clone`.
11. Existing `child_signal` and `children_signal_vec` APIs remain builder-valued; no `_with` replacement is required.
12. Stateful combinators retain `Clone` bounds where retaining and emitting independent owned values genuinely requires them.
13. Private shared identities such as `LazySignal` may remain internally cloneable without exposing `Clone` on ordinary public descriptions.

The intended graph shape for multiple builder consumers is:

```text
                             +-> map(build_view) -> Builder A -> child A
cloneable state -> broadcast |
                             +-> map(build_view) -> Builder B -> child B
```

The graph must not attempt to produce this invalid shape:

```text
state -> map(build_view) -> one move-only Builder -> multiple children
```

## Goals

- Preserve the current fluent builder-facing API.
- Make `Builder` honestly move-only.
- Permit scalar signals, signal vecs, and signal maps to carry move-only outputs through linear chains.
- Make value duplication explicit and visible in graph topology.
- Encode ordinary linearity and explicit broadcast shareability in public description types.
- Prevent ordinary accidental branching at compile time, with runtime validation as a defensive invariant.
- Remove cloning requirements caused only by graph transport.
- Preserve deterministic graph execution.
- Keep `no_std` compatibility.
- Avoid specialization, unsafe code, and type-detection tricks.
- Stage the work so each phase can be reviewed and tested independently.

## Non-goals

This refactor does not own:

- World-local cleanup routing, supplied by `world-local-cleanup-plan.md`.
- Registration counts, handle linearity, cached node/edge blueprints, transaction rollback, or dynamic lease ownership, supplied by `registration-lease-plan.md`.
- Known sort, empty replacement, and empty replay/switch semantics, supplied by `collection-correctness-plan.md`.
- Cross-schedule delivery semantics.
- Changing polling from re-evaluation to cached-value lookup.
- Making public signal registration broadly fallible.

Move-first work may extend existing transaction validation with delivery/fan-out rules and may mechanically adapt replay/dynamic code to move-only values, but it must preserve the prerequisite ownership and collection invariants.

## Cross-plan contracts

### World-local cleanup contract

This plan may rely on external cleanup roots being routed to the originating world, causal in-world follow-ups settling through the invocation-local worklist, and normal parallel test execution. It must not add process-global lifecycle state, plugin-time queue clearing, retry-wave constants, or transport cleanup follow-ups back through the concurrent inbox.

### Registration lease contract

This plan may rely on complete cached node/edge registration blueprints, active-edge counts, linear handles, world-aware identities, transactional activation batches, and entity-owned dynamic subscriptions. Broadcast registration composes one upstream blueprint with one broadcast-node and connecting-edge acquisition; cloning a broadcast description reacquires/reactivates that cached blueprint.

### Collection correctness contract

This plan may change collection value transport and clone bounds, but applying emitted diffs must continue to reconstruct exact state. In particular it must preserve:

- Correct forward sort indices.
- Stable equal-key source moves.
- Empty `map_value_signal` replacements.
- Authoritative empty vec/map replay.
- Non-empty to empty switch behavior.
- Versioned pending diffs and per-subscriber replay cursors that prevent snapshot overlap.

The complete oracle suite from `collection-correctness-plan.md` is required after every collection/replay phase.

## Current constraints

### Clone-based erased transport

The current graph erases outputs as `Box<dyn AnyClone>` in `src/graph.rs`. `AnyClone` requires every output to implement `Clone`, even when the node has only one downstream and no clone occurs at runtime.

Affected core concepts include:

- `SystemRunner`
- `Runnable`
- `SystemHolder`
- `SignalInputBuffer`
- `trigger_signal_subgraph`
- `poll_signal_one_shot`
- `poll_signal`
- `downcast_any_clone`

### Clone bounds propagate into all constructors and combinators

`from_system`, `from_function`, `map`, collection mapping, and related methods inherit `Clone` requirements from the erased graph transport rather than their own behavior.

### Public description cloning conflates capabilities

Most concrete signal descriptions currently implement `Clone` by cloning `LazySignal` identities and recursively cloning upstream descriptions. This conflates several different operations:

- Sharing graph identity.
- Creating another subscription.
- Re-registering an existing chain.
- Permitting output fan-out.
- Duplicating registration ownership.

It also makes invalid move-only fan-out representable through the ordinary API and contributes to unbalanced registration counts when recursively cloned chains are registered more than once.

The target model keeps private `LazySignal` identity cloneable but removes `Clone` from ordinary public descriptions. Only an explicit broadcast wrapper exposes cloneable description semantics.

### Builder is a one-shot recipe

`Builder` stores `FnOnce` callbacks that consume captured bundles, signals, observers, tasks, and child builders. A general, independently spawnable clone cannot be produced from these opaque captures.

The current `Clone` implementation panics in debug and returns an empty builder in release. This violates the semantic expectations of generic code and causes build-profile-dependent behavior.

## Target user-facing behavior

### Linear builder signal

This remains valid and unchanged:

```rust
let child = state.map_in(|state| {
    state.visible.then(|| {
        Builder::new()
            .insert(Node::default())
            .child(Builder::from(Text::new(state.label)))
    })
});

Builder::new().child_signal(child);
```

The `Option<Builder>` value moves through the final signal edge and is consumed by `child_signal`.

### Linear builder signal vec

This remains valid and unchanged:

```rust
Builder::new().children_signal_vec(
    items.signal_vec().map_in(item_view),
);
```

`VecDiff<Builder>` is moved into `children_signal_vec`. `VecDiff<T>` does not need to implement `Clone` merely to travel through the graph.

### Explicit scalar fan-out

```rust
let shared = state_signal.broadcast();

let left = shared.clone().map_in(render_left);
let right = shared.map_in(render_right);
```

### Correct builder fan-out

```rust
let shared_state = state_signal.broadcast();

let left_view = shared_state.clone().map_in(build_view);
let right_view = shared_state.map_in(build_view);

Builder::new()
    .child_signal(left_view)
    .child_signal(right_view);
```

Each branch independently constructs a fresh `Builder`.

### Invalid builder fan-out

This should fail at compile time because the ordinary mapped description is linear and does not implement `Clone`:

```rust,compile_fail
let view = state_signal.map_in(build_view);

Builder::new()
    .child_signal(view.clone())
    .child_signal(view);
```

Runtime fan-out validation remains necessary for custom signal implementations, dynamic/internal graph construction, and direct low-level edge creation. Its diagnostic should still explain that move-only state must be branched before producing `Builder`.

### Invalid builder broadcast

This should fail to compile because `Builder: Clone` is no longer implemented:

```rust,compile_fail
let view = state_signal.map_in(build_view).broadcast();
```

## Runtime design

### Erased value type

Replace clone-constrained erased values with move-only erased values:

```rust
type ErasedSignalValue = Box<dyn Any + Send + Sync>;
```

If a public alias is desirable:

```rust
pub type BoxedSignalValue = Box<dyn Any + Send + Sync>;
```

`dyn-clone` is no longer used for ordinary boxed signal descriptions or emitted value transport. It may remain narrowly for a sealed boxed broadcast-description capability if a concrete use case requires heterogeneous cloneable broadcast wrappers.

### System runner

Change `Runnable` and `SystemRunner` to return move-only erased outputs:

```rust
trait Runnable: Send + Sync {
    fn run(
        &self,
        world: &mut World,
        input: ErasedSignalValue,
    ) -> Option<ErasedSignalValue>;
}
```

Relax the output bound:

```rust
impl<I, O, S> Runnable for SystemHolder<I, O, S>
where
    I: 'static,
    O: Send + Sync + 'static,
    S: Into<Option<O>> + 'static,
{
    // ...
}
```

The input still downcasts to owned `I`, and the output is boxed without cloning.

### Input buffers

Change signal input storage to:

```rust
#[derive(Component, Default)]
pub(crate) struct SignalInputBuffer(Vec<ErasedSignalValue>);
```

Its operations remain move-based:

```rust
impl SignalInputBuffer {
    fn take(&mut self) -> Vec<ErasedSignalValue>;
    fn push(&mut self, value: ErasedSignalValue);
    fn clear(&mut self);
}
```

### Output delivery policy

Attach delivery metadata to each signal-system entity:

```rust
type CloneErasedSignalValue =
    dyn Fn(&(dyn Any + Send + Sync)) -> ErasedSignalValue + Send + Sync;

#[derive(Component, Clone)]
pub(crate) enum OutputDelivery {
    Move,
    Broadcast(Arc<CloneErasedSignalValue>),
}
```

Helper constructors:

```rust
impl OutputDelivery {
    fn move_only() -> Self {
        Self::Move
    }

    fn broadcast<T>() -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        Self::Broadcast(Arc::new(|value| {
            let value = value
                .downcast_ref::<T>()
                .expect("signal output type mismatch during broadcast");
            Box::new(value.clone())
        }))
    }

    fn can_broadcast(&self) -> bool {
        matches!(self, Self::Broadcast(_))
    }
}
```

The exact trait-object syntax should be validated against the supported Rust toolchain. A small private trait can replace the closure type if it produces clearer compiler errors.

### Signal registration primitives

Use one private implementation function with explicit delivery metadata, but do not expose it outside the smallest possible module scope:

```rust
fn spawn_signal_with_delivery<I, O, IOO, F, M>(
    world: &mut World,
    system: F,
    delivery: OutputDelivery,
) -> SignalSystem
where
    I: 'static,
    O: Send + Sync + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<I>, IOO, M> + Send + Sync + 'static;
```

The raw helper is not type-coupled strongly enough to prevent attaching `OutputDelivery::broadcast::<WrongType>()` to an `O` output. All callers must go through typed wrappers; alternatively, store `TypeId::of::<O>()` in broadcast metadata and verify it when installing the component as an additional invariant check.

Provide focused, type-safe wrappers:

```rust
fn spawn_move_signal<I, O, IOO, F, M>(...) -> SignalSystem
where
    O: Send + Sync + 'static;

fn spawn_broadcast_signal<I, O, IOO, F, M>(...) -> SignalSystem
where
    O: Clone + Send + Sync + 'static;
```

Likewise provide lazy wrappers:

```rust
fn lazy_move_signal_from_system(...);
fn lazy_broadcast_signal_from_system(...);
```

Do not overload a single helper with implicit clone detection. Stable Rust cannot reliably select cloneable versus move-only behavior without specialization or awkward API machinery.

### Default delivery policy

In the final design, ordinary constructors and combinators create move-only output nodes unless their documented purpose is explicitly broadcasting.

This produces one consistent public rule:

- A normal node may feed one downstream.
- A `.broadcast()` node may feed multiple downstreams.

Some internal source nodes are intrinsically shared and must be registered as broadcast-capable even without a user-visible `.broadcast()` call. In particular, every `MutableVec::signal_vec()` call reuses the same `MutableVecData::broadcaster` and adds a distinct replay-node downstream. `MutableBTreeMap` similarly shares one broadcaster across `signal_map`, `signal_vec_keys`, and `signal_vec_entries`. User-side `.broadcast()` cannot repair fan-out that occurs before each returned replay node.

Therefore:

- Shared mutable-source broadcaster systems use `OutputDelivery::Broadcast` internally.
- Each returned replay node remains move-only unless the user explicitly calls `.broadcast()` on it.
- Tests must cover multiple independent subscriptions to one mutable source, including mixed map/map-keys/map-entries subscriptions.

During migration, other existing clone-constrained nodes may temporarily be registered as broadcast-capable to preserve behavior while the forwarding infrastructure is introduced. The final phase should remove that compatibility behavior.

## Public description capability model

### Linear descriptions

Ordinary public descriptions such as `Source`, `Map`, `Filter`, `Dedupe`, scheduling wrappers, and collection wrappers should not implement `Clone` merely because their private `LazySignal` field is cloneable. Extension methods consume `self`, so normal linear composition remains unchanged:

```rust
source
    .map_in(first)
    .filter_in(predicate)
    .map_in(second)
```

A custom external signal type may still implement `Clone`, so runtime delivery-policy and edge validation remain mandatory. The crate itself should not advertise ordinary description cloning as a supported branching mechanism.

### Internal lazy identities

`LazySignal` remains a private `Arc`-backed identity and may implement `Clone`. Internal code can share a lazily registered node without exposing that capability on every public wrapper. Public cloneability and private identity sharing are deliberately separate.

### Broadcast descriptions

Only explicit broadcast wrappers generally implement public `Clone`. Their clone semantics are precise:

> Create another subscription starting at this explicitly shareable graph boundary.

Cloning a broadcast description shares the broadcast node identity; it does not recursively clone the upstream description and does not duplicate an emitted value until the node actually fans out.

Every combinator called after `.broadcast()` consumes one broadcast clone and produces a new linear output description. To branch after an expensive transformation, place `.broadcast()` after that transformation.

## Broadcast combinators

### Scalar signals

Add to `SignalExt`:

```rust
fn broadcast(self) -> Broadcast<Self::Item>
where
    Self: Sized,
    Self::Item: Clone + Send + Sync + 'static;
```

`Broadcast<T>` contains only a shared lazy identity and item marker:

```rust
pub struct Broadcast<T> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Clone> Clone for Broadcast<T> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}
```

The upstream description is moved into the `FnOnce` lazy registration closure when `.broadcast()` is constructed; it is not stored as a recursively cloneable public field. The first broadcast registration consumes and materializes that upstream. Later clones do not rerun recursive initializers or rebuild topology, but each clone does reacquire/reactivate every node and edge acquisition in the cached broadcast blueprint, including the upstream chain.

The identity system moves the value through:

```rust
|In(value): In<T>| value
```

Its graph shape is:

```text
upstream --move--> broadcast node --clone as needed--> downstreams
```

Do not mutate the upstream node's delivery policy. A dedicated node provides a stable ownership boundary, prevents recursive description cloning, and gives registration accounting one shared boundary.

### Signal vecs

Add to `SignalVecExt`:

```rust
fn broadcast(self) -> BroadcastVec<Self::Item>
where
    Self: Sized,
    Self::Item: Clone + Send + Sync + 'static;
```

The actual emitted type is `Vec<VecDiff<Self::Item>>`, which is cloneable when `Self::Item: Clone`.

The wrapper should be implemented as an identity `SignalVec` node with broadcast delivery metadata. Like scalar `Broadcast<T>`, `BroadcastVec<T>` stores only the shared lazy broadcast identity and does not recursively retain a cloneable upstream description. Its `Clone` implementation must be constrained by `T: Clone`, matching the capability exposed by its constructor.

### Signal maps

Add to `SignalMapExt`:

```rust
fn broadcast(self) -> BroadcastMap<Self::Key, Self::Value>
where
    Self: Sized,
    Self::Key: Clone + Send + Sync + 'static,
    Self::Value: Clone + Send + Sync + 'static;
```

The emitted `Vec<MapDiff<K, V>>` is cloneable under those bounds. `BroadcastMap<K, V>` stores only its shared lazy broadcast identity and key/value markers. Its `Clone` implementation must require `K: Clone` and `V: Clone`.

### Naming

Use `.broadcast()` consistently across scalar, vec, and map extension traits.

Avoid `.share()` because FRP libraries often use “share” to mean sharing a subscription or cached evaluation rather than cloning emitted values. “Broadcast” describes the graph operation directly.

## Edge validation

### Validation point

Extend the lease plan's `try_connect_edges` batch preflight with fan-out validation before mutating any endpoint relationships.

For the complete proposed edge batch:

1. Preserve duplicate edge acquisition multiplicity.
2. Separate already-active topology edges from topology edges requiring installation; do not return early unless the entire batch has no remaining validation/activation work.
3. Reject cycles across all newly activated topology edges.
4. Calculate each source's active downstream set after the complete proposed batch.
5. If any active downstream count would exceed one, require `OutputDelivery::Broadcast`.
6. Only after full validation commit edge counts, `Downstream`, `Upstream`, schedule inheritance, and graph invalidation state.

`registration-lease-plan.md` supplies a fallible, atomic node/edge activation transaction and exact rollback of newly acquired leases. Extend edge preflight with delivery validation before committing edge-count, topology, schedule, or cache mutations. Node acquisitions may occur transiently before final edge validation, but failure must release them exactly.

Do not add a second connection path or move-first-specific rollback mechanism. Static and dynamic registration continue using the shared lease transaction. On `MoveOnlyFanout`, that transaction releases only the newly acquired leases and leaves existing branches unchanged.

### Duplicate edges

Because downstreams are stored in a `HashSet`, piping the same source-target pair more than once must not count as fan-out. Registration counts and duplicate graph edges are separate concepts.

### Error representation

Public registration may continue returning `SignalHandle`. The supplied internal edge transaction returns structured errors and rolls back before the public boundary converts `MoveOnlyFanout` into an actionable panic.

Optionally define an internal error now:

```rust
enum SignalGraphError {
    Cycle {
        source: SignalSystem,
        target: SignalSystem,
    },
    MoveOnlyFanout {
        source: SignalSystem,
        existing: Vec<SignalSystem>,
        attempted: SignalSystem,
    },
}
```

Even if converted to a panic at the public boundary, this keeps validation logic structured and prepares for fallible registration later.

### Runtime defense

The forwarding function must still reject move-only fan-out defensively. Graph relationships can be manipulated internally, and a clear invariant failure is preferable to dropping branches or corrupting input buffers.

## Centralized output forwarding

Introduce one forwarding function used by all execution paths:

```rust
fn forward_output(
    world: &mut World,
    source: SignalSystem,
    downstreams: &[SignalSystem],
    output: ErasedSignalValue,
) {
    match downstreams {
        [] => {}
        [downstream] => push_signal_input(world, *downstream, output),
        [rest @ .., last] => {
            let delivery = world
                .get::<OutputDelivery>(*source)
                .expect("signal is missing OutputDelivery");

            let OutputDelivery::Broadcast(clone_value) = delivery else {
                panic!(
                    "move-only signal {source:?} reached runtime with multiple downstreams"
                );
            };

            let clones = rest
                .iter()
                .map(|downstream| {
                    (*downstream, clone_value(output.as_ref()))
                })
                .collect::<Vec<_>>();

            for (downstream, value) in clones {
                push_signal_input(world, downstream, value);
            }

            push_signal_input(world, *last, output);
        }
    }
}
```

Clone values before mutably borrowing downstream entities. This avoids borrow conflicts between reading delivery metadata and writing input buffers.

### Deterministic downstream order

`Downstream` is currently a `HashSet`. Sort downstreams by entity index before forwarding so the recipient of the original moved value is deterministic. Values should be equivalent, but deterministic behavior improves reproducibility and diagnostics.

A helper should centralize sorting and liveness filtering:

```rust
fn get_live_downstreams_sorted(
    world: &World,
    signal: SignalSystem,
) -> Vec<SignalSystem>;
```

A live downstream requires all of:

- An existing entity with the input sink/component required by the current execution path.
- Positive node activation/registration count.
- A positive active edge-acquisition count from the source.

Determine the live set before:

- Validating fan-out count.
- Calculating clone count.
- Selecting the recipient of the original moved value.

At registration time, fan-out counts active edge acquisitions supplied by the lease graph. Inactive cached blueprint edges do not count. Any stale structural remnants are ignored during preflight without mutation; if pruning is needed, include it only in the final `EdgeCommit` after full validation. At runtime, downstreams removed re-entrantly during the same frame are skipped. The invariant “N live downstreams perform N - 1 clones” must not count inactive/stale IDs or entities missing `SignalInputBuffer`.

## Execution-path integration

### Normal graph processing

Update `run_signal_node` to:

1. Take owned inputs.
2. Run the node according to existing multi-input semantics.
3. Obtain sorted, live downstreams.
4. Call `forward_output`.

The move-first refactor should not silently alter the current “run once per input and forward one final output” policy. A separate correctness change may later decide whether the final invocation returning `None` should suppress an earlier `Some`.

### Polling

`poll_signal_one_shot` requires special handling.

It currently computes nodes upstream of the target but forwards intermediate outputs to all real downstreams, including nodes outside the polled subgraph. The refactor should restrict forwarding to downstreams in the target's reachable set:

```rust
let downstreams = get_downstreams_sorted(world, signal)
    .into_iter()
    .filter(|downstream| reachable.contains(downstream))
    .collect::<Vec<_>>();
```

Then call the same forwarding primitive or a common helper that writes to the poll-local input map.

A shared abstraction may accept an output sink:

```rust
trait SignalInputSink {
    fn push(&mut self, world: &mut World, target: SignalSystem, value: ErasedSignalValue);
}
```

However, avoid over-generalizing if two small helpers sharing the same delivery-policy logic are easier to review.

Polling must not require a broadcast merely because the source has unrelated downstreams outside the target's upstream subgraph.

### Synchronous subgraph triggering

The existing `trigger_signal_subgraph` clones one input across multiple seed signals. Replace it with a move-first primitive:

```rust
fn trigger_signal_subgraph(
    world: &mut World,
    signal: SignalSystem,
    input: ErasedSignalValue,
);
```

For multiple seeds, accept independently owned inputs:

```rust
fn trigger_signal_subgraphs(
    world: &mut World,
    seeds: impl IntoIterator<Item = (SignalSystem, ErasedSignalValue)>,
);
```

Optionally add a typed convenience helper for cloneable inputs:

```rust
fn trigger_signal_subgraphs_cloned<T>(
    world: &mut World,
    signals: &[SignalSystem],
    input: T,
)
where
    T: Clone + Send + Sync + 'static;
```

Audit all current call sites. Most appear to trigger a single seed with `()` or an owned diff batch and should migrate to the single-seed API.

`trigger_signal_subgraph` must use the same downstream delivery policy as scheduled graph processing.

### Replay paths

Vector and map replay currently invoke synchronous triggering. Replay values should move into the replay node. If a returned replay signal fans out externally, it must feed an explicit broadcast node.

The broadcaster upstream of replay is different: mutable vec/map source handles deliberately reuse one hidden broadcaster across multiple independently created replay nodes. Register those shared broadcaster systems as intrinsically `OutputDelivery::Broadcast`; otherwise the second call to `signal_vec()`, `signal_map()`, `signal_vec_keys()`, or `signal_vec_entries()` can fail before user code has any opportunity to call `.broadcast()`.

Do not retain any other hidden cloning in replay-specific code.

### Dynamic combinators

Audit at minimum:

- `flatten`
- `switch`
- `switch_signal_vec`
- `switch_signal_map`
- `SignalVecExt::map_signal`
- `SignalVecExt::filter_signal`
- `SignalMapExt::map_value_signal`

`registration-lease-plan.md` has already converted these paths to consume each emitted inner description once, poll registered terminals directly, connect processors transactionally, and store active leases in entity-owned components.

Move-first work audits only delivery capability and bounds:

1. Mark dynamic forwarder outputs move-only unless they intentionally fan out.
2. Route manually triggered output through centralized delivery-policy forwarding.
3. Remove item/output `Clone` bounds that existed only for erased transport.
4. Preserve intrinsic bounds needed for retained initial values, replay, or explicit broadcast.
5. Verify same-identity replacement and owner-despawn lease tests remain green.
6. Verify collection semantic tests from `collection-correctness-plan.md` remain green.

Do not reintroduce inner-description cloning, identity-only registration, temporary polling edges, or `Local`-only active handle ownership.

### Moving descriptions and handles into closures

Distinguish three ownership cases.

#### One-shot registration closures

Moving a linear signal description into a `FnOnce` closure is correct and is the normal implementation technique for lazy combinators:

```rust
let signal = LazySignal::new(move |world| {
    let upstream = upstream_description.register(world);
    // ...
});
```

The closure runs once, so the captured description is consumed exactly once.

#### Reusable `FnMut` system closures

A Bevy system or mapping closure may run repeatedly. It cannot move the same captured linear description out on every invocation. Choose the semantics explicitly:

- Create a fresh description each invocation with a factory function.
- Capture an explicit `Broadcast` and clone that shareable description each invocation.
- For a genuinely one-shot emission, capture `Option<S>` and use `take()`, making later invocations return `None`.

Examples:

```rust
// Fresh signal each invocation.
outer.map(move |In(value)| make_inner_signal(value)).flatten()
```

```rust
// Reuse one explicitly shareable inner signal.
let inner = inner.broadcast();
outer.map(move |In(_)| inner.clone()).flatten()
```

```rust
// Move one linear signal exactly once.
let mut inner = Some(inner);
outer
    .map(move |In(_)| inner.take())
    .flatten()
```

`map` already treats `Option<S>` as conditional output with item type `S`, so no separate scalar `filter_map` combinator is required.

#### Registered `SignalHandle` cleanup tokens

`registration-lease-plan.md` has already made `SignalHandle` a non-cloneable, world-bound cleanup lease and moved active dynamic leases into entity-owned components.

The move-first description API must preserve that distinction:

- A broadcast description may be cloned to acquire another registration lease.
- A `SignalHandle` itself is never cloned.
- A copied `SignalSystem` remains identity only.
- Reusable closures clone `Broadcast` descriptions, not cleanup handles.
- One-shot cleanup closure captures may use `Option<SignalHandle>::take()`, but framework-managed active leases remain component-owned.

Move-first tests should assert that broadcast cloning acquires independent leases through the existing cached blueprint and that fan-out failure uses existing transactional rollback.

## Public polling API

### Return type

Change:

```rust
pub fn poll_signal(...) -> Option<Box<dyn AnyClone>>;
```

to:

```rust
pub fn poll_signal(...) -> Option<BoxedSignalValue>;
```

Add:

```rust
pub fn downcast_signal_value<T: 'static>(
    value: BoxedSignalValue,
) -> Option<T>;
```

### Compatibility

`AnyClone` is currently public. For a breaking release, prefer:

- Introduce `BoxedSignalValue` and `downcast_signal_value`.
- Decide explicitly whether `downcast_any_clone` is removed or changed to accept the new erased value type.
- Deprecate `AnyClone` and remove it once downstream users have migrated.

Changing `downcast_any_clone` to accept `BoxedSignalValue` is still a breaking signature change for callers that construct or pass their own `Box<dyn AnyClone>`; Rust cannot retain two functions with the same name and different erased argument types. Do not claim source compatibility for that wrapper. If this work lands in a breaking release, prefer the clean new API and a migration note.

If the move-first work is released only in a major/minor version already permitted to break API compatibility, remove `AnyClone` directly and document the migration.

## Combinator bound audit

Do not remove bounds mechanically. For every constructor and combinator, ask:

1. Does it merely transform and forward ownership?
2. Does it retain a value while also emitting an independent owned value?
3. Does it emit the same captured value more than once?
4. Does it replay retained state?
5. Does it branch internally?

### Likely move-only scalar operations

These should generally not require cloneable outputs:

- `from_system`
- `from_function`
- `once`
- `map`
- `map_in`
- `map_in_ref` output
- `take`
- `first`
- Boolean/option mapping outputs
- Component/entity mapping outputs where values are freshly produced
- Terminal task/registration methods
- Builder terminal consumers: `on_signal`, `on_signal_with_entity`, `on_signal_with_component`, and `component_signal`

`once` should store `Option<T>` and use `take()` rather than cloning its capture.

### Operations that likely retain `Clone` bounds

These may intrinsically need cloning, depending on their exact implementation:

- `always`
- `dedupe`
- `zip`
- `combine`-style operators
- Poll-and-forward dynamic operators
- Replayable sources
- Explicit `.broadcast()`
- Operations that keep the latest value while emitting an owned copy

Where possible, distinguish cloning input from cloning output. A mapping system may require a cloneable input due to internal retention while still producing a move-only output. For example, `filter` currently gives an owned item to its predicate and must still forward the accepted item, so it may retain an intrinsic `Clone` requirement unless its predicate API changes.

### Signal vecs

Key expectations:

- `VecDiff<T>` remains conditionally `Clone`; it is not required to be cloneable for linear transport.
- `SignalVecExt::map` should allow move-only output items if the mapped diffs are forwarded linearly.
- Mutable vector sources may retain `T: Clone` because they own current state and emit/replay independent diff values.
- Stateful vector combinators may retain clone bounds according to their own materialization needs.
- `SignalVecExt::broadcast()` requires `Item: Clone`.

### Signal maps

Key expectations:

- `MapDiff<K, V>` remains conditionally `Clone`.
- `map_value` should allow move-only mapped values when no retained copy is required, and its unchanged keys should be moved through `MapDiff::map_value` rather than requiring `Self::Key: Clone` solely for transport.
- Mutable map sources may retain key/value clone bounds for diff emission and replay.
- `SignalMapExt::broadcast()` requires cloneable keys and values.

### Boxed signals

Change the default boxed description to move-only:

```rust
pub type BoxedSignal<T> =
    Box<dyn Signal<Item = T> + Send + Sync>;
```

Likewise, default boxed signal vec/map descriptions should be move-only if public aliases are provided.

Remove the blanket `SignalDynClone` model from ordinary descriptions. If heterogeneous cloneable broadcast descriptions are demonstrably needed, introduce a separate sealed capability such as `BoxedBroadcastSignal<T>`. Only explicit broadcast wrappers should implement that sealed trait; do not use `impl<S: Signal + Clone> BroadcastSignal for S`, which would recreate the original ambiguity.

Prefer deferring boxed broadcast trait objects until a real call site requires them. Concrete `Broadcast<T>`, `BroadcastVec<T>`, and `BroadcastMap<K, V>` wrappers may be sufficient.

## Builder changes

### Remove invalid clone implementation

Delete `impl Clone for Builder` and its warning documentation.

`Builder` should remain:

```rust
#[derive(Default)]
pub struct Builder {
    on_spawns: Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>,
    next_block: usize,
}
```

Changing `AtomicUsize` to `usize` is optional but recommended. Builder methods consume `self`, so block assignment does not require shared atomic mutation.

### Preserve builder APIs

Keep:

```rust
pub fn child_signal(
    self,
    child_option: impl Signal<Item = Option<Builder>>,
) -> Self;

pub fn children_signal_vec(
    self,
    children: impl SignalVec<Item = Builder>,
) -> Self;
```

Their terminal systems consume builders immediately and return `()`, which is trivially movable and cloneable if later required.

Audit all builder terminal methods for transport-only bounds. `on_signal`, `on_signal_with_entity`, and `on_signal_with_component` consume their inputs and should accept non-`Clone` values when the upstream chain is linear. `component_signal` similarly consumes the component value; retain the required `Component`, `Send`, `Sync`, and `'static` constraints, but remove `Clone` when it is not semantically needed.

### Builder documentation

Document:

- `Builder` is a one-shot entity recipe.
- Builder-valued signals are valid in linear chains.
- Builder-valued signals cannot be broadcast.
- To render the same state into multiple entity branches, broadcast state before mapping it to builders.

Include an explicit branching example.

## Migration strategy

This is a breaking semantic change and should be released accordingly.

### Existing linear code

Expected to continue compiling after unrelated blanket `Clone` bounds are removed:

```rust
signal.map_in(build_view)
```

### Existing direct builder clones

Will stop compiling. This is intentional.

Replace factory-like builder cloning with functions:

```rust
fn build_view() -> Builder {
    // ...
}
```

### Existing signal fan-out

Ordinary descriptions no longer implement `Clone`. Code that clones a signal description to create multiple downstream edges must add `.broadcast()` at or before the branch:

Before:

```rust
let signal = source.map_in(transform);
let a = signal.clone().map_in(a);
let b = signal.map_in(b);
```

After:

```rust
let signal = source.map_in(transform).broadcast();
let a = signal.clone().map_in(a);
let b = signal.map_in(b);
```

### Reusable closure captures

Code that clones a captured inner signal from a repeatedly invoked closure must choose one of three migrations:

- Return a freshly constructed signal each invocation.
- Broadcast the inner signal first and clone the resulting `Broadcast` description.
- Store a linear signal in `Option` and `take()` it for one-shot behavior.

Do not replace description cloning with `SignalHandle` cloning. Handles are cleanup tokens and require explicit lifecycle ownership.

### Existing builder fan-out

Broadcast cloneable state before producing builders:

```rust
let state = state.broadcast();
let a = state.clone().map_in(build_view);
let b = state.map_in(build_view);
```

### Changelog guidance

Explain the change as an ownership correction and capability expansion:

- Signals now support move-only values in linear chains.
- Fan-out is explicit through `.broadcast()`.
- `Builder` no longer has an invalid `Clone` implementation.
- Some previous implicit branches require `.broadcast()`.
- `once` now supports move-only values.

## Implementation phases

### Phase 0: Verify prerequisite baselines

Before changing transport, require all prerequisite acceptance suites:

- World-local root cleanup, monotonic local causal worklists, and normal parallel test execution.
- Exact registration leases, transactional edge batches, and dynamic owner cleanup.
- Collection sort, empty replacement, authoritative replay, switch, and oracle tests.

Then add move-first characterization tests for:

- Linear scalar propagation.
- Multi-level and multi-input processing.
- Existing cloneable fan-out behavior.
- Polling through a branched graph.
- Synchronous trigger propagation.
- Builder child replacement and collection diffs.

Add compile-fail examples describing desired move-only descriptions and builder values. Do not proceed if prerequisite lifecycle or collection tests are red.

### Phase 1: Centralize current forwarding

Without changing `AnyClone` or public bounds:

1. Add sorted downstream retrieval.
2. Extract normal output forwarding into one helper.
3. Extract poll-local forwarding policy into a closely related helper.
4. Route `run_signal_node`, polling, and trigger paths through those helpers.
5. Restrict polling propagation to the target's reachable subgraph.

Acceptance criteria:

- Normal parallel tests and all prerequisite cleanup/lease/collection suites pass.
- Existing fan-out behavior is unchanged.
- No bounds have changed.

### Phase 2: Introduce delivery metadata while preserving clone behavior

1. Add `OutputDelivery` to signal-system entities.
2. Register all existing nodes as `Broadcast` temporarily because current bounds guarantee `Clone`.
3. Add defensive policy checks to centralized forwarding.
4. Validate `OutputDelivery` presence in tests.

Acceptance criteria:

- Existing fan-out tests pass.
- Clone counts match previous behavior.
- Missing metadata fails with a clear invariant message.

### Phase 3: Replace erased clone and trigger transport

1. Replace `Box<dyn AnyClone>` with `ErasedSignalValue` internally.
2. Remove `DynClone` from emitted-value transport.
3. Keep existing public `Clone` bounds temporarily and use delivery metadata for fan-out cloning.
4. Replace the clone-based multi-seed `trigger_signal_subgraph` API with the owned single-seed and `(seed, value)` multi-seed forms.
5. Migrate all trigger call sites; do not defer this because erased trigger input cloning becomes impossible once `AnyClone` is removed.
6. Introduce `BoxedSignalValue` and `downcast_signal_value` in this phase.
7. Change `poll_signal` to return the new erased value type.
8. Decide and implement the breaking removal/signature migration of `downcast_any_clone`; a non-generic polling API cannot reconstruct an `AnyClone` trait object after its clone vtable has been erased.

Acceptance criteria:

- Existing public behavior remains unchanged.
- Runtime, trigger transport, and public polling no longer require or return `AnyClone`.
- `BoxedSignalValue` and `downcast_signal_value` are available with documented migration from the old polling API.
- All execution paths use delivery metadata when cloning.
- Every synchronous trigger call site passes one owned value per seed.

### Phase 4: Add move-only registration and `.broadcast()`

1. Add move-only and broadcast node registration helpers that return existing exact leases.
2. Add scalar, vec, and map `.broadcast()` wrappers backed by cached node/edge registration blueprints.
3. Extend the existing fallible edge-batch preflight with registration-time fan-out validation.
4. Rely on the existing transaction to rollback newly acquired leases on fan-out failure.
5. Mark shared mutable vec/map broadcaster systems as intrinsically broadcast-capable.
6. Keep other legacy constructors broadcast-capable temporarily if needed for incremental migration.

Acceptance criteria:

- A dedicated move-only test value traverses a linear graph.
- A second downstream on a move-only node fails during registration.
- Broadcasting the same test value supports multiple downstreams.
- Duplicate source-target piping does not falsely trigger fan-out.
- Failed static and dynamic fan-out installs no partial edge mutation, leaks no acquisition, and leaves pre-existing active branches unchanged; newly materialized dormant definitions may remain when still owned by live descriptions.
- Two replay subscriptions to one mutable vec work without a user-visible broadcast.
- `signal_map`, `signal_vec_keys`, and `signal_vec_entries` can coexist on one mutable map.

### Phase 5: Prepare move-only public descriptions

The lease plan has already made handles linear and dynamic descriptions single-consumption. This phase changes only description capabilities:

1. Inventory every public scalar/vec/map `Clone` implementation and classify it as ordinary linear or explicit broadcast.
2. Change ordinary boxed scalar/vec/map descriptions to move-only trait objects.
3. Refactor `signal::option` to preserve the existing single-consumption dynamic registration path while removing its description-level cloneable return promise.
4. Add a sealed boxed broadcast capability only if a current heterogeneous use case requires it.
5. Verify reusable closures can return fresh descriptions, explicit broadcast clones, or one-shot `Option::take()` descriptions.
6. Verify all registration-lease dynamic cleanup tests remain unchanged.

Acceptance criteria:

- Ordinary boxed descriptions are consumed once.
- `signal::option(Some(move_only_boxed_signal)).flatten()` works without description cloning.
- Reusable closure examples express fresh, broadcast, or one-shot semantics explicitly.
- No handle or active-subscription ownership code is duplicated in this phase.

### Phase 6: Make scalar descriptions linear and relax scalar bounds

1. Remove `Clone` implementations from ordinary scalar public descriptions.
2. Retain `Clone` only on explicit scalar `Broadcast` and private internal identities.
3. Audit and relax scalar constructor/combinator bounds, beginning with:
   - `from_system`
   - `from_function`
   - `once`
   - `map`
   - `map_in`
   - `map_in_ref`
   - `take`
   - `first`
   - Builder terminal consumers whose inputs are only consumed
4. Keep explicit bounds where stateful semantics require them.

Acceptance criteria:

- `once(NonClone)` works.
- `from_system` can emit `NonClone`.
- `map` can produce `NonClone`.
- An ordinary scalar description cannot be cloned.
- `Broadcast<T>` is cloneable when `T: Clone`.
- Every combinator after a broadcast produces a new linear description.
- Existing scalar examples compile after adding explicit broadcasts where necessary.

### Phase 7: Make collection descriptions linear and remove `Builder: Clone`

1. Remove `Clone` implementations from ordinary signal vec/map public descriptions while retaining it on explicit broadcast wrappers and private identities.
2. Relax at least `SignalVecExt::map`, `map_in`, and `map_in_ref` output bounds so they can produce move-only items.
3. Verify the existing builder-valued signal-vec tests compile under those relaxed bounds.
4. Delete the builder clone implementation.
5. Remove transport-only `Clone` bounds from builder terminal consumers.
6. Optionally simplify the block counter to `usize`.
7. Update docs and examples.
8. Add builder-valued scalar, vec, non-`Clone` terminal, and description-linearity tests.

Acceptance criteria:

- Existing `child_signal(signal.map(...Builder...))` syntax works.
- Existing `children_signal_vec(signal_vec.map(...Builder...))` syntax works.
- Debug and release behavior is identical.
- Broadcasting a builder-valued signal fails to compile.
- Branching state before builder mapping produces independent entity trees.

### Phase 8: Complete signal vec and signal map bound audit

1. Audit the remaining collection combinators beyond the builder-critical mappings completed in Phase 7.
2. Permit move-only mapped output items where state retention does not require cloning.
3. Remove clone-only transport bounds from unchanged map keys where values are transformed by move.
4. Ensure replay sources retain only their intrinsic clone bounds.
5. Add explicit vec/map broadcast tests.

Acceptance criteria:

- `SignalVec<Item = Builder>` works linearly.
- `SignalMap<Value = NonClone>` works through supported linear mappings.
- Collection fan-out requires `.broadcast()`.
- All diff variants preserve ownership and ordering.

### Phase 9: Finalize dynamic, boxed, and public polling APIs

1. Verify dynamic edges obey fan-out validation and transactional rollback after ordinary description cloning is removed.
2. Verify all active dynamic handles are entity-owned and cleaned on replacement/despawn.
3. Remove any residual deprecated `AnyClone` compatibility surface left after the Phase 3 polling migration.
4. Remove ordinary `SignalDynClone`/boxed-clone APIs.
5. Add sealed boxed broadcast APIs only where required.
6. Update crate-level architecture documentation.

Acceptance criteria:

- Dynamic switching works with linear non-cloneable descriptions and outputs where semantically valid.
- Polling does not require broadcast because of unrelated downstreams.
- No `AnyClone` remains in runtime transport or public polling.
- No ordinary public boxed description promises cloneability.
- Public migration documentation is complete.

### Phase 10: Performance and cleanup

1. Remove now-unused emitted-value and ordinary-description `dyn-clone` code.
2. Retain `dyn-clone` only if a sealed heterogeneous boxed broadcast capability demonstrably needs it.
3. Benchmark linear and broadcast delivery.
4. Remove compatibility registration paths.
5. Tighten diagnostics and internal invariants.

## Test plan

### Core move semantics

Use a deliberately non-cloneable value:

```rust
#[derive(Debug, PartialEq, Eq)]
struct NonClone(String);
```

Test:

- Source to terminal consumer.
- Source through several maps.
- `once(NonClone)`.
- `from_function` producing a fresh `NonClone` each run.
- `map` producing `NonClone` from cloneable input.
- `Option<NonClone>` propagation.
- No output and filtered-output behavior.

### Description capabilities

Test and document:

- Ordinary scalar, vec, and map descriptions do not implement `Clone`.
- Private `LazySignal` identity remains internally cloneable.
- `Broadcast<T>`, `BroadcastVec<T>`, and `BroadcastMap<K, V>` are cloneable under their item bounds.
- Cloning a broadcast boundary does not recursively re-register its upstream chain.
- A combinator consuming a broadcast clone produces a new linear description.
- Ordinary boxed descriptions are move-only.
- Any boxed cloneable broadcast capability is sealed to explicit broadcast wrappers.
- Compile-fail examples reject ordinary description cloning and builder broadcasting.

### Closure capture and dynamic ownership

Test:

- A linear description can move into a one-shot lazy registration closure.
- A reusable closure can construct a fresh inner description on each invocation.
- A reusable closure can clone an explicitly broadcast inner description.
- `Option<S>::take()` supports a one-shot linear inner description.
- `flatten` and switch consume emitted inner descriptions without requiring `S: Clone`.
- Replacing an active dynamic subscription cleans the prior handle once.
- Despawning the dynamic manager/output owner cleans the final active handle once.
- A failure returned by the new fallible connection transaction preserves the previously active subscription.
- `signal::option(Some(move_only_boxed_signal)).flatten()` consumes the optional description without cloning.

### Fan-out validation

Test:

- Zero downstreams.
- One downstream.
- Two distinct downstreams on a move-only node.
- Duplicate insertion of the same edge.
- Multiple registrations without multiple downstream edges.
- Fan-out after graph levels have already been cached.
- Dynamic fan-out attempted during graph processing.
- Error message includes source and guidance to use `.broadcast()`.
- Failed registration restores registration counts and leaves no newly registered system entities or partial edges.
- A stale/removed downstream does not create a false fan-out rejection when a new live branch is connected.

### Broadcast behavior

Use a clone-counting value:

```rust
struct CloneCounter {
    clones: Arc<AtomicUsize>,
}
```

Test:

- One downstream performs zero clones.
- Two downstreams perform one clone.
- N live downstreams perform N - 1 clones.
- Stale downstream IDs and entities missing input buffers perform no clones.
- Every downstream receives an equivalent value.
- Recipient ordering is deterministic.
- Broadcast works across schedules where existing schedule semantics allow delivery.
- Broadcast node cleanup removes its edge relationships.

### Polling

Test:

- Polling a linear move-only chain.
- Polling a broadcast branch.
- Polling one branch while the source has unrelated downstreams.
- Diamond graph converging on the polled target.
- Polling a target with a move-only output.
- Downcast success and type mismatch.

### Synchronous triggers

Test:

- Single move-only seed input.
- Multiple independently owned seed inputs.
- Cloneable convenience triggering, if retained.
- Triggered downstream fan-out through a broadcast node.
- Triggered move-only fan-out rejection.
- Dynamic reader/forwarder paths used by flatten and switch.

### Builder

Test:

- `signal::once(Some(Builder))` creates one child.
- A state signal mapped to `Builder` works without `Clone`.
- `on_signal`, `on_signal_with_entity`, `on_signal_with_component`, and `component_signal` accept suitable non-`Clone` values.
- Conditional child replacement.
- Builder-valued signal vec replace, insert, push, update, move, remove, pop, and clear.
- Branching cloneable state before builder creation produces independent children.
- A compile-fail example demonstrates that `Builder` cannot be broadcast.
- No debug/release behavioral difference.

### Signal vec and map

Test:

- Move-only mapped vec items through a linear consumer.
- Move-only mapped map values through a linear consumer.
- Explicit vec/map broadcast.
- Two independent replay subscriptions to one mutable vec.
- Concurrent `signal_map`, `signal_vec_keys`, and `signal_vec_entries` subscriptions to one mutable map.
- Replay followed by broadcast.
- Switch followed by broadcast.
- Broadcast followed by move-only mapping.
- The complete deterministic and fixed-seed oracle suite from `collection-correctness-plan.md` remains green after every replay, switch, map-signal, and collection-bound change.

### Prerequisite lifecycle regression

Rerun the supplied registration-lease suite and add only broadcast-specific cases:

- One broadcast description clone acquires one independent lease.
- Cleaning either broadcast branch first leaves the other functional.
- Cleaning all branches balances the cached broadcast plan.
- Failed move-only fan-out leaves existing branches, counts, topology, schedules, and caches unchanged.
- Cleanup of broadcast nodes and upstream plans is exact.
- Dynamic owner replacement/despawn tests remain green under move-only delivery.

### Feature configurations

At minimum run:

```text
cargo test --locked --lib
cargo check --locked --no-default-features
cargo check --locked --no-default-features --features builder
cargo check --locked --all-features
```

Also validate wasm/no_std configurations already supported by CI.

Normal parallel tests are a prerequisite supplied by `world-local-cleanup-plan.md`. Any renewed cross-test lifecycle interference is a blocker, not an accepted limitation of this plan.

## Benchmark plan

Add focused benchmarks or instrumentation for:

- A long linear chain carrying a small value.
- A long linear chain carrying a large non-cloneable value.
- One broadcast node with 2, 8, and 64 downstreams.
- Existing clone-heavy collection diff propagation.
- Builder-valued vec updates.
- Polling a deep linear chain.
- Polling one branch of a wide graph.

Expected outcomes:

- Linear chains perform no value clones for transport.
- Broadcast performs exactly N - 1 clones for N live downstreams.
- The additional delivery-policy lookup has negligible impact relative to system execution.
- Polling no longer clones for unrelated downstream branches.

## Documentation updates

Update:

- Crate-level runtime overview.
- `Signal` trait documentation.
- `SignalExt::broadcast`.
- `SignalVecExt::broadcast`.
- `SignalMapExt::broadcast`.
- `Builder` documentation.
- `child_signal` and `children_signal_vec` examples.
- Polling/downcast API documentation.
- README architecture description.
- Changelog and migration guide.

Explicitly document the capability distinction:

- Ordinary descriptions are linear and move-only.
- Private lazy identities may be shared internally.
- Cloning an explicit broadcast description creates another subscription from that shared broadcast boundary.
- Broadcasting emitted values permits multiple downstream owners and requires cloneable items.
- `SignalHandle` is a linear cleanup token, not a reusable description.

## Risks and mitigations

### Risk: hidden internal fan-out

Some combinators may create multiple edges internally even when the public chain appears linear.

Mitigation:

- Add registration-time validation before relaxing bounds.
- Audit every `pipe_signal` call.
- Add graph diagnostics that print source and downstream entities.

### Risk: polling behaves differently from scheduled processing

Polling currently has a separate propagation implementation.

Mitigation:

- Centralize delivery-policy decisions.
- Filter polling to the target-reachable subgraph.
- Add parity tests for scheduled and polled execution.

### Risk: dynamic combinators bypass policy

Manual triggering and forwarder queues may avoid normal forwarding.

Mitigation:

- Replace multi-seed clone-based trigger APIs.
- Route dynamic downstream propagation through the same forwarding helper.
- Add switch/flatten-specific move-only tests.

### Risk: broad source breakage

Existing applications may depend on implicit signal fan-out.

Mitigation:

- Produce precise registration errors.
- Add `.broadcast()` before removing implicit clone behavior.
- Publish side-by-side migration examples.
- Consider a temporary tracing warning phase if releases permit gradual migration.

### Risk: custom/internal graph construction bypasses compile-time linearity

Removing `Clone` from ordinary crate-provided descriptions catches normal accidental branching at compile time, but custom signal types and internal low-level edge creation can still construct invalid fan-out.

Mitigation:

- Fail invalid low-level fan-out at graph registration, before frame execution.
- Keep `.broadcast()` compile-time constrained by `Item: Clone`.
- Keep runtime delivery-policy checks as a defensive invariant.
- Document that external `Signal` implementations must not treat ordinary `Clone` as broadcast permission.

### Risk: prerequisite invariant regression

Transport changes touch graph, replay, and dynamic code already hardened by prerequisite plans.

Mitigation:

- Treat world-local cleanup, lease, and collection suites as mandatory gates for every phase.
- Do not replace supplied transactions or lifecycle components with move-first-specific alternatives.
- Keep semantic diff changes out of transport commits.
- Stop and fix the violating transport change if a prerequisite regression appears.

## Commit and review strategy

Prefer a sequence of independently reviewable commits rather than one large rewrite:

1. Verify prerequisite cleanup, lease, and collection baselines.
2. Sorted live-downstream and centralized forwarding helpers.
3. Poll-reachable downstream filtering.
4. Delivery metadata with existing clone transport.
5. Move-only erased, synchronous-trigger, and public polling transport with existing public signal bounds.
6. Move/broadcast node registration using exact cached leases.
7. Extend transactional edge preflight with fan-out validation.
8. Scalar `.broadcast()` and shared mutable-source broadcaster policy.
9. Move-only boxed descriptions, `signal::option` migration, and optional sealed boxed broadcast capability.
10. Ordinary scalar description `Clone` removal plus scalar bound relaxation.
11. Collection description `Clone` removal and builder-critical signal-vec bound relaxation while preserving collection oracles.
12. Builder clone removal.
13. Remaining signal vec/map broadcast and bound audits.
14. Dynamic delivery-policy audit, residual compatibility removal, documentation, cleanup, and benchmarks.

Each commit should keep the test suite passing. Avoid combining mechanical bound removal with behavioral graph changes in the same commit.

## Acceptance criteria

The project is complete when all of the following are true:

- `Builder` does not implement `Clone`.
- Existing linear `child_signal` and `children_signal_vec` syntax works with builder-valued outputs.
- Scalar, vec, and map signals support move-only values through linear chains where their own combinator semantics permit it.
- Ordinary crate-provided scalar, vec, map, and boxed descriptions do not implement `Clone`.
- Only explicit broadcast wrappers generally expose public cloneable description semantics.
- Cloning a broadcast description does not recursively clone/re-register its upstream chain.
- Dynamic combinators consume emitted inner descriptions once and do not require blanket inner-description `Clone` bounds.
- Prerequisite lease tests continue proving non-cloneable handles and entity-owned active subscriptions.
- Every node has explicit delivery metadata.
- A move-only node cannot acquire a second distinct live downstream edge.
- Failed fan-out releases all newly acquired leases, installs no partial edge mutation, and preserves pre-existing active branches; live descriptions may retain newly materialized dormant definitions.
- Shared mutable vec/map broadcaster nodes support multiple replay subscribers without user intervention.
- Scalar, vec, and map `.broadcast()` combinators exist and require cloneable outputs.
- All execution paths enforce identical delivery rules.
- Polling ignores unrelated downstream branches.
- `once` supports move-only values.
- `always` retains its necessary clone requirement.
- Broadcast performs exactly N - 1 clones for N live downstreams.
- Invalid fan-out fails during registration with actionable guidance.
- No emitted-value runtime storage or public polling API depends on `AnyClone`.
- Builder behavior is identical in debug and release builds.
- Migration and architecture documentation are published.
- Normal parallel tests, all prerequisite suites, and supported feature checks pass.

## Follow-up opportunities

After this refactor stabilizes, consider separate designs for:

- Fully public fallible signal registration beyond the internal lease transaction.
- Schedule-aware buffered delivery.
- Cached-value polling.
- Graph visualization including move/broadcast edge capabilities.
- Static or debug-time reporting of expensive broadcast nodes.
