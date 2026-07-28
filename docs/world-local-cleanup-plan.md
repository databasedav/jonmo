# World-Local Lock-Free Deferred Cleanup

## Status

Issue 1 is implemented and ready to close. Signal, mutable-vec, and mutable-map producers use cleanup inboxes owned by individual Bevy `World`s; no process-global cleanup queue remains, and normal-parallel tests pass without reset helpers or forced serialization.

Same-invocation cleanup is guaranteed for explicit move-only obligations transferred into the current `CleanupWorklist`. Every request sent through `CleanupSender` is an external root; if an arbitrary destructor emits it after the current boundary, it is processed by the next scheduled drain. This limitation preserves finite root snapshots and deferral of unrelated post-boundary work. This document remains the canonical architecture and rationale for issue 1; registration leases, collection correctness, move-first signal delivery, and later landing steps remain separate work.

## Historical problem statement

Before this refactor, Jonmo stored world-relative `Entity` identifiers in three process-global queues:

- `STALE_SIGNALS` in `src/graph.rs`
- `STALE_MUTABLE_VECS` in `src/signal_vec.rs`
- `STALE_MUTABLE_BTREE_MAPS` in `src/signal_map.rs`

Each app installed systems that drained those same globals. The first world to run consumed all pending requests, regardless of which world owned the entities. Entity IDs are only meaningful inside their originating `World`, so this could:

- Despawn unrelated entities in another world when IDs collide.
- Consume another world's request and leak the intended target.
- Make parallel tests interfere with one another.
- Lose pending work when another plugin instance clears the globals.

The pre-refactor plugin also cleared all three queues from `JonmoPlugin::build`, allowing plugin construction in one app to mutate another app's cleanup state.

## Decision summary

The implementation uses one private lock-free cleanup inbox per `World`:

1. Each world owns a `WorldCleanupQueue` resource backed by an unbounded concurrent queue.
2. Cleanup-producing handles retain either an immediately bound sender or a deferred sender created only for `Commands` construction.
3. `Drop` enqueues a typed request without requiring `&mut World` or acquiring a Jonmo-managed mutex.
4. A cleanup system drains only its own world's inbox.
5. `BoundCleanupTarget` is the sole primitive that delivers requests to a world and retains only a `Weak` inbox reference.
6. Direct `World` constructors store `CleanupSender::Bound` and pay no `OnceLock`, pending queue, handoff atomic, or extra route-allocation cost.
7. Deferred `Commands` constructors store `CleanupSender::Deferred` and queue one non-cloneable, consuming `CleanupBinder`.
8. Only the deferred route uses a `OnceLock`, private concurrent pending queue, and one `AtomicUsize` handoff word shared by sender entry, sender exit, and binder publication.
9. The handoff word's top bit is a monotonic `BOUND` flag; its remaining bits are a checked in-flight sender count. This single atomic modification order closes the send-versus-bind visibility race.
10. Drain passes insert one queue boundary and take only requests linearized before that boundary as root work.
11. Root handlers transfer explicit move-only obligations directly into a local `CleanupWorklist`; they do not promote inbox sends based on logical causality.
12. The local worklist runs to exhaustion in the same drain invocation with no retry-wave constant.
13. Every local append consumes or disarms a finite cleanup ownership obligation; handlers never create new cleanup ownership.
14. Cleanup requests are idempotent and order-independent; queue arrival order is never semantic ownership state.
15. Mutable vec/map owners privately retain the exact full generational entity and broadcaster ownership created for that source.
16. Duplicate, stale, late, and post-world-destruction requests are harmless.
17. No process-global map from `WorldId` to queue is introduced.
18. Selected roots and explicit move-only obligations transferred into their `CleanupWorklist` settle in that invocation. Every `CleanupSender` send remains an external root selected only by a boundary that follows its linearization.
19. Strict lock-freedom is claimed only on targets with suitable native atomics; atomic-emulation targets may internally use critical sections.

## Goals

- Route every cleanup request exclusively to its originating world.
- Support handles dropped without world access.
- Support direct `World` constructors and deferred `Commands` constructors.
- Preserve `no_std`, `critical-section`, and wasm compatibility.
- Make test execution independent and parallel-safe.
- Preserve existing public mutable collection APIs.
- Provide a general typed inbox that issue 2 can later use for registration-definition cleanup.
- Keep cleanup enqueue and root snapshotting free of Jonmo-managed mutexes.
- Never mutate a Bevy world while collecting a root snapshot.
- Classify every `CleanupSender` send as an external root, irrespective of call context or logical cause.
- Settle finite chains of explicitly represented move-only `CleanupWorklist` obligations without an arbitrary retry or wave limit.

## Non-goals

This plan does not:

- Redesign `SignalHandle` or registration counts.
- Fix recursive registration cleanup.
- Remove `Clone` from signal descriptions or builders.
- Add move-first output transport or `.broadcast()`.
- Fix collection diff semantics.
- Redesign schedules.
- Promise immediate cleanup from `Drop`.
- Support materializing one already-registered lazy signal identity independently in multiple worlds.
- Add a public manual cleanup API.

## Required invariants

1. No process-global collection stores world-relative cleanup targets.
2. Every `BoundCleanupTarget` targets exactly one world inbox.
3. Only a `CleanupBinder` may publish a deferred destination.
4. `CleanupBinder` is non-cloneable, consumes itself during binding, and publishes at most one target.
5. Direct bound senders contain no deferred route state or pending queue.
6. Requests sent before deferred binding are retained and delivered exactly once after binding.
7. A request cannot be consumed by another world's drain system.
8. Dropping a handle does not require world access.
9. Handles that outlive their world do not keep the world inbox alive.
10. Sending after world destruction is a safe no-op.
11. Plugin initialization never replaces, clears, or invalidates an existing inbox.
12. Enqueue, deferred binding, and root snapshot collection acquire no Jonmo-managed mutex.
13. Every `CleanupSender` send enters the external root stream. A drain boundary selects a finite root prefix; sends linearized after it remain in the inbox for a later invocation.
14. No cleanup request handler runs until the complete root snapshot has been removed from the inbox.
15. Concurrent deferred send/bind/flush operations lose and duplicate no requests.
16. Every deferred sender entry, sender exit, and binder publication participates in one `AtomicUsize` modification order.
17. The handoff `BOUND` bit is monotonic and checked in-flight-count updates cannot carry into, borrow from, clear, or otherwise alter it.
18. Cleanup requests are idempotent and semantically order-independent; concurrent pending flushes may reorder root requests.
19. Same-invocation follow-up work enters `CleanupWorklist` only through typed methods that consume an explicit move-only obligation.
20. A local obligation must already be carried by a selected root or be explicitly moved from a known token- or lease-owning world component. ECS mutation, graph reachability, destructor execution, or logical causality alone cannot synthesize local work.
21. No local handler creates a new cleanup owner, reacquires a released obligation, or republishes the same obligation.
22. Duplicate, missing, and ineligible targets synthesize no new obligation; independently owned candidate tokens already carried by a request are still transferred exactly once.
23. Each mutable vec/map owner stores exactly the full generational entity and broadcaster created by its private constructor.
24. Mutable root requests can only be constructed through those private owner paths and target the stored entity unchanged.
25. Signal cleanup acts only on valid signal entities that satisfy existing eligibility checks.
26. A lazy signal registered in one world cannot silently access an entity with a colliding ID in another world.
27. The request enum and worklist can later carry owned registration-definition candidate batches without another transport redesign.
28. Queue closure is not used during ordinary operation; an enqueue failure is an internal invariant violation, not permission to discard live-world cleanup.

## Implemented module

The private module is:

```text
src/cleanup.rs
```

It is declared privately from `src/lib.rs`:

```rust
mod cleanup;
```

### Cleanup requests

The implementation uses a typed request enum rather than boxed arbitrary world closures:

```rust
// Defined beside SignalSystem in graph.rs. Its production constructor is private
// to graph.rs, and the token deliberately implements neither Clone nor Copy.
pub(crate) struct ReleasedSignalCandidate(SignalSystem);

enum CleanupRequest {
    StaleSignal(ReleasedSignalCandidate),
    MutableVec {
        entity: Entity,
        stale_broadcaster: Option<ReleasedSignalCandidate>,
    },
    MutableBTreeMap {
        entity: Entity,
        stale_broadcaster: Option<ReleasedSignalCandidate>,
    },

    // Added by issue 2; this is an owned non-cloneable batch token:
    // LazyDefinitionCandidates(ReleasedDefinitionCandidates),
}
```

Mutable root requests aggregate the exact backing entity with any signal candidate produced by explicitly releasing that source's broadcaster. `ReleasedSignalCandidate` is a crate-visible, move-only obligation token defined beside `SignalSystem` in `src/graph.rs`; its wrapped identity and production constructor remain private to that module. Other production modules may receive, inspect through a read-only identity accessor, and consume a token, but cannot fabricate one from a copied `SignalSystem`. A `#[cfg(test)]` factory supports infrastructure tests without broadening production authority. This preserves both causality and the local termination measure across the no-`World` boundary rather than relying on the broadcaster's field destructor to enqueue a second independent root.

Typed requests provide:

- Auditable centralized dispatch.
- A closed private request vocabulary.
- Predictable `Send + Sync` behavior.
- No arbitrary code execution from `Drop`.
- A stable place to add future lifecycle request kinds.

### Local causal worklist

The concurrent inbox is the destination for every `CleanupSender` send. Once a bounded root snapshot is collected, only explicit move-only obligations transferred through typed `CleanupWorklist` methods may participate in the current invocation:

```rust
enum CleanupAction {
    Root(CleanupRequest),

    // Added by issue 2:
    // ReleaseSignalLease(SignalHandle),
}

pub(crate) struct CleanupWorklist {
    actions: VecDeque<CleanupAction>,
    signal_candidates: Vec<ReleasedSignalCandidate>,

    // Replaced/extended by issue 2 with move-only definition batches.
}

impl CleanupWorklist {
    fn from_roots(roots: Vec<CleanupRequest>) -> Self;
    fn pop_action(&mut self) -> Option<CleanupAction>;
    fn take_signal_candidates(&mut self) -> Vec<ReleasedSignalCandidate>;
    fn is_empty(&self) -> bool;

    pub(crate) fn push_stale_signal(
        &mut self,
        candidate: ReleasedSignalCandidate,
    );

    // Added by issue 2:
    // pub(crate) fn push_signal_lease(&mut self, lease: SignalHandle);
    // pub(crate) fn push_lazy_definition_candidates(
    //     &mut self,
    //     candidates: ReleasedDefinitionCandidates,
    // );
}
```

The worklist is phased rather than a naïve FIFO:

1. Drain explicit move-only release actions, including future leases moved from known lease-owning components by issue 2.
2. Accumulate/deduplicate the move-only node/definition candidate obligations produced by those releases.
3. Only after the release-action queue is empty, reap candidates in deterministic reverse-topological order.
4. If reaping explicitly takes a known obligation-owning component and moves another release obligation into the worklist, return to the release phase before continuing candidate reaping.
5. Stop when both action and candidate stores are empty.

This ordering prevents premature reaping only for ownership explicitly represented in the worklist or explicitly moved from known components. It does not inspect arbitrary values captured by `SystemRunner`, Bevy systems, callbacks, or user closures. Any `CleanupSender` send caused by dropping such a capture remains an external root behind the current boundary. `CleanupWorklist` is not `Clone`, is never shared with producer threads, and has no general public `push` surface; typed methods keep local obligation transfer auditable.

### Queue dependency and portability boundary

The implementation uses the maintained `concurrent-queue` unbounded queue with default features disabled rather than implementing memory reclamation or lock-free linked storage inside Jonmo:

```toml
concurrent-queue = { version = "2.5", default-features = false }
```

Reasons:

- It provides an unbounded concurrent queue suitable for multiple drop producers and one world consumer.
- It supports `no_std` with a global allocator.
- Its optional `std` and `portable-atomic` integration can be mapped to Jonmo's feature matrix after target validation.
- Jonmo keeps `unsafe_code = "deny"`; the queue algorithm remains in an audited dependency rather than new local unsafe code.

Final validation records the exact feature wiring for Jonmo's supported native-atomic, wasm, and `critical-section` targets. An ad hoc atomic linked list remains outside the design.

“Lock-free” in this plan means Jonmo's enqueue and drain protocol acquires no explicit mutex and the queue uses lock-free atomics on targets that provide the required native operations. When atomics are emulated through `portable-atomic` or a target critical-section backend, the platform may internally mask interrupts or serialize operations; documentation must not claim universal hardware lock-freedom.

### World cleanup resource

The public request type remains distinct from an internal drain-boundary message:

```rust
enum CleanupMessage {
    Request(CleanupRequest),
    DrainBoundary,
}

struct CleanupInbox {
    messages: ConcurrentQueue<CleanupMessage>,
}

#[derive(Resource)]
pub(crate) struct WorldCleanupQueue {
    world_id: WorldId,
    inbox: Arc<CleanupInbox>,
}
```

Initialize with `FromWorld` so the resource records its owner:

```rust
impl FromWorld for WorldCleanupQueue {
    fn from_world(world: &mut World) -> Self {
        Self {
            world_id: world.id(),
            inbox: Arc::new(CleanupInbox::default()),
        }
    }
}
```

`WorldId` is used for validation and diagnostics. It is not used as a key into global state. The `Arc` allocation's identity distinguishes an existing inbox from a same-`WorldId` resource that was removed and reinitialized.

The live queue is never deliberately closed. A `push` failure therefore represents either a dead/orphan route or an internal invariant violation; it must never silently discard a request still destined for a live world.

### Cleanup routing capabilities

A cleanup-producing owner stores a sender rather than the resource itself. Direct and deferred construction are represented explicitly, but both eventually deliver through one `BoundCleanupTarget` primitive:

```rust
#[derive(Clone)]
pub(crate) enum CleanupSender {
    Bound(BoundCleanupTarget),
    Deferred(Arc<DeferredCleanupRoute>),
}

#[derive(Clone)]
struct BoundCleanupTarget {
    world_id: WorldId,
    inbox: Weak<CleanupInbox>,
}

struct DeferredCleanupRoute {
    target: OnceLock<BoundCleanupTarget>,
    pending: ConcurrentQueue<CleanupRequest>,
    handoff: AtomicUsize,
}

pub(crate) struct CleanupBinder {
    route: Arc<DeferredCleanupRoute>,
}
```

`CleanupBinder` deliberately does not implement `Clone`. It is the unique capability to publish one destination for one deferred route. `CleanupSender::Deferred` may be cloned with its owner, but it cannot bind itself.

Required operations:

```rust
impl CleanupSender {
    pub(crate) fn bound(world: &mut World) -> Self;
    pub(crate) fn deferred() -> (Self, CleanupBinder);
    pub(crate) fn send_stale_signal(
        &self,
        candidate: ReleasedSignalCandidate,
    );
    pub(crate) fn send_mutable_vec(
        &self,
        entity: Entity,
        stale_broadcaster: Option<ReleasedSignalCandidate>,
    );
    pub(crate) fn send_mutable_btree_map(
        &self,
        entity: Entity,
        stale_broadcaster: Option<ReleasedSignalCandidate>,
    );
    pub(crate) fn world_id(&self) -> Option<WorldId>;

    fn send(&self, request: CleanupRequest);
}

impl CleanupBinder {
    pub(crate) fn bind(self, world: &mut World);
}
```

`bevy_platform::sync::OnceLock`, the private pending queue, and the `AtomicUsize` handoff word exist only inside `DeferredCleanupRoute`. Directly bound senders do not allocate a deferred route and do not carry unused deferred synchronization state.

#### `BoundCleanupTarget`: the single delivery primitive

`BoundCleanupTarget::new(world)`:

1. Calls `world.get_resource_or_init::<WorldCleanupQueue>()` or the exact supported equivalent.
2. Records that resource's `WorldId` for affinity diagnostics.
3. Stores a `Weak` reference to the exact inbox allocation.

Its send operation is the only code that publishes into a world inbox:

```rust
impl BoundCleanupTarget {
    fn send(&self, request: CleanupRequest) {
        let Some(inbox) = self.inbox.upgrade() else {
            return;
        };

        inbox.push_request(request);
    }
}
```

A dead weak inbox is a safe no-op. The live inbox is never closed; its queue helper handles the impossible closed-queue branch consistently without exposing it to owner drop implementations.

#### Direct bound sender

`CleanupSender::bound(world)` returns:

```rust
CleanupSender::Bound(BoundCleanupTarget::new(world))
```

The common direct-construction send path is one enum match, one `Weak::upgrade`, and one concurrent world-inbox push. It performs no `OnceLock` access, owns no pending queue, and requires no extra route allocation.

#### Deferred sender and one-shot binder

`CleanupSender::deferred()` allocates one shared `DeferredCleanupRoute` with an unset target and empty pending queue, then returns two distinct capabilities:

```rust
let route = Arc::new(DeferredCleanupRoute::new());
(
    CleanupSender::Deferred(route.clone()),
    CleanupBinder { route },
)
```

The sender belongs to the returned mutable owner. The unique binder belongs to the queued command. This separation makes rebinding structurally unavailable: ordinary senders expose no `bind`, and the non-cloneable binder consumes itself.

`CleanupBinder::bind(self, world)`:

1. Obtains `BoundCleanupTarget::new(world)`.
2. Initializes the deferred route's `OnceLock`; an already-initialized target is an internal invariant violation because producing or applying the unique binder twice should be impossible.
3. Publishes the monotonic `BOUND` bit with an `AcqRel` `fetch_or` on the shared handoff word and verifies that the previous state was not already bound.
4. Flushes pending requests through `BoundCleanupTarget::send`.
5. Consumes and drops the binder.

The target is initialized before the release portion of the handoff publication, so a sender that acquires a handoff state containing `BOUND` may safely read it. No request is moved as part of destination publication. Publication and pending-queue flushing remain separate concurrent operations, but their visibility is coordinated through the single handoff atomic rather than inferred from independent `OnceLock` and queue observations.

#### Shared send dispatch

Typed sender helpers construct the private request once, then delegate:

```rust
fn send(&self, request: CleanupRequest) {
    match self {
        CleanupSender::Bound(target) => target.send(request),
        CleanupSender::Deferred(route) => route.send(request),
    }
}
```

The enum branch is paid only when cleanup is requested, normally on final owner drop. It does not create two world-delivery implementations: the deferred route only stages requests until it can delegate to `BoundCleanupTarget::send`.

#### Deferred single-atomic handoff protocol

The handoff word is partitioned as follows:

```rust
const HANDOFF_BOUND: usize = 1usize << (usize::BITS - 1);
const HANDOFF_IN_FLIGHT_MASK: usize = HANDOFF_BOUND - 1;
```

The top bit is a sticky `BOUND` flag. The lower bits count sends that have entered the deferred route but have not yet left it. All count changes and binder publication are `AcqRel` read-modify-write operations on this one atomic, creating one modification order for the handoff decision.

`DeferredCleanupRoute::send` uses this algorithm:

1. Enter with a checked `fetch_update` that increments only the lower in-flight count and returns the prior handoff state.
2. If that prior state already contains `BOUND`, acquire the initialized target, deliver directly through `BoundCleanupTarget::send`, and leave with a checked decrement.
3. Otherwise, push the request into `pending`.
4. Leave with a checked decrement on the same handoff atomic.
5. If the state observed while leaving contains sticky `BOUND`, acquire the initialized target and help flush pending requests. Otherwise, the unique binder will perform the eventual flush.

`CleanupBinder::bind` initializes the target, publishes `BOUND` with `fetch_or(AcqRel)`, then flushes. The shared atomic closes the race that independent `OnceLock::get()` and queue operations cannot close by themselves:

- **Sender exit precedes binder publication in the handoff modification order.** The pending push is sequenced before the release portion of the sender's exit RMW. The binder's later acquiring RMW observes that release directly or through intervening AcqRel RMWs, then flushes after publication. The completed pre-bind request is therefore visible to a binder/helper flush.
- **Binder publication precedes sender exit.** The bound bit is never cleared, so the sender's exit RMW observes a state containing `BOUND`. Its acquire side observes target initialization and it flushes after its pending push.
- **Sender enters after binder publication.** Its entry RMW observes `BOUND` with acquire semantics, reads the initialized target, and sends directly without staging.
- **Sender pauses after entry but before its pending push.** The binder may publish and flush an empty queue, but the resumed sender pushes before leaving; its exit observes sticky `BOUND` and flushes the newly staged request.
- **Several senders and flushers overlap.** Each request is either sent directly once or pushed once. Each successful concurrent queue pop transfers one staged request to only one flusher, preserving the exact multiset even though delivery order may change.

Checked `fetch_update` operations reject lower-bit overflow and underflow before arithmetic. Increment cannot carry into `BOUND`; decrement cannot borrow from or clear it; binder `fetch_or` preserves all in-flight bits. Invalid count states are invariant failures and leave the handoff word unchanged.

`flush_pending` repeatedly pops pending requests and forwards them through the published `BoundCleanupTarget` until the queue is observed empty. The binder and senders may call it concurrently. Cleanup requests must therefore be order-independent: one flusher can pop request A, pause, and enqueue it after another flusher has forwarded request B. Exact multiset preservation is required; preservation of concurrent producer order is not.

If the published weak inbox no longer upgrades, flushing drops the orphaned requests outside ECS mutation. If a deferred command queue is never applied, its binder disappears; once all returned owners also disappear, the private pending queue and requests disappear because no entity was ever materialized in a world.

Use explicit helpers for queue pushes so the impossible/closed-queue branch is handled consistently and never hidden behind an unconditional `unwrap` in a `Drop` path.

### Why use `Weak`

The world resource must own the inbox lifetime:

1. The world drops `WorldCleanupQueue`.
2. Its strong `Arc` disappears.
3. Externally retained senders can no longer upgrade their `Weak`.
4. Sends strictly after destruction fail to upgrade and become no-ops.
5. A send racing with destruction may briefly upgrade and append to an orphan inbox; that inbox is isolated and dropped when the temporary strong reference is released.
6. No dead-world request can reach a replacement world, because a reinitialized resource owns a different `Arc` allocation.

A deferred route whose command queue is never applied loses its unique binder. Its pending queue and requests are dropped once the returned owners release their deferred senders; no entity was materialized, so no world cleanup is required.

## Deferred `Commands` binding

Direct world constructors can create a bound sender immediately. Deferred constructors cannot know the destination world synchronously.

For every `spawnc` or `From<&mut Commands>` path:

1. Create one deferred sender/binder pair.
2. Spawn/reserve the target entity through `Commands`.
3. Store the deferred sender in the returned ownership handle.
4. Move the unique binder into a binding command queued after the spawn command:

```rust
let (cleanup, binder) = CleanupSender::deferred();
let entity = commands.spawn(data).id();

commands.queue(move |world: &mut World| {
    binder.bind(world);
});
```

If the public handle is dropped before commands apply, its cleanup request remains buffered because the queued binder still owns the deferred route. If the command queue is eventually applied and the spawn/bind commands succeed, the consumed binder publishes the correct bound target, flushes the request, and a later Jonmo drain cleans the entity. If the command queue is discarded, the entity is never spawned; the binder disappears, and the pending request is safely dropped when the remaining deferred sender state disappears. No eventual-cleanup guarantee applies to unapplied commands.

Do not infer world identity from a reserved `Entity` and do not introduce a global pending-command queue.

## Cleanup targets and private ownership

Mutable collection cleanup does not add category marker components. Its ownership proof is the direct private construction chain:

1. A constructor spawns or reserves the data entity and receives its full generational `Entity`.
2. That same constructor immediately stores the entity unchanged in a private `MutableVecOwner` or `MutableBTreeMapOwner`.
3. Only the final owner `Drop` creates the corresponding private cleanup request.
4. The request is routed through the exact originating world's inbox.
5. The handler attempts a fallible despawn of that full entity.

A missing or stale entity is a no-op. A reused entity index with a different generation does not resolve. Making a request target an unrelated same-world entity requires a Jonmo implementation bug in private owner/request construction; focused constructor tests and encapsulation enforce that invariant.

A zero-sized category marker would catch only some cross-kind internal bugs and would not detect a request accidentally targeting a different mutable source of the same category. An exact per-instance token would provide stronger runtime authentication but would add an allocation and reference-counted component per source without addressing a public or untrusted input boundary. Both are rejected for this internal protocol unless future extension makes cleanup requests externally constructible.

### Request handlers

Each handler is singular and idempotent:

```rust
pub(crate) fn cleanup_stale_signal(
    world: &mut World,
    candidate: ReleasedSignalCandidate,
    worklist: &mut CleanupWorklist,
);

pub(crate) fn cleanup_mutable_vec(
    world: &mut World,
    entity: Entity,
    stale_broadcaster: Option<ReleasedSignalCandidate>,
    worklist: &mut CleanupWorklist,
);

pub(crate) fn cleanup_mutable_btree_map(
    world: &mut World,
    entity: Entity,
    stale_broadcaster: Option<ReleasedSignalCandidate>,
    worklist: &mut CleanupWorklist,
);
```

Mutable collection handlers:

1. Receive the exact full generational entity and broadcaster candidate retained by the private aggregate root request.
2. Attempt a fallible despawn in the sender's originating world.
3. Transfer the explicitly released broadcaster candidate, if present, into the local worklist exactly once even if the data entity was already absent.
4. Treat a missing entity as an entity-cleanup no-op; the candidate remains independently meaningful because its description reference was already consumed before enqueue.

Eligibility is rechecked when the later `StaleSignal` work item runs. Duplicate candidate identities are harmless, but the transport and owner invariants still require each owned candidate payload to be transferred exactly once.

Keep `CleanupRequest` private to `src/cleanup.rs`. Expose only typed crate-private sender helpers such as `send_mutable_vec(entity, stale_broadcaster)` and `send_mutable_btree_map(entity, stale_broadcaster)` so root variants remain coupled to their owner implementations. Issue 2 adds corresponding typed inbox/worklist lazy-candidate helpers rather than exposing the enum.

Signal cleanup preserves its existing eligibility checks for issue 1, but final entity access/despawn must be fallible and duplicate-safe.

Issue 2 will later replace the registration-count semantics without changing the world-local routing mechanism.

## Root snapshot and causal draining

Install one exclusive drain function:

```rust
pub(crate) fn drain_world_cleanup(world: &mut World)
```

Root snapshot algorithm:

1. Obtain the current world's resource and clone its inbox `Arc`.
2. Push one private `CleanupMessage::DrainBoundary` into that inbox.
3. Pop messages into a root `Vec<CleanupRequest>` until that boundary is observed.
4. Stop touching the concurrent inbox.
5. Leave every request linearized after the boundary—including `CleanupSender` sends triggered by destructors while dispatching or reaping current roots—for a later drain invocation.

The queue must provide linearizable FIFO behavior. With one exclusive world consumer, the boundary splits the external root stream into a finite current prefix and later suffix even while producers continue sending.

After collecting roots, create the invocation-local worklist and process it to exhaustion:

```rust
let mut worklist = CleanupWorklist::from_roots(roots);

loop {
    while let Some(action) = worklist.pop_action() {
        dispatch_cleanup_action(world, action, &mut worklist);
    }

    let candidates = worklist.take_signal_candidates();
    if candidates.is_empty() {
        break;
    }

    reap_signal_candidates(world, candidates, &mut worklist);
}
```

Selected roots are dispatched in the current invocation. Beyond root dispatch, same-invocation processing applies only to move-only obligations explicitly transferred into `CleanupWorklist`; handlers that own such obligations append them directly through typed worklist methods. Candidate reaping starts only after the current explicit release-action queue reaches a fixed point.

Any `CleanupSender` call—including one synchronously triggered by dropping a field, component, `SystemRunner`, or opaque capture during dispatch—is a new external root. If it linearizes after the current boundary, it waits for a later invocation. No generic transitive single-drain guarantee is made for arbitrary Rust destructor graphs.

- The explicit local worklist runs to exhaustion without a retry wave or arbitrary maximum iteration count.
- Every post-boundary external root remains in the concurrent inbox until a later drain.
- No queue operation encloses ECS mutation.

The boundary itself never reaches a request handler. If root extraction panics unexpectedly, no ECS mutation has begun; implementation and tests should keep message extraction non-panicking.

### Opaque destructor boundary

Safe Rust provides no generic introspection of values captured by a type-erased `SystemRunner`, Bevy system, callback, or user closure. Cleanup-producing ownership hidden in such a capture is therefore not an explicit local obligation. If reaping drops the capture and its destructor calls `CleanupSender`, that request is an external root. A send linearized after the current boundary is processed by the next scheduled drain, which may be later in the same frame when multiple Jonmo schedules are configured.

The permanent `mutable_btree_map_world_cleanup_opaque_runner_drop_becomes_next_root` regression records this contract: the first drain reaps the stale signal and queues the opaque runner's mutable-map root behind the consumed boundary; the second drain removes the map; a third drain is idempotent. This preserves finite root snapshots and avoids capturing unrelated concurrent work.

### Termination invariant

The local worklist is finite over explicitly represented move-only obligations, not over the arbitrary Rust destructor graph. Every local append must consume an obligation already carried by a selected root or explicitly moved from a known obligation-owning component, such as a `ReleasedSignalCandidate`, future lease, or non-cloneable candidate batch.

ECS mutation, entity removal, graph traversal, and destructor execution do not themselves authorize a local append. Drops from opaque captures may send external roots, but those roots are outside the current worklist's termination measure. Handlers may not construct new cleanup owners, reacquire released obligations, or synthesize work not backed by a consumed payload. Test accounting covers explicit local obligations and proves that they are consumed exactly once.

This is a private-code invariant rather than protection against arbitrary user-supplied requests: `CleanupRequest` and unrestricted worklist insertion remain inaccessible outside the cleanup implementation and typed cooperating modules.

## Mutable ownership objects and explicit broadcaster release

The implementation replaces bespoke atomic final-owner detection for mutable collections with one private owner allocation per source. Broadcaster ownership lives in that authoritative source owner so final drop packages the explicitly released signal candidate into one aggregate external root.

### Mutable vec

```rust
struct MutableVecOwner {
    entity: Entity,
    cleanup: CleanupSender,
    broadcaster: Option<LazySignal>,
}

impl Drop for MutableVecOwner {
    fn drop(&mut self) {
        let stale_broadcaster = self
            .broadcaster
            .take()
            .and_then(LazySignal::into_stale_candidate);

        self.cleanup
            .send_mutable_vec(self.entity, stale_broadcaster);
    }
}

pub struct MutableVec<T> {
    owner: Arc<MutableVecOwner>,
    _marker: PhantomData<fn() -> T>,
}
```

`LazySignal::into_stale_candidate` consumes/disarms that description reference, performs the same liveness decrement ordinary `Drop` would perform, and returns the registered identity only when eligibility must later be rechecked. It never sends to the concurrent inbox itself.

`MutableVecData<T>` retains collection state and pending diffs but no longer owns the broadcaster. `signal_vec()` clones the broadcaster through the retained owner. The broadcaster initializer captures the data `LazyEntity`, not the owner, so this must be verified not to introduce an `Arc` cycle.

`MutableVec::clone` clones the owner `Arc`; only the final owner drop releases the broadcaster and emits one aggregate root. Add private `entity()` and `broadcaster()` accessors and migrate existing `read`, `write`, and `signal_vec` field accesses.

### Mutable map

Mirror the same design with `MutableBTreeMapOwner`, owner-held broadcaster, and the private `CleanupSender::send_mutable_btree_map(entity, stale_broadcaster)` helper. Migrate `read`, `write`, `signal_map`, `signal_vec_keys`, and `signal_vec_entries` to owner accessors.

This removes manually synchronized source reference counters, preserves public handle cloning, and makes the mutable-source → broadcaster cleanup edge explicit rather than destructor-inferred.

## Lazy signal world affinity and explicit release

Issue 1 prevents one registered lazy identity from silently reusing its entity in another world and separates ownership release from where the resulting candidate is delivered.

As temporary issue-1 protection, until issue 2 makes `SignalSystem` itself world-aware, the registered lazy state retains:

- The originating `WorldId` or cleanup-inbox identity.
- The registered `SignalSystem`.
- The bound cleanup sender.

On subsequent registration:

1. Compare the current world with the stored world affinity.
2. Reject a mismatch before entity lookup or count mutation.
3. Never interpret an equal `Entity` from another world as the same signal.

`LazySignal` has one centralized exactly-once release primitive. Each handle is armed when created/cloned and is disarmed only by an explicit consuming release method:

```rust
pub(crate) struct LazySignal {
    inner: Arc<LazySignalState>,
    armed: bool,
}

impl LazySignal {
    pub(crate) fn into_stale_candidate(mut self) -> Option<ReleasedSignalCandidate>;
    pub(crate) fn release_reaped_holder(mut self);
}
```

Both explicit methods perform the same reference/liveness decrement as ordinary `Drop`, set `armed = false`, and prevent a second release. Their delivery semantics differ:

- `into_stale_candidate` returns a candidate without enqueueing it so an aggregate root or local worklist can preserve causality.
- `release_reaped_holder` is used only when the corresponding definition is already definitively being reaped; it decrements holder ownership without generating a redundant candidate for the entity currently being removed.
- Ordinary armed `Drop` remains the no-`World` fallback: it releases once and sends any candidate through the stored bound cleanup sender as a new root.

Before despawning a stale signal entity under issue 1's temporary holder model, cleanup explicitly takes its known `LazySignalHolder` and calls `release_reaped_holder`; that known holder does not enqueue a duplicate root. This does not imply introspection of `SystemRunner` or user-closure captures: ordinary drops from opaque captures send external roots through `CleanupSender`.

Supporting independent materialization of one lazy definition into several worlds requires a per-world lazy cache and is out of scope. Issue 2 replaces the temporary affinity and holder/reference-threshold details with world-aware `SignalSystem`, explicit `live_descriptions`, weak definition metadata, and candidate batches; the exactly-once explicit-release/worklist boundary remains.

## Current plugin integration

`JonmoPlugin::build`:

1. Contains no `clear_stale_*` call.
2. Calls `app.init_resource::<WorldCleanupQueue>()` rather than inserting/replacing the resource.
3. Uses `drain_world_cleanup` instead of plural stale-drain systems.
4. Preserves ordering after signal graph processing.
5. Registers the exclusive drain function in each configured schedule; post-boundary roots wait for the next scheduled drain invocation.

On-demand initialization remains required because mutable sources may be created before `JonmoPlugin` is added. Plugin initialization must preserve an already initialized queue.

## Source touchpoints

### `Cargo.toml`

- Add the selected unbounded concurrent queue as a direct dependency with default features disabled.
- Map its `std` support into Jonmo's `std` feature.
- Validate and, if required, map its portable-atomic support into Jonmo's `critical-section` feature without weakening existing targets.

### New file

- `src/cleanup.rs`

### `src/lib.rs`

- Add the private cleanup module.
- Remove plugin-time global queue clearing.
- Initialize the per-world resource idempotently.
- Register the world-local drain system.

### `src/graph.rs`

- Remove `STALE_SIGNALS` and `clear_stale_signals`.
- Store cleanup sender/world affinity in registered lazy state.
- Centralize armed `LazySignal` release and add explicit candidate/disarm paths.
- Send root `StaleSignal` requests only from ordinary no-`World` drops.
- Remove/disarm `LazySignalHolder` explicitly before stale entity despawn.
- Convert plural drain logic into a singular idempotent worklist-aware handler.
- Preserve registration semantics for issue 2.

### `src/signal_vec.rs`

- Remove `STALE_MUTABLE_VECS` and its clear/drain functions.
- Replace manual owner count with a private `Arc<MutableVecOwner>` that stores the spawned entity and broadcaster.
- Remove broadcaster ownership from `MutableVecData<T>` and route `signal_vec()` access through the owner.
- Explicitly release the broadcaster into an aggregate mutable root on final owner drop.
- Keep cleanup request construction private and coupled to final owner drop.
- Create `CleanupSender::Bound` in every direct constructor and a deferred sender/binder pair in every `Commands` constructor.

### `src/signal_map.rs`

- Mirror mutable vec migration with a private map owner retaining the exact spawned entity and broadcaster.
- Move broadcaster access out of `MutableBTreeMapData<K, V>` and into the owner.
- Preserve unrelated map-diff work already present in the working tree.

### Tests and `justfile`

- Remove all manual global queue reset helpers/imports.
- Remove `--test-threads=1` only after the parallel suite is reliable.

## Historical implementation phases

The following records the completed issue-1 landing sequence.

### Phase 0: Characterize failures

Add focused tests that reproduce:

- World B consuming world A's mutable vec request.
- World B consuming world A's mutable map request.
- Colliding signal entity IDs across worlds.
- Plugin B clearing pending cleanup for app A.
- The existing parallel-suite abort.

Acceptance: tests fail for the expected cross-world reason before production changes.

### Phase 1: Add infrastructure, singular handlers, and transitional plugin wiring

Implement:

- Direct concurrent-queue dependency and feature wiring
- `WorldCleanupQueue`
- `CleanupInbox`
- `CleanupRequest`
- `BoundCleanupTarget` as the sole inbox-delivery primitive
- `CleanupSender::{Bound, Deferred}` without deferred storage on the direct path
- Non-cloneable, consuming `CleanupBinder`
- Deferred `OnceLock`, concurrent pending queue, and single-`AtomicUsize` bound-bit/in-flight-count handoff
- Boundary-delimited root snapshotting
- Invocation-local `CleanupWorklist` and typed explicit-obligation transfer methods
- Singular idempotent, order-independent signal/vec/map request handlers that accept the worklist
- Test-only monotonic cleanup-obligation accounting
- Focused module tests

Initialize the resource and register `drain_world_cleanup` alongside temporary legacy plural drains. Keep legacy producers and their reset helpers until each producer migrates, but stop adding new global behavior. This allows every phase to compile and gives the typed drain valid handlers before requests are produced.

Acceptance:

- Two inboxes are isolated.
- Direct bound senders allocate no deferred route or pending queue.
- Send-before-bind is lossless.
- A binder is one-shot and cannot be cloned or invoked through `CleanupSender`.
- Sending after world destruction is safe.
- Concurrent send/bind/flush loses and duplicates nothing.
- A drain boundary creates a finite root snapshot and no queue operation encloses dispatch.
- Every `CleanupSender` send linearized after the boundary remains queued, while explicit move-only obligations already transferred into the local worklist settle in the same invocation.
- The local worklist terminates by monotonic consumption of those explicit obligations without a retry constant.
- Cleanup enqueue and root snapshot collection acquire no Jonmo-managed mutex.

### Phase 2: Migrate signals atomically

- Bind lazy materialization to one world before entity access.
- Initialize the cleanup resource on demand even when signals register before the plugin.
- Route ordinary no-`World` stale signal roots through that world's sender.
- Add armed exactly-once release, `into_stale_candidate`, and reaped-holder disarm paths.
- Make stale signal handlers transfer explicitly owned candidates through `CleanupWorklist` and explicitly release known holders before despawn.
- Delete the signal global queue and clear function.
- Remove the legacy signal drain/clear references from plugin wiring in the same commit.
- Remove signal-specific test reset imports/helpers in the same commit.
- Make duplicate stale requests one idempotent eligibility-and-mutation operation that synthesizes no unowned follow-up work.
- In ordinary armed `LazySignal::drop`, extract candidate identity and clone the sender under the lazy-state read lock, release that lock, then send the root.

Acceptance:

- Two-world signal isolation passes.
- Cross-world lazy reuse fails before entity access.
- Plugin construction cannot discard another world's request.
- Reaping a holder does not enqueue a duplicate inbox root.
- Explicit and ordinary release each decrement liveness exactly once.

### Phase 3: Migrate mutable vec atomically

- Add the private owner with entity, cleanup sender, and optional broadcaster plus `entity()`/`broadcaster()` accessors.
- Remove broadcaster ownership from generic data.
- Convert all direct constructors, including `FromWorld` and resource initialization, so each owner stores exactly the entity and broadcaster returned by its construction path.
- Convert `spawnc` and `Commands` constructors.
- Delete global vec queue/reset helpers.
- Remove legacy vec plugin references and test reset calls in the same commit.

Acceptance:

- Direct and deferred handles route only to their world.
- Immediate drop before deferred binding does not leak.
- Every constructor test proves final drop despawns its own stored entity and leaves neighboring entities intact.
- Broadcaster release is aggregated into the mutable root and its explicit candidate is transferred into the worklist for same-invocation processing.
- No mutable-data field destructor emits a follow-up inbox root.

### Phase 4: Migrate mutable map atomically

Mirror phase 3 for maps, including `FromWorld`, resource initialization, all signal-producing methods, plugin references, and test reset helpers.

Acceptance mirrors vec behavior.

### Phase 5: Simplify plugin and tests

- Install only the world-local drain.
- Remove global reset scaffolding.
- Run the full suite repeatedly with normal parallelism.
- Remove forced test serialization.

Acceptance:

- Parallel tests pass repeatedly.
- No static cleanup queues remain.
- No plugin setup path clears process-global lifecycle state.

## Test plan

### Cleanup module

- `BoundCleanupTarget` routes only to its exact inbox.
- `CleanupSender::Bound` has no `OnceLock`, pending queue, or deferred-route allocation; assert representation/size expectations where stable enough to be useful.
- A bound sender delegates directly through `BoundCleanupTarget::send`.
- A deferred sender/binder pair shares exactly one `DeferredCleanupRoute`.
- `CleanupBinder` is non-cloneable and consuming; add a compile-fail assertion if the project's test tooling supports one.
- A deferred sender buffers one and several requests before binding.
- Binding flushes buffered requests exactly once through the bound-target primitive.
- Send before bind, send after bind, and instrumented pauses after sender entry and after pending push lose and duplicate nothing.
- Intermediate handoff-state assertions cover unbound in-flight senders, `BOUND` with in-flight senders, and the final bound/zero-in-flight state.
- Several concurrent helper flushers deliver each uniquely numbered request exactly once; delivery order is not asserted.
- A completed multi-sender pre-bind batch is delivered as the exact produced multiset after binding.
- Checked in-flight overflow/underflow tests prove invalid updates leave the handoff word, including `BOUND`, unchanged.
- Deterministic barrier-controlled std tests cover send racing with bind and send racing with resource destruction.
- A stress test repeats concurrent send/bind batches and compares the delivered request multiset with the produced multiset.
- A test-only inbox weak reference no longer upgrades after world/resource destruction.
- An old bound target cannot send to a replacement inbox allocation.
- A send strictly after destruction is a no-op; a racing send may only reach an orphan inbox.
- A drain boundary selects only the root prefix present before the boundary and leaves the external suffix for the next invocation.
- Every `CleanupSender` send after the boundary, including destructor-generated sends during dispatch, remains in the external suffix.
- A deep finite chain of explicitly transferred move-only obligations drains to an empty worklist in one invocation with no retry constant.
- Duplicate, missing, and ineligible targets synthesize no new obligations; pre-owned candidate tokens are transferred exactly once.
- Test-only obligation accounting decreases monotonically across representative explicit local token and candidate chains.
- An intentionally invalid test handler that republishes an obligation is rejected by private API structure or detected by debug/test accounting.
- The live inbox and deferred pending queues are never closed during ordinary operation.

### Graph

- Two fresh worlds with colliding entity indices remain isolated.
- Draining/updating B first cannot remove A's stale signal.
- Duplicate stale signal requests are idempotent and do not grow the local worklist indefinitely.
- Explicit holder release decrements once without enqueueing a duplicate root.
- Ordinary external description drop sends one root; explicit candidate release sends none.
- Adding plugin B does not clear A's pending request.
- Cross-world registration of one lazy identity fails clearly.
- Signal registration before plugin setup initializes an inbox that plugin `init_resource` preserves.
- Queued pre-plugin signal cleanup is drained after plugin installation.

### Mutable vec/map

For each collection type and every constructor family (`From<&mut World>`, `FromWorld`/`world.init_resource`, builder `spawn`, builder `spawnc`, and `From<&mut Commands>`):

- Final clone drop enqueues exactly one request.
- Non-final clone drop does nothing.
- Updating another app first leaves its source intact.
- Direct world spawn cleans correctly.
- `spawnc` retained handle cleans correctly.
- `spawnc` immediate drop before command application cleans eventually when that command queue is applied; discarding the queue is safe because no entity is spawned.
- `From<&mut Commands>` follows the same behavior.
- Already-despawned and missing entity requests are no-ops.
- For each constructor, final drop removes exactly the spawned/reserved data entity and leaves neighboring entities intact.
- Permuting independent cleanup requests produces the same final world state.
- Dropping handles after world destruction is safe.
- Deferred bind before plugin setup initializes an inbox preserved by later plugin setup.
- A signal description retaining a mutable source owner prevents premature collection cleanup.
- Mutable data no longer owns the broadcaster; owner access preserves source behavior and introduces no `Arc` cycle.
- Final source-owner drop releases its broadcaster once and embeds the candidate in the aggregate root.
- Processing that root transfers the candidate to the local worklist and settles eligible signal cleanup in the same invocation.
- Any `CleanupSender` send occurring during that explicit worklist chain—including one logically related through an opaque destructor capture—remains in the inbox for the next scheduled drain.
- The opaque-runner regression proves that candidate reaping may queue a captured final owner's root behind the first boundary, the second drain settles it, and a third drain is idempotent.
- Multiple configured schedules share the same world inbox and obey documented root-snapshot timing.

### Feature matrix

At minimum:

```text
cargo test --locked --lib
cargo check --locked --no-default-features
cargo check --locked --no-default-features --features critical-section
cargo check --locked --no-default-features --features builder
```

Also validate the supported wasm configuration and at least one supported target without native pointer-width atomics. The unrelated optional-time test compilation problem remains separate.

Runtime concurrency tests run under `std`; compile-only no_std checks do not prove interrupt-context behavior.

## Bevy API assumptions to validate

- `World::id() -> WorldId` behavior and required traits.
- `World::get_resource_or_init`/`init_resource` availability in Bevy 0.18.
- `Commands::queue` ordering after `commands.spawn`.
- Deferred application boundaries relative to the cleanup drain system.
- `bevy_platform::sync::{Arc, Weak, OnceLock}` and `bevy_platform::sync::atomic::AtomicUsize` across `std`, `no_std`, critical-section, and wasm.
- The selected concurrent queue's FIFO linearization, `no_std`, allocator, native-atomic, wasm, and portable-atomic behavior.
- Required feature unification between the queue's atomic backend and Bevy's `critical-section` configuration.
- Resource teardown ordering.
- Behavior when a command queue is dropped without application.
- Registering one exclusive drain function in multiple configured schedules.

Every `CleanupSender` send is an external root. A send linearized after the current boundary is not guaranteed to clean in the same frame or invocation, even when current cleanup triggered it. Same-invocation processing is guaranteed only for explicit move-only obligations transferred into the current `CleanupWorklist`; deferred command ordering remains a separate concern.

## no_std and drop-context constraints

The initial implementation is compatible with allocator-backed no_std task contexts, not arbitrary interrupt/ISR drop contexts:

- An unbounded concurrent queue may allocate a new segment while enqueueing.
- `Arc`/`Weak`, `OnceLock`, the `AtomicUsize` handoff, and queue operations require atomic support or a configured atomic-emulation backend.
- On targets without suitable native atomics, `portable-atomic` or the platform backend may internally use critical sections; such execution is not strictly lock-free at the hardware level.
- Queue algorithms may spin transiently under contention even though Jonmo acquires no explicit mutex.
- Allocator use and request-payload destruction remain unsuitable for arbitrary interrupt context.

Document that cleanup-producing handles must be dropped from normal task/application context on no_std targets. Supporting ISR-safe cleanup would require a separately designed bounded, preallocated protocol with an explicit overflow policy; an unbounded cleanup queue cannot honestly provide that property. Host `critical-section` feature checks alone do not validate interrupt safety.

Jonmo's drop paths must not panic on queue errors. Failed sends to a dead weak inbox are safe no-ops; failure to push into an open live inbox is an internal invariant violation that should be surfaced outside unwinding-sensitive `Drop` paths where possible.

## Risks and mitigations

### Deferred sender complexity

Risk: send-before-bind races can strand requests in the private pending queue after the binder has already flushed it.

Mitigation: isolate this state machine in `DeferredCleanupRoute`; initialize the destination with `OnceLock`, but publish readiness through the same `AtomicUsize` used for checked sender entry/exit; make `BOUND` sticky; have the binder and senders that leave after publication help flush; add deterministic state-gap, checked-count, and repeated multiset stress tests. Do not rely on a second `OnceLock::get()` to order an independently synchronized pending-queue push.

### Divergent bound/deferred behavior

Risk: splitting sender representation accidentally creates two world-delivery semantics.

Mitigation: make `BoundCleanupTarget::send` the only inbox-publishing primitive. `CleanupSender::Bound` calls it directly; `DeferredCleanupRoute` only stages and eventually delegates to it. Run shared isolation, destruction, and drain tests against both constructors.

### Concurrent flush reordering

Risk: two helpers can pop pending requests in one order and enqueue them into the world inbox in another.

Mitigation: require every cleanup request kind to be idempotent and semantically order-independent. Encode ordered lifecycle work as one aggregate request or generation-checked transaction, never as order-dependent adjacent messages.

### Queue dependency and target coverage

Risk: the chosen queue or atomic backend does not compile on a supported `no_std`, wasm, or `critical-section` target, or uses emulation where documentation promises native lock-freedom.

Mitigation: keep the queue behind the private `CleanupInbox` abstraction, disable default features, run the complete feature/target matrix, audit feature unification, and document lock-freedom only for native-atomic targets.

### Unbounded enqueue allocation

Risk: a final-owner `Drop` allocates a queue segment and can fail under allocator exhaustion.

Mitigation: retain the existing allocator-backed normal-task-context support boundary; do not claim ISR or allocation-free cleanup. A bounded queue is rejected because overflow cannot safely discard cleanup ownership.

### Dead-world handles

Risk: externally retained handles keep cleanup resources alive forever.

Mitigation: bound senders retain only `Weak<CleanupInbox>`.

### Wrong-target despawn

Risk: a Jonmo implementation bug stores or sends an unrelated same-world entity.

Mitigation: keep owner fields and request construction private; construct each owner immediately from the spawn/reservation result; expose only typed private send helpers; use full generational entities and fallible despawn; test every constructor with neighboring entities and exact target assertions. Category markers are intentionally omitted because they do not validate exact ownership.

### Cleanup timing and causal classification

Risk: code assumes logical causality causes destructor-generated cleanup to join the current invocation.

Mitigation: define the boundary structurally. Every `CleanupSender` send is an external root. Known cleanup-owning fields may be explicitly taken, disarmed, and moved into `CleanupWorklist` when stronger same-invocation timing is required, but opaque captures are not introspected. Their ordinary drops remain safe external roots and may require the next scheduled drain.

### Local worklist termination

Risk: a handler synthesizes new ownership or republishes an obligation, creating a non-terminating local worklist now that there is no retry cap.

Mitigation: expose only typed append methods, require each append to consume/disarm a finite owned obligation or batch, make duplicates/missing targets non-generative, add test-only monotonic accounting, and review every new request variant against the termination measure.

### Mutable broadcaster ownership move

Risk: moving the broadcaster from generic ECS data into the shared source owner changes retention behavior or creates an `Arc` cycle.

Mitigation: keep broadcaster initializers dependent only on the data entity identity, inspect captured closures, test source/description/registration drop orders, and assert that final source cleanup occurs after the last dependent description but without a permanently retained owner.

### Overlap with issue 2

Risk: issue 1 embeds current registration-count assumptions that issue 2 must undo.

Mitigation: keep root routing and `CleanupWorklist` generic; preserve temporary signal eligibility behind a singular worklist-aware handler that issue 2 can replace. Avoid issue-1 APIs tied to signed counts or recursive topology traversal.

## Handoff to registration leases

Issue 2 replaces the temporary `StaleSignal` threshold/holder path with batched `LazyDefinitionCandidates(Arc<[SignalSystem]>)` work. A final description drop outside world access sends that batch as one world-local root. Candidate batches explicitly owned while releasing leases or reaping known definition components under `&mut World` append directly to the current local worklist. One ownership transition must not emit both old and new variants. Issue 2 should reuse:

- `WorldCleanupQueue`
- `CleanupSender`
- `CleanupWorklist`
- World-affinity validation
- Boundary-delimited root snapshots
- Explicit release/disarm semantics
- Monotonic local obligation consumption
- Dead-world no-op behavior

Registration leases cover acquisitions represented in registration blueprints and leases stored in known world components. They do not enumerate or extract arbitrary values captured by `SystemRunner`, Bevy systems, callbacks, or user closures. Explicit lease, slot, or candidate obligations already available under `&mut World` must move directly into the active `CleanupWorklist`; any request sent through `CleanupSender`, including one caused by reaping an opaque capture, remains an external root. Issue 2 must not introduce another lifecycle queue. Automatic dropped-registration-lease release remains deferred unless a later plan defines an owned exact-plan root and exactly-once semantics.

## Acceptance criteria

The issue is complete when:

- No process-global stale cleanup queues remain.
- No plugin path globally clears cleanup work.
- Cleanup enqueue, deferred handoff, and boundary root snapshotting use no Jonmo-managed mutex.
- Direct senders contain no deferred route, `OnceLock`, pending queue, handoff atomic, or extra route allocation.
- Deferred constructors create exactly one non-cloneable binder and one shared deferred route.
- Every direct producer holds one bound target; every successfully applied deferred constructor publishes exactly one bound target through its consumed binder.
- Direct handles clean eventually; deferred handles dropped before command application clean eventually only when the command queue is applied successfully.
- Every mutable constructor retains exactly its own full generational entity and broadcaster without category marker components.
- Final mutable owner drop emits one aggregate root containing any explicitly released broadcaster candidate.
- Signal release is exactly once across ordinary drop, candidate extraction, and holder disarm.
- Signal cleanup is duplicate-safe.
- Cross-world lazy reuse fails before entity access.
- Handles dropped after world destruction are safe.
- Concurrent send/bind/flush tests prove request-multiset preservation.
- Boundary tests prove every `CleanupSender` send is an external root selected only when it precedes the current boundary; all suffix roots are deferred.
- Explicit move-only worklist tests settle their mutable/signal obligations in one invocation without a retry constant.
- Test accounting demonstrates monotonic finite consumption of explicit local obligations without claiming to bound arbitrary destructor graphs.
- Manual test queue clearing is removed.
- The full library suite passes repeatedly with normal parallel execution.
- Supported no_std/critical-section/wasm checks pass under the documented allocator-backed non-ISR constraint.
