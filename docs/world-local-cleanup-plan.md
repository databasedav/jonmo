# World-Local Lock-Free Deferred Cleanup

## Status

Proposed implementation plan for issue 1: replace process-global stale cleanup queues with cleanup inboxes owned by individual Bevy `World`s.

This plan is the first step in the canonical landing order:

1. World-local cleanup.
2. [`collection-correctness-plan.md`](collection-correctness-plan.md).
3. [`registration-lease-plan.md`](registration-lease-plan.md).
4. [`move-first-signal-delivery-plan.md`](move-first-signal-delivery-plan.md).

It lands first so all later semantic and ownership tests can run reliably in parallel and subsequent lease work has one safe world-bound lifecycle inbox.

## Problem statement

Jonmo currently stores world-relative `Entity` identifiers in three process-global queues:

- `STALE_SIGNALS` in `src/graph.rs`
- `STALE_MUTABLE_VECS` in `src/signal_vec.rs`
- `STALE_MUTABLE_BTREE_MAPS` in `src/signal_map.rs`

Every app installs systems that drain those same globals. The first world to run consumes all pending requests, regardless of which world owns the entities. Entity IDs are only meaningful inside their originating `World`, so this can:

- Despawn unrelated entities in another world when IDs collide.
- Consume another world's request and leak the intended target.
- Make parallel tests interfere with one another.
- Lose pending work when another plugin instance clears the globals.

The current working tree also clears all three queues from `JonmoPlugin::build`. That must be removed; plugin construction in one app must never mutate another app's cleanup state.

## Decision summary

Adopt one private lock-free cleanup inbox per `World`:

1. Each world owns a `WorldCleanupQueue` resource backed by an unbounded concurrent queue.
2. Cleanup-producing handles retain either an immediately bound sender or a deferred sender created only for `Commands` construction.
3. `Drop` enqueues a typed request without requiring `&mut World` or acquiring a Jonmo-managed mutex.
4. A cleanup system drains only its own world's inbox.
5. `BoundCleanupTarget` is the sole primitive that delivers requests to a world and retains only a `Weak` inbox reference.
6. Direct `World` constructors store `CleanupSender::Bound` and pay no `OnceLock`, pending-queue, or extra route-allocation cost.
7. Deferred `Commands` constructors store `CleanupSender::Deferred` and queue one non-cloneable, consuming `CleanupBinder`.
8. Only the deferred route uses a `OnceLock`, private concurrent pending queue, and push-then-recheck protocol.
9. Drain passes insert one queue boundary and take only requests linearized before that boundary as root work.
10. Root handlers append causally related follow-ups to a local `CleanupWorklist`, never back into the concurrent inbox.
11. The local worklist runs to exhaustion in the same drain invocation with no retry-wave constant.
12. Every local append consumes or disarms a finite cleanup ownership obligation; handlers never create new cleanup ownership.
13. Cleanup requests are idempotent and order-independent; queue arrival order is never semantic ownership state.
14. Mutable vec/map owners privately retain the exact full generational entity and broadcaster ownership created for that source.
15. Duplicate, stale, late, and post-world-destruction requests are harmless.
16. No process-global map from `WorldId` to queue is introduced.
17. Cleanup remains deferred until a scheduled drain pass, but causal follow-ups normally settle in that same invocation.
18. Strict lock-freedom is claimed only on targets with suitable native atomics; atomic-emulation targets may internally use critical sections.

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
- Distinguish external/concurrent root requests from explicit same-invocation causal follow-ups.
- Settle finite causal cleanup chains without an arbitrary retry or wave limit.

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
13. A drain boundary selects a finite root prefix: requests linearized after it remain in the inbox for a later invocation.
14. No cleanup request handler runs until the complete root snapshot has been removed from the inbox.
15. Concurrent deferred send/bind/flush operations lose and duplicate no requests.
16. Cleanup requests are idempotent and semantically order-independent; concurrent pending flushes may reorder root requests.
17. Handlers publish causal follow-ups only to the invocation-local `CleanupWorklist`.
18. Every local follow-up is derived by consuming/disarming an owned cleanup obligation, releasing an acquisition, or transitioning an existing target toward removal.
19. No local handler creates a new cleanup owner, reacquires a released obligation, or republishes the same obligation.
20. Duplicate, missing, and ineligible targets synthesize no new obligation; independently owned candidate tokens already carried by a request are still transferred exactly once.
21. Each mutable vec/map owner stores exactly the full generational entity and broadcaster created by its private constructor.
22. Mutable root requests can only be constructed through those private owner paths and target the stored entity unchanged.
23. Signal cleanup acts only on valid signal entities that satisfy existing eligibility checks.
24. A lazy signal registered in one world cannot silently access an entity with a colliding ID in another world.
25. The request enum and worklist can later carry owned registration-definition candidate batches without another transport redesign.
26. Queue closure is not used during ordinary operation; an enqueue failure is an internal invariant violation, not permission to discard live-world cleanup.

## Proposed module

Add a private module:

```text
src/cleanup.rs
```

Declare it privately from `src/lib.rs`:

```rust
mod cleanup;
```

### Cleanup requests

Use a typed request enum rather than boxed arbitrary world closures:

```rust
pub(crate) struct ReleasedSignalCandidate(SignalSystem); // deliberately non-Clone

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

Mutable root requests aggregate the exact backing entity with any signal candidate produced by explicitly releasing that source's broadcaster. `ReleasedSignalCandidate` is a private move-only obligation token: identity may be copied out for validation, but the token itself cannot be cloned or appended twice through safe APIs. This preserves both causality and the local termination measure across the no-`World` boundary rather than relying on the broadcaster's field destructor to enqueue a second independent root.

Typed requests provide:

- Auditable centralized dispatch.
- A closed private request vocabulary.
- Predictable `Send + Sync` behavior.
- No arbitrary code execution from `Drop`.
- A stable place to add future lifecycle request kinds.

### Local causal worklist

The concurrent inbox is only for root work produced without `&mut World`. Once a root snapshot is collected, handlers use a private local worklist:

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

1. Drain ownership-release actions, including nested dynamic leases added by issue 2.
2. Accumulate/deduplicate the move-only node/definition candidate obligations produced by those releases.
3. Only after the release-action queue is empty, reap candidates in deterministic reverse-topological order.
4. If explicit component draining during reaping discovers another owned release action, return to the release phase before continuing candidate reaping.
5. Stop when both action and candidate stores are empty.

This ordering ensures an outer dynamic node is not tested for reaping while a nested inner lease still owns an edge into it. `CleanupWorklist` is not `Clone`, is never shared with producer threads, and has no general public `push` surface. Typed methods keep local action/candidate creation auditable. External senders cannot access it, so unrelated requests arriving during dispatch remain in the world inbox behind the previously consumed root boundary.

### Queue dependency and portability boundary

Use a maintained unbounded concurrent queue rather than implementing memory reclamation or lock-free linked storage inside Jonmo. The initial candidate is `concurrent-queue` with default features disabled:

```toml
concurrent-queue = { version = "2.5", default-features = false }
```

Reasons:

- It provides an unbounded concurrent queue suitable for multiple drop producers and one world consumer.
- It supports `no_std` with a global allocator.
- Its optional `std` and `portable-atomic` integration can be mapped to Jonmo's feature matrix after target validation.
- Jonmo keeps `unsafe_code = "deny"`; the queue algorithm remains in an audited dependency rather than new local unsafe code.

Before landing, verify the exact feature wiring for Jonmo's supported native-atomic, wasm, and `critical-section` targets. `crossbeam-queue::SegQueue` is an acceptable alternative only if its pointer-atomic target restrictions cover the same supported matrix. Do not implement an ad hoc atomic linked list.

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

`bevy_platform::sync::OnceLock` and the private pending queue exist only inside `DeferredCleanupRoute`. Directly bound senders do not allocate a deferred route and do not carry an unused concurrent queue.

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
2. Publishes it into the deferred route's `OnceLock`.
3. Treats an already-initialized target as an internal invariant violation, because producing or applying the unique binder twice should be impossible.
4. Flushes all currently pending requests through `BoundCleanupTarget::send`.
5. Consumes and drops the binder.

No request is moved as part of destination publication. Publishing and pending-queue flushing are separate concurrent operations, avoiding a compound state transition that would require a mutex.

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

#### Deferred push-then-recheck protocol

`DeferredCleanupRoute::send` uses this algorithm:

1. If `target.get()` already returns a destination, delegate directly to `BoundCleanupTarget::send`.
2. Otherwise, push the request into `pending`.
3. Recheck `target` after the push.
4. If binding became visible, help flush pending requests through the published bound target.
5. If the target is still unset, return; the unique binder performs the eventual flush.

The second target check closes the only apparent lost-request race:

- If the pending push linearizes before binding's flush, the binder or a concurrent helper pops it.
- If binding's flush finishes before the pending push, the sender's second check observes the target and performs another flush.
- If they overlap, concurrent pop operations remove each pending request at most once.

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
5. Leave messages linearized after the boundary, including unrelated concurrent drops, for a later drain invocation.

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

Handlers never publish causal follow-ups through `CleanupSender`; they append release actions or candidate obligations through typed `CleanupWorklist` methods. Candidate reaping starts only after the current release-action queue reaches a fixed point.

- Cleanup generated from the consumed ownership represented by a root settles in the same invocation.
- Unrelated external drops during dispatch remain in the concurrent inbox.
- No retry wave or arbitrary maximum iteration count is required.
- No queue operation encloses ECS mutation.

The boundary itself never reaches a request handler. If root extraction panics unexpectedly, no ECS mutation has begun; implementation and tests should keep message extraction non-panicking.

### Termination invariant

The local worklist is finite by ownership consumption, not by a retry constant. Every appended follow-up must be justified by one of these monotonic transitions:

- An armed cleanup-producing description/reference is explicitly released exactly once.
- A registration/edge acquisition is released exactly once.
- An existing entity or definition transitions toward removal exactly once.
- A finite owned candidate batch is transferred into the worklist.

Handlers may not construct new cleanup owners, reacquire released obligations, or synthesize follow-ups not backed by a consumed payload/obligation. A missing entity may still transfer an independently owned candidate carried by its root request, but may not derive additional entity cleanup. Define a test-only accounting snapshot for armed description references, active acquisitions, existing cleanup targets, and queued local requests; representative deep chains must monotonically consume that finite potential and terminate with an empty worklist.

This is a private-code invariant rather than protection against arbitrary user-supplied requests: `CleanupRequest` and unrestricted worklist insertion remain inaccessible outside the cleanup implementation and typed cooperating modules.

## Mutable ownership objects and explicit broadcaster release

Replace bespoke atomic final-owner detection for mutable collections with one private owner allocation per source. Move broadcaster ownership out of the generic ECS data component and into that authoritative source owner so final drop can package the causal signal candidate into one root request.

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

Issue 1 must stop one registered lazy identity from silently reusing its entity in another world and must separate ownership release from where the resulting candidate is delivered.

As temporary issue-1 protection, until issue 2 makes `SignalSystem` itself world-aware, the registered lazy state should retain:

- The originating `WorldId` or cleanup-inbox identity.
- The registered `SignalSystem`.
- The bound cleanup sender.

On subsequent registration:

1. Compare the current world with the stored world affinity.
2. Reject a mismatch before entity lookup or count mutation.
3. Never interpret an equal `Entity` from another world as the same signal.

Add one centralized exactly-once release primitive inside `LazySignal`. Each handle is armed when created/cloned and is disarmed only by an explicit consuming release method:

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

Before despawning a stale signal entity under issue 1's temporary holder model, remove its `LazySignalHolder` and call `release_reaped_holder`; do not let that holder's field destructor enqueue a duplicate root. Audit all entities despawned by cleanup so no armed cleanup-producing field is implicitly dropped when it should be released into the current causal transaction.

Supporting independent materialization of one lazy definition into several worlds requires a per-world lazy cache and is out of scope. Issue 2 replaces the temporary affinity and holder/reference-threshold details with world-aware `SignalSystem`, explicit `live_descriptions`, weak definition metadata, and candidate batches; the exactly-once explicit-release/worklist boundary remains.

## Plugin integration

Update `JonmoPlugin::build`:

1. Delete all `clear_stale_*` calls.
2. Call `app.init_resource::<WorldCleanupQueue>()` rather than inserting/replacing the resource.
3. Replace the three plural stale-drain systems with `drain_world_cleanup`.
4. Preserve current ordering after signal graph processing.
5. If multiple schedules are configured, registering the same exclusive drain function in each schedule is acceptable only after validating Bevy's system-instance and ordering behavior.

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

## Implementation phases

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
- Deferred `OnceLock`, concurrent pending queue, and push-then-recheck race handling
- Boundary-delimited root snapshotting
- Invocation-local `CleanupWorklist` and typed causal append methods
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
- External roots arriving after the boundary remain queued while explicit local follow-ups settle in the same invocation.
- The local worklist terminates by monotonic obligation consumption without a retry constant.
- Cleanup enqueue and root snapshot collection acquire no Jonmo-managed mutex.

### Phase 2: Migrate signals atomically

- Bind lazy materialization to one world before entity access.
- Initialize the cleanup resource on demand even when signals register before the plugin.
- Route ordinary no-`World` stale signal roots through that world's sender.
- Add armed exactly-once release, `into_stale_candidate`, and reaped-holder disarm paths.
- Make stale signal handlers transfer causal candidates through `CleanupWorklist` and explicitly release holders before despawn.
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
- Broadcaster release is aggregated into the mutable root and processed causally in the same drain invocation.
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
- Send before bind, send after bind, and all instrumented interleavings around publication/push/recheck lose and duplicate nothing.
- Several concurrent helper flushers deliver each uniquely numbered request exactly once; delivery order is not asserted.
- Deterministic barrier-controlled std tests cover send racing with bind and send racing with resource destruction.
- A stress test repeats concurrent send/bind batches and compares the delivered request multiset with the produced multiset.
- A test-only inbox weak reference no longer upgrades after world/resource destruction.
- An old bound target cannot send to a replacement inbox allocation.
- A send strictly after destruction is a no-op; a racing send may only reach an orphan inbox.
- A drain boundary selects only the root prefix present before the boundary and leaves the external suffix for the next invocation.
- Root handlers append explicit causal follow-ups locally; unrelated concurrent roots remain in the inbox.
- A deep finite causal chain drains to an empty worklist in one invocation with no retry constant.
- Duplicate, missing, and ineligible targets synthesize no new obligations; pre-owned candidate tokens are transferred exactly once.
- Test-only obligation accounting decreases monotonically across representative mutable/signal chains.
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
- An unrelated concurrent root arriving during that causal chain remains in the inbox for the next invocation.
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
- `bevy_platform::sync::{Arc, Weak, OnceLock}` across `std`, `no_std`, critical-section, and wasm.
- The selected concurrent queue's FIFO linearization, `no_std`, allocator, native-atomic, wasm, and portable-atomic behavior.
- Required feature unification between the queue's atomic backend and Bevy's `critical-section` configuration.
- Resource teardown ordering.
- Behavior when a command queue is dropped without application.
- Registering one exclusive drain function in multiple configured schedules.

Do not promise that a root dropped after the current boundary cleans in the same frame. Once a root is selected, explicitly represented finite causal follow-ups should settle in that drain invocation; deferred command ordering still requires tests.

## no_std and drop-context constraints

The initial implementation is compatible with allocator-backed no_std task contexts, not arbitrary interrupt/ISR drop contexts:

- An unbounded concurrent queue may allocate a new segment while enqueueing.
- `Arc`/`Weak`, `OnceLock`, and queue operations require atomic support or a configured atomic-emulation backend.
- On targets without suitable native atomics, `portable-atomic` or the platform backend may internally use critical sections; such execution is not strictly lock-free at the hardware level.
- Queue algorithms may spin transiently under contention even though Jonmo acquires no explicit mutex.
- Allocator use and request-payload destruction remain unsuitable for arbitrary interrupt context.

Document that cleanup-producing handles must be dropped from normal task/application context on no_std targets. Supporting ISR-safe cleanup would require a separately designed bounded, preallocated protocol with an explicit overflow policy; an unbounded cleanup queue cannot honestly provide that property. Host `critical-section` feature checks alone do not validate interrupt safety.

Jonmo's drop paths must not panic on queue errors. Failed sends to a dead weak inbox are safe no-ops; failure to push into an open live inbox is an internal invariant violation that should be surfaced outside unwinding-sensitive `Drop` paths where possible.

## Risks and mitigations

### Deferred sender complexity

Risk: send-before-bind races can strand requests in the private pending queue after the binder has already flushed it.

Mitigation: isolate this state machine in `DeferredCleanupRoute`; publish the destination with `OnceLock`; every pre-bind send pushes before rechecking the target; both the unique binder and deferred senders help flush; add deterministic interleaving and repeated multiset stress tests.

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

Risk: a cleanup-producing value is dropped implicitly during a handler instead of being explicitly released, so its causally related request goes to the concurrent inbox and waits for a later invocation.

Mitigation: audit every cleanup-time despawn/removal for armed owners; extract/disarm known holders before despawn; aggregate mutable broadcaster release into the root; add tests asserting the inbox does not grow during representative local causal chains. Truly external requests linearized after the root boundary intentionally wait for a later invocation.

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

After this plan lands, issue 2 replaces the temporary `StaleSignal` threshold/holder path with batched `LazyDefinitionCandidates(Arc<[SignalSystem]>)` work. A final description drop outside world access sends that batch as one world-local root. Candidate batches discovered while releasing leases or reaping definitions under `&mut World` append directly to the current local worklist. One ownership transition must not emit both old and new variants. Issue 2 should reuse:

- `WorldCleanupQueue`
- `CleanupSender`
- `CleanupWorklist`
- World-affinity validation
- Boundary-delimited root snapshots
- Explicit release/disarm semantics
- Monotonic local obligation consumption
- Dead-world no-op behavior

Issue 2 must not introduce another lifecycle queue or route causal lease/reaping follow-ups back through the concurrent inbox. Automatic dropped-registration-lease release remains explicitly deferred unless a later plan defines an owned exact-plan root and exactly-once semantics.

## Acceptance criteria

The issue is complete when:

- No process-global stale cleanup queues remain.
- No plugin path globally clears cleanup work.
- Cleanup enqueue, deferred handoff, and boundary root snapshotting use no Jonmo-managed mutex.
- Direct senders contain no deferred route, `OnceLock`, pending queue, or extra route allocation.
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
- Boundary tests prove each root snapshot is finite and defers unrelated suffix roots.
- Explicit causal worklist tests settle related mutable/signal cleanup in one invocation without a retry constant.
- Test accounting demonstrates monotonic finite obligation consumption.
- Manual test queue clearing is removed.
- The full library suite passes repeatedly with normal parallel execution.
- Supported no_std/critical-section/wasm checks pass under the documented allocator-backed non-ISR constraint.
