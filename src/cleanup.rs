use alloc::{collections::VecDeque, vec::Vec};

use bevy_ecs::{
    entity::Entity,
    prelude::{Resource, World},
};
use bevy_platform::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicUsize, Ordering},
};
use concurrent_queue::{ConcurrentQueue, PopError, PushError};

use crate::graph::{PreflightedSignalClaimRelease, ReleasedSignalCandidate, ReleasedStaticDependencyClaim};

pub(crate) fn update_atomic_usize<F>(
    atomic: &AtomicUsize,
    set_order: Ordering,
    fetch_order: Ordering,
    update: F,
) -> Result<usize, usize>
where
    F: FnMut(usize) -> Option<usize>,
{
    #[cfg(target_has_atomic = "ptr")]
    {
        atomic.try_update(set_order, fetch_order, update)
    }

    #[cfg(not(target_has_atomic = "ptr"))]
    {
        atomic.fetch_update(set_order, fetch_order, update)
    }
}

fn push_open<T>(queue: &ConcurrentQueue<T>, value: T, name: &str) {
    match queue.push(value) {
        Ok(()) => {}
        Err(PushError::Full(_)) => panic!("unbounded {name} unexpectedly reported full"),
        Err(PushError::Closed(_)) => panic!("live {name} unexpectedly closed"),
    }
}

fn pop_open<T>(queue: &ConcurrentQueue<T>, name: &str) -> Option<T> {
    match queue.pop() {
        Ok(value) => Some(value),
        Err(PopError::Empty) => None,
        Err(PopError::Closed) => panic!("live {name} unexpectedly closed"),
    }
}

#[cfg(test)]
struct TestDropProbe(Arc<AtomicUsize>);

#[cfg(test)]
impl Drop for TestDropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

// Request construction stays coupled to the typed sender methods below.
enum CleanupRequest {
    StaleSignal(ReleasedSignalCandidate),
    IndexLookup(Entity),
    MutableVec(Entity),
    MutableBTreeMap(Entity),
    #[cfg(test)]
    Test(usize),
    #[cfg(test)]
    Tracked(TestDropProbe),
}

enum CleanupMessage {
    Request(CleanupRequest),
    DrainBoundary,
}

struct CleanupInbox {
    messages: ConcurrentQueue<CleanupMessage>,
}

impl Default for CleanupInbox {
    fn default() -> Self {
        Self {
            messages: ConcurrentQueue::unbounded(),
        }
    }
}

impl CleanupInbox {
    fn push_message(&self, message: CleanupMessage) {
        push_open(&self.messages, message, "world cleanup inbox");
    }

    fn pop_message(&self) -> Option<CleanupMessage> {
        pop_open(&self.messages, "world cleanup inbox")
    }

    fn collect_roots(&self) -> Vec<CleanupRequest> {
        self.collect_roots_with_hook(|| {} as ())
    }

    fn collect_roots_with_hook<F>(&self, after_boundary: F) -> Vec<CleanupRequest>
    where
        F: FnOnce(),
    {
        self.push_message(CleanupMessage::DrainBoundary);
        after_boundary();

        let mut roots = Vec::new();
        loop {
            match self.pop_message() {
                Some(CleanupMessage::Request(request)) => roots.push(request),
                Some(CleanupMessage::DrainBoundary) => break,
                None => panic!("cleanup drain boundary disappeared from its open inbox"),
            }
        }
        roots
    }
}

/// The unique cleanup inbox allocation owned by one [`World`].
#[derive(Resource, Default)]
pub(crate) struct WorldCleanupQueue {
    inbox: Arc<CleanupInbox>,
}

/// A weak capability targeting one exact world-inbox allocation.
#[derive(Clone)]
pub(crate) struct BoundCleanupTarget {
    inbox: Weak<CleanupInbox>,
}

impl BoundCleanupTarget {
    fn new(world: &mut World) -> Self {
        let queue = world.get_resource_or_init::<WorldCleanupQueue>();
        Self {
            inbox: Arc::downgrade(&queue.inbox),
        }
    }

    fn send(&self, message: CleanupMessage) {
        let Some(inbox) = self.inbox.upgrade() else {
            return;
        };
        inbox.push_message(message);
    }

    #[cfg(test)]
    fn send_test(&self, id: usize) {
        self.send(CleanupMessage::Request(CleanupRequest::Test(id)));
    }
}

/// A cleanup route that is either bound to one exact inbox or awaiting one command binding.
#[derive(Clone)]
pub(crate) enum CleanupSender {
    Bound(BoundCleanupTarget),
    Deferred(Arc<DeferredCleanupRoute>),
}

impl CleanupSender {
    pub(crate) fn bound(world: &mut World) -> Self {
        Self::Bound(BoundCleanupTarget::new(world))
    }

    pub(crate) fn deferred() -> (Self, CleanupBinder) {
        let route = Arc::new(DeferredCleanupRoute::default());
        (Self::Deferred(route.clone()), CleanupBinder { route })
    }

    pub(crate) fn send_stale_signal(&self, candidate: ReleasedSignalCandidate) {
        self.send(CleanupRequest::StaleSignal(candidate));
    }

    pub(crate) fn send_index_lookup(&self, entity: Entity) {
        self.send(CleanupRequest::IndexLookup(entity));
    }

    pub(crate) fn send_mutable_vec(&self, entity: Entity) {
        self.send(CleanupRequest::MutableVec(entity));
    }

    pub(crate) fn send_mutable_btree_map(&self, entity: Entity) {
        self.send(CleanupRequest::MutableBTreeMap(entity));
    }

    #[cfg(test)]
    pub(crate) fn inbox_identity(&self) -> Option<usize> {
        let target = match self {
            Self::Bound(target) => Some(target),
            Self::Deferred(route) => route.target.get(),
        }?;
        Some(target.inbox.as_ptr() as usize)
    }

    fn send(&self, request: CleanupRequest) {
        match self {
            Self::Bound(target) => target.send(CleanupMessage::Request(request)),
            Self::Deferred(route) => route.send(request),
        }
    }

    #[cfg(test)]
    fn send_test(&self, id: usize) {
        self.send(CleanupRequest::Test(id));
    }

    #[cfg(test)]
    fn send_tracked(&self, probe: TestDropProbe) {
        self.send(CleanupRequest::Tracked(probe));
    }
}

/// Stages requests until its one destination is published by [`CleanupBinder`].
pub(crate) struct DeferredCleanupRoute {
    target: OnceLock<BoundCleanupTarget>,
    pending: ConcurrentQueue<CleanupRequest>,
    handoff: AtomicUsize,
}

impl Default for DeferredCleanupRoute {
    fn default() -> Self {
        Self {
            target: OnceLock::new(),
            pending: ConcurrentQueue::unbounded(),
            handoff: AtomicUsize::new(0),
        }
    }
}

const HANDOFF_BOUND: usize = 1usize << (usize::BITS - 1);
const HANDOFF_IN_FLIGHT_MASK: usize = HANDOFF_BOUND - 1;

impl DeferredCleanupRoute {
    fn send(&self, request: CleanupRequest) {
        self.send_with_hooks(request, || {}, || {});
    }

    fn send_with_hooks<F, G>(&self, request: CleanupRequest, after_enter: F, after_pending_push: G)
    where
        F: FnOnce(),
        G: FnOnce(),
    {
        let entered = self.enter_send();
        if entered & HANDOFF_BOUND != 0 {
            self.published_target().send(CleanupMessage::Request(request));
            let leaving = self.leave_send();
            debug_assert_ne!(leaving & HANDOFF_BOUND, 0);
            return;
        }

        after_enter();
        push_open(&self.pending, request, "deferred cleanup queue");
        after_pending_push();

        let leaving = self.leave_send();
        if leaving & HANDOFF_BOUND != 0 {
            self.flush_pending(self.published_target());
        }
    }

    fn enter_send(&self) -> usize {
        update_atomic_usize(&self.handoff, Ordering::AcqRel, Ordering::Acquire, |state| {
            let in_flight = state & HANDOFF_IN_FLIGHT_MASK;
            if in_flight == HANDOFF_IN_FLIGHT_MASK {
                None
            } else {
                Some(state + 1)
            }
        })
        .unwrap_or_else(|state| {
            panic!(
                "deferred cleanup in-flight sender count overflowed at {}",
                state & HANDOFF_IN_FLIGHT_MASK
            )
        })
    }

    fn leave_send(&self) -> usize {
        update_atomic_usize(&self.handoff, Ordering::AcqRel, Ordering::Acquire, |state| {
            let in_flight = state & HANDOFF_IN_FLIGHT_MASK;
            if in_flight == 0 { None } else { Some(state - 1) }
        })
        .unwrap_or_else(|_| panic!("deferred cleanup in-flight sender count underflowed"))
    }

    fn publish_bound(&self) -> usize {
        self.handoff.fetch_or(HANDOFF_BOUND, Ordering::AcqRel)
    }

    fn published_target(&self) -> &BoundCleanupTarget {
        self.target
            .get()
            .unwrap_or_else(|| panic!("deferred cleanup target publication was not initialized"))
    }

    fn flush_pending(&self, target: &BoundCleanupTarget) {
        while let Some(request) = pop_open(&self.pending, "deferred cleanup queue") {
            target.send(CleanupMessage::Request(request));
        }
    }
}

/// The unique consuming capability that binds one deferred cleanup route.
pub(crate) struct CleanupBinder {
    route: Arc<DeferredCleanupRoute>,
}

impl CleanupBinder {
    pub(crate) fn bind(self, world: &mut World) {
        self.bind_with_hook(world, || {});
    }

    fn bind_with_hook<F>(self, world: &mut World, after_publish: F)
    where
        F: FnOnce(),
    {
        let target = BoundCleanupTarget::new(world);
        if self.route.target.set(target.clone()).is_err() {
            panic!("deferred cleanup route was bound more than once");
        }
        let previous = self.route.publish_bound();
        assert_eq!(
            previous & HANDOFF_BOUND,
            0,
            "deferred cleanup route was bound more than once"
        );
        after_publish();
        self.route.flush_pending(&target);
    }
}

fn collect_external_roots(world: &mut World) -> Vec<CleanupRequest> {
    let inbox = world.get_resource_or_init::<WorldCleanupQueue>().inbox.clone();
    inbox.collect_roots()
}

enum CleanupAction {
    Root(CleanupRequest),
    ReleaseSignalClaim(PreflightedSignalClaimRelease),
    ReleaseStaticDependency(ReleasedStaticDependencyClaim),
    ReleaseDynamicConnection(crate::graph::PreflightedDynamicConnectionRelease),
    #[cfg(test)]
    TestRelease(TestReleaseAction),
}

#[cfg(test)]
struct TestReleaseAction {
    candidates: Vec<ReleasedSignalCandidate>,
}

/// Invocation-local causal cleanup state, with release actions always taking priority.
pub(crate) struct CleanupWorklist {
    actions: VecDeque<CleanupAction>,
    signal_candidates: Vec<ReleasedSignalCandidate>,
}

impl CleanupWorklist {
    fn from_roots(roots: Vec<CleanupRequest>) -> Self {
        Self {
            actions: roots.into_iter().map(CleanupAction::Root).collect(),
            signal_candidates: Vec::new(),
        }
    }

    fn from_signal_claim(release: PreflightedSignalClaimRelease) -> Self {
        Self {
            actions: VecDeque::from([CleanupAction::ReleaseSignalClaim(release)]),
            signal_candidates: Vec::new(),
        }
    }

    fn from_signal_candidates(candidates: Vec<ReleasedSignalCandidate>) -> Self {
        Self {
            actions: VecDeque::new(),
            signal_candidates: candidates,
        }
    }

    fn pop_action(&mut self) -> Option<CleanupAction> {
        self.actions.pop_front()
    }

    fn take_signal_candidates(&mut self) -> Vec<ReleasedSignalCandidate> {
        assert!(
            self.actions.is_empty(),
            "signal candidates cannot be taken before release actions reach a fixed point"
        );
        core::mem::take(&mut self.signal_candidates)
    }

    pub(crate) fn push_stale_signal(&mut self, candidate: ReleasedSignalCandidate) {
        self.signal_candidates.push(candidate);
    }

    pub(crate) fn push_signal_claim_release(&mut self, release: PreflightedSignalClaimRelease) {
        self.actions.push_front(CleanupAction::ReleaseSignalClaim(release));
    }

    pub(crate) fn push_dynamic_connection_releases(
        &mut self,
        releases: impl IntoIterator<Item = crate::graph::PreflightedDynamicConnectionRelease, IntoIter: DoubleEndedIterator>,
    ) {
        for release in releases.into_iter().rev() {
            self.actions
                .push_front(CleanupAction::ReleaseDynamicConnection(release));
        }
    }

    pub(crate) fn push_static_dependency_releases(
        &mut self,
        releases: impl IntoIterator<Item = ReleasedStaticDependencyClaim, IntoIter: DoubleEndedIterator>,
    ) {
        for release in releases.into_iter().rev() {
            self.actions.push_front(CleanupAction::ReleaseStaticDependency(release));
        }
    }

    pub(crate) fn has_release_actions(&self) -> bool {
        !self.actions.is_empty()
    }

    fn is_empty(&self) -> bool {
        self.actions.is_empty() && self.signal_candidates.is_empty()
    }

    #[cfg(test)]
    fn push_test_release(&mut self, action: TestReleaseAction) {
        self.actions.push_back(CleanupAction::TestRelease(action));
    }
}

fn dispatch_cleanup_action(world: &mut World, action: CleanupAction, worklist: &mut CleanupWorklist) {
    match action {
        CleanupAction::Root(request) => match request {
            CleanupRequest::StaleSignal(candidate) => worklist.push_stale_signal(candidate),
            CleanupRequest::IndexLookup(entity)
            | CleanupRequest::MutableVec(entity)
            | CleanupRequest::MutableBTreeMap(entity) => {
                if let Ok(entity_mut) = world.get_entity_mut(entity) {
                    entity_mut.despawn();
                }
            }
            #[cfg(test)]
            CleanupRequest::Test(_) => {}
            #[cfg(test)]
            CleanupRequest::Tracked(probe) => drop(probe),
        },
        CleanupAction::ReleaseSignalClaim(release) => {
            crate::graph::commit_preflighted_signal_claim_release(world, release, worklist);
        }
        CleanupAction::ReleaseStaticDependency(release) => {
            crate::graph::commit_released_static_dependency_claim(world, release, worklist);
        }
        CleanupAction::ReleaseDynamicConnection(release) => {
            crate::graph::commit_preflighted_dynamic_connection_release(world, release, worklist);
        }
        #[cfg(test)]
        CleanupAction::TestRelease(release) => {
            for candidate in release.candidates {
                worklist.push_stale_signal(candidate);
            }
        }
    }
}

fn process_cleanup_worklist(world: &mut World, mut worklist: CleanupWorklist) {
    loop {
        while let Some(action) = worklist.pop_action() {
            dispatch_cleanup_action(world, action, &mut worklist);
        }

        let candidates = worklist.take_signal_candidates();
        if candidates.is_empty() {
            break;
        }

        crate::graph::reap_signal_candidates(world, candidates, &mut worklist);
    }

    debug_assert!(worklist.is_empty());
}

fn process_world_cleanup_roots(world: &mut World, roots: Vec<CleanupRequest>) {
    process_cleanup_worklist(world, CleanupWorklist::from_roots(roots));
}

pub(crate) fn process_signal_claim_release(world: &mut World, release: PreflightedSignalClaimRelease) {
    process_cleanup_worklist(world, CleanupWorklist::from_signal_claim(release));
}

pub(crate) fn process_signal_candidates(world: &mut World, candidates: Vec<ReleasedSignalCandidate>) {
    process_cleanup_worklist(world, CleanupWorklist::from_signal_candidates(candidates));
}

pub(crate) fn process_dynamic_connection_releases(
    world: &mut World,
    releases: Vec<crate::graph::PreflightedDynamicConnectionRelease>,
) {
    let mut worklist = CleanupWorklist::from_signal_candidates(Vec::new());
    worklist.push_dynamic_connection_releases(releases);
    process_cleanup_worklist(world, worklist);
}

pub(crate) fn drain_world_cleanup(world: &mut World) {
    let roots = collect_external_roots(world);
    process_world_cleanup_roots(world, roots);
}

#[cfg(test)]
pub(crate) fn drain_world_cleanup_with_stale_signal_roots(
    world: &mut World,
    inspect: impl FnOnce(&[crate::graph::SignalEntity]),
) {
    let roots = collect_external_roots(world);
    let signals = roots
        .iter()
        .map(|request| match request {
            CleanupRequest::StaleSignal(candidate) => candidate.signal(),
            _ => panic!("expected only stale-signal cleanup roots"),
        })
        .collect::<Vec<_>>();
    inspect(&signals);
    process_world_cleanup_roots(world, roots);
}

#[cfg(test)]
pub(crate) fn queued_message_count(world: &World) -> usize {
    world
        .get_resource::<WorldCleanupQueue>()
        .map(|queue| queue.inbox.messages.len())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) fn cleanup_inbox_identity(world: &World) -> Option<usize> {
    world
        .get_resource::<WorldCleanupQueue>()
        .map(|queue| Arc::as_ptr(&queue.inbox) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use bevy_platform::sync::Barrier;
    use bevy_platform::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(not(target_arch = "wasm32"))]
    use std::thread;

    fn request_ids(requests: Vec<CleanupRequest>) -> Vec<usize> {
        requests
            .into_iter()
            .map(|request| match request {
                CleanupRequest::Test(id) => id,
                _ => panic!("expected a test cleanup request"),
            })
            .collect()
    }

    fn candidate(world: &mut World) -> ReleasedSignalCandidate {
        ReleasedSignalCandidate::test_new(world.spawn_empty().id().into())
    }

    #[test]
    fn index_lookup_cleanup_is_one_data_only_action() {
        let mut world = World::new();
        let sender = CleanupSender::bound(&mut world);
        let entity = world.spawn_empty().id();
        let neighbor = world.spawn_empty().id();
        sender.send_index_lookup(entity);
        assert_eq!(queued_message_count(&world), 1);
        let roots = collect_external_roots(&mut world);
        assert!(matches!(roots.as_slice(), [CleanupRequest::IndexLookup(target)] if *target == entity));
        let mut worklist = CleanupWorklist::from_roots(roots);
        let action = worklist.pop_action().unwrap();
        dispatch_cleanup_action(&mut world, action, &mut worklist);
        assert!(
            worklist.is_empty(),
            "data cleanup must not introduce graph candidates or releases"
        );
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
        assert_eq!(queued_message_count(&world), 0);
    }

    #[test]
    fn index_lookup_cleanup_preserves_reused_generation() {
        let mut world = World::new();
        let sender = CleanupSender::bound(&mut world);
        let stale = world.spawn_empty().id();
        sender.send_index_lookup(stale);
        let next_gen = world.despawn_no_free(stale).unwrap();
        let replacement = world.spawn_empty_at(next_gen).unwrap().id();
        assert_eq!(stale.index(), replacement.index());
        assert_ne!(stale.generation(), replacement.generation());
        drain_world_cleanup(&mut world);
        assert!(world.get_entity(replacement).is_ok());
        assert_eq!(queued_message_count(&world), 0);
    }

    #[test]
    fn index_lookup_cleanup_targets_original_inbox_allocation() {
        let mut world = World::new();
        let sender = CleanupSender::bound(&mut world);
        let entity = world.spawn_empty().id();
        let original = world.remove_resource::<WorldCleanupQueue>().unwrap();
        world.init_resource::<WorldCleanupQueue>();
        sender.send_index_lookup(entity);
        assert_eq!(queued_message_count(&world), 0);
        let roots = original.inbox.collect_roots();
        assert!(matches!(roots.as_slice(), [CleanupRequest::IndexLookup(target)] if *target == entity));
        drop(original);
        sender.send_index_lookup(entity);
        drain_world_cleanup(&mut world);
        assert!(
            world.get_entity(entity).is_ok(),
            "an expired route must not target a replacement inbox"
        );
    }

    #[test]
    fn bound_targets_isolate_worlds() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let sender_a = CleanupSender::bound(&mut world_a);
        let sender_b = CleanupSender::bound(&mut world_b);

        sender_a.send_test(1);
        sender_b.send_test(2);

        assert_eq!(request_ids(collect_external_roots(&mut world_a)), [1]);
        assert_eq!(request_ids(collect_external_roots(&mut world_b)), [2]);
    }

    #[test]
    fn on_demand_initialization_preserves_one_exact_inbox_allocation() {
        let mut world = World::new();
        let first = BoundCleanupTarget::new(&mut world);
        let second = BoundCleanupTarget::new(&mut world);

        assert!(Weak::ptr_eq(&first.inbox, &second.inbox));
    }

    #[test]
    fn bound_target_routes_only_to_its_exact_inbox() {
        let mut world = World::new();
        let target = BoundCleanupTarget::new(&mut world);
        let inbox = world.resource::<WorldCleanupQueue>().inbox.clone();

        target.send_test(7);

        assert_eq!(request_ids(inbox.collect_roots()), [7]);
    }

    #[test]
    fn dead_bound_target_is_a_no_op() {
        let target = {
            let mut world = World::new();
            BoundCleanupTarget::new(&mut world)
        };

        target.send_test(1);
        assert!(target.inbox.upgrade().is_none());
    }

    #[test]
    fn old_target_cannot_reach_replacement_inbox() {
        let mut world = World::new();
        let old_target = BoundCleanupTarget::new(&mut world);
        let old_inbox = world.remove_resource::<WorldCleanupQueue>().unwrap().inbox;
        assert!(old_target.inbox.upgrade().is_some());
        drop(old_inbox);
        assert!(old_target.inbox.upgrade().is_none());

        let new_target = BoundCleanupTarget::new(&mut world);
        old_target.send_test(1);
        new_target.send_test(2);

        assert_eq!(request_ids(collect_external_roots(&mut world)), [2]);
    }

    #[test]
    fn direct_sender_has_only_the_bound_representation() {
        let mut world = World::new();
        let sender = CleanupSender::bound(&mut world);

        let CleanupSender::Bound(target) = &sender else {
            panic!("direct sender allocated a deferred route");
        };
        assert_eq!(
            target.inbox.as_ptr(),
            Arc::as_ptr(&world.resource::<WorldCleanupQueue>().inbox)
        );
        assert_eq!(Arc::strong_count(&world.resource::<WorldCleanupQueue>().inbox), 1);
    }

    #[test]
    fn deferred_sender_and_binder_share_one_route() {
        let (sender, binder) = CleanupSender::deferred();
        let CleanupSender::Deferred(route) = &sender else {
            panic!("expected deferred sender");
        };

        assert!(Arc::ptr_eq(route, &binder.route));
        assert_eq!(Arc::strong_count(route), 2);
        assert!(route.target.get().is_none());
        assert_eq!(route.handoff.load(Ordering::Acquire), 0);
    }

    #[test]
    fn send_fully_before_bind_is_delivered_once() {
        let mut world = World::new();
        let (sender, binder) = CleanupSender::deferred();
        sender.send_test(1);

        binder.bind(&mut world);

        assert_eq!(request_ids(collect_external_roots(&mut world)), [1]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn simultaneous_senders_complete_before_binding_without_loss() {
        const SENDERS: usize = 32;
        let mut world = World::new();
        let (sender, binder) = CleanupSender::deferred();
        let start = Arc::new(Barrier::new(SENDERS + 1));
        let mut threads = Vec::new();

        for id in 0..SENDERS {
            let sender = sender.clone();
            let start = start.clone();
            threads.push(thread::spawn(move || {
                start.wait();
                sender.send_test(id);
            }));
        }

        start.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        let CleanupSender::Deferred(route) = &sender else {
            unreachable!();
        };
        assert_eq!(route.handoff.load(Ordering::Acquire), 0);

        binder.bind(&mut world);

        let mut delivered = request_ids(collect_external_roots(&mut world));
        delivered.sort_unstable();
        assert_eq!(delivered, (0..SENDERS).collect::<Vec<_>>());
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND);
    }

    #[test]
    fn bind_fully_before_send_is_delivered_once() {
        let mut world = World::new();
        let (sender, binder) = CleanupSender::deferred();
        binder.bind(&mut world);

        sender.send_test(1);

        assert_eq!(request_ids(collect_external_roots(&mut world)), [1]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn binder_flushes_while_sender_is_paused_before_pending_push() {
        let mut world = World::new();
        let (sender, binder) = CleanupSender::deferred();
        let CleanupSender::Deferred(route) = sender else {
            unreachable!();
        };
        let reached_gap = Arc::new(Barrier::new(2));
        let resume = reached_gap.clone();
        let thread_route = route.clone();

        let sending = thread::spawn(move || {
            thread_route.send_with_hooks(
                CleanupRequest::Test(1),
                || {
                    resume.wait();
                    resume.wait();
                },
                || {},
            );
        });
        reached_gap.wait();
        assert_eq!(route.handoff.load(Ordering::Acquire), 1);
        binder.bind(&mut world);
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND | 1);
        reached_gap.wait();
        sending.join().unwrap();

        assert_eq!(request_ids(collect_external_roots(&mut world)), [1]);
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn bind_between_pending_push_and_sender_exit_is_delivered_once() {
        let mut world = World::new();
        let (sender, binder) = CleanupSender::deferred();
        let CleanupSender::Deferred(route) = sender else {
            unreachable!();
        };
        let reached_gap = Arc::new(Barrier::new(2));
        let resume = reached_gap.clone();
        let thread_route = route.clone();

        let sending = thread::spawn(move || {
            thread_route.send_with_hooks(
                CleanupRequest::Test(1),
                || {},
                || {
                    resume.wait();
                    resume.wait();
                },
            );
        });
        reached_gap.wait();
        assert_eq!(route.handoff.load(Ordering::Acquire), 1);
        binder.bind(&mut world);
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND | 1);
        reached_gap.wait();
        sending.join().unwrap();

        assert_eq!(request_ids(collect_external_roots(&mut world)), [1]);
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn concurrent_binder_and_sender_flushes_do_not_duplicate() {
        const SENDERS: usize = 16;
        let mut world = World::new();
        let (sender, binder) = CleanupSender::deferred();
        let CleanupSender::Deferred(route) = sender else {
            unreachable!();
        };
        let pending_ready = Arc::new(Barrier::new(SENDERS + 1));
        let target_published = Arc::new(Barrier::new(SENDERS + 1));
        let mut threads = Vec::new();

        for id in 0..SENDERS {
            let route = route.clone();
            let pending_ready = pending_ready.clone();
            let target_published = target_published.clone();
            threads.push(thread::spawn(move || {
                route.send_with_hooks(
                    CleanupRequest::Test(id),
                    || {},
                    || {
                        pending_ready.wait();
                        target_published.wait();
                    },
                );
            }));
        }

        pending_ready.wait();
        binder.bind_with_hook(&mut world, || {
            target_published.wait();
        });
        for thread in threads {
            thread.join().unwrap();
        }

        let mut delivered = request_ids(collect_external_roots(&mut world));
        delivered.sort_unstable();
        assert_eq!(delivered, (0..SENDERS).collect::<Vec<_>>());
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND);
    }

    #[test]
    fn in_flight_count_errors_do_not_modify_handoff_state() {
        let route = DeferredCleanupRoute::default();
        route.handoff.store(HANDOFF_IN_FLIGHT_MASK, Ordering::Release);
        let overflow = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| route.enter_send()));
        assert!(overflow.is_err());
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_IN_FLIGHT_MASK);

        route.handoff.store(HANDOFF_BOUND, Ordering::Release);
        let underflow = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| route.leave_send()));
        assert!(underflow.is_err());
        assert_eq!(route.handoff.load(Ordering::Acquire), HANDOFF_BOUND);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn deferred_send_bind_stress_preserves_the_request_multiset() {
        const ROUNDS: usize = 32;
        const SENDERS: usize = 8;
        const PER_SENDER: usize = 32;

        for round in 0..ROUNDS {
            let mut world = World::new();
            let (sender, binder) = CleanupSender::deferred();
            let start = Arc::new(Barrier::new(SENDERS + 1));
            let mut threads = Vec::new();

            for producer in 0..SENDERS {
                let sender = sender.clone();
                let start = start.clone();
                threads.push(thread::spawn(move || {
                    start.wait();
                    for offset in 0..PER_SENDER {
                        sender.send_test(producer * PER_SENDER + offset);
                    }
                }));
            }

            start.wait();
            binder.bind(&mut world);
            for thread in threads {
                thread.join().unwrap();
            }

            let mut delivered = request_ids(collect_external_roots(&mut world));
            delivered.sort_unstable();
            assert_eq!(
                delivered,
                (0..SENDERS * PER_SENDER).collect::<Vec<_>>(),
                "request multiset mismatch in round {round}"
            );
        }
    }

    #[test]
    fn dropping_unapplied_deferred_route_drops_pending_work() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (sender, binder) = CleanupSender::deferred();
        sender.send_tracked(TestDropProbe(drops.clone()));

        drop(sender);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(binder);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn boundary_selects_only_requests_before_it() {
        let mut world = World::new();
        let target = BoundCleanupTarget::new(&mut world);
        let inbox = world.resource::<WorldCleanupQueue>().inbox.clone();
        target.send_test(1);

        let roots = inbox.collect_roots_with_hook(|| {
            target.send_test(2);
        });

        assert_eq!(request_ids(roots), [1]);
        assert_eq!(request_ids(inbox.collect_roots()), [2]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn continued_producers_cannot_extend_the_selected_root_prefix() {
        const AFTER_BOUNDARY: usize = 1_000;
        let mut world = World::new();
        let target = BoundCleanupTarget::new(&mut world);
        let inbox = world.resource::<WorldCleanupQueue>().inbox.clone();
        target.send_test(0);
        let start = Arc::new(Barrier::new(2));
        let producer_target = target.clone();
        let producer_start = start.clone();
        let producer = thread::spawn(move || {
            producer_start.wait();
            for id in 1..=AFTER_BOUNDARY {
                producer_target.send_test(id);
            }
        });

        let roots = inbox.collect_roots_with_hook(|| {
            start.wait();
        });
        producer.join().unwrap();

        assert_eq!(request_ids(roots), [0]);
        assert_eq!(inbox.collect_roots().len(), AFTER_BOUNDARY);
    }

    #[test]
    fn boundary_message_never_becomes_a_root_action() {
        let mut world = World::new();
        let sender = CleanupSender::bound(&mut world);
        sender.send_test(1);

        let mut worklist = CleanupWorklist::from_roots(collect_external_roots(&mut world));
        let Some(CleanupAction::Root(CleanupRequest::Test(1))) = worklist.pop_action() else {
            panic!("expected exactly the request before the private boundary");
        };
        assert!(worklist.pop_action().is_none());
        assert!(worklist.take_signal_candidates().is_empty());
    }

    #[test]
    fn external_inbox_and_local_worklist_are_separate() {
        let mut world = World::new();
        let sender = CleanupSender::bound(&mut world);
        sender.send_test(1);
        let local_candidate = candidate(&mut world);
        let expected_signal = local_candidate.signal();
        let mut worklist = CleanupWorklist::from_roots(Vec::new());

        worklist.push_stale_signal(local_candidate);

        assert_eq!(request_ids(collect_external_roots(&mut world)), [1]);
        let candidates = worklist.take_signal_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].signal(), expected_signal);
    }

    #[test]
    fn release_actions_reach_a_fixed_point_before_candidates() {
        let mut world = World::new();
        let mut worklist = CleanupWorklist::from_roots(Vec::new());
        worklist.push_test_release(TestReleaseAction {
            candidates: vec![candidate(&mut world)],
        });
        worklist.push_test_release(TestReleaseAction {
            candidates: vec![candidate(&mut world)],
        });
        let mut order = Vec::new();

        while let Some(action) = worklist.pop_action() {
            match action {
                CleanupAction::TestRelease(mut release) => {
                    order.push('R');
                    if let Some(candidate) = release.candidates.pop() {
                        worklist.push_stale_signal(candidate);
                    }
                }
                CleanupAction::Root(_)
                | CleanupAction::ReleaseSignalClaim(_)
                | CleanupAction::ReleaseStaticDependency(_)
                | CleanupAction::ReleaseDynamicConnection(_) => unreachable!(),
            }
        }
        order.extend(worklist.take_signal_candidates().into_iter().map(|_| 'C'));

        assert_eq!(order, ['R', 'R', 'C', 'C']);
    }

    #[test]
    fn candidate_processing_yields_to_new_release_actions() {
        let mut world = World::new();
        let mut worklist = CleanupWorklist::from_roots(Vec::new());
        worklist.push_stale_signal(candidate(&mut world));
        worklist.push_stale_signal(candidate(&mut world));
        let mut order = Vec::new();

        let mut candidate_batch = worklist.take_signal_candidates();
        let _processed = candidate_batch.pop().unwrap();
        order.push('C');
        worklist.push_test_release(TestReleaseAction {
            candidates: vec![candidate(&mut world)],
        });
        for candidate in candidate_batch {
            worklist.push_stale_signal(candidate);
        }

        while let Some(action) = worklist.pop_action() {
            match action {
                CleanupAction::TestRelease(mut release) => {
                    order.push('R');
                    if let Some(candidate) = release.candidates.pop() {
                        worklist.push_stale_signal(candidate);
                    }
                }
                CleanupAction::Root(_)
                | CleanupAction::ReleaseSignalClaim(_)
                | CleanupAction::ReleaseStaticDependency(_)
                | CleanupAction::ReleaseDynamicConnection(_) => unreachable!(),
            }
        }
        order.extend(worklist.take_signal_candidates().into_iter().map(|_| 'C'));

        assert_eq!(order, ['C', 'R', 'C', 'C']);
    }

    #[test]
    fn duplicate_and_missing_test_targets_synthesize_no_obligations() {
        let mut world = World::new();
        let mut worklist = CleanupWorklist::from_roots(Vec::new());
        worklist.push_test_release(TestReleaseAction {
            candidates: vec![candidate(&mut world)],
        });
        worklist.push_test_release(TestReleaseAction { candidates: Vec::new() });
        worklist.push_test_release(TestReleaseAction { candidates: Vec::new() });
        let mut reaped = 0;

        while let Some(action) = worklist.pop_action() {
            match action {
                CleanupAction::TestRelease(mut release) => {
                    if let Some(candidate) = release.candidates.pop() {
                        worklist.push_stale_signal(candidate);
                    }
                }
                CleanupAction::Root(_)
                | CleanupAction::ReleaseSignalClaim(_)
                | CleanupAction::ReleaseStaticDependency(_)
                | CleanupAction::ReleaseDynamicConnection(_) => unreachable!(),
            }
        }
        reaped += worklist.take_signal_candidates().len();

        assert_eq!(reaped, 1);
    }

    #[test]
    fn finite_owned_release_actions_terminate_without_a_wave_limit() {
        const OBLIGATIONS: usize = 256;
        let mut world = World::new();
        let candidates = (0..OBLIGATIONS).map(|_| candidate(&mut world)).collect::<Vec<_>>();
        let mut worklist = CleanupWorklist::from_roots(Vec::new());
        worklist.push_test_release(TestReleaseAction { candidates });
        let mut releases = 0;
        let mut reaped = 0;

        while let Some(action) = worklist.pop_action() {
            match action {
                CleanupAction::TestRelease(mut release) => {
                    releases += 1;
                    let candidate = release.candidates.pop().unwrap();
                    worklist.push_stale_signal(candidate);
                    if !release.candidates.is_empty() {
                        worklist.push_test_release(release);
                    }
                }
                CleanupAction::Root(_)
                | CleanupAction::ReleaseSignalClaim(_)
                | CleanupAction::ReleaseStaticDependency(_)
                | CleanupAction::ReleaseDynamicConnection(_) => unreachable!(),
            }
        }
        reaped += worklist.take_signal_candidates().len();

        assert_eq!(releases, OBLIGATIONS);
        assert_eq!(reaped, OBLIGATIONS);
        assert!(worklist.is_empty());
    }
}
