//! Signal graph management and runtime.
//!
//! This module contains the core infrastructure for [jonmo](crate)'s reactive signal graph,
//! including graph topology tracking, signal registration, multi-schedule processing, and
//! the per-frame processing loop.
//!
//! # Graph Structure
//!
//! The signal graph is a directed acyclic graph (DAG) where each node is backed by a Bevy
//! [`System`]. Edges represent data dependencies: when an upstream signal produces output, that
//! output becomes the input to its downstream signals. Cycles are detected and rejected at
//! edge-creation time.
//!
//! # Multi-Schedule Processing
//!
//! Signals can be assigned to run in different Bevy schedules (e.g., `Update`, `PostUpdate`)
//! using [`SignalExt::schedule`](super::signal::SignalExt::schedule). This enables fine-grained
//! control over when signals execute within each frame.
//!
//! - **Schedule assignment**: Each signal can be assigned to a specific schedule. Unassigned
//!   signals inherit the schedule from upstream signals or default to the
//!   [`JonmoPlugin`](crate::JonmoPlugin)'s default schedule, see
//!   [`SignalExt::schedule`](super::signal::SignalExt::schedule) for the exact semantics.
//! - **Cross-schedule data flow**: Signals in different schedules can still be connected. Outputs
//!   from earlier schedules are available as inputs to signals in later schedules.
//! - **Per-schedule graph partitioning**: The graph is partitioned by schedule, with each schedule
//!   processing only its assigned signals in topological order.
//!
//! # Processing Semantics
//!
//! Each frame, within each configured schedule, signals are processed in topological order:
//!
//! 1. **Level assignment**: Each signal is assigned a level equal to 1 + the maximum level of its
//!    upstreams (roots have level 0). This is recomputed incrementally when edges change.
//! 2. **Level-order execution**: Signals are processed level by level, lowest first, ensuring every
//!    signal's upstreams have already run before it executes. Within each level, signals are
//!    processed in deterministic order (sorted by signal identity) for reproducible behavior.
//! 3. **Output forwarding**: When a signal produces [`Some`] value, that value is forwarded as
//!    input to all downstream signals. Returning [`None`] terminates propagation for that branch.
//! 4. **Dynamic registration**: Signals registered during processing (e.g., from UI element
//!    spawning) are integrated and processed within the same frame, avoiding one-frame delays.
//!
//! A signal runs **once per queued input**, in enqueue order. Scalar entities forward only the
//! **last successful output** from that drain; a later `None` does not erase an earlier `Some`.
//! Typed vector/map entities instead accept and forward every produced diff batch in order,
//! preserving batch boundaries through downstream processing. A scalar whose value is a `Vec`
//! still follows scalar selection, not collection-diff semantics.
//!
//! Unconsumed inputs survive frame boundaries until the consumer runs or lifecycle invalidation
//! clears them. A consumer whose schedule never runs can therefore retain unbounded queued work.
//!
//! # Lifecycle
//!
//! Each call to [`.register`](super::signal::SignalExt::register) returns one direct
//! [`SignalClaim`] for the terminal signal entity. Activating a count-zero terminal recursively
//! acquires one claim on each unique immediate canonical dependency and installs its active routes;
//! additional claims on an already-active terminal increment only that terminal. Claims are
//! released explicitly with [`SignalClaim::release`] or by storing them in a [`SignalClaims`]
//! component, which releases them when its owner is removed.
//!
//! A final `1 -> 0` release deactivates the entity immediately: it becomes unrunnable, leaves its
//! effective schedule, clears pending input, removes active dependency relationships, and releases
//! its immediate dependency claims. Entity reaping is a later candidate phase, distinct from
//! deactivation. An ordinary count-zero terminal can remain materialized while a live signal chain
//! retains it, and dependencies remain materialized while listed by an existing entity's immediate
//! structural dependencies. A retained replay terminal instead owns one explicit private
//! maintenance claim so its replay state stays active and current.
//!
//! A candidate is reaped only after all terminal retainers are dead, no structural materialized
//! dependents remain, replay maintenance has been released, its claim count is zero, and its
//! runtime is logically inactive. Active downstream routes alone neither retain an entity nor
//! determine reaping eligibility.
//!
//! # Replay Inspection and Imperative Evaluation
//!
//! Authoritative current state for framework-verified replayable terminals can be inspected
//! without running systems or mutating graph state via [`ReplayCacheInspection`],
//! [`inspect_scalar_replay_cache`], [`inspect_vec_replay_cache`], and
//! [`inspect_map_replay_cache`].
//!
//! For applications and tests that explicitly need out-of-schedule imperative subgraph stepping,
//! enable the non-default `imperative-signal-evaluation` feature to access
//! `jonmo::graph::run_signal_subgraph_once`, `jonmo::graph::BoxedSignalValue`, and
//! `jonmo::graph::downcast_signal_value`. Imperative evaluation executes systems and mutates
//! runtime state; it is not a read-only current-value query.
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use bevy_app::PostUpdate;
use bevy_derive::Deref;
use bevy_ecs::{
    lifecycle::HookContext,
    prelude::*,
    schedule::{InternedScheduleLabel, ScheduleLabel},
    system::{SystemId, SystemState},
    world::DeferredWorld,
};
use bevy_platform::{
    collections::{HashMap, HashSet},
    prelude::*,
    sync::Arc,
};
use core::{
    any::{Any, TypeId},
    hash::Hash,
};

#[cfg(test)]
use crate::registration::{OutputDeliveryKey, SignalKind, SignalMaterializer, verify_existing_canonical_with_graph};

use crate::{
    cleanup::{self, CleanupWorklist},
    dynamic::{DynamicSignalConnection, DynamicSignalOwner, DynamicSignalSubscription, DynamicSubscriptionInspection},
    registration::{
        AcceptedSignalOutput, ErasedOutputClone, ErasedReplaySnapshot, ExistingSignalEntityRegistration,
        OrderedInputConsumption, OutputDeliveryKind, PlannedReplayState, PlannedSignalEntityRef,
        RegistrationActivationCommit, RegistrationActivationPreflight, RegistrationActiveRouteCommit,
        RegistrationActiveRoutePreflight, RegistrationDynamicConnectionCommit,
        RegistrationEffectiveMembershipPreflight, RegistrationEntityCommit, RegistrationEntityPreflight,
        RegistrationMaterializedDependencyCommit, RegistrationPreflight, RegistrationReplayCommit,
        RegistrationRouteDisposition, RegistrationScheduleCommit, RegistrationTerminalRetainerCommit,
        ReplayRuntimeContract, ReplayRuntimeKey, SignalRegistrationError, SignalRegistrationGraphHooks,
        SignalTerminalRetainer, StoredOutputDelivery,
    },
    replay::{
        MapReplayCache, ReplayCacheReadiness, ReplayCacheSnapshot, ReplayRevision, ScalarReplayCache, VecReplayCache,
    },
};

/// Copyable full-generational identity for an [`Entity`] in the signal graph.
///
/// A `SignalEntity` carries no ownership, release authority, route permission, or dependency
/// multiplicity. It cannot be converted into a [`SignalClaim`].
#[derive(Clone, Copy, Deref, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignalEntity(Entity);

impl SignalEntity {
    /// Returns the complete generational Bevy entity identity.
    ///
    /// This observes identity only; it does not acquire or duplicate ownership.
    pub fn entity(self) -> Entity {
        self.0
    }
}

/// A move-only signal-cleanup obligation created by releasing one description owner.
pub(crate) struct ReleasedSignalCandidate(SignalEntity);

impl ReleasedSignalCandidate {
    pub(crate) fn new(signal: SignalEntity) -> Self {
        Self(signal)
    }

    pub(crate) fn signal(&self) -> SignalEntity {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn test_new(signal: SignalEntity) -> Self {
        Self(signal)
    }
}

#[must_use]
pub(crate) struct ReleasedStaticDependencyClaim {
    projection: ProjectedSignalRelease,
}

impl From<Entity> for SignalEntity {
    fn from(entity: Entity) -> Self {
        Self(entity)
    }
}

impl<I: 'static, O> From<SystemId<In<I>, O>> for SignalEntity {
    fn from(system_id: SystemId<In<I>, O>) -> Self {
        system_id.entity().into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SignalClaimCountError {
    Overflow,
    Underflow,
}

#[derive(Component, Deref)]
pub(crate) struct SignalClaimCount(usize);

#[derive(Component, Debug, PartialEq, Eq)]
pub(crate) struct StaticSignalDependencies(Box<[SignalEntity]>);

#[derive(Component, Debug, Default, PartialEq, Eq)]
pub(crate) struct MaterializedSignalDependents(BTreeSet<SignalEntity>);

#[derive(Component, Default)]
struct SignalTerminalRetainers(Vec<SignalTerminalRetainer>);

#[derive(Component)]
struct ReplayMaintenanceClaim(SignalClaim);

#[derive(Component)]
struct RegistrarInactive;

// A validated commit can outlive reentrant deactivation. This retains only the canonical
// entity, never a runnable claim, so final release still invalidates input epochs immediately.
#[derive(Component, Default)]
struct RegistrationCommitRetainers(usize);

pub(crate) fn retain_registration_entities(world: &mut World, entities: &[SignalEntity]) {
    for signal in entities {
        let mut entity = world.entity_mut(**signal);
        if let Some(mut retainers) = entity.get_mut::<RegistrationCommitRetainers>() {
            retainers.0 = retainers.0.checked_add(1).expect("registration retention overflow");
        } else {
            entity.insert(RegistrationCommitRetainers(1));
        }
    }
}

pub(crate) fn release_registration_entities(
    world: &mut World,
    entities: Vec<SignalEntity>,
) -> Vec<ReleasedSignalCandidate> {
    let mut candidates = Vec::new();
    for signal in entities {
        let Ok(mut entity) = world.get_entity_mut(*signal) else {
            continue;
        };
        let mut retainers = entity
            .get_mut::<RegistrationCommitRetainers>()
            .expect("registration retention disappeared");
        retainers.0 = retainers.0.checked_sub(1).expect("registration retention underflow");
        if retainers.0 == 0 {
            entity.remove::<RegistrationCommitRetainers>();
            candidates.push(ReleasedSignalCandidate::new(signal));
        }
    }
    candidates
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegistrationCommitStage {
    EntityPublishedWithoutReplay(SignalEntity),
    ReplayPublished(SignalEntity),
}

#[cfg(test)]
#[derive(Resource, Default)]
struct RegistrationCommitStageTrace(Vec<RegistrationCommitStage>);

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnershipWorklistStage {
    Acquired(SignalEntity),
    OrdinaryRoute(SignalEntity, SignalEntity),
    ReplayPendingRoute(SignalEntity, SignalEntity),
    ReplayReadyRoute(SignalEntity, SignalEntity, ReplayRevision),
    Released(SignalEntity),
    Candidate(SignalEntity),
    MaintenanceQueued(SignalEntity),
    Reaped(SignalEntity),
}

#[cfg(test)]
#[derive(Resource, Default)]
struct OwnershipWorklistTrace(Vec<OwnershipWorklistStage>);

#[cfg(test)]
fn trace_ownership_stage(world: &mut World, stage: OwnershipWorklistStage) {
    if let Some(mut trace) = world.get_resource_mut::<OwnershipWorklistTrace>() {
        trace.0.push(stage);
    }
}

#[derive(Component)]
pub(crate) struct RegistrationReplayCacheState {
    contract: Arc<dyn ReplayRuntimeContract>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayRelationshipState {
    Pending,
    Published { cursor: ReplayRevision },
}

#[derive(Component, Clone, Default)]
struct ReplayConsumingRelationships(BTreeMap<SignalEntity, ReplayRelationshipState>);

impl SignalClaimCount {
    fn new() -> Self {
        Self(0)
    }

    fn try_increment(&mut self) -> Result<(), SignalClaimCountError> {
        let incremented = self.0.checked_add(1).ok_or(SignalClaimCountError::Overflow)?;
        self.0 = incremented;
        Ok(())
    }

    fn try_decrement(&mut self) -> Result<(), SignalClaimCountError> {
        let decremented = self.0.checked_sub(1).ok_or(SignalClaimCountError::Underflow)?;
        self.0 = decremented;
        Ok(())
    }
}

fn increment_signal_claim_count(world: &mut World, signal: SignalEntity) {
    let mut count = world
        .get_mut::<SignalClaimCount>(*signal)
        .unwrap_or_else(|| panic!("cannot acquire a claim for non-signal entity {signal:?}"));
    count
        .try_increment()
        .unwrap_or_else(|error| panic!("cannot acquire a claim for {signal:?}: {error:?}"));
}

/// Component on signal system entities indicating which schedule they run in.
///
/// Used by the multi-schedule processing system to determine which signals
/// to run during each schedule's processing pass.
#[derive(Component, Clone, Copy)]
pub(crate) struct ScheduleTag(pub(crate) InternedScheduleLabel);

/// Component for downstream schedule inheritance during registration.
///
/// When a signal is planned from a dependency with a `ScheduleHint`, the dependent inherits the
/// schedule unless it already has a [`ScheduleTag`].
#[derive(Component, Clone, Copy)]
pub(crate) struct ScheduleHint(pub(crate) InternedScheduleLabel);

fn commit_cached_schedule_membership(
    world: &mut World,
    signal: SignalEntity,
    schedule: InternedScheduleLabel,
    level: u32,
) {
    let mut state = world.resource_mut::<SignalGraphState>();
    state.deferred_removals.remove(&signal);
    if let Some(previous) = state.signal_schedules.remove(&signal)
        && let Some(levels) = state.by_schedule.get_mut(&previous)
    {
        for members in levels {
            members.retain(|member| *member != signal);
        }
    }
    state.levels.insert(signal, level);
    state.signal_schedules.insert(signal, schedule);
    let levels = state.by_schedule.entry(schedule).or_default();
    let members = ensure_level_bucket(levels, level);
    insert_sorted_by_identity(members, signal);
}

fn reconcile_cached_schedule_membership(world: &mut World, signal: SignalEntity) {
    let schedule = world.get::<ScheduleTag>(*signal).map(|tag| tag.0).or_else(|| {
        world
            .get_resource::<SignalGraphState>()
            .map(|state| state.default_schedule)
    });
    let active = world.get::<RegistrarInactive>(*signal).is_none();
    world.resource_scope(|_world, mut state: Mut<SignalGraphState>| {
        if !active {
            remove_signal_from_graph_state_internal(&mut state, signal);
            return;
        }
        if let Some(previous) = state.signal_schedules.remove(&signal)
            && let Some(levels) = state.by_schedule.get_mut(&previous)
        {
            for members in levels {
                members.retain(|member| *member != signal);
            }
        }
        let Some(schedule) = schedule else {
            return;
        };
        let Some(level) = active.then(|| state.levels.get(&signal).copied()).flatten() else {
            return;
        };
        state.signal_schedules.insert(signal, schedule);
        let levels = state.by_schedule.entry(schedule).or_default();
        let members = ensure_level_bucket(levels, level);
        insert_sorted_by_identity(members, signal);
    });
}

#[derive(Component, Deref, Clone, Debug, Default)]
pub(crate) struct ActiveSignalDependents(HashSet<SignalEntity>);

impl<'a> IntoIterator for &'a ActiveSignalDependents {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = bevy_platform::collections::hash_set::Iter<'a, SignalEntity>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub(crate) type ErasedSignalValue = Box<dyn Any + Send + Sync>;

pub(crate) fn erased_type_id(value: &ErasedSignalValue) -> TypeId {
    (**value).type_id()
}

type TargetOutputObserver<'a> = &'a mut dyn FnMut(&SystemRunner, ErasedSignalValue);

#[derive(Component, Clone)]
struct SystemRunner {
    runner: Arc<crate::registration::RegistrationRunner>,
    input_type: TypeId,
    output_type: TypeId,
    output_delivery: StoredOutputDelivery,
    collection_poll_append: Option<crate::registration::CollectionPollAppend>,
    ordered_input_consumption: Option<OrderedInputConsumption>,
    replay_key: Option<ReplayRuntimeKey>,
    replay_contract: Option<Arc<dyn ReplayRuntimeContract>>,
}

impl SystemRunner {
    fn run(&self, world: &mut World, input: ErasedSignalValue) -> Option<ErasedSignalValue> {
        match (self.runner)(world, input) {
            Ok(output) => output,
            Err(crate::registration::RegistrationSystemRunError::InputTypeMismatch { expected }) => {
                panic!("signal system runner received mismatched erased input; expected {expected}");
            }
            Err(crate::registration::RegistrationSystemRunError::SystemUnavailable) => None,
        }
    }

    fn accept_output(
        &self,
        world: &mut World,
        entity: SignalEntity,
        output: ErasedSignalValue,
    ) -> Option<AcceptedSignalOutput> {
        match &self.replay_contract {
            Some(contract) => contract.accept_output(world, entity, output),
            None => Some(AcceptedSignalOutput {
                output,
                replay_revision: None,
            }),
        }
    }
}

/// Component that stores pending inputs for a signal.
///
/// Inputs are written by upstream signals and read by the signal during processing.
/// Unconsumed inputs survive frame boundaries; deactivation invalidates them.
#[derive(Component, Default)]
pub(crate) struct SignalInputBuffer(pub(crate) Vec<ErasedSignalValue>, BTreeMap<SignalEntity, usize>, u64);

pub(crate) fn signal_input_epoch(world: &World, signal: SignalEntity) -> u64 {
    world
        .get::<SignalInputBuffer>(*signal)
        .expect("executing signal must retain its input buffer")
        .2
}

impl SignalInputBuffer {
    /// Take all inputs, leaving the buffer empty.
    fn take(&mut self) -> Vec<ErasedSignalValue> {
        self.1.clear();
        core::mem::take(&mut self.0)
    }

    /// Push an input value.
    fn push(&mut self, value: ErasedSignalValue) {
        self.0.push(value);
    }

    fn push_replay_snapshot(&mut self, source: SignalEntity, value: ErasedSignalValue) {
        if let Some(previous_index) = self.1.remove(&source)
            && previous_index < self.0.len()
        {
            self.0.remove(previous_index);
            for index in self.1.values_mut() {
                if *index > previous_index {
                    *index -= 1;
                }
            }
        }
        let index = self.0.len();
        self.0.push(value);
        self.1.insert(source, index);
    }

    fn has_pending_replay_snapshot(&self, source: SignalEntity) -> bool {
        self.1.contains_key(&source)
    }

    /// Clear the buffer.
    fn clear(&mut self) {
        self.2 = self
            .2
            .checked_add(1)
            .expect("signal input invalidation epoch overflowed");
        self.0.clear();
        self.1.clear();
    }
}

/// Behavior when the signal registration recursion limit is exceeded.
///
/// During signal processing, signals can spawn new signals (e.g., UI elements registering
/// child signals). These new signals are processed in the same frame via recursive passes.
/// If signals keep spawning more signals indefinitely, this limit prevents infinite loops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecursionLimitBehavior {
    /// Panic with an error message. This is the default because hitting the limit
    /// almost always indicates a bug (infinite signal spawning loop).
    #[default]
    Panic,
    /// Log a warning (if the `tracing` feature is enabled) and stop processing
    /// new signals for this frame. The graph may be left in an incomplete state.
    Warn,
    /// Silently stop processing new signals. Use this only if you understand the
    /// implications and have a specific reason to suppress the error.
    Silent,
}

/// Tracks signal graph topology for level-based processing.
#[derive(Resource)]
pub(crate) struct SignalGraphState {
    /// Cached level for each signal (max distance from any root).
    levels: HashMap<SignalEntity, u32>,
    /// Signals partitioned by schedule for efficient per-schedule iteration.
    /// Updated incrementally alongside level updates.
    by_schedule: HashMap<InternedScheduleLabel, Vec<Vec<SignalEntity>>>,
    /// Cache of which schedule each signal belongs to, for O(1) lookup during removal.
    signal_schedules: HashMap<SignalEntity, InternedScheduleLabel>,
    /// Signals that seed level recomputation after edge changes.
    edge_change_seeds: HashSet<SignalEntity>,
    /// Signals queued for removal while the graph is being processed.
    deferred_removals: HashSet<SignalEntity>,
    /// Whether the signal graph is currently being processed.
    is_processing: bool,
    /// Default schedule for signals without explicit scheduling.
    default_schedule: InternedScheduleLabel,
    /// Maximum number of recursive signal registration passes per frame.
    registration_recursion_limit: usize,
    /// What to do when the recursion limit is exceeded.
    on_recursion_limit_exceeded: RecursionLimitBehavior,
    /// Schedules that have been registered with `JonmoPlugin`.
    registered_schedules: HashSet<InternedScheduleLabel>,
    /// Dynamic owners that queued deferred self-adapter cleanup during the current pass.
    pending_after_output_cleanup: Vec<SignalEntity>,
}

/// Default recursion limit for signal registration passes.
pub const DEFAULT_REGISTRATION_RECURSION_LIMIT: usize = 100;

impl Default for SignalGraphState {
    fn default() -> Self {
        Self::with_options(
            PostUpdate.intern(),
            DEFAULT_REGISTRATION_RECURSION_LIMIT,
            RecursionLimitBehavior::default(),
        )
    }
}

impl SignalGraphState {
    /// Create a new SignalGraphState with the specified default schedule.
    #[allow(unused)] // used in tests
    pub(crate) fn new(default_schedule: InternedScheduleLabel) -> Self {
        Self::with_options(
            default_schedule,
            DEFAULT_REGISTRATION_RECURSION_LIMIT,
            RecursionLimitBehavior::default(),
        )
    }

    /// Create a new SignalGraphState with full configuration options.
    pub(crate) fn with_options(
        default_schedule: InternedScheduleLabel,
        registration_recursion_limit: usize,
        on_recursion_limit_exceeded: RecursionLimitBehavior,
    ) -> Self {
        Self {
            levels: HashMap::default(),
            by_schedule: HashMap::default(),
            signal_schedules: HashMap::default(),
            edge_change_seeds: HashSet::default(),
            deferred_removals: HashSet::default(),
            is_processing: false,
            default_schedule,
            registration_recursion_limit,
            on_recursion_limit_exceeded,
            registered_schedules: HashSet::from([default_schedule]),
            pending_after_output_cleanup: Vec::new(),
        }
    }

    /// Register a schedule as valid for signal processing.
    pub(crate) fn register_schedule(&mut self, schedule: InternedScheduleLabel) {
        self.registered_schedules.insert(schedule);
    }
}

fn get_static_dependencies(world: &World, signal: SignalEntity) -> Vec<SignalEntity> {
    world
        .get::<StaticSignalDependencies>(*signal)
        .map(|dependencies| dependencies.0.to_vec())
        .unwrap_or_default()
}

fn get_execution_dependencies(world: &World, signal: SignalEntity) -> Vec<SignalEntity> {
    let mut upstreams = get_static_dependencies(world, signal);
    if let Some(owner) = world.get::<DynamicSignalOwner>(*signal) {
        upstreams.extend(
            owner
                .subscriptions()
                .into_iter()
                .filter(|subscription| subscription.target == signal)
                .map(|subscription| subscription.source),
        );
    }
    upstreams.sort_by(signal_identity_cmp);
    upstreams.dedup();
    upstreams
}

fn get_downstreams(world: &World, signal: SignalEntity) -> Vec<SignalEntity> {
    world
        .get::<ActiveSignalDependents>(*signal)
        .map(|d| d.0.iter().copied().collect())
        .unwrap_or_default()
}

fn get_execution_dependents(world: &World, signal: SignalEntity) -> Vec<SignalEntity> {
    let mut dependents = get_downstreams(world, signal);
    // Pending replay consumers still depend on the source's level, even before delivery is allowed.
    if let Some(relationships) = world.get::<ReplayConsumingRelationships>(*signal) {
        dependents.extend(relationships.0.keys().copied());
    }
    dependents.sort_by(signal_identity_cmp);
    dependents.dedup();
    dependents
}

fn ensure_level_bucket(levels: &mut Vec<Vec<SignalEntity>>, level: u32) -> &mut Vec<SignalEntity> {
    let index = usize::try_from(level).expect("signal level exceeded usize");
    let required_len = index
        .checked_add(1)
        .expect("signal level bucket length overflowed usize");
    if levels.len() < required_len {
        levels.resize_with(required_len, Vec::new);
    }
    &mut levels[index]
}

fn insert_sorted_by_identity(bucket: &mut Vec<SignalEntity>, signal: SignalEntity) {
    if let Err(index) = bucket.binary_search_by(|existing| signal_identity_cmp(existing, &signal)) {
        bucket.insert(index, signal);
    }
}

/// Result of computing signal levels via Kahn's algorithm.
struct LevelComputeResult {
    /// Computed levels for each signal in the working set.
    levels: HashMap<SignalEntity, u32>,
    /// Number of signals successfully processed.
    processed: usize,
    /// Total number of signals in the working set.
    total: usize,
}

impl LevelComputeResult {
    fn is_complete(&self) -> bool {
        self.processed == self.total
    }
}

/// Core Kahn's algorithm for computing topological levels on a subset of signals.
///
/// - `signals`: the working set of signals to compute levels for
/// - `upstream_filter`: determines which upstreams count toward in-degree (typically the working
///   set)
/// - `external_level`: provides levels for upstreams outside the working set
fn compute_signal_levels(
    world: &World,
    signals: &HashSet<SignalEntity>,
    upstream_filter: impl Fn(SignalEntity) -> bool,
    external_level: impl Fn(SignalEntity) -> Option<u32>,
) -> LevelComputeResult {
    if signals.is_empty() {
        return LevelComputeResult {
            levels: HashMap::new(),
            processed: 0,
            total: 0,
        };
    }

    let mut in_degree: HashMap<SignalEntity, usize> = HashMap::new();
    let mut upstreams_map: HashMap<SignalEntity, Vec<SignalEntity>> = HashMap::new();
    let mut downstreams_map: HashMap<SignalEntity, Vec<SignalEntity>> = HashMap::new();

    for &signal in signals {
        let upstreams = get_execution_dependencies(world, signal);
        let local_in_degree = upstreams.iter().filter(|u| upstream_filter(**u)).count();
        in_degree.insert(signal, local_in_degree);
        // Build local downstreams_map by inverting upstream relationships.
        // This ensures we only traverse to signals in our working set.
        for &upstream in upstreams.iter().filter(|u| signals.contains(*u)) {
            downstreams_map.entry(upstream).or_default().push(signal);
        }
        upstreams_map.insert(signal, upstreams);
    }

    let mut queue: VecDeque<SignalEntity> = in_degree.iter().filter_map(|(s, d)| (*d == 0).then_some(*s)).collect();

    let mut levels: HashMap<SignalEntity, u32> = HashMap::new();
    let mut processed = 0usize;

    while let Some(signal) = queue.pop_front() {
        processed += 1;
        let upstreams = upstreams_map.get(&signal).cloned().unwrap_or_default();
        let level = upstreams
            .iter()
            .filter_map(|u| levels.get(u).copied().or_else(|| external_level(*u)))
            .max()
            .map(|m| m.checked_add(1).expect("signal graph level overflowed u32::MAX"))
            .unwrap_or(0);

        levels.insert(signal, level);

        // Use local downstreams_map instead of get_downstreams to ensure we only
        // consider signals in our working set.
        if let Some(downstreams) = downstreams_map.get(&signal) {
            for &downstream in downstreams {
                if let Some(count) = in_degree.get_mut(&downstream) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        queue.push_back(downstream);
                    }
                }
            }
        }
    }

    LevelComputeResult {
        levels,
        processed,
        total: signals.len(),
    }
}

fn bucket_levels_sorted(levels: &HashMap<SignalEntity, u32>) -> Vec<Vec<SignalEntity>> {
    let mut by_level: Vec<Vec<SignalEntity>> = Vec::new();
    for (&signal, &level) in levels {
        ensure_level_bucket(&mut by_level, level).push(signal);
    }
    for bucket in &mut by_level {
        bucket.sort_by(signal_identity_cmp);
    }
    by_level
}

// Computes a topological ordering of signals reachable downstream from `seeds`.
// Uses cached levels when available, falls back to BFS level computation for uncached signals.
fn downstream_levels_from_seeds(world: &World, seeds: &[SignalEntity]) -> Vec<Vec<SignalEntity>> {
    let state = world.resource::<SignalGraphState>();

    // Collect all reachable signals via BFS.
    let mut reachable: HashSet<SignalEntity> = HashSet::new();
    let mut queue: VecDeque<SignalEntity> = seeds.iter().copied().collect();
    while let Some(signal) = queue.pop_front() {
        if reachable.insert(signal) {
            queue.extend(get_downstreams(world, signal));
        }
    }

    // Check if all reachable signals have cached levels.
    let all_cached = reachable.iter().all(|s| state.levels.contains_key(s));

    if all_cached {
        // Fast path: use cached levels.
        let mut by_level: Vec<Vec<SignalEntity>> = Vec::new();
        for signal in reachable {
            let level = state.levels.get(&signal).copied().unwrap_or(0);
            ensure_level_bucket(&mut by_level, level).push(signal);
        }
        for bucket in &mut by_level {
            bucket.sort_by(signal_identity_cmp);
        }
        by_level
    } else {
        // Slow path: compute levels via Kahn's algorithm on the reachable subgraph.
        // This handles newly created signals that aren't in the cache yet.
        let result = compute_signal_levels(
            world,
            &reachable,
            |u| reachable.contains(&u),        // only count upstreams in the subgraph
            |u| state.levels.get(&u).copied(), // use cached levels for external upstreams
        );
        bucket_levels_sorted(&result.levels)
    }
}

// Rebuilds per-signal levels using a Kahn-style topological traversal.
//
// - Roots (in-degree 0) start at level 0.
// - Each node's level is 1 + max(level of its upstreams).
// - Nodes are bucketed by level for deterministic per-level iteration.
// - If a cycle or inconsistent edges are detected (not all nodes processed), this panics because
//   the graph invariants were violated.
fn rebuild_levels(world: &mut World, state: &mut SignalGraphState) {
    state.levels.clear();
    state.by_schedule.clear();
    state.signal_schedules.clear();

    let mut all_signals_state =
        SystemState::<Query<Entity, (With<SystemRunner>, Without<RegistrarInactive>)>>::new(world);
    let all_signals: HashSet<SignalEntity> = all_signals_state
        .get(world)
        .expect("query params are always valid")
        .iter()
        .map(SignalEntity)
        .collect();

    let result = compute_signal_levels(
        world,
        &all_signals,
        |u| all_signals.contains(&u), // all upstreams count
        |_| None,                     // no external levels
    );

    if !result.is_complete() {
        panic!("signal graph contains a cycle or inconsistent edges during level rebuild");
    }

    state.levels = result.levels;

    // Build by_schedule partitioning and signal_schedules cache
    for (&signal, &level) in &state.levels {
        let schedule = world
            .get::<ScheduleTag>(*signal)
            .map(|tag| tag.0)
            .unwrap_or(state.default_schedule);

        // Cache the schedule for O(1) lookup during removal
        state.signal_schedules.insert(signal, schedule);

        let schedule_levels = state.by_schedule.entry(schedule).or_default();
        insert_sorted_by_identity(ensure_level_bucket(schedule_levels, level), signal);
    }
}

/// Remove a signal from its current position in by_schedule.
fn remove_signal_from_buckets(state: &mut SignalGraphState, signal: SignalEntity, old_level: u32) {
    // Remove from by_schedule using cached schedule for O(1) lookup.
    if let Some(&schedule) = state.signal_schedules.get(&signal)
        && let Some(schedule_levels) = state.by_schedule.get_mut(&schedule)
        && let Ok(index) = usize::try_from(old_level)
        && let Some(bucket) = schedule_levels.get_mut(index)
        && let Some(pos) = bucket.iter().position(|s| *s == signal)
    {
        bucket.remove(pos);
    }
}

/// Insert a signal at its new level in by_schedule.
fn insert_signal_into_buckets(world: &World, state: &mut SignalGraphState, signal: SignalEntity, new_level: u32) {
    // Get schedule and insert into by_schedule
    let schedule = world
        .get::<ScheduleTag>(*signal)
        .map(|tag| tag.0)
        .unwrap_or(state.default_schedule);

    // Update the schedule cache
    state.signal_schedules.insert(signal, schedule);

    let schedule_levels = state.by_schedule.entry(schedule).or_default();
    insert_sorted_by_identity(ensure_level_bucket(schedule_levels, new_level), signal);
}

fn update_levels_incremental(world: &mut World, state: &mut SignalGraphState, seeds: &[SignalEntity]) -> bool {
    // Collect all signals affected by edge changes (seeds + all their downstreams).
    let mut affected: HashSet<SignalEntity> = HashSet::new();
    let mut queue: VecDeque<SignalEntity> = seeds.iter().copied().collect();

    while let Some(signal) = queue.pop_front() {
        if affected.insert(signal) {
            queue.extend(get_execution_dependents(world, signal));
        }
    }
    if affected.is_empty() {
        return true;
    }

    let active_affected = affected
        .iter()
        .copied()
        .filter(|signal| world.get::<RegistrarInactive>(**signal).is_none())
        .collect::<HashSet<_>>();

    // Use Kahn's algorithm to compute new levels for affected runnable signals.
    let result = compute_signal_levels(
        world,
        &active_affected,
        |u| active_affected.contains(&u),  // only affected upstreams count for in-degree
        |u| state.levels.get(&u).copied(), // unaffected upstreams use cached levels
    );

    if !result.is_complete() {
        return false;
    }

    // Update state with short-circuit optimization:
    // Only update buckets for signals whose level actually changed.
    for &signal in &affected {
        let old_level = state.levels.get(&signal).copied();
        let new_level = result.levels.get(&signal).copied();

        // Short-circuit: if level didn't change, skip bucket updates
        if old_level == new_level {
            continue;
        }

        // Remove from old position
        if let Some(old) = old_level {
            remove_signal_from_buckets(state, signal, old);
        }

        // Insert at new position
        if let Some(new_level) = new_level {
            state.levels.insert(signal, new_level);
            insert_signal_into_buckets(world, state, signal, new_level);
        } else {
            state.levels.remove(&signal);
            state.signal_schedules.remove(&signal);
        }
    }

    true
}

/// Updates signal levels based on edge changes and returns the seeds that were processed.
///
/// Returns the signals that were in `edge_change_seeds` before processing. This allows
/// callers to know which signals triggered the update without needing to drain and re-add.
fn update_edge_change_levels(world: &mut World, state: &mut SignalGraphState) -> Vec<SignalEntity> {
    if state.levels.is_empty() {
        rebuild_levels(world, state);
        return state.edge_change_seeds.drain().collect();
    }

    if state.edge_change_seeds.is_empty() {
        return Vec::new();
    }

    // Drain seeds once and pass directly to update function
    let seeds: Vec<SignalEntity> = state.edge_change_seeds.drain().collect();

    if !update_levels_incremental(world, state, &seeds) {
        panic!("signal graph contains a cycle or inconsistent edges during incremental update");
    }

    seeds
}

/// Immediately synchronizes `state.levels` and `state.by_schedule` for a runnable dynamic owner
/// and its downstream subgraph after `DynamicSignalOwner::subscriptions` changes, without
/// draining `state.edge_change_seeds`.
pub(crate) fn sync_dynamic_owner_levels(world: &mut World, owner: SignalEntity) {
    if !signal_is_runnable(world, owner) {
        return;
    }
    if world.contains_resource::<SignalGraphState>() {
        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            if state.levels.is_empty() {
                rebuild_levels(world, &mut state);
                return;
            }
            if !update_levels_incremental(world, &mut state, &[owner]) {
                panic!("signal graph contains a cycle or inconsistent edges during dynamic level sync");
            }
        });
    }
}

fn remove_signal_from_graph_state_internal(state: &mut SignalGraphState, signal: SignalEntity) {
    if let Some(level) = state.levels.remove(&signal) {
        remove_signal_from_buckets(state, signal, level);
    }
    state.signal_schedules.remove(&signal);
    state.edge_change_seeds.remove(&signal);
}

fn remove_signal_from_graph_state(world: &mut World, signal: SignalEntity) {
    if let Some(mut state) = world.get_resource_mut::<SignalGraphState>() {
        if state.is_processing {
            state.deferred_removals.insert(signal);
        } else {
            remove_signal_from_graph_state_internal(&mut state, signal);
        }
    }
}

fn apply_deferred_removals(state: &mut SignalGraphState) {
    if state.deferred_removals.is_empty() {
        return;
    }
    let removals = state.deferred_removals.drain().collect::<Vec<_>>();
    for signal in removals {
        remove_signal_from_graph_state_internal(state, signal);
    }
}

fn push_signal_input_unchecked(world: &mut World, dependent: SignalEntity, input: ErasedSignalValue) -> bool {
    if let Ok(mut entity) = world.get_entity_mut(*dependent)
        && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
    {
        buffer.push(input);
        true
    } else {
        false
    }
}

fn push_replay_snapshot_input_unchecked(
    world: &mut World,
    source: SignalEntity,
    dependent: SignalEntity,
    input: ErasedSignalValue,
) -> bool {
    if let Ok(mut entity) = world.get_entity_mut(*dependent)
        && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
    {
        buffer.push_replay_snapshot(source, input);
        true
    } else {
        false
    }
}

fn push_replay_snapshot_input(
    world: &mut World,
    source: SignalEntity,
    dependent: SignalEntity,
    input: ErasedSignalValue,
) -> bool {
    if let Ok(mut entity) = world.get_entity_mut(*dependent)
        && !entity.contains::<RegistrarInactive>()
        && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
    {
        buffer.push_replay_snapshot(source, input);
        true
    } else {
        false
    }
}

fn push_signal_input(world: &mut World, dependent: SignalEntity, input: ErasedSignalValue) -> bool {
    if let Ok(mut entity) = world.get_entity_mut(*dependent)
        && !entity.contains::<RegistrarInactive>()
        && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
    {
        buffer.push(input);
        true
    } else {
        false
    }
}

#[derive(Clone, Copy)]
enum ReplayDeliveryUpdate {
    None,
    Publish(ReplayRevision),
    Advance(ReplayRevision),
}

/// Returns `Some(true)` if `dependent` is a live ordered-input consumer, `Some(false)` if it is a
/// live non-ordered input recipient, or `None` if it cannot currently receive inputs.
fn classify_live_recipient(world: &World, dependent: SignalEntity) -> Option<bool> {
    let entity = world.get_entity(*dependent).ok()?;
    if entity.contains::<RegistrarInactive>() || !entity.contains::<SignalInputBuffer>() {
        return None;
    }
    let ordered = entity
        .get::<SystemRunner>()
        .is_some_and(|runner| runner.ordered_input_consumption.is_some());
    Some(ordered)
}

fn consumes_ordered_input(world: &World, dependent: SignalEntity) -> bool {
    world
        .get::<SystemRunner>(*dependent)
        .is_some_and(|runner| runner.ordered_input_consumption.is_some())
}

fn is_live_input_recipient(world: &World, dependent: SignalEntity) -> bool {
    world
        .get_entity(*dependent)
        .is_ok_and(|entity| !entity.contains::<RegistrarInactive>() && entity.contains::<SignalInputBuffer>())
}

fn require_broadcast_clone_fn(world: &World, source: SignalEntity) -> ErasedOutputClone {
    let delivery = world
        .get::<StoredOutputDelivery>(*source)
        .unwrap_or_else(|| panic!("signal {source:?} is missing stored output delivery policy"));
    delivery
        .clone_value()
        .unwrap_or_else(|| panic!("move-only signal {source:?} reached runtime with multiple active dependents"))
}

fn ensure_imperative_observation_can_share_output(world: &World, source: SignalEntity) {
    let delivery = world
        .get::<StoredOutputDelivery>(*source)
        .unwrap_or_else(|| panic!("signal {source:?} is missing stored output delivery policy"));
    assert!(
        delivery.clone_value().is_some(),
        "cannot imperatively evaluate move-only signal {source:?} while it has active or pending dependents; \
         returning BoxedSignalValue while forwarding downstream requires .broadcast() / OutputDelivery::broadcast()"
    );
}

#[cfg(any(test, feature = "imperative-signal-evaluation"))]
fn has_live_downstream_consumer(world: &World, source: SignalEntity) -> bool {
    let has_active = world.get::<ActiveSignalDependents>(*source).is_some_and(|dependents| {
        dependents
            .0
            .iter()
            .copied()
            .any(|dependent| is_live_input_recipient(world, dependent))
    });
    let has_replay = world
        .get::<ReplayConsumingRelationships>(*source)
        .is_some_and(|relationships| {
            relationships
                .0
                .keys()
                .copied()
                .any(|dependent| is_live_input_recipient(world, dependent))
        });
    has_active || has_replay
}

fn get_live_dependents_sorted_filtered(
    world: &World,
    source: SignalEntity,
    mut predicate: impl FnMut(&World, SignalEntity) -> bool,
) -> Vec<SignalEntity> {
    let mut dependents = world
        .get::<ActiveSignalDependents>(*source)
        .map(|dependents| {
            dependents
                .0
                .iter()
                .copied()
                .filter(|&dependent| is_live_input_recipient(world, dependent) && predicate(world, dependent))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    dependents.sort_by(signal_identity_cmp);
    dependents.dedup();
    dependents
}

fn forward_output(world: &mut World, source: SignalEntity, dependents: &[SignalEntity], output: ErasedSignalValue) {
    match dependents {
        [] => {}
        [dependent] => {
            push_signal_input(world, *dependent, output);
        }
        [rest @ .., last] => {
            let clone_value = require_broadcast_clone_fn(world, source);
            let clones = rest
                .iter()
                .map(|dependent| (*dependent, clone_value(&*output)))
                .collect::<Vec<_>>();
            for (dependent, value) in clones {
                push_signal_input(world, dependent, value);
            }
            push_signal_input(world, *last, output);
        }
    }
}

#[cfg(test)]
fn deliver_accepted_signal_output(
    world: &mut World,
    source: SignalEntity,
    accepted: AcceptedSignalOutput,
    enqueue: impl FnMut(&mut World, SignalEntity, ErasedSignalValue) -> bool,
) {
    deliver_accepted_signal_output_with_observer(world, source, None, accepted, &mut None, enqueue);
}

fn deliver_accepted_signal_output_with_observer(
    world: &mut World,
    source: SignalEntity,
    runner: Option<&SystemRunner>,
    accepted: AcceptedSignalOutput,
    observe_target_output: &mut Option<TargetOutputObserver<'_>>,
    mut enqueue: impl FnMut(&mut World, SignalEntity, ErasedSignalValue) -> bool,
) {
    let AcceptedSignalOutput {
        output,
        replay_revision,
    } = accepted;

    // Fast path for ordinary/broadcast delivery when there are no replay-consuming relationships
    // and no imperative observer: zero BTreeSet allocations and zero Vec allocations for 0 or 1
    // active dependents.
    if observe_target_output.is_none()
        && world
            .get::<ReplayConsumingRelationships>(*source)
            .is_none_or(|relationships| relationships.0.is_empty())
    {
        let Some(dependents_set) = world.get::<ActiveSignalDependents>(*source) else {
            return;
        };
        match dependents_set.0.len() {
            0 => return,
            1 => {
                let dependent = *dependents_set.0.iter().next().expect("len == 1");
                if classify_live_recipient(world, dependent) == Some(false) {
                    enqueue(world, dependent, output);
                }
                return;
            }
            _ => {
                let mut accepted_dependents = dependents_set
                    .0
                    .iter()
                    .copied()
                    .filter(|&dependent| classify_live_recipient(world, dependent) == Some(false))
                    .collect::<Vec<_>>();
                accepted_dependents.sort_by(signal_identity_cmp);
                accepted_dependents.dedup();
                match accepted_dependents.as_slice() {
                    [] => {}
                    [dependent] => {
                        enqueue(world, *dependent, output);
                    }
                    [rest @ .., last] => {
                        let clone_value = require_broadcast_clone_fn(world, source);
                        let clones = rest
                            .iter()
                            .map(|&dependent| (dependent, clone_value(&*output)))
                            .collect::<Vec<_>>();
                        for (dependent, cloned) in clones {
                            enqueue(world, dependent, cloned);
                        }
                        enqueue(world, *last, output);
                    }
                }
                return;
            }
        }
    }

    let active_dependents = world
        .get::<ActiveSignalDependents>(*source)
        .map(|dependents| dependents.0.iter().copied().collect::<BTreeSet<_>>())
        .unwrap_or_default();
    let mut relationships = world.get::<ReplayConsumingRelationships>(*source).cloned();
    let mut dependents = active_dependents.clone();
    if let Some(relationships) = relationships.as_ref() {
        dependents.extend(relationships.0.keys().copied());
    }

    let mut deliveries = Vec::new();
    let mut accepted_dependents = Vec::new();
    for dependent in dependents {
        if consumes_ordered_input(world, dependent) || !is_live_input_recipient(world, dependent) {
            continue;
        }
        let Some(relationship_state) = relationships
            .as_ref()
            .and_then(|relationships| relationships.0.get(&dependent))
        else {
            if active_dependents.contains(&dependent) {
                accepted_dependents.push((dependent, ReplayDeliveryUpdate::None));
            }
            continue;
        };
        let accepted_revision = replay_revision
            .unwrap_or_else(|| panic!("replay-consuming relationship from {source:?} received a cursor-free output"));
        match relationship_state {
            ReplayRelationshipState::Pending => {
                let metadata = world.get::<RegistrationReplayCacheState>(*source).unwrap_or_else(|| {
                    panic!("replay-consuming relationship source {source:?} lost its coherent replay contract")
                });
                let expected_event_type = metadata.contract.event_type();
                let inspection = metadata.contract.inspect(world, source).unwrap_or_else(|| {
                    panic!("replay-consuming relationship source {source:?} lost its typed replay cache")
                });
                let ErasedReplaySnapshot::Ready {
                    snapshot,
                    revision: snapshot_revision,
                } = inspection
                else {
                    panic!("accepted replay output from {source:?} did not publish an authoritative snapshot");
                };
                assert_eq!(
                    erased_type_id(&snapshot),
                    expected_event_type,
                    "accepted replay output from {source:?} produced the wrong snapshot event type"
                );
                assert_eq!(
                    snapshot_revision, accepted_revision,
                    "accepted replay output and authoritative snapshot revisions diverged for {source:?}"
                );
                deliveries.push((
                    dependent,
                    snapshot,
                    ReplayDeliveryUpdate::Publish(accepted_revision),
                    true,
                ));
            }
            ReplayRelationshipState::Published { cursor } if *cursor < accepted_revision => {
                let metadata = world.get::<RegistrationReplayCacheState>(*source).unwrap_or_else(|| {
                    panic!("replay-consuming relationship source {source:?} lost its coherent replay contract")
                });
                let has_pending_snapshot = metadata.contract.coalesces_pending_snapshot()
                    && world
                        .get::<SignalInputBuffer>(*dependent)
                        .is_some_and(|buffer| buffer.has_pending_replay_snapshot(source));
                if has_pending_snapshot {
                    let metadata = world.get::<RegistrationReplayCacheState>(*source).unwrap_or_else(|| {
                        panic!("replay-consuming relationship source {source:?} lost its coherent replay contract")
                    });
                    let ErasedReplaySnapshot::Ready {
                        snapshot,
                        revision: snapshot_revision,
                    } = metadata.contract.inspect(world, source).unwrap_or_else(|| {
                        panic!("replay-consuming relationship source {source:?} lost its typed replay cache")
                    })
                    else {
                        panic!("accepted replay output from {source:?} did not publish an authoritative snapshot");
                    };
                    assert_eq!(
                        snapshot_revision, accepted_revision,
                        "accepted replay output and authoritative snapshot revisions diverged for {source:?}"
                    );
                    deliveries.push((
                        dependent,
                        snapshot,
                        ReplayDeliveryUpdate::Advance(accepted_revision),
                        true,
                    ));
                } else {
                    accepted_dependents.push((dependent, ReplayDeliveryUpdate::Advance(accepted_revision)));
                }
            }
            ReplayRelationshipState::Published { .. } => {}
        }
    }

    match (observe_target_output.as_deref_mut(), accepted_dependents.as_slice()) {
        (None, []) => {}
        (None, [(dependent, update)]) => {
            deliveries.push((*dependent, output, *update, false));
        }
        (None, [rest @ .., last]) => {
            let clone_value = require_broadcast_clone_fn(world, source);
            for &(dependent, update) in rest {
                deliveries.push((dependent, clone_value(&*output), update, false));
            }
            deliveries.push((last.0, output, last.1, false));
        }
        (Some(observe), []) => {
            if !deliveries.is_empty() {
                ensure_imperative_observation_can_share_output(world, source);
            }
            let runner = runner.expect("observed signal execution must supply SystemRunner");
            observe(runner, output);
        }
        (Some(observe), [rest @ .., last]) => {
            ensure_imperative_observation_can_share_output(world, source);
            let runner = runner.expect("observed signal execution must supply SystemRunner");
            let clone_value = require_broadcast_clone_fn(world, source);
            let observed_output = clone_value(&*output);
            for &(dependent, update) in rest {
                deliveries.push((dependent, clone_value(&*output), update, false));
            }
            deliveries.push((last.0, output, last.1, false));
            observe(runner, observed_output);
        }
    }

    let mut newly_published = BTreeSet::new();
    for (dependent, output, update, replay_snapshot) in deliveries {
        let enqueued = if replay_snapshot {
            push_replay_snapshot_input(world, source, dependent, output)
        } else {
            enqueue(world, dependent, output)
        };
        if !enqueued {
            continue;
        }
        match update {
            ReplayDeliveryUpdate::None => {}
            ReplayDeliveryUpdate::Publish(revision) => {
                let state = relationships
                    .as_mut()
                    .and_then(|relationships| relationships.0.get_mut(&dependent))
                    .unwrap_or_else(|| {
                        panic!("pending replay relationship from {source:?} disappeared during delivery")
                    });
                *state = ReplayRelationshipState::Published { cursor: revision };
                newly_published.insert(dependent);
            }
            ReplayDeliveryUpdate::Advance(revision) => {
                let state = relationships
                    .as_mut()
                    .and_then(|relationships| relationships.0.get_mut(&dependent))
                    .unwrap_or_else(|| {
                        panic!("published replay relationship from {source:?} disappeared during delivery")
                    });
                let ReplayRelationshipState::Published { cursor } = state else {
                    panic!("published replay relationship from {source:?} changed state during delivery");
                };
                if *cursor < revision {
                    *cursor = revision;
                }
            }
        }
    }

    if let Some(relationships) = relationships {
        world.entity_mut(*source).insert(relationships);
    }
    if !newly_published.is_empty() {
        {
            let mut source_entity = world.entity_mut(*source);
            if let Some(mut dependents) = source_entity.get_mut::<ActiveSignalDependents>() {
                dependents.0.extend(newly_published.iter().copied());
            } else {
                source_entity.insert(ActiveSignalDependents(newly_published.iter().copied().collect()));
            }
        }
        if let Some(mut graph) = world.get_resource_mut::<SignalGraphState>() {
            graph.edge_change_seeds.extend(newly_published);
        }
    }
}

fn signal_is_runnable(world: &World, signal: SignalEntity) -> bool {
    world.get_entity(*signal).is_ok_and(|entity| {
        !entity.contains::<RegistrarInactive>() && !entity.get::<SignalClaimCount>().is_some_and(|count| count.0 == 0)
    })
}

pub(crate) fn record_pending_after_output_cleanup(world: &mut World, owner: SignalEntity) {
    if let Some(mut state) = world.get_resource_mut::<SignalGraphState>() {
        state.pending_after_output_cleanup.push(owner);
    }
}

pub(crate) fn take_pending_after_output_cleanup(world: &mut World) -> Option<Vec<SignalEntity>> {
    world
        .get_resource_mut::<SignalGraphState>()
        .map(|mut state| core::mem::take(&mut state.pending_after_output_cleanup))
}

/// Runs a signal node: reads inputs from its [`SignalInputBuffer`], executes the system,
/// and writes outputs to downstream signals' buffers.
///
/// Avoids holding references to [`SignalGraphState`] during execution because signal
/// systems may materialize new registrar plans that update the graph.
fn run_signal_node(world: &mut World, signal: SignalEntity) {
    execute_signal_node(world, signal, None);
}

fn execute_signal_node(
    world: &mut World,
    signal: SignalEntity,
    mut observe_target_output: Option<TargetOutputObserver<'_>>,
) {
    // Single entity lookup for liveness, claim count, runner, static-root check, and input buffer.
    let (runner, signal_inputs, input_epoch, is_root) = {
        let Ok(mut entity) = world.get_entity_mut(*signal) else {
            return;
        };
        if entity.contains::<RegistrarInactive>() || entity.get::<SignalClaimCount>().is_some_and(|count| count.0 == 0)
        {
            return;
        }
        let is_root = entity
            .get::<StaticSignalDependencies>()
            .is_none_or(|dependencies| dependencies.0.is_empty());
        let Some(runner) = entity.get::<SystemRunner>().cloned() else {
            let upstreams = get_static_dependencies(world, signal);
            let downstreams = get_downstreams(world, signal);
            panic!(
                "missing SystemRunner for signal {:?} during processing (entity exists). upstreams={:?}, downstreams={:?}",
                signal, upstreams, downstreams
            );
        };
        if !is_root
            && entity
                .get::<SignalInputBuffer>()
                .is_none_or(|buffer| buffer.0.is_empty())
        {
            return;
        }
        let (signal_inputs, input_epoch) = entity
            .get_mut::<SignalInputBuffer>()
            .map(|mut buffer| (buffer.take(), buffer.2))
            .unwrap_or_default();

        (runner, signal_inputs, input_epoch, is_root)
    };

    let deliver_accepted =
        |world: &mut World, output: ErasedSignalValue, observe_target_output: &mut Option<TargetOutputObserver<'_>>| {
            if signal_is_runnable(world, signal)
                && let Some(accepted) = runner.accept_output(world, signal, output)
            {
                deliver_accepted_signal_output_with_observer(
                    world,
                    signal,
                    Some(&runner),
                    accepted,
                    observe_target_output,
                    push_signal_input,
                );
            }
        };

    let route_produced_output = |world: &mut World,
                                 output: ErasedSignalValue,
                                 observe_target_output: &mut Option<TargetOutputObserver<'_>>|
     -> Option<ErasedSignalValue> {
        if !signal_is_runnable(world, signal) {
            return None;
        }
        let has_replay_relationships = world
            .get::<ReplayConsumingRelationships>(*signal)
            .is_some_and(|relationships| !relationships.0.is_empty());
        if observe_target_output.is_none() && runner.replay_contract.is_none() && !has_replay_relationships {
            let dependents = world.get::<ActiveSignalDependents>(*signal)?;
            if dependents.0.is_empty() {
                return None;
            }
            if dependents.0.len() == 1 {
                let dependent = *dependents.0.iter().next().expect("len == 1");
                return match classify_live_recipient(world, dependent) {
                    None => None,
                    Some(true) => {
                        push_signal_input(world, dependent, output);
                        None
                    }
                    Some(false) => Some(output),
                };
            }
        }

        let ordered_dependents = get_live_dependents_sorted_filtered(world, signal, consumes_ordered_input);
        let has_non_ordered_live_dependent = world.get::<ActiveSignalDependents>(*signal).is_some_and(|deps| {
            deps.0
                .iter()
                .copied()
                .any(|dependent| classify_live_recipient(world, dependent) == Some(false))
        });
        let needs_accepted_path = observe_target_output.is_some()
            || runner.replay_contract.is_some()
            || has_replay_relationships
            || has_non_ordered_live_dependent;

        if ordered_dependents.is_empty() {
            return needs_accepted_path.then_some(output);
        }
        if !needs_accepted_path {
            forward_output(world, signal, &ordered_dependents, output);
            return None;
        }
        if ordered_dependents.len() == 1
            && !has_non_ordered_live_dependent
            && !has_replay_relationships
            && observe_target_output.is_none()
            && let Some(contract) = &runner.replay_contract
        {
            let cloned = contract.clone_event(&*output);
            push_signal_input(world, ordered_dependents[0], cloned);
            return Some(output);
        }
        if observe_target_output.is_some() {
            ensure_imperative_observation_can_share_output(world, signal);
        }
        let clone_value = require_broadcast_clone_fn(world, signal);
        let clones = ordered_dependents
            .iter()
            .map(|dependent| (*dependent, clone_value(&*output)))
            .collect::<Vec<_>>();
        for (dependent, value) in clones {
            push_signal_input(world, dependent, value);
        }
        Some(output)
    };

    if is_root {
        if let Some(output) = runner.run(world, Box::new(()))
            && let Some(output) = route_produced_output(world, output, &mut observe_target_output)
        {
            deliver_accepted(world, output, &mut observe_target_output);
        }
        return;
    }

    let mut final_output = None;
    let mut first_iteration = true;
    for input in signal_inputs {
        if !first_iteration {
            let still_valid = world.get_entity(*signal).is_ok_and(|entity| {
                !entity.contains::<RegistrarInactive>()
                    && !entity.get::<SignalClaimCount>().is_some_and(|count| count.0 == 0)
                    && entity
                        .get::<SignalInputBuffer>()
                        .is_some_and(|buffer| buffer.2 == input_epoch)
            });
            if !still_valid {
                break;
            }
        }
        first_iteration = false;
        let Some(output) = runner.run(world, input) else {
            continue;
        };
        let Some(output) = route_produced_output(world, output, &mut observe_target_output) else {
            continue;
        };
        if runner.collection_poll_append.is_some() {
            // Accept before the next callback so replay and forwarding share each batch's revision.
            deliver_accepted(world, output, &mut observe_target_output);
        } else {
            final_output = Some(output);
        }
    }
    if let Some(final_output) = final_output {
        deliver_accepted(world, final_output, &mut observe_target_output);
    }
}

fn sync_edge_change_levels(world: &mut World, state: &mut SignalGraphState) {
    drop(update_edge_change_levels(world, state));
}

fn process_signal_graph_for_schedule_inner(world: &mut World, schedule: InternedScheduleLabel) {
    world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
        sync_edge_change_levels(world, &mut state);
        state.is_processing = true;
    });

    // Track signals we've already processed this frame to avoid double-processing
    let mut processed: HashSet<SignalEntity> = HashSet::default();
    let mut level_buf: Vec<SignalEntity> = Vec::new();

    // Phase 2: Process signals level-by-level using persistent inputs.
    // Signal processing may materialize registrar plans or promote dynamic owners via
    // `sync_dynamic_owner_levels`, so fetch each bucket dynamically and verify each signal is
    // still active at `level_idx` in `schedule` before running it.
    let mut level_idx = 0usize;
    loop {
        {
            let state = world.resource::<SignalGraphState>();
            let Some(levels) = state.by_schedule.get(&schedule) else {
                break;
            };
            if level_idx >= levels.len() {
                break;
            }
            level_buf.clear();
            level_buf.extend_from_slice(&levels[level_idx]);
        }
        let expected_level = u32::try_from(level_idx).ok();
        for &signal in &level_buf {
            if processed.contains(&signal) {
                continue;
            }
            let still_at_level = {
                let state = world.resource::<SignalGraphState>();
                state.levels.get(&signal).copied() == expected_level
                    && state.signal_schedules.get(&signal).copied() == Some(schedule)
            };
            if still_at_level {
                processed.insert(signal);
                run_signal_node(world, signal);
            }
        }
        level_idx += 1;
    }

    // Phase 2b: Process signals registered during processing (registration recursion).
    // This handles cases like child_signal spawning elements with their own signals.
    let (recursion_limit, limit_behavior) = {
        let state = world.resource::<SignalGraphState>();
        (state.registration_recursion_limit, state.on_recursion_limit_exceeded)
    };
    let mut recursion_pass = 0usize;

    loop {
        // Update levels for new signals and get the seeds that were processed
        let new_seeds: Vec<SignalEntity> = world
            .resource_scope(|world, mut state: Mut<SignalGraphState>| update_edge_change_levels(world, &mut state));

        if new_seeds.is_empty() {
            break;
        }

        // Get signals in this schedule that we haven't processed yet
        let new_signals: Vec<SignalEntity> = new_seeds
            .into_iter()
            .filter(|signal| {
                (!processed.contains(signal)
                    || world
                        .get::<SignalInputBuffer>(**signal)
                        .is_some_and(|buffer| !buffer.0.is_empty()))
                    && world
                        .resource::<SignalGraphState>()
                        .signal_schedules
                        .get(signal)
                        .is_some_and(|effective_schedule| *effective_schedule == schedule)
            })
            .collect();

        if new_signals.is_empty() {
            break;
        }

        recursion_pass += 1;

        if recursion_pass > recursion_limit {
            match limit_behavior {
                RecursionLimitBehavior::Panic => {
                    panic!(
                        "Signal registration recursion limit exceeded ({} passes) in schedule {:?}. \
                             This usually indicates an infinite loop where signals keep spawning new signals. \
                             Processed {} signals before limit was reached. \
                             Use `JonmoPlugin::on_recursion_limit_exceeded` to change this behavior.",
                        recursion_limit,
                        schedule,
                        processed.len()
                    );
                }
                RecursionLimitBehavior::Warn => {
                    #[cfg(feature = "tracing")]
                    bevy_log::warn!(
                        "Signal registration recursion limit exceeded ({} passes) in schedule {:?}. \
                             This may indicate an infinite loop where signals keep spawning new signals. \
                             Processed {} signals so far.",
                        recursion_limit,
                        schedule,
                        processed.len()
                    );
                    break;
                }
                RecursionLimitBehavior::Silent => {
                    break;
                }
            }
        }

        // Process new signals and their downstreams in topological order
        let new_levels = downstream_levels_from_seeds(world, &new_signals);
        for level in new_levels {
            for signal in level {
                let has_pending_input = world
                    .get::<SignalInputBuffer>(*signal)
                    .is_some_and(|buffer| !buffer.0.is_empty());
                if (processed.insert(signal) || has_pending_input)
                    && world
                        .resource::<SignalGraphState>()
                        .signal_schedules
                        .get(&signal)
                        .is_some_and(|effective_schedule| *effective_schedule == schedule)
                {
                    run_signal_node(world, signal);
                }
            }
        }
    }

    // Phase 3: Cleanup
    crate::dynamic::flush_all_after_output_cleanup(world);
    let mut state = world.resource_mut::<SignalGraphState>();
    state.is_processing = false;
    apply_deferred_removals(&mut state);
}

/// Creates a system that processes only signals tagged for the specified schedule.
///
/// Uses persistent per-signal [`SignalInputBuffer`]s to enable cross-schedule data flow.
/// Includes a fixpoint loop to process signals that are registered during processing.
#[allow(clippy::std_instead_of_core)]
pub(crate) fn process_signal_graph_for_schedule(schedule: InternedScheduleLabel) -> impl FnMut(&mut World) {
    move |world: &mut World| {
        #[cfg(feature = "std")]
        {
            let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
                process_signal_graph_for_schedule_inner(world, schedule);
            }));
            if let Err(payload) = result {
                {
                    let mut state = world.resource_mut::<SignalGraphState>();
                    state.is_processing = false;
                    apply_deferred_removals(&mut state);
                }
                std::panic::resume_unwind(payload);
            }
        }
        #[cfg(not(feature = "std"))]
        process_signal_graph_for_schedule_inner(world, schedule);
    }
}

/// A detectable failure to release one signal claim.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignalReleaseError {
    /// The claimed signal entity no longer exists in this world.
    MissingEntity(SignalEntity),
    /// The claimed entity is not a materialized signal entity.
    MissingClaimCount(SignalEntity),
    /// Releasing the claim would underflow a signal's checked count.
    ClaimCountUnderflow(SignalEntity),
    /// Existing registrar-managed runtime state is inconsistent with its canonical topology.
    InconsistentExistingRuntimeState {
        /// The inconsistent registrar-managed entity.
        entity: SignalEntity,
    },
}

/// One move-only direct ownership token for a signal terminal.
///
/// A claim stores only its terminal identity; it does not store a transitive ownership path.
/// Registering another direct claim on an already-active terminal increments only that terminal.
/// Canonical immediate dependencies are recursively acquired only when a terminal transitions
/// `0 -> 1`.
///
/// A claim must be explicitly [`release`](Self::release)d or moved into a [`SignalClaims`] owner.
/// Dropping a raw claim does not release it. Claims deliberately implement neither [`Clone`] nor
/// [`Copy`] and must be released against the same [`World`] in which their terminal was
/// materialized. Jonmo does not add a separate world identifier to the token.
///
/// ```compile_fail,E0277
/// use jonmo::graph::SignalClaim;
/// fn require_clone<T: Clone>() {}
/// require_clone::<SignalClaim>();
/// ```
///
/// ```compile_fail,E0277
/// use jonmo::graph::SignalClaim;
/// fn require_copy<T: Copy>() {}
/// require_copy::<SignalClaim>();
/// ```
///
/// ```compile_fail,E0277
/// use jonmo::graph::{SignalClaim, SignalEntity};
/// fn counterfeit(entity: SignalEntity) -> SignalClaim { entity.into() }
/// ```
///
/// ```compile_fail,E0451
/// use jonmo::graph::{SignalClaim, SignalEntity};
/// fn counterfeit(terminal: SignalEntity) -> SignalClaim { SignalClaim { terminal } }
/// ```
///
/// ```compile_fail,E0614
/// use jonmo::graph::{SignalClaim, SignalEntity};
/// fn cannot_deref(claim: SignalClaim) -> SignalEntity { *claim }
/// ```
///
/// ```compile_fail,E0451
/// use jonmo::graph::{SignalClaim, SignalEntity};
/// fn extract(claim: SignalClaim) -> SignalEntity {
///     let SignalClaim { terminal } = claim;
///     terminal
/// }
/// ```
#[must_use = "a SignalClaim must be released or moved into a SignalClaims owner"]
#[derive(Debug)]
pub struct SignalClaim {
    terminal: SignalEntity,
}

impl SignalClaim {
    fn new(terminal: SignalEntity) -> Self {
        Self { terminal }
    }

    /// Returns the copyable full-generational terminal identity without consuming this claim.
    ///
    /// The returned [`SignalEntity`] carries no ownership or release authority.
    pub fn entity(&self) -> SignalEntity {
        self.terminal
    }

    /// Preflights and releases this direct claim.
    ///
    /// A final `1 -> 0` transition deactivates the terminal immediately, clears its active runtime
    /// state, and releases its canonical immediate dependency claims. Structural candidate reaping
    /// is a later cleanup operation. On a detectable rejection, committed graph state is unchanged
    /// and the error is returned together with the original still-armed claim.
    pub fn try_release(self, world: &mut World) -> Result<(), (SignalReleaseError, SignalClaim)> {
        let release = preflight_signal_claim_release(world, self)?;
        cleanup::process_signal_claim_release(world, release);
        Ok(())
    }

    /// Releases this claim.
    ///
    /// A final `1 -> 0` transition deactivates immediately; eligible entity reaping happens later.
    ///
    /// # Panics
    ///
    /// Panics if release preflight detects a missing entity, count error, or inconsistent runtime
    /// state. Use [`try_release`](Self::try_release) to recover the original armed claim on error.
    ///
    /// ```compile_fail,E0382
    /// use bevy_ecs::world::World;
    /// use jonmo::graph::SignalClaim;
    /// fn cannot_release_twice(claim: SignalClaim, world: &mut World) {
    ///     claim.release(world);
    ///     claim.release(world);
    /// }
    /// ```
    #[track_caller]
    pub fn release(self, world: &mut World) {
        if let Err((error, _claim)) = self.try_release(world) {
            panic!("failed to release signal claim: {error:?}");
        }
    }
}

fn release_signal_claims(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let claims = core::mem::take(&mut world.entity_mut(entity).get_mut::<SignalClaims>().unwrap().0);
    let mut commands = world.commands();
    for claim in claims {
        commands.queue(|world: &mut World| claim.release(world));
    }
}

/// Owns [`SignalClaim`]s for the lifetime of an [`Entity`].
///
/// Moving claims into this component transfers their release obligations. Component replacement,
/// removal, or owner despawn moves every stored claim into framework cleanup and releases each
/// exactly once.
///
/// ```compile_fail,E0277
/// use jonmo::graph::SignalClaims;
/// fn require_clone<T: Clone>() {}
/// require_clone::<SignalClaims>();
/// ```
///
/// Stored claims cannot be copied or moved out through a borrowed component:
///
/// ```compile_fail,E0616
/// use jonmo::graph::{SignalClaim, SignalClaims};
///
/// fn take_claim(claims: &mut SignalClaims) -> SignalClaim {
///     claims.0.pop().unwrap()
/// }
/// ```
#[derive(Component, Default)]
#[component(on_discard = release_signal_claims, on_remove = release_signal_claims)]
pub struct SignalClaims(Vec<SignalClaim>);

impl<T> From<T> for SignalClaims
where
    Vec<SignalClaim>: From<T>,
{
    #[inline]
    fn from(values: T) -> Self {
        SignalClaims(values.into())
    }
}

impl SignalClaims {
    /// Adds one owned claim to this component.
    pub fn add(&mut self, claim: SignalClaim) {
        self.0.push(claim);
    }

    /// Returns the number of claims owned by this component.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether this component owns no claims.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn reap_signal(world: &mut World, signal: SignalEntity) -> Vec<ReleasedSignalCandidate> {
    #[cfg(test)]
    trace_ownership_stage(world, OwnershipWorklistStage::Reaped(signal));
    let dependencies = world
        .get::<StaticSignalDependencies>(*signal)
        .unwrap_or_else(|| panic!("non-registrar signal reached reaping: {signal:?}"))
        .0
        .to_vec();
    remove_signal_from_graph_state(world, signal);
    if let Ok(entity) = world.get_entity_mut(*signal) {
        entity.despawn();
    }

    let mut exposed = Vec::new();
    for dependency in dependencies {
        let Ok(mut dependency_entity) = world.get_entity_mut(*dependency) else {
            continue;
        };
        let Some(mut dependents) = dependency_entity.get_mut::<MaterializedSignalDependents>() else {
            continue;
        };
        dependents.0.remove(&signal);
        if dependents.0.is_empty() {
            exposed.push(ReleasedSignalCandidate::new(dependency));
        }
    }
    exposed
}

fn cleanup_stale_signal(world: &mut World, signal: SignalEntity, worklist: &mut CleanupWorklist) {
    #[cfg(test)]
    trace_ownership_stage(world, OwnershipWorklistStage::Candidate(signal));
    let has_live_retainers = if let Ok(mut entity) = world.get_entity_mut(*signal) {
        if let Some(mut retainers) = entity.get_mut::<SignalTerminalRetainers>() {
            retainers.0.retain(SignalTerminalRetainer::is_live);
            !retainers.0.is_empty()
        } else {
            false
        }
    } else {
        false
    };
    if !has_live_retainers {
        let maintenance = preflight_replay_maintenance_release(world, signal)
            .unwrap_or_else(|error| panic!("failed to preflight replay maintenance release: {error:?}"));
        if let Some(release) = maintenance {
            #[cfg(test)]
            trace_ownership_stage(world, OwnershipWorklistStage::MaintenanceQueued(signal));
            worklist.push_signal_claim_release(release);
            return;
        }
    }

    let eligible = world.get::<StaticSignalDependencies>(*signal).is_some()
        && world.get::<RegistrarInactive>(*signal).is_some()
        && world
            .get::<SignalClaimCount>(*signal)
            .map(|claim_count| **claim_count == 0)
            .unwrap_or(false)
        && should_despawn_signal(world, signal);
    if !eligible {
        return;
    }

    for candidate in reap_signal(world, signal) {
        worklist.push_stale_signal(candidate);
    }
}

fn signal_identity_cmp(left: &SignalEntity, right: &SignalEntity) -> core::cmp::Ordering {
    left.0.cmp(&right.0)
}

fn order_signal_candidate_tokens(
    world: &World,
    candidates: Vec<ReleasedSignalCandidate>,
) -> Vec<ReleasedSignalCandidate> {
    let mut candidate_tokens = BTreeMap::new();
    for candidate in candidates {
        candidate_tokens.entry(candidate.signal()).or_insert(candidate);
    }
    let candidates = candidate_tokens.keys().copied().collect::<BTreeSet<_>>();
    let mut adjacency = BTreeMap::<SignalEntity, Vec<SignalEntity>>::new();
    let mut incoming = BTreeMap::<SignalEntity, usize>::new();
    let mut pending = candidates.iter().copied().collect::<Vec<_>>();
    let mut discovered = BTreeSet::new();
    while let Some(entity) = pending.pop() {
        if !discovered.insert(entity) {
            continue;
        }
        incoming.entry(entity).or_insert(0);
        let mut dependencies = world
            .get::<StaticSignalDependencies>(*entity)
            .map(|dependencies| dependencies.0.to_vec())
            .unwrap_or_default();
        dependencies.sort_by(signal_identity_cmp);
        dependencies.dedup();
        for dependency in dependencies.iter().copied() {
            let count = incoming.entry(dependency).or_insert(0);
            *count = count
                .checked_add(1)
                .unwrap_or_else(|| panic!("candidate topology incoming degree overflowed for {dependency:?}"));
            pending.push(dependency);
        }
        adjacency.insert(entity, dependencies);
    }

    let mut ready = incoming
        .iter()
        .filter_map(|(entity, count)| (*count == 0).then_some(*entity))
        .collect::<BTreeSet<_>>();
    let mut processed = BTreeSet::new();
    let mut ordered = Vec::with_capacity(candidates.len());
    while let Some(entity) = ready.iter().next().copied() {
        ready.remove(&entity);
        if !processed.insert(entity) {
            continue;
        }
        if candidates.contains(&entity) {
            ordered.push(entity);
        }
        for dependency in adjacency.get(&entity).into_iter().flatten().copied() {
            let Some(count) = incoming.get_mut(&dependency) else {
                continue;
            };
            *count = count
                .checked_sub(1)
                .unwrap_or_else(|| panic!("candidate topology incoming degree underflowed for {dependency:?}"));
            if *count == 0 {
                ready.insert(dependency);
            }
        }
    }

    // Canonical cycles are rejected before commit. If corrupted topology reaches cleanup,
    // retain deterministic behavior without indexing or dropping candidate obligations.
    ordered.extend(
        candidates
            .into_iter()
            .filter(|candidate| !processed.contains(candidate)),
    );
    ordered
        .into_iter()
        .filter_map(|signal| candidate_tokens.remove(&signal))
        .collect()
}

#[cfg(test)]
fn order_signal_candidates(world: &World, candidates: Vec<ReleasedSignalCandidate>) -> Vec<SignalEntity> {
    order_signal_candidate_tokens(world, candidates)
        .into_iter()
        .map(|candidate| candidate.signal())
        .collect()
}

pub(crate) fn reap_signal_candidates(
    world: &mut World,
    candidates: Vec<ReleasedSignalCandidate>,
    worklist: &mut CleanupWorklist,
) {
    let mut ordered = order_signal_candidate_tokens(world, candidates).into_iter();
    while let Some(candidate) = ordered.next() {
        cleanup_stale_signal(world, candidate.signal(), worklist);
        if worklist.has_release_actions() {
            for remaining in ordered {
                worklist.push_stale_signal(remaining);
            }
            return;
        }
    }
}

struct ProjectedSignalRelease {
    signal: SignalEntity,
    expected_count: usize,
    transition: ProjectedSignalReleaseTransition,
}

struct ProjectedDependencyRelease {
    projection: ProjectedSignalRelease,
    remove_active_route: bool,
    remove_replay_relationship: bool,
}

struct ProjectedDynamicSubscriptionRelease {
    inspection: DynamicSubscriptionInspection,
    projection: ProjectedSignalRelease,
}

enum ProjectedSignalReleaseTransition {
    DecrementOnly,
    Deactivate {
        dependencies: Box<[ProjectedDependencyRelease]>,
        dynamic_subscriptions: Box<[ProjectedDynamicSubscriptionRelease]>,
    },
}

pub(crate) struct PreflightedSignalClaimRelease {
    claim: SignalClaim,
    projection: ProjectedSignalRelease,
}

fn inconsistent_release_runtime(entity: SignalEntity) -> SignalReleaseError {
    SignalReleaseError::InconsistentExistingRuntimeState { entity }
}

fn preflight_dependency_route(
    world: &World,
    dependency: SignalEntity,
    dependent: SignalEntity,
) -> Result<(bool, bool), SignalReleaseError> {
    let dependency_ref = world
        .get_entity(*dependency)
        .map_err(|_| SignalReleaseError::MissingEntity(dependency))?;
    let active_route = dependency_ref
        .get::<ActiveSignalDependents>()
        .is_some_and(|dependents| dependents.0.contains(&dependent));
    let relationship = dependency_ref
        .get::<ReplayConsumingRelationships>()
        .and_then(|relationships| relationships.0.get(&dependent));

    let consistent = if consumes_ordered_input(world, dependent) {
        active_route && relationship.is_none()
    } else {
        match dependency_ref.get::<RegistrationReplayCacheState>() {
            None => active_route && relationship.is_none(),
            Some(replay) => match replay.contract.readiness_and_event_type(world, dependency) {
                Some((ReplayCacheReadiness::Warming, event_type)) => {
                    event_type == replay.contract.event_type()
                        && !active_route
                        && matches!(relationship, Some(ReplayRelationshipState::Pending))
                }
                Some((ReplayCacheReadiness::Ready { revision }, event_type)) => {
                    event_type == replay.contract.event_type()
                        && active_route
                        && matches!(
                            relationship,
                            Some(ReplayRelationshipState::Published { cursor }) if *cursor <= revision
                        )
                }
                None => false,
            },
        }
    };
    if !consistent {
        return Err(inconsistent_release_runtime(dependent));
    }
    Ok((active_route, relationship.is_some()))
}

fn project_signal_release(
    world: &World,
    signal: SignalEntity,
    planned_decrements: &mut BTreeMap<SignalEntity, usize>,
    validated_levels: &mut BTreeMap<SignalEntity, u32>,
    transferred_dynamic_subscriptions: &[DynamicSubscriptionInspection],
) -> Result<ProjectedSignalRelease, SignalReleaseError> {
    let entity = world
        .get_entity(*signal)
        .map_err(|_| SignalReleaseError::MissingEntity(signal))?;
    let committed_count = entity
        .get::<SignalClaimCount>()
        .map(|count| **count)
        .ok_or(SignalReleaseError::MissingClaimCount(signal))?;
    let already_planned = planned_decrements.get(&signal).copied().unwrap_or(0);
    let expected_count = committed_count
        .checked_sub(already_planned)
        .ok_or(SignalReleaseError::ClaimCountUnderflow(signal))?;
    let remaining = expected_count
        .checked_sub(1)
        .ok_or(SignalReleaseError::ClaimCountUnderflow(signal))?;
    planned_decrements.insert(
        signal,
        already_planned
            .checked_add(1)
            .ok_or(SignalReleaseError::ClaimCountUnderflow(signal))?,
    );

    validate_existing_registration_runtime_memoized(world, signal, validated_levels, transferred_dynamic_subscriptions)
        .map_err(|_| inconsistent_release_runtime(signal))?;
    if remaining != 0 {
        return Ok(ProjectedSignalRelease {
            signal,
            expected_count,
            transition: ProjectedSignalReleaseTransition::DecrementOnly,
        });
    }

    let mut dependencies = entity
        .get::<StaticSignalDependencies>()
        .ok_or_else(|| inconsistent_release_runtime(signal))?
        .0
        .to_vec();
    dependencies.sort_by(signal_identity_cmp);
    dependencies.dedup();

    let dynamic_inspections = entity
        .get::<DynamicSignalOwner>()
        .map(DynamicSignalOwner::subscriptions)
        .unwrap_or_default();
    let mut projected_dynamic = Vec::with_capacity(dynamic_inspections.len());
    for inspection in dynamic_inspections {
        if inspection.target != signal
            || inspection.source != inspection.adapter
            || !world
                .get::<ActiveSignalDependents>(*inspection.source)
                .is_some_and(|dependents| dependents.0.contains(&signal))
            || world
                .get::<ReplayConsumingRelationships>(*inspection.source)
                .is_some_and(|relationships| relationships.0.contains_key(&signal))
        {
            return Err(inconsistent_release_runtime(signal));
        }
        let projection = project_signal_release(
            world,
            inspection.adapter,
            planned_decrements,
            validated_levels,
            transferred_dynamic_subscriptions,
        )?;
        projected_dynamic.push(ProjectedDynamicSubscriptionRelease { inspection, projection });
    }

    let mut projected_dependencies = Vec::with_capacity(dependencies.len());
    for dependency in dependencies {
        let (remove_active_route, remove_replay_relationship) = preflight_dependency_route(world, dependency, signal)?;
        let projection = project_signal_release(
            world,
            dependency,
            planned_decrements,
            validated_levels,
            transferred_dynamic_subscriptions,
        )?;
        projected_dependencies.push(ProjectedDependencyRelease {
            projection,
            remove_active_route,
            remove_replay_relationship,
        });
    }

    Ok(ProjectedSignalRelease {
        signal,
        expected_count,
        transition: ProjectedSignalReleaseTransition::Deactivate {
            dependencies: projected_dependencies.into_boxed_slice(),
            dynamic_subscriptions: projected_dynamic.into_boxed_slice(),
        },
    })
}

fn preflight_signal_release_projection(
    world: &World,
    terminal: SignalEntity,
) -> Result<ProjectedSignalRelease, SignalReleaseError> {
    project_signal_release(world, terminal, &mut BTreeMap::new(), &mut BTreeMap::new(), &[])
}

fn preflight_signal_claim_release(
    world: &World,
    claim: SignalClaim,
) -> Result<PreflightedSignalClaimRelease, (SignalReleaseError, SignalClaim)> {
    match preflight_signal_release_projection(world, claim.terminal) {
        Ok(projection) => Ok(PreflightedSignalClaimRelease { claim, projection }),
        Err(error) => Err((error, claim)),
    }
}

fn preflight_replay_maintenance_release(
    world: &mut World,
    terminal: SignalEntity,
) -> Result<Option<PreflightedSignalClaimRelease>, SignalReleaseError> {
    let Some(claim_terminal) = world
        .get::<ReplayMaintenanceClaim>(*terminal)
        .map(|maintenance| maintenance.0.terminal)
    else {
        return Ok(None);
    };
    assert_eq!(claim_terminal, terminal, "replay maintenance claim changed identity");
    let projection = preflight_signal_release_projection(world, terminal)?;
    let maintenance = world
        .get_entity_mut(*terminal)
        .expect("preflighted replay maintenance terminal disappeared")
        .take::<ReplayMaintenanceClaim>()
        .expect("preflighted replay maintenance claim disappeared");
    Ok(Some(PreflightedSignalClaimRelease {
        claim: maintenance.0,
        projection,
    }))
}

#[cfg(test)]
pub(crate) fn test_has_signal_lifetime_retention(world: &World, entity: Entity) -> bool {
    world
        .get::<SignalTerminalRetainers>(entity)
        .is_some_and(|retainers| retainers.0.iter().any(SignalTerminalRetainer::is_live))
        || world
            .get::<MaterializedSignalDependents>(entity)
            .is_some_and(|dependents| !dependents.0.is_empty())
}

fn should_despawn_signal(world: &World, signal: SignalEntity) -> bool {
    world.get_entity(*signal).ok().is_some_and(|entity| {
        let no_live_terminal_retainers = entity
            .get::<SignalTerminalRetainers>()
            .is_none_or(|retainers| retainers.0.iter().all(|retainer| !retainer.is_live()));
        let no_materialized_dependents = entity
            .get::<MaterializedSignalDependents>()
            .is_none_or(|dependents| dependents.0.is_empty());
        entity.contains::<StaticSignalDependencies>()
            && no_live_terminal_retainers
            && no_materialized_dependents
            && !entity.contains::<RegistrationCommitRetainers>()
    })
}

fn commit_dependency_route_removal(
    world: &mut World,
    dependency: SignalEntity,
    dependent: SignalEntity,
    remove_active_route: bool,
    remove_replay_relationship: bool,
) {
    let mut dependency_entity = world
        .get_entity_mut(*dependency)
        .unwrap_or_else(|_| panic!("preflighted release dependency {dependency:?} disappeared"));
    if remove_active_route {
        let mut active_dependents = dependency_entity
            .get_mut::<ActiveSignalDependents>()
            .unwrap_or_else(|| panic!("preflighted release dependency {dependency:?} lost its active routes"));
        assert!(
            active_dependents.0.remove(&dependent),
            "preflighted active route {dependency:?} -> {dependent:?} disappeared"
        );
    }
    if remove_replay_relationship {
        let mut relationships = dependency_entity
            .get_mut::<ReplayConsumingRelationships>()
            .unwrap_or_else(|| panic!("preflighted release dependency {dependency:?} lost its replay relationships"));
        assert!(
            relationships.0.remove(&dependent).is_some(),
            "preflighted replay relationship {dependency:?} -> {dependent:?} disappeared"
        );
        if relationships.0.is_empty() {
            dependency_entity.remove::<ReplayConsumingRelationships>();
        }
    }
}

fn commit_projected_signal_release(
    world: &mut World,
    projection: ProjectedSignalRelease,
    worklist: &mut CleanupWorklist,
) {
    let ProjectedSignalRelease {
        signal,
        expected_count,
        transition,
    } = projection;
    #[cfg(test)]
    trace_ownership_stage(world, OwnershipWorklistStage::Released(signal));
    {
        let mut count = world
            .get_mut::<SignalClaimCount>(*signal)
            .expect("preflighted release signal must retain its claim count");
        assert_eq!(
            count.0, expected_count,
            "preflighted release count diverged for {signal:?}"
        );
        count
            .try_decrement()
            .expect("preflighted release decrement must remain valid");
        if matches!(transition, ProjectedSignalReleaseTransition::DecrementOnly) {
            assert!(count.0 > 0, "preflighted decrement-only transition reached zero");
            return;
        }
        assert_eq!(count.0, 0, "preflighted zero transition must commit at zero");
    }

    let ProjectedSignalReleaseTransition::Deactivate {
        dependencies,
        dynamic_subscriptions,
    } = transition
    else {
        unreachable!("decrement-only release returned before deactivation")
    };

    {
        let mut entity = world
            .get_entity_mut(*signal)
            .unwrap_or_else(|_| panic!("preflighted registrar signal {signal:?} disappeared"));
        assert!(
            !entity.contains::<RegistrarInactive>(),
            "preflighted active registrar signal {signal:?} was already inactive"
        );
        entity.insert(RegistrarInactive);
        entity
            .get_mut::<SignalInputBuffer>()
            .expect("preflighted registrar signal must retain its input buffer")
            .clear();
    }
    remove_signal_from_graph_state(world, signal);

    let dynamic_releases = if let Some(mut owner) = world.get_mut::<DynamicSignalOwner>(*signal) {
        let mut subscriptions = owner.drain();
        let mut releases = Vec::with_capacity(dynamic_subscriptions.len());
        for projected in dynamic_subscriptions {
            let index = subscriptions
                .iter()
                .position(|subscription| {
                    subscription.inspect().source == projected.inspection.source
                        && subscription.inspect().target == projected.inspection.target
                        && subscription.inspect().slot == projected.inspection.slot
                })
                .unwrap_or_else(|| panic!("preflighted dynamic subscription disappeared from {signal:?}"));
            releases.push(PreflightedDynamicConnectionRelease {
                subscription: subscriptions.remove(index),
                projection: projected.projection,
            });
        }
        assert!(
            subscriptions.is_empty(),
            "dynamic owner {signal:?} retained unprojected subscriptions"
        );
        releases
    } else {
        assert!(
            dynamic_subscriptions.is_empty(),
            "preflighted dynamic owner {signal:?} disappeared"
        );
        Vec::new()
    };

    let mut dependency_claims = Vec::with_capacity(dependencies.len());
    for dependency in dependencies {
        commit_dependency_route_removal(
            world,
            dependency.projection.signal,
            signal,
            dependency.remove_active_route,
            dependency.remove_replay_relationship,
        );
        dependency_claims.push(ReleasedStaticDependencyClaim {
            projection: dependency.projection,
        });
    }
    worklist.push_stale_signal(ReleasedSignalCandidate::new(signal));
    worklist.push_static_dependency_releases(dependency_claims);
    worklist.push_dynamic_connection_releases(dynamic_releases);
}

pub(crate) fn commit_preflighted_signal_claim_release(
    world: &mut World,
    release: PreflightedSignalClaimRelease,
    worklist: &mut CleanupWorklist,
) {
    let PreflightedSignalClaimRelease { claim, projection } = release;
    assert_eq!(
        claim.terminal, projection.signal,
        "preflighted root claim changed identity"
    );
    commit_projected_signal_release(world, projection, worklist);
}

pub(crate) struct PreflightedDynamicConnectionRelease {
    subscription: DynamicSignalSubscription,
    projection: ProjectedSignalRelease,
}

pub(crate) fn commit_preflighted_dynamic_connection_release(
    world: &mut World,
    release: PreflightedDynamicConnectionRelease,
    worklist: &mut CleanupWorklist,
) {
    let PreflightedDynamicConnectionRelease {
        subscription,
        projection,
    } = release;
    let inspection = subscription.inspect();
    assert_eq!(inspection.source, inspection.adapter);
    assert_eq!(subscription.adapter_claim.entity(), projection.signal);
    let mut source = world
        .get_entity_mut(*subscription.connection.source())
        .unwrap_or_else(|_| panic!("dynamic connection source {:?} disappeared", inspection.source));
    let mut dependents = source
        .get_mut::<ActiveSignalDependents>()
        .unwrap_or_else(|| panic!("dynamic connection source {:?} lost active routes", inspection.source));
    assert!(
        dependents.0.remove(&subscription.connection.target()),
        "dynamic connection route {:?} -> {:?} disappeared",
        inspection.source,
        inspection.target
    );
    if let Some(mut state) = world.get_resource_mut::<SignalGraphState>() {
        state.edge_change_seeds.insert(inspection.target);
    }
    sync_dynamic_owner_levels(world, inspection.target);
    commit_projected_signal_release(world, projection, worklist);
}

pub(crate) fn preflight_dynamic_connection_releases(
    world: &World,
    subscriptions: Vec<DynamicSignalSubscription>,
) -> Result<Vec<PreflightedDynamicConnectionRelease>, SignalReleaseError> {
    let transferred_dynamic_subscriptions = subscriptions
        .iter()
        .map(DynamicSignalSubscription::inspect)
        .collect::<Vec<_>>();
    let mut planned_decrements = BTreeMap::new();
    let mut validated_levels = BTreeMap::new();
    subscriptions
        .into_iter()
        .map(|subscription| {
            let inspection = subscription.inspect();
            if inspection.source != inspection.adapter
                || !world
                    .get::<ActiveSignalDependents>(*inspection.source)
                    .is_some_and(|dependents| dependents.0.contains(&inspection.target))
            {
                return Err(inconsistent_release_runtime(inspection.target));
            }
            let projection = project_signal_release(
                world,
                inspection.adapter,
                &mut planned_decrements,
                &mut validated_levels,
                &transferred_dynamic_subscriptions,
            )?;
            Ok(PreflightedDynamicConnectionRelease {
                subscription,
                projection,
            })
        })
        .collect()
}

pub(crate) fn release_dynamic_subscriptions(world: &mut World, subscriptions: Vec<DynamicSignalSubscription>) {
    if subscriptions.is_empty() {
        return;
    }
    let releases = preflight_dynamic_connection_releases(world, subscriptions)
        .unwrap_or_else(|error| panic!("dynamic subscription cleanup preflight failed: {error:?}"));
    cleanup::process_dynamic_connection_releases(world, releases);
}

pub(crate) fn commit_released_static_dependency_claim(
    world: &mut World,
    claim: ReleasedStaticDependencyClaim,
    worklist: &mut CleanupWorklist,
) {
    commit_projected_signal_release(world, claim.projection, worklist);
}

fn inconsistent_existing_runtime(entity: SignalEntity) -> SignalRegistrationError {
    SignalRegistrationError::InconsistentExistingRuntimeState { entity }
}

fn validate_existing_registration_runtime_memoized(
    world: &World,
    root: SignalEntity,
    validated_levels: &mut BTreeMap<SignalEntity, u32>,
    transferred_dynamic_subscriptions: &[DynamicSubscriptionInspection],
) -> Result<(), SignalRegistrationError> {
    fn visit(
        world: &World,
        entity: SignalEntity,
        visiting: &mut BTreeSet<SignalEntity>,
        validated_levels: &mut BTreeMap<SignalEntity, u32>,
        transferred_dynamic_subscriptions: &[DynamicSubscriptionInspection],
    ) -> Result<u32, SignalRegistrationError> {
        if let Some(level) = validated_levels.get(&entity).copied() {
            return Ok(level);
        }
        if !visiting.insert(entity) {
            return Err(SignalRegistrationError::CycleDetected);
        }
        let entity_ref = world
            .get_entity(*entity)
            .map_err(|_| SignalRegistrationError::EntityUnavailable)?;
        let dependencies_component = entity_ref
            .get::<StaticSignalDependencies>()
            .ok_or_else(|| inconsistent_existing_runtime(entity))?;
        if !entity_ref.contains::<SystemRunner>()
            || !entity_ref.contains::<StoredOutputDelivery>()
            || !entity_ref.contains::<SignalInputBuffer>()
            || !entity_ref.contains::<SignalClaimCount>()
            || !entity_ref.contains::<MaterializedSignalDependents>()
            || !entity_ref.contains::<SignalTerminalRetainers>()
            || !entity_ref.contains::<ActiveSignalDependents>()
        {
            return Err(inconsistent_existing_runtime(entity));
        }

        let dependencies = dependencies_component.0.clone();
        let mut dynamic_subscriptions = entity_ref
            .get::<DynamicSignalOwner>()
            .map(DynamicSignalOwner::subscriptions)
            .unwrap_or_default();
        for subscription in transferred_dynamic_subscriptions
            .iter()
            .filter(|subscription| subscription.target == entity)
        {
            if !dynamic_subscriptions.contains(subscription) {
                dynamic_subscriptions.push(*subscription);
            }
        }
        for subscription in &dynamic_subscriptions {
            if subscription.target != entity
                || subscription.source != subscription.adapter
                || world
                    .get::<ReplayConsumingRelationships>(*subscription.source)
                    .is_some_and(|relationships| relationships.0.contains_key(&entity))
                || !world
                    .get::<ActiveSignalDependents>(*subscription.source)
                    .is_some_and(|dependents| dependents.0.contains(&entity))
            {
                return Err(inconsistent_existing_runtime(entity));
            }
        }
        let mut execution_dependencies = dependencies.to_vec();
        execution_dependencies.extend(dynamic_subscriptions.iter().map(|subscription| subscription.source));
        execution_dependencies.sort_by(signal_identity_cmp);
        execution_dependencies.dedup();
        let mut canonical_dependencies = dependencies.to_vec();
        canonical_dependencies.sort_by(signal_identity_cmp);
        canonical_dependencies.dedup();
        if canonical_dependencies.as_slice() != dependencies.as_ref() {
            return Err(inconsistent_existing_runtime(entity));
        }

        let runner = entity_ref
            .get::<SystemRunner>()
            .ok_or_else(|| inconsistent_existing_runtime(entity))?;
        let delivery = entity_ref
            .get::<StoredOutputDelivery>()
            .copied()
            .ok_or_else(|| inconsistent_existing_runtime(entity))?;
        if !delivery.is_coherent(runner.output_type)
            || !runner.output_delivery.is_coherent(runner.output_type)
            || delivery.key() != runner.output_delivery.key()
        {
            return Err(inconsistent_existing_runtime(entity));
        }
        if runner.ordered_input_consumption.is_some_and(|consumption| {
            !consumption.accepts_shape(
                dependencies.len(),
                runner.input_type,
                runner.output_type,
                runner.replay_key.is_some(),
            )
        }) {
            return Err(inconsistent_existing_runtime(entity));
        }
        let materialized_dependents = entity_ref
            .get::<MaterializedSignalDependents>()
            .map(|dependents| dependents.0.clone())
            .ok_or_else(|| inconsistent_existing_runtime(entity))?;
        let claim_count = entity_ref
            .get::<SignalClaimCount>()
            .map(|count| **count)
            .ok_or_else(|| inconsistent_existing_runtime(entity))?;
        let replay = entity_ref.get::<RegistrationReplayCacheState>();
        let replay_readiness = match (runner.replay_key, &runner.replay_contract, replay) {
            (None, None, None) => None,
            (Some(key), Some(runtime_contract), Some(metadata))
                if Arc::ptr_eq(runtime_contract, &metadata.contract)
                    && runner.output_type == key.event_type()
                    && key.cache_type() == runtime_contract.cache_type()
                    && key.event_type() == runtime_contract.event_type() =>
            {
                let (readiness, event_type) = runtime_contract
                    .readiness_and_event_type(world, entity)
                    .ok_or_else(|| inconsistent_existing_runtime(entity))?;
                if event_type != key.event_type() {
                    return Err(inconsistent_existing_runtime(entity));
                }
                Some(readiness)
            }
            _ => return Err(inconsistent_existing_runtime(entity)),
        };
        let maintenance_armed = entity_ref.contains::<ReplayMaintenanceClaim>();
        let has_live_terminal_retainer = entity_ref
            .get::<SignalTerminalRetainers>()
            .is_some_and(|retainers| retainers.0.iter().any(SignalTerminalRetainer::is_live));
        if maintenance_armed && (replay_readiness.is_none() || claim_count == 0)
            || replay_readiness.is_some() && has_live_terminal_retainer && !maintenance_armed
        {
            return Err(inconsistent_existing_runtime(entity));
        }
        for dependency in dependencies.iter().copied() {
            let dependency_ref = world
                .get_entity(*dependency)
                .map_err(|_| inconsistent_existing_runtime(entity))?;
            if let Some(consumption) = runner.ordered_input_consumption {
                let dependency_runner = dependency_ref
                    .get::<SystemRunner>()
                    .ok_or_else(|| inconsistent_existing_runtime(dependency))?;
                if !consumption.accepts_dependency(
                    dependency_runner.output_type,
                    dependency_runner.replay_key.is_some() || dependency_ref.contains::<RegistrationReplayCacheState>(),
                    dependency_runner.ordered_input_consumption,
                ) {
                    return Err(inconsistent_existing_runtime(entity));
                }
            }
            let reverse = dependency_ref
                .get::<MaterializedSignalDependents>()
                .ok_or_else(|| inconsistent_existing_runtime(dependency))?;
            if !reverse.0.contains(&entity) {
                return Err(inconsistent_existing_runtime(dependency));
            }
        }
        for dependent in materialized_dependents.iter().copied() {
            let dependent_dependencies = world
                .get::<StaticSignalDependencies>(*dependent)
                .ok_or_else(|| inconsistent_existing_runtime(entity))?;
            if !dependent_dependencies.0.contains(&entity) {
                return Err(inconsistent_existing_runtime(entity));
            }
        }
        let inactive = world.get::<RegistrarInactive>(*entity).is_some();
        let graph = world
            .get_resource::<SignalGraphState>()
            .ok_or(SignalRegistrationError::ScheduleNotRegistered)?;
        let bucket_locations = graph
            .by_schedule
            .iter()
            .flat_map(|(schedule, levels)| {
                levels.iter().enumerate().flat_map(move |(level, members)| {
                    members
                        .iter()
                        .filter(move |member| **member == entity)
                        .map(move |_| (*schedule, level as u32))
                })
            })
            .collect::<Vec<_>>();

        let active_outgoing = entity_ref
            .get::<ActiveSignalDependents>()
            .map(|dependents| dependents.0.iter().copied().collect::<BTreeSet<_>>())
            .ok_or_else(|| inconsistent_existing_runtime(entity))?;
        let replay_outgoing = entity_ref
            .get::<ReplayConsumingRelationships>()
            .map(|relationships| relationships.0.clone())
            .unwrap_or_default();
        let mut outgoing_targets = active_outgoing.clone();
        outgoing_targets.extend(replay_outgoing.keys().copied());
        if claim_count == 0 && !outgoing_targets.is_empty()
            || outgoing_targets.len() > 1 && delivery.kind() != OutputDeliveryKind::Broadcast
        {
            return Err(inconsistent_existing_runtime(entity));
        }

        let registrar_outgoing_route_is_consistent = |target: SignalEntity| {
            let active_route = active_outgoing.contains(&target);
            let relationship = replay_outgoing.get(&target);
            if let Some(consumption) = world
                .get::<SystemRunner>(*target)
                .and_then(|runner| runner.ordered_input_consumption)
            {
                return active_route
                    && relationship.is_none()
                    && consumption.accepts_dependency(
                        runner.output_type,
                        replay_readiness.is_some(),
                        runner.ordered_input_consumption,
                    );
            }
            match replay_readiness {
                None => active_route && relationship.is_none(),
                Some(ReplayCacheReadiness::Warming) => {
                    !active_route && matches!(relationship, Some(ReplayRelationshipState::Pending))
                }
                Some(ReplayCacheReadiness::Ready { revision }) => {
                    active_route
                        && matches!(
                            relationship,
                            Some(ReplayRelationshipState::Published { cursor }) if *cursor <= revision
                        )
                }
            }
        };

        for target in outgoing_targets.iter().copied() {
            let target_ref = world
                .get_entity(*target)
                .map_err(|_| inconsistent_existing_runtime(entity))?;
            let target_count = target_ref
                .get::<SignalClaimCount>()
                .map(|count| **count)
                .ok_or_else(|| inconsistent_existing_runtime(target))?;
            if target_count == 0 || target_ref.contains::<RegistrarInactive>() {
                return Err(inconsistent_existing_runtime(entity));
            }
            let target_dependencies = target_ref
                .get::<StaticSignalDependencies>()
                .ok_or_else(|| inconsistent_existing_runtime(target))?;
            let dynamic_route = target_ref.get::<DynamicSignalOwner>().is_some_and(|owner| {
                owner
                    .subscriptions()
                    .iter()
                    .any(|subscription| subscription.source == entity && subscription.target == target)
            }) || transferred_dynamic_subscriptions
                .iter()
                .any(|subscription| subscription.source == entity && subscription.target == target);
            let target_level = graph.levels.get(&target).copied();
            let target_schedule = graph.signal_schedules.get(&target).copied();
            let target_bucket_count = target_schedule
                .zip(target_level)
                .map(|(schedule, level)| {
                    graph
                        .by_schedule
                        .get(&schedule)
                        .and_then(|levels| levels.get(level as usize))
                        .map(|members| members.iter().filter(|member| **member == target).count())
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            let target_bucket_consistent = if graph.is_processing {
                target_bucket_count <= 1
            } else {
                target_bucket_count == 1
            };
            let ownership_is_consistent = if dynamic_route {
                !target_dependencies.0.contains(&entity)
                    && !materialized_dependents.contains(&target)
                    && active_outgoing.contains(&target)
                    && !replay_outgoing.contains_key(&target)
            } else {
                target_dependencies.0.contains(&entity)
                    && materialized_dependents.contains(&target)
                    && registrar_outgoing_route_is_consistent(target)
            };
            if !ownership_is_consistent
                || !target_ref.contains::<SystemRunner>()
                || !target_ref.contains::<SignalInputBuffer>()
                || target_level.is_none()
                || target_schedule.is_none()
                || !target_bucket_consistent
            {
                return Err(inconsistent_existing_runtime(entity));
            }
        }

        for target in materialized_dependents.iter().copied() {
            let target_ref = world
                .get_entity(*target)
                .map_err(|_| inconsistent_existing_runtime(entity))?;
            let target_count = target_ref
                .get::<SignalClaimCount>()
                .map(|count| **count)
                .ok_or_else(|| inconsistent_existing_runtime(target))?;
            if target_count > 0 && !registrar_outgoing_route_is_consistent(target)
                || target_count == 0 && (active_outgoing.contains(&target) || replay_outgoing.contains_key(&target))
            {
                return Err(inconsistent_existing_runtime(entity));
            }
        }

        if claim_count == 0 {
            let has_incoming_route = execution_dependencies.iter().copied().any(|dependency| {
                world
                    .get::<ActiveSignalDependents>(*dependency)
                    .is_some_and(|dependents| dependents.0.contains(&entity))
                    || world
                        .get::<ReplayConsumingRelationships>(*dependency)
                        .is_some_and(|relationships| relationships.0.contains_key(&entity))
            });
            let deferred = graph.deferred_removals.contains(&entity);
            let cached_membership = graph
                .levels
                .get(&entity)
                .copied()
                .zip(graph.signal_schedules.get(&entity).copied());
            let membership_is_consistent = match cached_membership {
                Some((level, schedule)) if deferred && graph.is_processing => {
                    let expected_schedule = world
                        .get::<ScheduleTag>(*entity)
                        .map(|tag| tag.0)
                        .unwrap_or(graph.default_schedule);
                    schedule == expected_schedule && bucket_locations.as_slice() == [(schedule, level)]
                }
                _ => {
                    !graph.levels.contains_key(&entity)
                        && !graph.signal_schedules.contains_key(&entity)
                        && bucket_locations.is_empty()
                }
            };
            // Processing keeps old buckets until cleanup; inactivity and removed routes are
            // immediate.
            if !inactive
                || deferred && !graph.is_processing
                || !membership_is_consistent
                || !dynamic_subscriptions.is_empty()
                || entity_ref
                    .get::<SignalInputBuffer>()
                    .is_some_and(|buffer| !buffer.0.is_empty())
                || has_incoming_route
            {
                return Err(inconsistent_existing_runtime(entity));
            }
            visiting.remove(&entity);
            validated_levels.insert(entity, 0);
            return Ok(0);
        }

        if inactive || graph.deferred_removals.contains(&entity) {
            return Err(inconsistent_existing_runtime(entity));
        }
        let mut expected_level = 0;
        for dependency in execution_dependencies.iter().copied() {
            let dependency_count = world
                .get::<SignalClaimCount>(*dependency)
                .map(|count| **count)
                .ok_or(SignalRegistrationError::EntityUnavailable)?;
            if dependency_count == 0 {
                return Err(inconsistent_existing_runtime(entity));
            }
            let dependency_level = visit(
                world,
                dependency,
                visiting,
                validated_levels,
                transferred_dynamic_subscriptions,
            )?;
            let next_level = dependency_level
                .checked_add(1)
                .ok_or(SignalRegistrationError::GraphDepthOverflow)?;
            expected_level = expected_level.max(check_level_within_limit(next_level, u32::MAX)?);

            let active_route = world
                .get::<ActiveSignalDependents>(*dependency)
                .is_some_and(|dependents| dependents.0.contains(&entity));
            let relationship = world
                .get::<ReplayConsumingRelationships>(*dependency)
                .and_then(|relationships| relationships.0.get(&entity));
            let dynamic_route = dynamic_subscriptions
                .iter()
                .any(|subscription| subscription.source == dependency && subscription.target == entity);
            let route_is_consistent = if dynamic_route || runner.ordered_input_consumption.is_some() {
                relationship.is_none() && active_route
            } else {
                match world.get::<RegistrationReplayCacheState>(*dependency) {
                    None => relationship.is_none() && active_route,
                    Some(replay) => match replay.contract.readiness_and_event_type(world, dependency) {
                        Some((ReplayCacheReadiness::Warming, event_type)) => {
                            event_type == replay.contract.event_type()
                                && matches!(relationship, Some(ReplayRelationshipState::Pending))
                                && !active_route
                        }
                        Some((ReplayCacheReadiness::Ready { revision }, event_type)) => {
                            event_type == replay.contract.event_type()
                                && matches!(
                                    relationship,
                                    Some(ReplayRelationshipState::Published { cursor }) if *cursor <= revision
                                )
                                && active_route
                        }
                        None => return Err(inconsistent_existing_runtime(dependency)),
                    },
                }
            };
            if !route_is_consistent {
                return Err(inconsistent_existing_runtime(entity));
            }
        }

        let expected_schedule = world
            .get::<ScheduleTag>(*entity)
            .map(|tag| tag.0)
            .unwrap_or(graph.default_schedule);
        if graph.levels.get(&entity).copied() != Some(expected_level)
            || graph.signal_schedules.get(&entity).copied() != Some(expected_schedule)
        {
            return Err(inconsistent_existing_runtime(entity));
        }
        let expected_bucket_count = bucket_locations
            .iter()
            .filter(|(schedule, level)| *schedule == expected_schedule && *level == expected_level)
            .count();
        let bucket_is_consistent = if graph.is_processing {
            bucket_locations.is_empty() || (bucket_locations.len() == 1 && expected_bucket_count == 1)
        } else {
            bucket_locations.len() == 1 && expected_bucket_count == 1
        };
        if !bucket_is_consistent {
            return Err(inconsistent_existing_runtime(entity));
        }

        visiting.remove(&entity);
        validated_levels.insert(entity, expected_level);
        Ok(expected_level)
    }

    visit(
        world,
        root,
        &mut BTreeSet::new(),
        validated_levels,
        transferred_dynamic_subscriptions,
    )
    .map(|_| ())
}

#[cfg(test)]
fn validate_existing_registration_runtime(world: &World, root: SignalEntity) -> Result<(), SignalRegistrationError> {
    validate_existing_registration_runtime_memoized(world, root, &mut BTreeMap::new(), &[])
}

fn replay_route_disposition(
    dependency: Option<SignalEntity>,
    expected_event_type: TypeId,
    inspection: ErasedReplaySnapshot,
) -> Result<RegistrationRouteDisposition, SignalRegistrationError> {
    match inspection {
        ErasedReplaySnapshot::Warming => Ok(RegistrationRouteDisposition::ReplayPending),
        ErasedReplaySnapshot::Ready { snapshot, revision } => {
            if erased_type_id(&snapshot) != expected_event_type {
                return Err(dependency
                    .map(inconsistent_existing_runtime)
                    .unwrap_or(SignalRegistrationError::ConflictingCanonicalState));
            }
            Ok(RegistrationRouteDisposition::ReplayReady {
                snapshot,
                revision,
                event_type: expected_event_type,
            })
        }
    }
}

fn preflight_activation_route_disposition(
    world: &World,
    plan: &RegistrationPreflight,
    dependency: PlannedSignalEntityRef,
    dependent: PlannedSignalEntityRef,
) -> Result<RegistrationRouteDisposition, SignalRegistrationError> {
    let ordered_input_consumption = match dependent {
        PlannedSignalEntityRef::Pending { .. } => {
            plan.entities
                .iter()
                .find(|candidate| candidate.entity == dependent)
                .ok_or(SignalRegistrationError::EntityUnavailable)?
                .ordered_input_consumption
        }
        PlannedSignalEntityRef::Existing(dependent) => {
            let dependent_ref = world
                .get_entity(*dependent)
                .map_err(|_| SignalRegistrationError::EntityUnavailable)?;
            if !dependent_ref.contains::<SignalInputBuffer>() || !dependent_ref.contains::<RegistrarInactive>() {
                return Err(inconsistent_existing_runtime(dependent));
            }
            dependent_ref
                .get::<SystemRunner>()
                .ok_or_else(|| inconsistent_existing_runtime(dependent))?
                .ordered_input_consumption
        }
    };
    let disposition = preflight_dependency_replay(world, plan, dependency)?;
    match ordered_input_consumption {
        Some(OrderedInputConsumption::ManagerBridge) => {
            // A scalar replay value is one bootstrap command, not a revisioned command history.
            let bootstrap = match disposition {
                RegistrationRouteDisposition::ReplayReady { snapshot, .. } => Some(snapshot),
                RegistrationRouteDisposition::Ordinary | RegistrationRouteDisposition::ReplayPending => None,
                RegistrationRouteDisposition::OrderedInput { .. } => unreachable!("replay inspection is unordered"),
            };
            Ok(RegistrationRouteDisposition::OrderedInput { bootstrap })
        }
        Some(OrderedInputConsumption::Chain { .. }) => {
            if !matches!(disposition, RegistrationRouteDisposition::Ordinary) {
                return Err(SignalRegistrationError::ConflictingCanonicalState);
            }
            Ok(RegistrationRouteDisposition::OrderedInput { bootstrap: None })
        }
        None => Ok(disposition),
    }
}

fn preflight_dependency_replay(
    world: &World,
    plan: &RegistrationPreflight,
    dependency: PlannedSignalEntityRef,
) -> Result<RegistrationRouteDisposition, SignalRegistrationError> {
    match dependency {
        PlannedSignalEntityRef::Pending { .. } => {
            let dependency = plan
                .entities
                .iter()
                .find(|candidate| candidate.entity == dependency)
                .ok_or(SignalRegistrationError::EntityUnavailable)?;
            let Some(replay) = dependency.planned_replay.as_ref() else {
                if dependency.replayable {
                    return Err(SignalRegistrationError::ConflictingCanonicalState);
                }
                return Ok(RegistrationRouteDisposition::Ordinary);
            };
            if replay.key.cache_type() != replay.contract.cache_type()
                || replay.key.event_type() != replay.contract.event_type()
            {
                return Err(SignalRegistrationError::ConflictingCanonicalState);
            }
            replay_route_disposition(None, replay.key.event_type(), replay.inspection.snapshot())
                .map_err(|_| SignalRegistrationError::ConflictingCanonicalState)
        }
        PlannedSignalEntityRef::Existing(dependency) => {
            let Some(metadata) = world.get::<RegistrationReplayCacheState>(*dependency) else {
                return Ok(RegistrationRouteDisposition::Ordinary);
            };
            let inspection = metadata
                .contract
                .inspect(world, dependency)
                .ok_or_else(|| inconsistent_existing_runtime(dependency))?;
            replay_route_disposition(Some(dependency), metadata.contract.event_type(), inspection)
        }
    }
}

fn projected_execution_dependencies(
    world: &World,
    plan: &RegistrationPreflight,
    reference: PlannedSignalEntityRef,
) -> Result<Vec<PlannedSignalEntityRef>, SignalRegistrationError> {
    let mut dependencies = match reference {
        PlannedSignalEntityRef::Pending { .. } => plan
            .entities
            .iter()
            .find(|entity| entity.entity == reference)
            .map(|entity| entity.dependencies.to_vec())
            .ok_or(SignalRegistrationError::EntityUnavailable)?,
        PlannedSignalEntityRef::Existing(entity) => {
            if world.get::<StaticSignalDependencies>(*entity).is_none() {
                return Err(SignalRegistrationError::EntityUnavailable);
            }
            get_execution_dependencies(world, entity)
                .into_iter()
                .map(PlannedSignalEntityRef::Existing)
                .collect()
        }
    };
    dependencies.extend(
        plan.dynamic_connections
            .iter()
            .filter(|connection| connection.target == reference)
            .map(|connection| connection.source),
    );
    dependencies.sort_unstable();
    dependencies.dedup();
    Ok(dependencies)
}

fn check_level_within_limit(level: u32, max_level: u32) -> Result<u32, SignalRegistrationError> {
    if level > max_level {
        return Err(SignalRegistrationError::GraphDepthOverflow);
    }
    let index = usize::try_from(level).map_err(|_| SignalRegistrationError::GraphDepthOverflow)?;
    index
        .checked_add(1)
        .ok_or(SignalRegistrationError::GraphDepthOverflow)?;
    Ok(level)
}

fn collect_dynamic_connection_downstream_closure(
    world: &World,
    plan: &RegistrationPreflight,
    planned_forward_edges: &BTreeSet<(PlannedSignalEntityRef, PlannedSignalEntityRef)>,
) -> BTreeSet<PlannedSignalEntityRef> {
    if plan.dynamic_connections.is_empty() {
        return BTreeSet::new();
    }
    let mut forward_map = BTreeMap::<PlannedSignalEntityRef, Vec<PlannedSignalEntityRef>>::new();
    for &(dependency, dependent) in planned_forward_edges {
        forward_map.entry(dependency).or_default().push(dependent);
    }
    for connection in plan.dynamic_connections.iter().copied() {
        forward_map
            .entry(connection.source)
            .or_default()
            .push(connection.target);
    }
    let incremented = plan
        .claim_deltas
        .iter()
        .filter(|delta| delta.increment != 0)
        .map(|delta| delta.entity)
        .collect::<BTreeSet<_>>();
    let mut affected = BTreeSet::new();
    let mut stack = plan
        .dynamic_connections
        .iter()
        .flat_map(|connection| [connection.source, connection.target])
        .collect::<Vec<_>>();
    while let Some(entity) = stack.pop() {
        if !affected.insert(entity) {
            continue;
        }
        if let Some(dependents) = forward_map.get(&entity) {
            stack.extend(dependents.iter().copied());
        }
        if let PlannedSignalEntityRef::Existing(existing) = entity {
            for dependent in get_execution_dependents(world, existing) {
                let reference = PlannedSignalEntityRef::Existing(dependent);
                let active_existing = world
                    .get::<SignalClaimCount>(*dependent)
                    .is_some_and(|count| **count != 0)
                    && world.get::<RegistrarInactive>(*dependent).is_none();
                if active_existing || incremented.contains(&reference) {
                    stack.push(reference);
                }
            }
        }
    }
    affected
}

fn compute_projected_execution_levels_with_limit(
    world: &World,
    plan: &RegistrationPreflight,
    seeds: impl IntoIterator<Item = PlannedSignalEntityRef>,
    max_level: u32,
) -> Result<BTreeMap<PlannedSignalEntityRef, u32>, SignalRegistrationError> {
    let mut needed = BTreeSet::new();
    let mut stack = seeds.into_iter().collect::<Vec<_>>();
    while let Some(entity) = stack.pop() {
        if !needed.insert(entity) {
            continue;
        }
        stack.extend(projected_execution_dependencies(world, plan, entity)?);
    }
    let mut in_degrees = BTreeMap::<PlannedSignalEntityRef, usize>::new();
    let mut upstreams_by_entity = BTreeMap::<PlannedSignalEntityRef, Vec<PlannedSignalEntityRef>>::new();
    let mut dependents_by_entity = BTreeMap::<PlannedSignalEntityRef, Vec<PlannedSignalEntityRef>>::new();
    for entity in needed.iter().copied() {
        let dependencies = projected_execution_dependencies(world, plan, entity)?;
        let mut in_degree = 0usize;
        for &dependency in &dependencies {
            if needed.contains(&dependency) {
                in_degree += 1;
                dependents_by_entity.entry(dependency).or_default().push(entity);
            }
        }
        in_degrees.insert(entity, in_degree);
        upstreams_by_entity.insert(entity, dependencies);
    }
    let mut ready_queue = in_degrees
        .iter()
        .filter_map(|(&entity, &degree)| (degree == 0).then_some(entity))
        .collect::<VecDeque<_>>();
    let mut projected_levels = BTreeMap::<PlannedSignalEntityRef, u32>::new();
    while let Some(entity) = ready_queue.pop_front() {
        let level = match upstreams_by_entity
            .get(&entity)
            .into_iter()
            .flatten()
            .filter_map(|dependency| projected_levels.get(dependency).copied())
            .max()
        {
            Some(level) => {
                let next = level
                    .checked_add(1)
                    .ok_or(SignalRegistrationError::GraphDepthOverflow)?;
                check_level_within_limit(next, max_level)?
            }
            None => check_level_within_limit(0, max_level)?,
        };
        projected_levels.insert(entity, level);
        if let Some(dependents) = dependents_by_entity.get(&entity) {
            for &dependent in dependents {
                let degree = in_degrees
                    .get_mut(&dependent)
                    .expect("needed dependent must have an in-degree entry");
                *degree -= 1;
                if *degree == 0 {
                    ready_queue.push_back(dependent);
                }
            }
        }
    }
    if projected_levels.len() != needed.len() {
        return Err(SignalRegistrationError::CycleDetected);
    }
    Ok(projected_levels)
}

fn compute_projected_execution_levels(
    world: &World,
    plan: &RegistrationPreflight,
    seeds: impl IntoIterator<Item = PlannedSignalEntityRef>,
) -> Result<BTreeMap<PlannedSignalEntityRef, u32>, SignalRegistrationError> {
    compute_projected_execution_levels_with_limit(world, plan, seeds, u32::MAX)
}

fn preflight_dynamic_connections(world: &World, plan: &RegistrationPreflight) -> Result<(), SignalRegistrationError> {
    preflight_dynamic_connections_with_limit(world, plan, u32::MAX)
}

fn preflight_dynamic_connections_with_limit(
    world: &World,
    plan: &RegistrationPreflight,
    max_level: u32,
) -> Result<(), SignalRegistrationError> {
    let entity_types = |reference: PlannedSignalEntityRef| -> Result<(TypeId, TypeId), SignalRegistrationError> {
        match reference {
            PlannedSignalEntityRef::Pending { .. } => plan
                .entities
                .iter()
                .find(|entity| entity.entity == reference)
                .map(|entity| (entity.input_type, entity.output_type))
                .ok_or(SignalRegistrationError::EntityUnavailable),
            PlannedSignalEntityRef::Existing(entity) => {
                let runner = world
                    .get::<SystemRunner>(*entity)
                    .ok_or(SignalRegistrationError::EntityUnavailable)?;
                Ok((runner.input_type, runner.output_type))
            }
        }
    };
    let is_projected_active = |reference: PlannedSignalEntityRef| -> bool {
        let incremented = plan
            .claim_deltas
            .iter()
            .any(|delta| delta.entity == reference && delta.increment != 0);
        incremented
            || match reference {
                PlannedSignalEntityRef::Pending { .. } => false,
                PlannedSignalEntityRef::Existing(entity) => {
                    world.get::<SignalClaimCount>(*entity).is_some_and(|count| **count != 0)
                        && world.get::<RegistrarInactive>(*entity).is_none()
                }
            }
    };

    let mut pairs = BTreeSet::new();
    for connection in plan.dynamic_connections.iter().copied() {
        if connection.source == connection.target || !pairs.insert((connection.source, connection.target)) {
            return Err(SignalRegistrationError::DynamicRouteConflict);
        }
        let (target_input, _) = entity_types(connection.target)?;
        let (_, source_output) = entity_types(connection.source)?;
        if source_output != TypeId::of::<()>() || target_input != TypeId::of::<()>() {
            return Err(SignalRegistrationError::DynamicRouteConflict);
        }
        if !is_projected_active(connection.source) || !is_projected_active(connection.target) {
            return Err(SignalRegistrationError::EntityUnavailable);
        }
        match (connection.source, connection.target) {
            (PlannedSignalEntityRef::Existing(source), PlannedSignalEntityRef::Existing(target)) => {
                if world
                    .get::<StaticSignalDependencies>(*target)
                    .is_some_and(|dependencies| dependencies.0.contains(&source))
                    || world.get::<DynamicSignalOwner>(*target).is_some_and(|owner| {
                        owner
                            .subscriptions()
                            .iter()
                            .any(|subscription| subscription.source == source)
                    })
                {
                    return Err(SignalRegistrationError::DynamicRouteConflict);
                }
            }
            (source, PlannedSignalEntityRef::Pending { .. })
                if plan
                    .entities
                    .iter()
                    .find(|entity| entity.entity == connection.target)
                    .is_some_and(|target| target.dependencies.contains(&source)) =>
            {
                return Err(SignalRegistrationError::DynamicRouteConflict);
            }
            _ => {}
        }
    }

    if !plan.dynamic_connections.is_empty() {
        let mut planned_forward_edges = BTreeSet::new();
        for entity in plan.entities.iter().filter(|entity| is_projected_active(entity.entity)) {
            for &dependency in &entity.dependencies {
                planned_forward_edges.insert((dependency, entity.entity));
            }
        }
        let affected = collect_dynamic_connection_downstream_closure(world, plan, &planned_forward_edges);
        compute_projected_execution_levels_with_limit(world, plan, affected, max_level)?;
    }
    Ok(())
}

fn preflight_output_fanout(
    world: &World,
    plan: &RegistrationPreflight,
    relationship_pairs: &BTreeSet<(PlannedSignalEntityRef, PlannedSignalEntityRef)>,
) -> Result<(), SignalRegistrationError> {
    let mut distinct_dependents = BTreeMap::<PlannedSignalEntityRef, BTreeSet<PlannedSignalEntityRef>>::new();
    for &(dependency, dependent) in relationship_pairs {
        distinct_dependents.entry(dependency).or_default().insert(dependent);
    }
    for connection in plan.dynamic_connections.iter().copied() {
        distinct_dependents
            .entry(connection.source)
            .or_default()
            .insert(connection.target);
    }
    for (source, dependents) in &mut distinct_dependents {
        if let PlannedSignalEntityRef::Existing(existing_source) = *source {
            if let Some(active) = world.get::<ActiveSignalDependents>(*existing_source) {
                dependents.extend(active.0.iter().copied().map(PlannedSignalEntityRef::Existing));
            }
            if let Some(replay_relationships) = world.get::<ReplayConsumingRelationships>(*existing_source) {
                dependents.extend(
                    replay_relationships
                        .0
                        .keys()
                        .copied()
                        .map(PlannedSignalEntityRef::Existing),
                );
            }
        }
        if dependents.len() <= 1 {
            continue;
        }
        match *source {
            PlannedSignalEntityRef::Pending { .. } => {
                let candidate = plan
                    .entities
                    .iter()
                    .find(|entity| entity.entity == *source)
                    .ok_or(SignalRegistrationError::EntityUnavailable)?;
                let delivery = candidate
                    .pending_output_delivery
                    .ok_or(SignalRegistrationError::ConflictingCanonicalState)?;
                if !delivery.is_coherent(candidate.output_type) {
                    return Err(SignalRegistrationError::ConflictingCanonicalState);
                }
                if delivery.kind() != OutputDeliveryKind::Broadcast {
                    return Err(SignalRegistrationError::MoveOnlyFanout);
                }
            }
            PlannedSignalEntityRef::Existing(existing_source) => {
                let Some(delivery) = world.get::<StoredOutputDelivery>(*existing_source).copied() else {
                    return Err(inconsistent_existing_runtime(existing_source));
                };
                if delivery.kind() != OutputDeliveryKind::Broadcast {
                    validate_existing_registration_runtime_memoized(world, existing_source, &mut BTreeMap::new(), &[])?;
                    return Err(SignalRegistrationError::MoveOnlyFanout);
                }
            }
        }
    }
    Ok(())
}

impl SignalRegistrationGraphHooks for World {
    fn registration_world(&mut self) -> &mut World {
        self
    }

    fn registration_existing_signal(&self, entity: SignalEntity) -> Option<ExistingSignalEntityRegistration> {
        let entity_ref = self.get_entity(*entity).ok()?;
        let runner = entity_ref.get::<SystemRunner>();
        let dependencies = entity_ref.get::<StaticSignalDependencies>()?;
        Some(ExistingSignalEntityRegistration {
            dependencies: dependencies.0.clone(),
            replayable: entity_ref.contains::<RegistrationReplayCacheState>(),
            ordered_input_consumption: runner.and_then(|runner| runner.ordered_input_consumption),
            input_type: runner.map(|runner| runner.input_type).unwrap_or(TypeId::of::<()>()),
            output_type: runner.map(|runner| runner.output_type).unwrap_or(TypeId::of::<()>()),
            output_delivery: entity_ref.get::<StoredOutputDelivery>().map(|delivery| delivery.key()),
        })
    }

    fn registration_preflight_existing_runtimes(
        &self,
        entities: &[SignalEntity],
    ) -> Result<(), SignalRegistrationError> {
        let mut validated_levels = BTreeMap::new();
        for &entity in entities {
            validate_existing_registration_runtime_memoized(self, entity, &mut validated_levels, &[])?;
        }
        Ok(())
    }

    fn registration_schedule_is_registered(&self, schedule: InternedScheduleLabel) -> bool {
        self.get_resource::<SignalGraphState>()
            .is_some_and(|state| state.registered_schedules.contains(&schedule))
    }

    fn registration_existing_schedule(
        &self,
        entity: SignalEntity,
    ) -> Result<(Option<InternedScheduleLabel>, Option<InternedScheduleLabel>), SignalRegistrationError> {
        let entity = self
            .get_entity(*entity)
            .map_err(|_| SignalRegistrationError::EntityUnavailable)?;
        Ok((
            entity.get::<ScheduleTag>().map(|tag| tag.0),
            entity.get::<ScheduleHint>().map(|hint| hint.0),
        ))
    }

    fn registration_claim_count(&self, entity: SignalEntity) -> Option<usize> {
        self.get::<SignalClaimCount>(*entity).map(|count| **count)
    }

    fn registration_replay_maintenance_armed(&self, entity: SignalEntity) -> Option<bool> {
        self.get_entity(*entity)
            .ok()
            .map(|entity| entity.contains::<ReplayMaintenanceClaim>())
    }

    fn registration_preflight_canonical_state(
        &self,
        entity: &RegistrationEntityPreflight,
    ) -> Result<(), SignalRegistrationError> {
        if entity.ordered_input_consumption.is_some_and(|consumption| {
            !consumption.accepts_shape(
                entity.dependencies.len(),
                entity.input_type,
                entity.output_type,
                entity.replayable,
            )
        }) {
            return Err(SignalRegistrationError::ConflictingCanonicalState);
        }
        let PlannedSignalEntityRef::Existing(existing_entity) = entity.entity else {
            return Ok(());
        };
        let existing = self
            .registration_existing_signal(existing_entity)
            .ok_or(SignalRegistrationError::EntityUnavailable)?;
        let mut proposed = Vec::with_capacity(entity.dependencies.len());
        for dependency in entity.dependencies.iter().copied() {
            let PlannedSignalEntityRef::Existing(dependency) = dependency else {
                return Err(SignalRegistrationError::ConflictingCanonicalState);
            };
            proposed.push(dependency);
        }
        proposed.sort_by(signal_identity_cmp);
        proposed.dedup();
        if proposed.as_slice() != existing.dependencies.as_ref()
            || entity.replayable != existing.replayable
            || entity.ordered_input_consumption != existing.ordered_input_consumption
        {
            return Err(SignalRegistrationError::ConflictingCanonicalState);
        }
        Ok(())
    }

    fn registration_preflight_reserved_output_policy(
        &self,
        entity: &RegistrationEntityPreflight,
    ) -> Result<(), SignalRegistrationError> {
        if let Some(pending_delivery) = entity.pending_output_delivery
            && !pending_delivery.is_coherent(entity.output_type)
        {
            return Err(SignalRegistrationError::ConflictingCanonicalState);
        }
        if let PlannedSignalEntityRef::Existing(existing_entity) = entity.entity {
            let Some(runner) = self.get::<SystemRunner>(*existing_entity) else {
                return Err(inconsistent_existing_runtime(existing_entity));
            };
            let Some(stored) = self.get::<StoredOutputDelivery>(*existing_entity).copied() else {
                return Err(inconsistent_existing_runtime(existing_entity));
            };
            if !stored.is_coherent(runner.output_type)
                || !runner.output_delivery.is_coherent(runner.output_type)
                || stored.key() != runner.output_delivery.key()
            {
                return Err(inconsistent_existing_runtime(existing_entity));
            }
            if let Some(expected_key) = entity.expected_output_delivery
                && stored.key() != expected_key
            {
                return Err(SignalRegistrationError::ConflictingCanonicalState);
            }
        }
        Ok(())
    }

    fn registration_preflight_dynamic_connections(
        &self,
        plan: &RegistrationPreflight,
    ) -> Result<(), SignalRegistrationError> {
        preflight_dynamic_connections(self, plan)
    }

    fn registration_preflight_activation(
        &self,
        plan: &RegistrationPreflight,
    ) -> Result<RegistrationActivationPreflight, SignalRegistrationError> {
        let mut projected_counts = BTreeMap::<PlannedSignalEntityRef, usize>::new();
        let mut count_increments = BTreeMap::<PlannedSignalEntityRef, usize>::new();
        let mut relationship_pairs = BTreeSet::new();
        let mut activated = BTreeSet::new();
        for delta in plan.claim_deltas.iter() {
            for _ in 0..delta.increment {
                let mut pending = vec![delta.entity];
                while let Some(entity) = pending.pop() {
                    let current = match projected_counts.get(&entity).copied() {
                        Some(current) => current,
                        None => match entity {
                            PlannedSignalEntityRef::Pending { .. } => 0,
                            PlannedSignalEntityRef::Existing(existing_entity) => self
                                .registration_claim_count(existing_entity)
                                .ok_or(SignalRegistrationError::EntityUnavailable)?,
                        },
                    };
                    let incremented = current
                        .checked_add(1)
                        .ok_or(SignalRegistrationError::ClaimCountOverflow)?;
                    projected_counts.insert(entity, incremented);
                    let total_increment = count_increments.get(&entity).copied().unwrap_or(0);
                    count_increments.insert(
                        entity,
                        total_increment
                            .checked_add(1)
                            .ok_or(SignalRegistrationError::ClaimCountOverflow)?,
                    );
                    if current != 0 {
                        continue;
                    }
                    activated.insert(entity);

                    let mut dependencies = match entity {
                        PlannedSignalEntityRef::Pending { .. } => plan
                            .entities
                            .iter()
                            .find(|candidate| candidate.entity == entity)
                            .map(|candidate| candidate.dependencies.to_vec())
                            .ok_or(SignalRegistrationError::EntityUnavailable)?,
                        PlannedSignalEntityRef::Existing(entity) => self
                            .registration_existing_signal(entity)
                            .map(|existing| {
                                existing
                                    .dependencies
                                    .iter()
                                    .copied()
                                    .map(PlannedSignalEntityRef::Existing)
                                    .collect()
                            })
                            .ok_or(SignalRegistrationError::EntityUnavailable)?,
                    };
                    dependencies.sort_unstable();
                    dependencies.dedup();
                    for dependency in dependencies.iter().copied() {
                        relationship_pairs.insert((dependency, entity));
                    }
                    pending.extend(dependencies.into_iter().rev());
                }
            }
        }
        // Dynamic connections affect execution order across `owner` and its downstream subgraph,
        // without adding static ownership obligations.
        let dynamic_affected = collect_dynamic_connection_downstream_closure(self, plan, &relationship_pairs);
        let projected_levels =
            compute_projected_execution_levels(self, plan, count_increments.keys().copied().chain(dynamic_affected))?;
        preflight_output_fanout(self, plan, &relationship_pairs)?;
        let schedule_projection = plan
            .schedules
            .iter()
            .map(|schedule| (schedule.entity, *schedule))
            .collect::<BTreeMap<_, _>>();
        let default_schedule = self
            .get_resource::<SignalGraphState>()
            .map(|state| state.default_schedule)
            .ok_or(SignalRegistrationError::ScheduleNotRegistered)?;
        let mut ordered_activated = activated.into_iter().collect::<Vec<_>>();
        ordered_activated.sort_by_key(|entity| (projected_levels.get(entity).copied().unwrap_or(u32::MAX), *entity));
        let effective_membership = ordered_activated
            .iter()
            .copied()
            .map(|entity| {
                let schedule = schedule_projection
                    .get(&entity)
                    .and_then(|state| state.tag)
                    .or_else(|| match entity {
                        PlannedSignalEntityRef::Existing(entity) => self.get::<ScheduleTag>(*entity).map(|tag| tag.0),
                        PlannedSignalEntityRef::Pending { .. } => None,
                    })
                    .unwrap_or(default_schedule);
                let level = projected_levels
                    .get(&entity)
                    .copied()
                    .ok_or(SignalRegistrationError::EntityUnavailable)?;
                Ok(RegistrationEffectiveMembershipPreflight {
                    entity,
                    schedule,
                    level,
                })
            })
            .collect::<Result<Vec<_>, SignalRegistrationError>>()?;
        let mut ordered_count_increments = count_increments.into_iter().collect::<Vec<_>>();
        ordered_count_increments
            .sort_by_key(|(entity, _)| (projected_levels.get(entity).copied().unwrap_or(u32::MAX), *entity));
        let active_routes = relationship_pairs
            .into_iter()
            .map(|(dependency, dependent)| {
                Ok(RegistrationActiveRoutePreflight {
                    dependency,
                    dependent,
                    disposition: preflight_activation_route_disposition(self, plan, dependency, dependent)?,
                })
            })
            .collect::<Result<Vec<_>, SignalRegistrationError>>()?;
        Ok(RegistrationActivationPreflight {
            count_increments: ordered_count_increments
                .into_iter()
                .map(|(entity, increment)| crate::registration::RegistrationClaimDelta { entity, increment })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            active_routes: active_routes.into_boxed_slice(),
            effective_membership: effective_membership.into_boxed_slice(),
            activated: ordered_activated.into_boxed_slice(),
        })
    }

    fn registration_commit_signal_entity(&mut self, entity: RegistrationEntityCommit) {
        let RegistrationEntityCommit {
            entity,
            dependencies,
            runtime,
        } = entity;
        let input_type = runtime.input_type();
        let output_type = runtime.output_type();
        let output_delivery = runtime.output_delivery();
        assert!(
            output_delivery.is_coherent(output_type),
            "committed signal entity {entity:?} had incoherent output delivery policy"
        );
        let collection_poll_append = runtime.collection_poll_append();
        let ordered_input_consumption = runtime.ordered_input_consumption();
        let replay_key = runtime.replay_key();
        let replay_contract = runtime.replay_contract().cloned();
        let runner = SystemRunner {
            runner: runtime.runner().clone(),
            input_type,
            output_type,
            output_delivery,
            collection_poll_append,
            ordered_input_consumption,
            replay_key,
            replay_contract,
        };
        {
            let mut entity_mut = self.entity_mut(*entity);
            entity_mut.insert((
                SignalClaimCount::new(),
                SignalInputBuffer::default(),
                runner,
                output_delivery,
                StaticSignalDependencies(dependencies),
                MaterializedSignalDependents::default(),
                SignalTerminalRetainers::default(),
                ActiveSignalDependents::default(),
                RegistrarInactive,
            ));
        }
        if let Some(mut state) = self.get_resource_mut::<SignalGraphState>() {
            state.edge_change_seeds.insert(entity);
        }
        #[cfg(test)]
        if self.contains_resource::<RegistrationCommitStageTrace>() {
            assert!(self.get::<RegistrationReplayCacheState>(*entity).is_none());
            self.resource_mut::<RegistrationCommitStageTrace>()
                .0
                .push(RegistrationCommitStage::EntityPublishedWithoutReplay(entity));
        }
    }

    fn registration_commit_replay(&mut self, replay: RegistrationReplayCommit) {
        let RegistrationReplayCommit {
            entity,
            replay,
            installer,
        } = replay;
        if let Some(installer) = installer {
            installer(self, entity);
        }
        #[cfg(test)]
        let replayable = matches!(&replay, PlannedReplayState::Replayable { .. });
        match replay {
            PlannedReplayState::NotReplayable => {}
            PlannedReplayState::Replayable { key, contract, .. } => {
                let runner = self
                    .get::<SystemRunner>(*entity)
                    .unwrap_or_else(|| panic!("preflighted replay entity {entity:?} lost its runtime"));
                let runtime_key = runner
                    .replay_key
                    .unwrap_or_else(|| panic!("preflighted replay entity {entity:?} runtime lost its replay key"));
                let runtime_contract = runner
                    .replay_contract
                    .as_ref()
                    .unwrap_or_else(|| panic!("preflighted replay entity {entity:?} runtime lost its replay contract"));
                assert_eq!(
                    runtime_key, key,
                    "preflighted replay entity {entity:?} runtime key diverged"
                );
                assert!(
                    Arc::ptr_eq(runtime_contract, &contract),
                    "preflighted replay entity {entity:?} runtime and metadata contracts diverged"
                );
                assert_eq!(
                    runner.output_type,
                    key.event_type(),
                    "preflighted replay entity {entity:?} output and runtime-key event types diverged"
                );
                assert_eq!(
                    key.cache_type(),
                    contract.cache_type(),
                    "preflighted replay entity {entity:?} cache contract and runtime key diverged"
                );
                assert_eq!(
                    key.event_type(),
                    contract.event_type(),
                    "preflighted replay entity {entity:?} event contract and runtime key diverged"
                );
                let inspection = contract.inspect(self, entity).unwrap_or_else(|| {
                    panic!("preflighted replay entity {entity:?} contract could not inspect its installed cache")
                });
                if let ErasedReplaySnapshot::Ready { snapshot, .. } = &inspection {
                    assert_eq!(
                        erased_type_id(snapshot),
                        key.event_type(),
                        "preflighted replay entity {entity:?} produced the wrong snapshot event type"
                    );
                }
                self.entity_mut(*entity)
                    .insert(RegistrationReplayCacheState { contract });
            }
        }
        #[cfg(test)]
        if replayable && self.contains_resource::<RegistrationCommitStageTrace>() {
            assert!(self.get::<RegistrationReplayCacheState>(*entity).is_some());
            self.resource_mut::<RegistrationCommitStageTrace>()
                .0
                .push(RegistrationCommitStage::ReplayPublished(entity));
        }
    }

    fn registration_commit_materialized_dependency(&mut self, link: RegistrationMaterializedDependencyCommit) {
        let mut dependency = self
            .get_entity_mut(*link.dependency)
            .unwrap_or_else(|_| panic!("preflighted materialized dependency {:?} disappeared", link.dependency));
        if let Some(mut dependents) = dependency.get_mut::<MaterializedSignalDependents>() {
            dependents.0.insert(link.dependent);
        } else {
            dependency.insert(MaterializedSignalDependents(BTreeSet::from([link.dependent])));
        }
    }

    fn registration_commit_schedule(&mut self, schedule: RegistrationScheduleCommit) {
        {
            let mut entity = self.entity_mut(*schedule.entity);
            match schedule.tag {
                Some(tag) => {
                    entity.insert(ScheduleTag(tag));
                }
                None => {
                    entity.remove::<ScheduleTag>();
                }
            }
            match schedule.hint {
                Some(hint) => {
                    entity.insert(ScheduleHint(hint));
                }
                None => {
                    entity.remove::<ScheduleHint>();
                }
            }
        }
        reconcile_cached_schedule_membership(self, schedule.entity);
    }

    fn registration_commit_activation(&mut self, activation: RegistrationActivationCommit) {
        for claim in activation.count_increments {
            for _ in 0..claim.increment {
                increment_signal_claim_count(self, claim.entity);
                #[cfg(test)]
                trace_ownership_stage(self, OwnershipWorklistStage::Acquired(claim.entity));
            }
        }
        for RegistrationActiveRouteCommit {
            dependency,
            dependent,
            disposition,
        } in activation.active_routes
        {
            let (relationship_state, install_active_route) = match disposition {
                RegistrationRouteDisposition::OrderedInput { bootstrap } => {
                    if let Some(bootstrap) = bootstrap {
                        assert!(
                            push_signal_input_unchecked(self, dependent, bootstrap),
                            "preflighted ordered-input consumer {dependent:?} lost its input buffer during activation"
                        );
                    }
                    (None, true)
                }
                RegistrationRouteDisposition::Ordinary => {
                    #[cfg(test)]
                    trace_ownership_stage(self, OwnershipWorklistStage::OrdinaryRoute(dependency, dependent));
                    (None, true)
                }
                RegistrationRouteDisposition::ReplayPending => {
                    #[cfg(test)]
                    trace_ownership_stage(self, OwnershipWorklistStage::ReplayPendingRoute(dependency, dependent));
                    (Some(ReplayRelationshipState::Pending), false)
                }
                RegistrationRouteDisposition::ReplayReady {
                    snapshot,
                    revision,
                    event_type,
                } => {
                    #[cfg(test)]
                    trace_ownership_stage(
                        self,
                        OwnershipWorklistStage::ReplayReadyRoute(dependency, dependent, revision),
                    );
                    assert_eq!(
                        erased_type_id(&snapshot),
                        event_type,
                        "preflighted replay dependency {dependency:?} snapshot changed event type"
                    );
                    assert!(
                        push_replay_snapshot_input_unchecked(self, dependency, dependent, snapshot),
                        "preflighted replay dependent {dependent:?} lost its input buffer during activation"
                    );
                    (Some(ReplayRelationshipState::Published { cursor: revision }), true)
                }
            };
            let mut dependency_entity = self
                .get_entity_mut(*dependency)
                .unwrap_or_else(|_| panic!("preflighted dependency {dependency:?} disappeared during activation"));
            if let Some(relationship_state) = relationship_state {
                if let Some(mut relationships) = dependency_entity.get_mut::<ReplayConsumingRelationships>() {
                    relationships.0.insert(dependent, relationship_state);
                } else {
                    dependency_entity.insert(ReplayConsumingRelationships(BTreeMap::from([(
                        dependent,
                        relationship_state,
                    )])));
                }
            }
            if install_active_route {
                if let Some(mut dependents) = dependency_entity.get_mut::<ActiveSignalDependents>() {
                    dependents.0.insert(dependent);
                } else {
                    dependency_entity.insert(ActiveSignalDependents(HashSet::from([dependent])));
                }
            }
        }
        for membership in activation.effective_membership {
            commit_cached_schedule_membership(self, membership.entity, membership.schedule, membership.level);
        }
        for entity in activation.activated.iter().copied() {
            if let Some(mut state) = self.get_resource_mut::<SignalGraphState>() {
                state.edge_change_seeds.insert(entity);
            }
        }
        for entity in activation.activated {
            let mut entity_mut = self.entity_mut(*entity);
            entity_mut.remove::<RegistrarInactive>();
            if let Some(mut owner) = entity_mut.get_mut::<DynamicSignalOwner>() {
                owner.mark_activated();
            }
        }
    }

    fn registration_commit_dynamic_connection(
        &mut self,
        connection: RegistrationDynamicConnectionCommit,
    ) -> DynamicSignalConnection {
        let RegistrationDynamicConnectionCommit { source, target } = connection;
        assert_ne!(source, target, "preflighted dynamic connection became a self-route");
        let mut source_entity = self
            .get_entity_mut(*source)
            .unwrap_or_else(|_| panic!("preflighted dynamic source {source:?} disappeared"));
        let inserted = source_entity
            .get_mut::<ActiveSignalDependents>()
            .unwrap_or_else(|| panic!("preflighted dynamic source {source:?} lost its active-route component"))
            .0
            .insert(target);
        assert!(
            inserted,
            "preflighted dynamic route {source:?} -> {target:?} already existed"
        );
        if let Some(mut state) = self.get_resource_mut::<SignalGraphState>() {
            state.edge_change_seeds.insert(target);
        }
        DynamicSignalConnection::new(source, target)
    }

    fn registration_commit_replay_maintenance(&mut self, terminal: SignalEntity) {
        let mut entity = self
            .get_entity_mut(*terminal)
            .unwrap_or_else(|_| panic!("preflighted replay terminal {terminal:?} disappeared"));
        assert!(
            !entity.contains::<ReplayMaintenanceClaim>(),
            "preflighted replay terminal {terminal:?} already had a maintenance claim"
        );
        entity.insert(ReplayMaintenanceClaim(SignalClaim::new(terminal)));
    }

    fn registration_commit_terminal_retainer(&mut self, retainer: RegistrationTerminalRetainerCommit) {
        let mut entity = self
            .get_entity_mut(*retainer.terminal)
            .unwrap_or_else(|_| panic!("preflighted terminal {:?} disappeared", retainer.terminal));
        if let Some(mut retainers) = entity.get_mut::<SignalTerminalRetainers>() {
            retainers.0.push(retainer.retainer);
        } else {
            entity.insert(SignalTerminalRetainers(vec![retainer.retainer]));
        }
    }

    fn registration_publish_initial_claim(&mut self, terminal: SignalEntity) -> SignalClaim {
        SignalClaim::new(terminal)
    }
}

#[cfg(test)]
pub(crate) fn materialize_materializer<Kind: SignalKind>(
    world: &mut World,
    materializer: SignalMaterializer<Kind>,
) -> Result<SignalEntity, SignalRegistrationError> {
    materializer.materialize_with_graph(world)
}

#[cfg(test)]
fn verify_existing_canonical_dependencies(
    world: &mut World,
    entity: SignalEntity,
    dependencies: impl IntoIterator<Item = SignalEntity>,
) -> Result<(), SignalRegistrationError> {
    let existing = world
        .registration_existing_signal(entity)
        .ok_or(SignalRegistrationError::EntityUnavailable)?;
    let mut dependencies = dependencies.into_iter().collect::<Vec<_>>();
    dependencies.sort_by(signal_identity_cmp);
    dependencies.dedup();
    verify_existing_canonical_with_graph(
        world,
        entity,
        dependencies.into_boxed_slice(),
        existing.replayable,
        existing.output_delivery,
    )
}

#[cfg(test)]
fn verify_existing_canonical_output_delivery(
    world: &mut World,
    entity: SignalEntity,
    output_delivery: OutputDeliveryKey,
) -> Result<(), SignalRegistrationError> {
    let existing = world
        .registration_existing_signal(entity)
        .ok_or(SignalRegistrationError::EntityUnavailable)?;
    verify_existing_canonical_with_graph(
        world,
        entity,
        existing.dependencies,
        existing.replayable,
        Some(output_delivery),
    )
}

/// Non-executing readiness and current-state observation of a framework-verified replay cache.
///
/// Unlike opt-in imperative subgraph evaluation, inspecting a replay cache is purely read-only:
/// it never executes Bevy systems, advances revisions, mutates `Local` state or resources, or
/// publishes dynamic generations.
#[cfg_attr(
    not(feature = "imperative-signal-evaluation"),
    doc = concat!(
        "\n\nImperative subgraph evaluation is unavailable under default features unless the ",
        "non-default `imperative-signal-evaluation` feature is enabled:\n\n",
        "```compile_fail,E0432\n",
        "use jonmo::graph::run_signal_subgraph_once;\n",
        "```\n",
    )
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayCacheInspection<T> {
    /// Authoritative state has not been established yet.
    Warming,
    /// Authoritative current state and terminal-local revision.
    Ready {
        /// Cloned current logical state.
        current: T,
        /// Monotonic terminal-local revision.
        revision: u64,
    },
}

fn verify_coherent_replay_entity<Cache: 'static, Event: 'static>(world: &World, signal: SignalEntity) -> Option<()> {
    let entity_ref = world.get_entity(*signal).ok()?;
    let runner = entity_ref.get::<SystemRunner>()?;
    let metadata = entity_ref.get::<RegistrationReplayCacheState>()?;
    let expected_key = ReplayRuntimeKey::new::<Cache, Event>();
    let runtime_contract = runner.replay_contract.as_ref()?;
    (runner.replay_key == Some(expected_key)
        && runner.output_type == expected_key.event_type()
        && Arc::ptr_eq(runtime_contract, &metadata.contract)
        && metadata.contract.cache_type() == expected_key.cache_type()
        && metadata.contract.event_type() == expected_key.event_type())
    .then_some(())
}

/// Inspects a scalar replay cache on `signal` without executing systems or mutating graph state.
///
/// Returns `None` if `signal` is unavailable, is not a framework-managed scalar replay terminal,
/// or stores a different value type.
pub fn inspect_scalar_replay_cache<T: Clone + Send + Sync + 'static>(
    world: &World,
    signal: SignalEntity,
) -> Option<ReplayCacheInspection<T>> {
    verify_coherent_replay_entity::<ScalarReplayCache<T>, T>(world, signal)?;
    match world.get::<ScalarReplayCache<T>>(*signal)?.snapshot() {
        ReplayCacheSnapshot::Warming => Some(ReplayCacheInspection::Warming),
        ReplayCacheSnapshot::Ready { current, revision } => Some(ReplayCacheInspection::Ready {
            current: current.clone(),
            revision: revision.get(),
        }),
    }
}

/// Inspects a vector replay cache on `signal` without executing systems or mutating graph state.
///
/// Returns `None` if `signal` is unavailable, is not a framework-managed vector replay terminal,
/// or stores a different item type.
pub fn inspect_vec_replay_cache<T: Clone + Send + Sync + 'static>(
    world: &World,
    signal: SignalEntity,
) -> Option<ReplayCacheInspection<Vec<T>>> {
    verify_coherent_replay_entity::<VecReplayCache<T>, Vec<crate::signal_vec::VecDiff<T>>>(world, signal)?;
    match world.get::<VecReplayCache<T>>(*signal)?.snapshot() {
        ReplayCacheSnapshot::Warming => Some(ReplayCacheInspection::Warming),
        ReplayCacheSnapshot::Ready { current, revision } => Some(ReplayCacheInspection::Ready {
            current: current.clone(),
            revision: revision.get(),
        }),
    }
}

/// Inspects a map replay cache on `signal` without executing systems or mutating graph state.
///
/// Returns `None` if `signal` is unavailable, is not a framework-managed map replay terminal,
/// or stores different key/value types.
pub fn inspect_map_replay_cache<K, V>(
    world: &World,
    signal: SignalEntity,
) -> Option<ReplayCacheInspection<BTreeMap<K, V>>>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    verify_coherent_replay_entity::<MapReplayCache<K, V>, Vec<crate::signal_map::MapDiff<K, V>>>(world, signal)?;
    match world.get::<MapReplayCache<K, V>>(*signal)?.snapshot() {
        ReplayCacheSnapshot::Warming => Some(ReplayCacheInspection::Warming),
        ReplayCacheSnapshot::Ready { current, revision } => Some(ReplayCacheInspection::Ready {
            current: current.clone(),
            revision: revision.get(),
        }),
    }
}

#[cfg(any(test, feature = "imperative-signal-evaluation"))]
fn run_signal_subgraph_once_one_shot(In(target): In<SignalEntity>, world: &mut World) -> Option<ErasedSignalValue> {
    if !signal_is_runnable(world, target) {
        return None;
    }
    if has_live_downstream_consumer(world, target) {
        ensure_imperative_observation_can_share_output(world, target);
    }

    let (recursion_limit, limit_behavior) = world
        .get_resource::<SignalGraphState>()
        .map(|state| (state.registration_recursion_limit, state.on_recursion_limit_exceeded))
        .unwrap_or((DEFAULT_REGISTRATION_RECURSION_LIMIT, RecursionLimitBehavior::default()));

    let mut stepped: HashSet<SignalEntity> = HashSet::default();
    let mut target_output: Option<ErasedSignalValue> = None;
    let mut pass = 0usize;

    loop {
        if !signal_is_runnable(world, target) {
            break;
        }

        // Collect all runnable signals reachable upstream from `target`, including active
        // dynamic subscriptions (`adapter -> owner`), not only static dependencies.
        let mut reachable: HashSet<SignalEntity> = HashSet::default();
        let mut queue: VecDeque<SignalEntity> = VecDeque::new();
        queue.push_back(target);
        reachable.insert(target);

        while let Some(signal) = queue.pop_front() {
            for upstream in get_execution_dependencies(world, signal) {
                if signal_is_runnable(world, upstream) && reachable.insert(upstream) {
                    queue.push_back(upstream);
                }
            }
        }

        let any_pending = reachable.iter().any(|&signal| {
            !stepped.contains(&signal)
                || world
                    .get::<SignalInputBuffer>(*signal)
                    .is_some_and(|buffer| !buffer.0.is_empty())
        });
        if !any_pending {
            break;
        }

        if pass > recursion_limit {
            match limit_behavior {
                RecursionLimitBehavior::Panic => {
                    panic!(
                        "Signal registration recursion limit exceeded ({} passes) during one-shot evaluation of {:?}. \
                             This usually indicates an infinite loop where signals keep spawning new signals. \
                             Processed {} signals before limit was reached. \
                             Use `JonmoPlugin::on_recursion_limit_exceeded` to change this behavior.",
                        recursion_limit,
                        target,
                        stepped.len()
                    );
                }
                RecursionLimitBehavior::Warn => {
                    #[cfg(feature = "tracing")]
                    bevy_log::warn!(
                        "Signal registration recursion limit exceeded ({} passes) during one-shot evaluation of {:?}. \
                             This may indicate an infinite loop where signals keep spawning new signals. \
                             Processed {} signals so far.",
                        recursion_limit,
                        target,
                        stepped.len()
                    );
                    break;
                }
                RecursionLimitBehavior::Silent => {
                    break;
                }
            }
        }
        pass += 1;

        let level_result = compute_signal_levels(world, &reachable, |u| reachable.contains(&u), |_| None);
        let by_level = bucket_levels_sorted(&level_result.levels);

        let mut topology_changed = false;
        'levels: for level in by_level {
            for signal in level {
                // If an earlier node in this pass attached a new dynamic dependency upstream of
                // `signal`, recompute levels so the new upstream subgraph executes before `signal`.
                if get_execution_dependencies(world, signal)
                    .into_iter()
                    .any(|upstream| signal_is_runnable(world, upstream) && !reachable.contains(&upstream))
                {
                    topology_changed = true;
                    break 'levels;
                }

                let has_pending_input = world
                    .get::<SignalInputBuffer>(*signal)
                    .is_some_and(|buffer| !buffer.0.is_empty());
                if stepped.insert(signal) || has_pending_input {
                    if signal == target {
                        let mut capture = |runner: &SystemRunner, output: ErasedSignalValue| {
                            if let Some(history) = target_output.as_mut()
                                && let Some(append) = runner.collection_poll_append
                            {
                                append(history, output);
                            } else {
                                target_output = Some(output);
                            }
                        };
                        execute_signal_node(world, signal, Some(&mut capture));
                    } else {
                        execute_signal_node(world, signal, None);
                    }
                }
            }
        }

        if !topology_changed && stepped.contains(&target) {
            let target_has_input = world
                .get::<SignalInputBuffer>(*target)
                .is_some_and(|buffer| !buffer.0.is_empty());
            if !target_has_input {
                break;
            }
        }
    }

    crate::dynamic::flush_all_after_output_cleanup(world);
    target_output
}

#[cfg(any(test, feature = "imperative-signal-evaluation"))]
#[track_caller]
pub(crate) fn run_signal_subgraph_once_internal(world: &mut World, signal: SignalEntity) -> Option<ErasedSignalValue> {
    if signal_is_runnable(world, signal) && has_live_downstream_consumer(world, signal) {
        ensure_imperative_observation_can_share_output(world, signal);
    }
    world
        .run_system_cached_with(run_signal_subgraph_once_one_shot, signal)
        .ok()
        .flatten()
}

#[cfg(any(test, feature = "imperative-signal-evaluation"))]
pub(crate) fn downcast_erased_signal_value<T: 'static>(value: ErasedSignalValue) -> Option<T> {
    value.downcast::<T>().map(|value| *value).ok()
}

/// Owned type-erased value returned by [`run_signal_subgraph_once`].
#[cfg(feature = "imperative-signal-evaluation")]
pub type BoxedSignalValue = Box<dyn Any + Send + Sync>;

/// Imperatively executes `signal` and its reachable upstream subgraph once outside their assigned
/// Bevy schedules.
///
/// # State Mutation Warning
///
/// This function is **not** a read-only current-value query (use [`inspect_scalar_replay_cache`],
/// [`inspect_vec_replay_cache`], or [`inspect_map_replay_cache`] for non-executing state
/// inspection). Calling `run_signal_subgraph_once`:
///
/// - Immediately executes reachable signal systems outside their configured Bevy schedules.
/// - Consumes queued inputs and forwards accepted outputs to downstream buffers.
/// - Mutates system `Local` state, `once` / `dedupe` state, world resources, replay cache
///   readiness/state/revisions, and dynamic slots, cohorts, and generations.
/// - Can alter subsequent scheduled execution and future application behavior.
///
/// Scalars return the last accepted output from `signal`'s drain. Signal vectors and maps return
/// all accepted diffs concatenated in order for this return value while preserving separate batch
/// boundaries for downstream consumers.
///
/// # Panics
///
/// Panics before executing reachable systems if `signal` is runnable, has move-only
/// [`OutputDelivery`](crate::OutputDelivery), and also has an active or pending downstream consumer
/// (ordinary, ordered, or replay). Returning [`BoxedSignalValue`] while forwarding downstream
/// requires broadcast fan-out (`.broadcast()` or
/// [`OutputDelivery::broadcast`](crate::OutputDelivery::broadcast)).
///
/// This API is intentionally absent from [`jonmo::prelude`](crate::prelude):
///
/// ```compile_fail,E0432
/// use jonmo::prelude::run_signal_subgraph_once;
/// ```
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use jonmo::{
///     graph::{downcast_signal_value, run_signal_subgraph_once},
///     prelude::*,
/// };
///
/// let mut app = App::new();
/// app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
/// let claim = signal::once(42usize).register(app.world_mut());
/// let value = run_signal_subgraph_once(app.world_mut(), claim.entity())
///     .and_then(downcast_signal_value::<usize>);
/// assert_eq!(value, Some(42));
/// claim.release(app.world_mut());
/// ```
#[cfg(feature = "imperative-signal-evaluation")]
#[track_caller]
pub fn run_signal_subgraph_once(world: &mut World, signal: SignalEntity) -> Option<BoxedSignalValue> {
    run_signal_subgraph_once_internal(world, signal)
}

/// Extracts a concrete value of type `T` from a [`BoxedSignalValue`] returned by
/// [`run_signal_subgraph_once`].
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use jonmo::{
///     graph::{downcast_signal_value, run_signal_subgraph_once},
///     prelude::*,
/// };
///
/// let mut app = App::new();
/// app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
/// let claim = signal::once(1usize).register(app.world_mut());
/// let value = run_signal_subgraph_once(app.world_mut(), claim.entity())
///     .and_then(downcast_signal_value::<usize>);
/// assert_eq!(value, Some(1));
/// claim.release(app.world_mut());
/// ```
#[cfg(feature = "imperative-signal-evaluation")]
pub fn downcast_signal_value<T: 'static>(value: BoxedSignalValue) -> Option<T> {
    downcast_erased_signal_value(value)
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ScheduleSnapshot(String);

#[cfg(test)]
impl ScheduleSnapshot {
    pub(crate) fn new(schedule: InternedScheduleLabel) -> Self {
        Self(alloc::format!("{schedule:?}"))
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ScheduleBucketSnapshot {
    pub(crate) schedule: ScheduleSnapshot,
    pub(crate) level: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReplayRelationshipSnapshotState {
    Pending,
    Published { cursor: ReplayRevision },
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReplayRelationshipSnapshot {
    pub(crate) dependent: Entity,
    pub(crate) state: ReplayRelationshipSnapshotState,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DynamicSubscriptionSnapshot {
    pub(crate) selected: Entity,
    pub(crate) adapter: Entity,
    pub(crate) target: Entity,
    pub(crate) slot_identity: u64,
    pub(crate) generation: u64,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GraphNodeSnapshot {
    pub(crate) entity: Entity,
    pub(crate) exists: bool,
    pub(crate) claim_count: Option<usize>,
    pub(crate) canonical_static_dependencies: Option<Vec<Entity>>,
    pub(crate) materialized_dependents: Vec<Entity>,
    pub(crate) active_downstreams: Vec<Entity>,
    pub(crate) dynamic_subscriptions: Vec<DynamicSubscriptionSnapshot>,
    pub(crate) schedule_tag: Option<ScheduleSnapshot>,
    pub(crate) schedule_hint: Option<ScheduleSnapshot>,
    pub(crate) cached_level: Option<u32>,
    pub(crate) cached_schedule: Option<ScheduleSnapshot>,
    pub(crate) schedule_buckets: Vec<ScheduleBucketSnapshot>,
    pub(crate) edge_change_seed: bool,
    pub(crate) deferred_removal: bool,
    pub(crate) has_system_runner: bool,
    pub(crate) pending_input_count: Option<usize>,
    pub(crate) terminal_retainer_count: usize,
    pub(crate) live_terminal_retainer_count: usize,
    pub(crate) replay_maintenance_armed: bool,
    pub(crate) replay_relationships: Vec<ReplayRelationshipSnapshot>,
    pub(crate) replay_cache_readiness: Option<ReplayCacheReadiness>,
    pub(crate) replay_cache_type: Option<String>,
    pub(crate) registrar_inactive: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SignalGraphCacheSnapshot {
    pub(crate) default_schedule: ScheduleSnapshot,
    pub(crate) registered_schedules: Vec<ScheduleSnapshot>,
    pub(crate) is_processing: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GraphSnapshot {
    pub(crate) graph_cache: Option<SignalGraphCacheSnapshot>,
    pub(crate) nodes: Vec<GraphNodeSnapshot>,
}

#[cfg(test)]
impl GraphSnapshot {
    pub(crate) fn node(&self, entity: Entity) -> Option<&GraphNodeSnapshot> {
        self.nodes.iter().find(|node| node.entity == entity)
    }
}

#[cfg(test)]
fn sorted_entities(signals: impl IntoIterator<Item = SignalEntity>) -> Vec<Entity> {
    let mut entities = signals.into_iter().map(|signal| signal.0).collect::<Vec<_>>();
    entities.sort();
    entities.dedup();
    entities
}

#[cfg(test)]
pub(crate) fn graph_snapshot(world: &mut World) -> GraphSnapshot {
    let entities = world.query::<Entity>().iter(world).collect::<Vec<_>>();
    let mut identities: HashSet<SignalEntity> = HashSet::default();
    for entity in entities {
        if world.get::<SignalClaimCount>(entity).is_some()
            || world.get::<StaticSignalDependencies>(entity).is_some()
            || world.get::<MaterializedSignalDependents>(entity).is_some()
            || world.get::<ActiveSignalDependents>(entity).is_some()
            || world.get::<RegistrationReplayCacheState>(entity).is_some()
            || world.get::<ScheduleTag>(entity).is_some()
            || world.get::<ScheduleHint>(entity).is_some()
            || world.get::<SystemRunner>(entity).is_some()
            || world.get::<StoredOutputDelivery>(entity).is_some()
            || world.get::<SignalInputBuffer>(entity).is_some()
        {
            identities.insert(SignalEntity(entity));
        }

        if let Some(dependencies) = world.get::<StaticSignalDependencies>(entity) {
            identities.extend(dependencies.0.iter().copied());
        }
        if let Some(dependents) = world.get::<MaterializedSignalDependents>(entity) {
            identities.extend(dependents.0.iter().copied());
        }
        if let Some(downstreams) = world.get::<ActiveSignalDependents>(entity) {
            identities.extend(downstreams.iter().copied());
        }
    }

    let graph_state = world.get_resource::<SignalGraphState>();
    if let Some(graph_state) = graph_state {
        identities.extend(graph_state.levels.keys().copied());
        identities.extend(graph_state.signal_schedules.keys().copied());
        identities.extend(graph_state.edge_change_seeds.iter().copied());
        identities.extend(graph_state.deferred_removals.iter().copied());
        for levels in graph_state.by_schedule.values() {
            for level in levels {
                identities.extend(level.iter().copied());
            }
        }
    }

    let mut identities = identities.into_iter().collect::<Vec<_>>();
    identities.sort_by(signal_identity_cmp);
    identities.dedup();

    let mut nodes = Vec::with_capacity(identities.len());
    for signal in identities {
        let entity = signal.0;
        let mut schedule_buckets = Vec::new();
        if let Some(graph_state) = graph_state {
            for (schedule, levels) in &graph_state.by_schedule {
                for (level, members) in levels.iter().enumerate() {
                    if members.contains(&signal) {
                        schedule_buckets.push(ScheduleBucketSnapshot {
                            schedule: ScheduleSnapshot::new(*schedule),
                            level,
                        });
                    }
                }
            }
        }
        schedule_buckets.sort();

        let (terminal_retainer_count, live_terminal_retainer_count) = world
            .get::<SignalTerminalRetainers>(entity)
            .map(|retainers| {
                (
                    retainers.0.len(),
                    retainers.0.iter().filter(|retainer| retainer.is_live()).count(),
                )
            })
            .unwrap_or_default();
        let replay_metadata = world.get::<RegistrationReplayCacheState>(entity);
        let replay_inspection = replay_metadata.and_then(|metadata| metadata.contract.inspect(world, signal));
        let replay_cache_readiness = replay_inspection.as_ref().map(|inspection| match inspection {
            ErasedReplaySnapshot::Warming => ReplayCacheReadiness::Warming,
            ErasedReplaySnapshot::Ready { revision, .. } => ReplayCacheReadiness::Ready { revision: *revision },
        });
        let replay_cache_type = replay_metadata.map(|metadata| alloc::format!("{:?}", metadata.contract.cache_type()));
        let replay_relationships = world
            .get::<ReplayConsumingRelationships>(entity)
            .map(|relationships| {
                relationships
                    .0
                    .iter()
                    .map(|(dependent, state)| ReplayRelationshipSnapshot {
                        dependent: dependent.entity(),
                        state: match state {
                            ReplayRelationshipState::Pending => ReplayRelationshipSnapshotState::Pending,
                            ReplayRelationshipState::Published { cursor } => {
                                ReplayRelationshipSnapshotState::Published { cursor: *cursor }
                            }
                        },
                    })
                    .collect()
            })
            .unwrap_or_default();

        nodes.push(GraphNodeSnapshot {
            entity,
            exists: world.get_entity(entity).is_ok(),
            claim_count: world.get::<SignalClaimCount>(entity).map(|count| **count),
            canonical_static_dependencies: world
                .get::<StaticSignalDependencies>(entity)
                .map(|dependencies| sorted_entities(dependencies.0.iter().copied())),
            materialized_dependents: world
                .get::<MaterializedSignalDependents>(entity)
                .map(|dependents| sorted_entities(dependents.0.iter().copied()))
                .unwrap_or_default(),
            active_downstreams: world
                .get::<ActiveSignalDependents>(entity)
                .map(|downstreams| sorted_entities(downstreams.iter().copied()))
                .unwrap_or_default(),
            dynamic_subscriptions: world
                .get::<DynamicSignalOwner>(entity)
                .map(|owner| {
                    owner
                        .subscriptions()
                        .into_iter()
                        .map(|subscription| DynamicSubscriptionSnapshot {
                            selected: *subscription.selected,
                            adapter: *subscription.adapter,
                            target: *subscription.target,
                            slot_identity: subscription.slot.identity,
                            generation: subscription.slot.generation,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            schedule_tag: world.get::<ScheduleTag>(entity).map(|tag| ScheduleSnapshot::new(tag.0)),
            schedule_hint: world
                .get::<ScheduleHint>(entity)
                .map(|hint| ScheduleSnapshot::new(hint.0)),
            cached_level: graph_state.and_then(|state| state.levels.get(&signal).copied()),
            cached_schedule: graph_state
                .and_then(|state| state.signal_schedules.get(&signal).copied())
                .map(ScheduleSnapshot::new),
            schedule_buckets,
            edge_change_seed: graph_state.is_some_and(|state| state.edge_change_seeds.contains(&signal)),
            deferred_removal: graph_state.is_some_and(|state| state.deferred_removals.contains(&signal)),
            has_system_runner: world.get::<SystemRunner>(entity).is_some(),
            pending_input_count: world.get::<SignalInputBuffer>(entity).map(|inputs| inputs.0.len()),
            terminal_retainer_count,
            live_terminal_retainer_count,
            replay_maintenance_armed: world.get::<ReplayMaintenanceClaim>(entity).is_some(),
            replay_relationships,
            replay_cache_readiness,
            replay_cache_type,
            registrar_inactive: world.get::<RegistrarInactive>(entity).is_some(),
        });
    }

    let graph_cache = graph_state.map(|state| {
        let mut registered_schedules = state
            .registered_schedules
            .iter()
            .copied()
            .map(ScheduleSnapshot::new)
            .collect::<Vec<_>>();
        registered_schedules.sort();
        registered_schedules.dedup();
        SignalGraphCacheSnapshot {
            default_schedule: ScheduleSnapshot::new(state.default_schedule),
            registered_schedules,
            is_processing: state.is_processing,
        }
    });

    GraphSnapshot { graph_cache, nodes }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DirectOwnershipKind {
    External,
    Dynamic,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DirectOwnershipToken {
    node: Entity,
    kind: DirectOwnershipKind,
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct DirectOwnershipModel {
    dependencies: Vec<(Entity, Vec<Entity>)>,
    external_counts: Vec<(Entity, usize)>,
    dynamic_counts: Vec<(Entity, usize)>,
}

#[cfg(test)]
impl DirectOwnershipModel {
    pub(crate) fn add_node(&mut self, node: Entity, dependencies: impl IntoIterator<Item = Entity>) {
        let mut dependencies = dependencies.into_iter().collect::<Vec<_>>();
        dependencies.sort();
        dependencies.dedup();
        if let Some((_, current)) = self.dependencies.iter_mut().find(|(current, _)| *current == node) {
            *current = dependencies;
        } else {
            self.dependencies.push((node, dependencies));
            self.dependencies.sort_by_key(|(entity, _)| *entity);
        }
    }

    pub(crate) fn acquire_external(&mut self, node: Entity) -> DirectOwnershipToken {
        self.increment_base(node, DirectOwnershipKind::External);
        DirectOwnershipToken {
            node,
            kind: DirectOwnershipKind::External,
        }
    }

    pub(crate) fn acquire_dynamic(&mut self, node: Entity) -> DirectOwnershipToken {
        self.increment_base(node, DirectOwnershipKind::Dynamic);
        DirectOwnershipToken {
            node,
            kind: DirectOwnershipKind::Dynamic,
        }
    }

    pub(crate) fn release(&mut self, token: DirectOwnershipToken) {
        let counts = match token.kind {
            DirectOwnershipKind::External => &mut self.external_counts,
            DirectOwnershipKind::Dynamic => &mut self.dynamic_counts,
        };
        let Some((_, count)) = counts.iter_mut().find(|(node, _)| *node == token.node) else {
            panic!("direct ownership model released an unknown token for {:?}", token.node);
        };
        assert!(*count > 0, "direct ownership model underflow for {:?}", token.node);
        *count -= 1;
    }

    pub(crate) fn count(&self, node: Entity) -> usize {
        self.counts()
            .into_iter()
            .find_map(|(current, count)| (current == node).then_some(count))
            .unwrap_or_default()
    }

    pub(crate) fn active_routes(&self) -> Vec<(Entity, Entity)> {
        let counts = self.counts();
        let mut routes = Vec::new();
        for (dependent, dependencies) in &self.dependencies {
            let active = counts
                .iter()
                .find_map(|(node, count)| (node == dependent).then_some(*count > 0))
                .unwrap_or(false);
            if active {
                routes.extend(dependencies.iter().map(|dependency| (*dependency, *dependent)));
            }
        }
        routes.sort();
        routes.dedup();
        routes
    }

    fn increment_base(&mut self, node: Entity, kind: DirectOwnershipKind) {
        let counts = match kind {
            DirectOwnershipKind::External => &mut self.external_counts,
            DirectOwnershipKind::Dynamic => &mut self.dynamic_counts,
        };
        if let Some((_, count)) = counts.iter_mut().find(|(current, _)| *current == node) {
            *count = count
                .checked_add(1)
                .unwrap_or_else(|| panic!("direct ownership model overflow for {node:?}"));
        } else {
            counts.push((node, 1));
            counts.sort_by_key(|(entity, _)| *entity);
        }
    }

    fn base_count(&self, node: Entity) -> usize {
        self.external_counts
            .iter()
            .chain(&self.dynamic_counts)
            .filter_map(|(current, count)| (current == &node).then_some(*count))
            .sum()
    }

    fn counts(&self) -> Vec<(Entity, usize)> {
        let mut counts = self
            .dependencies
            .iter()
            .map(|(node, _)| (*node, self.base_count(*node)))
            .collect::<Vec<_>>();
        loop {
            let mut next = self
                .dependencies
                .iter()
                .map(|(node, _)| (*node, self.base_count(*node)))
                .collect::<Vec<_>>();
            for (dependent, dependencies) in &self.dependencies {
                let dependent_active = counts
                    .iter()
                    .find_map(|(node, count)| (node == dependent).then_some(*count > 0))
                    .unwrap_or(false);
                if dependent_active {
                    for dependency in dependencies {
                        let Some((_, count)) = next.iter_mut().find(|(node, _)| node == dependency) else {
                            panic!("direct ownership model is missing dependency node {dependency:?}");
                        };
                        *count = count
                            .checked_add(1)
                            .unwrap_or_else(|| panic!("direct ownership model overflow for {dependency:?}"));
                    }
                }
            }
            if next == counts {
                return counts;
            }
            counts = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bevy::prelude::{App, MinimalPlugins};
    use bevy_ecs::{
        prelude::{In, World},
        schedule::ScheduleLabel,
    };

    use crate::{JonmoPlugin, OutputDelivery, cleanup};

    #[test]
    fn ordered_chain_rejects_duplicate_or_replay_inputs_before_commit() {
        use crate::signal_vec::{ChainInput, VecDiff};

        for (duplicate, replayable) in [(true, false), (false, true)] {
            for activate in [false, true] {
                let mut world = lazy_cleanup_world();
                let materializer = crate::VecSignalMaterializer::<u8>::new(move |registrar| {
                    let left = if replayable {
                        registrar.replayable_signal_source(
                            ChainInput::Left(Vec::new()),
                            |In(())| None::<ChainInput<u8>>,
                            OutputDelivery::broadcast(),
                        )?
                    } else {
                        registrar.signal_entity(
                            (),
                            |In(())| Some(ChainInput::<u8>::Left(Vec::new())),
                            OutputDelivery::broadcast(),
                        )?
                    };
                    let right = if duplicate {
                        left
                    } else {
                        registrar.signal_entity(
                            (),
                            |In(())| Some(ChainInput::<u8>::Right(Vec::new())),
                            OutputDelivery::move_only(),
                        )?
                    };
                    registrar.ordered_chain_consumer(
                        [left, right],
                        |_input, _epoch| None::<Vec<VecDiff<u8>>>,
                        OutputDelivery::broadcast(),
                    )
                });
                let before = graph_snapshot(&mut world);
                let result = if activate {
                    materializer.try_register(&mut world).map(|claim| claim.entity())
                } else {
                    materialize_materializer(&mut world, materializer)
                };
                assert_eq!(result, Err(SignalRegistrationError::ConflictingCanonicalState));
                assert_eq!(graph_snapshot(&mut world), before);
            }
        }
    }

    #[test]
    fn ordered_chain_routes_preserve_replay_taggers_and_static_lifecycle() {
        use crate::signal_vec::{ChainInput, VecDiff};

        for processing in [false, true] {
            let mut world = lazy_cleanup_world();
            let materializer = crate::VecSignalMaterializer::<u8>::new(|registrar| {
                let source = registrar.replayable_signal_vec_source(
                    vec![1],
                    |In(())| None::<Vec<VecDiff<u8>>>,
                    OutputDelivery::broadcast(),
                )?;
                let left = registrar.signal_entity(
                    source,
                    |In(diffs)| Some(ChainInput::Left(diffs)),
                    OutputDelivery::move_only(),
                )?;
                let right = registrar.signal_entity(
                    source,
                    |In(diffs)| Some(ChainInput::Right(diffs)),
                    OutputDelivery::move_only(),
                )?;
                registrar.ordered_chain_consumer(
                    [left, right],
                    |input, _epoch| match input {
                        ChainInput::Left(diffs) | ChainInput::Right(diffs) => Some(diffs),
                    },
                    OutputDelivery::broadcast(),
                )
            });
            let retained = materializer.shared_clone();
            let mut claim = materializer.register(&mut world);
            let consumer = claim.entity();
            let taggers = get_static_dependencies(&world, consumer);
            assert_eq!(taggers.len(), 2);
            let source = get_static_dependencies(&world, taggers[0])[0];
            let consumption = world
                .get::<SystemRunner>(*consumer)
                .expect("consumer runner")
                .ordered_input_consumption;
            assert!(matches!(consumption, Some(OrderedInputConsumption::Chain { .. })));
            assert!(
                world
                    .get::<SystemRunner>(*consumer)
                    .expect("consumer runner")
                    .collection_poll_append
                    .is_some()
            );

            for activation in 0..2 {
                assert!(
                    world
                        .get::<SignalInputBuffer>(*consumer)
                        .expect("consumer buffer")
                        .0
                        .is_empty()
                );
                for tagger in &taggers {
                    let runner = world.get::<SystemRunner>(**tagger).expect("tagger runner");
                    assert!(runner.ordered_input_consumption.is_none());
                    assert!(runner.collection_poll_append.is_none());
                    assert!(runner.replay_key.is_none());
                    assert_eq!(get_downstreams(&world, *tagger), vec![consumer]);
                    assert!(world.get::<ReplayConsumingRelationships>(**tagger).is_none());
                    assert!(matches!(
                        world
                            .get::<ReplayConsumingRelationships>(*source)
                            .expect("source cursors")
                            .0
                            .get(tagger),
                        Some(ReplayRelationshipState::Published { .. })
                    ));
                    assert_eq!(
                        world
                            .get::<SignalInputBuffer>(**tagger)
                            .expect("tagger bootstrap")
                            .0
                            .len(),
                        1
                    );
                }
                assert!(push_signal_input(
                    &mut world,
                    taggers[0],
                    Box::new(vec![VecDiff::Push { value: 2_u8 }])
                ));
                for tagger in &taggers {
                    run_signal_node(&mut world, *tagger);
                }
                let mut batches = 0;
                let mut count_batches = |_: &SystemRunner, _: ErasedSignalValue| batches += 1;
                execute_signal_node(&mut world, consumer, Some(&mut count_batches));
                assert_eq!(batches, 3, "ordered inputs must not use tagger scalar selection");
                validate_existing_registration_runtime(&world, consumer).expect("active ordered runtime");

                world.resource_mut::<SignalGraphState>().is_processing = processing;
                claim.try_release(&mut world).expect("ordered final release");
                for entity in taggers.iter().copied().chain([consumer, source]) {
                    assert_eq!(world.get::<SignalClaimCount>(*entity).expect("count").0, 0);
                    assert!(world.get::<RegistrarInactive>(*entity).is_some());
                    assert!(get_downstreams(&world, entity).is_empty());
                    validate_existing_registration_runtime(&world, entity).expect("inactive ordered runtime");
                }
                assert_eq!(
                    world
                        .get::<SystemRunner>(*consumer)
                        .expect("retained runner")
                        .ordered_input_consumption,
                    consumption
                );
                if activation == 0 {
                    claim = retained.shared_clone().register(&mut world);
                    assert_eq!(claim.entity(), consumer);
                } else {
                    break;
                }
                world.resource_mut::<SignalGraphState>().is_processing = false;
                apply_deferred_removals(&mut world.resource_mut::<SignalGraphState>());
            }
            world.resource_mut::<SignalGraphState>().is_processing = false;
            apply_deferred_removals(&mut world.resource_mut::<SignalGraphState>());
        }
    }

    #[test]
    fn ordered_chain_runtime_rejects_changed_type_metadata() {
        use crate::signal_vec::ChainInput;

        let mut world = lazy_cleanup_world();
        let materializer = crate::VecSignalMaterializer::<u8>::new(|registrar| {
            let left = registrar.signal_entity(
                (),
                |In(())| Some(ChainInput::<u8>::Left(Vec::new())),
                OutputDelivery::move_only(),
            )?;
            let right = registrar.signal_entity(
                (),
                |In(())| Some(ChainInput::<u8>::Right(Vec::new())),
                OutputDelivery::move_only(),
            )?;
            registrar.ordered_chain_consumer(
                [left, right],
                |input, _epoch| match input {
                    ChainInput::Left(diffs) | ChainInput::Right(diffs) => Some(diffs),
                },
                OutputDelivery::broadcast(),
            )
        });
        let claim = materializer.register(&mut world);
        let consumer = claim.entity();
        let runner = world.get::<SystemRunner>(*consumer).expect("consumer runner");
        let declaration = RegistrationEntityPreflight {
            ordered_input_consumption: None,
            entity: PlannedSignalEntityRef::Existing(consumer),
            dependencies: get_static_dependencies(&world, consumer)
                .into_iter()
                .map(PlannedSignalEntityRef::Existing)
                .collect(),
            input_type: runner.input_type,
            output_type: runner.output_type,
            pending_output_delivery: None,
            expected_output_delivery: None,
            replayable: false,
            planned_replay: None,
        };
        assert_eq!(
            world.registration_preflight_canonical_state(&declaration),
            Err(SignalRegistrationError::ConflictingCanonicalState)
        );
        world
            .get_mut::<SystemRunner>(*consumer)
            .expect("consumer runner")
            .input_type = TypeId::of::<u8>();
        let before = graph_snapshot(&mut world);
        assert_eq!(
            validate_existing_registration_runtime(&world, consumer),
            Err(inconsistent_existing_runtime(consumer))
        );
        assert_eq!(graph_snapshot(&mut world), before);
    }

    #[test]
    fn graph_lifecycle_deferred_reactivation_restores_exact_projection() {
        let mut world = lazy_cleanup_world();
        let (retained, claim, source, terminal) = active_unary_registration_fixture(&mut world);
        let schedule = world.resource::<SignalGraphState>().default_schedule;
        world.resource_mut::<SignalGraphState>().is_processing = true;
        claim.try_release(&mut world).expect("processing-time final release");
        for entity in [source, terminal] {
            assert_eq!(world.get::<SignalClaimCount>(*entity).expect("count").0, 0);
            assert!(world.get::<RegistrarInactive>(*entity).is_some());
            assert!(world.resource::<SignalGraphState>().deferred_removals.contains(&entity));
            assert!(get_downstreams(&world, entity).is_empty());
        }
        let claim = retained
            .shared_clone()
            .try_register(&mut world)
            .expect("deferred reactivation");
        assert_eq!(claim.entity(), terminal);
        for (entity, level) in [(source, 0), (terminal, 1)] {
            let snapshot = graph_snapshot(&mut world);
            let node = snapshot.node(*entity).expect("reactivated node");
            assert_eq!(node.claim_count, Some(1));
            assert!(!node.registrar_inactive);
            assert!(!node.deferred_removal);
            assert_eq!(node.cached_level, Some(level));
            assert_eq!(
                node.schedule_buckets,
                vec![ScheduleBucketSnapshot {
                    schedule: ScheduleSnapshot::new(schedule),
                    level: level as usize
                }]
            );
            validate_existing_registration_runtime(&world, entity).expect("immediately valid runtime");
        }
        assert_eq!(get_downstreams(&world, source), vec![terminal]);
        world.resource_mut::<SignalGraphState>().is_processing = false;
        apply_deferred_removals(&mut world.resource_mut::<SignalGraphState>());
        validate_existing_registration_runtime(&world, terminal).expect("cleanup preserves reactivation");
        claim.try_release(&mut world).expect("immediate final release");
    }

    #[test]
    fn graph_lifecycle_deferred_projection_corruption_rejects_atomically() {
        for corruption in 0..7 {
            let mut world = lazy_cleanup_world();
            let (retained, claim, _source, terminal) = active_unary_registration_fixture(&mut world);
            world.resource_mut::<SignalGraphState>().is_processing = true;
            claim.try_release(&mut world).expect("release");
            let schedule = world.resource::<SignalGraphState>().default_schedule;
            {
                let mut state = world.resource_mut::<SignalGraphState>();
                match corruption {
                    0 => {
                        state.deferred_removals.remove(&terminal);
                    }
                    1 => {
                        state.is_processing = false;
                    }
                    2 => {
                        state.by_schedule.get_mut(&schedule).expect("buckets")[1].push(terminal);
                    }
                    3 => {
                        state.levels.remove(&terminal);
                    }
                    4 => {
                        state.signal_schedules.remove(&terminal);
                    }
                    5 => {
                        state.levels.insert(terminal, 2);
                    }
                    6 => {
                        state.by_schedule.get_mut(&schedule).expect("buckets")[0].push(terminal);
                    }
                    _ => unreachable!(),
                }
            }
            assert_runtime_inconsistency_is_atomic(&mut world, retained.shared_clone(), terminal);
        }
    }

    #[test]
    fn graph_lifecycle_active_deferred_removal_rejects_atomically() {
        let mut world = lazy_cleanup_world();
        let (retained, _claim, _source, terminal) = active_unary_registration_fixture(&mut world);
        let mut state = world.resource_mut::<SignalGraphState>();
        state.is_processing = true;
        state.deferred_removals.insert(terminal);
        assert_runtime_inconsistency_is_atomic(&mut world, retained.shared_clone(), terminal);
    }

    #[test]
    fn graph_lifecycle_projected_dynamic_connection_sets_execution_not_ownership_level() {
        use crate::registration::{RegistrationClaimDelta, RegistrationDynamicConnectionPreflight};

        let mut world = lazy_cleanup_world();
        let deep = crate::ScalarSignalMaterializer::<()>::new(|registrar| {
            let mut source = registrar.signal_entity((), |In(())| Some(()), OutputDelivery::broadcast())?;
            for _ in 0..8 {
                source = registrar.signal_entity(source, |In(())| Some(()), OutputDelivery::broadcast())?;
            }
            Ok(source)
        });
        let deep_claim = deep.register(&mut world);
        let target = crate::ScalarSignalMaterializer::<()>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(()), OutputDelivery::broadcast())
        });
        let retained = target.shared_clone();
        let target = materialize_materializer(&mut world, target).expect("inactive target");
        let target_ref = PlannedSignalEntityRef::Existing(target);
        let source_ref = PlannedSignalEntityRef::Existing(deep_claim.entity());
        let plan = RegistrationPreflight {
            entities: Box::new([]),
            schedules: Box::new([]),
            claim_deltas: Box::new([RegistrationClaimDelta {
                entity: target_ref,
                increment: 1,
            }]),
            dynamic_connections: Box::new([RegistrationDynamicConnectionPreflight {
                source: source_ref,
                target: target_ref,
            }]),
        };
        let before = graph_snapshot(&mut world);
        world
            .registration_preflight_dynamic_connections(&plan)
            .expect("projected connection");
        let activation = world
            .registration_preflight_activation(&plan)
            .expect("activation projection");
        assert_eq!(graph_snapshot(&mut world), before, "preflight must be read-only");
        assert_eq!(
            activation.count_increments.len(),
            1,
            "dynamic connection is not a static claim"
        );
        assert_eq!(activation.count_increments[0].entity, target_ref);
        assert!(
            activation.active_routes.is_empty(),
            "dynamic connection is not a static route"
        );
        assert_eq!(activation.effective_membership[0].level, 9);
        drop(retained);
        deep_claim.try_release(&mut world).expect("deep release");
    }

    #[test]
    fn graph_lifecycle_poll_skips_target_reaped_by_upstream_callback() {
        #[derive(Resource, Default)]
        struct ReleaseTarget(Option<SignalClaim>);

        for step_imperatively in [false, true] {
            let mut world = lazy_cleanup_world();
            world.init_resource::<ReleaseTarget>();
            let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let source = registrar.signal_entity(
                    (),
                    |In(()), world: &mut World| {
                        let claim = world.resource_mut::<ReleaseTarget>().0.take().expect("target claim");
                        claim.try_release(world).expect("upstream releases target");
                        Some(1)
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_entity(
                    source,
                    |In(_): In<i32>| -> Option<i32> { panic!("reaped target must not execute") },
                    OutputDelivery::move_only(),
                )
            });
            let claim = materializer.register(&mut world);
            let target = claim.entity();
            world.resource_mut::<ReleaseTarget>().0 = Some(claim);
            if step_imperatively {
                assert!(run_signal_subgraph_once_one_shot(In(target), &mut world).is_none());
            } else {
                let schedule = world.resource::<SignalGraphState>().default_schedule;
                process_signal_graph_for_schedule(schedule)(&mut world);
            }
            assert!(world.get_entity(*target).is_err());
        }
    }

    #[test]
    fn graph_lifecycle_late_descendant_levels_include_current_and_pending_owned_connections() {
        use crate::{
            DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery, ReplayInitialization,
            SameSignalIdentity, ScalarSignalMaterializer, ScalarSlotCommand, prelude::*,
        };

        struct DeepSelection {
            depth: usize,
            ready: bool,
        }
        impl Signal for DeepSelection {
            type Item = i32;
            fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
                ScalarSignalMaterializer::new(move |registrar| {
                    let mut selected = if self.ready {
                        registrar.replayable_signal_source(41, |In(())| None::<i32>, OutputDelivery::broadcast())?
                    } else {
                        registrar.replayable_signal_entity(
                            (),
                            |In(())| None::<i32>,
                            ReplayInitialization::Warming,
                            OutputDelivery::broadcast(),
                        )?
                    };
                    for _ in 0..self.depth {
                        selected = registrar.replayable_signal_entity(
                            selected,
                            |In(value): In<i32>| Some(value),
                            ReplayInitialization::Warming,
                            OutputDelivery::broadcast(),
                        )?;
                    }
                    Ok(selected)
                })
            }
        }

        for pending in [false, true] {
            let mut app = App::new();
            app.add_plugins(JonmoPlugin::default());
            let mut selections = VecDeque::from([DeepSelection { depth: 4, ready: true }.cache().boxed_replayable()]);
            if pending {
                selections.push_back(DeepSelection { depth: 8, ready: false }.cache().boxed_replayable());
            }
            let owner = ScalarSignalMaterializer::<i32>::new(move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        selections.pop_front().map(|signal| {
                            DynamicManagerUpdate::new(
                                [ScalarSlotCommand::Select { signal }],
                                DynamicEmissions::<i32>::None,
                            )
                        })
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(|values| values.into_iter().last())
                        .same_signal_identity(SameSignalIdentity::Restart)
                        .on_candidate_error(|error| panic!("candidate failed: {error:?}")),
                )
            });
            let retained = owner.shared_clone();
            let claim = owner.register(app.world_mut());
            for _ in 0..12 {
                app.update();
            }
            let owner = claim.entity();
            let subscriptions = app
                .world()
                .get::<DynamicSignalOwner>(*owner)
                .expect("dynamic owner")
                .subscriptions();
            assert_eq!(subscriptions.len(), if pending { 2 } else { 1 });
            let before = graph_snapshot(app.world_mut());
            let owner_level = before
                .node(*owner)
                .expect("owner node")
                .cached_level
                .expect("owner level");
            let expected_level = subscriptions
                .iter()
                .map(|subscription| {
                    before
                        .node(*subscription.adapter)
                        .expect("adapter")
                        .cached_level
                        .expect("adapter level")
                        + 1
                })
                .max()
                .expect("owned adapters");
            assert_eq!(owner_level, expected_level);
            assert!(owner_level >= if pending { 10 } else { 6 });
            let descendant = ScalarSignalMaterializer::<i32>::new(move |registrar| {
                let owner = registrar.materialize_signal(retained)?;
                let child =
                    registrar.signal_entity(owner, |In(value): In<i32>| Some(value), OutputDelivery::move_only())?;
                registrar.signal_entity(child, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
            })
            .register(app.world_mut());
            let terminal = descendant.entity();
            let child = get_static_dependencies(app.world(), terminal)[0];
            for (entity, expected_level) in [(child, owner_level + 1), (terminal, owner_level + 2)] {
                let after = graph_snapshot(app.world_mut());
                let node = after.node(*entity).expect("late descendant");
                assert_eq!(node.cached_level, Some(expected_level));
                assert_eq!(node.schedule_buckets.len(), 1);
                assert_eq!(node.schedule_buckets[0].level, expected_level as usize);
                validate_existing_registration_runtime(app.world(), entity).expect("no scheduler repair needed");
            }
            for subscription in &subscriptions {
                assert_eq!(
                    app.world()
                        .get::<SignalClaimCount>(*subscription.adapter)
                        .expect("adapter count")
                        .0,
                    before
                        .node(*subscription.adapter)
                        .expect("adapter before")
                        .claim_count
                        .expect("count before")
                );
            }
            descendant
                .try_release(app.world_mut())
                .expect("immediate descendant release");
            claim
                .try_release(app.world_mut())
                .expect("owner releases current and pending ownership");
        }
    }

    #[test]
    fn graph_lifecycle_pending_replay_descendant_levels_follow_dynamic_connection_changes() {
        use crate::{
            DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery, ReplayInitialization,
            SameSignalIdentity, ScalarSignalMaterializer, ScalarSlotCommand, prelude::*,
        };

        for remove_connection in [false, true] {
            let mut app = App::new();
            app.add_plugins(JonmoPlugin::default());
            let mut candidate = Some(
                crate::signal::from_system(|In(())| None::<i32>)
                    .map(|In(value): In<i32>| value)
                    .map(|In(value): In<i32>| value)
                    .dedupe()
                    .boxed_replayable(),
            );
            let mut phase = 0;
            let claim = ScalarSignalMaterializer::<i32>::new(move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        phase += 1;
                        let command = match phase {
                            2 => ScalarSlotCommand::Select {
                                signal: candidate.take().expect("phase 2 runs once"),
                            },
                            3 => ScalarSlotCommand::Clear,
                            _ => return None,
                        };
                        Some(DynamicManagerUpdate::new([command], DynamicEmissions::<i32>::None))
                    },
                    OutputDelivery::move_only(),
                )?;
                let owner = registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .replayable_scalar_output(|_| None::<i32>)
                        .same_signal_identity(SameSignalIdentity::KeepCurrent),
                )?;
                let child = registrar.replayable_signal_entity(
                    owner,
                    |In(value): In<i32>| Some(value),
                    ReplayInitialization::Warming,
                    OutputDelivery::move_only(),
                )?;
                registrar.replayable_signal_entity(
                    child,
                    |In(value): In<i32>| Some(value),
                    ReplayInitialization::Warming,
                    OutputDelivery::move_only(),
                )
            })
            .register(app.world_mut());
            let terminal = claim.entity();
            let child = get_static_dependencies(app.world(), terminal)
                .first()
                .copied()
                .expect("terminal dependency");
            let owner = get_static_dependencies(app.world(), child)
                .first()
                .copied()
                .expect("child dependency");
            let schedule = app.world().resource::<SignalGraphState>().default_schedule;
            let expected_owner_levels = if remove_connection { vec![2, 5, 2] } else { vec![2, 5] };
            for owner_level in expected_owner_levels {
                app.update();
                let snapshot = graph_snapshot(app.world_mut());
                for (entity, level) in [
                    (owner, owner_level),
                    (child, owner_level + 1),
                    (terminal, owner_level + 2),
                ] {
                    let node = snapshot.node(*entity).expect("warming node");
                    assert_eq!(node.cached_level, Some(level));
                    assert_eq!(
                        node.schedule_buckets,
                        vec![ScheduleBucketSnapshot {
                            schedule: ScheduleSnapshot::new(schedule),
                            level: level as usize,
                        }]
                    );
                    assert_eq!(node.pending_input_count, Some(0));
                    let replay = app
                        .world()
                        .get::<RegistrationReplayCacheState>(*entity)
                        .expect("replay metadata");
                    assert!(matches!(
                        replay.contract.inspect(app.world(), entity),
                        Some(ErasedReplaySnapshot::Warming)
                    ));
                    validate_existing_registration_runtime(app.world(), entity)
                        .expect("exact pending execution levels");
                }
                for (dependency, dependent) in [(owner, child), (child, terminal)] {
                    assert!(matches!(
                        app.world()
                            .get::<ReplayConsumingRelationships>(*dependency)
                            .and_then(|relationships| relationships.0.get(&dependent)),
                        Some(ReplayRelationshipState::Pending)
                    ));
                    assert!(!get_downstreams(app.world(), dependency).contains(&dependent));
                }
                assert_eq!(
                    app.world()
                        .get::<DynamicSignalOwner>(*owner)
                        .expect("owner")
                        .subscriptions()
                        .len(),
                    usize::from(owner_level == 5)
                );
            }

            let terminal_level = if remove_connection { 4 } else { 7 };
            commit_cached_schedule_membership(app.world_mut(), terminal, schedule, terminal_level + 1);
            let before = graph_snapshot(app.world_mut());
            let (error, claim) = claim
                .try_release(app.world_mut())
                .expect_err("corrupt level must reject release");
            assert_eq!(
                error,
                SignalReleaseError::InconsistentExistingRuntimeState { entity: terminal }
            );
            assert_eq!(graph_snapshot(app.world_mut()), before, "rejected release is atomic");
            commit_cached_schedule_membership(app.world_mut(), terminal, schedule, terminal_level);
            claim
                .try_release(app.world_mut())
                .expect("immediate release without readiness or another update");
        }
    }

    #[test]
    fn graph_lifecycle_deferred_reactivation_uses_latest_schedule_after_partial_cleanup() {
        for repair_levels in [false, true] {
            let mut world = lazy_cleanup_world();
            let (retained, claim, source, terminal) = active_unary_registration_fixture(&mut world);
            let schedule = RegistrarScheduleA.intern();
            world.resource_mut::<SignalGraphState>().register_schedule(schedule);
            world.resource_mut::<SignalGraphState>().is_processing = true;
            claim.try_release(&mut world).expect("release");
            if repair_levels {
                world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
                    update_edge_change_levels(world, &mut state);
                });
            }
            world.registration_commit_schedule(RegistrationScheduleCommit {
                entity: terminal,
                tag: Some(schedule),
                hint: Some(schedule),
            });
            let claim = retained
                .shared_clone()
                .try_register(&mut world)
                .expect("reactivate with latest assignment");
            assert_eq!(
                world.resource::<SignalGraphState>().signal_schedules.get(&terminal),
                Some(&schedule)
            );
            assert!(!world.resource::<SignalGraphState>().deferred_removals.contains(&source));
            assert!(
                !world
                    .resource::<SignalGraphState>()
                    .deferred_removals
                    .contains(&terminal)
            );
            world.resource_mut::<SignalGraphState>().is_processing = false;
            apply_deferred_removals(&mut world.resource_mut::<SignalGraphState>());
            validate_existing_registration_runtime(&world, terminal).expect("exact restored membership");
            claim.try_release(&mut world).expect("final release");
        }
    }

    struct CustomScalarSource;

    impl crate::signal::Signal for CustomScalarSource {
        type Item = i32;

        fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
            crate::ScalarSignalMaterializer::new(|registrar| {
                registrar.signal_entity((), |In(())| Some(2), OutputDelivery::broadcast())
            })
        }
    }

    struct CustomStatefulUnary;

    impl crate::signal::Signal for CustomStatefulUnary {
        type Item = i32;

        fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
            crate::ScalarSignalMaterializer::new(|registrar| {
                let source = registrar.signal_entity((), |In(())| Some(2), OutputDelivery::broadcast())?;
                registrar.signal_entity(
                    source,
                    |In(value): In<i32>, mut total: Local<i32>| {
                        *total += value;
                        Some(*total)
                    },
                    OutputDelivery::broadcast(),
                )
            })
        }
    }

    struct CustomVectorSource;

    impl crate::signal_vec::SignalVec for CustomVectorSource {
        type Item = i32;

        fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
            crate::VecSignalMaterializer::new(|registrar| {
                registrar.signal_vec_entity(
                    (),
                    |In(())| Some(vec![crate::signal_vec::VecDiff::Replace { values: vec![1] }]),
                    OutputDelivery::broadcast(),
                )
            })
        }
    }

    struct CustomMapSource;

    struct ScalarReadinessVectorSnapshotContract;
    struct MalformedScalarSnapshotContract;

    impl ReplayRuntimeContract for ScalarReadinessVectorSnapshotContract {
        fn cache_type(&self) -> TypeId {
            TypeId::of::<crate::replay::ScalarReplayCache<i32>>()
        }

        fn event_type(&self) -> TypeId {
            TypeId::of::<i32>()
        }

        fn inspect(&self, world: &World, entity: SignalEntity) -> Option<ErasedReplaySnapshot> {
            match world.get::<crate::replay::ScalarReplayCache<i32>>(*entity)?.snapshot() {
                crate::replay::ReplayCacheSnapshot::Warming => Some(ErasedReplaySnapshot::Warming),
                crate::replay::ReplayCacheSnapshot::Ready { revision, .. } => {
                    let vector = world.get::<crate::replay::VecReplayCache<i32>>(*entity)?;
                    match vector.snapshot() {
                        crate::replay::ReplayCacheSnapshot::Warming => Some(ErasedReplaySnapshot::Warming),
                        crate::replay::ReplayCacheSnapshot::Ready { current, .. } => {
                            Some(ErasedReplaySnapshot::Ready {
                                snapshot: Box::new(vec![crate::signal_vec::VecDiff::Replace {
                                    values: current.clone(),
                                }]),
                                revision,
                            })
                        }
                    }
                }
            }
        }

        fn accept_output(
            &self,
            world: &mut World,
            entity: SignalEntity,
            output: ErasedSignalValue,
        ) -> Option<AcceptedSignalOutput> {
            crate::registration::scalar_replay_runtime_contract::<i32>().accept_output(world, entity, output)
        }
    }

    impl ReplayRuntimeContract for MalformedScalarSnapshotContract {
        fn cache_type(&self) -> TypeId {
            TypeId::of::<crate::replay::ScalarReplayCache<i32>>()
        }

        fn event_type(&self) -> TypeId {
            TypeId::of::<i32>()
        }

        fn inspect(&self, world: &World, entity: SignalEntity) -> Option<ErasedReplaySnapshot> {
            match world.get::<crate::replay::ScalarReplayCache<i32>>(*entity)?.snapshot() {
                crate::replay::ReplayCacheSnapshot::Warming => Some(ErasedReplaySnapshot::Warming),
                crate::replay::ReplayCacheSnapshot::Ready { revision, .. } => Some(ErasedReplaySnapshot::Ready {
                    snapshot: Box::new(String::from("not a scalar event")),
                    revision,
                }),
            }
        }

        fn accept_output(
            &self,
            world: &mut World,
            entity: SignalEntity,
            output: ErasedSignalValue,
        ) -> Option<AcceptedSignalOutput> {
            crate::registration::scalar_replay_runtime_contract::<i32>().accept_output(world, entity, output)
        }
    }

    impl crate::signal_map::SignalMap for CustomMapSource {
        type Key = i32;
        type Value = i32;

        fn into_signal_map_materializer(self: Box<Self>) -> crate::MapSignalMaterializer<Self::Key, Self::Value> {
            crate::MapSignalMaterializer::new(|registrar| {
                registrar.signal_map_entity(
                    (),
                    |In(())| Some(vec![crate::signal_map::MapDiff::Replace { entries: vec![(1, 2)] }]),
                    OutputDelivery::broadcast(),
                )
            })
        }
    }

    fn active_unary_registration_fixture(
        world: &mut World,
    ) -> (
        crate::ScalarSignalMaterializer<i32>,
        SignalClaim,
        SignalEntity,
        SignalEntity,
    ) {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?;
            registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::broadcast())
        });
        let retained = materializer.shared_clone();
        let claim = materializer.register(world);
        let terminal = claim.entity();
        let source = world
            .get::<StaticSignalDependencies>(*terminal)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("unary fixture should have one canonical dependency");
        (retained, claim, source, terminal)
    }

    fn linear_registration_entities(
        world: &World,
        terminal: SignalEntity,
    ) -> (SignalEntity, SignalEntity, SignalEntity) {
        let middle = world
            .get::<StaticSignalDependencies>(*terminal)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("linear terminal should have one dependency");
        let source = world
            .get::<StaticSignalDependencies>(*middle)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("linear middle should have one dependency");
        (source, middle, terminal)
    }

    fn assert_runtime_inconsistency_is_atomic<Kind: SignalKind>(
        world: &mut World,
        attempt: SignalMaterializer<Kind>,
        expected_entity: SignalEntity,
    ) {
        let before = graph_snapshot(world);
        assert!(matches!(
            attempt.try_register(world),
            Err(SignalRegistrationError::InconsistentExistingRuntimeState { entity })
                if entity == expected_entity
        ));
        assert_eq!(graph_snapshot(world), before);
    }

    fn assert_missing_runtime_component_rejects<T: Component>() {
        let mut world = lazy_cleanup_world();
        let (retained, _claim, _source, terminal) = active_unary_registration_fixture(&mut world);
        let attempt = retained.shared_clone();
        world.entity_mut(*terminal).remove::<T>();
        assert_runtime_inconsistency_is_atomic(&mut world, attempt, terminal);
    }

    fn inspect_coherent_replay_contract(
        world: &World,
        entity: SignalEntity,
        expected_cache_type: TypeId,
        expected_event_type: TypeId,
    ) -> ErasedReplaySnapshot {
        let runner = world
            .get::<SystemRunner>(*entity)
            .expect("replay entity should retain its fixed runtime");
        let key = runner.replay_key.expect("replay runtime should retain its fixed key");
        let runtime_contract = runner
            .replay_contract
            .as_ref()
            .expect("replay runtime should retain its contract");
        let metadata_contract = &world
            .get::<RegistrationReplayCacheState>(*entity)
            .expect("replay entity should retain its metadata contract")
            .contract;
        assert!(Arc::ptr_eq(runtime_contract, metadata_contract));
        assert_eq!(runner.output_type, expected_event_type);
        assert_eq!(key.cache_type(), expected_cache_type);
        assert_eq!(key.event_type(), expected_event_type);
        assert_eq!(runtime_contract.cache_type(), expected_cache_type);
        assert_eq!(runtime_contract.event_type(), expected_event_type);
        let inspection = runtime_contract
            .inspect(world, entity)
            .expect("coherent replay contract should inspect its typed cache");
        if let ErasedReplaySnapshot::Ready { snapshot, .. } = &inspection {
            assert_eq!(erased_type_id(snapshot), expected_event_type);
        }
        inspection
    }

    #[test]
    fn public_custom_materializers_share_registrar_bookkeeping_registration_ownership_observability() {
        use crate::{
            signal::{Signal, SignalExt},
            signal_map::SignalMapExt,
            signal_vec::SignalVecExt,
        };

        let mut world = lazy_cleanup_world();
        let scalar_materializer = Box::new(CustomScalarSource).into_signal_materializer();
        let retained_scalar = scalar_materializer.shared_clone();
        let scalar = materialize_materializer(&mut world, scalar_materializer)
            .expect("custom scalar source should materialize without a claim");
        assert_eq!(**world.get::<SignalClaimCount>(*scalar).unwrap(), 0);

        let scalar_claim = CustomStatefulUnary
            .try_register(&mut world)
            .expect("custom scalar should register through the structured path");
        let vector_claim = CustomVectorSource
            .try_register(&mut world)
            .expect("custom vector should register through the structured path");
        let map_claim = CustomMapSource
            .try_register(&mut world)
            .expect("custom map should register through the structured path");
        assert_eq!(**world.get::<SignalClaimCount>(*scalar_claim.entity()).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*vector_claim.entity()).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*map_claim.entity()).unwrap(), 1);

        let unary = graph_snapshot(&mut world)
            .node(scalar_claim.entity().entity())
            .expect("custom unary terminal should be observable")
            .clone();
        assert_eq!(unary.canonical_static_dependencies.as_ref().map(Vec::len), Some(1));

        scalar_claim.release(&mut world);
        vector_claim.release(&mut world);
        map_claim.release(&mut world);
        drop(retained_scalar);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn claimless_materialization_stays_out_of_runnable_membership_registration_ownership_observability() {
        use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let unused_runs = Arc::new(AtomicUsize::new(0));
        let terminal_runs = Arc::new(AtomicUsize::new(0));
        let materializer = crate::ScalarSignalMaterializer::<i32>::new({
            let unused_runs = unused_runs.clone();
            let terminal_runs = terminal_runs.clone();
            move |registrar| {
                let _unused: crate::ScalarSignalEntityRef<'_, i32> = registrar.signal_entity(
                    (),
                    move |In(())| {
                        unused_runs.fetch_add(1, AtomicOrdering::SeqCst);
                        Some(1)
                    },
                    OutputDelivery::broadcast(),
                )?;
                registrar.signal_entity(
                    (),
                    move |In(())| {
                        terminal_runs.fetch_add(1, AtomicOrdering::SeqCst);
                        Some(2)
                    },
                    OutputDelivery::broadcast(),
                )
            }
        });
        let owner = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let terminal =
            materialize_materializer(&mut world, materializer).expect("claimless materialization should succeed");
        let schedule = world.resource::<SignalGraphState>().default_schedule;
        process_signal_graph_for_schedule(schedule)(&mut world);

        assert_eq!(unused_runs.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(terminal_runs.load(AtomicOrdering::SeqCst), 0);
        let snapshot = graph_snapshot(&mut world);
        assert!(snapshot.nodes.iter().all(|node| node.registrar_inactive));
        assert!(snapshot.nodes.iter().all(|node| node.cached_level.is_none()));

        let claim = owner
            .try_register(&mut world)
            .expect("a claim should activate the ready terminal");
        process_signal_graph_for_schedule(schedule)(&mut world);
        assert_eq!(unused_runs.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(terminal_runs.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(claim.entity(), terminal);
        claim.release(&mut world);
    }

    #[test]
    fn count_zero_schedule_bucket_corruption_rejects_atomically_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let entity = materialize_materializer(&mut world, materializer)
            .expect("claimless fixture should materialize without activation");
        let attempt = retained.shared_clone();
        let schedule = world.resource::<SignalGraphState>().default_schedule;
        world
            .resource_mut::<SignalGraphState>()
            .by_schedule
            .entry(schedule)
            .or_default()
            .push(vec![entity]);

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, entity);
    }

    #[test]
    fn missing_immediate_active_route_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let (retained, _claim, source, terminal) = active_unary_registration_fixture(&mut world);
        let attempt = retained.shared_clone();
        world
            .get_mut::<ActiveSignalDependents>(*source)
            .expect("active unary source should have an active route")
            .0
            .remove(&terminal);

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, source);
    }

    #[test]
    fn active_inactive_marker_corruption_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let (retained, _claim, _source, terminal) = active_unary_registration_fixture(&mut world);
        let attempt = retained.shared_clone();
        world.entity_mut(*terminal).insert(RegistrarInactive);

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, terminal);
    }

    #[test]
    fn wrong_effective_schedule_bucket_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let (retained, _claim, source, terminal) = active_unary_registration_fixture(&mut world);
        let attempt = retained.shared_clone();
        let wrong_schedule = RegistrarScheduleA.intern();
        let expected_level = world
            .resource::<SignalGraphState>()
            .levels
            .get(&terminal)
            .copied()
            .expect("active unary terminal should have a cached level");
        {
            let mut graph = world.resource_mut::<SignalGraphState>();
            graph.register_schedule(wrong_schedule);
            for levels in graph.by_schedule.values_mut() {
                for members in levels {
                    members.retain(|member| *member != terminal);
                }
            }
            let wrong_levels = graph.by_schedule.entry(wrong_schedule).or_default();
            let level = expected_level as usize;
            if wrong_levels.len() <= level {
                wrong_levels.resize_with(level + 1, Vec::new);
            }
            wrong_levels[level].push(terminal);
        }

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, source);
    }

    #[test]
    fn replay_readiness_relationship_corruption_rejects_atomically_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<()>::new(|registrar| {
            let source = registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())?;
            registrar.signal_entity(source, |In(_value): In<i32>| None::<()>, OutputDelivery::broadcast())
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let source = world
            .get::<StaticSignalDependencies>(*terminal)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("replay fixture should have one canonical dependency");
        let attempt = retained.shared_clone();
        world
            .get_mut::<ReplayConsumingRelationships>(*source)
            .expect("ready replay dependency should have a published relationship")
            .0
            .insert(terminal, ReplayRelationshipState::Pending);

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, source);
    }

    #[test]
    fn missing_mandatory_runtime_components_reject_atomically_registration_ownership_observability() {
        assert_missing_runtime_component_rejects::<SystemRunner>();
        assert_missing_runtime_component_rejects::<StoredOutputDelivery>();
        assert_missing_runtime_component_rejects::<SignalInputBuffer>();
        assert_missing_runtime_component_rejects::<MaterializedSignalDependents>();
        assert_missing_runtime_component_rejects::<SignalTerminalRetainers>();
        assert_missing_runtime_component_rejects::<ActiveSignalDependents>();
    }

    #[test]
    fn replay_metadata_without_matching_typed_cache_rejects_atomically_registration_ownership_observability() {
        let mut missing_world = lazy_cleanup_world();
        let missing = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let missing_attempt = missing.shared_clone();
        let _missing_retained = missing.shared_clone();
        let _missing_claim = missing.register(&mut missing_world);
        let missing_terminal = _missing_claim.entity();
        missing_world
            .entity_mut(*missing_terminal)
            .remove::<crate::replay::ScalarReplayCache<i32>>();
        assert_runtime_inconsistency_is_atomic(&mut missing_world, missing_attempt, missing_terminal);

        let mut wrong_world = lazy_cleanup_world();
        let wrong = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let wrong_attempt = wrong.shared_clone();
        let _wrong_retained = wrong.shared_clone();
        let _wrong_claim = wrong.register(&mut wrong_world);
        let wrong_terminal = _wrong_claim.entity();
        wrong_world
            .entity_mut(*wrong_terminal)
            .remove::<crate::replay::ScalarReplayCache<i32>>()
            .insert(crate::replay::VecReplayCache::<i32>::ready(Vec::new()));
        assert_runtime_inconsistency_is_atomic(&mut wrong_world, wrong_attempt, wrong_terminal);
    }

    #[test]
    fn scalar_and_vector_replay_contract_mismatches_reject_atomically_registration_ownership_observability() {
        let mut scalar_world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let scalar_attempt = scalar.shared_clone();
        let _scalar_retained = scalar.shared_clone();
        let scalar_claim = scalar.register(&mut scalar_world);
        let scalar_entity = scalar_claim.entity();
        scalar_world
            .get_mut::<RegistrationReplayCacheState>(*scalar_entity)
            .expect("scalar replay metadata should exist")
            .contract = crate::registration::vec_replay_runtime_contract::<i32>();
        assert_runtime_inconsistency_is_atomic(&mut scalar_world, scalar_attempt, scalar_entity);
        run_signal_node(&mut scalar_world, scalar_entity);
        assert!(matches!(
            scalar_world
                .get::<crate::replay::ScalarReplayCache<i32>>(*scalar_entity)
                .expect("scalar cache should remain authoritative")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready { current: 2, .. }
        ));

        let mut vector_world = lazy_cleanup_world();
        let vector = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_source(
                Vec::new(),
                |In(())| Some(Vec::<crate::signal_vec::VecDiff<i32>>::new()),
                OutputDelivery::broadcast(),
            )
        });
        let vector_attempt = vector.shared_clone();
        let _vector_retained = vector.shared_clone();
        let vector_claim = vector.register(&mut vector_world);
        let vector_entity = vector_claim.entity();
        vector_world
            .get_mut::<RegistrationReplayCacheState>(*vector_entity)
            .expect("vector replay metadata should exist")
            .contract = crate::registration::scalar_replay_runtime_contract::<i32>();
        assert_runtime_inconsistency_is_atomic(&mut vector_world, vector_attempt, vector_entity);
    }

    #[test]
    fn split_readiness_and_snapshot_contract_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let attempt = scalar.shared_clone();
        let _retained = scalar.shared_clone();
        let claim = scalar.register(&mut world);
        let entity = claim.entity();
        world
            .entity_mut(*entity)
            .insert(crate::replay::VecReplayCache::<i32>::ready(vec![9]));
        let incompatible: Arc<dyn ReplayRuntimeContract> = Arc::new(ScalarReadinessVectorSnapshotContract);
        world
            .get_mut::<SystemRunner>(*entity)
            .expect("scalar replay runtime should exist")
            .replay_contract = Some(incompatible.clone());
        world
            .get_mut::<RegistrationReplayCacheState>(*entity)
            .expect("scalar replay metadata should exist")
            .contract = incompatible;

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, entity);
        run_signal_node(&mut world, entity);
        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<i32>>(*entity)
                .expect("scalar cache should remain runnable after rejection")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready { current: 2, .. }
        ));
    }

    #[test]
    fn malformed_ready_snapshot_payload_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let attempt = scalar.shared_clone();
        let _retained = scalar.shared_clone();
        let claim = scalar.register(&mut world);
        let entity = claim.entity();
        let malformed: Arc<dyn ReplayRuntimeContract> = Arc::new(MalformedScalarSnapshotContract);
        world
            .get_mut::<SystemRunner>(*entity)
            .expect("scalar replay runtime should exist")
            .replay_contract = Some(malformed.clone());
        world
            .get_mut::<RegistrationReplayCacheState>(*entity)
            .expect("scalar replay metadata should exist")
            .contract = malformed;

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, entity);
    }

    #[test]
    fn runtime_and_metadata_contract_identity_mismatch_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let attempt = scalar.shared_clone();
        let _retained = scalar.shared_clone();
        let claim = scalar.register(&mut world);
        let entity = claim.entity();
        world
            .get_mut::<SystemRunner>(*entity)
            .expect("scalar replay runtime should exist")
            .replay_contract = Some(crate::registration::scalar_replay_runtime_contract::<i32>());

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, entity);
    }

    #[test]
    fn wrong_contract_with_both_typed_caches_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let attempt = scalar.shared_clone();
        let _retained = scalar.shared_clone();
        let claim = scalar.register(&mut world);
        let entity = claim.entity();
        world
            .entity_mut(*entity)
            .insert(crate::replay::VecReplayCache::<i32>::ready(Vec::new()));
        let wrong_contract = crate::registration::vec_replay_runtime_contract::<i32>();
        world
            .get_mut::<SystemRunner>(*entity)
            .expect("scalar replay runtime should exist")
            .replay_contract = Some(wrong_contract.clone());
        world
            .get_mut::<RegistrationReplayCacheState>(*entity)
            .expect("scalar replay metadata should exist")
            .contract = wrong_contract;

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, entity);
    }

    #[test]
    fn scalar_vector_event_collision_requires_cache_type_anchor_registration_ownership_observability() {
        type CollidingEvent = Vec<crate::signal_vec::VecDiff<i32>>;

        let mut world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<CollidingEvent>::new(|registrar| {
            registrar.replayable_signal_source(Vec::new(), |In(())| Some(Vec::new()), OutputDelivery::broadcast())
        });
        let attempt = scalar.shared_clone();
        let _retained = scalar.shared_clone();
        let claim = scalar.register(&mut world);
        let entity = claim.entity();
        world
            .entity_mut(*entity)
            .insert(crate::replay::VecReplayCache::<i32>::ready(Vec::new()));
        let wrong_contract = crate::registration::vec_replay_runtime_contract::<i32>();
        let runtime_key = world
            .get::<SystemRunner>(*entity)
            .and_then(|runner| runner.replay_key)
            .expect("scalar replay runtime should retain its fixed key");
        assert_eq!(runtime_key.event_type(), wrong_contract.event_type());
        assert_ne!(runtime_key.cache_type(), wrong_contract.cache_type());
        world
            .get_mut::<SystemRunner>(*entity)
            .expect("scalar replay runtime should exist")
            .replay_contract = Some(wrong_contract.clone());
        world
            .get_mut::<RegistrationReplayCacheState>(*entity)
            .expect("scalar replay metadata should exist")
            .contract = wrong_contract;

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, entity);
    }

    #[test]
    fn coherent_warming_and_ready_replay_contracts_pass_preflight_registration_ownership_observability() {
        let mut warming_world = lazy_cleanup_world();
        let warming = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(())| Some(1),
                crate::ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        });
        let warming_attempt = warming.shared_clone();
        let warming_claim = warming.register(&mut warming_world);
        let warming_entity = warming_claim.entity();
        assert!(matches!(
            inspect_coherent_replay_contract(
                &warming_world,
                warming_entity,
                TypeId::of::<crate::replay::ScalarReplayCache<i32>>(),
                TypeId::of::<i32>(),
            ),
            ErasedReplaySnapshot::Warming
        ));
        let second_warming_claim = warming_attempt
            .try_register(&mut warming_world)
            .expect("a coherent warming replay contract should pass existing-runtime preflight");
        second_warming_claim.release(&mut warming_world);
        warming_claim.release(&mut warming_world);

        let mut ready_world = lazy_cleanup_world();
        let ready = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(7, |In(())| Some(8), OutputDelivery::broadcast())
        });
        let ready_attempt = ready.shared_clone();
        let ready_claim = ready.register(&mut ready_world);
        let ready_entity = ready_claim.entity();
        let ErasedReplaySnapshot::Ready { snapshot, revision } = inspect_coherent_replay_contract(
            &ready_world,
            ready_entity,
            TypeId::of::<crate::replay::ScalarReplayCache<i32>>(),
            TypeId::of::<i32>(),
        ) else {
            panic!("stored-state scalar replay source should be ready");
        };
        assert_eq!(snapshot.downcast_ref::<i32>(), Some(&7));
        assert_eq!(revision, ReplayRevision::FIRST);
        let second_ready_claim = ready_attempt
            .try_register(&mut ready_world)
            .expect("a coherent ready scalar replay contract should pass existing-runtime preflight");
        second_ready_claim.release(&mut ready_world);
        ready_claim.release(&mut ready_world);

        let mut vector_world = lazy_cleanup_world();
        let vector = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_source(
                vec![1],
                |In(())| None::<Vec<crate::signal_vec::VecDiff<i32>>>,
                OutputDelivery::broadcast(),
            )
        });
        let vector_attempt = vector.shared_clone();
        let vector_claim = vector.register(&mut vector_world);
        let vector_entity = vector_claim.entity();
        let ErasedReplaySnapshot::Ready { snapshot, revision } = inspect_coherent_replay_contract(
            &vector_world,
            vector_entity,
            TypeId::of::<crate::replay::VecReplayCache<i32>>(),
            TypeId::of::<Vec<crate::signal_vec::VecDiff<i32>>>(),
        ) else {
            panic!("stored-state vector replay source should be ready");
        };
        assert_eq!(
            snapshot.downcast_ref::<Vec<crate::signal_vec::VecDiff<i32>>>(),
            Some(&vec![crate::signal_vec::VecDiff::Replace { values: vec![1] }])
        );
        assert_eq!(revision, ReplayRevision::FIRST);
        let second_vector_claim = vector_attempt
            .try_register(&mut vector_world)
            .expect("a coherent ready vector replay contract should pass existing-runtime preflight");
        second_vector_claim.release(&mut vector_world);
        vector_claim.release(&mut vector_world);

        let mut map_world = lazy_cleanup_world();
        let map = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            registrar.replayable_signal_map_source(
                BTreeMap::from([(1, 2)]),
                |In(())| None::<Vec<crate::signal_map::MapDiff<i32, i32>>>,
                OutputDelivery::broadcast(),
            )
        });
        let map_attempt = map.shared_clone();
        let map_claim = map.register(&mut map_world);
        let map_entity = map_claim.entity();
        let ErasedReplaySnapshot::Ready { snapshot, revision } = inspect_coherent_replay_contract(
            &map_world,
            map_entity,
            TypeId::of::<crate::replay::MapReplayCache<i32, i32>>(),
            TypeId::of::<Vec<crate::signal_map::MapDiff<i32, i32>>>(),
        ) else {
            panic!("stored-state map replay source should be ready");
        };
        assert_eq!(
            snapshot.downcast_ref::<Vec<crate::signal_map::MapDiff<i32, i32>>>(),
            Some(&vec![crate::signal_map::MapDiff::Replace { entries: vec![(1, 2)] }])
        );
        assert_eq!(revision, ReplayRevision::FIRST);
        let second_map_claim = map_attempt
            .try_register(&mut map_world)
            .expect("a coherent ready map replay contract should pass existing-runtime preflight");
        second_map_claim.release(&mut map_world);
        map_claim.release(&mut map_world);
    }

    #[test]
    fn retained_replay_terminal_missing_maintenance_rejects_atomically_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let replay = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let retained = replay.shared_clone();
        let attempt = retained.shared_clone();
        let _claim = replay.register(&mut world);
        let terminal = _claim.entity();
        world.entity_mut(*terminal).remove::<ReplayMaintenanceClaim>();

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, terminal);
    }

    #[test]
    fn structural_reverse_link_corruption_rejects_atomically_registration_ownership_observability() {
        let mut missing_world = lazy_cleanup_world();
        let (missing_retained, _claim, source, terminal) = active_unary_registration_fixture(&mut missing_world);
        let missing_attempt = missing_retained.shared_clone();
        missing_world
            .get_mut::<MaterializedSignalDependents>(*source)
            .expect("canonical dependency should retain its structural dependent")
            .0
            .remove(&terminal);
        assert_runtime_inconsistency_is_atomic(&mut missing_world, missing_attempt, source);

        let mut dangling_world = lazy_cleanup_world();
        let (dangling_retained, _claim, source, terminal) = active_unary_registration_fixture(&mut dangling_world);
        let dangling_attempt = dangling_retained.shared_clone();
        let dangling = SignalEntity::from(dangling_world.spawn_empty().id());
        dangling_world
            .get_mut::<MaterializedSignalDependents>(*source)
            .expect("canonical dependency should retain its structural dependent")
            .0
            .insert(dangling);
        assert_runtime_inconsistency_is_atomic(&mut dangling_world, dangling_attempt, source);
        assert!(
            dangling_world
                .get::<StaticSignalDependencies>(*terminal)
                .is_some_and(|dependencies| dependencies.0.contains(&source))
        );
    }

    #[test]
    fn side_only_corrupted_existing_entity_rejects_atomically_registration_ownership_observability() {
        let existing = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let side_owner = existing.shared_clone();
        let _retained_owner = existing.shared_clone();
        let mut world = lazy_cleanup_world();
        let existing_entity =
            materialize_materializer(&mut world, existing).expect("side fixture should materialize without a claim");
        world.entity_mut(*existing_entity).remove::<SystemRunner>();
        let attempt = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let _side = registrar.materialize_signal(side_owner)?;
            registrar.signal_entity((), |In(())| Some(2), OutputDelivery::broadcast())
        });

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, existing_entity);
    }

    #[test]
    fn schedule_only_corrupted_existing_entity_rejects_atomically_registration_ownership_observability() {
        let existing = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let schedule_owner = existing.shared_clone();
        let _retained_owner = existing.shared_clone();
        let mut world = lazy_cleanup_world();
        world
            .resource_mut::<SignalGraphState>()
            .register_schedule(RegistrarScheduleA.intern());
        let existing_entity = materialize_materializer(&mut world, existing)
            .expect("schedule fixture should materialize without a claim");
        world.entity_mut(*existing_entity).remove::<SignalInputBuffer>();
        let attempt = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let scheduled = registrar.materialize_signal(schedule_owner)?;
            let _scheduled = registrar.apply_schedule(scheduled, RegistrarScheduleA)?;
            registrar.signal_entity((), |In(())| Some(2), OutputDelivery::broadcast())
        });

        assert_runtime_inconsistency_is_atomic(&mut world, attempt, existing_entity);
    }

    #[test]
    fn recursive_claim_overflow_rejects_without_mutation_registration_ownership_observability() {
        let source_materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let source_for_terminal = source_materializer.shared_clone();
        let _source_retainer = source_for_terminal.shared_clone();
        let mut world = lazy_cleanup_world();
        let source = materialize_materializer(&mut world, source_materializer)
            .expect("source should materialize without a claim");
        world.get_mut::<SignalClaimCount>(*source).unwrap().0 = usize::MAX;
        let before = graph_snapshot(&mut world);
        let entity_count_before = world.query::<Entity>().iter(&world).count();
        let terminal = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let source = registrar.materialize_signal(source_for_terminal)?;
            registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::broadcast())
        });

        assert!(matches!(
            terminal.try_register(&mut world),
            Err(SignalRegistrationError::ClaimCountOverflow)
        ));
        assert_eq!(world.query::<Entity>().iter(&world).count(), entity_count_before);
        assert_eq!(graph_snapshot(&mut world), before);
    }

    #[test]
    fn late_first_materialization_rejection_publishes_nothing_registration_ownership_observability() {
        use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let initializer_calls = Arc::new(AtomicUsize::new(0));
        let system_runs = Arc::new(AtomicUsize::new(0));
        let child = crate::ScalarSignalMaterializer::<i32>::new({
            let initializer_calls = initializer_calls.clone();
            let system_runs = system_runs.clone();
            move |registrar| {
                initializer_calls.fetch_add(1, AtomicOrdering::SeqCst);
                let source = registrar.signal_entity(
                    (),
                    move |In(())| {
                        system_runs.fetch_add(1, AtomicOrdering::SeqCst);
                        Some(1)
                    },
                    OutputDelivery::broadcast(),
                )?;
                let intermediate = registrar.signal_entity(
                    source,
                    |In(value): In<i32>| Some(value + 1),
                    OutputDelivery::broadcast(),
                )?;
                registrar.replayable_signal_entity(
                    intermediate,
                    |In(value): In<i32>| Some(value + 1),
                    crate::ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            }
        });
        let later_child = child.shared_clone();
        let outer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let terminal = registrar.materialize_signal(child)?;
            registrar.apply_schedule(terminal, UnregisteredMaterializationSchedule)
        });
        let later_outer = outer.shared_clone();
        let mut world = lazy_cleanup_world();
        let before = graph_snapshot(&mut world);
        let entity_count_before = world.query::<Entity>().iter(&world).count();
        let inbox_before = cleanup::queued_message_count(&world);

        assert!(matches!(
            outer.try_register(&mut world),
            Err(SignalRegistrationError::ScheduleNotRegistered)
        ));

        assert_eq!(initializer_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(system_runs.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(world.query::<Entity>().iter(&world).count(), entity_count_before);
        assert_eq!(graph_snapshot(&mut world), before);
        assert_eq!(cleanup::queued_message_count(&world), inbox_before);
        assert!(matches!(
            later_child.try_register(&mut world),
            Err(SignalRegistrationError::SharedMaterializerRejected)
        ));
        assert!(matches!(
            later_outer.try_register(&mut world),
            Err(SignalRegistrationError::SharedMaterializerRejected)
        ));
        assert_eq!(world.query::<Entity>().iter(&world).count(), entity_count_before);
        assert_eq!(graph_snapshot(&mut world), before);
        assert_eq!(cleanup::queued_message_count(&world), inbox_before);
    }

    #[test]
    fn rejected_enclosing_plan_handles_ready_child_owner_release_locally_registration_ownership_observability() {
        let child = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let child_for_parent = child.shared_clone();
        let mut world = lazy_cleanup_world();
        let child_claim = child.register(&mut world);
        let child_entity = child_claim.entity();
        child_claim.release(&mut world);
        assert!(world.get_entity(*child_entity).is_ok());
        assert_eq!(cleanup::queued_message_count(&world), 0);
        let outer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let child = registrar.materialize_signal(child_for_parent)?;
            registrar.apply_schedule(child, UnregisteredMaterializationSchedule)
        });

        assert!(matches!(
            outer.try_register(&mut world),
            Err(SignalRegistrationError::ScheduleNotRegistered)
        ));

        assert_eq!(cleanup::queued_message_count(&world), 0);
        assert!(world.get_entity(*child_entity).is_err());
    }

    #[test]
    fn rejected_enclosing_plan_preserves_ready_child_with_live_sibling_registration_ownership_observability() {
        let child = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let child_for_parent = child.shared_clone();
        let retained_child = child.shared_clone();
        let mut world = lazy_cleanup_world();
        let child_claim = child.register(&mut world);
        let child_entity = child_claim.entity();
        child_claim.release(&mut world);
        let before = graph_snapshot(&mut world);
        let outer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let child = registrar.materialize_signal(child_for_parent)?;
            registrar.apply_schedule(child, UnregisteredMaterializationSchedule)
        });

        assert!(matches!(
            outer.try_register(&mut world),
            Err(SignalRegistrationError::ScheduleNotRegistered)
        ));

        assert_eq!(graph_snapshot(&mut world), before);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        let recovered = retained_child.shared_clone().register(&mut world);
        assert_eq!(recovered.entity(), child_entity);
        recovered.release(&mut world);
        drop(retained_child);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(*child_entity).is_err());
    }

    #[test]
    fn ready_materializer_survives_claim_overflow_rejection_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        });
        let rejected_owner = materializer.shared_clone();
        let recovered_owner = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let signal = materialize_materializer(&mut world, materializer)
            .expect("materializer should become ready without acquiring a claim");
        world.get_mut::<SignalClaimCount>(*signal).unwrap().0 = usize::MAX;
        let before = graph_snapshot(&mut world);

        assert!(matches!(
            rejected_owner.try_register(&mut world),
            Err(SignalRegistrationError::ClaimCountOverflow)
        ));
        assert_eq!(graph_snapshot(&mut world), before);

        world.get_mut::<SignalClaimCount>(*signal).unwrap().0 = 0;
        let claim = recovered_owner
            .try_register(&mut world)
            .expect("a ready materializer should remain usable after a later acquisition rejection");
        assert_eq!(claim.entity(), signal);
        assert_eq!(**world.get::<SignalClaimCount>(*signal).unwrap(), 1);
        claim.release(&mut world);
    }

    #[test]
    fn custom_cache_wrappers_add_one_entity_and_are_graph_idempotent_registration_ownership_observability() {
        use crate::{signal::SignalExt, signal_map::SignalMapExt, signal_vec::SignalVecExt};

        let mut scalar_world = lazy_cleanup_world();
        let scalar_claim = CustomScalarSource.cache().cache().register(&mut scalar_world);
        assert_eq!(graph_snapshot(&mut scalar_world).nodes.len(), 2);
        assert!(
            scalar_world
                .get::<crate::replay::ScalarReplayCache<i32>>(*scalar_claim.entity())
                .is_some()
        );

        let mut vector_world = lazy_cleanup_world();
        let vector_claim = CustomVectorSource.cache().cache().register(&mut vector_world);
        assert_eq!(graph_snapshot(&mut vector_world).nodes.len(), 2);
        assert!(
            vector_world
                .get::<crate::replay::VecReplayCache<i32>>(*vector_claim.entity())
                .is_some()
        );

        let mut map_world = lazy_cleanup_world();
        let map_claim = CustomMapSource.cache().cache().register(&mut map_world);
        assert_eq!(graph_snapshot(&mut map_world).nodes.len(), 2);
        assert!(
            map_world
                .get::<crate::replay::MapReplayCache<i32, i32>>(*map_claim.entity())
                .is_some()
        );

        scalar_claim.release(&mut scalar_world);
        vector_claim.release(&mut vector_world);
        map_claim.release(&mut map_world);
    }

    #[test]
    fn graph_snapshot_ordering_is_deterministic_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source_a = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let source_b = registrar.signal_entity((), |In(())| Some(2), OutputDelivery::move_only())?;
            registrar.signal_entity(
                [source_b, source_a, source_b],
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        });
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let expected_dependencies = world
            .get::<StaticSignalDependencies>(*terminal)
            .expect("registered terminal should have canonical dependencies")
            .0
            .iter()
            .map(|dependency| dependency.entity())
            .collect::<Vec<_>>();

        let first = graph_snapshot(&mut world);
        let second = graph_snapshot(&mut world);
        assert_eq!(first, second, "capturing unchanged graph state must be deterministic");

        let mut expected_entities = expected_dependencies.clone();
        expected_entities.push(terminal.entity());
        expected_entities.sort();
        assert_eq!(
            first.nodes.iter().map(|node| node.entity).collect::<Vec<_>>(),
            expected_entities,
            "snapshot rows must use full generational entity ordering"
        );

        let terminal_snapshot = first
            .node(terminal.entity())
            .expect("registered terminal should be present in the graph snapshot");
        assert_eq!(
            terminal_snapshot.canonical_static_dependencies.as_deref(),
            Some(expected_dependencies.as_slice())
        );
        assert_eq!(terminal_snapshot.claim_count, Some(1));
        assert!(terminal_snapshot.has_system_runner);
        assert_eq!(terminal_snapshot.pending_input_count, Some(0));

        claim.release(&mut world);
    }

    #[test]
    fn canonical_structure_is_deterministic_and_idempotent_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let source_materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let source_for_terminal = source_materializer.shared_clone();
        let source =
            materialize_materializer(&mut world, source_materializer).expect("source materialization should succeed");
        let terminal_materializer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let source = registrar.materialize_signal(source_for_terminal)?;
            registrar.signal_entity(
                [source, source],
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        });
        let retained_terminal = terminal_materializer.shared_clone();
        let terminal = materialize_materializer(&mut world, terminal_materializer)
            .expect("terminal materialization should succeed");

        let snapshot = graph_snapshot(&mut world);
        let source_snapshot = snapshot.node(source.entity()).expect("source snapshot should exist");
        let terminal_snapshot = snapshot
            .node(terminal.entity())
            .expect("terminal snapshot should exist");
        assert_eq!(source_snapshot.canonical_static_dependencies, Some(Vec::new()));
        assert_eq!(source_snapshot.materialized_dependents, vec![terminal.entity()]);
        assert_eq!(
            terminal_snapshot.canonical_static_dependencies,
            Some(vec![source.entity()])
        );
        assert_eq!(terminal_snapshot.claim_count, Some(0));
        assert_eq!(source_snapshot.claim_count, Some(0));

        let before = graph_snapshot(&mut world);
        verify_existing_canonical_dependencies(&mut world, terminal, [source, source])
            .expect("equivalent repeated dependencies should canonicalize idempotently");
        assert_eq!(graph_snapshot(&mut world), before);
        drop(retained_terminal);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn conflicting_canonical_state_rejects_without_mutation_registration_ownership_characterization() {
        let mut world = lazy_cleanup_world();
        let source_a_materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let source_a_for_terminal = source_a_materializer.shared_clone();
        let source_a = materialize_materializer(&mut world, source_a_materializer)
            .expect("first source materialization should succeed");
        let source_b_materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(2), OutputDelivery::move_only())
        });
        let retained_source_b = source_b_materializer.shared_clone();
        let source_b = materialize_materializer(&mut world, source_b_materializer)
            .expect("second source materialization should succeed");
        let terminal_materializer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let source = registrar.materialize_signal(source_a_for_terminal)?;
            registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
        });
        let retained_terminal = terminal_materializer.shared_clone();
        let terminal = materialize_materializer(&mut world, terminal_materializer)
            .expect("terminal materialization should succeed");

        let before = graph_snapshot(&mut world);
        let entity_count_before = world.query::<Entity>().iter(&world).count();
        let result = verify_existing_canonical_dependencies(&mut world, terminal, [source_b]);

        assert_eq!(result, Err(SignalRegistrationError::ConflictingCanonicalState));
        assert_eq!(world.query::<Entity>().iter(&world).count(), entity_count_before);
        assert_eq!(graph_snapshot(&mut world), before);
        assert_eq!(
            before
                .node(terminal.entity())
                .expect("committed terminal should remain")
                .canonical_static_dependencies,
            Some(vec![source_a.entity()])
        );
        drop(retained_terminal);
        drop(retained_source_b);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn retained_builtin_source_reuses_live_identity_then_reaps_registration_ownership_observability() {
        use crate::signal::{self, SignalExt};

        let mut world = lazy_cleanup_world();
        let source = signal::from_system(|In(())| -> i32 { 1 }).broadcast();
        let first_claim = source.clone().register(&mut world);
        let entity = first_claim.entity();
        first_claim.release(&mut world);

        assert!(world.get_entity(*entity).is_ok());
        assert_eq!(world.get::<SignalClaimCount>(*entity).map(|count| **count), Some(0));
        assert!(world.get::<RegistrarInactive>(*entity).is_some());

        let second_claim = source
            .clone()
            .try_register(&mut world)
            .expect("retained source should reactivate");
        assert_eq!(second_claim.entity(), entity);
        second_claim.release(&mut world);
        drop(source);
        cleanup::drain_world_cleanup(&mut world);

        assert!(world.get_entity(*entity).is_err());
    }

    #[test]
    fn native_final_external_drop_targets_only_originating_world_registration_ownership_observability() {
        let materializer_a = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained_a = materializer_a.shared_clone();
        let materializer_b = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(2), OutputDelivery::move_only())
        });
        let retained_b = materializer_b.shared_clone();
        let mut world_a = lazy_cleanup_world();
        let mut world_b = lazy_cleanup_world();
        let claim_a = materializer_a.register(&mut world_a);
        let signal_a = claim_a.entity();
        let claim_b = materializer_b.register(&mut world_b);
        let signal_b = claim_b.entity();
        assert_eq!(signal_a.entity(), signal_b.entity());
        claim_a.release(&mut world_a);
        claim_b.release(&mut world_b);

        drop(retained_a);

        assert_eq!(cleanup::queued_message_count(&world_a), 1);
        assert_eq!(cleanup::queued_message_count(&world_b), 0);
        cleanup::drain_world_cleanup(&mut world_b);
        assert!(world_b.get_entity(*signal_b).is_ok());
        cleanup::drain_world_cleanup(&mut world_a);
        assert!(world_a.get_entity(*signal_a).is_err());
        assert!(world_b.get_entity(*signal_b).is_ok());

        drop(retained_b);
        cleanup::drain_world_cleanup(&mut world_b);
        assert!(world_b.get_entity(*signal_b).is_err());
    }

    #[test]
    fn native_final_drop_after_originating_world_destruction_is_safe_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        claim.release(&mut world);

        drop(world);
        drop(retained);
    }

    #[test]
    fn ready_replay_relationship_targets_only_new_dependent_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let source = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let first_source = source.shared_clone();
        let second_source = source.shared_clone();
        let retained_source = source.shared_clone();
        let source_claim = source.register(&mut world);
        let source_entity = source_claim.entity();

        let first_values = Arc::new(std::sync::Mutex::new(Vec::new()));
        let first_claim = crate::ScalarSignalMaterializer::<()>::new({
            let first_values = first_values.clone();
            move |registrar| {
                let source = registrar.materialize_signal(first_source)?;
                registrar.signal_entity(
                    source,
                    move |In(value): In<i32>| {
                        first_values
                            .lock()
                            .expect("first replay observer mutex poisoned")
                            .push(value);
                        None::<()>
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(&mut world);
        run_signal_node(&mut world, first_claim.entity());
        assert_eq!(
            *first_values.lock().expect("first replay observer mutex poisoned"),
            vec![1]
        );

        let second_values = Arc::new(std::sync::Mutex::new(Vec::new()));
        let second_claim = crate::ScalarSignalMaterializer::<()>::new({
            let second_values = second_values.clone();
            move |registrar| {
                let source = registrar.materialize_signal(second_source)?;
                registrar.signal_entity(
                    source,
                    move |In(value): In<i32>| {
                        second_values
                            .lock()
                            .expect("second replay observer mutex poisoned")
                            .push(value);
                        None::<()>
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(&mut world);

        run_signal_node(&mut world, first_claim.entity());
        run_signal_node(&mut world, second_claim.entity());
        assert_eq!(
            *first_values.lock().expect("first replay observer mutex poisoned"),
            vec![1]
        );
        assert_eq!(
            *second_values.lock().expect("second replay observer mutex poisoned"),
            vec![1]
        );

        run_signal_node(&mut world, source_entity);
        run_signal_node(&mut world, first_claim.entity());
        run_signal_node(&mut world, second_claim.entity());
        assert_eq!(
            *first_values.lock().expect("first replay observer mutex poisoned"),
            vec![1, 2]
        );
        assert_eq!(
            *second_values.lock().expect("second replay observer mutex poisoned"),
            vec![1, 2]
        );
        let relationships = world
            .get::<ReplayConsumingRelationships>(*source_entity)
            .expect("ready replay source should retain relationship cursors");
        assert!(relationships.0.values().all(|relationship| {
            matches!(
                relationship,
                ReplayRelationshipState::Published { cursor } if cursor.get() == 2
            )
        }));

        first_claim.release(&mut world);
        second_claim.release(&mut world);
        source_claim.release(&mut world);
        drop(retained_source);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn replay_relationship_cursors_suppress_represented_history_independently_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let source_claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        })
        .register(&mut world);
        let source = source_claim.entity();
        let first = SignalEntity::from(world.spawn(SignalInputBuffer::default()).id());
        let second = SignalEntity::from(world.spawn(SignalInputBuffer::default()).id());
        let cursor_free = SignalEntity::from(world.spawn(SignalInputBuffer::default()).id());
        world.entity_mut(*source).insert((
            ActiveSignalDependents(HashSet::from([first, second, cursor_free])),
            ReplayConsumingRelationships(BTreeMap::from([
                (
                    first,
                    ReplayRelationshipState::Published {
                        cursor: ReplayRevision::FIRST,
                    },
                ),
                (
                    second,
                    ReplayRelationshipState::Published {
                        cursor: ReplayRevision::FIRST,
                    },
                ),
            ])),
        ));

        let mut represented = Vec::new();
        deliver_accepted_signal_output(
            &mut world,
            source,
            AcceptedSignalOutput {
                output: Box::new(1_i32),
                replay_revision: Some(ReplayRevision::FIRST),
            },
            |_world, dependent, output| {
                represented.push((dependent, output));
                true
            },
        );
        assert_eq!(represented.len(), 1);
        assert_eq!(represented[0].0, cursor_free);
        assert_eq!(represented[0].1.downcast_ref::<i32>(), Some(&1));

        let update = world
            .get_mut::<crate::replay::ScalarReplayCache<i32>>(*source)
            .expect("ready replay source should retain its typed cache")
            .accept(2)
            .expect("test replay revision should not overflow");
        let crate::replay::ReplayCacheUpdate::Accepted { output, revision } = update else {
            unreachable!("scalar replay accepts every output");
        };
        world
            .get_mut::<ReplayConsumingRelationships>(*source)
            .expect("test source should retain relationship cursors")
            .0
            .insert(second, ReplayRelationshipState::Published { cursor: revision });

        let mut newer = Vec::new();
        deliver_accepted_signal_output(
            &mut world,
            source,
            AcceptedSignalOutput {
                output: Box::new(output),
                replay_revision: Some(revision),
            },
            |_world, dependent, output| {
                newer.push((dependent, output));
                true
            },
        );
        let delivered = newer.iter().map(|(dependent, _)| *dependent).collect::<BTreeSet<_>>();
        assert_eq!(delivered, BTreeSet::from([first, cursor_free]));
        assert!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .expect("test source should retain relationship cursors")
                .0
                .values()
                .all(|state| matches!(state, ReplayRelationshipState::Published { cursor } if *cursor == revision))
        );

        world.get_mut::<ActiveSignalDependents>(*source).unwrap().0.clear();
        world.entity_mut(*source).remove::<ReplayConsumingRelationships>();
        source_claim.release(&mut world);
    }

    #[test]
    fn warming_replay_relationship_publishes_on_first_output_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let source = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(())| Some(3),
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        let dependent_source = source.shared_clone();
        let source_claim = source.register(&mut world);
        let source_entity = source_claim.entity();
        let values = Arc::new(std::sync::Mutex::new(Vec::new()));
        let dependent_claim = crate::ScalarSignalMaterializer::<()>::new({
            let values = values.clone();
            move |registrar| {
                let source = registrar.materialize_signal(dependent_source)?;
                registrar.signal_entity(
                    source,
                    move |In(value): In<i32>| {
                        values
                            .lock()
                            .expect("warming replay observer mutex poisoned")
                            .push(value);
                        None::<()>
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(&mut world);

        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source_entity)
                .and_then(|relationships| relationships.0.get(&dependent_claim.entity())),
            Some(ReplayRelationshipState::Pending)
        ));
        assert!(
            !world
                .get::<ActiveSignalDependents>(*source_entity)
                .is_some_and(|dependents| dependents.0.contains(&dependent_claim.entity()))
        );
        run_signal_node(&mut world, dependent_claim.entity());
        assert!(
            values
                .lock()
                .expect("warming replay observer mutex poisoned")
                .is_empty()
        );

        run_signal_node(&mut world, source_entity);
        run_signal_node(&mut world, dependent_claim.entity());
        assert_eq!(*values.lock().expect("warming replay observer mutex poisoned"), vec![3]);
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source_entity)
                .and_then(|relationships| relationships.0.get(&dependent_claim.entity())),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));

        dependent_claim.release(&mut world);
        source_claim.release(&mut world);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn collection_snapshot_coalescing_preserves_cross_source_enqueue_order_registration_ownership_observability() {
        fn take_vector_batches(buffer: &mut SignalInputBuffer) -> Vec<Vec<crate::signal_vec::VecDiff<i32>>> {
            buffer
                .take()
                .into_iter()
                .map(|input| {
                    downcast_erased_signal_value::<Vec<crate::signal_vec::VecDiff<i32>>>(input)
                        .expect("vector replay input should retain its event type")
                })
                .collect()
        }

        for reverse_source_identity_order in [false, true] {
            let mut world = World::new();
            let first = SignalEntity::from(world.spawn_empty().id());
            let second = SignalEntity::from(world.spawn_empty().id());
            let (source_a, source_b) = if reverse_source_identity_order {
                (second, first)
            } else {
                (first, second)
            };
            assert_ne!(source_a, source_b);

            let mut buffer = SignalInputBuffer::default();
            buffer.push_replay_snapshot(
                source_a,
                Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![0] }]),
            );
            buffer.push_replay_snapshot(
                source_b,
                Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![10] }]),
            );
            buffer.push(Box::new(vec![crate::signal_vec::VecDiff::Push { value: 9 }]));
            buffer.push_replay_snapshot(
                source_a,
                Box::new(vec![
                    crate::signal_vec::VecDiff::Replace { values: vec![2] },
                    crate::signal_vec::VecDiff::Push { value: 3 },
                ]),
            );
            buffer.push_replay_snapshot(
                source_b,
                Box::new(vec![
                    crate::signal_vec::VecDiff::Replace { values: vec![20] },
                    crate::signal_vec::VecDiff::Push { value: 21 },
                ]),
            );

            let batches = take_vector_batches(&mut buffer);
            assert!(!buffer.has_pending_replay_snapshot(source_a));
            assert!(!buffer.has_pending_replay_snapshot(source_b));
            assert_eq!(
                batches,
                vec![
                    vec![crate::signal_vec::VecDiff::Push { value: 9 }],
                    vec![
                        crate::signal_vec::VecDiff::Replace { values: vec![2] },
                        crate::signal_vec::VecDiff::Push { value: 3 },
                    ],
                    vec![
                        crate::signal_vec::VecDiff::Replace { values: vec![20] },
                        crate::signal_vec::VecDiff::Push { value: 21 },
                    ],
                ]
            );
            let mut current = Vec::new();
            for diff in batches.into_iter().flatten() {
                diff.apply_to_vec(&mut current);
            }
            assert_eq!(current, vec![20, 21]);
        }
    }

    #[test]
    fn two_replay_sources_reposition_only_the_superseded_snapshot_registration_ownership_observability() {
        let mut world = World::new();
        let source_a = SignalEntity::from(world.spawn_empty().id());
        let source_b = SignalEntity::from(world.spawn_empty().id());
        let mut buffer = SignalInputBuffer::default();
        buffer.push_replay_snapshot(
            source_a,
            Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![0] }]),
        );
        buffer.push_replay_snapshot(
            source_b,
            Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![10] }]),
        );
        buffer.push_replay_snapshot(
            source_a,
            Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![2] }]),
        );
        buffer.push_replay_snapshot(
            source_b,
            Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![12] }]),
        );

        let batches = buffer
            .take()
            .into_iter()
            .map(|input| {
                downcast_erased_signal_value::<Vec<crate::signal_vec::VecDiff<i32>>>(input)
                    .expect("vector replay input should retain its event type")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            batches,
            vec![
                vec![crate::signal_vec::VecDiff::Replace { values: vec![2] }],
                vec![crate::signal_vec::VecDiff::Replace { values: vec![12] }],
            ]
        );
        assert!(!buffer.has_pending_replay_snapshot(source_a));
        assert!(!buffer.has_pending_replay_snapshot(source_b));
        let mut current = Vec::new();
        for diff in batches.into_iter().flatten() {
            diff.apply_to_vec(&mut current);
        }
        assert_eq!(current, vec![12]);
    }

    #[test]
    fn replay_snapshot_coalescing_uses_full_generational_source_identity_registration_ownership_observability() {
        let mut world = World::new();
        let stale_entity = world.spawn_empty().id();
        let next_gen = world.despawn_no_free(stale_entity).unwrap();
        let live_entity = world.spawn_empty_at(next_gen).unwrap().id();
        assert_eq!(stale_entity.index(), live_entity.index());
        assert_ne!(stale_entity.generation(), live_entity.generation());
        let stale_source = SignalEntity::from(stale_entity);
        let live_source = SignalEntity::from(live_entity);

        let mut buffer = SignalInputBuffer::default();
        buffer.push_replay_snapshot(stale_source, Box::new(1_i32));
        buffer.push_replay_snapshot(live_source, Box::new(2_i32));
        buffer.push_replay_snapshot(stale_source, Box::new(3_i32));
        assert!(buffer.has_pending_replay_snapshot(stale_source));
        assert!(buffer.has_pending_replay_snapshot(live_source));

        let values = buffer
            .take()
            .into_iter()
            .map(|input| downcast_erased_signal_value::<i32>(input).expect("snapshot should retain its value type"))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![2, 3]);
        assert!(!buffer.has_pending_replay_snapshot(stale_source));
        assert!(!buffer.has_pending_replay_snapshot(live_source));

        buffer.push_replay_snapshot(stale_source, Box::new(4_i32));
        buffer.push_replay_snapshot(live_source, Box::new(5_i32));
        buffer.clear();
        assert!(buffer.0.is_empty());
        assert!(!buffer.has_pending_replay_snapshot(stale_source));
        assert!(!buffer.has_pending_replay_snapshot(live_source));
    }

    #[test]
    fn polling_preserves_mixed_source_collection_order_registration_ownership_observability() {
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let current = Arc::new(std::sync::Mutex::new(Vec::new()));
        let materializer = crate::VecSignalMaterializer::<i32>::new({
            let observed = observed.clone();
            let current = current.clone();
            move |registrar| {
                let replay = registrar.replayable_signal_vec_source(
                    vec![0],
                    |In(())| {
                        Some(vec![
                            crate::signal_vec::VecDiff::Replace { values: vec![2] },
                            crate::signal_vec::VecDiff::Push { value: 3 },
                        ])
                    },
                    OutputDelivery::move_only(),
                )?;
                let ordinary = registrar.signal_vec_entity(
                    (),
                    |In(())| Some(vec![crate::signal_vec::VecDiff::Push { value: -1 }]),
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_vec_entity(
                    vec![ordinary, replay],
                    move |In(diffs): In<Vec<crate::signal_vec::VecDiff<i32>>>| {
                        observed
                            .lock()
                            .expect("mixed-source observer mutex poisoned")
                            .push(diffs.clone());
                        let mut current = current.lock().expect("mixed-source current mutex poisoned");
                        for diff in diffs {
                            diff.apply_to_vec(&mut current);
                        }
                        None::<Vec<crate::signal_vec::VecDiff<i32>>>
                    },
                    OutputDelivery::move_only(),
                )
            }
        });
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        let claim = materializer.register(app.world_mut());
        let dependencies = app
            .world()
            .get::<StaticSignalDependencies>(*claim.entity())
            .expect("mixed-source observer should have canonical dependencies");
        assert_eq!(dependencies.0.len(), 2);
        assert!(
            dependencies
                .0
                .windows(2)
                .all(|pair| signal_identity_cmp(&pair[0], &pair[1]).is_lt())
        );

        assert!(run_signal_subgraph_once_internal(app.world_mut(), claim.entity()).is_none());

        assert_eq!(
            *observed.lock().expect("mixed-source observer mutex poisoned"),
            vec![
                vec![crate::signal_vec::VecDiff::Push { value: -1 }],
                vec![crate::signal_vec::VecDiff::Replace { values: vec![2, 3] }],
            ]
        );
        assert_eq!(
            *current.lock().expect("mixed-source current mutex poisoned"),
            vec![2, 3]
        );
        claim.release(app.world_mut());
    }

    #[test]
    fn reactivation_replaces_pending_collection_snapshot_at_newest_position_registration_ownership_observability() {
        let source = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_source(
                vec![0],
                |In(()), mut value: Local<i32>| {
                    *value += 1;
                    Some(vec![crate::signal_vec::VecDiff::Push { value: *value }])
                },
                OutputDelivery::move_only(),
            )
        });
        let retained_source = source.shared_clone();
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observer = crate::ScalarSignalMaterializer::<()>::new({
            let observed = observed.clone();
            move |registrar| {
                let source = registrar.materialize_signal_vec(source)?;
                registrar.signal_entity(
                    source,
                    move |In(diffs): In<Vec<crate::signal_vec::VecDiff<i32>>>| {
                        observed
                            .lock()
                            .expect("reactivated collection observer mutex poisoned")
                            .push(diffs);
                        None::<()>
                    },
                    OutputDelivery::move_only(),
                )
            }
        });
        let retained_observer = observer.shared_clone();
        let mut world = lazy_cleanup_world();
        let first_claim = observer.register(&mut world);
        let observer_entity = first_claim.entity();
        let source_entity = world
            .get::<StaticSignalDependencies>(*observer_entity)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("observer should have one source");
        first_claim.release(&mut world);
        assert!(
            world
                .get::<SignalInputBuffer>(*observer_entity)
                .expect("retained observer should keep its input buffer")
                .0
                .is_empty()
        );

        run_signal_node(&mut world, source_entity);
        let second_claim = retained_observer.register(&mut world);
        assert_eq!(second_claim.entity(), observer_entity);
        run_signal_node(&mut world, source_entity);
        run_signal_node(&mut world, observer_entity);

        assert_eq!(
            *observed.lock().expect("reactivated collection observer mutex poisoned"),
            vec![vec![crate::signal_vec::VecDiff::Replace { values: vec![0, 1, 2] }]]
        );
        second_claim.release(&mut world);
        drop(retained_source);
    }

    #[test]
    fn polling_processes_queued_replay_snapshot_before_newer_output_registration_ownership_observability() {
        let observed = Arc::new(std::sync::Mutex::new(Vec::<i32>::new()));
        let observed_for_system = observed.clone();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let source = registrar.replayable_signal_source(
                0,
                |In(()), mut invocation: Local<i32>| {
                    *invocation += 1;
                    Some(*invocation)
                },
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                source,
                move |In(value): In<i32>| {
                    observed_for_system.lock().expect("observer mutex poisoned").push(value);
                    Some(value)
                },
                OutputDelivery::move_only(),
            )
        });
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        let claim = materializer.register(app.world_mut());

        assert_eq!(
            run_signal_subgraph_once_internal(app.world_mut(), claim.entity())
                .and_then(downcast_erased_signal_value::<i32>),
            Some(1)
        );
        app.update();

        let observed = observed.lock().expect("observer mutex poisoned").clone();
        assert_eq!(observed, vec![0, 1, 2]);
        assert_ne!(observed, vec![1, 0, 2]);
        claim.release(app.world_mut());
    }

    #[test]
    fn polling_one_vector_branch_preserves_all_replay_relationship_revisions_registration_ownership_observability() {
        fn vector_observer(
            source: crate::VecSignalMaterializer<i32>,
            current: Arc<std::sync::Mutex<Vec<i32>>>,
        ) -> crate::VecSignalMaterializer<i32> {
            crate::VecSignalMaterializer::new(move |registrar| {
                let source = registrar.materialize_signal_vec(source)?;
                registrar.signal_vec_entity(
                    source,
                    move |In(diffs): In<Vec<crate::signal_vec::VecDiff<i32>>>| {
                        let mut current = current.lock().expect("vector observer mutex poisoned");
                        for diff in diffs {
                            diff.apply_to_vec(&mut current);
                        }
                        None::<Vec<crate::signal_vec::VecDiff<i32>>>
                    },
                    OutputDelivery::move_only(),
                )
            })
        }

        let source = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_source(
                vec![0],
                |In(()), mut invocation: Local<i32>| {
                    *invocation += 1;
                    Some(vec![crate::signal_vec::VecDiff::Push { value: *invocation }])
                },
                OutputDelivery::broadcast(),
            )
        });
        let retained_source = source.shared_clone();
        let first_state = Arc::new(std::sync::Mutex::new(Vec::new()));
        let second_state = Arc::new(std::sync::Mutex::new(Vec::new()));
        let first = vector_observer(source.shared_clone(), first_state.clone());
        let second = vector_observer(source, second_state.clone());
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        let first_claim = first.register(app.world_mut());
        let second_claim = second.register(app.world_mut());
        let source_entity = app
            .world()
            .get::<StaticSignalDependencies>(*first_claim.entity())
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("vector observer should have one replayable dependency");

        assert!(run_signal_subgraph_once_internal(app.world_mut(), first_claim.entity()).is_none());
        assert_eq!(*first_state.lock().expect("vector observer mutex poisoned"), vec![0, 1]);
        assert!(second_state.lock().expect("vector observer mutex poisoned").is_empty());
        app.update();

        assert_eq!(
            *first_state.lock().expect("vector observer mutex poisoned"),
            vec![0, 1, 2]
        );
        assert_eq!(
            *second_state.lock().expect("vector observer mutex poisoned"),
            vec![0, 1, 2]
        );
        let relationships = app
            .world()
            .get::<ReplayConsumingRelationships>(*source_entity)
            .expect("vector source should retain replay relationship cursors");
        assert!(relationships.0.values().all(|relationship| {
            matches!(relationship, ReplayRelationshipState::Published { cursor } if cursor.get() == 3)
        }));

        first_claim.release(app.world_mut());
        second_claim.release(app.world_mut());
        drop(retained_source);
    }

    #[test]
    fn polling_one_map_branch_preserves_all_replay_relationship_revisions_registration_ownership_observability() {
        fn map_observer(
            source: crate::MapSignalMaterializer<i32, i32>,
            current: Arc<std::sync::Mutex<BTreeMap<i32, i32>>>,
        ) -> crate::MapSignalMaterializer<i32, i32> {
            crate::MapSignalMaterializer::new(move |registrar| {
                let source = registrar.materialize_signal_map(source)?;
                registrar.signal_map_entity(
                    source,
                    move |In(diffs): In<Vec<crate::signal_map::MapDiff<i32, i32>>>| {
                        let mut current = current.lock().expect("map observer mutex poisoned");
                        for diff in diffs {
                            match diff {
                                crate::signal_map::MapDiff::Replace { entries } => {
                                    *current = entries.into_iter().collect();
                                }
                                crate::signal_map::MapDiff::Insert { key, value }
                                | crate::signal_map::MapDiff::Update { key, value } => {
                                    current.insert(key, value);
                                }
                                crate::signal_map::MapDiff::Remove { key } => {
                                    current.remove(&key);
                                }
                                crate::signal_map::MapDiff::Clear => current.clear(),
                            }
                        }
                        None::<Vec<crate::signal_map::MapDiff<i32, i32>>>
                    },
                    OutputDelivery::move_only(),
                )
            })
        }

        let source = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            registrar.replayable_signal_map_source(
                BTreeMap::from([(0, 0)]),
                |In(()), mut invocation: Local<i32>| {
                    *invocation += 1;
                    Some(vec![crate::signal_map::MapDiff::Insert {
                        key: *invocation,
                        value: *invocation,
                    }])
                },
                OutputDelivery::broadcast(),
            )
        });
        let retained_source = source.shared_clone();
        let first_state = Arc::new(std::sync::Mutex::new(BTreeMap::new()));
        let second_state = Arc::new(std::sync::Mutex::new(BTreeMap::new()));
        let first = map_observer(source.shared_clone(), first_state.clone());
        let second = map_observer(source, second_state.clone());
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        let first_claim = first.register(app.world_mut());
        let second_claim = second.register(app.world_mut());
        let source_entity = app
            .world()
            .get::<StaticSignalDependencies>(*first_claim.entity())
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("map observer should have one replayable dependency");

        assert!(run_signal_subgraph_once_internal(app.world_mut(), first_claim.entity()).is_none());
        assert_eq!(
            *first_state.lock().expect("map observer mutex poisoned"),
            BTreeMap::from([(0, 0), (1, 1)])
        );
        assert!(second_state.lock().expect("map observer mutex poisoned").is_empty());
        app.update();

        let expected = BTreeMap::from([(0, 0), (1, 1), (2, 2)]);
        assert_eq!(*first_state.lock().expect("map observer mutex poisoned"), expected);
        assert_eq!(*second_state.lock().expect("map observer mutex poisoned"), expected);
        let relationships = app
            .world()
            .get::<ReplayConsumingRelationships>(*source_entity)
            .expect("map source should retain replay relationship cursors");
        assert!(relationships.0.values().all(|relationship| {
            matches!(relationship, ReplayRelationshipState::Published { cursor } if cursor.get() == 3)
        }));

        first_claim.release(app.world_mut());
        second_claim.release(app.world_mut());
        drop(retained_source);
    }

    #[test]
    fn polling_replayable_target_preserves_downstream_relationship_order_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let source = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(
                0,
                |In(()), mut invocation: Local<i32>| {
                    *invocation += 1;
                    Some(*invocation)
                },
                OutputDelivery::broadcast(),
            )
        });
        let dependent_source = source.shared_clone();
        let source_claim = source.register(&mut world);
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let dependent_claim = crate::ScalarSignalMaterializer::<()>::new({
            let observed = observed.clone();
            move |registrar| {
                let source = registrar.materialize_signal(dependent_source)?;
                registrar.signal_entity(
                    source,
                    move |In(value): In<i32>| {
                        observed.lock().expect("downstream observer mutex poisoned").push(value);
                        None::<()>
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(&mut world);

        assert_eq!(
            run_signal_subgraph_once_internal(&mut world, source_claim.entity())
                .and_then(downcast_erased_signal_value::<i32>),
            Some(1)
        );
        run_signal_node(&mut world, dependent_claim.entity());
        assert_eq!(
            *observed.lock().expect("downstream observer mutex poisoned"),
            vec![0, 1]
        );
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source_claim.entity())
                .and_then(|relationships| relationships.0.get(&dependent_claim.entity())),
            Some(ReplayRelationshipState::Published { cursor }) if cursor.get() == 2
        ));

        dependent_claim.release(&mut world);
        source_claim.release(&mut world);
    }

    #[test]
    fn replay_cursor_advances_only_after_output_is_enqueued_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let source_claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        })
        .register(&mut world);
        let source = source_claim.entity();
        let dependent = SignalEntity::from(world.spawn_empty().id());
        world.entity_mut(*source).insert((
            ActiveSignalDependents(HashSet::from([dependent])),
            ReplayConsumingRelationships(BTreeMap::from([(
                dependent,
                ReplayRelationshipState::Published {
                    cursor: ReplayRevision::FIRST,
                },
            )])),
        ));
        let update = world
            .get_mut::<crate::replay::ScalarReplayCache<i32>>(*source)
            .expect("ready replay source should retain its typed cache")
            .accept(2)
            .expect("test replay revision should not overflow");
        let crate::replay::ReplayCacheUpdate::Accepted { output, revision } = update else {
            unreachable!("scalar replay accepts every output");
        };

        deliver_accepted_signal_output(
            &mut world,
            source,
            AcceptedSignalOutput {
                output: Box::new(output),
                replay_revision: Some(revision),
            },
            push_signal_input,
        );

        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&dependent)),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));
        world.get_mut::<ActiveSignalDependents>(*source).unwrap().0.clear();
        world.entity_mut(*source).remove::<ReplayConsumingRelationships>();
        source_claim.release(&mut world);
    }

    #[test]
    fn polling_uses_replay_acceptance_boundary_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(())| Some(7),
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);

        let output = run_signal_subgraph_once_internal(&mut world, claim.entity())
            .and_then(downcast_erased_signal_value::<i32>)
            .expect("imperative evaluation should return the accepted replay output");
        assert_eq!(output, 7);
        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<i32>>(*claim.entity())
                .expect("polled replay cache should exist")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready {
                current: 7,
                revision,
            } if revision.get() == 1
        ));
        claim.release(&mut world);
    }

    #[test]
    fn replay_components_publish_only_at_validated_commit_stage_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        world.insert_resource(RegistrationCommitStageTrace::default());
        let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(())| Some(1),
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);

        assert_eq!(
            world.resource::<RegistrationCommitStageTrace>().0,
            vec![
                RegistrationCommitStage::EntityPublishedWithoutReplay(claim.entity()),
                RegistrationCommitStage::ReplayPublished(claim.entity()),
            ]
        );
        claim.release(&mut world);
    }

    #[test]
    fn replay_revision_matches_final_forwarded_output_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let first = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let second = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(2), OutputDelivery::move_only())
        });
        let claim = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let first = registrar.materialize_signal(first)?;
            let second = registrar.materialize_signal(second)?;
            registrar.replayable_signal_entity(
                [first, second],
                |In(value): In<i32>| Some(value),
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);
        let signal = claim.entity();
        {
            let mut inputs = world
                .get_mut::<SignalInputBuffer>(*signal)
                .expect("replayable terminal should have an input buffer");
            inputs.push(Box::new(10_i32));
            inputs.push(Box::new(20_i32));
        }

        run_signal_node(&mut world, signal);

        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<i32>>(*signal)
                .expect("managed replay cache should exist")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready {
                current: 20,
                revision,
            } if revision.get() == 1
        ));
        claim.release(&mut world);
    }

    #[test]
    fn warming_collection_relationships_publish_authoritative_replace_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let vector_source = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_entity(
                (),
                |In(())| {
                    Some(vec![
                        crate::signal_vec::VecDiff::Replace { values: Vec::new() },
                        crate::signal_vec::VecDiff::Push { value: 2 },
                    ])
                },
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        let vector_dependent_source = vector_source.shared_clone();
        let vector_claim = vector_source.register(&mut world);
        let vector_batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let vector_dependent = crate::VecSignalMaterializer::<i32>::new({
            let vector_batches = vector_batches.clone();
            move |registrar| {
                let source = registrar.materialize_signal_vec(vector_dependent_source)?;
                registrar.signal_vec_entity(
                    source,
                    move |In(batch): In<Vec<crate::signal_vec::VecDiff<i32>>>| {
                        vector_batches
                            .lock()
                            .expect("vector replay observer mutex poisoned")
                            .push(batch);
                        None::<Vec<crate::signal_vec::VecDiff<i32>>>
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(&mut world);

        run_signal_node(&mut world, vector_claim.entity());
        run_signal_node(&mut world, vector_dependent.entity());
        assert_eq!(
            *vector_batches.lock().expect("vector replay observer mutex poisoned"),
            vec![vec![crate::signal_vec::VecDiff::Replace { values: vec![2] }]]
        );

        let map_source = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            registrar.replayable_signal_map_entity(
                (),
                |In(())| Some(vec![crate::signal_map::MapDiff::Replace { entries: Vec::new() }]),
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        let map_dependent_source = map_source.shared_clone();
        let map_claim = map_source.register(&mut world);
        let map_batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let map_dependent = crate::MapSignalMaterializer::<i32, i32>::new({
            let map_batches = map_batches.clone();
            move |registrar| {
                let source = registrar.materialize_signal_map(map_dependent_source)?;
                registrar.signal_map_entity(
                    source,
                    move |In(batch): In<Vec<crate::signal_map::MapDiff<i32, i32>>>| {
                        map_batches
                            .lock()
                            .expect("map replay observer mutex poisoned")
                            .push(batch);
                        None::<Vec<crate::signal_map::MapDiff<i32, i32>>>
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(&mut world);

        run_signal_node(&mut world, map_claim.entity());
        run_signal_node(&mut world, map_dependent.entity());
        assert_eq!(
            *map_batches.lock().expect("map replay observer mutex poisoned"),
            vec![vec![crate::signal_map::MapDiff::Replace { entries: Vec::new() }]]
        );

        vector_dependent.release(&mut world);
        vector_claim.release(&mut world);
        map_dependent.release(&mut world);
        map_claim.release(&mut world);
    }

    #[test]
    fn retained_replay_terminal_owns_one_maintenance_claim_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(1, |In(())| Some(2), OutputDelivery::broadcast())
        });
        let retained = materializer.shared_clone();
        let claim = materializer.register(&mut world);
        let entity = claim.entity();
        let active = graph_snapshot(&mut world);
        let active = active
            .node(entity.entity())
            .expect("retained replay terminal should exist");
        assert_eq!(active.claim_count, Some(2));
        assert!(active.replay_maintenance_armed);
        assert_eq!(active.live_terminal_retainer_count, 1);

        claim.release(&mut world);
        let retained_snapshot = graph_snapshot(&mut world);
        let retained_snapshot = retained_snapshot
            .node(entity.entity())
            .expect("maintenance claim should retain replay terminal");
        assert_eq!(retained_snapshot.claim_count, Some(1));
        assert!(retained_snapshot.replay_maintenance_armed);
        assert!(!retained_snapshot.registrar_inactive);

        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(*entity).is_err());
    }

    #[test]
    fn consumed_replay_dependency_with_sibling_retains_one_maintenance_claim_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let next_value = Arc::new(core::sync::atomic::AtomicUsize::new(1));
        let dependency = crate::ScalarSignalMaterializer::<usize>::new({
            let next_value = next_value.clone();
            move |registrar| {
                registrar.replayable_signal_source(
                    0,
                    move |In(())| Some(next_value.fetch_add(1, core::sync::atomic::Ordering::SeqCst)),
                    OutputDelivery::broadcast(),
                )
            }
        });
        let sibling = dependency.shared_clone();
        let derived = crate::ScalarSignalMaterializer::<usize>::new(move |registrar| {
            let dependency = registrar.materialize_signal(dependency)?;
            registrar.signal_entity(
                dependency,
                |In(value): In<usize>| Some(value),
                OutputDelivery::move_only(),
            )
        });

        let derived_claim = derived.register(&mut world);
        let derived_entity = derived_claim.entity();
        let dependency_entity = world
            .get::<StaticSignalDependencies>(*derived_entity)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("derived entity should retain its replayable dependency");
        derived_claim.release(&mut world);

        let retained = graph_snapshot(&mut world)
            .node(dependency_entity.entity())
            .cloned()
            .expect("retained replayable dependency should remain materialized");
        assert_eq!(retained.claim_count, Some(1));
        assert!(retained.replay_maintenance_armed);
        assert!(!retained.registrar_inactive);

        let schedule = world.resource::<SignalGraphState>().default_schedule;
        process_signal_graph_for_schedule(schedule)(&mut world);
        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<usize>>(*dependency_entity)
                .expect("retained replayable dependency should keep its cache")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready {
                current: 1,
                revision,
            } if revision.get() == 2
        ));

        let sibling_claim = sibling
            .shared_clone()
            .try_register(&mut world)
            .expect("the remaining semantic owner should reuse the retained dependency");
        assert_eq!(sibling_claim.entity(), dependency_entity);
        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<usize>>(*dependency_entity)
                .expect("reused dependency should preserve its current cache")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready { current: 1, .. }
        ));
        sibling_claim.release(&mut world);
        assert_eq!(
            world.get::<SignalClaimCount>(*dependency_entity).map(|count| **count),
            Some(1)
        );

        drop(sibling);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(*dependency_entity).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn final_drop_concurrent_with_ready_publication_selects_one_local_candidate_registration_ownership_observability() {
        let inner = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(0, |In(())| Some(1), OutputDelivery::broadcast())
        });
        let final_inner_owner = inner.shared_clone();
        let start_drop = Arc::new(std::sync::Barrier::new(2));
        let decrement_observed = Arc::new(std::sync::Barrier::new(2));
        let finish_drop = Arc::new(std::sync::Barrier::new(2));
        final_inner_owner.test_after_final_live_chain_decrement(Arc::new({
            let decrement_observed = decrement_observed.clone();
            let finish_drop = finish_drop.clone();
            move || {
                decrement_observed.wait();
                finish_drop.wait();
            }
        }));
        let drop_thread = std::thread::spawn({
            let start_drop = start_drop.clone();
            move || {
                start_drop.wait();
                drop(final_inner_owner);
            }
        });
        let outer = crate::ScalarSignalMaterializer::<i32>::new({
            let start_drop = start_drop.clone();
            let decrement_observed = decrement_observed.clone();
            move |registrar| {
                let terminal = registrar.materialize_signal(inner)?;
                start_drop.wait();
                decrement_observed.wait();
                Ok(terminal)
            }
        });
        let retained_outer = outer.shared_clone();
        let mut world = lazy_cleanup_world();
        world.insert_resource(OwnershipWorklistTrace::default());

        let claim = outer.register(&mut world);
        let terminal = claim.entity();
        let local_candidates = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Candidate(candidate) => Some(*candidate),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(local_candidates, [terminal]);
        cleanup::drain_world_cleanup_with_stale_signal_roots(&mut world, |roots| {
            assert!(roots.is_empty());
        });
        finish_drop.wait();
        drop_thread
            .join()
            .expect("final materializer drop thread should finish");
        cleanup::drain_world_cleanup_with_stale_signal_roots(&mut world, |roots| {
            assert!(roots.is_empty());
        });

        claim.release(&mut world);
        drop(retained_outer);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn independent_publications_share_one_replay_maintenance_claim_registration_ownership_observability() {
        let inner = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(0, |In(())| Some(1), OutputDelivery::broadcast())
        });
        let retained_inner = inner.shared_clone();
        let outer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| registrar.materialize_signal(inner));
        let retained_outer = outer.shared_clone();
        let mut world = lazy_cleanup_world();

        let claim = outer.register(&mut world);
        let terminal = claim.entity();
        let active = graph_snapshot(&mut world);
        let active_terminal = active
            .node(terminal.entity())
            .expect("shared replay terminal should be committed once");
        assert_eq!(active.nodes.len(), 1);
        assert_eq!(active_terminal.terminal_retainer_count, 2);
        assert_eq!(active_terminal.live_terminal_retainer_count, 2);
        assert_eq!(active_terminal.claim_count, Some(2));
        assert!(active_terminal.replay_maintenance_armed);

        drop(retained_inner);
        cleanup::drain_world_cleanup_with_stale_signal_roots(&mut world, |roots| {
            assert_eq!(roots, [terminal]);
        });
        let one_publication = graph_snapshot(&mut world);
        let one_publication = one_publication
            .node(terminal.entity())
            .expect("remaining publication should retain the terminal");
        assert_eq!(one_publication.live_terminal_retainer_count, 1);
        assert_eq!(one_publication.claim_count, Some(2));
        assert!(one_publication.replay_maintenance_armed);

        drop(retained_outer);
        cleanup::drain_world_cleanup_with_stale_signal_roots(&mut world, |roots| {
            assert_eq!(roots, [terminal]);
        });
        let no_publications = graph_snapshot(&mut world);
        let no_publications = no_publications
            .node(terminal.entity())
            .expect("direct claim should keep the terminal alive after maintenance release");
        assert_eq!(no_publications.live_terminal_retainer_count, 0);
        assert_eq!(no_publications.claim_count, Some(1));
        assert!(!no_publications.replay_maintenance_armed);

        claim.release(&mut world);
        assert!(world.get_entity(*terminal).is_err());
    }

    #[test]
    fn managed_replay_warms_from_scheduled_output_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let claim = crate::ScalarSignalMaterializer::<Option<i32>>::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(())| Some(None::<i32>),
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);
        let signal = claim.entity();

        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<Option<i32>>>(*signal)
                .expect("managed scalar cache should be installed")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Warming
        ));
        run_signal_node(&mut world, signal);
        assert!(matches!(
            world
                .get::<crate::replay::ScalarReplayCache<Option<i32>>>(*signal)
                .expect("managed scalar cache should remain installed")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready {
                current: None,
                revision,
            } if revision.get() == 1
        ));
        assert_eq!(
            graph_snapshot(&mut world)
                .node(signal.entity())
                .expect("managed scalar should be observable")
                .replay_cache_readiness,
            Some(ReplayCacheReadiness::Ready {
                revision: ReplayRevision::FIRST,
            })
        );
        claim.release(&mut world);
    }

    #[test]
    fn managed_vector_replay_suppresses_incrementals_until_replace_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let claim = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_entity(
                (),
                |In(()), mut invocation: Local<usize>| {
                    *invocation += 1;
                    if *invocation == 1 {
                        Some(vec![crate::signal_vec::VecDiff::Push { value: 9 }])
                    } else {
                        Some(vec![
                            crate::signal_vec::VecDiff::Replace { values: Vec::new() },
                            crate::signal_vec::VecDiff::Push { value: 2 },
                        ])
                    }
                },
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);
        let signal = claim.entity();

        run_signal_node(&mut world, signal);
        assert!(matches!(
            world
                .get::<crate::replay::VecReplayCache<i32>>(*signal)
                .expect("managed vector cache should be installed")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Warming
        ));
        run_signal_node(&mut world, signal);
        assert!(matches!(
            world
                .get::<crate::replay::VecReplayCache<i32>>(*signal)
                .expect("managed vector cache should remain installed")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready {
                current,
                revision,
            } if current == &vec![2] && revision.get() == 1
        ));
        claim.release(&mut world);
    }

    #[test]
    fn managed_map_replay_suppresses_incrementals_until_replace_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let claim = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            registrar.replayable_signal_map_entity(
                (),
                |In(()), mut invocation: Local<usize>| {
                    *invocation += 1;
                    if *invocation == 1 {
                        Some(vec![crate::signal_map::MapDiff::Insert { key: 9, value: 9 }])
                    } else {
                        Some(vec![
                            crate::signal_map::MapDiff::Replace { entries: Vec::new() },
                            crate::signal_map::MapDiff::Insert { key: 2, value: 3 },
                        ])
                    }
                },
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);
        let signal = claim.entity();

        run_signal_node(&mut world, signal);
        assert!(matches!(
            world
                .get::<crate::replay::MapReplayCache<i32, i32>>(*signal)
                .expect("managed map cache should be installed")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Warming
        ));
        run_signal_node(&mut world, signal);
        assert!(matches!(
            world
                .get::<crate::replay::MapReplayCache<i32, i32>>(*signal)
                .expect("managed map cache should remain installed")
                .snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready {
                current,
                revision,
            } if current == &alloc::collections::BTreeMap::from([(2, 3)]) && revision.get() == 1
        ));
        claim.release(&mut world);
    }

    #[test]
    fn intrinsic_replay_state_is_ready_without_execution_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let scalar = crate::ScalarSignalMaterializer::<Option<i32>>::new(|registrar| {
            registrar.replayable_signal_source(None, |In(())| None::<Option<i32>>, OutputDelivery::move_only())
        })
        .register(&mut world);
        let vector = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_vec_source(
                Vec::new(),
                |In(())| None::<Vec<crate::signal_vec::VecDiff<i32>>>,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);
        let map = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            registrar.replayable_signal_map_source(
                alloc::collections::BTreeMap::new(),
                |In(())| None::<Vec<crate::signal_map::MapDiff<i32, i32>>>,
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);

        assert!(matches!(
            world.get::<crate::replay::ScalarReplayCache<Option<i32>>>(*scalar.entity()).unwrap().snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready { current: None, revision } if revision.get() == 1
        ));
        assert!(matches!(
            world.get::<crate::replay::VecReplayCache<i32>>(*vector.entity()).unwrap().snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready { current, revision }
                if current.is_empty() && revision.get() == 1
        ));
        assert!(matches!(
            world.get::<crate::replay::MapReplayCache<i32, i32>>(*map.entity()).unwrap().snapshot(),
            crate::replay::ReplayCacheSnapshot::Ready { current, revision }
                if current.is_empty() && revision.get() == 1
        ));

        scalar.release(&mut world);
        vector.release(&mut world);
        map.release(&mut world);
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
    struct UnregisteredMaterializationSchedule;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
    struct RegistrarScheduleA;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
    struct RegistrarScheduleB;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
    struct RegistrarScheduleC;

    #[test]
    fn registrar_schedule_projection_preserves_declaration_order_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        {
            let mut graph = world.resource_mut::<SignalGraphState>();
            graph.register_schedule(RegistrarScheduleA.intern());
            graph.register_schedule(RegistrarScheduleB.intern());
            graph.register_schedule(RegistrarScheduleC.intern());
        }
        let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let source = registrar.apply_schedule(source, RegistrarScheduleA)?;
            let terminal =
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())?;
            let original_terminal = terminal;
            let terminal = registrar.apply_schedule(terminal, RegistrarScheduleB)?;
            let terminal = registrar.apply_schedule(terminal, RegistrarScheduleC)?;
            registrar.signal_entity(
                [original_terminal, terminal],
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        })
        .register(&mut world);

        let snapshot = graph_snapshot(&mut world);
        let sink = snapshot
            .node(claim.entity().entity())
            .expect("registrar sink should be observable");
        let terminal_entity = sink
            .canonical_static_dependencies
            .as_ref()
            .and_then(|dependencies| dependencies.first())
            .copied()
            .expect("sink should depend once on the unchanged scheduled reference");
        assert_eq!(sink.canonical_static_dependencies.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            sink.schedule_tag,
            Some(ScheduleSnapshot::new(RegistrarScheduleC.intern()))
        );
        assert_eq!(
            sink.schedule_hint,
            Some(ScheduleSnapshot::new(RegistrarScheduleC.intern()))
        );

        let terminal = snapshot.node(terminal_entity).expect("terminal should be observable");
        let source_entity = terminal
            .canonical_static_dependencies
            .as_ref()
            .and_then(|dependencies| dependencies.first())
            .copied()
            .expect("terminal should retain its source");
        assert_eq!(
            terminal.schedule_tag,
            Some(ScheduleSnapshot::new(RegistrarScheduleC.intern()))
        );
        assert_eq!(
            terminal.schedule_hint,
            Some(ScheduleSnapshot::new(RegistrarScheduleC.intern()))
        );

        let source = snapshot.node(source_entity).expect("source should be observable");
        assert_eq!(
            source.schedule_tag,
            Some(ScheduleSnapshot::new(RegistrarScheduleA.intern()))
        );
        assert_eq!(
            source.schedule_hint,
            Some(ScheduleSnapshot::new(RegistrarScheduleA.intern()))
        );
        assert_eq!(source.cached_level, Some(0));
        assert_eq!(
            source.cached_schedule,
            Some(ScheduleSnapshot::new(RegistrarScheduleA.intern()))
        );
        assert_eq!(
            source.schedule_buckets,
            vec![ScheduleBucketSnapshot {
                schedule: ScheduleSnapshot::new(RegistrarScheduleA.intern()),
                level: 0,
            }]
        );
        assert_eq!(terminal.cached_level, Some(1));
        assert_eq!(
            terminal.cached_schedule,
            Some(ScheduleSnapshot::new(RegistrarScheduleC.intern()))
        );
        assert_eq!(sink.cached_level, Some(2));
        assert_eq!(
            sink.cached_schedule,
            Some(ScheduleSnapshot::new(RegistrarScheduleC.intern()))
        );
        assert!(!source.registrar_inactive);
        assert!(!terminal.registrar_inactive);
        assert!(!sink.registrar_inactive);

        claim.release(&mut world);
    }

    #[test]
    fn unregistered_schedule_rejects_before_allocation_registration_ownership_observability() {
        use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, AtomicOrdering::SeqCst);
            }
        }

        let mut world = lazy_cleanup_world();
        let before = graph_snapshot(&mut world);
        let entity_count_before = world.query::<Entity>().iter(&world).count();
        let drops = Arc::new(AtomicUsize::new(0));
        let probe = DropProbe(drops.clone());
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let signal = registrar.signal_entity(
                (),
                move |In(())| {
                    core::hint::black_box(&probe);
                    Some(1)
                },
                OutputDelivery::move_only(),
            )?;
            registrar.apply_schedule(signal, UnregisteredMaterializationSchedule)
        });

        assert!(matches!(
            materializer.try_register(&mut world),
            Err(SignalRegistrationError::ScheduleNotRegistered)
        ));
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(world.query::<Entity>().iter(&world).count(), entity_count_before);
        assert_eq!(graph_snapshot(&mut world), before);
    }

    #[test]
    fn signal_entity_copy_is_identity_only_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained = materializer.shared_clone();
        let claim = materializer.register(&mut world);
        let identity = claim.entity();
        let copied_identity = identity;

        assert_eq!(identity, copied_identity);
        assert_eq!(claim.entity(), copied_identity);
        assert_eq!(**world.get::<SignalClaimCount>(*identity).unwrap(), 1);

        claim.release(&mut world);
        assert_eq!(**world.get::<SignalClaimCount>(*identity).unwrap(), 0);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn independent_claims_balance_once_each_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained = materializer.shared_clone();
        let first = materializer.register(&mut world);
        let second = retained.shared_clone().register(&mut world);
        let identity = first.entity();

        assert_eq!(second.entity(), identity);
        assert_eq!(**world.get::<SignalClaimCount>(*identity).unwrap(), 2);

        first.try_release(&mut world).expect("the first claim should release");
        assert_eq!(**world.get::<SignalClaimCount>(*identity).unwrap(), 1);
        second.release(&mut world);
        assert_eq!(**world.get::<SignalClaimCount>(*identity).unwrap(), 0);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn linear_registrar_activation_and_release_are_phased_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let middle = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                middle,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        world.insert_resource(OwnershipWorklistTrace::default());

        let claim = materializer.register(&mut world);
        let (source, middle, terminal) = linear_registration_entities(&world, claim.entity());
        let acquisitions = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Acquired(signal) => Some(*signal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(acquisitions, [source, middle, terminal]);
        for signal in [source, middle, terminal] {
            assert_eq!(**world.get::<SignalClaimCount>(*signal).unwrap(), 1);
            assert!(world.get::<RegistrarInactive>(*signal).is_none());
        }
        world
            .get_mut::<SignalInputBuffer>(*terminal)
            .unwrap()
            .push(Box::new(99_i32));
        world.resource_mut::<OwnershipWorklistTrace>().0.clear();
        let inbox_before = cleanup::queued_message_count(&world);

        claim.release(&mut world);

        let trace = &world.resource::<OwnershipWorklistTrace>().0;
        assert_eq!(
            &trace[..3],
            [
                OwnershipWorklistStage::Released(terminal),
                OwnershipWorklistStage::Released(middle),
                OwnershipWorklistStage::Released(source),
            ]
        );
        assert!(
            trace[3..]
                .iter()
                .all(|stage| !matches!(stage, OwnershipWorklistStage::Released(_)))
        );
        assert_eq!(cleanup::queued_message_count(&world), inbox_before);
        for signal in [source, middle, terminal] {
            assert_eq!(**world.get::<SignalClaimCount>(*signal).unwrap(), 0);
            assert!(world.get::<RegistrarInactive>(*signal).is_some());
            assert!(world.resource::<SignalGraphState>().levels.get(&signal).is_none());
            assert!(
                world
                    .resource::<SignalGraphState>()
                    .signal_schedules
                    .get(&signal)
                    .is_none()
            );
            assert_eq!(world.get::<SignalInputBuffer>(*signal).unwrap().0.len(), 0);
        }
        assert!(
            !world
                .get::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .contains(&middle)
        );
        assert!(
            !world
                .get::<ActiveSignalDependents>(*middle)
                .unwrap()
                .0
                .contains(&terminal)
        );
        assert!(world.get_entity(*source).is_ok());
        assert!(world.get_entity(*middle).is_ok());
        assert!(world.get_entity(*terminal).is_ok());

        world.resource_mut::<OwnershipWorklistTrace>().0.clear();
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
        let cleanup_trace = &world.resource::<OwnershipWorklistTrace>().0;
        assert!(
            cleanup_trace
                .iter()
                .all(|stage| !matches!(stage, OwnershipWorklistStage::Released(_))),
            "structural reaping must not fabricate claim-release ownership"
        );
        let reaped = cleanup_trace
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Reaped(signal) => Some(*signal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reaped, [terminal, middle, source]);
        assert!(world.get_entity(*terminal).is_err());
        assert!(world.get_entity(*middle).is_err());
        assert!(world.get_entity(*source).is_err());
    }

    #[test]
    fn dormant_existing_reactivation_acquires_in_topology_order_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let middle = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                middle,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let registration = retained.shared_clone();
        let mut world = lazy_cleanup_world();
        let terminal = materialize_materializer(&mut world, materializer)
            .expect("linear graph should materialize without an initial claim");
        let (source, middle, terminal) = linear_registration_entities(&world, terminal);
        let mut identity_order = vec![source, middle, terminal];
        identity_order.sort_by(signal_identity_cmp);
        assert_ne!(identity_order, [source, middle, terminal]);
        world.insert_resource(OwnershipWorklistTrace::default());

        let claim = registration.register(&mut world);

        let acquisitions = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Acquired(signal) => Some(*signal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(acquisitions, [source, middle, terminal]);
        claim.release(&mut world);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn activation_preflight_projects_ordinary_warming_and_ready_routes_registration_ownership_observability() {
        {
            let mut world = lazy_cleanup_world();
            world.insert_resource(OwnershipWorklistTrace::default());
            let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
            })
            .register(&mut world);
            let terminal = claim.entity();
            let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
            assert!(
                world
                    .resource::<OwnershipWorklistTrace>()
                    .0
                    .contains(&OwnershipWorklistStage::OrdinaryRoute(source, terminal))
            );
            claim.release(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            world.insert_resource(OwnershipWorklistTrace::default());
            let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let source = registrar.replayable_signal_entity(
                    (),
                    |In(())| None::<i32>,
                    crate::ReplayInitialization::Warming,
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
            })
            .register(&mut world);
            let terminal = claim.entity();
            let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
            assert!(
                world
                    .resource::<OwnershipWorklistTrace>()
                    .0
                    .contains(&OwnershipWorklistStage::ReplayPendingRoute(source, terminal))
            );
            assert!(matches!(
                world
                    .get::<ReplayConsumingRelationships>(*source)
                    .and_then(|relationships| relationships.0.get(&terminal)),
                Some(ReplayRelationshipState::Pending)
            ));
            claim.release(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            world.insert_resource(OwnershipWorklistTrace::default());
            let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let source = registrar.replayable_signal_source(7, |In(())| Some(8), OutputDelivery::move_only())?;
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
            })
            .register(&mut world);
            let terminal = claim.entity();
            let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
            assert!(
                world
                    .resource::<OwnershipWorklistTrace>()
                    .0
                    .contains(&OwnershipWorklistStage::ReplayReadyRoute(
                        source,
                        terminal,
                        ReplayRevision::FIRST
                    ))
            );
            assert_eq!(world.get::<SignalInputBuffer>(*terminal).unwrap().0.len(), 1);
            assert!(matches!(
                world
                    .get::<ReplayConsumingRelationships>(*source)
                    .and_then(|relationships| relationships.0.get(&terminal)),
                Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
            ));
            claim.release(&mut world);
        }

        {
            let source = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                registrar.replayable_signal_source(11, |In(())| Some(12), OutputDelivery::move_only())
            });
            let source_for_derived = source.shared_clone();
            let retained_source = source.shared_clone();
            let derived = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
                let source = registrar.materialize_signal(source_for_derived)?;
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
            });
            let registration = derived.shared_clone();
            let retained_derived = derived.shared_clone();
            let mut world = lazy_cleanup_world();
            let terminal = materialize_materializer(&mut world, derived)
                .expect("existing ready route fixture should materialize without a claim");
            let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
            world.insert_resource(OwnershipWorklistTrace::default());

            let claim = registration.register(&mut world);

            assert!(
                world
                    .resource::<OwnershipWorklistTrace>()
                    .0
                    .contains(&OwnershipWorklistStage::ReplayReadyRoute(
                        source,
                        terminal,
                        ReplayRevision::FIRST
                    ))
            );
            assert_eq!(world.get::<SignalInputBuffer>(*terminal).unwrap().0.len(), 1);
            claim.release(&mut world);
            drop(retained_derived);
            drop(retained_source);
            cleanup::drain_world_cleanup(&mut world);
        }
    }

    #[test]
    fn sole_top_level_materializer_hands_final_owner_candidate_to_local_cleanup_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        world.insert_resource(OwnershipWorklistTrace::default());

        let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        })
        .register(&mut world);
        let terminal = claim.entity();

        assert_eq!(cleanup::queued_message_count(&world), 0);
        let local_candidates = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Candidate(candidate) => Some(*candidate),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(local_candidates, [terminal]);
        let terminal_snapshot = graph_snapshot(&mut world)
            .node(terminal.entity())
            .expect("the returned claim must keep the terminal valid")
            .clone();
        assert_eq!(terminal_snapshot.claim_count, Some(1));
        assert_eq!(terminal_snapshot.live_terminal_retainer_count, 0);

        claim.release(&mut world);

        assert_eq!(cleanup::queued_message_count(&world), 0);
        assert!(world.get_entity(*terminal).is_err());
    }

    #[test]
    fn sole_child_materializer_hands_candidate_to_parent_local_cleanup_registration_ownership_observability() {
        let child = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let parent = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let child = registrar.materialize_signal(child)?;
            registrar.signal_entity(child, |In(value): In<i32>| Some(value + 1), OutputDelivery::move_only())
        });
        let mut world = lazy_cleanup_world();
        world.insert_resource(OwnershipWorklistTrace::default());

        let claim = parent.register(&mut world);
        let terminal = claim.entity();
        let child = world
            .get::<StaticSignalDependencies>(*terminal)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("parent should have the child as its canonical dependency");

        assert_eq!(cleanup::queued_message_count(&world), 0);
        assert_eq!(
            world
                .get::<MaterializedSignalDependents>(*child)
                .unwrap()
                .0
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [terminal]
        );
        let snapshot = graph_snapshot(&mut world);
        for signal in [child, terminal] {
            let node = snapshot
                .node(signal.entity())
                .expect("local candidates must not reap active structure");
            assert_eq!(node.live_terminal_retainer_count, 0);
        }
        let local_candidates = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Candidate(candidate) => Some(*candidate),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(local_candidates, [terminal, child]);
        world.resource_mut::<OwnershipWorklistTrace>().0.clear();

        claim.release(&mut world);

        assert_eq!(cleanup::queued_message_count(&world), 0);
        let reaped = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Reaped(signal) => Some(*signal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reaped, [terminal, child]);
        assert!(world.get_entity(*terminal).is_err());
        assert!(world.get_entity(*child).is_err());
    }

    #[test]
    fn retained_linear_chain_uses_terminal_only_retention_and_current_dormant_schedule_registration_ownership_observability()
     {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let middle = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                middle,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        world
            .resource_mut::<SignalGraphState>()
            .register_schedule(RegistrarScheduleA.intern());
        let first_claim = materializer.register(&mut world);
        let (source, middle, terminal) = linear_registration_entities(&world, first_claim.entity());
        let active = graph_snapshot(&mut world);

        assert_eq!(active.node(source.entity()).unwrap().live_terminal_retainer_count, 0);
        assert_eq!(active.node(middle.entity()).unwrap().live_terminal_retainer_count, 0);
        assert_eq!(active.node(terminal.entity()).unwrap().live_terminal_retainer_count, 1);
        assert_eq!(
            active.node(source.entity()).unwrap().materialized_dependents,
            [middle.entity()]
        );
        assert_eq!(
            active.node(middle.entity()).unwrap().materialized_dependents,
            [terminal.entity()]
        );
        assert!(
            active
                .node(terminal.entity())
                .unwrap()
                .materialized_dependents
                .is_empty()
        );

        first_claim.release(&mut world);
        for signal in [source, middle, terminal] {
            assert_eq!(world.get::<SignalClaimCount>(*signal).map(|count| **count), Some(0));
            assert!(world.get::<RegistrarInactive>(*signal).is_some());
            assert!(world.resource::<SignalGraphState>().levels.get(&signal).is_none());
        }
        let scheduled_materializer = crate::ScalarSignalMaterializer::<i32>::new({
            let retained = retained.shared_clone();
            move |registrar| {
                let terminal = registrar.materialize_signal(retained)?;
                registrar.apply_schedule(terminal, RegistrarScheduleA)
            }
        });
        let second_claim = scheduled_materializer.register(&mut world);
        assert_eq!(second_claim.entity(), terminal);
        assert_eq!(
            world.resource::<SignalGraphState>().signal_schedules.get(&terminal),
            Some(&RegistrarScheduleA.intern())
        );
        for signal in [source, middle, terminal] {
            assert!(world.get::<RegistrarInactive>(*signal).is_none());
        }

        second_claim.release(&mut world);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        drop(retained);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        for signal in [source, middle, terminal] {
            assert!(world.get_entity(*signal).is_err());
        }
    }

    #[test]
    fn retained_diamond_uses_structural_links_and_reaps_shared_dependency_once_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?;
            let left = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )?;
            let right = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 2),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                [left, right],
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        world.insert_resource(OwnershipWorklistTrace::default());
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let branches = world.get::<StaticSignalDependencies>(*terminal).unwrap().0.to_vec();
        let first_branch = branches.first().copied().expect("diamond should have two branches");
        let source = world
            .get::<StaticSignalDependencies>(*first_branch)
            .and_then(|dependencies| dependencies.0.first().copied())
            .expect("diamond branch should depend on the shared source");
        let active = graph_snapshot(&mut world);

        assert_eq!(active.node(terminal.entity()).unwrap().live_terminal_retainer_count, 1);
        assert_eq!(active.node(source.entity()).unwrap().live_terminal_retainer_count, 0);
        assert_eq!(active.node(source.entity()).unwrap().materialized_dependents.len(), 2);
        for branch in branches.iter().copied() {
            let branch_snapshot = active.node(branch.entity()).unwrap();
            assert_eq!(branch_snapshot.live_terminal_retainer_count, 0);
            assert_eq!(branch_snapshot.materialized_dependents, [terminal.entity()]);
        }

        claim.release(&mut world);
        world.resource_mut::<OwnershipWorklistTrace>().0.clear();
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);

        let reaped = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter_map(|stage| match stage {
                OwnershipWorklistStage::Reaped(signal) => Some(*signal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reaped.first(), Some(&terminal));
        assert_eq!(reaped.last(), Some(&source));
        assert_eq!(reaped.iter().filter(|signal| **signal == source).count(), 1);
        assert!(branches.iter().all(|branch| reaped.contains(branch)));
        assert_eq!(reaped.len(), 4);
    }

    #[test]
    fn retained_vector_and_map_materializers_reactivate_from_terminal_retention_registration_ownership_observability() {
        let vector = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_vec_entity(
                (),
                |In(())| Some(vec![crate::signal_vec::VecDiff::Replace { values: vec![1] }]),
                OutputDelivery::move_only(),
            )
        });
        let retained_vector = vector.shared_clone();
        let map = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            registrar.signal_map_entity(
                (),
                |In(())| Some(vec![crate::signal_map::MapDiff::Replace { entries: vec![(1, 2)] }]),
                OutputDelivery::move_only(),
            )
        });
        let retained_map = map.shared_clone();
        let mut world = lazy_cleanup_world();

        let vector_claim = vector.register(&mut world);
        let vector_entity = vector_claim.entity();
        let map_claim = map.register(&mut world);
        let map_entity = map_claim.entity();
        vector_claim.release(&mut world);
        map_claim.release(&mut world);

        let vector_claim = retained_vector.shared_clone().register(&mut world);
        let map_claim = retained_map.shared_clone().register(&mut world);
        assert_eq!(vector_claim.entity(), vector_entity);
        assert_eq!(map_claim.entity(), map_entity);

        vector_claim.release(&mut world);
        map_claim.release(&mut world);
        drop(retained_vector);
        drop(retained_map);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(*vector_entity).is_err());
        assert!(world.get_entity(*map_entity).is_err());
    }

    #[test]
    fn repeated_terminal_claim_decrements_only_terminal_until_final_release_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let middle =
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())?;
            registrar.signal_entity(middle, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
        });
        let second_owner = materializer.shared_clone();
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let first = materializer.register(&mut world);
        let second = second_owner.register(&mut world);
        let (source, middle, terminal) = linear_registration_entities(&world, first.entity());

        assert_eq!(**world.get::<SignalClaimCount>(*terminal).unwrap(), 2);
        assert_eq!(**world.get::<SignalClaimCount>(*middle).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*source).unwrap(), 1);
        let source_before = graph_snapshot(&mut world).node(source.entity()).unwrap().clone();
        let middle_before = graph_snapshot(&mut world).node(middle.entity()).unwrap().clone();

        first.release(&mut world);

        assert_eq!(**world.get::<SignalClaimCount>(*terminal).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*middle).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*source).unwrap(), 1);
        assert_eq!(graph_snapshot(&mut world).node(source.entity()), Some(&source_before));
        assert_eq!(graph_snapshot(&mut world).node(middle.entity()), Some(&middle_before));
        assert!(world.get::<RegistrarInactive>(*terminal).is_none());

        second.release(&mut world);
        for signal in [source, middle, terminal] {
            assert_eq!(**world.get::<SignalClaimCount>(*signal).unwrap(), 0);
            assert!(world.get::<RegistrarInactive>(*signal).is_some());
        }
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn diamond_release_balances_shared_dependency_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?;
            let left = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::move_only(),
            )?;
            let right = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 2),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                [left, right],
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let join = claim.entity();
        let branches = world.get::<StaticSignalDependencies>(*join).unwrap().0.to_vec();
        assert_eq!(branches.len(), 2);
        let shared = world
            .get::<StaticSignalDependencies>(*branches[0])
            .and_then(|dependencies| dependencies.0.first().copied())
            .unwrap();
        assert_eq!(
            world
                .get::<StaticSignalDependencies>(*branches[1])
                .and_then(|dependencies| dependencies.0.first().copied()),
            Some(shared)
        );
        assert_eq!(**world.get::<SignalClaimCount>(*join).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*branches[0]).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*branches[1]).unwrap(), 1);
        assert_eq!(**world.get::<SignalClaimCount>(*shared).unwrap(), 2);

        claim.release(&mut world);

        for signal in [shared, branches[0], branches[1], join] {
            assert_eq!(**world.get::<SignalClaimCount>(*signal).unwrap(), 0);
        }
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn duplicate_static_dependency_creates_one_acquisition_and_release_obligation_registration_ownership_observability()
    {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            registrar.signal_entity(
                [source, source],
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        world.insert_resource(OwnershipWorklistTrace::default());
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let dependencies = world.get::<StaticSignalDependencies>(*terminal).unwrap();
        assert_eq!(dependencies.0.len(), 1);
        let source = dependencies.0[0];
        assert_eq!(**world.get::<SignalClaimCount>(*source).unwrap(), 1);
        assert_eq!(world.get::<ActiveSignalDependents>(*source).unwrap().0.len(), 1);
        world.resource_mut::<OwnershipWorklistTrace>().0.clear();

        claim.release(&mut world);

        let releases = world
            .resource::<OwnershipWorklistTrace>()
            .0
            .iter()
            .filter(|stage| matches!(stage, OwnershipWorklistStage::Released(signal) if *signal == source))
            .count();
        assert_eq!(releases, 1);
        assert_eq!(**world.get::<SignalClaimCount>(*source).unwrap(), 0);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn warming_replay_relationship_is_removed_on_zero_transition_registration_ownership_observability() {
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let warming = registrar.replayable_signal_entity(
                (),
                |In(())| None::<i32>,
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(warming, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let warming = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*warming)
                .and_then(|relationships| relationships.0.get(&terminal)),
            Some(ReplayRelationshipState::Pending)
        ));
        assert!(
            !world
                .get::<ActiveSignalDependents>(*warming)
                .unwrap()
                .0
                .contains(&terminal)
        );

        claim.release(&mut world);

        assert!(
            world
                .get::<ReplayConsumingRelationships>(*warming)
                .is_none_or(|relationships| !relationships.0.contains_key(&terminal))
        );
        assert!(
            !world
                .get::<ActiveSignalDependents>(*warming)
                .unwrap()
                .0
                .contains(&terminal)
        );
        assert_eq!(**world.get::<SignalClaimCount>(*warming).unwrap(), 0);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn checked_claim_count_transitions_registration_ownership_observability() {
        let mut count = SignalClaimCount::new();
        let _: usize = count.0;
        count.try_increment().expect("zero should increment to one");
        assert_eq!(count.0, 1);
        count.try_decrement().expect("one should decrement to zero");
        assert_eq!(count.0, 0);

        let mut overflow = SignalClaimCount(usize::MAX);
        assert_eq!(overflow.try_increment(), Err(SignalClaimCountError::Overflow));
        assert_eq!(overflow.0, usize::MAX, "overflow must not mutate the count");

        let mut underflow = SignalClaimCount(0);
        assert_eq!(underflow.try_decrement(), Err(SignalClaimCountError::Underflow));
        assert_eq!(underflow.0, 0, "underflow must not mutate the count");
    }

    #[test]
    fn try_release_missing_count_returns_claim_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained = materializer.shared_clone();
        let claim = materializer.register(&mut world);
        let signal = claim.entity();
        world.entity_mut(*signal).remove::<SignalClaimCount>();
        let before = graph_snapshot(&mut world);

        let (error, claim) = claim
            .try_release(&mut world)
            .expect_err("a missing count must reject release");
        assert_eq!(error, SignalReleaseError::MissingClaimCount(signal));
        assert_eq!(
            graph_snapshot(&mut world),
            before,
            "rejected release must not mutate graph state"
        );

        world.entity_mut(*signal).insert(SignalClaimCount(1));
        claim
            .try_release(&mut world)
            .expect("the returned armed claim should remain releasable");
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(*signal).is_err());
    }

    #[test]
    fn registrar_release_corruption_is_atomic_and_returns_armed_claim_registration_ownership_observability() {
        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, source, terminal) = active_unary_registration_fixture(&mut world);
            world
                .get_mut::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .remove(&terminal);
            let before = graph_snapshot(&mut world);
            let (error, claim) = claim
                .try_release(&mut world)
                .expect_err("missing active route must reject");
            assert_eq!(error, inconsistent_release_runtime(terminal));
            assert_eq!(claim.entity(), terminal);
            assert_eq!(graph_snapshot(&mut world), before);
            world
                .get_mut::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .insert(terminal);
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, _source, terminal) = active_unary_registration_fixture(&mut world);
            world.entity_mut(*terminal).remove::<SignalInputBuffer>();
            let before = graph_snapshot(&mut world);
            let (error, claim) = claim
                .try_release(&mut world)
                .expect_err("missing mandatory runtime component must reject");
            assert_eq!(error, inconsistent_release_runtime(terminal));
            assert_eq!(claim.entity(), terminal);
            assert_eq!(graph_snapshot(&mut world), before);
            world.entity_mut(*terminal).insert(SignalInputBuffer::default());
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, _source, terminal) = active_unary_registration_fixture(&mut world);
            world.entity_mut(*terminal).insert(RegistrarInactive);
            let before = graph_snapshot(&mut world);
            let (error, claim) = claim
                .try_release(&mut world)
                .expect_err("contradictory inactive marker must reject");
            assert_eq!(error, inconsistent_release_runtime(terminal));
            assert_eq!(graph_snapshot(&mut world), before);
            world.entity_mut(*terminal).remove::<RegistrarInactive>();
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, _source, terminal) = active_unary_registration_fixture(&mut world);
            let schedule = world
                .resource_mut::<SignalGraphState>()
                .signal_schedules
                .remove(&terminal)
                .expect("active terminal should have effective schedule membership");
            let before = graph_snapshot(&mut world);
            let (error, claim) = claim
                .try_release(&mut world)
                .expect_err("inconsistent effective schedule membership must reject");
            assert_eq!(error, inconsistent_release_runtime(terminal));
            assert_eq!(graph_snapshot(&mut world), before);
            world
                .resource_mut::<SignalGraphState>()
                .signal_schedules
                .insert(terminal, schedule);
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }
    }

    #[test]
    fn stale_outgoing_routes_reject_atomically_and_repair_registration_ownership_observability() {
        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, source, _terminal) = active_unary_registration_fixture(&mut world);
            let rejected = retained.shared_clone();
            let recovered = retained.shared_clone();
            claim.release(&mut world);
            let unrelated_claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                registrar.signal_entity((), |In(())| Some(9), OutputDelivery::move_only())
            })
            .register(&mut world);
            let unrelated = unrelated_claim.entity();
            world
                .get_mut::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .insert(unrelated);
            let before = graph_snapshot(&mut world);

            assert!(matches!(
                rejected.try_register(&mut world),
                Err(SignalRegistrationError::InconsistentExistingRuntimeState { entity }) if entity == source
            ));
            assert_eq!(graph_snapshot(&mut world), before);
            run_signal_node(&mut world, source);
            assert!(world.get::<SignalInputBuffer>(*unrelated).unwrap().0.is_empty());

            world
                .get_mut::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .remove(&unrelated);
            let claim = recovered
                .try_register(&mut world)
                .expect("repairing the stale outgoing route should permit reactivation");
            assert!(world.get::<SignalInputBuffer>(*unrelated).unwrap().0.is_empty());
            claim.release(&mut world);
            unrelated_claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, source, _terminal) = active_unary_registration_fixture(&mut world);
            let rejected = retained.shared_clone();
            let recovered = retained.shared_clone();
            claim.release(&mut world);
            let dangling_entity = world.spawn_empty().id();
            world.entity_mut(dangling_entity).despawn();
            let dangling = SignalEntity::from(dangling_entity);
            world
                .get_mut::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .insert(dangling);
            let before = graph_snapshot(&mut world);

            assert!(matches!(
                rejected.try_register(&mut world),
                Err(SignalRegistrationError::InconsistentExistingRuntimeState { entity }) if entity == source
            ));
            assert_eq!(graph_snapshot(&mut world), before);

            world
                .get_mut::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .remove(&dangling);
            let claim = recovered
                .try_register(&mut world)
                .expect("removing the dangling route should permit reactivation");
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }

        {
            let mut world = lazy_cleanup_world();
            let (retained, claim, source, terminal) = active_unary_registration_fixture(&mut world);
            let rejected = retained.shared_clone();
            let recovered = retained.shared_clone();
            world
                .entity_mut(*source)
                .insert(ReplayConsumingRelationships(BTreeMap::from([(
                    terminal,
                    ReplayRelationshipState::Pending,
                )])));
            let before = graph_snapshot(&mut world);

            assert!(matches!(
                rejected.try_register(&mut world),
                Err(SignalRegistrationError::InconsistentExistingRuntimeState { entity }) if entity == source
            ));
            assert_eq!(graph_snapshot(&mut world), before);

            world.entity_mut(*source).remove::<ReplayConsumingRelationships>();
            let recovered_claim = recovered
                .try_register(&mut world)
                .expect("removing contradictory replay state should permit registration");
            recovered_claim.release(&mut world);
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }
    }

    #[test]
    fn registrar_release_replay_and_projected_underflow_reject_atomically_registration_ownership_observability() {
        {
            let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let source = registrar.replayable_signal_source(0, |In(())| Some(1), OutputDelivery::move_only())?;
                registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
            });
            let retained = materializer.shared_clone();
            let mut world = lazy_cleanup_world();
            let claim = materializer.register(&mut world);
            let terminal = claim.entity();
            let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
            let relationship = world
                .get_mut::<ReplayConsumingRelationships>(*source)
                .unwrap()
                .0
                .remove(&terminal)
                .expect("ready replay activation should publish a relationship");
            let before = graph_snapshot(&mut world);
            let (error, claim) = claim
                .try_release(&mut world)
                .expect_err("missing replay relationship must reject");
            assert_eq!(error, inconsistent_release_runtime(terminal));
            assert_eq!(claim.entity(), terminal);
            assert_eq!(graph_snapshot(&mut world), before);
            world
                .get_mut::<ReplayConsumingRelationships>(*source)
                .unwrap()
                .0
                .insert(terminal, relationship);
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }

        {
            let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let source = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?;
                let left =
                    registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())?;
                let right =
                    registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())?;
                registrar.signal_entity(
                    [left, right],
                    |In(value): In<i32>| Some(value),
                    OutputDelivery::move_only(),
                )
            });
            let retained = materializer.shared_clone();
            let mut world = lazy_cleanup_world();
            let claim = materializer.register(&mut world);
            let join = claim.entity();
            let branch = world.get::<StaticSignalDependencies>(*join).unwrap().0[0];
            let source = world.get::<StaticSignalDependencies>(*branch).unwrap().0[0];
            world.get_mut::<SignalClaimCount>(*source).unwrap().0 = 1;
            let before = graph_snapshot(&mut world);
            let (error, claim) = claim
                .try_release(&mut world)
                .expect_err("diamond source projected underflow must reject");
            assert_eq!(error, SignalReleaseError::ClaimCountUnderflow(source));
            assert_eq!(claim.entity(), join);
            assert_eq!(graph_snapshot(&mut world), before);
            world.get_mut::<SignalClaimCount>(*source).unwrap().0 = 2;
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }
    }

    #[test]
    fn replay_maintenance_preflight_rejection_preserves_armed_claim_registration_ownership_observability() {
        let replay = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(0, |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained = replay.shared_clone();
        let mut world = lazy_cleanup_world();
        let claim = replay.register(&mut world);
        let terminal = claim.entity();
        world.entity_mut(*terminal).remove::<SignalInputBuffer>();
        let before = graph_snapshot(&mut world);

        let Err(error) = preflight_replay_maintenance_release(&mut world, terminal) else {
            panic!("corrupted maintenance runtime should reject before taking its claim");
        };
        assert_eq!(error, inconsistent_release_runtime(terminal));
        assert_eq!(graph_snapshot(&mut world), before);
        assert!(world.get::<ReplayMaintenanceClaim>(*terminal).is_some());

        world.entity_mut(*terminal).insert(SignalInputBuffer::default());
        claim.release(&mut world);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn missing_terminal_release_rejects_without_mutation_registration_ownership_observability() {
        let mut world = lazy_cleanup_world();
        let (_retained, claim, _source, terminal) = active_unary_registration_fixture(&mut world);
        world.entity_mut(*terminal).despawn();
        let before = graph_snapshot(&mut world);

        let (error, claim) = claim
            .try_release(&mut world)
            .expect_err("missing terminal must reject release");

        assert_eq!(error, SignalReleaseError::MissingEntity(terminal));
        assert_eq!(claim.entity(), terminal);
        assert_eq!(graph_snapshot(&mut world), before);
    }

    #[test]
    fn inactive_registrar_rejects_direct_and_stale_replay_input_registration_ownership_observability() {
        use core::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let runs = Arc::new(AtomicUsize::new(0));
        let runs_for_system = runs.clone();
        let materializer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let source = registrar.replayable_signal_source(0, |In(())| Some(1), OutputDelivery::move_only())?;
            registrar.signal_entity(
                source,
                move |In(value): In<i32>| {
                    runs_for_system.fetch_add(1, AtomicOrdering::SeqCst);
                    Some(value)
                },
                OutputDelivery::move_only(),
            )
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
        run_signal_node(&mut world, terminal);
        let active_runs = runs.load(AtomicOrdering::SeqCst);

        claim.release(&mut world);
        run_signal_node(&mut world, terminal);
        assert_eq!(runs.load(AtomicOrdering::SeqCst), active_runs);
        assert!(!push_signal_input(&mut world, terminal, Box::new(7_i32)));
        assert!(world.get::<SignalInputBuffer>(*terminal).unwrap().0.is_empty());
        assert!(world.resource::<SignalGraphState>().levels.get(&terminal).is_none());

        if world.get::<ReplayConsumingRelationships>(*source).is_none() {
            world
                .entity_mut(*source)
                .insert(ReplayConsumingRelationships::default());
        }
        world
            .get_mut::<ReplayConsumingRelationships>(*source)
            .unwrap()
            .0
            .insert(
                terminal,
                ReplayRelationshipState::Published {
                    cursor: ReplayRevision::FIRST,
                },
            );
        let contract = world
            .get::<RegistrationReplayCacheState>(*source)
            .unwrap()
            .contract
            .clone();
        let accepted = contract
            .accept_output(&mut world, source, Box::new(2_i32))
            .expect("scalar replay output should be accepted");
        deliver_accepted_signal_output(&mut world, source, accepted, push_signal_input);
        assert!(world.get::<SignalInputBuffer>(*terminal).unwrap().0.is_empty());
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&terminal)),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));

        drop(retained);
    }

    #[test]
    fn inactive_pending_vector_relationship_rejects_first_runtime_snapshot_registration_ownership_observability() {
        let materializer = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.replayable_signal_vec_entity(
                (),
                |In(())| None::<Vec<crate::signal_vec::VecDiff<i32>>>,
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )?;
            registrar.signal_vec_entity(
                source,
                |In(diffs): In<Vec<crate::signal_vec::VecDiff<i32>>>| Some(diffs),
                OutputDelivery::move_only(),
            )
        });
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let target = claim.entity();
        let source = world.get::<StaticSignalDependencies>(*target).unwrap().0[0];
        world.entity_mut(*target).insert(RegistrarInactive);
        world.get_mut::<SignalInputBuffer>(*target).unwrap().clear();

        let contract = world
            .get::<RegistrationReplayCacheState>(*source)
            .unwrap()
            .contract
            .clone();
        let accepted = contract
            .accept_output(
                &mut world,
                source,
                Box::new(vec![crate::signal_vec::VecDiff::Replace { values: vec![1] }]),
            )
            .expect("first vector replacement should establish readiness");
        deliver_accepted_signal_output(&mut world, source, accepted, push_signal_input);

        assert!(world.get::<SignalInputBuffer>(*target).unwrap().0.is_empty());
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&target)),
            Some(ReplayRelationshipState::Pending)
        ));
        assert!(
            !world
                .get::<ActiveSignalDependents>(*source)
                .is_some_and(|dependents| dependents.0.contains(&target))
        );

        world
            .get_mut::<ReplayConsumingRelationships>(*source)
            .unwrap()
            .0
            .insert(
                target,
                ReplayRelationshipState::Published {
                    cursor: ReplayRevision::FIRST,
                },
            );
        world
            .get_mut::<ActiveSignalDependents>(*source)
            .unwrap()
            .0
            .insert(target);
        world.entity_mut(*target).remove::<RegistrarInactive>();
        claim.release(&mut world);
    }

    #[test]
    fn inactive_pending_map_relationship_rejects_first_runtime_snapshot_registration_ownership_observability() {
        let materializer = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            let source = registrar.replayable_signal_map_entity(
                (),
                |In(())| None::<Vec<crate::signal_map::MapDiff<i32, i32>>>,
                crate::ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )?;
            registrar.signal_map_entity(
                source,
                |In(diffs): In<Vec<crate::signal_map::MapDiff<i32, i32>>>| Some(diffs),
                OutputDelivery::move_only(),
            )
        });
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let target = claim.entity();
        let source = world.get::<StaticSignalDependencies>(*target).unwrap().0[0];
        world.entity_mut(*target).insert(RegistrarInactive);
        world.get_mut::<SignalInputBuffer>(*target).unwrap().clear();

        let contract = world
            .get::<RegistrationReplayCacheState>(*source)
            .unwrap()
            .contract
            .clone();
        let accepted = contract
            .accept_output(
                &mut world,
                source,
                Box::new(vec![crate::signal_map::MapDiff::Replace { entries: vec![(1, 2)] }]),
            )
            .expect("first map replacement should establish readiness");
        deliver_accepted_signal_output(&mut world, source, accepted, push_signal_input);

        assert!(world.get::<SignalInputBuffer>(*target).unwrap().0.is_empty());
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&target)),
            Some(ReplayRelationshipState::Pending)
        ));
        assert!(
            !world
                .get::<ActiveSignalDependents>(*source)
                .is_some_and(|dependents| dependents.0.contains(&target))
        );

        world
            .get_mut::<ReplayConsumingRelationships>(*source)
            .unwrap()
            .0
            .insert(
                target,
                ReplayRelationshipState::Published {
                    cursor: ReplayRevision::FIRST,
                },
            );
        world
            .get_mut::<ActiveSignalDependents>(*source)
            .unwrap()
            .0
            .insert(target);
        world.entity_mut(*target).remove::<RegistrarInactive>();
        claim.release(&mut world);
    }

    #[test]
    fn inactive_published_vector_relationship_does_not_advance_or_restore_route_registration_ownership_observability() {
        let materializer = crate::VecSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.replayable_signal_vec_source(
                vec![0],
                |In(())| None::<Vec<crate::signal_vec::VecDiff<i32>>>,
                OutputDelivery::move_only(),
            )?;
            registrar.signal_vec_entity(
                source,
                |In(diffs): In<Vec<crate::signal_vec::VecDiff<i32>>>| Some(diffs),
                OutputDelivery::move_only(),
            )
        });
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let target = claim.entity();
        let source = world.get::<StaticSignalDependencies>(*target).unwrap().0[0];
        assert!(
            world
                .get::<SignalInputBuffer>(*target)
                .unwrap()
                .has_pending_replay_snapshot(source)
        );
        world.get_mut::<SignalInputBuffer>(*target).unwrap().clear();
        world.entity_mut(*target).insert(RegistrarInactive);
        world
            .get_mut::<ActiveSignalDependents>(*source)
            .unwrap()
            .0
            .remove(&target);

        let contract = world
            .get::<RegistrationReplayCacheState>(*source)
            .unwrap()
            .contract
            .clone();
        let accepted = contract
            .accept_output(
                &mut world,
                source,
                Box::new(vec![crate::signal_vec::VecDiff::Push { value: 1 }]),
            )
            .expect("newer vector output should be accepted");
        deliver_accepted_signal_output(&mut world, source, accepted, push_signal_input);

        assert!(world.get::<SignalInputBuffer>(*target).unwrap().0.is_empty());
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&target)),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));
        assert!(
            !world
                .get::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .contains(&target)
        );

        world
            .get_mut::<ActiveSignalDependents>(*source)
            .unwrap()
            .0
            .insert(target);
        world.entity_mut(*target).remove::<RegistrarInactive>();
        claim.release(&mut world);
    }

    #[test]
    fn inactive_published_map_relationship_does_not_advance_or_restore_route_registration_ownership_observability() {
        let materializer = crate::MapSignalMaterializer::<i32, i32>::new(|registrar| {
            let source = registrar.replayable_signal_map_source(
                BTreeMap::new(),
                |In(())| None::<Vec<crate::signal_map::MapDiff<i32, i32>>>,
                OutputDelivery::move_only(),
            )?;
            registrar.signal_map_entity(
                source,
                |In(diffs): In<Vec<crate::signal_map::MapDiff<i32, i32>>>| Some(diffs),
                OutputDelivery::move_only(),
            )
        });
        let mut world = lazy_cleanup_world();
        let claim = materializer.register(&mut world);
        let target = claim.entity();
        let source = world.get::<StaticSignalDependencies>(*target).unwrap().0[0];
        assert!(
            world
                .get::<SignalInputBuffer>(*target)
                .unwrap()
                .has_pending_replay_snapshot(source)
        );
        world.get_mut::<SignalInputBuffer>(*target).unwrap().clear();
        world.entity_mut(*target).insert(RegistrarInactive);
        world
            .get_mut::<ActiveSignalDependents>(*source)
            .unwrap()
            .0
            .remove(&target);

        let contract = world
            .get::<RegistrationReplayCacheState>(*source)
            .unwrap()
            .contract
            .clone();
        let accepted = contract
            .accept_output(
                &mut world,
                source,
                Box::new(vec![crate::signal_map::MapDiff::Insert { key: 1, value: 2 }]),
            )
            .expect("newer map output should be accepted");
        deliver_accepted_signal_output(&mut world, source, accepted, push_signal_input);

        assert!(world.get::<SignalInputBuffer>(*target).unwrap().0.is_empty());
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&target)),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));
        assert!(
            !world
                .get::<ActiveSignalDependents>(*source)
                .unwrap()
                .0
                .contains(&target)
        );

        world
            .get_mut::<ActiveSignalDependents>(*source)
            .unwrap()
            .0
            .insert(target);
        world.entity_mut(*target).remove::<RegistrarInactive>();
        claim.release(&mut world);
    }

    #[test]
    fn polling_does_not_advance_cursor_after_reentrant_inactivation_registration_ownership_observability() {
        #[derive(Resource, Default)]
        struct PollDeactivateTarget(Option<SignalEntity>);

        let materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let source = registrar.replayable_signal_source(
                0,
                |In(()), world: &mut World| {
                    if let Some(target) = world.resource::<PollDeactivateTarget>().0 {
                        world.entity_mut(*target).insert(RegistrarInactive);
                    }
                    Some(1)
                },
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
        });
        let retained = materializer.shared_clone();
        let mut world = lazy_cleanup_world();
        world.insert_resource(PollDeactivateTarget::default());
        let claim = materializer.register(&mut world);
        let terminal = claim.entity();
        let source = world.get::<StaticSignalDependencies>(*terminal).unwrap().0[0];
        world.resource_mut::<PollDeactivateTarget>().0 = Some(terminal);
        world.get_mut::<SignalInputBuffer>(*terminal).unwrap().clear();
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&terminal)),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));

        assert!(run_signal_subgraph_once_one_shot(In(terminal), &mut world).is_none());

        assert!(world.get::<SignalInputBuffer>(*terminal).unwrap().0.is_empty());
        assert!(matches!(
            world
                .get::<ReplayConsumingRelationships>(*source)
                .and_then(|relationships| relationships.0.get(&terminal)),
            Some(ReplayRelationshipState::Published { cursor }) if *cursor == ReplayRevision::FIRST
        ));
        world.entity_mut(*terminal).remove::<RegistrarInactive>();
        claim.release(&mut world);
        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn graph_snapshot_uses_entity_generation_registration_ownership_observability() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let stale_entity = world.spawn_empty().id();
        let next_gen = world.despawn_no_free(stale_entity).unwrap();
        let live_entity = world.spawn_empty_at(next_gen).unwrap().id();
        assert_eq!(stale_entity.index(), live_entity.index());
        assert_ne!(stale_entity.generation(), live_entity.generation());

        {
            let mut state = world.resource_mut::<SignalGraphState>();
            state.levels.insert(SignalEntity(stale_entity), 7);
            state.edge_change_seeds.insert(SignalEntity(live_entity));
        }

        let snapshot = graph_snapshot(&mut world);
        let mut expected_entities = vec![stale_entity, live_entity];
        expected_entities.sort();
        assert_eq!(
            snapshot.nodes.iter().map(|node| node.entity).collect::<Vec<_>>(),
            expected_entities,
            "same-index generations must remain distinct and fully ordered"
        );

        let stale = snapshot
            .node(stale_entity)
            .expect("stale cached generation should remain observable");
        assert!(!stale.exists);
        assert_eq!(stale.cached_level, Some(7));

        let live = snapshot
            .node(live_entity)
            .expect("live reused generation should remain observable");
        assert!(live.exists);
        assert!(live.edge_change_seed);
        assert_ne!(stale, live, "entity generation must participate in snapshot equality");
    }

    fn assert_direct_ownership_model_matches_world(world: &mut World, model: &DirectOwnershipModel) {
        let snapshot = graph_snapshot(world);
        let mut expected_nodes = model.dependencies.iter().map(|(entity, _)| *entity).collect::<Vec<_>>();
        expected_nodes.sort();
        let mut actual_managed_nodes = snapshot
            .nodes
            .iter()
            .filter(|node| node.exists && node.canonical_static_dependencies.is_some())
            .map(|node| node.entity)
            .collect::<Vec<_>>();
        actual_managed_nodes.sort();
        assert_eq!(actual_managed_nodes, expected_nodes);

        for entity in expected_nodes {
            let node = snapshot
                .node(entity)
                .unwrap_or_else(|| panic!("expected model node {entity:?} disappeared"));
            assert!(node.exists, "expected model node {entity:?} was reaped");
            assert!(
                node.canonical_static_dependencies.is_some(),
                "expected model node {entity:?} lost canonical registrar state"
            );
            validate_existing_registration_runtime(world, SignalEntity(entity))
                .unwrap_or_else(|error| panic!("expected model node {entity:?} became invalid: {error:?}"));
            assert_eq!(
                node.claim_count,
                Some(model.count(entity)),
                "direct count diverged for {entity:?}"
            );
        }

        let mut actual_routes = snapshot
            .nodes
            .iter()
            .flat_map(|node| {
                node.active_downstreams
                    .iter()
                    .map(move |dependent| (node.entity, *dependent))
            })
            .collect::<Vec<_>>();
        actual_routes.sort();
        actual_routes.dedup();
        assert_eq!(actual_routes, model.active_routes());
    }

    #[derive(Component)]
    struct GeneratedModelNode(usize);

    fn all_generated_nodes_reach_terminal(dependencies: &[Vec<usize>]) -> bool {
        let terminal = dependencies.len().saturating_sub(1);
        (0..terminal).all(|start| {
            let mut pending = vec![start];
            let mut visited = vec![false; dependencies.len()];
            while let Some(node) = pending.pop() {
                if node == terminal {
                    return true;
                }
                let Some(seen) = visited.get_mut(node) else {
                    return false;
                };
                if core::mem::replace(seen, true) {
                    continue;
                }
                pending.extend(
                    dependencies
                        .iter()
                        .enumerate()
                        .filter_map(|(dependent, inputs)| inputs.contains(&node).then_some(dependent)),
                );
            }
            false
        })
    }

    #[test]
    fn exhaustive_four_node_terminal_connected_dags_and_fixed_seed_claim_stream_match_independent_model_registration_ownership_observability()
     {
        let mut enumerated_graphs = 0;
        let mut maximum_simultaneous_claims = 0;
        let mut released_oldest = false;
        let mut released_newest = false;
        let mut released_middle = false;

        for edge_mask in 0_u8..64 {
            let mut expected_logical_dependencies = vec![Vec::new(); 4];
            let mut edge_bit = 0;
            for (dependent, dependencies) in expected_logical_dependencies.iter_mut().enumerate().skip(1) {
                for dependency in 0..dependent {
                    if edge_mask & (1 << edge_bit) != 0 {
                        dependencies.push(dependency);
                    }
                    edge_bit += 1;
                }
            }
            if !all_generated_nodes_reach_terminal(&expected_logical_dependencies) {
                continue;
            }
            enumerated_graphs += 1;

            for drain_strategy in 0_u64..3 {
                let generated_dependencies = expected_logical_dependencies.clone();
                let materializer = crate::ScalarSignalMaterializer::<i32>::new(move |registrar| {
                    let mut entities = Vec::with_capacity(generated_dependencies.len());
                    for (logical_node, logical_dependencies) in generated_dependencies.iter().enumerate() {
                        let entity = if logical_dependencies.is_empty() {
                            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?
                        } else {
                            let mut dependencies = logical_dependencies
                                .iter()
                                .map(|dependency| {
                                    entities
                                        .get(*dependency)
                                        .copied()
                                        .expect("ordered DAG dependencies must precede their dependent")
                                })
                                .collect::<Vec<_>>();
                            dependencies.push(
                                dependencies
                                    .first()
                                    .copied()
                                    .expect("non-root generated nodes have at least one dependency"),
                            );
                            registrar.signal_entity(
                                dependencies,
                                |In(value): In<i32>| Some(value),
                                OutputDelivery::broadcast(),
                            )?
                        };
                        registrar.install_entity_component(entity, move |world, entity, _dependencies| {
                            world.entity_mut(*entity).insert(GeneratedModelNode(logical_node));
                        })?;
                        entities.push(entity);
                    }
                    entities
                        .last()
                        .copied()
                        .ok_or(SignalRegistrationError::EntityUnavailable)
                });
                let retained = materializer.shared_clone();
                let mut world = lazy_cleanup_world();
                let first_claim = materializer.register(&mut world);
                let terminal = first_claim.entity();

                let mut logical_entities = vec![None; expected_logical_dependencies.len()];
                let mut generated_query = world.query::<(Entity, &GeneratedModelNode)>();
                for (entity, marker) in generated_query.iter(&world) {
                    let slot = logical_entities
                        .get_mut(marker.0)
                        .expect("generated node marker must name a logical node");
                    assert!(
                        slot.replace(entity).is_none(),
                        "each logical node must map to one entity"
                    );
                }
                let logical_entities = logical_entities
                    .into_iter()
                    .map(|entity| entity.expect("every generated logical node must be committed"))
                    .collect::<Vec<_>>();
                assert_eq!(logical_entities.last().copied(), Some(terminal.entity()));

                let mut model = DirectOwnershipModel::default();
                for (logical_node, expected_dependencies) in expected_logical_dependencies.iter().enumerate() {
                    let entity = *logical_entities
                        .get(logical_node)
                        .expect("each logical node has a committed identity");
                    let mut expected_dependencies = expected_dependencies
                        .iter()
                        .map(|dependency| {
                            *logical_entities
                                .get(*dependency)
                                .expect("each logical dependency has a committed identity")
                        })
                        .collect::<Vec<_>>();
                    expected_dependencies
                        .sort_by(|left, right| signal_identity_cmp(&SignalEntity(*left), &SignalEntity(*right)));
                    let stored_dependencies = world
                        .get::<StaticSignalDependencies>(entity)
                        .unwrap_or_else(|| panic!("logical node {logical_node} lost canonical dependencies"))
                        .0
                        .iter()
                        .map(|dependency| dependency.entity())
                        .collect::<Vec<_>>();
                    assert_eq!(
                        stored_dependencies, expected_dependencies,
                        "raw canonical dependencies must match the generator for logical node {logical_node}"
                    );
                    model.add_node(entity, expected_dependencies);
                }

                let mut claims = vec![first_claim];
                let mut tokens = vec![model.acquire_external(terminal.entity())];
                assert_direct_ownership_model_matches_world(&mut world, &model);

                for _ in 0..8 {
                    claims.push(retained.shared_clone().register(&mut world));
                    tokens.push(model.acquire_external(terminal.entity()));
                    maximum_simultaneous_claims = maximum_simultaneous_claims.max(claims.len());
                    assert_direct_ownership_model_matches_world(&mut world, &model);
                }

                claims.remove(0).release(&mut world);
                model.release(tokens.remove(0));
                released_oldest = true;
                assert_direct_ownership_model_matches_world(&mut world, &model);

                claims
                    .pop()
                    .expect("the acquisition burst leaves a newest claim")
                    .release(&mut world);
                model.release(tokens.pop().expect("the newest claim has a model token"));
                released_newest = true;
                assert_direct_ownership_model_matches_world(&mut world, &model);

                let middle = claims.len() / 2;
                claims.remove(middle).release(&mut world);
                model.release(tokens.remove(middle));
                released_middle = true;
                assert_direct_ownership_model_matches_world(&mut world, &model);

                let mut operation_state = u64::from(edge_mask)
                    .wrapping_add(drain_strategy.wrapping_mul(0xd1b5_4a32_d192_ed03))
                    .wrapping_add(0x9e37_79b9_7f4a_7c15);
                for _ in 0..96 {
                    operation_state = operation_state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let acquire = ((operation_state >> 37) % 5) < 3;
                    if claims.is_empty() || (acquire && claims.len() < 12) {
                        let claim = retained.shared_clone().register(&mut world);
                        assert_eq!(claim.entity(), terminal);
                        claims.push(claim);
                        tokens.push(model.acquire_external(terminal.entity()));
                        maximum_simultaneous_claims = maximum_simultaneous_claims.max(claims.len());
                    } else {
                        operation_state = operation_state
                            .wrapping_mul(2_862_933_555_777_941_757)
                            .wrapping_add(3_037_000_493);
                        let index = match (operation_state >> 43) % 3 {
                            0 => 0,
                            1 => claims.len() - 1,
                            _ => claims.len() / 2,
                        };
                        claims.remove(index).release(&mut world);
                        model.release(tokens.remove(index));
                    }
                    assert_direct_ownership_model_matches_world(&mut world, &model);
                }

                while !claims.is_empty() {
                    let index = match drain_strategy {
                        0 => 0,
                        1 => claims.len() - 1,
                        _ => claims.len() / 2,
                    };
                    claims.remove(index).release(&mut world);
                    model.release(tokens.remove(index));
                    assert_direct_ownership_model_matches_world(&mut world, &model);
                }

                drop(retained);
                cleanup::drain_world_cleanup(&mut world);
                for entity in &logical_entities {
                    assert!(world.get_entity(*entity).is_err(), "generated entity {entity:?} leaked");
                }
                let graph = world.resource::<SignalGraphState>();
                for entity in &logical_entities {
                    let signal = SignalEntity(*entity);
                    assert!(!graph.levels.contains_key(&signal));
                    assert!(!graph.signal_schedules.contains_key(&signal));
                    assert!(!graph.edge_change_seeds.contains(&signal));
                    assert!(!graph.deferred_removals.contains(&signal));
                    assert!(
                        graph
                            .by_schedule
                            .values()
                            .flatten()
                            .flatten()
                            .all(|member| member != &signal)
                    );
                }
            }
        }

        assert_eq!(enumerated_graphs, 21);
        assert_eq!(maximum_simultaneous_claims, 12);
        assert!(released_oldest && released_newest && released_middle);
    }

    #[test]
    fn direct_count_model_registration_ownership_observability() {
        let mut world = World::new();
        let source = world.spawn_empty().id();
        let terminal = world.spawn_empty().id();

        let mut model = DirectOwnershipModel::default();
        model.add_node(source, []);
        model.add_node(terminal, [source, source]);
        let first = model.acquire_external(terminal);
        let second = model.acquire_external(terminal);

        assert_eq!(model.count(terminal), 2);
        assert_eq!(
            model.count(source),
            1,
            "an already-active dependent owns one canonical immediate dependency claim"
        );
        assert_eq!(model.active_routes(), vec![(source, terminal)]);

        model.release(first);
        assert_eq!(model.count(terminal), 1);
        assert_eq!(model.count(source), 1);
        model.release(second);
        assert_eq!(model.count(terminal), 0);
        assert_eq!(model.count(source), 0);
        assert!(model.active_routes().is_empty());

        let left = world.spawn_empty().id();
        let right = world.spawn_empty().id();
        let join = world.spawn_empty().id();
        let mut diamond = DirectOwnershipModel::default();
        diamond.add_node(source, []);
        diamond.add_node(left, [source]);
        diamond.add_node(right, [source]);
        diamond.add_node(join, [left, right]);
        let external = diamond.acquire_external(join);
        let dynamic = diamond.acquire_dynamic(join);

        assert_eq!(diamond.count(join), 2);
        assert_eq!(diamond.count(left), 1);
        assert_eq!(diamond.count(right), 1);
        assert_eq!(diamond.count(source), 2);

        diamond.release(dynamic);
        assert_eq!(diamond.count(join), 1);
        assert_eq!(diamond.count(source), 2);
        diamond.release(external);
        assert_eq!(diamond.count(join), 0);
        assert_eq!(diamond.count(left), 0);
        assert_eq!(diamond.count(right), 0);
        assert_eq!(diamond.count(source), 0);
    }

    fn lazy_cleanup_world() -> World {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());
        world
    }

    #[test]
    fn lazy_world_cleanup_candidate_order_uses_full_entity_identity() {
        let mut world = lazy_cleanup_world();
        let stale_entity = world.spawn_empty().id();
        let next_gen = world.despawn_no_free(stale_entity).unwrap();
        let live_entity = world.spawn_empty_at(next_gen).unwrap().id();
        assert_eq!(stale_entity.index(), live_entity.index());
        assert_ne!(stale_entity.generation(), live_entity.generation());

        let stale = SignalEntity(stale_entity);
        let live = SignalEntity(live_entity);
        let identity_order = signal_identity_cmp(&stale, &live);
        assert_ne!(identity_order, core::cmp::Ordering::Equal);
        let expected = if identity_order.is_lt() {
            [stale, live]
        } else {
            [live, stale]
        };

        let first = order_signal_candidates(
            &world,
            vec![
                ReleasedSignalCandidate::new(live),
                ReleasedSignalCandidate::new(stale),
                ReleasedSignalCandidate::new(live),
                ReleasedSignalCandidate::new(stale),
            ],
        );
        let second = order_signal_candidates(
            &world,
            vec![
                ReleasedSignalCandidate::new(stale),
                ReleasedSignalCandidate::new(live),
                ReleasedSignalCandidate::new(stale),
                ReleasedSignalCandidate::new(live),
            ],
        );

        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn canonical_output_delivery_key_mismatch_rejects_without_mutation() {
        let mut world = lazy_cleanup_world();
        let move_materializer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let retained = move_materializer.shared_clone();
        let entity =
            materialize_materializer(&mut world, move_materializer).expect("move-only entity should materialize");
        let before = graph_snapshot(&mut world);

        verify_existing_canonical_output_delivery(
            &mut world,
            entity,
            OutputDeliveryKey {
                kind: OutputDeliveryKind::Move,
                output_type: TypeId::of::<i32>(),
            },
        )
        .expect("matching canonical output delivery key should succeed");
        assert_eq!(graph_snapshot(&mut world), before);

        assert_eq!(
            verify_existing_canonical_output_delivery(
                &mut world,
                entity,
                OutputDeliveryKey {
                    kind: OutputDeliveryKind::Broadcast,
                    output_type: TypeId::of::<i32>(),
                },
            ),
            Err(SignalRegistrationError::ConflictingCanonicalState),
        );
        assert_eq!(
            verify_existing_canonical_output_delivery(
                &mut world,
                entity,
                OutputDeliveryKey {
                    kind: OutputDeliveryKind::Move,
                    output_type: TypeId::of::<u64>(),
                },
            ),
            Err(SignalRegistrationError::ConflictingCanonicalState),
        );
        assert_eq!(graph_snapshot(&mut world), before);

        drop(retained);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn incoherent_stored_output_delivery_rejects_atomically() {
        fn dummy_clone(_value: &dyn Any) -> ErasedSignalValue {
            Box::new(0_i32)
        }

        for incoherent in [
            StoredOutputDelivery::test_incoherent(
                OutputDeliveryKey {
                    kind: OutputDeliveryKind::Move,
                    output_type: TypeId::of::<i32>(),
                },
                Some(dummy_clone),
            ),
            StoredOutputDelivery::test_incoherent(
                OutputDeliveryKey {
                    kind: OutputDeliveryKind::Broadcast,
                    output_type: TypeId::of::<i32>(),
                },
                None,
            ),
            StoredOutputDelivery::test_incoherent(
                OutputDeliveryKey {
                    kind: OutputDeliveryKind::Broadcast,
                    output_type: TypeId::of::<u64>(),
                },
                Some(dummy_clone),
            ),
        ] {
            let mut world = lazy_cleanup_world();
            let (retained, claim, _source, terminal) = active_unary_registration_fixture(&mut world);
            let attempt = retained.shared_clone();
            world.entity_mut(*terminal).insert(incoherent);
            assert_runtime_inconsistency_is_atomic(&mut world, attempt, terminal);
            world
                .entity_mut(*terminal)
                .insert(OutputDelivery::<i32>::broadcast().into_stored());
            claim.release(&mut world);
            drop(retained);
            cleanup::drain_world_cleanup(&mut world);
        }
    }

    #[test]
    fn mid_pass_dynamic_level_promotion_skips_stale_bucket_and_executes_once_in_topological_order() {
        use crate::{
            DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, ReplayInitialization,
            SameSignalIdentity, ScalarSlotCommand,
            signal::{Signal, SignalExt},
        };
        use bevy_ecs::prelude::ResMut;

        #[derive(Resource, Default)]
        struct DeepDrive(Option<i32>);

        struct SharedDeepSelected(crate::ScalarSignalMaterializer<i32>);

        impl Signal for SharedDeepSelected {
            type Item = i32;

            fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<i32> {
                self.0
            }
        }

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::new::<PostUpdate>()));
        app.init_resource::<DeepDrive>();
        // Restrict to a single pass so premature execution at a stale level cannot be masked by a
        // retry pass.
        app.world_mut()
            .resource_mut::<SignalGraphState>()
            .registration_recursion_limit = 0;

        let execution_order = Arc::new(bevy_platform::sync::Mutex::new(Vec::<&'static str>::new()));
        let deep_materializer = crate::ScalarSignalMaterializer::<i32>::new({
            let execution_order = execution_order.clone();
            move |registrar| {
                let d0 = registrar.signal_entity(
                    (),
                    |In(()), mut drive: ResMut<DeepDrive>| drive.0.take(),
                    OutputDelivery::move_only(),
                )?;
                let d1 = registrar.signal_entity(d0, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
                let d2 = registrar.signal_entity(d1, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
                let d3 = registrar.signal_entity(d2, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
                registrar.replayable_signal_entity(
                    d3,
                    move |In(v): In<i32>| {
                        execution_order
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push("deep_selected");
                        Some(v)
                    },
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            }
        });
        let deep_cached = SharedDeepSelected(deep_materializer).cache().broadcast();
        // Keep the deep chain active at levels 0..=4 and warm it to 10 in frame 0.
        let deep_claim = deep_cached.clone().register(app.world_mut());
        app.world_mut().resource_mut::<DeepDrive>().0 = Some(10);
        process_signal_graph_for_schedule(PostUpdate.intern())(app.world_mut());
        execution_order
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();

        // Register dynamic owner + sink: manager (0) -> manager_bridge (1) -> owner (2) -> sink
        // (3).
        let select_once = Arc::new(bevy_platform::sync::Mutex::new(Some(deep_cached)));
        let observed_batches = Arc::new(bevy_platform::sync::Mutex::new(Vec::<Vec<i32>>::new()));
        let owner_claim = crate::ScalarSignalMaterializer::<()>::new({
            let select_once = select_once.clone();
            let execution_order = execution_order.clone();
            let observed_batches = observed_batches.clone();
            move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        select_once
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .take()
                            .map(|signal| {
                                DynamicManagerUpdate::new(
                                    [ScalarSlotCommand::Select { signal }],
                                    DynamicEmissions::None,
                                )
                            })
                    },
                    OutputDelivery::move_only(),
                )?;
                let owner = registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(move |queued| {
                            execution_order
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .push("owner");
                            (!queued.is_empty()).then_some(queued)
                        })
                        .same_signal_identity(SameSignalIdentity::Restart),
                )?;
                registrar.signal_entity(
                    owner,
                    move |In(batch): In<Vec<i32>>| {
                        observed_batches
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(batch);
                        Some(())
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(app.world_mut());

        // Drive a fresh value (20) through d0..deep_selected in the same pass that manager_bridge
        // promotes owner (2 -> 6) and sink (3 -> 7). Owner and sink must skip their stale buckets
        // and execute once in topological order after deep_selected updates to 20.
        app.world_mut().resource_mut::<DeepDrive>().0 = Some(20);
        process_signal_graph_for_schedule(PostUpdate.intern())(app.world_mut());

        assert_eq!(
            *execution_order.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec!["deep_selected", "owner"]
        );
        assert_eq!(
            *observed_batches.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![10, 20]]
        );

        owner_claim.release(app.world_mut());
        deep_claim.release(app.world_mut());
        cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
    }

    #[test]
    fn cross_schedule_after_output_cleanup_flushes_at_end_of_producer_pass() {
        use crate::{
            DynamicEmissions, DynamicManagerStep, DynamicManagerUpdate, DynamicSignalOperatorSpec,
            ReplayInitialization, SameSignalIdentity, ScalarSlotCommand,
            signal::{BoxedReplayableSignal, Signal, SignalExt},
        };
        use bevy_ecs::prelude::ResMut;

        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash, ScheduleLabel)]
        struct ProducerSchedule;

        #[derive(Resource, Default)]
        struct ProducerTransientValue(Option<i32>);

        struct ProducerWarmingSignal(crate::ScalarSignalMaterializer<i32>);

        impl Signal for ProducerWarmingSignal {
            type Item = i32;

            fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<i32> {
                self.0
            }
        }

        let mut app = App::new();
        app.init_schedule(ProducerSchedule);
        app.add_plugins((
            MinimalPlugins,
            JonmoPlugin::new::<PostUpdate>().with_schedule::<ProducerSchedule>(),
        ));
        app.init_resource::<ProducerTransientValue>();

        let transient_producer: BoxedReplayableSignal<i32> =
            ProducerWarmingSignal(crate::ScalarSignalMaterializer::new(|registrar| {
                let entity = registrar.replayable_signal_entity(
                    (),
                    |In(()), mut pending: ResMut<ProducerTransientValue>| pending.0.take(),
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )?;
                registrar.apply_schedule(entity, ProducerSchedule)
            }))
            .cache()
            .boxed_replayable();
        let ready_final: BoxedReplayableSignal<i32> = crate::signal::once(5).cache().boxed_replayable();

        type Step = DynamicManagerStep<ScalarSlotCommand<BoxedReplayableSignal<i32>>, i32>;
        let initial_update = Arc::new(bevy_platform::sync::Mutex::new(Some(DynamicManagerUpdate::sequenced(
            [
                Step::command(
                    ScalarSlotCommand::Select {
                        signal: transient_producer,
                    },
                    DynamicEmissions::None,
                ),
                Step::command(
                    ScalarSlotCommand::Select { signal: ready_final },
                    DynamicEmissions::None,
                ),
            ],
        ))));

        let observed = Arc::new(bevy_platform::sync::Mutex::new(Vec::<Vec<i32>>::new()));
        let sink_claim = crate::ScalarSignalMaterializer::<()>::new({
            let initial_update = initial_update.clone();
            let observed = observed.clone();
            move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        initial_update
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .take()
                    },
                    OutputDelivery::move_only(),
                )?;
                let owner = registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                        .same_signal_identity(SameSignalIdentity::Restart),
                )?;
                registrar.signal_entity(
                    owner,
                    move |In(batch): In<Vec<i32>>| {
                        observed
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(batch);
                        Some(())
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(app.world_mut());
        let owner = get_static_dependencies(app.world(), sink_claim.entity())[0];

        // Stage the pending cohort on PostUpdate (owner's schedule); transient_producer is still
        // Warming.
        process_signal_graph_for_schedule(PostUpdate.intern())(app.world_mut());
        let staged = world_dynamic_subscriptions(app.world(), owner);
        assert_eq!(staged.len(), 2);
        let transient_adapter = staged[0].adapter;

        // Warm transient_producer on ProducerSchedule: transient_adapter runs on ProducerSchedule,
        // defers self-release during its own execution, and end-of-pass cleanup immediately flushes
        // after_output_cleanup BEFORE PostUpdate runs!
        app.world_mut().resource_mut::<ProducerTransientValue>().0 = Some(77);
        process_signal_graph_for_schedule(ProducerSchedule.intern())(app.world_mut());

        assert_eq!(
            world_dynamic_subscriptions(app.world(), owner).len(),
            1,
            "retired cross-schedule transient adapter must be flushed at the end of ProducerSchedule"
        );
        assert!(
            app.world().get_entity(*transient_adapter).is_err()
                || app.world().get::<RegistrarInactive>(*transient_adapter).is_some(),
            "retired transient adapter must no longer be active after ProducerSchedule completes"
        );

        // Now run PostUpdate: owner still consumes the woken cohort output [77, 5].
        process_signal_graph_for_schedule(PostUpdate.intern())(app.world_mut());
        assert_eq!(
            *observed.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![77, 5]]
        );

        sink_claim.release(app.world_mut());
        cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
    }

    fn world_dynamic_subscriptions(world: &World, owner: SignalEntity) -> Vec<DynamicSubscriptionInspection> {
        world
            .get::<DynamicSignalOwner>(*owner)
            .expect("dynamic owner component")
            .subscriptions()
    }

    #[test]
    fn dynamic_connection_preflight_rejects_graph_depth_overflow() {
        let mut world = lazy_cleanup_world();
        let adapter_chain = crate::ScalarSignalMaterializer::<()>::new(|registrar| {
            let selected = registrar.signal_entity((), |In(())| Some(1_i32), OutputDelivery::move_only())?;
            registrar.signal_entity(selected, |In(_): In<i32>| Some(()), OutputDelivery::move_only())
        });
        let downstream_chain = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let owner = registrar.signal_entity((), |In(())| Some(1_i32), OutputDelivery::move_only())?;
            registrar.signal_entity(owner, |In(v): In<i32>| Some(v + 1), OutputDelivery::move_only())
        });

        let adapter_claim = adapter_chain.register(&mut world);
        let downstream_claim = downstream_chain.register(&mut world);
        let adapter = adapter_claim.entity();
        let downstream = downstream_claim.entity();
        let owner = get_static_dependencies(&world, downstream)[0];

        let plan = RegistrationPreflight {
            entities: Vec::new().into_boxed_slice(),
            schedules: Vec::new().into_boxed_slice(),
            claim_deltas: Vec::new().into_boxed_slice(),
            dynamic_connections: vec![crate::registration::RegistrationDynamicConnectionPreflight {
                source: PlannedSignalEntityRef::Existing(adapter),
                target: PlannedSignalEntityRef::Existing(owner),
            }]
            .into_boxed_slice(),
        };

        // Chain depths with adapter -> owner connected:
        // selected (0) -> adapter (1) -> owner (2) -> downstream (3).
        assert_eq!(preflight_dynamic_connections_with_limit(&world, &plan, 3), Ok(()));
        assert_eq!(
            preflight_dynamic_connections_with_limit(&world, &plan, 2),
            Err(SignalRegistrationError::GraphDepthOverflow),
            "overflow on owner's active downstream subgraph must be rejected before commit"
        );
        assert_eq!(
            preflight_dynamic_connections_with_limit(&world, &plan, 1),
            Err(SignalRegistrationError::GraphDepthOverflow),
            "overflow on owner itself must be rejected before commit"
        );

        downstream_claim.release(&mut world);
        adapter_claim.release(&mut world);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn retained_replay_caches_preserve_monotonic_ready_state_across_deactivation() {
        use crate::{ReplayInitialization, signal_map::MapDiff, signal_vec::VecDiff};

        let mut world = lazy_cleanup_world();
        let outer = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let warming_scalar = registrar.replayable_signal_entity(
                (),
                |In(())| Some(11_i32),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )?;
            let warming_vec = registrar.replayable_signal_vec_entity(
                (),
                |In(())| Some(vec![VecDiff::Replace { values: vec![22_i32] }]),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )?;
            let warming_map = registrar.replayable_signal_map_entity(
                (),
                |In(())| {
                    Some(vec![MapDiff::Replace {
                        entries: vec![(1_i32, 33_i32)],
                    }])
                },
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )?;
            let source_scalar =
                registrar.replayable_signal_source(10_i32, |In(())| Some(20_i32), OutputDelivery::broadcast())?;
            let s1 = registrar.signal_entity(warming_scalar, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
            let s2 = registrar.signal_entity(
                warming_vec,
                |In(_): In<Vec<VecDiff<i32>>>| Some(1_i32),
                OutputDelivery::move_only(),
            )?;
            let s3 = registrar.signal_entity(
                warming_map,
                |In(_): In<Vec<MapDiff<i32, i32>>>| Some(1_i32),
                OutputDelivery::move_only(),
            )?;
            let s4 = registrar.signal_entity(source_scalar, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
            registrar.signal_entity([s1, s2, s3, s4], |In(v): In<i32>| Some(v), OutputDelivery::move_only())
        });
        let retained_outer = outer.shared_clone();
        let claim = outer.register(&mut world);
        let schedule = world.resource::<SignalGraphState>().default_schedule;
        process_signal_graph_for_schedule(schedule)(&mut world);

        let terminal = claim.entity();
        let middle = get_static_dependencies(&world, terminal);
        assert_eq!(middle.len(), 4);
        let mut warming_scalar = None;
        let mut warming_vec = None;
        let mut warming_map = None;
        let mut source_scalar = None;
        for node in middle {
            let dep = get_static_dependencies(&world, node)[0];
            if inspect_vec_replay_cache::<i32>(&world, dep).is_some() {
                warming_vec = Some(dep);
            } else if inspect_map_replay_cache::<i32, i32>(&world, dep).is_some() {
                warming_map = Some(dep);
            } else if matches!(
                inspect_scalar_replay_cache::<i32>(&world, dep),
                Some(ReplayCacheInspection::Ready { current: 11, .. })
            ) {
                warming_scalar = Some(dep);
            } else {
                source_scalar = Some(dep);
            }
        }
        let warming_scalar = warming_scalar.expect("warming scalar entity");
        let warming_vec = warming_vec.expect("warming vec entity");
        let warming_map = warming_map.expect("warming map entity");
        let source_scalar = source_scalar.expect("source scalar entity");

        // Release 1 -> 0 while retained_outer keeps all static dependencies materialized.
        // Per the replay contract, readiness is monotonic while retained across deactivation.
        claim.release(&mut world);

        assert!(matches!(
            inspect_scalar_replay_cache::<i32>(&world, warming_scalar),
            Some(ReplayCacheInspection::Ready { current: 11, .. })
        ));
        assert!(matches!(
            inspect_vec_replay_cache::<i32>(&world, warming_vec),
            Some(ReplayCacheInspection::Ready { .. })
        ));
        assert!(matches!(
            inspect_map_replay_cache::<i32, i32>(&world, warming_map),
            Some(ReplayCacheInspection::Ready { .. })
        ));
        assert!(matches!(
            inspect_scalar_replay_cache::<i32>(&world, source_scalar),
            Some(ReplayCacheInspection::Ready { current: 20, .. })
        ));

        drop(retained_outer);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn one_shot_subgraph_evaluation_respects_silent_and_warn_recursion_limits() {
        use crate::{
            DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, SameSignalIdentity, ScalarSlotCommand,
            signal::SignalExt,
        };

        for behavior in [RecursionLimitBehavior::Silent, RecursionLimitBehavior::Warn] {
            let mut app = App::new();
            app.add_plugins((
                MinimalPlugins,
                JonmoPlugin::default()
                    .registration_recursion_limit(0)
                    .on_recursion_limit_exceeded(behavior),
            ));

            let claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
                let manager = registrar.signal_entity(
                    (),
                    |In(())| {
                        Some(DynamicManagerUpdate::new(
                            [ScalarSlotCommand::Select {
                                signal: crate::signal::once(7_i32).cache(),
                            }],
                            DynamicEmissions::None,
                        ))
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(|queued| queued.last().copied())
                        .same_signal_identity(SameSignalIdentity::Restart),
                )
            })
            .register(app.world_mut());

            // Pass 0 attaches a new dynamic upstream and breaks to recompute topology; Pass 1 hits
            // `pass > 0` and exits cleanly without panicking under Silent and Warn.
            assert!(run_signal_subgraph_once_internal(app.world_mut(), claim.entity()).is_none());

            claim.release(app.world_mut());
            cleanup::drain_world_cleanup(app.world_mut());
            app.world_mut().flush();
        }
    }

    #[test]
    #[should_panic(expected = "Signal registration recursion limit exceeded")]
    fn one_shot_subgraph_evaluation_panics_when_recursion_limit_behavior_is_panic() {
        use crate::{
            DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, SameSignalIdentity, ScalarSlotCommand,
            signal::SignalExt,
        };

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            JonmoPlugin::default()
                .registration_recursion_limit(0)
                .on_recursion_limit_exceeded(RecursionLimitBehavior::Panic),
        ));

        let _claim = crate::ScalarSignalMaterializer::<i32>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                |In(())| {
                    Some(DynamicManagerUpdate::new(
                        [ScalarSlotCommand::Select {
                            signal: crate::signal::once(7_i32).cache(),
                        }],
                        DynamicEmissions::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|queued| queued.last().copied())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        })
        .register(app.world_mut());

        let _ = run_signal_subgraph_once_internal(app.world_mut(), _claim.entity());
    }
}
