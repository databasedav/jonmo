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
//!    processed in deterministic order (sorted by entity index) for reproducible behavior.
//! 3. **Output forwarding**: When a signal produces [`Some`] value, that value is forwarded as
//!    input to all downstream signals. Returning [`None`] terminates propagation for that branch.
//! 4. **Dynamic registration**: Signals registered during processing (e.g., from UI element
//!    spawning) are integrated and processed within the same frame, avoiding one-frame delays.
//!
//! A signal with multiple upstreams runs **once per upstream** that fires in a given frame. This
//! allows a signal to act as a collection point, processing each upstream's output in turn.
//! However, only the **final output** (from the last run) is forwarded to downstream signals,
//! ensuring that downstreams see a single, consolidated result rather than receiving multiple
//! inputs.
//!
//! # Lifecycle
//!
//! Signal systems are usage-tracked; each call to [`.register`](super::signal::SignalExt::register)
//! increments a registration count on the signal and its upstreams. To release a signal,
//! [`SignalHandle::cleanup`] must be called, either explicitly or implicitly by storing handles in
//! a [`SignalHandles`] component (which calls cleanup when the entity is despawned). When a
//! signal's registration count reaches zero and no downstream dependents remain, the signal's
//! backing system is automatically despawned.
//!
//! # Polling
//!
//! In addition to the standard push-based flow, signals can be polled synchronously to retrieve
//! their most recent output. This is useful when a system needs to read signal state on-demand
//! rather than receiving it as pushed input.
use alloc::collections::VecDeque;
use bevy_app::PostUpdate;
use bevy_derive::Deref;
use bevy_ecs::{
    entity_disabling::Internal,
    lifecycle::HookContext,
    prelude::*,
    query::{QueryData, QueryFilter},
    schedule::{InternedScheduleLabel, ScheduleLabel},
    system::{SystemId, SystemState},
    world::DeferredWorld,
};
use bevy_platform::{
    collections::{HashMap, HashSet},
    prelude::*,
    sync::{
        Arc, LazyLock, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};
use core::{any::Any, hash::Hash, marker::PhantomData};
use dyn_clone::{DynClone, clone_trait_object};

/// Newtype wrapper for [`Entity`]s that hold systems in the signal graph.
#[derive(Clone, Copy, Deref, Debug, PartialEq, Eq, Hash)]
pub struct SignalSystem(pub Entity);

impl From<Entity> for SignalSystem {
    fn from(entity: Entity) -> Self {
        Self(entity)
    }
}

impl<I: 'static, O> From<SystemId<In<I>, O>> for SignalSystem {
    fn from(system_id: SystemId<In<I>, O>) -> Self {
        system_id.entity().into()
    }
}

#[derive(Component, Deref)]
pub(crate) struct SignalRegistrationCount(i32);

impl SignalRegistrationCount {
    fn new() -> Self {
        Self(1)
    }

    fn increment(&mut self) {
        self.0 += 1
    }

    fn decrement(&mut self) {
        self.0 -= 1
    }
}

/// Component on signal system entities indicating which schedule they run in.
///
/// Used by the multi-schedule processing system to determine which signals
/// to run during each schedule's processing pass.
#[derive(Component, Clone, Copy)]
pub(crate) struct SignalScheduleTag(pub(crate) InternedScheduleLabel);

/// Component for downstream schedule inheritance during registration.
///
/// When a signal is connected to an upstream via [`pipe_signal`], if the upstream
/// has a `ScheduleHint`, the downstream inherits the schedule (unless it already
/// has a [`SignalScheduleTag`]).
#[derive(Component, Clone, Copy)]
pub(crate) struct ScheduleHint(pub(crate) InternedScheduleLabel);

/// Apply schedule tagging to a signal: tag the signal itself, propagate to unscheduled
/// upstreams, and set hint for downstream inheritance.
///
/// This is the common logic used by [`SignalExt::schedule`](super::signal::SignalExt::schedule),
/// [`SignalVecExt::schedule`](super::signal_vec::SignalVecExt::schedule), and
/// [`SignalMapExt::schedule`](super::signal_map::SignalMapExt::schedule).
pub(crate) fn apply_schedule_to_signal(world: &mut World, signal: SignalSystem, schedule: InternedScheduleLabel) {
    // Directly tag caller (overwrites any inherited schedule)
    world.entity_mut(*signal).insert(SignalScheduleTag(schedule));

    // Propagate to unscheduled upstreams
    tag_unscheduled_upstreams(world, signal, schedule);

    // Set hint for downstream inheritance
    world.entity_mut(*signal).insert(ScheduleHint(schedule));
}

/// Tags all upstream signals that don't already have a [`SignalScheduleTag`].
///
/// Traverses the upstream graph from `start` and applies the given `schedule` to any
/// signal that hasn't been explicitly scheduled. This ensures that when a downstream
/// signal is scheduled, its entire upstream chain runs in a compatible schedule.
pub(crate) fn tag_unscheduled_upstreams(world: &mut World, start: SignalSystem, schedule: InternedScheduleLabel) {
    let mut stack = vec![start];
    let mut visited = HashSet::new();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }

        // Tag if not already tagged
        if world.get::<SignalScheduleTag>(*current).is_none() {
            world.entity_mut(*current).insert(SignalScheduleTag(schedule));
        }

        // Continue to upstreams
        if let Some(upstream) = world.get::<Upstream>(*current) {
            for &up in upstream.iter() {
                if !visited.contains(&up) {
                    stack.push(up);
                }
            }
        }
    }
}

pub(crate) fn register_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: 'static,
    O: Clone + Send + Sync + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<I>, IOO, M> + Send + Sync + 'static,
{
    lazy_signal_from_system(system).register(world)
}

// TODO: many to many relationships
#[derive(Component, Deref, Clone)]
pub(crate) struct Upstream(pub(crate) HashSet<SignalSystem>);

impl<'a> IntoIterator for &'a Upstream {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = bevy_platform::collections::hash_set::Iter<'a, SignalSystem>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// TODO: many to many relationships
#[derive(Component, Deref, Clone)]
pub(crate) struct Downstream(HashSet<SignalSystem>);

impl<'a> IntoIterator for &'a Downstream {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = bevy_platform::collections::hash_set::Iter<'a, SignalSystem>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// TODO: this isn't sensitive to switchers ?
fn would_create_cycle(world: &World, source: SignalSystem, target: SignalSystem) -> bool {
    if source == target {
        return true;
    }
    let mut stack = vec![target];
    let mut visited = HashSet::new();
    while let Some(node) = stack.pop() {
        if node == source {
            return true;
        }
        if visited.insert(node)
            && let Some(down) = world.get::<Downstream>(*node)
        {
            stack.extend(down.iter().copied());
        }
    }
    false
}

pub(crate) fn pipe_signal(world: &mut World, source: SignalSystem, target: SignalSystem) {
    if would_create_cycle(world, source, target) {
        // TODO: panic instead ?
        #[cfg(feature = "tracing")]
        bevy_log::error!("cycle detected when attempting to pipe {:?} → {:?}", source, target);
        return;
    }
    let mut upstream = world.entity_mut(*source);
    if let Some(mut downstream) = upstream.get_mut::<Downstream>() {
        downstream.0.insert(target);
    } else {
        upstream.insert(Downstream(HashSet::from([target])));
    }
    let mut downstream = world.entity_mut(*target);
    if let Some(mut upstream) = downstream.get_mut::<Upstream>() {
        upstream.0.insert(source);
    } else {
        downstream.insert(Upstream(HashSet::from([source])));
    }

    // Inherit schedule from upstream if target doesn't already have one
    if world.get::<SignalScheduleTag>(*target).is_none()
        && let Some(hint) = world.get::<ScheduleHint>(*source).copied()
    {
        world
            .entity_mut(*target)
            .insert(SignalScheduleTag(hint.0))
            .insert(ScheduleHint(hint.0)); // Pass it on to further downstreams
    }

    world
        .resource_mut::<SignalGraphState>()
        .edge_change_seeds
        .insert(target);
}

#[derive(Component, Clone)]
struct SystemRunner {
    #[allow(clippy::type_complexity)]
    runner: Arc<Box<dyn Fn(&mut World, Box<dyn Any>) -> Option<Box<dyn AnyClone>> + Send + Sync>>,
}

trait Runnable: Send + Sync {
    fn run(&self, w: &mut World, i: Box<dyn Any>) -> Option<Box<dyn AnyClone>>;
}

struct SystemHolder<I, O, S>
where
    I: 'static,
    O: 'static,
    S: Into<Option<O>>,
{
    system: SystemId<In<I>, S>,
    _marker: PhantomData<fn() -> O>,
}

impl<I, O, S> Runnable for SystemHolder<I, O, S>
where
    I: 'static,
    O: Clone + Send + Sync,
    S: Into<Option<O>> + 'static,
{
    fn run(&self, world: &mut World, input: Box<dyn Any>) -> Option<Box<dyn AnyClone>> {
        match input.downcast::<I>() {
            Ok(bx) => world
                .run_system_with(self.system, *bx)
                .ok()
                .and_then(Into::into)
                .map(|o| Box::new(o) as Box<dyn AnyClone>),
            Err(_) => {
                cfg_if::cfg_if! {
                    if #[cfg(feature = "tracing")] {
                        let expected_type = core::any::type_name::<I>();
                        bevy_log::error!(
                            "failed to downcast input for system {:?}. expected input type: `{}`",
                            self.system, expected_type
                        );
                    }
                }
                None
            }
        }
    }
}

impl SystemRunner {
    fn run(&self, world: &mut World, input: Box<dyn Any>) -> Option<Box<dyn AnyClone>> {
        (self.runner)(world, input)
    }
}

/// An extension trait for [`Any`] types that implement [`Clone`], [`Send`], and [`Sync`].
pub trait AnyClone: Any + DynClone + Send + Sync {}

clone_trait_object!(AnyClone);

impl<T: Clone + Send + Sync + 'static> AnyClone for T {}

/// Component that stores pending inputs for a signal.
///
/// Inputs are written by upstream signals and read by the signal during processing.
/// The buffer is cleared at the end of each frame.
#[derive(Component, Default)]
pub(crate) struct SignalInputBuffer(pub(crate) Vec<Box<dyn AnyClone>>);

impl SignalInputBuffer {
    /// Take all inputs, leaving the buffer empty.
    fn take(&mut self) -> Vec<Box<dyn AnyClone>> {
        core::mem::take(&mut self.0)
    }

    /// Push an input value.
    fn push(&mut self, value: Box<dyn AnyClone>) {
        self.0.push(value);
    }

    /// Clear the buffer.
    fn clear(&mut self) {
        self.0.clear();
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
    levels: HashMap<SignalSystem, u32>,
    /// Signals partitioned by schedule for efficient per-schedule iteration.
    /// Updated incrementally alongside level updates.
    by_schedule: HashMap<InternedScheduleLabel, Vec<Vec<SignalSystem>>>,
    /// Cache of which schedule each signal belongs to, for O(1) lookup during removal.
    signal_schedules: HashMap<SignalSystem, InternedScheduleLabel>,
    /// Signals that seed level recomputation after edge changes.
    edge_change_seeds: HashSet<SignalSystem>,
    /// Signals queued for removal while the graph is being processed.
    deferred_removals: HashSet<SignalSystem>,
    /// Whether the signal graph is currently being processed.
    is_processing: bool,
    /// Default schedule for signals without explicit scheduling.
    default_schedule: InternedScheduleLabel,
    /// Maximum number of recursive signal registration passes per frame.
    registration_recursion_limit: usize,
    /// What to do when the recursion limit is exceeded.
    on_recursion_limit_exceeded: RecursionLimitBehavior,
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
        }
    }
}

fn get_upstreams(world: &World, signal: SignalSystem) -> Vec<SignalSystem> {
    world
        .get::<Upstream>(*signal)
        .map(|u| u.0.iter().copied().collect())
        .unwrap_or_default()
}

fn get_downstreams(world: &World, signal: SignalSystem) -> Vec<SignalSystem> {
    world
        .get::<Downstream>(*signal)
        .map(|d| d.0.iter().copied().collect())
        .unwrap_or_default()
}

fn insert_sorted_by_index(bucket: &mut Vec<SignalSystem>, signal: SignalSystem) {
    if bucket.contains(&signal) {
        return;
    }
    let key = signal.index();
    let index = bucket.binary_search_by_key(&key, |s| s.index()).unwrap_or_else(|i| i);
    bucket.insert(index, signal);
}

/// Result of computing signal levels via Kahn's algorithm.
struct LevelComputeResult {
    /// Computed levels for each signal in the working set.
    levels: HashMap<SignalSystem, u32>,
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
    signals: &HashSet<SignalSystem>,
    upstream_filter: impl Fn(SignalSystem) -> bool,
    external_level: impl Fn(SignalSystem) -> Option<u32>,
) -> LevelComputeResult {
    if signals.is_empty() {
        return LevelComputeResult {
            levels: HashMap::new(),
            processed: 0,
            total: 0,
        };
    }

    let mut in_degree: HashMap<SignalSystem, usize> = HashMap::new();
    let mut upstreams_map: HashMap<SignalSystem, Vec<SignalSystem>> = HashMap::new();
    let mut downstreams_map: HashMap<SignalSystem, Vec<SignalSystem>> = HashMap::new();

    for &signal in signals {
        let upstreams = get_upstreams(world, signal);
        let local_in_degree = upstreams.iter().filter(|u| upstream_filter(**u)).count();
        in_degree.insert(signal, local_in_degree);
        // Build local downstreams_map by inverting upstream relationships.
        // This ensures we only traverse to signals in our working set.
        for &upstream in upstreams.iter().filter(|u| signals.contains(*u)) {
            downstreams_map.entry(upstream).or_default().push(signal);
        }
        upstreams_map.insert(signal, upstreams);
    }

    let mut queue: VecDeque<SignalSystem> = in_degree.iter().filter_map(|(s, d)| (*d == 0).then_some(*s)).collect();

    let mut levels: HashMap<SignalSystem, u32> = HashMap::new();
    let mut processed = 0usize;

    while let Some(signal) = queue.pop_front() {
        processed += 1;
        let upstreams = upstreams_map.get(&signal).cloned().unwrap_or_default();
        let level = upstreams
            .iter()
            .filter_map(|u| levels.get(u).copied().or_else(|| external_level(*u)))
            .max()
            .map(|m| m + 1)
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

fn bucket_levels_sorted(levels: &HashMap<SignalSystem, u32>) -> Vec<Vec<SignalSystem>> {
    let mut by_level: Vec<Vec<SignalSystem>> = Vec::new();
    for (&signal, &level) in levels {
        while by_level.len() <= level as usize {
            by_level.push(Vec::new());
        }
        by_level[level as usize].push(signal);
    }
    for bucket in &mut by_level {
        bucket.sort_by_key(|s| s.index());
    }
    by_level
}

// Computes a topological ordering of signals reachable downstream from `seeds`.
// Uses cached levels when available, falls back to BFS level computation for uncached signals.
fn downstream_levels_from_seeds(world: &World, seeds: &[SignalSystem]) -> Vec<Vec<SignalSystem>> {
    let state = world.resource::<SignalGraphState>();

    // Collect all reachable signals via BFS.
    let mut reachable: HashSet<SignalSystem> = HashSet::new();
    let mut queue: VecDeque<SignalSystem> = seeds.iter().copied().collect();
    while let Some(signal) = queue.pop_front() {
        if reachable.insert(signal) {
            queue.extend(get_downstreams(world, signal));
        }
    }

    // Check if all reachable signals have cached levels.
    let all_cached = reachable.iter().all(|s| state.levels.contains_key(s));

    if all_cached {
        // Fast path: use cached levels.
        let mut by_level: Vec<Vec<SignalSystem>> = Vec::new();
        for signal in reachable {
            let level = state.levels.get(&signal).copied().unwrap_or(0) as usize;
            while by_level.len() <= level {
                by_level.push(Vec::new());
            }
            by_level[level].push(signal);
        }
        for bucket in &mut by_level {
            bucket.sort_by_key(|s| s.index());
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

    let mut all_signals_state = SystemState::<Query<Entity, (With<SystemRunner>, Allow<Internal>)>>::new(world);
    let all_signals: HashSet<SignalSystem> = all_signals_state.get(world).iter().map(SignalSystem).collect();

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
            .get::<SignalScheduleTag>(*signal)
            .map(|tag| tag.0)
            .unwrap_or(state.default_schedule);

        // Cache the schedule for O(1) lookup during removal
        state.signal_schedules.insert(signal, schedule);

        let schedule_levels = state.by_schedule.entry(schedule).or_default();
        while schedule_levels.len() <= level as usize {
            schedule_levels.push(Vec::new());
        }
        insert_sorted_by_index(&mut schedule_levels[level as usize], signal);
    }
}

/// Remove a signal from its current position in by_schedule.
fn remove_signal_from_buckets(state: &mut SignalGraphState, signal: SignalSystem, old_level: u32) {
    // Remove from by_schedule using cached schedule for O(1) lookup.
    if let Some(&schedule) = state.signal_schedules.get(&signal)
        && let Some(schedule_levels) = state.by_schedule.get_mut(&schedule)
        && let Some(bucket) = schedule_levels.get_mut(old_level as usize)
        && let Some(pos) = bucket.iter().position(|s| *s == signal)
    {
        bucket.remove(pos);
    }
}

/// Insert a signal at its new level in by_schedule.
fn insert_signal_into_buckets(world: &World, state: &mut SignalGraphState, signal: SignalSystem, new_level: u32) {
    // Get schedule and insert into by_schedule
    let schedule = world
        .get::<SignalScheduleTag>(*signal)
        .map(|tag| tag.0)
        .unwrap_or(state.default_schedule);

    // Update the schedule cache
    state.signal_schedules.insert(signal, schedule);

    let schedule_levels = state.by_schedule.entry(schedule).or_default();
    while schedule_levels.len() <= new_level as usize {
        schedule_levels.push(Vec::new());
    }
    insert_sorted_by_index(&mut schedule_levels[new_level as usize], signal);
}

fn update_levels_incremental(world: &mut World, state: &mut SignalGraphState, seeds: &[SignalSystem]) -> bool {
    // Collect all signals affected by edge changes (seeds + all their downstreams).
    let mut affected: HashSet<SignalSystem> = HashSet::new();
    let mut queue: VecDeque<SignalSystem> = seeds.iter().copied().collect();

    while let Some(signal) = queue.pop_front() {
        if affected.insert(signal) {
            for downstream in get_downstreams(world, signal) {
                queue.push_back(downstream);
            }
        }
    }
    if affected.is_empty() {
        return true;
    }

    // Use Kahn's algorithm to compute new levels for affected signals
    let result = compute_signal_levels(
        world,
        &affected,
        |u| affected.contains(&u),         // only affected upstreams count for in-degree
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
fn update_edge_change_levels(world: &mut World, state: &mut SignalGraphState) -> Vec<SignalSystem> {
    if state.levels.is_empty() {
        rebuild_levels(world, state);
        return state.edge_change_seeds.drain().collect();
    }

    if state.edge_change_seeds.is_empty() {
        return Vec::new();
    }

    // Drain seeds once and pass directly to update function
    let seeds: Vec<SignalSystem> = state.edge_change_seeds.drain().collect();

    if !update_levels_incremental(world, state, &seeds) {
        panic!("signal graph contains a cycle or inconsistent edges during incremental update");
    }

    seeds
}

fn remove_signal_from_graph_state_internal(state: &mut SignalGraphState, signal: SignalSystem) {
    if let Some(level) = state.levels.remove(&signal) {
        remove_signal_from_buckets(state, signal, level);
    }
    state.signal_schedules.remove(&signal);
    state.edge_change_seeds.remove(&signal);
}

fn remove_signal_from_graph_state(world: &mut World, signal: SignalSystem) {
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

/// Runs a signal node: reads inputs from its [`SignalInputBuffer`], executes the system,
/// and writes outputs to downstream signals' buffers.
///
/// Avoids holding references to [`SignalGraphState`] during execution, since signal
/// systems may spawn new signals that call [`pipe_signal`].
fn run_signal_node(world: &mut World, signal: SignalSystem) {
    // Get runner and inputs before running (to avoid borrow conflicts)
    let (runner, signal_inputs, upstreams) = {
        let runner = match world.get::<SystemRunner>(*signal).cloned() {
            Some(runner) => runner,
            None => {
                if world.get_entity(*signal).is_err() {
                    // Re-entrant combinators can despawn signals during the same frame; skip these stale IDs.
                    return;
                }
                let upstreams = get_upstreams(world, signal);
                let downstreams = get_downstreams(world, signal);
                panic!(
                    "missing SystemRunner for signal {:?} during processing (entity exists). upstreams={:?}, downstreams={:?}",
                    signal, upstreams, downstreams
                );
            }
        };

        // Take inputs from the signal's component
        let signal_inputs = world
            .get_mut::<SignalInputBuffer>(*signal)
            .map(|mut buffer| buffer.take())
            .unwrap_or_default();

        let upstreams = get_upstreams(world, signal);

        (runner, signal_inputs, upstreams)
    };

    // Run the signal system
    let final_output = if upstreams.is_empty() {
        // Root signal - run with unit input
        runner.run(world, Box::new(()))
    } else if !signal_inputs.is_empty() {
        // Run once per upstream input received, forwarding only the final output.
        // This means a signal with multiple upstreams acts as a "collection" point:
        // it processes each upstream's output, but only its last output propagates downstream.
        let mut output = None;
        for input in signal_inputs {
            if let Some(o) = runner.run(world, input) {
                output = Some(o);
            }
        }
        output
    } else {
        None
    };

    // Write outputs directly to downstream components
    if let Some(output) = final_output {
        let downstreams = get_downstreams(world, signal);
        if let Some((last, rest)) = downstreams.split_last() {
            // Clone for all but last
            for downstream in rest {
                if let Ok(mut entity) = world.get_entity_mut(**downstream)
                    && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
                {
                    buffer.push(output.clone());
                }
            }
            // Last downstream gets the original (no clone)
            if let Ok(mut entity) = world.get_entity_mut(**last)
                && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
            {
                buffer.push(output);
            }
        }
    }
}

pub(crate) fn trigger_signal_subgraph(
    world: &mut World,
    signals: impl AsRef<[SignalSystem]>,
    input: Box<dyn AnyClone>,
) {
    let signals = signals.as_ref();
    if signals.is_empty() {
        return;
    }

    // Pre-populate inputs for seed signals by writing to their components
    if let Some((last, rest)) = signals.split_last() {
        for signal in rest {
            if let Ok(mut entity) = world.get_entity_mut(**signal)
                && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
            {
                buffer.push(input.clone());
            }
        }
        // Last signal gets the original (no clone)
        if let Ok(mut entity) = world.get_entity_mut(**last)
            && let Some(mut buffer) = entity.get_mut::<SignalInputBuffer>()
        {
            buffer.push(input);
        }
    }

    // Process seeds and all their downstreams in topological order.
    let by_level = downstream_levels_from_seeds(world, signals);
    for level in by_level {
        for signal in level {
            run_signal_node(world, signal);
        }
    }
}

/// Creates a system that processes only signals tagged for the specified schedule.
///
/// Uses persistent inputs stored in [`SignalGraphState`] to enable cross-schedule data flow.
/// Includes a fixpoint loop to process signals that are registered during processing.
pub(crate) fn process_signal_graph_for_schedule(schedule: InternedScheduleLabel) -> impl FnMut(&mut World) {
    move |world: &mut World| {
        // Phase 1: Update graph if needed, take this schedule's signals (avoids cloning)
        let levels_for_schedule: Vec<Vec<SignalSystem>> = {
            world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
                // Recompute levels if edges changed (also updates partition incrementally)
                let _ = update_edge_change_levels(world, &mut state);
                state.is_processing = true;

                // Take this schedule's signals to avoid cloning; we'll put them back after processing
                state.by_schedule.remove(&schedule).unwrap_or_default()
            })
        };

        // Track signals we've already processed this frame to avoid double-processing
        let mut processed: HashSet<SignalSystem> = HashSet::default();

        // Phase 2: Process signals level-by-level using persistent inputs
        // Note: We don't use resource_scope here because signal processing may spawn
        // new elements that register new signals, which calls pipe_signal, which needs
        // to access SignalGraphState. Using resource_scope would temporarily remove
        // the resource and cause a panic.
        for level in &levels_for_schedule {
            for &signal in level {
                processed.insert(signal);
                run_signal_node(world, signal);
            }
        }

        // Put levels back
        world
            .resource_mut::<SignalGraphState>()
            .by_schedule
            .insert(schedule, levels_for_schedule);

        // Phase 2b: Process signals registered during processing (registration recursion).
        // This handles cases like child_signal spawning elements with their own signals.
        let (recursion_limit, limit_behavior) = {
            let state = world.resource::<SignalGraphState>();
            (state.registration_recursion_limit, state.on_recursion_limit_exceeded)
        };
        let mut recursion_pass = 0usize;

        loop {
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

            // Update levels for new signals and get the seeds that were processed
            let new_seeds: Vec<SignalSystem> = world
                .resource_scope(|world, mut state: Mut<SignalGraphState>| update_edge_change_levels(world, &mut state));

            if new_seeds.is_empty() {
                break;
            }

            // Get signals in this schedule that we haven't processed yet
            let new_signals: Vec<SignalSystem> = new_seeds
                .into_iter()
                .filter(|s| {
                    !processed.contains(s)
                        && world
                            .get::<SignalScheduleTag>(**s)
                            .map(|tag| tag.0 == schedule)
                            .unwrap_or(false)
                })
                .collect();

            if new_signals.is_empty() {
                break;
            }

            // Process new signals and their downstreams in topological order
            let new_levels = downstream_levels_from_seeds(world, &new_signals);
            for level in new_levels {
                for signal in level {
                    if processed.insert(signal) {
                        // Only process if we haven't already
                        if world
                            .get::<SignalScheduleTag>(*signal)
                            .map(|tag| tag.0 == schedule)
                            .unwrap_or(false)
                        {
                            run_signal_node(world, signal);
                        }
                    }
                }
            }
        }

        // Phase 3: Cleanup
        let mut state = world.resource_mut::<SignalGraphState>();
        state.is_processing = false;
        apply_deferred_removals(&mut state);
    }
}

/// Clears persistent inputs at the end of each frame.
///
/// This should run after all signal processing systems in the frame.
/// Clears all signal input buffers at the end of each frame.
///
/// This should run after all signal processing systems in the frame.
pub(crate) fn clear_signal_inputs(mut buffers: Query<&mut SignalInputBuffer>) {
    for mut buffer in &mut buffers {
        buffer.clear();
    }
}

/// Handle to a particular node of the signal graph, returned by
/// [`SignalExt::register`](super::signal::SignalExt),
/// [`SignalVecExt::register`](super::signal_vec::SignalVecExt::register), and
/// [`SignalMapExt::register`](super::signal_map::SignalMapExt::register). In order
/// for signals to be appropriately cleaned up, for every call to `.register` made
/// to some particular signal or its clones, [`SignalHandle::cleanup`] must be
/// called on a corresponding [`SignalHandle`] or a downstream [`SignalHandle`].
/// Adding [`SignalHandle`]s to the [`SignalHandles`] [`Component`] will take care
/// of this when the corresponding [`Entity`] is despawned, and using the
/// [`jonmo::Builder`](super::builder::Builder) will manage this internally.
#[derive(Clone, Deref, Debug)]
pub struct SignalHandle(pub SignalSystem);

impl From<SignalSystem> for SignalHandle {
    fn from(signal: SignalSystem) -> Self {
        Self(signal)
    }
}

impl SignalHandle {
    #[allow(missing_docs)]
    pub(crate) fn new(signal: SignalSystem) -> Self {
        Self(signal)
    }

    /// Decrements the usage tracking of the corresponding signal and all its
    /// upstreams, potentially despawning the backing [`System`], see [`SignalHandle`].
    pub fn cleanup(self, world: &mut World) {
        cleanup_recursive(world, *self);
    }
}

fn cleanup_signal_handles(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let handles: Vec<_> = world
        .entity_mut(entity)
        .get_mut::<SignalHandles>()
        .unwrap()
        .0
        .drain(..)
        .collect();
    let mut commands = world.commands();
    for handle in handles {
        commands.queue(|world: &mut World| handle.cleanup(world));
    }
}

/// Stores [`SignalHandle`]s tied to the lifetime of some [`Entity`],
/// [`.cleanup`](SignalHandle::cleanup)-ing them when the [`Entity`] is despawned.
#[derive(Component, Default, Deref, Clone)]
#[component(on_remove = cleanup_signal_handles)]
pub struct SignalHandles(Vec<SignalHandle>);

impl<T> From<T> for SignalHandles
where
    Vec<SignalHandle>: From<T>,
{
    #[inline]
    fn from(values: T) -> Self {
        SignalHandles(values.into())
    }
}

impl SignalHandles {
    #[allow(missing_docs)]
    pub fn add(&mut self, handle: SignalHandle) {
        self.0.push(handle);
    }
}

fn spawn_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: 'static,
    O: Clone + Send + Sync + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<I>, IOO, M> + 'static,
{
    let signal_system = world.register_system(system);
    let runner: Arc<Box<dyn Runnable>> = Arc::new(Box::new(SystemHolder::<I, O, IOO> {
        system: signal_system,
        _marker: PhantomData,
    }));
    world.entity_mut(signal_system.entity()).insert((
        SignalRegistrationCount::new(),
        SignalInputBuffer::default(),
        SystemRunner {
            runner: Arc::new(Box::new(move |w, inp| runner.run(w, inp))),
        },
    ));
    if let Some(mut state) = world.get_resource_mut::<SignalGraphState>() {
        state.edge_change_seeds.insert(signal_system.entity().into());
    }
    signal_system.into()
}

pub(crate) struct LazySignalState {
    references: AtomicUsize,
    pub(crate) system: RwLock<LazySystem>,
}

pub(crate) enum LazySystem {
    #[allow(clippy::type_complexity)]
    System(Option<Box<dyn FnOnce(&mut World) -> SignalSystem + Send + Sync>>),
    Registered(SignalSystem),
}

impl LazySystem {
    fn register(&mut self, world: &mut World) -> SignalSystem {
        match self {
            LazySystem::System(f) => {
                let signal = f.take().unwrap()(world);
                *self = LazySystem::Registered(signal);
                signal
            }
            LazySystem::Registered(signal) => {
                world
                    .entity_mut(**signal)
                    .get_mut::<SignalRegistrationCount>()
                    .unwrap()
                    .increment();
                *signal
            }
        }
    }
}

pub(crate) struct LazySignal {
    pub(crate) inner: Arc<LazySignalState>,
}

impl LazySignal {
    pub(crate) fn new<F: FnOnce(&mut World) -> SignalSystem + Send + Sync + 'static>(system: F) -> Self {
        LazySignal {
            inner: Arc::new(LazySignalState {
                references: AtomicUsize::new(1),
                system: RwLock::new(LazySystem::System(Some(Box::new(system)))),
            }),
        }
    }

    pub(crate) fn register(self, world: &mut World) -> SignalSystem {
        let signal = self.inner.system.write().unwrap().register(world);
        let mut entity = world.entity_mut(*signal);
        if !entity.contains::<LazySignalHolder>() {
            entity.insert(LazySignalHolder(self));
        }
        signal
    }
}

impl Clone for LazySignal {
    fn clone(&self) -> Self {
        self.inner.references.fetch_add(1, Ordering::SeqCst);
        LazySignal {
            inner: self.inner.clone(),
        }
    }
}

impl Drop for LazySignal {
    fn drop(&mut self) {
        // <= 2 because we also wna queue if only the holder remains
        if self.inner.references.fetch_sub(1, Ordering::SeqCst) <= 2
            && let LazySystem::Registered(signal) = *self.inner.system.read().unwrap()
        {
            STALE_SIGNALS.lock().unwrap().push(signal);
        }
    }
}

#[derive(Component)]
pub(crate) struct LazySignalHolder(LazySignal);

pub(crate) static STALE_SIGNALS: LazyLock<Mutex<Vec<SignalSystem>>> = LazyLock::new(|| Mutex::new(Vec::new()));

fn unlink_downstreams_and_mark(world: &mut World, signal: SignalSystem) {
    if let Some(downstreams) = world.get::<Downstream>(*signal).cloned() {
        for &downstream in downstreams.iter() {
            if let Ok(mut downstream_entity) = world.get_entity_mut(*downstream)
                && let Some(mut upstream) = downstream_entity.get_mut::<Upstream>()
            {
                upstream.0.remove(&signal);
                if upstream.0.is_empty() {
                    downstream_entity.remove::<Upstream>();
                }
            }
            world
                .resource_mut::<SignalGraphState>()
                .edge_change_seeds
                .insert(downstream);
        }
    }
}

pub(crate) fn despawn_stale_signals(world: &mut World) {
    let signals = STALE_SIGNALS.lock().unwrap().drain(..).collect::<Vec<_>>();
    for signal in signals {
        let should_despawn = world
            .get::<SignalRegistrationCount>(*signal)
            .map(|registration_count| **registration_count == 0)
            .unwrap_or(false);
        if should_despawn {
            unlink_downstreams_and_mark(world, signal);
            remove_signal_from_graph_state(world, signal);
            world.entity_mut(*signal).despawn();
        }
    }
}

pub(crate) fn lazy_signal_from_system<I, O, IOO, F, M>(system: F) -> LazySignal
where
    I: 'static,
    O: Clone + Send + Sync + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<I>, IOO, M> + Send + Sync + 'static,
{
    LazySignal::new(move |world: &mut World| spawn_signal(world, system))
}

#[allow(dead_code)]
pub(crate) struct UpstreamIter<'w, 's, D: QueryData, F: QueryFilter>
where
    D::ReadOnly: QueryData<Item<'w, 's> = &'w Upstream>,
{
    upstreams_query: &'w Query<'w, 's, D, F>,
    upstreams: Vec<SignalSystem>,
}

#[allow(dead_code)]
impl<'w, 's, D: QueryData, F: QueryFilter> UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w, 's> = &'w Upstream>,
{
    /// Returns a new [`DescendantIter`].
    pub fn new(upstreams_query: &'w Query<'w, 's, D, F>, signal: SignalSystem) -> Self {
        UpstreamIter {
            upstreams_query,
            upstreams: upstreams_query.get(*signal).into_iter().flatten().copied().collect(),
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w, 's> = &'w Upstream>,
{
    type Item = SignalSystem;

    fn next(&mut self) -> Option<Self::Item> {
        let signal = self.upstreams.pop()?;
        if let Ok(upstream) = self.upstreams_query.get(*signal) {
            self.upstreams.extend(upstream);
        }
        Some(signal)
    }
}

#[allow(dead_code)]
pub(crate) struct DownstreamIter<'w, 's, D: QueryData, F: QueryFilter>
where
    D::ReadOnly: QueryData<Item<'w, 's> = &'w Downstream>,
{
    downstreams_query: &'w Query<'w, 's, D, F>,
    downstreams: Vec<SignalSystem>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w, 's> = &'w Downstream>,
{
    #[allow(dead_code)]
    pub fn new(downstreams_query: &'w Query<'w, 's, D, F>, signal: SignalSystem) -> Self {
        DownstreamIter {
            downstreams_query,
            downstreams: downstreams_query.get(*signal).into_iter().flatten().copied().collect(),
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w, 's> = &'w Downstream>,
{
    type Item = SignalSystem;

    fn next(&mut self) -> Option<Self::Item> {
        let signal = self.downstreams.pop()?;
        if let Ok(downstream) = self.downstreams_query.get(*signal) {
            self.downstreams.extend(downstream);
        }
        Some(signal)
    }
}

fn decrement_registration_and_needs_cleanup(world: &mut World, signal: SignalSystem) -> bool {
    if let Ok(mut entity) = world.get_entity_mut(*signal)
        && let Some(mut count) = entity.get_mut::<SignalRegistrationCount>()
    {
        count.decrement();
        return **count == 0;
    }
    false
}

fn should_despawn_signal(world: &World, signal: SignalSystem) -> bool {
    world
        .get_entity(*signal)
        .ok()
        .map(|entity| {
            if let Some(lazy_holder) = entity.get::<LazySignalHolder>() {
                lazy_holder.0.inner.references.load(Ordering::SeqCst) == 1
            } else {
                // This is a dynamically created signal without a holder, it can be despawned once
                // its registrations are gone.
                true
            }
        })
        .unwrap_or(false)
}

fn unlink_from_upstream(world: &mut World, upstream_system: SignalSystem, signal: SignalSystem) {
    if let Ok(mut upstream_entity) = world.get_entity_mut(*upstream_system)
        && let Some(mut downstream) = upstream_entity.get_mut::<Downstream>()
    {
        downstream.0.remove(&signal);
        if downstream.0.is_empty() {
            upstream_entity.remove::<Downstream>();
        }
    }
}

fn cleanup_recursive(world: &mut World, signal: SignalSystem) {
    // Decrement registration and bail if the node is still in use.
    if !decrement_registration_and_needs_cleanup(world, signal) {
        return;
    }

    // The count is zero. Perform the full cleanup. First, get the list of parents.
    let upstreams = world.get::<Upstream>(*signal).cloned();

    // Unlink downstream edges and mark affected nodes for level recomputation.
    unlink_downstreams_and_mark(world, signal);

    if should_despawn_signal(world, signal) {
        remove_signal_from_graph_state(world, signal);
        if let Ok(entity) = world.get_entity_mut(*signal) {
            entity.despawn();
        }
    }

    // Notify parents and recurse after processing this node.
    if let Some(upstreams) = upstreams {
        for &upstream_system in upstreams.iter() {
            unlink_from_upstream(world, upstream_system, signal);
            cleanup_recursive(world, upstream_system);
        }
    }
}

/// Computes topological levels for signals upstream of a target that aren't in the cached state.
/// Uses Kahn's algorithm on the reachable subgraph.
fn compute_levels_for_uncached(
    world: &World,
    reachable: &HashSet<SignalSystem>,
    cached_levels: &HashMap<SignalSystem, u32>,
) -> HashMap<SignalSystem, u32> {
    let uncached: HashSet<SignalSystem> = reachable
        .iter()
        .filter(|s| !cached_levels.contains_key(*s))
        .copied()
        .collect();

    // Note: upstream_filter uses `reachable` (not `uncached`) because we need to count
    // in-degree based on the full reachable subgraph, but only compute levels for uncached signals.
    // Cached upstreams don't contribute to in-degree since their levels are already known.
    let result = compute_signal_levels(
        world,
        &uncached,
        |u| reachable.contains(&u) && !cached_levels.contains_key(&u),
        |u| cached_levels.get(&u).copied(),
    );

    result.levels
}

fn poll_signal_one_shot(In(target): In<SignalSystem>, world: &mut World) -> Option<Box<dyn AnyClone>> {
    // Collect all signals reachable upstream from target
    let mut reachable: HashSet<SignalSystem> = HashSet::new();
    let mut queue: VecDeque<SignalSystem> = VecDeque::new();
    queue.push_back(target);
    reachable.insert(target);

    while let Some(signal) = queue.pop_front() {
        for upstream in get_upstreams(world, signal) {
            if reachable.insert(upstream) {
                queue.push_back(upstream);
            }
        }
    }

    // Get cached levels and compute levels for any uncached signals.
    // We avoid cloning the entire levels HashMap by collecting only the levels we need.

    // First pass: check which signals need level computation
    let uncached: HashSet<SignalSystem> = {
        let state = world.resource::<SignalGraphState>();
        reachable
            .iter()
            .filter(|s| !state.levels.contains_key(*s))
            .copied()
            .collect()
    };

    // Compute levels for uncached signals if any (requires &World, so state borrow must end first)
    let uncached_levels = if uncached.is_empty() {
        HashMap::default()
    } else {
        compute_levels_for_uncached(world, &reachable, &world.resource::<SignalGraphState>().levels)
    };

    // Bucket by level using references to avoid cloning
    let by_level = {
        let state = world.resource::<SignalGraphState>();
        let mut by_level: Vec<Vec<SignalSystem>> = Vec::new();
        for signal in &reachable {
            let level = state
                .levels
                .get(signal)
                .or_else(|| uncached_levels.get(signal))
                .copied()
                .unwrap_or(0) as usize;
            while by_level.len() <= level {
                by_level.push(Vec::new());
            }
            by_level[level].push(*signal);
        }

        // Sort each level for determinism
        for level in &mut by_level {
            level.sort_by_key(|s| s.index());
        }

        by_level
    };

    // Process level by level, running each signal once per upstream input received.
    // We only track the target's output directly instead of storing all outputs.
    let mut inputs: HashMap<SignalSystem, Vec<Box<dyn AnyClone>>> = HashMap::new();
    let mut target_output: Option<Box<dyn AnyClone>> = None;

    for level in by_level {
        for signal in level {
            let runner = world
                .get::<SystemRunner>(*signal)
                .cloned()
                .unwrap_or_else(|| panic!("missing SystemRunner for signal {:?} during poll", signal));

            let upstreams = get_upstreams(world, signal);

            let output = if upstreams.is_empty() {
                // Source signal - run with unit input
                runner.run(world, Box::new(()))
            } else if let Some(input_list) = inputs.remove(&signal) {
                // Has inputs from upstreams - run once per input, keep final output
                let mut final_output = None;
                for input in input_list {
                    if let Some(out) = runner.run(world, input) {
                        final_output = Some(out);
                    }
                }
                final_output
            } else {
                // No input received - signal doesn't fire
                None
            };

            // Only store output if this is the target we're polling
            if signal == target {
                target_output = output;
                // Target found - no need to propagate further since we're done
                continue;
            }

            // Propagate output to downstreams
            if let Some(out) = output {
                let downstreams = get_downstreams(world, signal);
                if let Some((last, rest)) = downstreams.split_last() {
                    for downstream in rest {
                        inputs.entry(*downstream).or_default().push(out.clone());
                    }
                    inputs.entry(*last).or_default().push(out);
                }
            }
        }
    }

    target_output
}

/// Get a signal's current value by running all of it's dependencies.
pub fn poll_signal(world: &mut World, signal: SignalSystem) -> Option<Box<dyn AnyClone>> {
    world
        .run_system_cached_with(poll_signal_one_shot, signal)
        .ok()
        .flatten()
}

/// Utility function for extracting values from [`AnyClone`]s, e.g. those returned by
/// [`poll_signal`].
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use jonmo::{prelude::*, graph::*};
///
/// let mut app = App::new();
/// app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
/// let signal = *signal::from_system(|_: In<()>| 1).register(app.world_mut());
/// poll_signal(app.world_mut(), signal).and_then(downcast_any_clone::<usize>); // outputs an `Option<usize>`
/// ```
pub fn downcast_any_clone<T: 'static>(any_clone: Box<dyn AnyClone>) -> Option<T> {
    (any_clone as Box<dyn Any>).downcast::<T>().map(|o| *o).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    use bevy_ecs::{
        prelude::{In, Mut, World},
        schedule::ScheduleLabel,
    };

    #[derive(Resource, Default)]
    struct Order(Vec<&'static str>);

    /// Helper to process all signals in the default schedule for tests.
    fn process_signals(world: &mut World) {
        let default_schedule = world.resource::<SignalGraphState>().default_schedule;
        let mut system = process_signal_graph_for_schedule(default_schedule);
        system(world);
    }

    #[test]
    #[should_panic(expected = "signal graph contains a cycle or inconsistent edges during incremental update")]
    fn incremental_update_panics_on_cycle() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));
        let signal_b = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(2));

        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        world
            .entity_mut(*signal_a)
            .insert(Downstream(HashSet::from([signal_b])))
            .insert(Upstream(HashSet::from([signal_b])));
        world
            .entity_mut(*signal_b)
            .insert(Downstream(HashSet::from([signal_a])))
            .insert(Upstream(HashSet::from([signal_a])));

        world
            .resource_mut::<SignalGraphState>()
            .edge_change_seeds
            .insert(signal_a);

        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            update_edge_change_levels(world, &mut state);
        });
    }

    #[test]
    fn ordering_is_deterministic_with_multiple_roots() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());
        world.insert_resource(Order::default());

        let signal_a = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, mut order: ResMut<Order>| {
            order.0.push("a");
            Some(())
        });
        let signal_b = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, mut order: ResMut<Order>| {
            order.0.push("b");
            Some(())
        });

        process_signals(&mut world);

        let order = world.resource::<Order>().0.clone();
        if signal_a.0.index() < signal_b.0.index() {
            assert_eq!(order, vec!["a", "b"]);
        } else {
            assert_eq!(order, vec!["b", "a"]);
        }
    }

    #[test]
    fn piping_updates_levels_for_same_frame_execution() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());
        world.insert_resource(Order::default());

        let signal_a = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, mut order: ResMut<Order>| {
            order.0.push("a");
            Some(())
        });
        let signal_b = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, mut order: ResMut<Order>| {
            order.0.push("b");
            Some(())
        });

        process_signals(&mut world);
        world.resource_mut::<Order>().0.clear();

        pipe_signal(&mut world, signal_a, signal_b);
        process_signals(&mut world);

        let order = world.resource::<Order>().0.clone();
        assert_eq!(order, vec!["a", "b"]);
    }

    #[test]
    fn incremental_matches_full_rebuild_after_edge_change() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let a = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));
        let b = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));
        let c = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));
        let d = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));

        pipe_signal(&mut world, a, b);
        pipe_signal(&mut world, b, c);
        pipe_signal(&mut world, a, d);

        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        pipe_signal(&mut world, c, d);

        let incremental_levels = world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            update_edge_change_levels(world, &mut state);
            state.levels.clone()
        });

        let mut expected = SignalGraphState::default();
        rebuild_levels(&mut world, &mut expected);

        assert_eq!(incremental_levels, expected.levels);
    }

    #[test]
    fn cleanup_updates_downstream_levels() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let a = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));
        let b = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));
        let c = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>| Some(()));

        pipe_signal(&mut world, a, b);
        pipe_signal(&mut world, b, c);

        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        cleanup_recursive(&mut world, a);

        let levels_after = world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            update_edge_change_levels(world, &mut state);
            state.levels.clone()
        });

        assert_eq!(levels_after.get(&b), Some(&0));
        assert_eq!(levels_after.get(&c), Some(&1));
        assert!(!levels_after.contains_key(&a));
    }

    #[test]
    fn schedule_tag_assigns_signal_to_schedule() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));

        // Tag the signal with Update schedule
        world.entity_mut(*signal_a).insert(SignalScheduleTag(Update.intern()));

        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        let state = world.resource::<SignalGraphState>();
        let update_signals = state.by_schedule.get(&Update.intern());
        assert!(update_signals.is_some());
        assert!(update_signals.unwrap().iter().flatten().any(|s| *s == signal_a));
    }

    #[test]
    fn schedule_hint_propagates_to_downstream_via_pipe() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));
        let signal_b = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(2));

        // Tag signal_a with Update schedule and set a hint for downstream
        world
            .entity_mut(*signal_a)
            .insert(SignalScheduleTag(Update.intern()))
            .insert(ScheduleHint(Update.intern()));

        // Pipe a -> b (b should inherit the schedule from hint)
        pipe_signal(&mut world, signal_a, signal_b);

        // signal_b should now have the SignalScheduleTag from the hint
        let tag = world.get::<SignalScheduleTag>(*signal_b);
        assert!(tag.is_some());
        assert_eq!(tag.unwrap().0, Update.intern());
    }

    #[test]
    fn schedule_hint_does_not_override_existing_tag() {
        use bevy_app::{Last, Update};

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));
        let signal_b = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(2));

        // Tag signal_a with Update schedule and set a hint
        world
            .entity_mut(*signal_a)
            .insert(SignalScheduleTag(Update.intern()))
            .insert(ScheduleHint(Update.intern()));

        // Tag signal_b with Last schedule (explicit tag)
        world.entity_mut(*signal_b).insert(SignalScheduleTag(Last.intern()));

        // Pipe a -> b (b's explicit tag should NOT be overridden)
        pipe_signal(&mut world, signal_a, signal_b);

        // signal_b should still have Last schedule
        let tag = world.get::<SignalScheduleTag>(*signal_b);
        assert!(tag.is_some());
        assert_eq!(tag.unwrap().0, Last.intern());
    }

    #[test]
    fn schedule_propagates_to_all_upstreams_for_multi_upstream_signal() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        // Create three independent root signals (simulating branches that will be combined)
        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));
        let signal_b = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(2));
        let signal_c = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(3));

        // Create a multi-upstream signal that depends on all three
        // (simulating what happens with signal::zip or signal::any!)
        let combined = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(100));

        // Manually connect all three as upstreams of `combined`
        pipe_signal(&mut world, signal_a, combined);
        pipe_signal(&mut world, signal_b, combined);
        pipe_signal(&mut world, signal_c, combined);

        // Verify the combined signal has all three upstreams
        let upstreams = world.get::<Upstream>(*combined).unwrap();
        assert_eq!(upstreams.iter().count(), 3);

        // Now apply schedule to the combined signal - this should propagate to ALL upstreams
        apply_schedule_to_signal(&mut world, combined, Update.intern());

        // Verify the combined signal itself has the schedule
        let combined_tag = world.get::<SignalScheduleTag>(*combined);
        assert!(combined_tag.is_some());
        assert_eq!(combined_tag.unwrap().0, Update.intern());

        // Verify ALL upstream signals got the schedule tag
        let tag_a = world.get::<SignalScheduleTag>(*signal_a);
        assert!(tag_a.is_some(), "signal_a should have schedule tag");
        assert_eq!(tag_a.unwrap().0, Update.intern());

        let tag_b = world.get::<SignalScheduleTag>(*signal_b);
        assert!(tag_b.is_some(), "signal_b should have schedule tag");
        assert_eq!(tag_b.unwrap().0, Update.intern());

        let tag_c = world.get::<SignalScheduleTag>(*signal_c);
        assert!(tag_c.is_some(), "signal_c should have schedule tag");
        assert_eq!(tag_c.unwrap().0, Update.intern());
    }

    #[test]
    fn schedule_propagates_to_deep_multi_upstream_graph() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        // Create a deeper graph:
        //       root_a    root_b    root_c
        //          \       /           |
        //         mid_ab              mid_c
        //              \             /
        //               \           /
        //                  final
        let root_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));
        let root_b = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(2));
        let root_c = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(3));

        let mid_ab = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(10));
        let mid_c = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(20));

        let final_signal = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(100));

        // Connect the graph
        pipe_signal(&mut world, root_a, mid_ab);
        pipe_signal(&mut world, root_b, mid_ab);
        pipe_signal(&mut world, root_c, mid_c);
        pipe_signal(&mut world, mid_ab, final_signal);
        pipe_signal(&mut world, mid_c, final_signal);

        // Apply schedule only to the final signal
        apply_schedule_to_signal(&mut world, final_signal, Update.intern());

        // ALL signals in the graph should now have the schedule tag
        for (name, signal) in [
            ("root_a", root_a),
            ("root_b", root_b),
            ("root_c", root_c),
            ("mid_ab", mid_ab),
            ("mid_c", mid_c),
            ("final", final_signal),
        ] {
            let tag = world.get::<SignalScheduleTag>(*signal);
            assert!(tag.is_some(), "{name} should have schedule tag");
            assert_eq!(tag.unwrap().0, Update.intern(), "{name} should have Update schedule");
        }
    }

    #[test]
    fn by_schedule_is_partitioned_correctly() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let signal_update = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));
        let signal_default = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(2));

        // Tag one signal with Update
        world
            .entity_mut(*signal_update)
            .insert(SignalScheduleTag(Update.intern()));

        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        let state = world.resource::<SignalGraphState>();

        // Check Update schedule has signal_update
        let update_signals = state.by_schedule.get(&Update.intern());
        assert!(update_signals.is_some());
        let update_flat: Vec<_> = update_signals.unwrap().iter().flatten().collect();
        assert!(update_flat.contains(&&signal_update));
        assert!(!update_flat.contains(&&signal_default));

        // Check default schedule (PostUpdate) has signal_default
        let default_signals = state.by_schedule.get(&state.default_schedule);
        assert!(default_signals.is_some());
        let default_flat: Vec<_> = default_signals.unwrap().iter().flatten().collect();
        assert!(default_flat.contains(&&signal_default));
        assert!(!default_flat.contains(&&signal_update));
    }

    #[test]
    fn process_for_schedule_only_runs_scheduled_signals() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());
        world.insert_resource(Order::default());

        let signal_update =
            spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, mut order: ResMut<Order>| {
                order.0.push("update");
                Some(())
            });
        let _signal_default =
            spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, mut order: ResMut<Order>| {
                order.0.push("default");
                Some(())
            });

        // Tag one signal with Update schedule
        world
            .entity_mut(*signal_update)
            .insert(SignalScheduleTag(Update.intern()));

        // Build levels first
        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        // Process only Update schedule
        let mut process_update = process_signal_graph_for_schedule(Update.intern());
        process_update(&mut world);

        let order = world.resource::<Order>().0.clone();
        assert_eq!(order, vec!["update"]);

        // Now process default schedule
        world.resource_mut::<Order>().0.clear();
        let default_schedule = world.resource::<SignalGraphState>().default_schedule;
        let mut process_default = process_signal_graph_for_schedule(default_schedule);
        process_default(&mut world);

        let order = world.resource::<Order>().0.clone();
        assert_eq!(order, vec!["default"]);
    }

    #[test]
    fn cross_schedule_data_flow_via_inputs() {
        use bevy_app::Update;

        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        #[derive(Resource, Default)]
        struct CollectedValues(Vec<i32>);
        world.insert_resource(CollectedValues::default());

        // signal_a runs in Update, outputs 42
        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(42));

        // signal_b runs in PostUpdate (default), collects input
        let signal_b = spawn_signal::<i32, (), Option<()>, _, _>(
            &mut world,
            |In(value): In<i32>, mut collected: ResMut<CollectedValues>| {
                collected.0.push(value);
                Some(())
            },
        );

        // Tag signal_a with Update
        world.entity_mut(*signal_a).insert(SignalScheduleTag(Update.intern()));

        // Pipe a -> b (cross-schedule dependency)
        pipe_signal(&mut world, signal_a, signal_b);

        // Build levels
        world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
            rebuild_levels(world, &mut state);
        });

        // Process Update schedule first (signal_a runs, stores output in inputs)
        let mut process_update = process_signal_graph_for_schedule(Update.intern());
        process_update(&mut world);

        // signal_b hasn't run yet
        assert!(world.resource::<CollectedValues>().0.is_empty());

        // Process PostUpdate schedule (signal_b runs, gets input from signal_a)
        let default_schedule = world.resource::<SignalGraphState>().default_schedule;
        let mut process_default = process_signal_graph_for_schedule(default_schedule);
        process_default(&mut world);

        // signal_b should have received 42 from signal_a
        assert_eq!(world.resource::<CollectedValues>().0, vec![42]);
    }

    #[test]
    fn clear_signal_inputs_clears_inputs() {
        let mut world = World::new();
        world.insert_resource(SignalGraphState::default());

        let signal_a = spawn_signal::<(), i32, Option<i32>, _, _>(&mut world, |_: In<()>| Some(1));

        // Manually insert some inputs into the signal's buffer component
        world
            .get_mut::<SignalInputBuffer>(*signal_a)
            .unwrap()
            .push(Box::new(42i32) as Box<dyn AnyClone>);

        assert!(!world.get::<SignalInputBuffer>(*signal_a).unwrap().0.is_empty());

        // Clear inputs by clearing the component
        world.get_mut::<SignalInputBuffer>(*signal_a).unwrap().clear();

        assert!(world.get::<SignalInputBuffer>(*signal_a).unwrap().0.is_empty());
    }

    #[test]
    fn signals_registered_during_processing_are_processed_same_frame() {
        use bevy_app::Update;

        // This test verifies the fixpoint loop behavior:
        // When a signal spawns new elements that register new signals during processing,
        // those new signals should be processed in the same frame.

        #[derive(Resource, Default)]
        struct ProcessOrder(Vec<&'static str>);

        #[derive(Resource)]
        struct ChildSignalHandle(Option<SignalSystem>);

        let mut world = World::new();
        world.insert_resource(SignalGraphState::new(Update.intern()));
        world.insert_resource(ProcessOrder::default());
        world.insert_resource(ChildSignalHandle(None));

        // Create a "parent" signal that, when processed, registers a new "child" signal
        let parent_signal = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, world: &mut World| {
            world.resource_mut::<ProcessOrder>().0.push("parent");

            // Check if child already exists (to avoid infinite loop)
            if world.resource::<ChildSignalHandle>().0.is_none() {
                // Register a new child signal during parent's processing
                let child_signal =
                    spawn_signal::<(), (), Option<()>, _, _>(world, |_: In<()>, mut order: ResMut<ProcessOrder>| {
                        order.0.push("child");
                        Some(())
                    });
                // Tag child with same schedule as parent
                world
                    .entity_mut(*child_signal)
                    .insert(SignalScheduleTag(Update.intern()));
                world.resource_mut::<ChildSignalHandle>().0 = Some(child_signal);
            }
            Some(())
        });

        // Tag parent signal
        world
            .entity_mut(*parent_signal)
            .insert(SignalScheduleTag(Update.intern()));

        // Process signals - the fixpoint loop should process both parent and child
        let mut process_system = process_signal_graph_for_schedule(Update.intern());
        process_system(&mut world);

        // Both parent and child should have been processed in the same frame
        let order = &world.resource::<ProcessOrder>().0;
        assert!(order.contains(&"parent"), "Parent signal should have been processed");
        assert!(
            order.contains(&"child"),
            "Child signal registered during processing should also be processed"
        );
    }

    #[test]
    fn fixpoint_loop_handles_multiple_levels_of_spawning() {
        use bevy_app::Update;

        // Test that the fixpoint loop can handle chains: A spawns B, B spawns C

        #[derive(Resource, Default)]
        struct ProcessOrder(Vec<&'static str>);

        #[derive(Resource, Default)]
        struct SpawnedSignals(Vec<SignalSystem>);

        let mut world = World::new();
        world.insert_resource(SignalGraphState::new(Update.intern()));
        world.insert_resource(ProcessOrder::default());
        world.insert_resource(SpawnedSignals::default());

        // Signal A: spawns signal B
        let signal_a = spawn_signal::<(), (), Option<()>, _, _>(&mut world, |_: In<()>, world: &mut World| {
            world.resource_mut::<ProcessOrder>().0.push("A");

            let spawned = &world.resource::<SpawnedSignals>().0;
            if spawned.is_empty() {
                // Spawn B
                let signal_b = spawn_signal::<(), (), Option<()>, _, _>(world, |_: In<()>, world: &mut World| {
                    world.resource_mut::<ProcessOrder>().0.push("B");

                    let spawned = &world.resource::<SpawnedSignals>().0;
                    if spawned.len() == 1 {
                        // B spawns C
                        let signal_c = spawn_signal::<(), (), Option<()>, _, _>(
                            world,
                            |_: In<()>, mut order: ResMut<ProcessOrder>| {
                                order.0.push("C");
                                Some(())
                            },
                        );
                        world.entity_mut(*signal_c).insert(SignalScheduleTag(Update.intern()));
                        world.resource_mut::<SpawnedSignals>().0.push(signal_c);
                    }
                    Some(())
                });
                world.entity_mut(*signal_b).insert(SignalScheduleTag(Update.intern()));
                world.resource_mut::<SpawnedSignals>().0.push(signal_b);
            }
            Some(())
        });

        world.entity_mut(*signal_a).insert(SignalScheduleTag(Update.intern()));

        // Process signals
        let mut process_system = process_signal_graph_for_schedule(Update.intern());
        process_system(&mut world);

        // All three should have been processed
        let order = &world.resource::<ProcessOrder>().0;
        assert!(order.contains(&"A"), "Signal A should have been processed");
        assert!(
            order.contains(&"B"),
            "Signal B (spawned by A) should have been processed"
        );
        assert!(
            order.contains(&"C"),
            "Signal C (spawned by B) should have been processed"
        );
    }
}
