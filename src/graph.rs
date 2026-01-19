//! Signal graph management and runtime.
use super::utils::SSs;
use alloc::collections::VecDeque;
use bevy_derive::Deref;
use bevy_ecs::{
    entity_disabling::Internal,
    lifecycle::HookContext,
    prelude::*,
    query::{QueryData, QueryFilter},
    system::{SystemId, SystemState},
    world::DeferredWorld,
};
#[cfg(feature = "tracing")]
use bevy_log::prelude::*;
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

pub(crate) fn register_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: 'static,
    O: Clone + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<I>, IOO, M> + SSs,
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
        error!("cycle detected when attempting to pipe {:?} → {:?}", source, target);
        return;
    }
    if let Ok(mut upstream) = world.get_entity_mut(*source) {
        if let Some(mut downstream) = upstream.get_mut::<Downstream>() {
            downstream.0.insert(target);
        } else {
            upstream.insert(Downstream(HashSet::from([target])));
        }
    }
    if let Ok(mut downstream) = world.get_entity_mut(*target) {
        if let Some(mut upstream) = downstream.get_mut::<Upstream>() {
            upstream.0.insert(source);
        } else {
            downstream.insert(Upstream(HashSet::from([source])));
        }
    }
    world.resource_mut::<SignalGraphState>().edge_change_seeds.insert(target);
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
    O: Clone,
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
                        error!(
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

/// An extension trait for [`Any`] types that implement [`Clone`].
pub trait AnyClone: Any + DynClone {}

clone_trait_object!(AnyClone);

impl<T: Clone + 'static> AnyClone for T {}


/// Tracks signal graph topology for level-based processing.
#[derive(Resource, Default)]
pub(crate) struct SignalGraphState {
    /// Cached level for each signal (max distance from any root).
    levels: HashMap<SignalSystem, u32>,
    /// Signals grouped by level for efficient iteration.
    by_level: Vec<Vec<SignalSystem>>,
    /// Signals that seed level recomputation after edge changes.
    edge_change_seeds: HashSet<SignalSystem>,
    /// Signals queued for removal while the graph is being processed.
    deferred_removals: HashSet<SignalSystem>,
    /// Whether the signal graph is currently being processed.
    is_processing: bool,
}

// Fan out a signal output to downstream input queues. When there is exactly one downstream, move the value without cloning.
fn enqueue_inputs(
    world: &World,
    inputs: &mut HashMap<SignalSystem, VecDeque<Box<dyn AnyClone>>>,
    signal: SignalSystem,
    value: Box<dyn AnyClone>,
) {
    let downstreams = get_downstreams(world, signal);
    if downstreams.is_empty() {
        return;
    }

    if downstreams.len() == 1 {
        // avoid cloning if there's only a single downstream
        inputs
            .entry(downstreams[0])
            .or_default()
            .push_back(value);
        return;
    }

    if let Some((last, rest)) = downstreams.split_last() {
        for downstream in rest {
            inputs
                .entry(*downstream)
                .or_default()
                .push_back(value.clone());
        }
        inputs.entry(*last).or_default().push_back(value);
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
    if bucket.iter().any(|s| *s == signal) {
        return;
    }
    let key = signal.index();
    let index = bucket
        .binary_search_by_key(&key, |s| s.index())
        .unwrap_or_else(|i| i);
    bucket.insert(index, signal);
}

// Computes a per-call, local topological ordering of signals reachable downstream
// from `seeds`. This intentionally does NOT use `SignalGraphState` because it only
// needs a lightweight traversal for the provided subset and should not mutate or
// depend on the global cached topology.
fn downstream_levels_from_seeds(
    world: &World,
    seeds: &[SignalSystem],
) -> Vec<Vec<SignalSystem>> {
    let mut levels: HashMap<SignalSystem, u32> = HashMap::new();
    let mut by_level: Vec<HashSet<SignalSystem>> = Vec::new();
    let mut queue: VecDeque<SignalSystem> = VecDeque::new();

    for signal in seeds {
        levels.insert(*signal, 0);
        queue.push_back(*signal);
    }

    while let Some(signal) = queue.pop_front() {
        let level = *levels.get(&signal).unwrap_or(&0);
        for downstream in get_downstreams(world, signal) {
            let next_level = level.saturating_add(1);
            let current = levels.get(&downstream).copied().unwrap_or(0);
            if next_level > current {
                levels.insert(downstream, next_level);
                queue.push_back(downstream);
            }
        }
    }

    for (signal, level) in levels.iter() {
        while by_level.len() <= *level as usize {
            by_level.push(HashSet::new());
        }
        by_level[*level as usize].insert(*signal);
    }

    by_level
        .into_iter()
        .map(|level| {
            let mut signals_at_level: Vec<SignalSystem> = level.into_iter().collect();
            // Sort by Entity index to get a deterministic, stable order independent of hash iteration.
            signals_at_level.sort_by_key(|signal| signal.index());
            signals_at_level
        })
        .collect()
}

// Rebuilds per-signal levels using a Kahn-style topological traversal.
//
// - Roots (in-degree 0) start at level 0.
// - Each node's level is 1 + max(level of its upstreams).
// - Nodes are bucketed by level for deterministic per-level iteration.
// - If a cycle or inconsistent edges are detected (not all nodes processed),
//   this panics because the graph invariants were violated.
fn rebuild_levels(world: &mut World, state: &mut SignalGraphState) {
    state.levels.clear();
    state.by_level.clear();

    let mut all_signals =
        SystemState::<Query<Entity, (With<SystemRunner>, Allow<Internal>)>>::new(world);
    let signals = all_signals
        .get(world)
        .iter()
        .map(SignalSystem)
        .collect::<Vec<_>>();

    let mut in_degree: HashMap<SignalSystem, usize> = HashMap::new();
    let mut upstreams_map: HashMap<SignalSystem, Vec<SignalSystem>> = HashMap::new();
    let mut downstreams_map: HashMap<SignalSystem, Vec<SignalSystem>> = HashMap::new();

    for signal in signals {
        let upstreams = get_upstreams(world, signal);
        in_degree.insert(signal, upstreams.len());
        upstreams_map.insert(signal, upstreams.clone());
        for upstream in upstreams {
            downstreams_map.entry(upstream).or_default().push(signal);
        }
    }

    let mut queue: VecDeque<SignalSystem> = in_degree
        .iter()
        .filter_map(|(signal, degree)| if *degree == 0 { Some(*signal) } else { None })
        .collect();

    let mut processed = 0usize;
    while let Some(signal) = queue.pop_front() {
        processed += 1;
        let upstreams = upstreams_map.get(&signal).cloned().unwrap_or_default();
        let level = if upstreams.is_empty() {
            0
        } else {
            upstreams
                .iter()
                .filter_map(|u| state.levels.get(u))
                .max()
                .map(|m| m + 1)
                .unwrap_or(0)
        };

        while state.by_level.len() <= level as usize {
            state.by_level.push(Vec::new());
        }
        state.by_level[level as usize].push(signal);
        state.levels.insert(signal, level);

        if let Some(downstreams) = downstreams_map.get(&signal) {
            for downstream in downstreams {
                if let Some(count) = in_degree.get_mut(downstream) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        queue.push_back(*downstream);
                    }
                }
            }
        }
    }

    if processed < in_degree.len() {
        panic!("signal graph contains a cycle or inconsistent edges during level rebuild");
    }

    for bucket in state.by_level.iter_mut() {
        bucket.sort_by_key(|signal| signal.index());
    }
}

fn update_levels_incremental(world: &mut World, state: &mut SignalGraphState) -> bool {
    let mut affected: HashSet<SignalSystem> = HashSet::new();
    let mut queue: VecDeque<SignalSystem> = state.edge_change_seeds.iter().copied().collect();

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

    let mut in_degree: HashMap<SignalSystem, usize> = HashMap::new();
    let mut upstreams_map: HashMap<SignalSystem, Vec<SignalSystem>> = HashMap::new();

    for signal in affected.iter().copied() {
        let upstreams = get_upstreams(world, signal);
        let local_in_degree = upstreams
            .iter()
            .filter(|u| affected.contains(&(**u)))
            .count();
        in_degree.insert(signal, local_in_degree);
        upstreams_map.insert(signal, upstreams);
    }

    let mut queue: VecDeque<SignalSystem> = in_degree
        .iter()
        .filter_map(|(signal, degree)| if *degree == 0 { Some(*signal) } else { None })
        .collect();

    let mut new_levels: HashMap<SignalSystem, u32> = HashMap::new();
    let mut processed = 0usize;

    while let Some(signal) = queue.pop_front() {
        processed += 1;
        let upstreams = upstreams_map.get(&signal).cloned().unwrap_or_default();

        let mut level = 0u32;
        for upstream in upstreams {
            let upstream_level = new_levels
                .get(&upstream)
                .copied()
                .or_else(|| state.levels.get(&upstream).copied())
                .unwrap_or(0);
            level = level.max(upstream_level.saturating_add(1));
        }

        new_levels.insert(signal, level);

        for downstream in get_downstreams(world, signal) {
            if !affected.contains(&downstream) {
                continue;
            }
            if let Some(count) = in_degree.get_mut(&downstream) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    queue.push_back(downstream);
                }
            }
        }
    }

    if processed < affected.len() {
        return false;
    }

    for signal in affected {
        if let Some(old_level) = state.levels.get(&signal).copied() {
            if let Some(bucket) = state.by_level.get_mut(old_level as usize) {
                if let Some(pos) = bucket.iter().position(|s| *s == signal) {
                    bucket.remove(pos);
                }
            }
        }

        if let Some(new_level) = new_levels.get(&signal).copied() {
            while state.by_level.len() <= new_level as usize {
                state.by_level.push(Vec::new());
            }
            insert_sorted_by_index(&mut state.by_level[new_level as usize], signal);
            state.levels.insert(signal, new_level);
        } else {
            state.levels.remove(&signal);
        }
    }

    true
}

fn update_edge_change_levels(world: &mut World, state: &mut SignalGraphState) {
    if state.levels.is_empty() {
        rebuild_levels(world, state);
        state.edge_change_seeds.clear();
        return;
    }

    if state.edge_change_seeds.is_empty() {
        return;
    }

    if !update_levels_incremental(world, state) {
        panic!("signal graph contains a cycle or inconsistent edges during incremental update");
    }
    state.edge_change_seeds.clear();
}

fn remove_signal_from_graph_state_internal(state: &mut SignalGraphState, signal: SignalSystem) {
    if let Some(level) = state.levels.remove(&signal) {
        if let Some(bucket) = state.by_level.get_mut(level as usize) {
            if let Some(pos) = bucket.iter().position(|s| *s == signal) {
                bucket.remove(pos);
            }
        }
    }
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

fn process_signal(
    world: &mut World,
    signal: SignalSystem,
    inputs: &mut HashMap<SignalSystem, VecDeque<Box<dyn AnyClone>>>,
) {
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
                signal,
                upstreams,
                downstreams
            );
        }
    };

    let upstreams = get_upstreams(world, signal);

    if upstreams.is_empty() {
        if let Some(output) = runner.run(world, Box::new(())) {
            enqueue_inputs(world, inputs, signal, output);
        }
        return;
    }

    let mut queue = inputs.remove(&signal).unwrap_or_default();
    while let Some(input) = queue.pop_front() {
        if let Some(output) = runner.run(world, input) {
            enqueue_inputs(world, inputs, signal, output);
        }
    }
}

pub(crate) fn process_signals(
    world: &mut World,
    signals: impl AsRef<[SignalSystem]>,
    input: Box<dyn AnyClone>,
) {
    let signals = signals.as_ref();
    if signals.is_empty() {
        return;
    }

    let mut inputs: HashMap<SignalSystem, VecDeque<Box<dyn AnyClone>>> = HashMap::new();
    let mut iter = signals.iter().copied().peekable();
    if let Some(first) = iter.next() {
        let mut run_with_input = |signal: SignalSystem, input: Box<dyn AnyClone>| {
            let runner = world
                .get::<SystemRunner>(*signal)
                .cloned()
                .unwrap_or_else(|| {
                    panic!(
                        "missing SystemRunner for signal {:?} during processing",
                        signal
                    )
                });
            if let Some(output) = runner.run(world, input) {
                enqueue_inputs(world, &mut inputs, signal, output);
            }
        };

        if iter.peek().is_none() {
            // avoid cloning if there's only a single downstream
            run_with_input(first, input);
        } else {
            let rest: Vec<SignalSystem> = iter.collect();
            if let Some((last, rest)) = rest.split_last() {
                for signal in core::iter::once(first).chain(rest.iter().copied()) {
                    run_with_input(signal, input.clone());
                }
                run_with_input(*last, input);
            }
        }
    }

    let by_level = downstream_levels_from_seeds(world, signals);
    let skip: HashSet<SignalSystem> = signals.iter().copied().collect();
    for level in by_level.into_iter().skip(1) {
        for signal in level {
            if skip.contains(&signal) {
                continue;
            }
            process_signal(world, signal, &mut inputs);
        }
    }
}

pub(crate) fn process_signal_graph(world: &mut World) {
    let mut levels_snapshot: Vec<Vec<SignalSystem>> = Vec::new();
    world.resource_scope(|world, mut state: Mut<SignalGraphState>| {
        update_edge_change_levels(world, &mut state);
        state.is_processing = true;
        levels_snapshot = core::mem::take(&mut state.by_level);
    });

    let mut inputs: HashMap<SignalSystem, VecDeque<Box<dyn AnyClone>>> = HashMap::new();
    for level in &levels_snapshot {
        for &signal in level {
            process_signal(world, signal, &mut inputs);
        }
    }

    let mut state = world.resource_mut::<SignalGraphState>();
    state.is_processing = false;
    state.by_level = levels_snapshot;
    // this indirection allows us to avoid cloning the topological ordering every frame
    apply_deferred_removals(&mut state);
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
/// [`JonmoBuilder`](super::builder::JonmoBuilder) will manage this internally.
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
    if let Some(handles) = world.get_entity_mut(entity).ok().and_then(|mut entity| {
        entity
            .get_mut::<SignalHandles>()
            .map(|mut handles| handles.0.drain(..).collect::<Vec<_>>())
    }) {
        let mut commands = world.commands();
        for handle in handles {
            commands.queue(|world: &mut World| handle.cleanup(world));
        }
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
    O: Clone + 'static,
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
                if let Ok(mut system) = world.get_entity_mut(**signal)
                    && let Some(mut registration_count) = system.get_mut::<SignalRegistrationCount>()
                {
                    registration_count.increment();
                }
                *signal
            }
        }
    }
}

pub(crate) struct LazySignal {
    pub(crate) inner: Arc<LazySignalState>,
}

impl LazySignal {
    pub(crate) fn new<F: FnOnce(&mut World) -> SignalSystem + SSs>(system: F) -> Self {
        LazySignal {
            inner: Arc::new(LazySignalState {
                references: AtomicUsize::new(1),
                system: RwLock::new(LazySystem::System(Some(Box::new(system)))),
            }),
        }
    }

    pub(crate) fn register(self, world: &mut World) -> SignalSystem {
        let signal = self.inner.system.write().unwrap().register(world);
        if let Ok(mut entity) = world.get_entity_mut(*signal)
            && !entity.contains::<LazySignalHolder>()
        {
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
            if let Some(mut state) = world.get_resource_mut::<SignalGraphState>() {
                state.edge_change_seeds.insert(downstream);
            }
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
            if let Ok(entity) = world.get_entity_mut(*signal) {
                entity.despawn();
            }
        }
    }
}

pub(crate) fn lazy_signal_from_system<I, O, IOO, F, M>(system: F) -> LazySignal
where
    I: 'static,
    O: Clone + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<I>, IOO, M> + SSs,
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
    // Stage 1: Decrement registration and bail if the node is still in use.
    if !decrement_registration_and_needs_cleanup(world, signal) {
        return;
    }

    // Stage 2: The count is zero. Perform the full cleanup. First, get the list of parents.
    let upstreams = world.get::<Upstream>(*signal).cloned();

    // Unlink downstream edges and mark affected nodes for level recomputation.
    unlink_downstreams_and_mark(world, signal);

    if should_despawn_signal(world, signal) {
        remove_signal_from_graph_state(world, signal);
        if let Ok(entity) = world.get_entity_mut(*signal) {
            entity.despawn();
        }
    }

    // Stage 3: Notify parents and recurse after processing this node.
    if let Some(upstreams) = upstreams {
        for &upstream_system in upstreams.iter() {
            unlink_from_upstream(world, upstream_system, signal);
            cleanup_recursive(world, upstream_system);
        }
    }
}

fn poll_signal_one_shot(In(signal): In<SignalSystem>, world: &mut World) -> Option<Box<dyn AnyClone>> {
    fn visit(
        world: &mut World,
        node: SignalSystem,
        cache: &mut HashMap<SignalSystem, Option<Box<dyn AnyClone>>>,
    ) -> Option<Box<dyn AnyClone>> {
        // 1. memoisation fast-path
        if let Some(cached) = cache.get(&node) {
            return cached.clone();
        }

        // 2. pull runner + upstream list
        let runner = world
            .get::<SystemRunner>(*node)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "missing SystemRunner for signal {:?} during processing",
                    node
                )
            });
        let upstreams: Vec<SignalSystem> = world
            .get::<Upstream>(*node)
            .map(|u| {
                let mut v: Vec<_> = u.0.iter().copied().collect();
                v.sort_by_key(|s| **s);
                v
            })
            .unwrap_or_default();

        // 3. run the node (depth-first)
        let mut last_output = None;
        if upstreams.is_empty() {
            last_output = runner.run(world, Box::new(()));
        } else {
            for up in upstreams {
                if let Some(input) = visit(world, up, cache)
                    && let Some(out) = runner.run(world, input)
                {
                    last_output = Some(out);
                }
            }
        }
        cache.insert(node, last_output.clone());
        last_output
    }

    let mut cache = HashMap::new();
    visit(world, signal, &mut cache)
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
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, graph::*};
///
/// let mut world = World::new();
/// let signal = *SignalBuilder::from_system(|_: In<()>| 1).register(&mut world);
/// poll_signal(&mut world, signal).and_then(downcast_any_clone::<usize>); // outputs an `Option<usize>`
/// ```
pub fn downcast_any_clone<T: 'static>(any_clone: Box<dyn AnyClone>) -> Option<T> {
    (any_clone as Box<dyn Any>).downcast::<T>().map(|o| *o).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{In, Mut, World};

    #[derive(Resource, Default)]
    struct Order(Vec<&'static str>);

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

        process_signal_graph(&mut world);

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

        process_signal_graph(&mut world);
        world.resource_mut::<Order>().0.clear();

        pipe_signal(&mut world, signal_a, signal_b);
        process_signal_graph(&mut world);

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
}
