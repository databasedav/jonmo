//! Signal graph management and runtime.

use super::utils::*;

use bevy_derive::Deref;
use bevy_ecs::{
    component::HookContext,
    prelude::*,
    query::{QueryData, QueryFilter},
    system::{RunSystemOnce, SystemId, SystemState},
    world::DeferredWorld,
};
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

fn downstream_syncer(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    world.commands().queue(move |world: &mut World| {
        let _ = world.run_system_once(
            move |upstreams: Query<&Upstream>, mut downstreams: Query<&mut Downstream>, mut commands: Commands| {
                if let Ok(upstream) = upstreams.get(entity) {
                    for &upstream_system in upstream.iter() {
                        if let Ok(mut downstreams) = downstreams.get_mut(*upstream_system) {
                            downstreams.0.remove(&SignalSystem(entity));
                            if downstreams.0.is_empty()
                                && let Ok(mut entity) = commands.get_entity(*upstream_system)
                            {
                                entity.remove::<Downstream>();
                            }
                        }
                    }
                }
            },
        );
    });
}

// TODO: many to many relationships
#[derive(Component, Deref, Clone)]
#[component(on_remove = downstream_syncer)]
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
            Err(error) => {
                error!(
                    "failed to downcast input to type for system {:?}: {:?}",
                    self.system, error
                );
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

// #[derive(Component)]
// pub(crate) struct SkipOnce;

pub(crate) fn process_signals(
    world: &mut World,
    signals: impl IntoIterator<Item = SignalSystem>,
    input: Box<dyn AnyClone>,
) {
    for signal in signals {
        if let Some(runner) = world
            .get_entity(*signal)
            .ok()
            .and_then(|entity| entity.get::<SystemRunner>().cloned())
            && let Some(output) = runner.run(world, input.clone())
            && let Some(downstream) = world.get::<Downstream>(*signal).cloned()
        {
            let targets = downstream
                .iter()
                .copied()
                // .filter(|&signal| {
                //     let mut entity = world.entity_mut(*signal);
                //     let filter = entity.get::<SkipOnce>().is_none();
                //     if !filter {
                //         entity.remove::<SkipOnce>();
                //     }
                //     filter
                // })
                .collect::<Vec<_>>();
            process_signals(world, targets, output);
        }
    }
}

pub(crate) fn process_signal_graph(world: &mut World) {
    let mut orphan_parents =
        SystemState::<Query<Entity, (With<SystemRunner>, Without<Upstream>, With<Downstream>)>>::new(world);
    let orphan_parents = orphan_parents.get(world);
    let orphan_parents = orphan_parents.iter().map(SignalSystem).collect::<Vec<_>>();
    process_signals(world, orphan_parents, Box::new(()));
}

/// Handle to a particular node of the signal graph, returned by
/// [`SignalExt::register`](super::signal::SignalExt),
/// [`SignalVecExt::register`](super::signal_vec::SignalVecExt::register), and
/// [`SignalMapExt::register`](super::signal_map::SignalMapExt::register). In order for signals to
/// be appropriately cleaned up, for every call to `.register` made to some particular signal or its
/// clones, [`SignalHandle::cleanup`] must be called on a corresponding [`SignalHandle`] or a
/// downstream [`SignalHandle`]. Adding [`SignalHandle`]s to the [`SignalHandles`] [`Component`]
/// will take care of this when the corresponding [`Entity`] is despawned, and using the
/// [`JonmoBuilder`](super::builder::JonmoBuilder) will manage this internally.
#[derive(Clone, Deref)]
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

    /// Decrements the usage tracking of the corresponding signal and all its upstreams,
    /// potentially despawning the backing [`System`], see [`SignalHandle`].
    pub fn cleanup(self, world: &mut World) {
        signal_handle_cleanup_helper(world, [self.0]);
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
#[derive(Component, Default)]
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
    let sys_id = world.register_system(system);
    let entity = sys_id.entity();

    // Wrap the typed node behind NodeErased
    let runner: Arc<Box<dyn Runnable>> = Arc::new(Box::new(SystemHolder::<I, O, IOO> {
        system: sys_id,
        _marker: PhantomData,
    }));

    world.entity_mut(entity).insert((
        SignalRegistrationCount::new(),
        SystemRunner {
            runner: Arc::new(Box::new(move |w, inp| runner.run(w, inp))),
        },
    ));

    entity.into()
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
            CLEANUP_SIGNALS.lock().unwrap().push(signal);
        }
    }
}

#[derive(Component)]
pub(crate) struct LazySignalHolder(LazySignal);

static CLEANUP_SIGNALS: LazyLock<Mutex<Vec<SignalSystem>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub(crate) fn flush_cleanup_signals(world: &mut World) {
    let signals = CLEANUP_SIGNALS.lock().unwrap().drain(..).collect::<Vec<_>>();
    for signal in signals {
        if let Ok(entity) = world.get_entity_mut(*signal)
            && let Some(registration_count) = entity.get::<SignalRegistrationCount>()
            && **registration_count == 0
        {
            entity.despawn();
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
    D::ReadOnly: QueryData<Item<'w> = &'w Upstream>,
{
    upstreams_query: &'w Query<'w, 's, D, F>,
    upstreams: Vec<SignalSystem>,
}

#[allow(dead_code)]
impl<'w, 's, D: QueryData, F: QueryFilter> UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Upstream>,
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
    D::ReadOnly: QueryData<Item<'w> = &'w Upstream>,
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
    D::ReadOnly: QueryData<Item<'w> = &'w Downstream>,
{
    downstreams_query: &'w Query<'w, 's, D, F>,
    downstreams: Vec<SignalSystem>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Downstream>,
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
    D::ReadOnly: QueryData<Item<'w> = &'w Downstream>,
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

fn signal_handle_cleanup_helper(world: &mut World, signals: impl IntoIterator<Item = SignalSystem>) {
    for signal in signals {
        if let Some(upstreams) = world.get::<Upstream>(*signal).cloned() {
            signal_handle_cleanup_helper(world, upstreams.0);
        }
        if let Ok(mut entity) = world.get_entity_mut(*signal) {
            let mut no_registrations = false;
            if let Some(mut registration_count) = entity.get_mut::<SignalRegistrationCount>() {
                registration_count.decrement();
                if **registration_count == 0 {
                    entity.remove::<Upstream>();
                    entity.remove::<Downstream>();
                    no_registrations = true;
                }
            }
            if no_registrations
                && let Some(LazySignalHolder(lazy_signal)) = entity.get::<LazySignalHolder>()
                && lazy_signal.inner.references.load(Ordering::SeqCst) == 1
            {
                entity.despawn();
            }
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
        let runner = match world.get::<SystemRunner>(*node) {
            Some(r) => r.clone(),
            None => {
                cache.insert(node, None);
                return None;
            }
        };

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
/// ```no_run
/// poll_signal(world, signal).and_then(downcast_any_clone::<usize>) // assuming this `signal` outputs a `usize`, this will return an `Option<usize>`
/// ```
pub fn downcast_any_clone<T: 'static>(any_clone: Box<dyn AnyClone>) -> Option<T> {
    (any_clone as Box<dyn Any>).downcast::<T>().map(|o| *o).ok()
}
