use super::utils::*;

use bevy_platform::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, LazyLock, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};
use core::hash::Hash;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::component::HookContext;
use bevy_ecs::{
    prelude::*,
    query::{QueryData, QueryFilter},
    system::{RunSystemOnce, SystemId, SystemState},
    world::DeferredWorld,
};
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, PartialReflect, Reflect};

#[derive(Clone, Copy, Deref, Debug, PartialEq, Eq, Hash, Reflect)]
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

impl SignalSystem {
    /// Returns the underlying [`Entity`] of this signal system.
    pub fn entity(&self) -> Entity {
        self.0
    }
}

// /// Component storing metadata for signal system nodes, primarily for reference counting.
#[derive(Component, Deref)]
pub(crate) struct SignalRegistrationCount(i32);

impl SignalRegistrationCount {
    /// Creates metadata with an initial reference count of 1.
    pub(crate) fn new() -> Self {
        Self(1)
    }

    pub(crate) fn increment(&mut self) {
        self.0 += 1;
    }

    pub(crate) fn decrement(&mut self) {
        self.0 -= 1;
    }
}

/// Helper to register a system, add the [`SystemRunner`] component, and manage [`SignalNodeMetadata`].
///
/// Ensures the system is registered, attaches a runner component, and handles the
/// reference counting via `SignalNodeMetadata`. Returns the `SystemId`.
pub fn register_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    F: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    lazy_signal_from_system(system).register(world)
}

fn downstream_syncer(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    world.commands().queue(move |world: &mut World| {
        let _ = world.run_system_once(
            move |upstreams: Query<&Upstream>,
                  mut downstreams: Query<&mut Downstream>,
                  mut commands: Commands| {
                if let Ok(upstream) = upstreams.get(entity) {
                    for &upstream_system in upstream.iter() {
                        if let Ok(mut downstreams) = downstreams.get_mut(*upstream_system) {
                            downstreams.0.remove(&SignalSystem(entity));
                            if downstreams.0.is_empty() {
                                if let Ok(mut entity) = commands.get_entity(*upstream_system) {
                                    entity.remove::<Downstream>();
                                }
                            }
                        }
                    }
                }
            },
        );
    });
}

// TODO: 0.16 relationships
#[derive(Component, Deref, Clone)]
#[component(on_remove = downstream_syncer)]
pub(crate) struct Upstream(pub(crate) HashSet<SignalSystem>);

impl<'a> IntoIterator for &'a Upstream {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = std::collections::hash_set::Iter<'a, SignalSystem>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Component, Deref)]
pub(crate) struct Downstream(HashSet<SignalSystem>);

impl<'a> IntoIterator for &'a Downstream {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = std::collections::hash_set::Iter<'a, SignalSystem>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub(crate) fn pipe_signal(world: &mut World, source: SignalSystem, target: SignalSystem) {
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
    let mut signals = world.get_resource_mut::<Signals>().unwrap();

}

/// Component holding the type-erased system runner function.
///
/// This component is attached to the entity associated with each registered signal system.
/// It contains an `Arc<Box<dyn Fn(...)>>` that captures the specific `SystemId` and
/// handles the type-erased execution logic, including downcasting inputs and boxing outputs.
#[derive(Component, Clone)]
pub(crate) struct SystemRunner {
    /// The type-erased function to execute the system.
    pub(crate) runner: Arc<
        Box<
            dyn Fn(&mut World, Box<dyn PartialReflect>) -> Option<Box<dyn PartialReflect>>
                + Send
                + Sync
                + 'static,
        >,
    >,
}

impl SystemRunner {
    /// Executes the stored system function with the given type-erased input.
    ///
    /// Takes the `World` and a `Box<dyn PartialReflect>` input, runs the system,
    /// and returns an optional `Box<dyn PartialReflect>` output.
    pub(crate) fn run(
        &self,
        world: &mut World,
        input: Box<dyn PartialReflect>,
    ) -> Option<Box<dyn PartialReflect>> {
        (self.runner)(world, input)
    }
}

fn clone_downstream(downstream: &Downstream) -> Vec<SignalSystem> {
    downstream.iter().cloned().collect()
}

pub(crate) fn process_signals_helper(
    world: &mut World,
    signals: impl IntoIterator<Item = SignalSystem>,
    input: Box<dyn PartialReflect>,
) {
    for signal in signals {
        if let Some(runner) = world
            .get_entity(*signal)
            .ok()
            .and_then(|entity| entity.get::<SystemRunner>().cloned())
        {
            if let Some(output) = runner.run(world, input.to_dynamic()) {
                if let Some(downstream) = world.get::<Downstream>(*signal).map(clone_downstream) {
                    process_signals_helper(world, downstream, output);
                }
            }
        }
    }
}

/// System that drives signal propagation by calling [`SignalPropagator::execute`].
/// Added to the `Update` schedule by the [`JonmoPlugin`]. This system runs once per frame.
/// It temporarily removes the [`SignalPropagator`] resource to allow mutable access to the `World`
/// during system execution within the propagator.
pub(crate) fn process_signals(world: &mut World) {
    let mut orphan_parents = SystemState::<
        Query<Entity, (With<SystemRunner>, Without<Upstream>, With<Downstream>)>,
    >::new(world);
    let orphan_parents = orphan_parents.get(world);
    let orphan_parents = orphan_parents.iter().map(SignalSystem).collect::<Vec<_>>();
    process_signals_helper(world, orphan_parents, Box::new(()));
}

/// Handle returned by [`SignalExt::register`] used for managing the lifecycle of a registered signal chain.
///
/// Contains the [`SignalSystem`] entity representing the final node of the signal chain
/// registered by a specific `register` call.
///
/// Dropping the handle does *not* automatically clean up the underlying systems.
/// Use the [`cleanup`](SignalHandle::cleanup) method for explicit cleanup, which decrements
/// reference counts and potentially despawns systems if their count reaches zero.
#[derive(Clone, Deref, DerefMut, Debug)]
pub struct SignalHandle(pub SignalSystem);

impl From<SignalSystem> for SignalHandle {
    fn from(signal: SignalSystem) -> Self {
        Self(signal)
    }
}

impl SignalHandle {
    /// Creates a new SignalHandle.
    /// This is crate-public to allow construction from other modules.
    pub(crate) fn new(signal: SignalSystem) -> Self {
        Self(signal)
    }

    /// Cleans up the registered signal chain associated with this handle.
    ///
    /// This method traverses upstream from the signal system entity stored in the handle.
    /// For each system encountered (including the starting one), it decrements its
    /// [`SignalRegistrationCount`]. If a system's count reaches zero, its [`Upstream`]
    /// and [`Downstream`] components are removed, effectively disconnecting it from the
    /// signal graph and allowing it to be potentially despawned later if unused elsewhere.
    ///
    /// **Note:** This performs reference counting. The actual systems are only fully removed
    /// when their registration count drops to zero, meaning no other active `SignalHandle`
    /// is still referencing them.
    pub fn cleanup(self, world: &mut World) {
        signal_handle_cleanup_helper(world, [self.0]);
    }
}

trait Runnable<T> {
    fn run(&self, world: &mut World, input: T);
    fn register(&mut self);
}

type DynRunnable<T> = Arc<Mutex<dyn Runnable<T> + Send + Sync + 'static>>;

struct SignalData<I, O, IOO>
where
    I: 'static,
    O: 'static + Clone,
    IOO: Into<Option<O>> + SSs,
{
    system: SystemId<In<I>, IOO>,
    registrations: usize,
    upstreams: HashSet<SignalSystem>,
    downstreams: HashMap<SignalSystem, DynRunnable<O>>,
}

impl<I, O, IOO> SignalData<I, O, IOO>
where
    I: 'static,
    O: 'static + Clone,
    IOO: Into<Option<O>> + SSs,
{
    pub fn new(
        system: SystemId<In<I>, IOO>,
    ) -> Self {
        Self {
            system,
            registrations: 1,
            upstreams: HashSet::new(),
            downstreams: HashMap::new(),
        }
    }
}

impl<I, O, IOO> Runnable<I> for SignalData<I, O, IOO>
where
    I: 'static,
    O: 'static + Clone,
    IOO: Into<Option<O>> + SSs,
{
    fn run(&self, world: &mut World, input: I) {
        if let Ok(Some(output)) = world.run_system_with(self.system, input).map(Into::into) {
            for downstream in self.downstreams.values() {
                // TODO: will deadlock on cycles
                downstream.lock().unwrap().run(world, output.clone())
            }
        }
    }

    fn register(&mut self) {
        self.registrations += 1;
    }
}

#[derive(Resource, Default)]
struct Signals {
    signals: HashSet<SignalSystem>,
    roots: HashMap<SignalSystem, DynRunnable<()>>,
}

impl Signals {
    pub fn add_signal<I, O, IOO>(&mut self, system: F) -> SignalSystem
    where
        I: 'static,
        O: 'static + Clone,
        IOO: Into<Option<O>> + SSs,
    {
        let signal = spawn_signal(self, system);
        self.signals.insert(signal, Box::new(()));
        signal
    }
}

pub(crate) fn spawn_signal<I, O, IOO, F, M>(world: &mut World, system: F) -> SignalSystem
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    F: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    let system = world.register_system(system);
    let entity = system.entity();
    let mut signals = world
        .get_resource_mut::<Signals>()
        .unwrap();
    
    world.entity_mut(entity).insert((
        SignalRegistrationCount::new(),
        SystemRunner {
            runner: Arc::new(Box::new(move |world, input| {
                match I::from_reflect(input.as_ref()) {
                    Some(input) => match world.run_system_with(system, input) {
                        Ok(output) => {
                            if let Some(output) = Into::<Option<O>>::into(output) {
                                Some(Box::new(output) as Box<dyn PartialReflect>)
                            } else {
                                None // terminate
                            }
                        }
                        Err(err) => {
                            warn!("error running system {:?}: {}", system, err);
                            None // terminate on error
                        }
                    },
                    None => {
                        warn!(
                            "failed to downcast input for system {:?}<{:?}>: {:?}",
                            system,
                            input.reflect_type_path(),
                            input
                        );
                        None
                    }
                }
            })),
        },
    ));
    system.into()
}

/// Internal enum used by `RegisterOnceSignal` to track registration state.
pub(crate) struct LazySignalState {
    references: AtomicUsize,
    system: RwLock<LazySystem>,
}

enum LazySystem {
    System(Option<Box<dyn FnOnce(&mut World) -> SignalSystem + Send + Sync + 'static>>),
    Registered(SignalSystem),
}

impl LazySystem {
    /// Registers the system if it hasn't been registered yet.
    /// Returns the system ID of the registered system.
    pub fn register(&mut self, world: &mut World) -> SignalSystem {
        match self {
            LazySystem::System(f) => {
                let signal = f.take().unwrap()(world).into();
                *self = LazySystem::Registered(signal);
                signal
            }
            LazySystem::Registered(signal) => {
                let mut signals = world
                    .get_resource_mut::<Signals>()
                    .unwrap();
                if let Some(data) = signals.signals.get_mut(signal) {
                    data.register()
                }
                if let Ok(mut system) = world.get_entity_mut(**signal) {
                    if let Some(mut registration_count) =
                        system.get_mut::<SignalRegistrationCount>()
                    {
                        registration_count.increment();
                    }
                }
                *signal
            }
        }
    }
}

/// A helper struct to ensure a signal system is registered only once in the `World`.
///
/// This struct wraps the registration logic. When `register` is called, it checks
/// if the system has already been registered. If not, it runs the provided closure
/// to create and register the system. If it has, it increments the registration count
/// for the existing system.
#[derive(Reflect)]
#[reflect(Clone)]
#[reflect(opaque)]
pub(crate) struct LazySignal {
    inner: Arc<LazySignalState>,
}

impl LazySignal {
    pub fn new<F: FnOnce(&mut World) -> SignalSystem + SSs>(system: F) -> Self {
        LazySignal {
            inner: Arc::new(LazySignalState {
                references: AtomicUsize::new(1),
                system: RwLock::new(LazySystem::System(Some(Box::new(system)))),
            }),
        }
    }

    pub fn register(self, world: &mut World) -> SignalSystem {
        let signal = self.inner.system.write().unwrap().register(world);
        if let Ok(mut entity) = world.get_entity_mut(*signal) {
            if !entity.contains::<LazySignalHolder>() {
                entity.insert(LazySignalHolder(self));
            }
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
        if self.inner.references.fetch_sub(1, Ordering::SeqCst) <= 2 {
            if let LazySystem::Registered(signal) = *self.inner.system.read().unwrap() {                
                CLEANUP_SIGNALS.lock().unwrap().push(signal);
            }
        }
    }
}

#[derive(Component)]
pub(crate) struct LazySignalHolder(LazySignal);

pub(crate) static CLEANUP_SIGNALS: LazyLock<Mutex<Vec<SignalSystem>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

pub(crate) fn flush_cleanup_signals(world: &mut World) {
    let signals = CLEANUP_SIGNALS
        .lock()
        .unwrap()
        .drain(..)
        .collect::<Vec<_>>();
    for signal in signals {
        if let Ok(entity) = world.get_entity_mut(*signal) {
            if let Some(registration_count) = entity.get::<SignalRegistrationCount>() {
                if **registration_count == 0 {
                    println!("LazySignal despawning for signal: {:?}", signal);
                    entity.despawn();
                }
            }
        }
    }
}

pub(crate) fn lazy_signal_from_system<I, O, IOO, F, M>(system: F) -> LazySignal
where
    I: FromReflect + SSs,
    O: FromReflect + SSs,
    IOO: Into<Option<O>> + SSs,
    F: IntoSystem<In<I>, IOO, M> + SSs,
    M: SSs,
{
    LazySignal::new(move |world: &mut World| spawn_signal(world, system))
}

/// An iterator that traverses *upstream* signal dependencies.
///
/// Starting from a given signal system entity, it yields the entity IDs of its direct
/// and indirect upstream dependencies (systems that provide input to it).
pub(crate) struct UpstreamIter<'w, 's, D: QueryData, F: QueryFilter>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Upstream>,
{
    upstreams_query: &'w Query<'w, 's, D, F>,
    upstreams: VecDeque<SignalSystem>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Upstream>,
{
    /// Returns a new [`DescendantIter`].
    pub fn new(upstreams_query: &'w Query<'w, 's, D, F>, signal: SignalSystem) -> Self {
        UpstreamIter {
            upstreams_query,
            upstreams: upstreams_query
                .get(*signal)
                .into_iter()
                .flatten()
                .copied()
                .collect(),
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for UpstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Upstream>,
{
    type Item = SignalSystem;

    fn next(&mut self) -> Option<Self::Item> {
        let signal = self.upstreams.pop_front()?;

        if let Ok(upstream) = self.upstreams_query.get(*signal) {
            self.upstreams.extend(upstream);
        }

        Some(signal)
    }
}

/// An iterator that traverses *downstream* signal dependencies.
///
/// Starting from a given signal system entity, it yields the entity IDs of its direct
/// and indirect downstream dependencies (systems that consume its output).
#[allow(dead_code)] // Currently unused within the crate, but potentially useful
pub(crate) struct DownstreamIter<'w, 's, D: QueryData, F: QueryFilter>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Downstream>,
{
    downstreams_query: &'w Query<'w, 's, D, F>,
    downstreams: VecDeque<SignalSystem>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Downstream>,
{
    /// Returns a new [`DescendantIter`].
    #[allow(dead_code)] // Currently unused within the crate
    pub fn new(downstreams_query: &'w Query<'w, 's, D, F>, signal: SignalSystem) -> Self {
        DownstreamIter {
            downstreams_query,
            downstreams: downstreams_query
                .get(*signal)
                .into_iter()
                .flatten()
                .copied()
                .collect(),
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for DownstreamIter<'w, 's, D, F>
where
    D::ReadOnly: QueryData<Item<'w> = &'w Downstream>,
{
    type Item = SignalSystem;

    fn next(&mut self) -> Option<Self::Item> {
        let signal = self.downstreams.pop_front()?;

        if let Ok(downstream) = self.downstreams_query.get(*signal) {
            self.downstreams.extend(downstream);
        }

        Some(signal)
    }
}

pub(crate) fn signal_handle_cleanup_helper(
    world: &mut World,
    signals: impl IntoIterator<Item = SignalSystem>,
) {
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
            if no_registrations {
                if let Some(LazySignalHolder(lazy_signal)) = entity.get::<LazySignalHolder>() {
                    if lazy_signal.inner.references.load(Ordering::SeqCst) == 1 {
                        entity.despawn();
                    }
                }
            }
        }
    }
}

// TODO: optimize this
pub fn poll_signal_one_shot(
    In(signal): In<SignalSystem>,
    world: &mut World,
) -> Option<Box<dyn PartialReflect>> {
    // TODO: implications of this cached system "leaking" ? i guess it will be used for all flattens, but will truly leak if all flattens are dropped ...
    if let Ok(ancestor_orphans) = world.run_system_cached_with(
        move |In(signal): In<SignalSystem>, upstreams: Query<&Upstream>| {
            UpstreamIter::new(&upstreams, signal)
                .filter(|upstream| !upstreams.contains(**upstream))
                .collect::<Vec<_>>()
        },
        signal,
    ) {
        return process_signals_helper(world, ancestor_orphans, Box::new(()), Some(signal));
    }
    None
}

pub fn poll_signal(world: &mut World, signal: SignalSystem) -> Option<Box<dyn PartialReflect>> {
    world
        .run_system_cached_with(poll_signal_one_shot, signal)
        .ok()
        .flatten()
}
