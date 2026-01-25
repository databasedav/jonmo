//! Reactive entity builder ported from [Dominator](https://github.com/Pauan/rust-dominator)'s [`DomBuilder`](https://docs.rs/dominator/latest/dominator/struct.DomBuilder.html).
use super::{
    graph::{SignalHandle, SignalHandles},
    signal::{Signal, SignalExt},
    signal_map::{SignalMap, SignalMapExt},
    signal_vec::{SignalVec, SignalVecExt, VecDiff},
    utils::{LazyEntity, SSs},
};
use bevy_ecs::{
    component::Mutable,
    lifecycle::HookContext,
    prelude::*,
    system::{IntoObserverSystem, RunSystemOnce},
    world::{DeferredWorld, error::EntityMutableFetchError},
};
use bevy_platform::prelude::*;
use core::sync::atomic::{AtomicUsize, Ordering};

// TODO: the fluent interface link breaks cargo fmt ??
/// A thin facade over a Bevy [`Entity`] enabling the ergonomic registration of
/// reactive components and children using a declarative
/// [fluent](https://en.wikipedia.org/wiki/Fluent_interface) builder pattern. All
/// its methods are deferred until the corresponding [`Entity`] is spawned so its
/// state *and how that state should change* depending on the state of the
/// [`World`] can be specified up front, in a tidy colocated package, without a
/// `&mut World` or [`Commands`].
///
/// Port of [Dominator](https://github.com/Pauan/rust-dominator)'s
/// [`DomBuilder`](https://docs.rs/dominator/latest/dominator/struct.DomBuilder.html).
///
/// # `Clone` semantics
///
/// This type implements [`Clone`] **only** to satisfy trait bounds required by signal combinators.
/// **Cloning [`jonmo::Builder`](Builder)s at runtime is a bug.** See
/// [`jonmo::Builder::clone`](Builder::clone) for details.
#[derive(Default)]
pub struct Builder {
    #[allow(clippy::type_complexity)]
    on_spawns: Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>,
    /// Counter for assigning block indices during builder construction.
    /// Each child/children/child_signal/children_signal_vec call gets a unique block index.
    next_block: AtomicUsize,
}

impl Clone for Builder {
    /// # Warning
    ///
    /// This clone implementation exists **only** to satisfy trait bounds required by signal
    /// combinators (e.g., [`SignalExt::map`], [`SignalVecExt::filter_map`]). **Cloning
    /// [`jonmo::Builder`](Builder)s at runtime is a bug and will lead to unexpected behavior.**
    ///
    /// Clones produce an empty builder with no on-spawn hooks. Use factory functions instead
    /// if you need reusable UI templates:
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// fn my_widget(label: &str) -> jonmo::Builder {
    ///     jonmo::Builder::new().insert(Name::new(label.to_string()))
    ///     // ... other configuration
    /// }
    ///
    /// // Correct: each call creates a fresh builder
    /// let mut world = World::new();
    /// let widget1 = my_widget("First").spawn(&mut world);
    /// let widget2 = my_widget("Second").spawn(&mut world);
    /// ```
    #[track_caller]
    fn clone(&self) -> Self {
        let msg = format!(
            "Cloning `jonmo::Builder` at {} is a bug! Use a factory function instead.",
            core::panic::Location::caller()
        );
        if cfg!(debug_assertions) {
            panic!("{}", msg);
        }
        #[cfg(feature = "tracing")]
        bevy_log::error!("{}", msg);

        Self::default()
    }
}

impl<T: Bundle> From<T> for Builder {
    fn from(bundle: T) -> Self {
        Self::new().insert(bundle)
    }
}

/// Component that tracks the population of each child block for offset calculation.
/// Used internally by reactive child methods.
#[derive(Component, Default)]
struct ChildBlockPopulations(Vec<usize>);

impl ChildBlockPopulations {
    /// Ensure the vector is large enough to store population for the given block index.
    fn ensure_block(&mut self, block: usize) {
        if self.0.len() <= block {
            self.0.resize(block + 1, 0);
        }
    }
}

impl Builder {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a function with mutable access to the [`World`] and this builder's [`Entity`].
    pub fn on_spawn(mut self, on_spawn: impl FnOnce(&mut World, Entity) + SSs) -> Self {
        self.on_spawns.push(Box::new(on_spawn));
        self
    }

    /// Run a [`System`] which takes [`In`] this builder's [`Entity`].
    pub fn on_spawn_with_system<T, M>(self, system: T) -> Self
    where
        T: IntoSystem<In<Entity>, (), M> + SSs,
    {
        self.on_spawn(|world, entity| {
            #[allow(unused_variables)]
            if let Err(error) = world.run_system_once_with(system, entity) {
                #[cfg(feature = "tracing")]
                bevy_log::error!("failed to run system on spawn: {}", error);
            }
        })
    }

    /// Attach [`SignalTask`]s to this builder for automatic cleanup on despawn.
    ///
    /// Use [`.task()`](SignalTaskExt::task) to convert a [`Signal`], [`SignalVec`], or
    /// [`SignalMap`] into a [`SignalTask`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone)]
    /// struct MyResource {
    ///     value: i32,
    /// }
    ///
    /// let my_signal_task = signal::from_resource::<MyResource>()
    ///     .map_in(|r: MyResource| println!("{}", r.value))
    ///     .task();
    ///
    /// let mut world = World::new();
    /// let my_mutable_vec = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
    /// let my_signal_vec_task = my_mutable_vec
    ///     .signal_vec()
    ///     .map_in(|item: i32| println!("{}", item))
    ///     .task();
    ///
    /// jonmo::Builder::new().hold_tasks([my_signal_task, my_signal_vec_task]);
    /// ```
    pub fn hold_tasks(self, tasks: impl IntoIterator<Item = Box<dyn SignalTask>> + SSs) -> Self {
        self.on_spawn(move |world, entity| {
            let handles: Vec<_> = tasks.into_iter().map(|task| task.register_task(world)).collect();
            add_handles(world, entity, handles);
        })
    }

    /// Run a function with this builder's [`EntityWorldMut`].
    pub fn with_entity(self, f: impl FnOnce(EntityWorldMut) + SSs) -> Self {
        self.on_spawn(move |world, entity| {
            f(world.entity_mut(entity));
        })
    }

    /// Adds a [`Bundle`] onto this builder's [`Entity`].
    pub fn insert<T: Bundle>(self, bundle: T) -> Self {
        self.with_entity(move |mut entity| {
            entity.insert(bundle);
        })
    }

    /// Run a function with mutable access (via [`Mut`]) to this builder's `C` [`Component`] if it
    /// exists.
    pub fn with_component<C: Component<Mutability = Mutable>>(self, f: impl FnOnce(Mut<C>) + SSs) -> Self {
        self.with_entity(|mut entity| {
            if let Some(component) = entity.get_mut::<C>() {
                f(component);
            }
        })
    }

    /// Attach an [`Observer`] to this builder.
    pub fn observe<E: EntityEvent, B: Bundle, Marker>(
        self,
        observer: impl IntoObserverSystem<E, B, Marker> + Sync,
    ) -> Self {
        self.on_spawn(|world, entity| {
            world.entity_mut(entity).observe(observer);
        })
    }

    /// When this builder's [`Entity`] is despawned, run a function with mutable access to the
    /// [`DeferredWorld`] and this builder's [`Entity`].
    pub fn on_despawn(self, on_despawn: impl FnOnce(&mut DeferredWorld, Entity) + Send + Sync + 'static) -> Self {
        self.on_spawn(|world, entity| {
            let mut entity_mut = world.entity_mut(entity);
            if let Some(mut on_despawn_component) = entity_mut.get_mut::<OnDespawnCallbacks>() {
                on_despawn_component.0.push(Box::new(on_despawn));
            } else {
                entity_mut.insert(OnDespawnCallbacks(vec![Box::new(on_despawn)]));
            }
        })
    }

    /// Set the [`LazyEntity`] to this builder's [`Entity`].
    #[track_caller]
    pub fn lazy_entity(self, entity: LazyEntity) -> Self {
        self.on_spawn(move |_, e| entity.set(e))
    }

    /// Reactively run a [`System`] which takes [`In`] this builder's [`Entity`] and the output of a
    /// [`Signal`].
    ///
    /// The `signal` will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn on_signal<I, S, F, M>(self, signal: S, system: F) -> Self
    where
        I: Clone + 'static,
        S: Signal<Item = I> + SSs,
        F: IntoSystem<In<(Entity, I)>, (), M> + SSs,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            let handle =
                Signal::register_signal(signal.map(move |In(input): In<I>| (entity, input)).map(system), world);
            add_handles(world, entity, [handle]);
        };
        self.on_spawn(on_spawn)
    }

    /// Reactively run a function with this builder's [`EntityWorldMut`] and the output of a
    /// [`Signal`].
    ///
    /// The `signal` will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn on_signal_with_entity<I, S, F>(self, signal: S, mut f: F) -> Self
    where
        I: Clone + 'static,
        S: Signal<Item = I> + SSs,
        F: FnMut(EntityWorldMut, I) + SSs,
    {
        self.on_signal(signal, move |In((entity, value)), world: &mut World| {
            f(world.entity_mut(entity), value)
        })
    }

    /// Reactively run a function with mutable access (via [`Mut`]) to this builder's [`Entity`]'s
    /// `C` [`Component`] if it exists and the output of a [`Signal`].
    ///
    /// The `signal` will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn on_signal_with_component<C, I, S, F>(self, signal: S, mut f: F) -> Self
    where
        C: Component<Mutability = Mutable>,
        I: Clone + 'static,
        S: Signal<Item = I> + SSs,
        F: FnMut(Mut<C>, I) + SSs,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            let handle = Signal::register_signal(
                signal.map(move |In(input): In<I>, mut components: Query<&mut C>| {
                    if let Ok(component) = components.get_mut(entity) {
                        f(component, input)
                    }
                }),
                world,
            );
            add_handles(world, entity, [handle]);
        };
        self.on_spawn(on_spawn)
    }

    /// Reactively set this builder's [`Entity`]'s `C` [`Component`] with a [`Signal`] that outputs
    /// an [`Option`]al `C`; if the [`Signal`] outputs [`None`], the `C` [`Component`] is removed.
    pub fn component_signal<C, S>(self, signal: S) -> Self
    where
        C: Component + Clone,
        S: Signal<Item = Option<C>> + SSs,
    {
        self.on_signal(
            signal,
            move |In((entity, component_option)): In<(Entity, Option<C>)>, world: &mut World| {
                let mut entity = world.entity_mut(entity);
                if let Some(component) = component_option {
                    entity.insert(component);
                } else {
                    entity.remove::<C>();
                }
            },
        )
    }

    /// Declare a static child.
    pub fn child(self, child: impl Into<Builder>) -> Self {
        let block = self.next_block.fetch_add(1, Ordering::Relaxed);
        let child = child.into();
        let on_spawn = move |world: &mut World, parent| {
            let child_entity = world.spawn_empty().id();
            // Ensure the populations vec is large enough and set this block's population
            let mut pops = world.get_mut::<ChildBlockPopulations>(parent).unwrap();
            pops.ensure_block(block);
            pops.0[block] = 1;
            let insert_offset = offset(block, &pops.0);
            // need to call like this to avoid type ambiguity
            EntityWorldMut::insert_children(&mut world.entity_mut(parent), insert_offset, &[child_entity]);
            child.spawn_on_entity(world, child_entity).unwrap();
        };
        self.on_spawn(on_spawn)
    }

    /// Declare a reactive child. When the [`Signal`] outputs [`None`], the child is removed.
    pub fn child_signal(self, child_option: impl Signal<Item = Option<Builder>>) -> Self {
        let block = self.next_block.fetch_add(1, Ordering::Relaxed);
        let on_spawn = move |world: &mut World, parent: Entity| {
            // Initialize this block's population to 0
            let mut pops = world.get_mut::<ChildBlockPopulations>(parent).unwrap();
            pops.ensure_block(block);

            let system = move |In(child_option): In<Option<Builder>>,
                               world: &mut World,
                               mut existing_child_option: Local<Option<Entity>>| {
                if let Some(child) = child_option {
                    if let Some(existing_child) = existing_child_option.take() {
                        world.entity_mut(existing_child).despawn();
                    }
                    let child_entity = world.spawn_empty().id();
                    let pops = world.get::<ChildBlockPopulations>(parent).unwrap();
                    let insert_offset = offset(block, &pops.0);
                    world.entity_mut(parent).insert_children(insert_offset, &[child_entity]);
                    child.spawn_on_entity(world, child_entity).unwrap();
                    *existing_child_option = Some(child_entity);
                    world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] = 1;
                } else {
                    if let Some(existing_child) = existing_child_option.take() {
                        world.entity_mut(existing_child).despawn();
                    }
                    world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] = 0;
                }
            };
            let handle = child_option.map(system).register(world);
            add_handles(world, parent, [handle]);
        };
        self.on_spawn(on_spawn)
    }

    /// Declare static children.
    pub fn children(self, children: impl IntoIterator<Item = impl Into<Builder>> + Send + 'static) -> Self {
        let block = self.next_block.fetch_add(1, Ordering::Relaxed);
        let children_vec: Vec<Builder> = children.into_iter().map(Into::into).collect();
        let population = children_vec.len();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let mut children_entities = Vec::with_capacity(children_vec.len());
            for _ in 0..children_vec.len() {
                children_entities.push(world.spawn_empty().id());
            }
            // Set this block's population
            let mut pops = world.get_mut::<ChildBlockPopulations>(parent).unwrap();
            pops.ensure_block(block);
            pops.0[block] = population;
            let insert_offset = offset(block, &pops.0);
            world
                .entity_mut(parent)
                .insert_children(insert_offset, &children_entities);
            for (child, child_entity) in children_vec.into_iter().zip(children_entities.iter().copied()) {
                child.spawn_on_entity(world, child_entity).unwrap();
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare reactive children.
    pub fn children_signal_vec(self, children_signal_vec: impl SignalVec<Item = Builder>) -> Self {
        let block = self.next_block.fetch_add(1, Ordering::Relaxed);
        let on_spawn = move |world: &mut World, parent: Entity| {
            // Initialize this block's population to 0
            let mut pops = world.get_mut::<ChildBlockPopulations>(parent).unwrap();
            pops.ensure_block(block);

            let system = move |In(diffs): In<Vec<VecDiff<Builder>>>,
                               world: &mut World,
                               mut children_entities: Local<Vec<Entity>>| {
                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values: children } => {
                            for child_entity in children_entities.drain(..) {
                                world.entity_mut(child_entity).despawn();
                            }
                            *children_entities = children.iter().map(|_| world.spawn_empty().id()).collect();
                            let pops = world.get::<ChildBlockPopulations>(parent).unwrap();
                            let insert_offset = offset(block, &pops.0);
                            world
                                .entity_mut(parent)
                                .insert_children(insert_offset, &children_entities);
                            for (child, child_entity) in children.into_iter().zip(children_entities.iter().copied()) {
                                child.spawn_on_entity(world, child_entity).unwrap();
                            }
                            world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] = children_entities.len();
                        }
                        VecDiff::InsertAt { index, value: child } => {
                            let child_entity = world.spawn_empty().id();
                            let pops = world.get::<ChildBlockPopulations>(parent).unwrap();
                            let insert_offset = offset(block, &pops.0);
                            world
                                .entity_mut(parent)
                                .insert_children(insert_offset + index, &[child_entity]);
                            child.spawn_on_entity(world, child_entity).unwrap();
                            children_entities.insert(index, child_entity);
                            world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] = children_entities.len();
                        }
                        VecDiff::Push { value: child } => {
                            let child_entity = world.spawn_empty().id();
                            let pops = world.get::<ChildBlockPopulations>(parent).unwrap();
                            let insert_offset = offset(block, &pops.0);
                            world
                                .entity_mut(parent)
                                .insert_children(insert_offset + children_entities.len(), &[child_entity]);
                            child.spawn_on_entity(world, child_entity).unwrap();
                            children_entities.push(child_entity);
                            world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] = children_entities.len();
                        }
                        VecDiff::UpdateAt { index, value: node } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                world.entity_mut(existing_child).despawn(); // removes from parent
                            }
                            let child_entity = world.spawn_empty().id();
                            let pops = world.get::<ChildBlockPopulations>(parent).unwrap();
                            let insert_offset = offset(block, &pops.0);
                            world
                                .entity_mut(parent)
                                .insert_children(insert_offset + index, &[child_entity]);
                            node.spawn_on_entity(world, child_entity).unwrap();
                            children_entities[index] = child_entity;
                        }
                        VecDiff::Move { old_index, new_index } => {
                            // First, update our local tracker to match the new logical order. This is the
                            // source of truth for which entity corresponds to which index.
                            let moved_entity = children_entities.remove(old_index);
                            children_entities.insert(new_index, moved_entity);

                            // Now, apply the same reordering to the actual parent entity in the world.
                            let mut parent = world.entity_mut(parent);
                            // Bevy's `remove_children` finds the entity and removes it from its
                            // current position, correctly shifting subsequent children.
                            parent.remove_children(&[moved_entity]);
                            let parent_entity = parent.id();

                            // The new insertion index must be calculated with the offset from any
                            // preceding static children.
                            let pops = world.get::<ChildBlockPopulations>(parent_entity).unwrap();
                            let insert_offset = offset(block, &pops.0);
                            world
                                .entity_mut(parent_entity)
                                .insert_children(insert_offset + new_index, &[moved_entity]);
                        }
                        VecDiff::RemoveAt { index } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                world.entity_mut(existing_child).despawn();
                                children_entities.remove(index);
                                world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::Pop => {
                            if let Some(child_entity) = children_entities.pop() {
                                world.entity_mut(child_entity).despawn();
                                world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::Clear => {
                            for child_entity in children_entities.drain(..) {
                                world.entity_mut(child_entity).despawn();
                            }
                            world.get_mut::<ChildBlockPopulations>(parent).unwrap().0[block] = children_entities.len();
                        }
                    }
                }
            };
            let handle = children_signal_vec.for_each(system).register(world);
            add_handles(world, parent, [handle]);
        };
        self.on_spawn(on_spawn)
    }

    /// Spawn this builder on an existing [`Entity`].
    ///
    /// # Errors
    ///
    /// Returns an error if the entity does not exist in the world.
    pub fn spawn_on_entity(self, world: &mut World, entity: Entity) -> Result<(), EntityMutableFetchError> {
        let mut entity_mut = world.get_entity_mut(entity)?;
        let id = entity_mut.id();
        entity_mut.insert((SignalHandles::default(), ChildBlockPopulations::default()));
        for on_spawn in self.on_spawns {
            on_spawn(world, id);
        }
        Ok(())
    }

    /// Spawn this builder into the [`World`].
    pub fn spawn(self, world: &mut World) -> Entity {
        let entity = world.spawn_empty().id();
        self.spawn_on_entity(world, entity).unwrap();
        entity
    }
}

/// A type-erased signal that can be managed as a free-floating "task".
///
/// This trait enables [`Builder::hold_tasks`] to accept signals that haven't
/// been registered yet, deferring their registration until the entity is spawned.
///
/// Use the [`.task()`](SignalTaskExt::task) method to convert a [`Signal`],
/// [`SignalVec`], or [`SignalMap`] into a [`SignalTask`].
pub trait SignalTask: Send + Sync + 'static {
    /// Register this signal task with the world and return its handle.
    fn register_task(self: Box<Self>, world: &mut World) -> SignalHandle;
}

struct SignalTaskSignal<S>(S);

impl<S: Signal + SSs> SignalTask for SignalTaskSignal<S> {
    fn register_task(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.0.register(world)
    }
}

/// Extension trait for converting a [`Signal`] into a [`SignalTask`].
pub trait SignalTaskExt: Signal + Sized + SSs {
    /// Convert this signal into a type-erased [`SignalTask`] for use with
    /// [`Builder::hold_tasks`].
    fn task(self) -> Box<dyn SignalTask> {
        Box::new(SignalTaskSignal(self))
    }
}

impl<S: Signal + Sized + SSs> SignalTaskExt for S {}

struct SignalTaskSignalVec<S>(S);

impl<S: SignalVec + SSs> SignalTask for SignalTaskSignalVec<S> {
    fn register_task(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.0.register(world)
    }
}

/// Extension trait for converting a [`SignalVec`] into a [`SignalTask`].
pub trait SignalVecTaskExt: SignalVec + Sized + SSs {
    /// Convert this signal vec into a type-erased [`SignalTask`] for use with
    /// [`Builder::hold_tasks`].
    fn task(self) -> Box<dyn SignalTask> {
        Box::new(SignalTaskSignalVec(self))
    }
}

impl<S: SignalVec + Sized + SSs> SignalVecTaskExt for S {}

struct SignalTaskSignalMap<S>(S);

impl<S: SignalMap + SSs> SignalTask for SignalTaskSignalMap<S> {
    fn register_task(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.0.register(world)
    }
}

/// Extension trait for converting a [`SignalMap`] into a [`SignalTask`].
pub trait SignalMapTaskExt: SignalMap + Sized + SSs {
    /// Convert this signal map into a type-erased [`SignalTask`] for use with
    /// [`Builder::hold_tasks`].
    fn task(self) -> Box<dyn SignalTask> {
        Box::new(SignalTaskSignalMap(self))
    }
}

impl<S: SignalMap + Sized + SSs> SignalMapTaskExt for S {}

fn on_despawn_hook(mut world: DeferredWorld, ctx: HookContext) {
    let entity = ctx.entity;
    let fs = world
        .get_mut::<OnDespawnCallbacks>(entity)
        .unwrap()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for f in fs {
        f(&mut world, entity);
    }
}

#[allow(clippy::type_complexity)]
#[derive(Component)]
#[component(on_remove = on_despawn_hook)]
struct OnDespawnCallbacks(Vec<Box<dyn FnOnce(&mut DeferredWorld, Entity) + Send + Sync + 'static>>);

fn add_handles<I>(world: &mut World, entity: Entity, handles: I)
where
    I: IntoIterator<Item = SignalHandle>,
{
    let mut entity = world.entity_mut(entity);
    let mut existing_handles = entity.get_mut::<SignalHandles>().unwrap();
    for handle in handles {
        existing_handles.add(handle);
    }
}

fn offset(i: usize, child_block_populations: &[usize]) -> usize {
    child_block_populations[..i].iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        self as jonmo, JonmoPlugin,
        graph::STALE_SIGNALS,
        signal::{self, SignalExt},
        signal_vec::MutableVec,
    };
    use bevy::prelude::*;
    use bevy_platform::{
        collections::HashSet,
        sync::{Arc, Mutex},
    };

    /// Helper to create a minimal Bevy App with the JonmoPlugin for testing.
    fn create_test_app() -> App {
        cleanup();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        app
    }

    fn cleanup() {
        STALE_SIGNALS.lock().unwrap().clear();
        crate::signal_vec::tests::cleanup(true);
    }

    #[test]
    fn test_on_signal() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // A resource to capture the output from our test systems.
        // The Vec will store tuples of (Entity, i32) received by the system.
        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<(Entity, i32)>>>);

        app.init_resource::<TestOutput>();

        // The system that will be triggered by `on_signal`.
        // It receives the entity and the signal's value, which is Option<i32>.
        // We only record non-None values for simplicity.
        fn capturing_system(In((entity, value_opt)): In<(Entity, Option<i32>)>, output: Res<TestOutput>) {
            if let Some(value) = value_opt {
                output.0.lock().unwrap().push((entity, value));
            }
        }

        // A resource to act as the signal's source.
        #[derive(Resource, Default, Clone, PartialEq)]
        struct SignalSource(Option<i32>);

        app.init_resource::<SignalSource>();

        // The signal reads the resource, extracts the Option, and deduplicates.
        // It will only fire when the i32 value inside SignalSource actually changes.
        let source_signal = signal::from_resource::<SignalSource>()
            .map_in(|source: SignalSource| source.0)
            .dedupe();

        // --- 2. Test Basic Execution & Correct Parameters ---
        let builder1 = jonmo::Builder::new()
            .insert(Name::new("Entity 1"))
            .on_signal(source_signal.clone(), capturing_system);

        let entity1 = builder1.spawn(app.world_mut());

        // The initial `None` value from the resource should not trigger anything.
        app.update();
        assert!(app.world().resource::<TestOutput>().0.lock().unwrap().is_empty());

        // Change the source value to trigger the signal.
        app.world_mut().resource_mut::<SignalSource>().0 = Some(100);
        app.update();

        // Verify the output
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "System should have run once.");
        assert_eq!(
            output_guard[0],
            (entity1, 100),
            "System received incorrect entity or value."
        );
        drop(output_guard);
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // --- 3. Test Multi-Entity Support ---
        let builder2 = jonmo::Builder::new()
            .insert(Name::new("Entity 2"))
            .on_signal(source_signal.clone(), capturing_system);

        let entity2 = builder2.spawn(app.world_mut());

        // Change the source value again, it should trigger systems for both entities.
        app.world_mut().resource_mut::<SignalSource>().0 = Some(200);
        app.update();

        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 2, "Systems for both entities should have run.");
        // Use a HashSet because the order of execution is not guaranteed.
        let received_set: HashSet<(Entity, i32)> = output_guard.iter().cloned().collect();
        let expected_set: HashSet<(Entity, i32)> = [(entity1, 200), (entity2, 200)].into();
        assert_eq!(
            received_set, expected_set,
            "Systems received incorrect parameters for multi-entity test."
        );
        drop(output_guard);
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // --- 4. Test Automatic Cleanup ---
        // Despawn the first entity.
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // Process the despawn command.

        // Change the source value one last time.
        app.world_mut().resource_mut::<SignalSource>().0 = Some(300);
        app.update();

        // Verify that only the system for the second entity ran.
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Only the system for the remaining entity should have run."
        );
        assert_eq!(
            output_guard[0],
            (entity2, 300),
            "The remaining system received incorrect parameters."
        );
        drop(output_guard);

        // Verify that the despawned entity no longer exists.
        assert!(
            app.world().get_entity(entity1).is_err(),
            "Despawned entity should not exist."
        );

        // Despawn the second entity and check that everything is cleaned up.
        app.world_mut().entity_mut(entity2).despawn();
        app.update();
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        app.world_mut().resource_mut::<SignalSource>().0 = Some(400);
        app.update();

        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "No systems should run after all entities are despawned."
        );
    }

    #[test]
    fn test_on_signal_with_component() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The component that will be mutated by the closure.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TestComponent(i32);

        // A resource to act as the signal's source.
        #[derive(Resource, Default, Clone, PartialEq)]
        struct SignalSource(i32);

        app.init_resource::<SignalSource>();

        // The signal reads the resource and deduplicates it, so it only fires on change.
        let source_signal = signal::from_resource::<SignalSource>()
            .map_in(|source: SignalSource| source.0)
            .dedupe();

        // The closure that will be passed to `on_signal_with_component`.
        // It adds the signal's value to the component's value.
        let mutator_closure = |mut component: Mut<TestComponent>, value: i32| {
            component.0 += value;
        };

        // --- 2. Test Basic Functionality ---
        let builder1 = jonmo::Builder::new()
            .insert(TestComponent(10))
            .on_signal_with_component(source_signal.clone(), mutator_closure);

        let entity1 = builder1.spawn(app.world_mut());

        // Trigger the signal by changing the resource.
        app.world_mut().resource_mut::<SignalSource>().0 = 5;
        app.update();

        // Verify the component was mutated correctly.
        let component1 = app.world().get::<TestComponent>(entity1).unwrap();
        assert_eq!(component1.0, 15, "Component should be 10 + 5 = 15.");

        // --- 3. Test Graceful Failure (Component Missing) ---
        let builder_no_comp = jonmo::Builder::new()
            // IMPORTANT: We do *not* insert TestComponent here.
            .on_signal_with_component(source_signal.clone(), mutator_closure);

        let entity_no_comp = builder_no_comp.spawn(app.world_mut());

        // Trigger the signal again. This should not panic.
        app.world_mut().resource_mut::<SignalSource>().0 = 10;
        app.update();

        // Verify that the entity still does not have the component.
        assert!(
            app.world().get::<TestComponent>(entity_no_comp).is_none(),
            "Entity should not have TestComponent added to it."
        );
        // Also verify that the other entity was updated correctly.
        let component1 = app.world().get::<TestComponent>(entity1).unwrap();
        assert_eq!(component1.0, 25, "Existing entity should be updated (15 + 10).");

        // --- 4. Test Multi-Entity Independence ---
        let builder2 = jonmo::Builder::new()
            .insert(TestComponent(100))
            .on_signal_with_component(source_signal.clone(), mutator_closure);

        let entity2 = builder2.spawn(app.world_mut());

        // Trigger the signal again.
        app.world_mut().resource_mut::<SignalSource>().0 = 20;
        app.update();

        // Verify both entities were updated independently.
        let component1 = app.world().get::<TestComponent>(entity1).unwrap();
        assert_eq!(component1.0, 45, "Entity 1 should be 25 + 20 = 45.");

        let component2 = app.world().get::<TestComponent>(entity2).unwrap();
        assert_eq!(component2.0, 120, "Entity 2 should be 100 + 20 = 120.");

        // --- 5. Test Automatic Cleanup ---
        // Despawn the first entity.
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // Process the despawn command.

        // Trigger the signal one last time.
        app.world_mut().resource_mut::<SignalSource>().0 = 30;
        app.update();

        // Verify the despawned entity is gone.
        assert!(
            app.world().get_entity(entity1).is_err(),
            "Despawned entity should not exist."
        );

        // Verify that only the second entity was updated.
        let component2 = app.world().get::<TestComponent>(entity2).unwrap();
        assert_eq!(component2.0, 150, "Only Entity 2 should be updated (120 + 30).");
    }

    #[test]
    fn test_component_signal() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The component that will be reactively added, updated, and removed.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TargetComponent(String);

        // A resource to act as the signal's source.
        // The Option<String> will be mapped to Option<TargetComponent>.
        #[derive(Resource, Default, Clone, PartialEq)]
        struct SignalSource(Option<String>);

        app.init_resource::<SignalSource>();

        // The signal reads the resource and maps its value to the target component.
        // `.dedupe()` is important to ensure we only process actual changes.
        let source_signal = signal::from_resource::<SignalSource>()
            .dedupe()
            .map_in(|source: SignalSource| {
                source.0.map(TargetComponent) // Option<String> -> Option<TargetComponent>
            });

        // --- 2. Test Initial Insertion ---
        let builder = jonmo::Builder::new().component_signal(source_signal.clone());
        let entity1 = builder.spawn(app.world_mut());

        // The signal hasn't run yet, so the component should not be present.
        assert!(app.world().get::<TargetComponent>(entity1).is_none());

        // Set the source to an initial value.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Initial".to_string());
        app.update();

        // Now the component should exist with the initial value.
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Initial".to_string())),
            "Component should be inserted on first Some signal."
        );

        // --- 3. Test Reactive Update ---
        // Change the source value.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Updated".to_string());
        app.update();

        // The component on the entity should be updated.
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Updated".to_string())),
            "Component should be updated on subsequent Some signal."
        );

        // --- 4. Test Reactive Removal ---
        // Set the source to None.
        app.world_mut().resource_mut::<SignalSource>().0 = None;
        app.update();

        // The component should now be removed from the entity.
        assert!(
            app.world().get::<TargetComponent>(entity1).is_none(),
            "Component should be removed on None signal."
        );

        // --- 5. Test Reactive Re-insertion ---
        // Set the source back to a Some value.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Reinserted".to_string());
        app.update();

        // The component should be added back to the entity.
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Reinserted".to_string())),
            "Component should be re-inserted after being removed."
        );

        // --- 6. Test Multi-Entity Independence ---
        let builder2 = jonmo::Builder::new().component_signal(source_signal.clone());
        let entity2 = builder2.spawn(app.world_mut());

        // The new entity should not have the component yet (signal will fire next update).
        assert!(app.world().get::<TargetComponent>(entity2).is_none());

        // Update the source. This will trigger updates for both entities.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Multi".to_string());
        app.update();

        // Both entities should now have the new component value.
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Multi".to_string())),
            "Entity 1 should be updated in multi-entity test."
        );
        assert_eq!(
            app.world().get::<TargetComponent>(entity2),
            Some(&TargetComponent("Multi".to_string())),
            "Entity 2 should be updated in multi-entity test."
        );

        // --- 7. Test Automatic Cleanup ---
        // Despawn the first entity.
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // Process the despawn command.

        // Change the source value one last time.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Post-Despawn".to_string());
        app.update();

        // Verify that only the second entity was updated.
        assert_eq!(
            app.world().get::<TargetComponent>(entity2),
            Some(&TargetComponent("Post-Despawn".to_string())),
            "Only the remaining entity should have been updated."
        );

        // Verify the first entity is gone.
        assert!(
            app.world().get_entity(entity1).is_err(),
            "Despawned entity should not exist."
        );
    }

    #[test]
    fn test_child() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();

            // Marker components for identifying entities
            #[derive(Component, Debug, PartialEq)]
            struct ParentComp;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompA;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompB;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompC;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompD;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompE;

            // Components for testing signal interaction
            #[derive(Component, Clone, Debug, PartialEq)]
            struct SourceComp(i32);
            #[derive(Component, Clone, Debug, PartialEq)]
            struct TargetComp(i32);

            // --- 2. Test: Simple Parent-Child Relationship ---
            let child_builder_simple = jonmo::Builder::new().insert((ChildCompA, Name::new("SimpleChild")));
            let parent_builder_simple = jonmo::Builder::new().insert(ParentComp).child(child_builder_simple);

            let parent_entity_simple = parent_builder_simple.spawn(app.world_mut());
            app.update();

            let mut children_query = app.world_mut().query::<&Children>();
            let parent_children = children_query.get(app.world(), parent_entity_simple).unwrap();
            assert_eq!(parent_children.len(), 1, "Parent should have exactly one child.");
            let child_entity_simple = parent_children[0];

            // Verify child's components
            let child_entity_ref = app.world().entity(child_entity_simple);
            assert!(child_entity_ref.contains::<ChildCompA>());
            assert_eq!(child_entity_ref.get::<Name>().unwrap().as_str(), "SimpleChild");

            // Verify parent relationship from child's perspective
            let mut parent_query = app.world_mut().query::<&bevy::prelude::ChildOf>();
            let childs_parent = parent_query.get(app.world(), child_entity_simple).unwrap();
            assert_eq!(childs_parent.parent(), parent_entity_simple);

            // Cleanup for next test
            app.world_mut().entity_mut(parent_entity_simple).despawn();
            app.update();

            // --- 3. Test: Multiple Chained Children & Correct Order ---
            let parent_builder_chained = jonmo::Builder::new()
                .insert(ParentComp)
                .child(jonmo::Builder::new().insert(ChildCompA))
                .child(jonmo::Builder::new().insert(ChildCompB));

            let parent_entity_chained = parent_builder_chained.spawn(app.world_mut());
            app.update();

            let children = app.world().get::<Children>(parent_entity_chained).unwrap();
            assert_eq!(children.len(), 2, "Parent should have two children.");

            let child_a_entity = children[0];
            let child_b_entity = children[1];

            assert!(
                app.world().entity(child_a_entity).contains::<ChildCompA>(),
                "First child should be A"
            );
            assert!(
                app.world().entity(child_b_entity).contains::<ChildCompB>(),
                "Second child should be B"
            );

            // Cleanup
            app.world_mut().entity_mut(parent_entity_chained).despawn();
            app.update();

            // --- 4. Test: Nested Hierarchy ---
            let grandchild_builder = jonmo::Builder::new().insert(Name::new("Grandchild"));
            let child_builder_nested = jonmo::Builder::new()
                .insert(Name::new("Child"))
                .child(grandchild_builder);
            let grandparent_builder = jonmo::Builder::new()
                .insert(Name::new("Grandparent"))
                .child(child_builder_nested);

            let grandparent_entity = grandparent_builder.spawn(app.world_mut());
            app.update();

            let gp_children = app.world().get::<Children>(grandparent_entity).unwrap();
            assert_eq!(gp_children.len(), 1);
            let child_entity_nested = gp_children[0];
            assert_eq!(app.world().get::<Name>(child_entity_nested).unwrap().as_str(), "Child");

            let child_children = app.world().get::<Children>(child_entity_nested).unwrap();
            assert_eq!(child_children.len(), 1);
            let grandchild_entity = child_children[0];
            assert_eq!(
                app.world().get::<Name>(grandchild_entity).unwrap().as_str(),
                "Grandchild"
            );

            // Cleanup
            app.world_mut().entity_mut(grandparent_entity).despawn();
            app.update();

            // --- 5. Test: Mixed Child Types for Correct Ordering ---
            #[derive(Resource, Default, Deref, DerefMut, Clone)]
            struct SignalTrigger(bool);
            app.init_resource::<SignalTrigger>();

            let child_a = jonmo::Builder::new().insert(ChildCompA);
            let child_b = jonmo::Builder::new().insert(ChildCompB);
            let child_c = jonmo::Builder::new().insert(ChildCompC);
            let child_d = jonmo::Builder::new().insert(ChildCompD);
            // Use a factory function instead of cloning
            fn make_child_e() -> jonmo::Builder {
                jonmo::Builder::new().insert(ChildCompE)
            }

            let parent_builder_mixed = jonmo::Builder::new()
                .insert(ParentComp)
                .child(child_a) // Block 0, offset 0
                .children([child_b, child_c]) // Block 1, offset 1
                .child(child_d) // Block 2, offset 3
                .child_signal(
                    // Block 3, offset 4
                    signal::from_resource::<SignalTrigger>().map_in::<Option<Builder>, _, _>(
                        move |trigger: SignalTrigger| if trigger.0 { Some(make_child_e()) } else { None },
                    ),
                );

            let parent_entity_mixed = parent_builder_mixed.spawn(app.world_mut());

            // Initially, signal is false, so child E should not exist.
            app.update();
            let children_mixed = app.world().get::<Children>(parent_entity_mixed).unwrap();
            assert_eq!(children_mixed.len(), 4, "Should have 4 children initially.");
            assert!(app.world().entity(children_mixed[0]).contains::<ChildCompA>());
            assert!(app.world().entity(children_mixed[1]).contains::<ChildCompB>());
            assert!(app.world().entity(children_mixed[2]).contains::<ChildCompC>());
            assert!(app.world().entity(children_mixed[3]).contains::<ChildCompD>());

            // Trigger the signal to add child E.
            app.world_mut().resource_mut::<SignalTrigger>().0 = true;
            app.update();

            let children_mixed_after = app.world().get::<Children>(parent_entity_mixed).unwrap();
            assert_eq!(
                children_mixed_after.len(),
                5,
                "Should have 5 children after signal trigger."
            );
            // Verify the full order
            assert!(app.world().entity(children_mixed_after[0]).contains::<ChildCompA>());
            assert!(app.world().entity(children_mixed_after[1]).contains::<ChildCompB>());
            assert!(app.world().entity(children_mixed_after[2]).contains::<ChildCompC>());
            assert!(app.world().entity(children_mixed_after[3]).contains::<ChildCompD>());
            assert!(
                app.world().entity(children_mixed_after[4]).contains::<ChildCompE>(),
                "Child E should be last."
            );

            // Cleanup
            app.world_mut().entity_mut(parent_entity_mixed).despawn();
            app.update();

            // --- 6. Test: Child's Signals Function Correctly ---
            // Use LazyEntity + signal::Builder pattern instead of removed component_signal_from_parent
            let child_entity = LazyEntity::new();
            let child_builder_with_signal = jonmo::Builder::new().lazy_entity(child_entity.clone()).on_signal(
                signal::from_parent(child_entity.clone())
                    .component::<SourceComp>()
                    .map_in(|source: SourceComp| Some(TargetComp(source.0 * 2))),
                |In((entity, comp_opt)): In<(Entity, Option<TargetComp>)>, world: &mut World| {
                    if let Ok(mut e) = world.get_entity_mut(entity) {
                        if let Some(comp) = comp_opt {
                            e.insert(comp);
                        } else {
                            e.remove::<TargetComp>();
                        }
                    }
                },
            );

            let parent_builder_with_signal = jonmo::Builder::new()
                .insert(SourceComp(10))
                .child(child_builder_with_signal);

            let parent_entity_signal = parent_builder_with_signal.spawn(app.world_mut());
            app.update();

            let child_entity_signal = app.world().get::<Children>(parent_entity_signal).unwrap()[0];
            let child_target_comp = app.world().get::<TargetComp>(child_entity_signal);

            assert_eq!(
                child_target_comp,
                Some(&TargetComp(20)),
                "Child's signal did not correctly read parent component and update itself."
            );

            // Test reactivity: change parent component, check child.
            app.world_mut().get_mut::<SourceComp>(parent_entity_signal).unwrap().0 = 50;
            app.update();
            let child_target_comp_updated = app.world().get::<TargetComp>(child_entity_signal);
            assert_eq!(
                child_target_comp_updated,
                Some(&TargetComp(100)),
                "Child's signal did not react to parent component change."
            );

            // Cleanup
            app.world_mut().entity_mut(parent_entity_signal).despawn();
            app.update();
        }

        cleanup()
    }

    // Marker components for easy identification in the child_signal test
    #[derive(Component, Debug, PartialEq)]
    struct ParentComp;
    #[derive(Component, Debug, PartialEq)]
    struct StaticChildBefore;
    #[derive(Component, Debug, PartialEq)]
    struct StaticChildAfter;
    #[derive(Component, Debug, PartialEq)]
    struct ReactiveChild(u32); // To distinguish different reactive children

    // The resource that will drive our test signal
    #[derive(Resource, Default, Clone, PartialEq)]
    struct SignalSource(Option<u32>);

    /// Helper to get a Vec of strings representing the types of children in order.
    fn get_child_types(world: &mut World, parent: Entity) -> Vec<String> {
        let mut children_query = world.query_filtered::<&Children, With<ParentComp>>();
        let Ok(children) = children_query.get(world, parent) else {
            return vec![];
        };

        children
            .iter()
            .map(|child_entity| {
                let entity_ref = world.entity(child_entity);
                if entity_ref.contains::<StaticChildBefore>() {
                    "Before".to_string()
                } else if entity_ref.contains::<StaticChildAfter>() {
                    "After".to_string()
                } else if let Some(rc) = entity_ref.get::<ReactiveChild>() {
                    format!("Reactive({})", rc.0)
                } else {
                    "Unknown".to_string()
                }
            })
            .collect()
    }

    #[test]
    fn test_child_signal() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalSource>();

            // A factory function to create reactive child builders based on a number
            let reactive_child_factory = |id: u32| jonmo::Builder::new().insert(ReactiveChild(id));

            // The signal that maps the resource to an Option<jonmo::Builder>
            let source_signal = signal::from_resource::<SignalSource>()
                .dedupe()
                .map_in(move |source: SignalSource| source.0.map(reactive_child_factory));

            // --- 2. Build the Parent Entity ---
            // This builder has static children sandwiching the reactive one to test ordering.
            let parent_builder = jonmo::Builder::new()
                .insert(ParentComp)
                .child(jonmo::Builder::new().insert(StaticChildBefore)) // Child in block 0
                .child_signal(source_signal) // Child in block 1
                .child(jonmo::Builder::new().insert(StaticChildAfter)); // Child in block 2

            let parent_entity = parent_builder.spawn(app.world_mut());

            // --- 3. Run Test Cases ---

            // Case A: Initial state (Source is None).
            // Should only have the static children.
            app.update();
            assert_eq!(
                get_child_types(app.world_mut(), parent_entity),
                vec!["Before", "After"],
                "Initial state with None should only have static children"
            );
            assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 2);

            // Case B: Transition None -> Some(1)
            app.world_mut().resource_mut::<SignalSource>().0 = Some(1);
            app.update();
            assert_eq!(
                get_child_types(app.world_mut(), parent_entity),
                vec!["Before", "Reactive(1)", "After"],
                "Transition None->Some failed to create and order child correctly"
            );
            assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 3);

            // Case C: No change (still Some(1)).
            // Because of .dedupe(), the signal doesn't fire, nothing should change.
            app.update();
            assert_eq!(
                get_child_types(app.world_mut(), parent_entity),
                vec!["Before", "Reactive(1)", "After"],
                "No-op update should not change children"
            );
            assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 3);

            // Case D: Transition Some(1) -> Some(2)
            // The old child should be despawned, and the new one spawned in its place.
            let old_child_entity = app.world().get::<Children>(parent_entity).unwrap()[1];
            app.world_mut().resource_mut::<SignalSource>().0 = Some(2);
            app.update();
            assert_eq!(
                get_child_types(app.world_mut(), parent_entity),
                vec!["Before", "Reactive(2)", "After"],
                "Transition Some->Some failed to replace and order child correctly"
            );
            assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 3);
            assert!(
                app.world().get_entity(old_child_entity).is_err(),
                "Old reactive child was not despawned on replacement"
            );

            // Case E: Transition Some(2) -> None
            let old_child_entity = app.world().get::<Children>(parent_entity).unwrap()[1];
            app.world_mut().resource_mut::<SignalSource>().0 = None;
            app.update();
            assert_eq!(
                get_child_types(app.world_mut(), parent_entity),
                vec!["Before", "After"],
                "Transition Some->None failed to remove child"
            );
            assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 2);
            assert!(
                app.world().get_entity(old_child_entity).is_err(),
                "Reactive child was not despawned on transition to None"
            );

            // Case F: Transition back to Some(3)
            app.world_mut().resource_mut::<SignalSource>().0 = Some(3);
            app.update();
            assert_eq!(
                get_child_types(app.world_mut(), parent_entity),
                vec!["Before", "Reactive(3)", "After"],
                "Transition back to Some failed"
            );
            assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 3);

            // --- 4. Test Cleanup ---
            let child_to_despawn = app.world().get::<Children>(parent_entity).unwrap()[1];
            app.world_mut().entity_mut(parent_entity).despawn();
            app.update(); // Process despawn command.

            assert!(
                app.world().get_entity(parent_entity).is_err(),
                "Parent should be despawned."
            );
            assert!(
                app.world().get_entity(child_to_despawn).is_err(),
                "Reactive child should be despawned with parent."
            );

            // Trigger the signal again. This should not panic or create any entities,
            // as the signal system tied to the parent should have been cleaned up.
            app.world_mut().resource_mut::<SignalSource>().0 = Some(4);
            app.update();

            let mut reactive_children_query = app.world_mut().query::<&ReactiveChild>();
            assert_eq!(
                reactive_children_query.iter(app.world()).count(),
                0,
                "No reactive children should exist after parent is despawned."
            );
        }

        cleanup()
    }

    #[test]
    fn test_children() {
        {
            // --- 1. SETUP ---
            // This comprehensive test validates multiple aspects of the `children` method.
            let mut app = create_test_app();

            // Marker components for identifying entities in assertions.
            #[derive(Component, Debug, PartialEq)]
            struct ParentComp;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompA;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompB;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompC;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompD;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompE;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompF;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompG;
            #[derive(Component, Debug, PartialEq)]
            struct ChildCompH;
            #[derive(Component, Debug, PartialEq)]
            struct GrandchildComp;

            // Components for testing signal interaction within children.
            #[derive(Component, Clone, Debug, PartialEq)]
            struct SourceComp(i32);
            #[derive(Component, Clone, Debug, PartialEq)]
            struct TargetComp(i32);

            // --- 2. TEST CASE: Basic Functionality & Correct Order ---
            // Verifies that a simple list of builders is spawned correctly as children and in the
            // specified order.
            {
                let child_builders = vec![
                    jonmo::Builder::new().insert(ChildCompA),
                    jonmo::Builder::new().insert(ChildCompB),
                    jonmo::Builder::new().insert(ChildCompC),
                ];
                let parent_builder = jonmo::Builder::new().insert(ParentComp).children(child_builders);

                let parent_entity = parent_builder.spawn(app.world_mut());
                app.update();

                let children = app
                    .world()
                    .get::<Children>(parent_entity)
                    .expect("Parent should have Children component.")
                    .iter()
                    .collect::<Vec<_>>();
                assert_eq!(children.len(), 3, "Parent should have exactly 3 children.");

                // Verify order and components
                assert!(
                    app.world().entity(children[0]).contains::<ChildCompA>(),
                    "Child 0 should be A"
                );
                assert!(
                    app.world().entity(children[1]).contains::<ChildCompB>(),
                    "Child 1 should be B"
                );
                assert!(
                    app.world().entity(children[2]).contains::<ChildCompC>(),
                    "Child 2 should be C"
                );

                // Verify parent relationship from each child's perspective
                let mut parent_rel_query = app.world_mut().query::<&bevy::prelude::ChildOf>();
                for child_entity in children.into_iter() {
                    assert_eq!(
                        parent_rel_query.get(app.world(), child_entity).unwrap().parent(),
                        parent_entity
                    );
                }

                app.world_mut().entity_mut(parent_entity).despawn();
                app.update();
            }

            // --- 3. TEST CASE: Empty Iterator ---
            // Verifies that providing an empty iterator results in no children being added.
            {
                // The type hint is needed for an empty vec.
                let parent_builder_empty = jonmo::Builder::new()
                    .insert(ParentComp)
                    .children(vec![] as Vec<Builder>);
                let parent_entity_empty = parent_builder_empty.spawn(app.world_mut());
                app.update();
                assert!(
                    app.world().get::<Children>(parent_entity_empty).is_none(),
                    "Parent with empty children iterator should not have a Children component."
                );

                app.world_mut().entity_mut(parent_entity_empty).despawn();
                app.update();
            }

            // --- 4. TEST CASE: Complex Children and Cleanup ---
            // Verifies that children can have their own complex logic (signals, hierarchy) and that
            // they are properly cleaned up when the parent is despawned.
            {
                // Use LazyEntity + signal::Builder pattern instead of removed component_signal_from_parent
                let complex_child_entity = LazyEntity::new();
                let complex_child_builder = jonmo::Builder::new()
                    .lazy_entity(complex_child_entity.clone())
                    .insert(ChildCompD)
                    .child(jonmo::Builder::new().insert(GrandchildComp)) // Nested child
                    .on_signal(
                        signal::from_parent(complex_child_entity.clone())
                            .component::<SourceComp>()
                            .map_in(|source: SourceComp| Some(TargetComp(source.0 * 10))),
                        |In((entity, comp_opt)): In<(Entity, Option<TargetComp>)>, world: &mut World| {
                            if let Ok(mut e) = world.get_entity_mut(entity) {
                                if let Some(comp) = comp_opt {
                                    e.insert(comp);
                                } else {
                                    e.remove::<TargetComp>();
                                }
                            }
                        },
                    );

                let parent_builder_complex = jonmo::Builder::new()
                    .insert((ParentComp, SourceComp(5)))
                    .children(vec![complex_child_builder]);

                let parent_entity_complex = parent_builder_complex.spawn(app.world_mut());
                app.update();

                let complex_children = app.world().get::<Children>(parent_entity_complex).unwrap();
                let complex_child_entity_id = complex_children[0];

                // Verify signal ran correctly
                assert_eq!(
                    app.world().get::<TargetComp>(complex_child_entity_id),
                    Some(&TargetComp(50))
                );

                // Verify nested hierarchy
                let grandchild_entity = app.world().get::<Children>(complex_child_entity_id).unwrap()[0];
                assert!(app.world().entity(grandchild_entity).contains::<GrandchildComp>());

                // Test reactivity
                app.world_mut().get_mut::<SourceComp>(parent_entity_complex).unwrap().0 = 7;
                app.update();
                assert_eq!(
                    app.world().get::<TargetComp>(complex_child_entity_id),
                    Some(&TargetComp(70)),
                    "Signal did not react to parent's component change."
                );

                // Test cleanup
                app.world_mut().entity_mut(parent_entity_complex).despawn();
                app.update();

                assert!(
                    app.world().get_entity(complex_child_entity_id).is_err(),
                    "Complex child should be despawned with parent."
                );
                assert!(
                    app.world().get_entity(grandchild_entity).is_err(),
                    "Grandchild should be despawned with parent."
                );
            }

            // --- 5. TEST CASE: Mixed Ordering with Other Child Methods ---
            // This is a critical test to ensure the internal offset calculation is correct when
            // `children` is mixed with `child`, `child_signal`, etc.
            {
                // Helper to verify the exact order of children by their components.
                fn get_ordered_child_markers(world: &World, parent: Entity) -> Vec<&'static str> {
                    let Some(children) = world.get::<Children>(parent) else {
                        return vec![];
                    };
                    children
                        .iter()
                        .map(|child_entity| {
                            let e = world.entity(child_entity);
                            if e.contains::<ChildCompA>() {
                                "A"
                            } else if e.contains::<ChildCompB>() {
                                "B"
                            } else if e.contains::<ChildCompC>() {
                                "C"
                            } else if e.contains::<ChildCompD>() {
                                "D"
                            } else if e.contains::<ChildCompE>() {
                                "E"
                            } else if e.contains::<ChildCompF>() {
                                "F"
                            } else if e.contains::<ChildCompG>() {
                                "G"
                            } else if e.contains::<ChildCompH>() {
                                "H"
                            } else {
                                "Unknown"
                            }
                        })
                        .collect()
                }

                #[derive(Resource, Default, Deref, DerefMut, Clone)]
                struct SignalTrigger(bool);
                app.init_resource::<SignalTrigger>();

                let parent_builder = jonmo::Builder::new()
                    .insert(ParentComp)
                    .child(jonmo::Builder::new().insert(ChildCompA)) // Block 0, size 1
                    .children([
                        // Block 1, size 2
                        jonmo::Builder::new().insert(ChildCompB),
                        jonmo::Builder::new().insert(ChildCompC),
                    ])
                    .child_signal(
                        // Block 2, size 0 -> 1
                        signal::from_resource::<SignalTrigger>().map_in(|trigger: SignalTrigger| {
                            if trigger.0 {
                                Some(jonmo::Builder::new().insert(ChildCompD))
                            } else {
                                None
                            }
                        }),
                    )
                    .children(vec![
                        // Block 3, size 3
                        jonmo::Builder::new().insert(ChildCompE),
                        jonmo::Builder::new().insert(ChildCompF),
                        jonmo::Builder::new().insert(ChildCompG),
                    ])
                    .child(jonmo::Builder::new().insert(ChildCompH)); // Block 4, size 1

                let parent_entity = parent_builder.spawn(app.world_mut());

                // Test Initial Order (Signal is false)
                app.update();
                assert_eq!(
                    get_ordered_child_markers(app.world(), parent_entity),
                    vec!["A", "B", "C", "E", "F", "G", "H"],
                    "Initial mixed child order is incorrect"
                );
                assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 7);

                // Test Order After Signal Trigger
                app.world_mut().resource_mut::<SignalTrigger>().0 = true;
                app.update();
                assert_eq!(
                    get_ordered_child_markers(app.world(), parent_entity),
                    vec!["A", "B", "C", "D", "E", "F", "G", "H"],
                    "Mixed child order after signal trigger is incorrect"
                );
                assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 8);

                // Test Order After Signal Un-triggers
                app.world_mut().resource_mut::<SignalTrigger>().0 = false;
                app.update();
                assert_eq!(
                    get_ordered_child_markers(app.world(), parent_entity),
                    vec!["A", "B", "C", "E", "F", "G", "H"],
                    "Mixed child order after signal un-triggers is incorrect"
                );
                assert_eq!(app.world().get::<Children>(parent_entity).unwrap().len(), 7);
            }
        }

        cleanup()
    }

    /// Helper to get a Vec of all children's entity IDs.
    fn get_children_entities(world: &mut World, parent: Entity) -> Vec<Entity> {
        world
            .get::<Children>(parent)
            .map(|children| children.iter().collect())
            .unwrap_or_default()
    }

    /// Helper to get a textual representation of all children for ordering assertions.
    fn get_all_child_types(world: &mut World, parent: Entity) -> Vec<String> {
        let Ok(children) = world.query::<&Children>().get(world, parent) else {
            return vec![];
        };

        children
            .iter()
            .map(|child_entity| {
                let entity_ref = world.entity(child_entity);
                if entity_ref.contains::<StaticChildBefore>() {
                    "StaticBefore".to_string()
                } else if entity_ref.contains::<StaticChildAfter>() {
                    "StaticAfter".to_string()
                } else if let Some(rc) = entity_ref.get::<ReactiveChild>() {
                    format!("Reactive({})", rc.0)
                } else {
                    "Unknown".to_string()
                }
            })
            .collect()
    }

    #[test]
    fn test_children_signal_vec() {
        {
            // --- 1. SETUP ---
            let mut app = create_test_app();
            let source_vec = MutableVec::builder().values([10u32, 20u32]).spawn(app.world_mut());

            // A factory function to create a simple jonmo::Builder for a reactive child.
            let child_builder_factory = |id: u32| jonmo::Builder::new().insert(ReactiveChild(id));

            // The SignalVec that will drive the children.
            let children_signal = source_vec.signal_vec().map_in(child_builder_factory);

            // The parent builder, with static children sandwiching the reactive ones to test ordering.
            let parent_builder = jonmo::Builder::new()
                .insert(ParentComp)
                .child(jonmo::Builder::new().insert(StaticChildBefore))
                .children_signal_vec(children_signal)
                .child(jonmo::Builder::new().insert(StaticChildAfter));

            let parent_entity = parent_builder.spawn(app.world_mut());

            // --- 2. INITIAL STATE ---
            // The first update runs the on_spawn closures and processes the initial `Replace` diff.
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec!["StaticBefore", "Reactive(10)", "Reactive(20)", "StaticAfter"],
                "Initial child order and content is incorrect."
            );

            // --- 3. TEST `PUSH` ---
            source_vec.write(app.world_mut()).push(30);
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec![
                    "StaticBefore",
                    "Reactive(10)",
                    "Reactive(20)",
                    "Reactive(30)",
                    "StaticAfter"
                ],
                "State after Push is incorrect."
            );

            // --- 4. TEST `INSERT_AT` ---
            source_vec.write(app.world_mut()).insert(1, 15); // Insert 15 between 10 and 20
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec![
                    "StaticBefore",
                    "Reactive(10)",
                    "Reactive(15)",
                    "Reactive(20)",
                    "Reactive(30)",
                    "StaticAfter"
                ],
                "State after InsertAt is incorrect."
            );

            // --- 5. TEST `UPDATE_AT` ---
            let old_child_entities = get_children_entities(app.world_mut(), parent_entity);
            let entity_to_be_replaced = old_child_entities[3]; // The "Reactive(20)" entity
            source_vec.write(app.world_mut()).set(2, 25); // Update 20 to 25
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec![
                    "StaticBefore",
                    "Reactive(10)",
                    "Reactive(15)",
                    "Reactive(25)",
                    "Reactive(30)",
                    "StaticAfter"
                ],
                "State after UpdateAt is incorrect."
            );
            assert!(
                app.world().get_entity(entity_to_be_replaced).is_err(),
                "Old child entity should be despawned after UpdateAt."
            );

            // --- 6. TEST `REMOVE_AT` ---
            let old_child_entities = get_children_entities(app.world_mut(), parent_entity);
            let entity_to_be_removed = old_child_entities[2]; // The "Reactive(15)" entity
            source_vec.write(app.world_mut()).remove(1); // Remove 15
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec![
                    "StaticBefore",
                    "Reactive(10)",
                    "Reactive(25)",
                    "Reactive(30)",
                    "StaticAfter"
                ],
                "State after RemoveAt is incorrect."
            );
            assert!(
                app.world().get_entity(entity_to_be_removed).is_err(),
                "Child entity should be despawned after RemoveAt."
            );

            // --- 7. TEST `MOVE` ---
            source_vec.write(app.world_mut()).move_item(2, 0); // Move 30 (now at index 2) to the front
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec![
                    "StaticBefore",
                    "Reactive(30)",
                    "Reactive(10)",
                    "Reactive(25)",
                    "StaticAfter"
                ],
                "State after Move is incorrect."
            );

            // --- 8. TEST `POP` ---
            let old_child_entities = get_children_entities(app.world_mut(), parent_entity);
            let entity_to_be_popped = old_child_entities[3]; // The "Reactive(25)" entity
            source_vec.write(app.world_mut()).pop(); // Removes 25
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec!["StaticBefore", "Reactive(30)", "Reactive(10)", "StaticAfter"],
                "State after Pop is incorrect."
            );
            assert!(
                app.world().get_entity(entity_to_be_popped).is_err(),
                "Child entity should be despawned after Pop."
            );

            // --- 9. TEST `CLEAR` ---
            let reactive_children_before_clear = get_children_entities(app.world_mut(), parent_entity)
                .into_iter()
                .filter(|e| app.world().get::<ReactiveChild>(*e).is_some())
                .collect::<Vec<_>>();
            source_vec.write(app.world_mut()).clear();
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec!["StaticBefore", "StaticAfter"],
                "State after Clear is incorrect."
            );
            for child in reactive_children_before_clear {
                assert!(
                    app.world().get_entity(child).is_err(),
                    "All reactive children should be despawned after Clear."
                );
            }

            // --- 10. TEST `REPLACE` (after Clear) ---
            source_vec.write(app.world_mut()).replace(vec![100, 200]);
            app.update();
            assert_eq!(
                get_all_child_types(app.world_mut(), parent_entity),
                vec!["StaticBefore", "Reactive(100)", "Reactive(200)", "StaticAfter"],
                "State after Replace is incorrect."
            );

            // --- 11. TEST PARENT DESPAWN ---
            let all_children_before_despawn = get_children_entities(app.world_mut(), parent_entity);
            app.world_mut().entity_mut(parent_entity).despawn();
            app.update(); // Process despawn commands

            assert!(
                app.world().get_entity(parent_entity).is_err(),
                "Parent entity should be despawned."
            );
            for child in all_children_before_despawn {
                assert!(
                    app.world().get_entity(child).is_err(),
                    "All children (static and reactive) should be despawned with parent."
                );
            }

            // Verify signal cleanup by flushing one more change. This should not panic.
            source_vec.write(app.world_mut()).push(999);
            app.update();
            // The test passes if the above update doesn't panic.
        }

        cleanup()
    }

    #[test]
    fn test_on_despawn() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // A resource to track whether the on_despawn callback was executed.
        #[derive(Resource, Default, Clone)]
        struct DespawnTracker(Arc<Mutex<Vec<(Entity, String)>>>);

        app.init_resource::<DespawnTracker>();

        // --- 2. Test Basic Callback Execution ---
        let tracker = app.world().resource::<DespawnTracker>().clone();
        let builder1 = jonmo::Builder::new().on_despawn({
            let tracker = tracker.clone();
            move |_world, entity| {
                tracker.0.lock().unwrap().push((entity, "callback1".to_string()));
            }
        });

        let entity1 = builder1.spawn(app.world_mut());
        app.update();

        // Callback should not have been called yet.
        assert!(
            tracker.0.lock().unwrap().is_empty(),
            "on_despawn callback should not run before despawn."
        );

        // Despawn the entity.
        app.world_mut().entity_mut(entity1).despawn();
        app.update();

        // Callback should have been called with the correct entity.
        let tracker_guard = tracker.0.lock().unwrap();
        assert_eq!(tracker_guard.len(), 1, "Callback should have been called exactly once.");
        assert_eq!(tracker_guard[0], (entity1, "callback1".to_string()));
        drop(tracker_guard);
        tracker.0.lock().unwrap().clear();

        // --- 3. Test Multiple Callbacks on Same Entity ---
        let builder2 = jonmo::Builder::new()
            .on_despawn({
                let tracker = tracker.clone();
                move |_world, entity| {
                    tracker.0.lock().unwrap().push((entity, "first".to_string()));
                }
            })
            .on_despawn({
                let tracker = tracker.clone();
                move |_world, entity| {
                    tracker.0.lock().unwrap().push((entity, "second".to_string()));
                }
            })
            .on_despawn({
                let tracker = tracker.clone();
                move |_world, entity| {
                    tracker.0.lock().unwrap().push((entity, "third".to_string()));
                }
            });

        let entity2 = builder2.spawn(app.world_mut());
        app.update();

        // Despawn the entity.
        app.world_mut().entity_mut(entity2).despawn();
        app.update();

        // All three callbacks should have been called.
        let tracker_guard = tracker.0.lock().unwrap();
        assert_eq!(tracker_guard.len(), 3, "All three callbacks should have been called.");
        // Callbacks are stored in a Vec and called in order.
        assert_eq!(tracker_guard[0], (entity2, "first".to_string()));
        assert_eq!(tracker_guard[1], (entity2, "second".to_string()));
        assert_eq!(tracker_guard[2], (entity2, "third".to_string()));
        drop(tracker_guard);
        tracker.0.lock().unwrap().clear();

        // --- 4. Test Multi-Entity Independence ---
        let builder3 = jonmo::Builder::new().on_despawn({
            let tracker = tracker.clone();
            move |_world, entity| {
                tracker.0.lock().unwrap().push((entity, "entity3".to_string()));
            }
        });
        let entity3 = builder3.spawn(app.world_mut());

        let builder4 = jonmo::Builder::new().on_despawn({
            let tracker = tracker.clone();
            move |_world, entity| {
                tracker.0.lock().unwrap().push((entity, "entity4".to_string()));
            }
        });
        let entity4 = builder4.spawn(app.world_mut());
        app.update();

        // Despawn only entity3.
        app.world_mut().entity_mut(entity3).despawn();
        app.update();

        // Only entity3's callback should have been called.
        let tracker_guard = tracker.0.lock().unwrap();
        assert_eq!(tracker_guard.len(), 1, "Only one callback should have been called.");
        assert_eq!(tracker_guard[0], (entity3, "entity3".to_string()));
        drop(tracker_guard);
        tracker.0.lock().unwrap().clear();

        // Now despawn entity4.
        app.world_mut().entity_mut(entity4).despawn();
        app.update();

        // entity4's callback should have been called.
        let tracker_guard = tracker.0.lock().unwrap();
        assert_eq!(tracker_guard.len(), 1, "entity4's callback should have been called.");
        assert_eq!(tracker_guard[0], (entity4, "entity4".to_string()));
        drop(tracker_guard);
        tracker.0.lock().unwrap().clear();

        // --- 5. Test DeferredWorld Access ---
        // Verify that the callback can actually interact with the world.
        #[derive(Resource, Default)]
        struct DespawnCounter(u32);

        app.init_resource::<DespawnCounter>();

        let builder5 = jonmo::Builder::new().on_despawn(|world, _entity| {
            world.resource_mut::<DespawnCounter>().0 += 1;
        });

        let entity5 = builder5.spawn(app.world_mut());
        app.update();

        assert_eq!(
            app.world().resource::<DespawnCounter>().0,
            0,
            "Counter should be 0 before despawn."
        );

        app.world_mut().entity_mut(entity5).despawn();
        app.update();

        assert_eq!(
            app.world().resource::<DespawnCounter>().0,
            1,
            "Counter should be incremented by on_despawn callback."
        );

        // Spawn and despawn another entity to verify counter increments again.
        let builder6 = jonmo::Builder::new().on_despawn(|world, _entity| {
            world.resource_mut::<DespawnCounter>().0 += 10;
        });
        let entity6 = builder6.spawn(app.world_mut());
        app.update();
        app.world_mut().entity_mut(entity6).despawn();
        app.update();

        assert_eq!(
            app.world().resource::<DespawnCounter>().0,
            11,
            "Counter should be 1 + 10 = 11 after second despawn."
        );

        // --- 6. Test Combination with Other Builder Methods ---
        // Ensure on_despawn works correctly with other builder methods like insert and child.
        #[derive(Component)]
        struct TestMarker;

        let child_despawn_tracker = Arc::new(Mutex::new(Vec::new()));
        let parent_despawn_tracker = Arc::new(Mutex::new(Vec::new()));

        let child_tracker = child_despawn_tracker.clone();
        let parent_tracker = parent_despawn_tracker.clone();

        let builder_parent = jonmo::Builder::new()
            .insert(TestMarker)
            .on_despawn(move |_world, entity| {
                parent_tracker.lock().unwrap().push(entity);
            })
            .child(jonmo::Builder::new().on_despawn(move |_world, entity| {
                child_tracker.lock().unwrap().push(entity);
            }));

        let parent_entity = builder_parent.spawn(app.world_mut());
        app.update();

        // Get the child entity.
        let children: Vec<Entity> = app
            .world()
            .get::<Children>(parent_entity)
            .map(|c| c.iter().collect())
            .unwrap_or_default();
        assert_eq!(children.len(), 1, "Parent should have one child.");
        let child_entity = children[0];

        // Despawn the parent (which should also despawn the child due to Bevy's hierarchy).
        app.world_mut().entity_mut(parent_entity).despawn();
        app.update();

        // Both callbacks should have been called.
        assert_eq!(
            parent_despawn_tracker.lock().unwrap().len(),
            1,
            "Parent's on_despawn should have been called."
        );
        assert_eq!(
            parent_despawn_tracker.lock().unwrap()[0],
            parent_entity,
            "Parent callback should receive parent entity."
        );
        assert_eq!(
            child_despawn_tracker.lock().unwrap().len(),
            1,
            "Child's on_despawn should have been called when parent is despawned."
        );
        assert_eq!(
            child_despawn_tracker.lock().unwrap()[0],
            child_entity,
            "Child callback should receive child entity."
        );
    }
}
