//! Declarative entity builder using jonmo signals.

use super::{
    prelude::*,
    signal::{SignalBuilder, Signal, SignalExt, Source, Map},
    signal_vec::{VecDiff},
    tree::SignalHandle,
    utils::{LazyEntity, SSs},
};
use bevy_ecs::component::HookContext;
use bevy_ecs::{prelude::*, world::DeferredWorld};
use bevy_reflect::{FromReflect, GetTypeRegistration, Reflect, Typed};
use std::sync::{Arc, Mutex};

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

/// Component storing handles to reactive systems attached to an entity.
/// These handles are used to clean up the systems when the entity is despawned.
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
    /// Add a signal handle to the component.
    pub fn add(&mut self, handle: SignalHandle) {
        self.0.push(handle);
    }
}

fn add_handle(world: &mut World, entity: Entity, handle: SignalHandle) {
    if let Ok(mut entity) = world.get_entity_mut(entity) {
        if let Some(mut handlers) = entity.get_mut::<SignalHandles>() {
            handlers.add(handle);
        }
    }
}

/// A thin facade over a Bevy [`Entity`] enabling the ergonomic registration of reactive systems and
/// children using a declarative builder pattern. Inspired by Dominator's DomBuilder and Haalka's NodeBuilder.
#[derive(Default, Clone, Reflect)] // Removed Clone
#[reflect(opaque)]
pub struct JonmoBuilder {
    #[allow(clippy::type_complexity)]
    on_spawns: Arc<Mutex<Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>>>, // Changed type
    child_block_populations: Arc<Mutex<Vec<usize>>>, // Keep this for child logic for now
}

impl<T: Bundle> From<T> for JonmoBuilder {
    fn from(bundle: T) -> Self {
        Self::new().insert(bundle)
    }
}

impl JonmoBuilder {
    /// Create a new, empty [`NodeBuilder`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a function with mutable access to the [`World`] and this node's [`Entity`]
    /// immediately after the entity is spawned, but before signal systems are fully registered.
    pub fn on_spawn(
        self, // Remove mut
        on_spawn: impl FnOnce(&mut World, Entity) + SSs,
    ) -> Self {
        self.on_spawns.lock().unwrap().push(Box::new(on_spawn)); // Direct access
        self
    }

    /// Insert a [`Bundle`] onto the node's entity.
    pub fn insert<T: Bundle>(self, bundle: T) -> Self {
        self.on_spawn(move |world, entity| {
            if let Ok(mut entity) = world.get_entity_mut(entity) {
                entity.insert(bundle);
            }
        })
    }

    /// Sync the entity with an [`OnceLock<Entity>`].
    pub fn entity_sync(self, entity: LazyEntity) -> Self {
        self.on_spawn(move |_, e| {
            let _ = entity.set(e);
        })
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    /// The system receives the value emitted by the signal.
    /// Note: The system is registered *directly* on the builder and will be transferred
    /// to the entity's `SignalHandlers` component upon spawning.
    pub fn on_signal<I, S, F, M>(self, signal: S, system: F) -> Self
    where
        I: FromReflect + GetTypeRegistration + Typed + SSs,
        S: Signal<Item = I> + SSs,
        F: IntoSystem<In<(Entity, I)>, (), M> + SSs,
        M: SSs,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            let handle = Signal::register_signal(
                signal
                    .map(move |In(input): In<I>| (entity, input))
                    .map(system),
                world,
            );
            add_handle(world, entity, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn signal_from_entity<O, OS, F>(self, f: F) -> Self
    where
        O: FromReflect + GetTypeRegistration + Typed + SSs,
        OS: Signal<Item = O> + SSs,
        F: FnOnce(Source<Entity>) -> OS + SSs,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            let handle = Signal::register_signal(f(SignalBuilder::from_entity(entity)), world);
            add_handle(world, entity, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn signal_from_component<C, OS, F, O>(self, f: F) -> Self
    where
        C: Component + Clone + FromReflect + GetTypeRegistration + Typed,
        OS: Signal<Item = O> + SSs,
        F: FnOnce(Map<Source<Entity>, C>) -> OS + SSs,
        O: FromReflect + GetTypeRegistration + Typed + SSs,
    {
        self.signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&C>| {
                components.get(entity).ok().cloned()
            }))
        })
    }

    pub fn signal_from_ancestor<O, OS, F>(self, generations: usize, f: F) -> Self
    where
        O: FromReflect + GetTypeRegistration + Typed + SSs,
        OS: Signal<Item = O> + SSs,
        F: FnOnce(Map<Source<Entity>, Entity>) -> OS + SSs,
    {
        self.signal_from_entity(move |signal| {
            f(
                signal.map(move |In(entity): In<Entity>, parents: Query<&ChildOf>| {
                    parents
                        .iter_ancestors(entity)
                        .nth(generations.saturating_sub(1))
                }),
            )
        })
    }

    pub fn signal_from_parent<O, OS, F>(self, f: F) -> Self
    where
        O: FromReflect + GetTypeRegistration + Typed + SSs,
        OS: Signal<Item = O> + SSs,
        F: FnOnce(Map<Source<Entity>, Entity>) -> OS + SSs,
    {
        self.signal_from_ancestor(1, f)
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn component_signal<S, O>(self, signal: S) -> Self
    where
        S: Signal<Item = Option<O>> + SSs,
        O: Component + FromReflect + GetTypeRegistration + Typed + SSs,
    {
        self.on_signal(
            signal,
            move |In((entity, component_option)): In<(Entity, Option<O>)>, world: &mut World| {
                if let Ok(mut entity) = world.get_entity_mut(entity) {
                    if let Some(component) = component_option {
                        entity.insert(component);
                    } else {
                        entity.remove::<O>();
                    }
                }
            },
        )
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn component_signal_from_entity<C, S, F>(self, f: F) -> Self
    where
        C: Component + FromReflect + GetTypeRegistration + Typed + SSs,
        S: Signal<Item = Option<C>> + SSs,
        F: FnOnce(Source<Entity>) -> S + SSs,
    {
        let entity = LazyEntity::new();
        self.entity_sync(entity.clone())
            .signal_from_entity(move |signal| {
                f(signal).map(
                    move |In(component_option): In<Option<C>>, world: &mut World| {
                        if let Ok(mut entity) = world.get_entity_mut(entity.get()) {
                            if let Some(component) = component_option {
                                entity.insert(component);
                            } else {
                                entity.remove::<C>();
                            }
                        }
                    },
                )
            })
    }

    /// Register a reactive system that runs when the given [`Signal`] emits a value.
    pub fn component_signal_from_component<I, O, S, F>(self, f: F) -> Self
    where
        I: Component + Clone + FromReflect + GetTypeRegistration + Typed,
        O: Component + FromReflect + GetTypeRegistration + Typed + SSs,
        S: Signal<Item = Option<O>> + SSs,
        F: FnOnce(Map<Source<Entity>, I>) -> S + SSs,
    {
        self.component_signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&I>| {
                components.get(entity).ok().cloned()
            }))
        })
    }

    /// Declare a static child node.
    /// The child is spawned and added to the parent when the parent is spawned.
    pub fn child(self, child: JonmoBuilder) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(1);
        let offset = offset(block, &self.child_block_populations.lock().unwrap());
        let on_spawn = move |world: &mut World, parent| {
            let child_entity = world.spawn_empty().id();
            if let Ok(ref mut parent) = world.get_entity_mut(parent) {
                EntityWorldMut::insert_children(parent, offset, &[child_entity]);
                child.spawn_on_entity(world, child_entity);
            } else {
                if let Ok(child) = world.get_entity_mut(child_entity) {
                    child.despawn();
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare a reactive child. When the [`Signal`] outputs [`None`], the child is removed.
    pub fn child_signal<T: Into<Option<JonmoBuilder>> + FromReflect>(
        self,
        child_option: impl Signal<Item = T> + SSs,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let system =
                move |In(child_option): In<T>,
                      world: &mut World,
                      mut existing_child_option: Local<Option<Entity>>| {
                    if let Some(child) = child_option.into() {
                        if let Some(existing_child) = existing_child_option.take() {
                            if let Ok(entity) = world.get_entity_mut(existing_child) {
                                entity.despawn();
                            }
                        }
                        let child_entity = world.spawn_empty().id();
                        if let Ok(mut parent) = world.get_entity_mut(parent) {
                            let offset = offset(block, &child_block_populations.lock().unwrap());
                            parent.insert_children(offset, &[child_entity]);
                            child.spawn_on_entity(world, child_entity);
                            *existing_child_option = Some(child_entity);
                        } else {
                            if let Ok(child) = world.get_entity_mut(child_entity) {
                                child.despawn();
                            }
                        }
                        child_block_populations.lock().unwrap()[block] = 1;
                    } else {
                        if let Some(existing_child) = existing_child_option.take() {
                            if let Ok(entity) = world.get_entity_mut(existing_child) {
                                entity.despawn();
                            }
                        }
                        child_block_populations.lock().unwrap()[block] = 0;
                    }
                };
            let signal = child_option.map(system);
            let handle = Signal::register_signal(signal, world);
            add_handle(world, parent, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Declare static children.
    pub fn children(
        self,
        children: impl IntoIterator<Item = JonmoBuilder> + Send + 'static,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        let children_vec: Vec<JonmoBuilder> = children.into_iter().collect(); // Collect into Vec
        let population = children_vec.len();
        self.child_block_populations
            .lock()
            .unwrap()
            .push(population);
        let child_block_populations = self.child_block_populations.clone(); // Clone Arc

        let on_spawn = move |world: &mut World, parent: Entity| {
            let mut children_entities = Vec::with_capacity(children_vec.len());
            for _ in 0..children_vec.len() {
                children_entities.push(world.spawn_empty().id());
            }

            if let Ok(mut parent) = world.get_entity_mut(parent) {
                let offset = offset(block, &child_block_populations.lock().unwrap()); // Recalculate offset
                parent.insert_children(offset, &children_entities);
                for (child, child_entity) in children_vec
                    .into_iter()
                    .zip(children_entities.iter().copied())
                {
                    // Use copied iterator
                    child.spawn_on_entity(world, child_entity);
                }
            } else {
                // Parent despawned during child spawning
                for child_entity in children_entities {
                    if let Ok(child) = world.get_entity_mut(child_entity) {
                        child.despawn();
                    }
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare reactive children based on a `SignalVec`.
    pub fn children_signal_vec(
        self,
        children_signal_vec: impl SignalVec<Item = JonmoBuilder>,
    ) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let system = move |In(diffs): In<Vec<VecDiff<JonmoBuilder>>>,
                               world: &mut World,
                               mut children_entities: Local<Vec<Entity>>| {
                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values: children } => {
                            for child_entity in children_entities.drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                            }
                            *children_entities =
                                children.iter().map(|_| world.spawn_empty().id()).collect();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset, &children_entities);
                                for (child, child_entity) in
                                    children.into_iter().zip(children_entities.iter().copied())
                                {
                                    child.spawn_on_entity(world, child_entity);
                                }
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::InsertAt {
                            index,
                            value: child,
                        } => {
                            let child_entity = world.spawn_empty().id();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + index, &[child_entity]);
                                child.spawn_on_entity(world, child_entity);
                                children_entities.insert(index, child_entity);
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            } else {
                                // Parent despawned during child insertion
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                            }
                        }
                        VecDiff::Push { value: child } => {
                            let child_entity = world.spawn_empty().id();
                            let mut push_child_entity = false;
                            {
                                if let Ok(mut parent) = world.get_entity_mut(parent) {
                                    let offset =
                                        offset(block, &child_block_populations.lock().unwrap());
                                    parent.insert_children(
                                        offset + children_entities.len(),
                                        &[child_entity],
                                    );
                                    child.spawn_on_entity(world, child_entity);
                                    push_child_entity = true;
                                    child_block_populations.lock().unwrap()[block] =
                                        children_entities.len();
                                } else {
                                    // parent despawned during child spawning
                                    if let Ok(child) = world.get_entity_mut(child_entity) {
                                        child.despawn();
                                    }
                                }
                            }
                            if push_child_entity {
                                children_entities.push(child_entity);
                            }
                        }
                        VecDiff::UpdateAt { index, value: node } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    child.despawn(); // removes from parent
                                }
                            }
                            let child_entity = world.spawn_empty().id();
                            let mut set_child_entity = false;
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                set_child_entity = true;
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + index, &[child_entity]);
                                node.spawn_on_entity(world, child_entity);
                            } else {
                                // parent despawned during child spawning
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                            }
                            if set_child_entity {
                                children_entities[index] = child_entity;
                            }
                        }
                        VecDiff::Move {
                            old_index,
                            new_index,
                        } => {
                            children_entities.swap(old_index, new_index);
                            fn move_from_to(
                                parent: &mut EntityWorldMut,
                                children_entities: &[Entity],
                                old_index: usize,
                                new_index: usize,
                            ) {
                                if old_index != new_index {
                                    if let Some(old_entity) =
                                        children_entities.get(old_index).copied()
                                    {
                                        parent.remove_children(&[old_entity]);
                                        parent.insert_children(new_index, &[old_entity]);
                                    }
                                }
                            }
                            fn swap(
                                parent: &mut EntityWorldMut,
                                children_entities: &[Entity],
                                a: usize,
                                b: usize,
                            ) {
                                move_from_to(parent, children_entities, a, b);
                                match a.cmp(&b) {
                                    std::cmp::Ordering::Less => {
                                        move_from_to(parent, children_entities, b - 1, a);
                                    }
                                    std::cmp::Ordering::Greater => {
                                        move_from_to(parent, children_entities, b + 1, a)
                                    }
                                    _ => {}
                                }
                            }
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset =
                                    offset(block, &child_block_populations.lock().unwrap());
                                swap(
                                    &mut parent,
                                    &children_entities,
                                    offset + old_index,
                                    offset + new_index,
                                );
                            }
                        }
                        VecDiff::RemoveAt { index } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    child.despawn(); // removes from parent
                                }
                                children_entities.remove(index);
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::Pop => {
                            if let Some(child_entity) = children_entities.pop() {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                                child_block_populations.lock().unwrap()[block] =
                                    children_entities.len();
                            }
                        }
                        VecDiff::Clear => {
                            for child_entity in children_entities.drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                            }
                            child_block_populations.lock().unwrap()[block] =
                                children_entities.len();
                        }
                    }
                }
            };
            let handle =
                SignalVec::register_signal_vec(children_signal_vec.for_each(system), world);
            add_handle(world, parent, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Spawn the node with its components and reactive systems onto an existing [`Entity`].
    ///
    /// Note: This consumes the builder.
    pub fn spawn_on_entity(self, world: &mut World, entity: Entity) {
        if let Ok(mut entity) = world.get_entity_mut(entity) {
            let id = entity.id();
            entity.insert(SignalHandles::default());
            for on_spawn in self.on_spawns.lock().unwrap().drain(..) {
                on_spawn(world, id);
            }
        }
    }

    /// Spawn the node with its components and reactive systems onto a new [`Entity`].
    pub fn spawn(self, world: &mut World) -> Entity {
        let entity = world.spawn_empty().id();
        self.spawn_on_entity(world, entity);
        entity
    }
}

fn offset(i: usize, child_block_populations: &[usize]) -> usize {
    child_block_populations[0..i].iter().copied().sum()
}
