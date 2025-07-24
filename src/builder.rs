//! Reactive entity builder ported from [Dominator](https://github.com/Pauan/rust-dominator)'s [`DomBuilder`](https://docs.rs/dominator/latest/dominator/struct.DomBuilder.html).
use super::{
    graph::{SignalHandle, SignalHandles},
    signal::{Signal, SignalBuilder, SignalExt},
    signal_vec::{SignalVec, SignalVecExt, VecDiff},
    utils::{LazyEntity, SSs, ancestor_map},
};
use bevy_ecs::{component::Mutable, prelude::*};
use bevy_platform::{
    prelude::*,
    sync::{Arc, Mutex},
};
use core::cmp::Ordering;

fn add_handle(world: &mut World, entity: Entity, handle: SignalHandle) {
    if let Ok(mut entity) = world.get_entity_mut(entity)
        && let Some(mut handles) = entity.get_mut::<SignalHandles>()
    {
        handles.add(handle);
    }
}

// TODO: the fluent interface link breaks cargo fmt ??
/// A thin facade over a Bevy [`Entity`] enabling the ergonomic registration of
/// reactive components and children using a declarative
/// [fluent](https://en.wikipedia.org/wiki/Fluent_interface) builder pattern. All
/// its methods are deferred until the corresponding [`Entity`] is spawned so its
/// state _and how that state should change_ depending on the state of the
/// [`World`] can be specified up front, in a tidy colocated package, without a
/// `&mut World` or [`Commands`].
///
/// Port of [Dominator](https://github.com/Pauan/rust-dominator)'s
/// [`DomBuilder`](https://docs.rs/dominator/latest/dominator/struct.DomBuilder.html),
/// and [haalka](https://github.com/databasedav/haalka)'s
/// [`NodeBuilder`](https://docs.rs/haalka/latest/haalka/node_builder/struct.NodeBuilder.html).
#[derive(Clone)]
pub struct JonmoBuilder {
    #[allow(clippy::type_complexity)]
    on_spawns: Arc<Mutex<Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>>>,
    child_block_populations: Arc<Mutex<Vec<usize>>>,
}

impl<T: Bundle> From<T> for JonmoBuilder {
    fn from(bundle: T) -> Self {
        Self::new().insert(bundle)
    }
}

impl JonmoBuilder {
    #[allow(clippy::new_without_default, missing_docs)]
    pub fn new() -> Self {
        Self {
            on_spawns: Arc::new(Mutex::new(Vec::new())),
            child_block_populations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Run a function with mutable access to the [`World`] and this builder's [`Entity`].
    pub fn on_spawn(self, on_spawn: impl FnOnce(&mut World, Entity) + SSs) -> Self {
        self.on_spawns.lock().unwrap().push(Box::new(on_spawn));
        self
    }

    /// Run a function with this builder's [`EntityWorldMut`].
    pub fn with_entity(self, f: impl FnOnce(EntityWorldMut) + SSs) -> Self {
        self.on_spawn(move |world, entity| {
            if let Ok(entity) = world.get_entity_mut(entity) {
                f(entity);
            }
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

    /// Set the [`LazyEntity`] to this builder's [`Entity`].
    pub fn entity_sync(self, entity: LazyEntity) -> Self {
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
            add_handle(world, entity, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Reactively run a function with mutable access (via [`Mut`]) to this builder's [`Entity`]'s
    /// `C` [`Component`] if it exists and the output of a [`Signal`].
    ///
    /// The `signal` will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn on_signal_with_component<C, I, S, F, M>(self, signal: S, mut f: F) -> Self
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
            add_handle(world, entity, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`] and returns a
    /// [`Signal`].
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_entity<S, F>(self, f: F) -> Self
    where
        S: Signal,
        F: FnOnce(super::signal::Source<Entity>) -> S + SSs,
    {
        let on_spawn = move |world: &mut World, entity: Entity| {
            let handle = Signal::register_signal(f(SignalBuilder::from_entity(entity)), world);
            add_handle(world, entity, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s `C`
    /// [`Component`] and returns a [`Signal`]; if this builder's [`Entity`] does not have a `C`
    /// [`Component`], the [`Signal`] chain will terminate for that frame.
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_component<C, S, F>(self, f: F) -> Self
    where
        C: Component + Clone,
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, C>) -> S + SSs,
    {
        self.signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&C>| components.get(entity).ok().cloned()))
        })
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s `C`
    /// [`Component`] wrapped in an [`Option`] and returns a [`Signal`]; if this builder's
    /// [`Entity`] does not have a `C` [`Component`], the [`Signal`] will output [`None`] and
    /// continue propagation.
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_component_option<C, S, F>(self, f: F) -> Self
    where
        C: Component + Clone,
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Option<C>>) -> S + SSs,
    {
        self.signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&C>| Some(components.get(entity).ok().cloned())))
        })
    }

    // TODO: ehh these could be useful but one should really just use LazyEntity ...
    /* pub fn signal_from_ancestor_find<S, F, P, M>(self, f: F, predicate: P) -> Self
    where
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Option<Entity>>) -> S + SSs,
        P: IntoSystem<In<Entity>, bool, M> + SSs
    {
        let system = Arc::new(OnceLock::new());
        self.on_spawn(clone!((system) move|world, entity| {
            let system_id = world.register_system(predicate);
            world.entity_mut(entity).add_child(system_id.entity());
            let _ =system.set(system_id);
        }))
        .signal_from_entity(move |signal| {
            f(signal.map(move |In(entity): In<Entity>, world: &mut World| {
                let mutancestors = SystemState::<Query<&ChildOf>>::new(world);
                let ancestors = ancestors.get(world);
                let ancestors = ancestors.iter_ancestors(entity).collect::<Vec<_>>();
                ancestors.into_iter().find(|&ancestor| {
                    world.run_system_with(system.get().copied().unwrap(), ancestor) .ok().unwrap_or(false)
                })
            }))
        })
    }

    pub fn signal_from_ancestor_with_component<C, S, F>(self, f: F) -> Self
    where
        C: Component,
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Option<Entity>>)-> S + SSs
    {
        self.signal_from_ancestor_find(f, |In(entity), components:Query<&C>| components.contains(entity))
    } */

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s
    /// `generations`-nth generation ancestor and returns a [`Signal`]. Passing `0` to `generations`
    /// will return this builder's [`Entity`] itself.
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_ancestor<S, F>(self, generations: usize, f: F) -> Self
    where
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        self.signal_from_entity(move |signal| f(signal.map(ancestor_map(generations))))
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's parent's [`Entity`] and
    /// returns a [`Signal`].
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_parent<S, F>(self, f: F) -> Self
    where
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        self.signal_from_ancestor(1, f)
    }

    /// Reactively set this builder's [`Entity`]'s `C` [`Component`] with a [`Signal`] that outputs
    /// an [`Option`]al `C`; if the [`Signal`] outputs [`None`], the `C` [`Component`] is removed.
    /// If the [`Signal`]'s output is infallible, wrapping the result in an [`Option`] is
    /// unnecessary.
    pub fn component_signal<C, S>(self, signal: S) -> Self
    where
        C: Component + Clone,
        S: Signal<Item = Option<C>> + SSs,
    {
        self.on_signal(
            signal,
            move |In((entity, component_option)): In<(Entity, Option<C>)>, world: &mut World| {
                if let Ok(mut entity) = world.get_entity_mut(entity) {
                    if let Some(component) = component_option {
                        entity.insert(component);
                    } else {
                        entity.remove::<C>();
                    }
                }
            },
        )
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`] and returns a
    /// [`Signal`] that outputs an [`Option`]al `C`; this resulting [`Signal`] reactively sets this
    /// builder's [`Entity`]'s `C` [`Component`]; if the [`Signal`] outputs [`None`], the `C`
    /// [`Component`] is removed. If the resulting [`Signal`]'s output is infallible, wrapping the
    /// result in an [`Option`] is unnecessary.
    pub fn component_signal_from_entity<C, IOC, S, F>(self, f: F) -> Self
    where
        C: Component,
        IOC: Into<Option<C>> + 'static,
        S: Signal<Item = IOC>,
        F: FnOnce(super::signal::Source<Entity>) -> S + SSs,
    {
        let entity = LazyEntity::new();
        self.entity_sync(entity.clone()).signal_from_entity(move |signal| {
            f(signal).map(move |In(component_option): In<IOC>, world: &mut World| {
                if let Ok(mut entity) = world.get_entity_mut(entity.get()) {
                    if let Some(component) = component_option.into() {
                        entity.insert(component);
                    } else {
                        entity.remove::<C>();
                    }
                }
            })
        })
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`] and returns a
    /// [`Signal`] that outputs an [`Option`]al `C`; this resulting [`Signal`] reactively sets this
    /// builder's [`Entity`]'s `C` [`Component`]; if the [`Signal`] outputs [`None`], the `C`
    /// [`Component`] is removed. If the resulting [`Signal`]'s output is infallible, wrapping the
    /// result in an [`Option`] is unnecessary.
    pub fn component_signal_from_ancestor<C, IOC, S, F>(self, generations: usize, f: F) -> Self
    where
        C: Component,
        IOC: Into<Option<C>> + 'static,
        S: Signal<Item = IOC>,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        let entity = LazyEntity::new();
        self.entity_sync(entity.clone()).signal_from_entity(move |signal| {
            f(signal.map(move |In(entity): In<Entity>, parents: Query<&ChildOf>| {
                [entity]
                    .into_iter()
                    .chain(parents.iter_ancestors(entity))
                    .nth(generations)
            }))
            .map(move |In(component_option): In<IOC>, world: &mut World| {
                if let Ok(mut entity) = world.get_entity_mut(entity.get()) {
                    if let Some(component) = component_option.into() {
                        entity.insert(component);
                    } else {
                        entity.remove::<C>();
                    }
                }
            })
        })
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`] and returns a
    /// [`Signal`] that outputs an [`Option`]al `C`; this resulting [`Signal`] reactively sets this
    /// builder's [`Entity`]'s `C` [`Component`]; if the [`Signal`] outputs [`None`], the `C`
    /// [`Component`] is removed. If the resulting [`Signal`]'s output is infallible, wrapping the
    /// result in an [`Option`] is unnecessary.
    pub fn component_signal_from_parent<C, IOC, S, F>(self, f: F) -> Self
    where
        C: Component,
        IOC: Into<Option<C>> + 'static,
        S: Signal<Item = IOC>,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        self.component_signal_from_ancestor(1, f)
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s `C`
    /// [`Component`] and returns a [`Signal`] that outputs an [`Option`]al `C`; this resulting
    /// [`Signal`] reactively sets this builder's [`Entity`]'s `C` [`Component`]; if the [`Signal`]
    /// outputs [`None`], the `C` [`Component`] is removed. If this builder's [`Entity`] does not
    /// have a `C` [`Component`], the [`Signal`]'s execution path will terminate for that frame. If
    /// the resulting [`Signal`]'s output is infallible, wrapping the result in an [`Option`] is
    /// unnecessary.
    pub fn component_signal_from_component<I, O, IOO, S, F>(self, f: F) -> Self
    where
        I: Component + Clone,
        O: Component,
        IOO: Into<Option<O>> + 'static,
        S: Signal<Item = IOO> + SSs,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, I>) -> S + SSs,
    {
        self.component_signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&I>| components.get(entity).ok().cloned()))
        })
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s `C`
    /// [`Component`] and returns a [`Signal`] that outputs an [`Option`]al `C`; this resulting
    /// [`Signal`] reactively sets this builder's [`Entity`]'s `C` [`Component`]; if the [`Signal`]
    /// outputs [`None`], the `C` [`Component`] is removed. If this builder's [`Entity`] does not
    /// have a `C` [`Component`], the input [`Signal`] will output [`None`] and continue
    /// propagation. If the resulting [`Signal`]'s output is infallible, wrapping the result in an
    /// [`Option`] is unnecessary.
    pub fn component_signal_from_component_option<I, O, IOO, S, F>(self, f: F) -> Self
    where
        I: Component + Clone,
        O: Component,
        IOO: Into<Option<O>> + 'static,
        S: Signal<Item = IOO> + SSs,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Option<I>>) -> S + SSs,
    {
        self.component_signal_from_entity(|signal| {
            f(signal.map(|In(entity): In<Entity>, components: Query<&I>| Some(components.get(entity).ok().cloned())))
        })
    }

    /// Declare a static child.
    pub fn child(self, child: impl Into<JonmoBuilder>) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(1);
        let offset = offset(block, &self.child_block_populations.lock().unwrap());
        let child = child.into();
        let on_spawn = move |world: &mut World, parent| {
            let child_entity = world.spawn_empty().id();
            if let Ok(ref mut parent) = world.get_entity_mut(parent) {
                // need to call like this to avoid type ambiguity
                EntityWorldMut::insert_children(parent, offset, &[child_entity]);
                child.spawn_on_entity(world, child_entity);
            } else if let Ok(child) = world.get_entity_mut(child_entity) {
                child.despawn();
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare a reactive child. When the [`Signal`] outputs [`None`], the child is removed.
    pub fn child_signal<T: Into<Option<JonmoBuilder>> + 'static>(self, child_option: impl Signal<Item = T>) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let system =
                move |In(child_option): In<T>, world: &mut World, mut existing_child_option: Local<Option<Entity>>| {
                    if let Some(child) = child_option.into() {
                        if let Some(existing_child) = existing_child_option.take()
                            && let Ok(entity) = world.get_entity_mut(existing_child)
                        {
                            entity.despawn();
                        }
                        let child_entity = world.spawn_empty().id();
                        if let Ok(mut parent) = world.get_entity_mut(parent) {
                            let offset = offset(block, &child_block_populations.lock().unwrap());
                            parent.insert_children(offset, &[child_entity]);
                            child.spawn_on_entity(world, child_entity);
                            *existing_child_option = Some(child_entity);
                        } else if let Ok(child) = world.get_entity_mut(child_entity) {
                            child.despawn();
                        }
                        child_block_populations.lock().unwrap()[block] = 1;
                    } else {
                        if let Some(existing_child) = existing_child_option.take()
                            && let Ok(entity) = world.get_entity_mut(existing_child)
                        {
                            entity.despawn();
                        }
                        child_block_populations.lock().unwrap()[block] = 0;
                    }
                };
            let handle = child_option.map(system).register(world);
            add_handle(world, parent, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Declare static children.
    pub fn children(self, children: impl IntoIterator<Item = impl Into<JonmoBuilder>> + Send + 'static) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        let children_vec: Vec<JonmoBuilder> = children.into_iter().map(Into::into).collect();
        let population = children_vec.len();
        self.child_block_populations.lock().unwrap().push(population);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let mut children_entities = Vec::with_capacity(children_vec.len());
            for _ in 0..children_vec.len() {
                children_entities.push(world.spawn_empty().id());
            }
            if let Ok(mut parent) = world.get_entity_mut(parent) {
                let offset = offset(block, &child_block_populations.lock().unwrap());
                parent.insert_children(offset, &children_entities);
                for (child, child_entity) in children_vec.into_iter().zip(children_entities.iter().copied()) {
                    child.spawn_on_entity(world, child_entity);
                }
            } else {
                for child_entity in children_entities {
                    if let Ok(child) = world.get_entity_mut(child_entity) {
                        child.despawn(); // removes from parent
                    }
                }
            }
        };
        self.on_spawn(on_spawn)
    }

    /// Declare reactive children.
    pub fn children_signal_vec(self, children_signal_vec: impl SignalVec<Item = JonmoBuilder>) -> Self {
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
                            *children_entities = children.iter().map(|_| world.spawn_empty().id()).collect();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset, &children_entities);
                                for (child, child_entity) in children.into_iter().zip(children_entities.iter().copied())
                                {
                                    child.spawn_on_entity(world, child_entity);
                                }
                                child_block_populations.lock().unwrap()[block] = children_entities.len();
                            }
                        }
                        VecDiff::InsertAt { index, value: child } => {
                            let child_entity = world.spawn_empty().id();
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + index, &[child_entity]);
                                child.spawn_on_entity(world, child_entity);
                                children_entities.insert(index, child_entity);
                                child_block_populations.lock().unwrap()[block] = children_entities.len();
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
                                    let offset = offset(block, &child_block_populations.lock().unwrap());
                                    parent.insert_children(offset + children_entities.len(), &[child_entity]);
                                    child.spawn_on_entity(world, child_entity);
                                    push_child_entity = true;
                                    child_block_populations.lock().unwrap()[block] = children_entities.len();
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
                            if let Some(existing_child) = children_entities.get(index).copied()
                                && let Ok(child) = world.get_entity_mut(existing_child)
                            {
                                child.despawn(); // removes from parent
                            }
                            let child_entity = world.spawn_empty().id();
                            let mut set_child_entity = false;
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                set_child_entity = true;
                                let offset = offset(block, &child_block_populations.lock().unwrap());
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
                        VecDiff::Move { old_index, new_index } => {
                            children_entities.swap(old_index, new_index);

                            fn move_from_to(
                                parent: &mut EntityWorldMut,
                                children_entities: &[Entity],
                                old_index: usize,
                                new_index: usize,
                            ) {
                                if old_index != new_index
                                    && let Some(old_entity) = children_entities.get(old_index).copied()
                                {
                                    parent.remove_children(&[old_entity]);
                                    parent.insert_children(new_index, &[old_entity]);
                                }
                            }

                            fn swap(parent: &mut EntityWorldMut, children_entities: &[Entity], a: usize, b: usize) {
                                move_from_to(parent, children_entities, a, b);
                                match a.cmp(&b) {
                                    Ordering::Less => {
                                        move_from_to(parent, children_entities, b - 1, a);
                                    }
                                    Ordering::Greater => move_from_to(parent, children_entities, b + 1, a),
                                    _ => {}
                                }
                            }

                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                swap(&mut parent, &children_entities, offset + old_index, offset + new_index);
                            }
                        }
                        VecDiff::RemoveAt { index } => {
                            if let Some(existing_child) = children_entities.get(index).copied() {
                                if let Ok(child) = world.get_entity_mut(existing_child) {
                                    // removes from parent
                                    child.despawn();
                                }
                                children_entities.remove(index);
                                child_block_populations.lock().unwrap()[block] = children_entities.len();
                            }
                        }
                        VecDiff::Pop => {
                            if let Some(child_entity) = children_entities.pop() {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                                child_block_populations.lock().unwrap()[block] = children_entities.len();
                            }
                        }
                        VecDiff::Clear => {
                            for child_entity in children_entities.drain(..) {
                                if let Ok(child) = world.get_entity_mut(child_entity) {
                                    child.despawn();
                                }
                            }
                            child_block_populations.lock().unwrap()[block] = children_entities.len();
                        }
                    }
                }
            };
            let handle = children_signal_vec.for_each(system).register(world);
            add_handle(world, parent, handle);
        };
        self.on_spawn(on_spawn)
    }

    /// Spawn this builder on an existing [`Entity`].
    pub fn spawn_on_entity(self, world: &mut World, entity: Entity) {
        if let Ok(mut entity) = world.get_entity_mut(entity) {
            let id = entity.id();
            entity.insert(SignalHandles::default());
            for on_spawn in self.on_spawns.lock().unwrap().drain(..) {
                on_spawn(world, id);
            }
        }
    }

    /// Spawn this builder into the [`World`].
    pub fn spawn(self, world: &mut World) -> Entity {
        let entity = world.spawn_empty().id();
        self.spawn_on_entity(world, entity);
        entity
    }
}

fn offset(i: usize, child_block_populations: &[usize]) -> usize {
    child_block_populations[..i].iter().sum()
}
