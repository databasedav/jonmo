//! Reactive entity builder ported from [Dominator](https://github.com/Pauan/rust-dominator)'s [`DomBuilder`](https://docs.rs/dominator/latest/dominator/struct.DomBuilder.html).
use super::{
    graph::{SignalHandle, SignalHandles},
    signal::{Signal, SignalBuilder, SignalExt},
    signal_vec::{SignalVec, SignalVecExt, VecDiff},
    utils::{LazyEntity, SSs, ancestor_map},
};
use bevy_ecs::{
    component::Mutable,
    prelude::*,
    system::{IntoObserverSystem, RunSystemOnce},
};
use bevy_platform::{
    prelude::*,
    sync::{Arc, Mutex},
};

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
/// state *and how that state should change* depending on the state of the
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

    /// Run a [`System`] which takes [`In`] this builder's [`Entity`].
    pub fn on_spawn_with_system<T, M>(self, system: T) -> Self
    where
        T: IntoSystem<In<Entity>, (), M> + SSs,
    {
        self.on_spawn(|world, entity| {
            if let Err(error) = world.run_system_once_with(system, entity) {
                #[cfg(feature = "tracing")]
                bevy_log::error!("failed to run system on spawn: {}", error);
            }
        })
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

    /// Attach an [`Observer`] to this builder.
    pub fn observe<E: Event, B: Bundle, Marker>(self, observer: impl IntoObserverSystem<E, B, Marker> + Sync) -> Self {
        self.on_spawn(|world, entity| {
            if let Ok(mut entity) = world.get_entity_mut(entity) {
                entity.observe(observer);
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
            if let Ok(entity) = world.get_entity_mut(entity) {
                f(entity, value)
            }
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
    /// [`Component`], the [`Signal`] chain will terminate for the frame.
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

    // Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s `C`
    /// [`Component`] on frames it has [`Changed`] and returns a [`Signal`]; if this builder's
    /// [`Entity`] does not have a `C` [`Component`], the [`Signal`] chain will terminate for
    /// the frame.
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_component_changed<C, S, F>(self, f: F) -> Self
    where
        C: Component + Clone,
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, C>) -> S + SSs,
    {
        self.signal_from_entity(|signal| {
            f(signal
                .map(|In(entity): In<Entity>, components: Query<&C, Changed<C>>| components.get(entity).ok().cloned()))
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
    /// `generations`-th generation ancestor if it exists (terminating for the frame otherwise) and
    /// returns a [`Signal`]. Passing `0` to `generations` will output this builder's [`Entity`]
    /// itself.
    ///
    /// The resulting [`Signal`] will be automatically cleaned up when the [`Entity`] is despawned.
    pub fn signal_from_ancestor<S, F>(self, generations: usize, f: F) -> Self
    where
        S: Signal,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        self.signal_from_entity(move |signal| f(signal.map(ancestor_map(generations))))
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's parent's [`Entity`] if
    /// it exists (terminating for the frame otherwise) and returns a [`Signal`].
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
    /// [`Component`] is removed.
    pub fn component_signal_from_entity<C, S, F>(self, f: F) -> Self
    where
        C: Component,
        S: Signal<Item = Option<C>>,
        F: FnOnce(super::signal::Source<Entity>) -> S + SSs,
    {
        let entity = LazyEntity::new();
        self.entity_sync(entity.clone()).signal_from_entity(move |signal| {
            f(signal).map(move |In(component_option): In<Option<C>>, world: &mut World| {
                if let Ok(mut entity) = world.get_entity_mut(entity.get()) {
                    if let Some(component) = component_option {
                        entity.insert(component);
                    } else {
                        entity.remove::<C>();
                    }
                }
            })
        })
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s
    /// `generations`-th generation ancestor's [`Entity`] if it exists (terminating for the frame
    /// otherwise) and returns a [`Signal`] that outputs an [`Option`]al `C`; this resulting
    /// [`Signal`] reactively sets this builder's [`Entity`]'s `C` [`Component`]; if the
    /// [`Signal`] outputs [`None`], the `C` [`Component`] is removed.
    pub fn component_signal_from_ancestor<C, S, F>(self, generations: usize, f: F) -> Self
    where
        C: Component,
        S: Signal<Item = Option<C>>,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        let entity = LazyEntity::new();
        self.entity_sync(entity.clone()).signal_from_entity(move |signal| {
            f(signal.map(ancestor_map(generations))).map(
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

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s parent
    /// [`Entity`] if it exists (terminating for the frame otherwise) and returns a [`Signal`] that
    /// outputs an [`Option`]al `C`; this resulting [`Signal`] reactively sets this builder's
    /// [`Entity`]'s `C` [`Component`]; if the [`Signal`] outputs [`None`], the `C`
    /// [`Component`] is removed.
    pub fn component_signal_from_parent<C, S, F>(self, f: F) -> Self
    where
        C: Component,
        S: Signal<Item = Option<C>>,
        F: FnOnce(super::signal::Map<super::signal::Source<Entity>, Entity>) -> S + SSs,
    {
        self.component_signal_from_ancestor(1, f)
    }

    /// Run a function that takes a [`Signal`] which outputs this builder's [`Entity`]'s `C`
    /// [`Component`] and returns a [`Signal`] that outputs an [`Option`]al `C`; this resulting
    /// [`Signal`] reactively sets this builder's [`Entity`]'s `C` [`Component`]; if the [`Signal`]
    /// outputs [`None`], the `C` [`Component`] is removed. If this builder's [`Entity`] does not
    /// have a `C` [`Component`], the [`Signal`]'s execution path will terminate for the frame.
    pub fn component_signal_from_component<I, O, S, F>(self, f: F) -> Self
    where
        I: Component + Clone,
        O: Component,
        S: Signal<Item = Option<O>> + SSs,
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
    /// propagation.
    pub fn component_signal_from_component_option<I, O, S, F>(self, f: F) -> Self
    where
        I: Component + Clone,
        O: Component,
        S: Signal<Item = Option<O>> + SSs,
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
    pub fn child_signal(self, child_option: impl Signal<Item = Option<JonmoBuilder>>) -> Self {
        let block = self.child_block_populations.lock().unwrap().len();
        self.child_block_populations.lock().unwrap().push(0);
        let child_block_populations = self.child_block_populations.clone();
        let on_spawn = move |world: &mut World, parent: Entity| {
            let system = move |In(child_option): In<Option<JonmoBuilder>>,
                               world: &mut World,
                               mut existing_child_option: Local<Option<Entity>>| {
                if let Some(child) = child_option {
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
                            // First, update our local tracker to match the new logical order. This is the
                            // source of truth for which entity corresponds to which index.
                            let moved_entity = children_entities.remove(old_index);
                            children_entities.insert(new_index, moved_entity);

                            // Now, apply the same reordering to the actual parent entity in the world.
                            if let Ok(mut parent) = world.get_entity_mut(parent) {
                                // Bevy's `remove_children` finds the entity and removes it from its
                                // current position, correctly shifting subsequent children.
                                parent.remove_children(&[moved_entity]);

                                // The new insertion index must be calculated with the offset from any
                                // preceding static children.
                                let offset = offset(block, &child_block_populations.lock().unwrap());
                                parent.insert_children(offset + new_index, &[moved_entity]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        JonmoPlugin,
        prelude::MutableVec,
        signal::{SignalBuilder, SignalExt},
    };
    use bevy::prelude::*;
    use bevy_platform::collections::HashSet;

    /// Helper to create a minimal Bevy App with the JonmoPlugin for testing.
    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app
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
        let source_signal = SignalBuilder::from_resource::<SignalSource>()
            .map_in(|source: SignalSource| source.0)
            .dedupe();

        // --- 2. Test Basic Execution & Correct Parameters ---
        let builder1 = JonmoBuilder::new()
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
        let builder2 = JonmoBuilder::new()
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
        let source_signal = SignalBuilder::from_resource::<SignalSource>()
            .map_in(|source: SignalSource| source.0)
            .dedupe();

        // The closure that will be passed to `on_signal_with_component`.
        // It adds the signal's value to the component's value.
        let mutator_closure = |mut component: Mut<TestComponent>, value: i32| {
            component.0 += value;
        };

        // --- 2. Test Basic Functionality ---
        let builder1 = JonmoBuilder::new()
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
        let builder_no_comp = JonmoBuilder::new()
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
        let builder2 = JonmoBuilder::new()
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
    fn test_signal_from_entity() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // A resource to capture the final output of the created signal chains.
        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<String>>>);

        app.init_resource::<TestOutput>();

        // The component that the signal chain will read from.
        #[derive(Component, Clone)]
        struct TestComponent(String);

        // A system to capture the final output of the signal chain.
        fn capturing_system(In(value): In<String>, output: Res<TestOutput>) {
            output.0.lock().unwrap().push(value);
        }

        // The factory closure that will be passed to `signal_from_entity`.
        // This represents the user's logic.
        let signal_chain_factory = |entity_signal: crate::signal::Source<Entity>| {
            // The user takes the entity signal...
            entity_signal
                // ...maps it to get a component from that entity...
                .map(|In(entity): In<Entity>, query: Query<&TestComponent>| query.get(entity).ok().cloned())
                // ...extracts the inner string...
                .map_in(|component: TestComponent| component.0)
                // ...and finally pipes it to our capturing system.
                .map(capturing_system)
        };

        // --- 2. Test Basic Functionality ---
        let builder1 = JonmoBuilder::new()
            .insert(TestComponent("Data A".to_string()))
            .signal_from_entity(signal_chain_factory);

        let entity1 = builder1.spawn(app.world_mut());
        app.update();

        // Verify the output.
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "Signal chain should have run once.");
        assert_eq!(output_guard[0], "Data A", "Signal chain produced incorrect output.");
        drop(output_guard);
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // --- 3. Test Multi-Entity Independence ---
        let builder2 = JonmoBuilder::new()
            .insert(TestComponent("Data B".to_string()))
            .signal_from_entity(signal_chain_factory);

        let entity2 = builder2.spawn(app.world_mut());
        app.update();

        // Verify that both signal chains ran independently. Order is not guaranteed.
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 2, "Both signal chains should have run.");
        let received_set: HashSet<String> = output_guard.iter().cloned().collect();
        let expected_set: HashSet<String> = ["Data A".to_string(), "Data B".to_string()].into();
        assert_eq!(
            received_set, expected_set,
            "Multi-entity test produced incorrect or incomplete output."
        );
        drop(output_guard);
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // --- 4. Test Automatic Cleanup ---
        // Despawn the first entity.
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // This update processes the despawn AND runs the graph for entity2.

        // Clear the output from the update call above to isolate the next frame's result.
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // Update again to run the signal graph, now that entity1's signal is gone.
        app.update();

        // Verify that only the signal for the second entity ran in this last frame.
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Only the signal for the remaining entity should have run."
        );
        assert_eq!(
            output_guard[0], "Data B",
            "The remaining signal produced incorrect output."
        );
        drop(output_guard);

        // Despawn the second entity.
        app.world_mut().entity_mut(entity2).despawn();
        app.update();
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // Update again. Nothing should run now.
        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "No signals should run after all entities are despawned."
        );
    }

    #[test]
    fn test_signal_from_component() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<String>>>);
        app.init_resource::<TestOutput>();

        #[derive(Component, Clone, Debug, PartialEq)]
        struct TestComponent(String);

        fn capturing_system(In(value): In<String>, output: Res<TestOutput>) {
            output.0.lock().unwrap().push(value);
        }

        let signal_chain_factory =
            |component_signal: crate::signal::Map<crate::signal::Source<Entity>, TestComponent>| {
                component_signal
                    .map_in(|component: TestComponent| component.0)
                    .dedupe()
                    .map(capturing_system)
            };

        // --- 2. Test Basic Functionality & Reactivity ---
        let builder1 = JonmoBuilder::new()
            .insert(TestComponent("Initial A".to_string()))
            .signal_from_component(signal_chain_factory);

        let entity1 = builder1.spawn(app.world_mut());

        app.update();
        let output = app.world_mut().resource_mut::<TestOutput>();
        let mut output_guard = output.0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard[0], "Initial A");
        output_guard.clear();
        drop(output_guard);

        app.update();
        assert!(app.world().resource::<TestOutput>().0.lock().unwrap().is_empty());

        app.world_mut().get_mut::<TestComponent>(entity1).unwrap().0 = "Updated A".to_string();
        app.update();
        let output = app.world_mut().resource_mut::<TestOutput>();
        let mut output_guard = output.0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard[0], "Updated A");
        output_guard.clear();
        drop(output_guard);

        // --- 3. Test Graceful Failure (Component Missing) ---
        let builder_no_comp = JonmoBuilder::new().signal_from_component(signal_chain_factory);
        let _entity_no_comp = builder_no_comp.spawn(app.world_mut());
        app.update();
        assert!(app.world().resource::<TestOutput>().0.lock().unwrap().is_empty());

        // --- 4. Test Multi-Entity Independence ---
        let builder2 = JonmoBuilder::new()
            .insert(TestComponent("Initial B".to_string()))
            .signal_from_component(signal_chain_factory);

        let entity2 = builder2.spawn(app.world_mut());

        // CORRECTED: To ensure both signals fire, we must change entity1's data again
        // so that its `dedupe` operator will allow the signal to pass.
        app.world_mut().get_mut::<TestComponent>(entity1).unwrap().0 = "Final A".to_string();
        app.update();

        let output = app.world_mut().resource_mut::<TestOutput>();
        let mut output_guard = output.0.lock().unwrap();
        // assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard.len(), 2, "Both signals should have run.");
        let received_set: HashSet<String> = output_guard.iter().cloned().collect();
        let expected_set: HashSet<String> = ["Final A".to_string(), "Initial B".to_string()].into();
        assert_eq!(received_set, expected_set);
        output_guard.clear();
        drop(output_guard);

        // --- 5. Test Automatic Cleanup ---
        app.world_mut().entity_mut(entity1).despawn();
        app.update();

        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        app.world_mut().get_mut::<TestComponent>(entity2).unwrap().0 = "Updated B".to_string();
        app.update();

        let output = app.world_mut().resource_mut::<TestOutput>();
        let mut output_guard = output.0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(
            output_guard.len(),
            1,
            "Only the signal for the remaining entity should run."
        );
        assert_eq!(output_guard[0], "Updated B");
        output_guard.clear();
    }

    #[test]
    fn test_signal_from_component_option() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // Resource to capture the final string output of the signal chains.
        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<String>>>);
        app.init_resource::<TestOutput>();

        // The component that the signal chain will read from.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TestComponent(String);

        // System to capture the final output.
        fn capturing_system(In(value): In<String>, output: Res<TestOutput>) {
            output.0.lock().unwrap().push(value);
        }

        // The factory closure that will be passed to `signal_from_component_option`.
        // It defines the signal logic after the component is read.
        let signal_chain_factory =
            |component_opt_signal: crate::signal::Map<crate::signal::Source<Entity>, Option<TestComponent>>| {
                component_opt_signal
                    .dedupe() // Use dedupe to test reactivity correctly.
                    .map_in(|opt_component: Option<TestComponent>| {
                        // This logic is key: we transform both Some and None cases into a string.
                        match opt_component {
                            Some(component) => format!("Some({})", component.0),
                            None => "None".to_string(),
                        }
                    })
                    .map(capturing_system)
            };

        // --- 2. Test Case: Component Present ---
        let builder_with_comp = JonmoBuilder::new()
            .insert(TestComponent("Data A".to_string()))
            .signal_from_component_option(signal_chain_factory);

        let entity_with_comp = builder_with_comp.spawn(app.world_mut());
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard[0], "Some(Data A)");
        output_guard.clear();
        drop(output_guard);

        // --- 3. Test Case: Component Missing ---
        // This is the main difference from `signal_from_component`. The chain should still run.
        let builder_without_comp = JonmoBuilder::new().signal_from_component_option(signal_chain_factory);

        let entity_without_comp = builder_without_comp.spawn(app.world_mut());
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Signal for entity without component should have run."
        );
        assert_eq!(
            output_guard[0], "None",
            "Signal should have received None for the missing component."
        );
        output_guard.clear();
        drop(output_guard);

        // --- 4. Test Case: Reactivity ---
        // Change the component on the first entity.
        app.world_mut().get_mut::<TestComponent>(entity_with_comp).unwrap().0 = "Updated A".to_string();
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "Signal should re-run on component change.");
        assert_eq!(output_guard[0], "Some(Updated A)");
        output_guard.clear();
        drop(output_guard);

        // Run again without changes. Due to `.dedupe()`, it shouldn't fire.
        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not run when component is unchanged due to dedupe."
        );
        drop(output_guard);

        // --- 5. Test Case: Automatic Cleanup ---
        // Despawn the entity with the component.
        app.world_mut().entity_mut(entity_with_comp).despawn();
        app.update(); // Process despawn commands.

        // To verify cleanup, we'll add the component to the *other* entity.
        // Only its signal should run, proving the first one was cleaned up.
        app.world_mut()
            .entity_mut(entity_without_comp)
            .insert(TestComponent("Data B".to_string()));
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Only the signal for the remaining entity should run."
        );
        assert_eq!(output_guard[0], "Some(Data B)");
        output_guard.clear();
        drop(output_guard);
    }

    #[test]
    fn test_signal_from_ancestor() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<String>>>);
        app.init_resource::<TestOutput>();

        #[derive(Component, Clone, Debug, PartialEq)]
        struct TestComponent(String);

        fn capturing_system(In(value): In<String>, output: Res<TestOutput>) {
            output.0.lock().unwrap().push(value);
        }

        // A reusable factory for the signal logic. It takes a signal of an ancestor's
        // entity, gets its TestComponent, and sends the inner string to the output.
        let signal_chain_factory =
            |ancestor_entity_signal: crate::signal::Map<crate::signal::Source<Entity>, Entity>| {
                ancestor_entity_signal
                    // Map the ancestor entity to its component. This will terminate if the
                    // ancestor doesn't have the component.
                    .map(|In(entity): In<Entity>, query: Query<&TestComponent>| query.get(entity).ok().cloned())
                    .map_in(|component: TestComponent| component.0) // Extract the string
                    .map(capturing_system)
            };

        // Create a 3-level entity hierarchy for testing: Grandparent -> Parent -> Child
        let grandparent = app.world_mut().spawn(TestComponent("Grandparent".to_string())).id();
        let parent = app.world_mut().spawn(TestComponent("Parent".to_string())).id();
        app.world_mut().entity_mut(grandparent).add_child(parent);

        // --- 2. Test Case: generations = 0 (Self) ---
        // The builder should target its own entity.
        let child_for_self_test = app
            .world_mut()
            .spawn(TestComponent("Child".to_string())) // Give the child its own component
            .id();
        app.world_mut().entity_mut(parent).add_child(child_for_self_test);

        let builder_self = JonmoBuilder::new().signal_from_ancestor(0, signal_chain_factory);
        builder_self.spawn_on_entity(app.world_mut(), child_for_self_test);
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard[0], "Child");
        output_guard.clear();
        drop(output_guard);
        app.world_mut().entity_mut(child_for_self_test).despawn();
        app.update();

        // --- 3. Test Case: generations = 1 (Parent) ---
        let child_for_parent_test = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child_for_parent_test);
        let builder_parent = JonmoBuilder::new().signal_from_ancestor(1, signal_chain_factory);
        builder_parent.spawn_on_entity(app.world_mut(), child_for_parent_test);
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard[0], "Parent");
        output_guard.clear();
        drop(output_guard);
        app.world_mut().entity_mut(child_for_parent_test).despawn();
        app.update();

        // --- 4. Test Case: generations = 2 (Grandparent) ---
        let child_for_gp_test = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child_for_gp_test);
        let builder_gp = JonmoBuilder::new().signal_from_ancestor(2, signal_chain_factory);
        builder_gp.spawn_on_entity(app.world_mut(), child_for_gp_test);
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1);
        assert_eq!(output_guard[0], "Grandparent");
        output_guard.clear();
        drop(output_guard);
        app.world_mut().entity_mut(child_for_gp_test).despawn();
        app.update();

        // --- 5. Test Case: Invalid Ancestor (Too many generations) ---
        let child_for_invalid_test = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child_for_invalid_test);
        let builder_invalid = JonmoBuilder::new().signal_from_ancestor(3, signal_chain_factory);
        builder_invalid.spawn_on_entity(app.world_mut(), child_for_invalid_test);
        app.update();

        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not run for an invalid ancestor."
        );
        drop(output_guard);
        app.world_mut().entity_mut(child_for_invalid_test).despawn();
        app.update();

        // --- 6. Test Case: Reactivity ---
        let child_for_reactivity_test = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child_for_reactivity_test);
        let builder_reactivity = JonmoBuilder::new().signal_from_ancestor(1, signal_chain_factory);
        builder_reactivity.spawn_on_entity(app.world_mut(), child_for_reactivity_test);
        app.update(); // Initial run
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear(); // Clear initial output

        // Modify the parent's component
        app.world_mut().get_mut::<TestComponent>(parent).unwrap().0 = "Parent Updated".to_string();
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "Signal did not re-run on ancestor change.");
        assert_eq!(output_guard[0], "Parent Updated");
        output_guard.clear();
        drop(output_guard);

        // --- 7. Test Case: Automatic Cleanup ---
        // Despawn the child entity that holds the signal
        app.world_mut().entity_mut(child_for_reactivity_test).despawn();
        app.update(); // Process despawn commands

        // Modify the parent's component again
        app.world_mut().get_mut::<TestComponent>(parent).unwrap().0 = "Parent Updated Again".to_string();
        app.update();

        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not run after its entity is despawned."
        );
    }

    #[test]
    fn test_signal_from_parent() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // A resource to capture the final output of the signal chains.
        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<String>>>);
        app.init_resource::<TestOutput>();

        // The component that the signal chain will read from the parent.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TestComponent(String);

        // A system to capture the final output of the signal chain.
        fn capturing_system(In(value): In<String>, output: Res<TestOutput>) {
            output.0.lock().unwrap().push(value);
        }

        // The factory closure that will be passed to `signal_from_parent`.
        // This represents the user's logic. It gets a signal for the parent entity,
        // reads its TestComponent, extracts the string, and sends it to be captured.
        let signal_chain_factory = |parent_entity_signal: crate::signal::Map<crate::signal::Source<Entity>, Entity>| {
            parent_entity_signal
                // Map the parent entity to its component. This will terminate if the
                // parent doesn't have the component.
                .map(|In(entity): In<Entity>, query: Query<&TestComponent>| query.get(entity).ok().cloned())
                .map_in(|component: TestComponent| component.0) // Extract the string
                .dedupe() // Add dedupe for robust reactivity testing
                .map(capturing_system)
        };

        // --- 2. Test Case: Basic Functionality & Reactivity ---
        let parent = app.world_mut().spawn(TestComponent("Parent A".to_string())).id();
        let child = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child);

        let builder_with_parent = JonmoBuilder::new().signal_from_parent(signal_chain_factory);
        builder_with_parent.spawn_on_entity(app.world_mut(), child);

        // Initial run
        app.update();
        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "Signal should run on initialization.");
        assert_eq!(output_guard[0], "Parent A");
        output_guard.clear();
        drop(output_guard);

        // Reactivity test
        app.world_mut().get_mut::<TestComponent>(parent).unwrap().0 = "Parent B".to_string();
        app.update();
        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Signal should re-run on parent component change."
        );
        assert_eq!(output_guard[0], "Parent B");
        output_guard.clear();
        drop(output_guard);

        // --- 3. Test Case: No Parent ---
        // A top-level entity with this signal should not run the chain.
        let _top_level_entity = JonmoBuilder::new()
            .signal_from_parent(signal_chain_factory)
            .spawn(app.world_mut());
        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not run for an entity with no parent."
        );
        drop(output_guard);

        // --- 4. Test Case: Parent Without Component ---
        // The signal chain should terminate gracefully if the parent is missing the component.
        let parent_no_comp = app.world_mut().spawn_empty().id();
        let child_of_plain_parent = app.world_mut().spawn_empty().id();
        app.world_mut()
            .entity_mut(parent_no_comp)
            .add_child(child_of_plain_parent);

        let builder_plain_parent = JonmoBuilder::new().signal_from_parent(signal_chain_factory);
        builder_plain_parent.spawn_on_entity(app.world_mut(), child_of_plain_parent);
        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not run if parent is missing the target component."
        );
        drop(output_guard);

        // --- 5. Test Case: Automatic Cleanup ---
        // Despawn the child entity that holds the signal.
        app.world_mut().entity_mut(child).despawn();
        app.update(); // Process despawn commands.

        // Modify the parent's component again. The signal should no longer be active.
        app.world_mut().get_mut::<TestComponent>(parent).unwrap().0 = "Parent C".to_string();
        app.update();

        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not run after its entity is despawned."
        );
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
        let source_signal = SignalBuilder::from_resource::<SignalSource>()
            .dedupe()
            .map_in(|source: SignalSource| {
                source.0.map(TargetComponent) // Option<String> -> Option<TargetComponent>
            });

        // --- 2. Test Initial Insertion ---
        let builder = JonmoBuilder::new().component_signal(source_signal.clone());
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
        let builder2 = JonmoBuilder::new().component_signal(source_signal.clone());
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
    fn test_signal_from_component_changed() {
        // --- 1. Setup ---
        // Create the standard test app with the JonmoPlugin.
        let mut app = create_test_app();

        // A resource to capture the final string output of the signal chains.
        #[derive(Resource, Default, Clone)]
        struct TestOutput(Arc<Mutex<Vec<String>>>);
        app.init_resource::<TestOutput>();

        // A system to capture the final output of the signal chain.
        fn capturing_system(In(value): In<String>, output: Res<TestOutput>) {
            output.0.lock().unwrap().push(value);
        }

        // The component that will be tracked for changes.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TrackedComponent(String);

        // The factory closure that will be passed to `signal_from_component_changed`.
        // This represents the user's logic: it takes the signal of the changed component,
        // extracts its inner string, and sends it to the capturing system.
        let signal_chain_factory =
            |component_signal: crate::signal::Map<crate::signal::Source<Entity>, TrackedComponent>| {
                component_signal
                    .map_in(|component: TrackedComponent| component.0)
                    // Note: `.dedupe()` is not needed here, as the `Changed<C>` filter
                    // in the implementation already prevents firing on unchanged frames.
                    .map(capturing_system)
            };

        // --- 2. Test Initial State & No-Change Frame ---
        let builder1 = JonmoBuilder::new()
            .insert(TrackedComponent("Initial".to_string()))
            .signal_from_component_changed(signal_chain_factory);

        let entity1 = builder1.spawn(app.world_mut());

        // Frame 1: The component is newly added (`Added<C>`), which counts as `Changed<C>`.
        // The signal should fire.
        app.update();
        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Signal should fire on initial component insertion."
        );
        assert_eq!(output_guard[0], "Initial");
        output_guard.clear();
        drop(output_guard);

        // Frame 2: No changes were made to the component. The signal should NOT fire.
        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not fire when component is unchanged."
        );
        drop(output_guard);

        // --- 3. Test Component Changed ---
        // Explicitly mutate the component to mark it as `Changed`.
        app.world_mut().get_mut::<TrackedComponent>(entity1).unwrap().0 = "Updated".to_string();

        // Frame 3: The component is now `Changed`. The signal should fire again.
        app.update();
        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "Signal should fire after component is mutated.");
        assert_eq!(output_guard[0], "Updated");
        output_guard.clear();
        drop(output_guard);

        // Frame 4: No changes were made in this frame. The signal should NOT fire.
        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not fire on the frame after a change."
        );
        drop(output_guard);

        // --- 4. Test Graceful Failure (Component Missing) ---
        // Spawn an entity with the signal but without the component.
        // The signal chain should terminate gracefully and not panic or produce output.
        let builder_no_comp = JonmoBuilder::new().signal_from_component_changed(signal_chain_factory);
        let _entity_no_comp = builder_no_comp.spawn(app.world_mut());

        app.update();
        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert!(
            output_guard.is_empty(),
            "Signal should not fire for an entity missing the component."
        );
        drop(output_guard);

        // --- 5. Test Multi-Entity Independence ---
        let builder2 = JonmoBuilder::new()
            .insert(TrackedComponent("Entity 2 Initial".to_string()))
            .signal_from_component_changed(signal_chain_factory);
        let entity2 = builder2.spawn(app.world_mut());

        // Frame 5: Only entity2's component is new. Only its signal should fire.
        app.update();
        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(output_guard.len(), 1, "Only the new entity's signal should fire.");
        assert_eq!(output_guard[0], "Entity 2 Initial");
        output_guard.clear();
        drop(output_guard);

        // Frame 6: Mutate both entities' components in the same frame.
        app.world_mut().get_mut::<TrackedComponent>(entity1).unwrap().0 = "Entity 1 Final".to_string();
        app.world_mut().get_mut::<TrackedComponent>(entity2).unwrap().0 = "Entity 2 Final".to_string();
        app.update();

        let output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            2,
            "Both signals should fire when both components change."
        );
        // Use a HashSet because the order of execution is not guaranteed.
        let received_set: HashSet<String> = output_guard.iter().cloned().collect();
        let expected_set: HashSet<String> = ["Entity 1 Final".to_string(), "Entity 2 Final".to_string()].into();
        assert_eq!(
            received_set, expected_set,
            "Multi-entity change produced incorrect output."
        );
        drop(output_guard);

        // --- 6. Test Automatic Cleanup ---
        // Despawn the first entity.
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // Process the despawn command.

        // Clear output from the last update cycle.
        app.world_mut().resource_mut::<TestOutput>().0.lock().unwrap().clear();

        // Mutate the remaining entity's component.
        app.world_mut().get_mut::<TrackedComponent>(entity2).unwrap().0 = "Entity 2 Post-Despawn".to_string();
        app.update();

        let mut output_guard = app.world().resource::<TestOutput>().0.lock().unwrap();
        assert_eq!(
            output_guard.len(),
            1,
            "Only the remaining entity's signal should fire after despawn."
        );
        assert_eq!(output_guard[0], "Entity 2 Post-Despawn");
        output_guard.clear();
        drop(output_guard);

        // Verify entity1 is truly gone.
        assert!(
            app.world().get_entity(entity1).is_err(),
            "Despawned entity should not exist."
        );
    }

    #[test]
    fn test_component_signal_from_entity() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The component that will be reactively added, updated, and removed.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TargetComponent(String);

        // A resource to act as the signal's external trigger. Its Option<String>
        // will be mapped to Option<TargetComponent>.
        #[derive(Resource, Default, Clone, PartialEq)]
        struct SignalSource(Option<String>);
        app.init_resource::<SignalSource>();

        // The factory closure that defines the signal logic.
        // It demonstrates that the closure correctly receives the entity signal
        // by using the entity's `Name` component in its logic.
        let signal_factory = |entity_signal: crate::signal::Source<Entity>| {
            // Combine the entity signal with the global source signal.
            entity_signal
                .combine(SignalBuilder::from_resource::<SignalSource>().dedupe())
                // The `.map` combinator gives us access to other `SystemParam`s, like `Query`.
                .map(
                    |In((entity, source)): In<(Entity, SignalSource)>, names: Query<&Name>| {
                        // Get the name from the specific entity this signal is for.
                        let name = names.get(entity).map_or("Unnamed", |n| n.as_str());
                        // Map the source data into the target component, including the entity's name.
                        source.0.map(|s| TargetComponent(format!("{s} for {name}")))
                    },
                )
        };

        // --- 2. Test Initial Insertion ---
        let builder1 = JonmoBuilder::new()
            .insert(Name::new("Entity1"))
            .component_signal_from_entity(signal_factory);

        // Set the source *before* spawning.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Initial".to_string());
        let entity1 = builder1.spawn(app.world_mut());

        // The signal chain runs on the first update after spawning.
        app.update();

        // Verify the component was inserted correctly.
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Initial for Entity1".to_string())),
            "Component should be inserted on first run."
        );

        // --- 3. Test Reactive Update ---
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Updated".to_string());
        app.update();

        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Updated for Entity1".to_string())),
            "Component should be updated on signal change."
        );

        // --- 4. Test Reactive Removal ---
        app.world_mut().resource_mut::<SignalSource>().0 = None;
        app.update();

        assert!(
            app.world().get::<TargetComponent>(entity1).is_none(),
            "Component should be removed when signal emits None."
        );

        // --- 5. Test Reactive Re-insertion ---
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Reinserted".to_string());
        app.update();

        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Reinserted for Entity1".to_string())),
            "Component should be re-inserted after being removed."
        );

        // --- 6. Test Multi-Entity Independence ---
        let builder2 = JonmoBuilder::new()
            .insert(Name::new("Entity2"))
            .component_signal_from_entity(signal_factory);

        let entity2 = builder2.spawn(app.world_mut());

        // Trigger an update for both.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Multi".to_string());
        app.update();

        // Verify both entities were updated correctly and independently.
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Multi for Entity1".to_string())),
            "Entity 1 should be updated in multi-entity test."
        );
        assert_eq!(
            app.world().get::<TargetComponent>(entity2),
            Some(&TargetComponent("Multi for Entity2".to_string())),
            "Entity 2 should be updated in multi-entity test."
        );

        // --- 7. Test Automatic Cleanup ---
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // Process despawn command.

        // Trigger another update.
        app.world_mut().resource_mut::<SignalSource>().0 = Some("Post-Despawn".to_string());
        app.update();

        // Verify the despawned entity is gone and its signal didn't run.
        assert!(
            app.world().get_entity(entity1).is_err(),
            "Despawned entity should not exist."
        );

        // Verify that only the second entity was updated.
        assert_eq!(
            app.world().get::<TargetComponent>(entity2),
            Some(&TargetComponent("Post-Despawn for Entity2".to_string())),
            "Only the remaining entity should have been updated."
        );
    }

    // Add the components required for the new test
    #[derive(Component, Clone, Debug, PartialEq)]
    struct SourceComponent(String);

    #[derive(Component, Clone, Debug, PartialEq)]
    struct TargetComponent(String);

    #[test]
    fn test_component_signal_from_ancestor() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The factory closure defines the reusable signal logic.
        // It takes a signal of an ancestor's entity, gets its `SourceComponent`,
        // and maps it to an `Option<TargetComponent>`. The final `Option` is crucial,
        // as `None` will trigger the removal of the `TargetComponent`.
        let signal_factory = |ancestor_signal: crate::signal::Map<crate::signal::Source<Entity>, Entity>| {
            ancestor_signal
                // Map the ancestor entity to its `SourceComponent`.
                // This produces `None` if the component doesn't exist, which is desired.
                .map(|In(entity): In<Entity>, query: Query<&SourceComponent>| query.get(entity).ok().cloned())
                // Map the `Option<SourceComponent>` to `Option<TargetComponent>`.
                .map_in(|opt_source: Option<SourceComponent>| opt_source.map(|source| TargetComponent(source.0)))
                // Dedupe to avoid unnecessary updates if the source value doesn't change.
                .dedupe()
        };

        // Create a 3-level entity hierarchy for testing different `generations`.
        let grandparent = app.world_mut().spawn(SourceComponent("Grandparent".to_string())).id();
        let parent = app.world_mut().spawn(SourceComponent("Parent".to_string())).id();
        app.world_mut().entity_mut(grandparent).add_child(parent);

        // --- 2. Test Parent (`generations = 1`) ---
        let child_from_parent = JonmoBuilder::new()
            .component_signal_from_ancestor(1, signal_factory)
            .spawn(app.world_mut());
        app.world_mut().entity_mut(parent).add_child(child_from_parent);

        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child_from_parent),
            Some(&TargetComponent("Parent".to_string())),
            "Child should get component from parent (gen 1)."
        );

        // --- 3. Test Grandparent (`generations = 2`) ---
        let child_from_gp = JonmoBuilder::new()
            .component_signal_from_ancestor(2, signal_factory)
            .spawn(app.world_mut());
        app.world_mut().entity_mut(parent).add_child(child_from_gp);

        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child_from_gp),
            Some(&TargetComponent("Grandparent".to_string())),
            "Child should get component from grandparent (gen 2)."
        );

        // --- 4. Test Self (`generations = 0`) ---
        let child_from_self = JonmoBuilder::new()
            .insert(SourceComponent("Self".to_string()))
            .component_signal_from_ancestor(0, signal_factory)
            .spawn(app.world_mut());
        app.world_mut().entity_mut(parent).add_child(child_from_self);

        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child_from_self),
            Some(&TargetComponent("Self".to_string())),
            "Child should get component from itself (gen 0)."
        );

        // --- 5. Test Reactivity (Update, Remove, Re-add) ---
        // Use the `child_from_parent` for this test.

        // Update the parent's source component.
        app.world_mut().get_mut::<SourceComponent>(parent).unwrap().0 = "Parent Updated".to_string();
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child_from_parent),
            Some(&TargetComponent("Parent Updated".to_string())),
            "Child's component should update when parent's component changes."
        );

        // Remove the source component from the parent. This should cause the signal to emit `None`.
        app.world_mut().entity_mut(parent).remove::<SourceComponent>();
        app.update();
        assert!(
            app.world().get::<TargetComponent>(child_from_parent).is_none(),
            "Child's component should be removed when parent's source component is removed."
        );

        // Re-add the source component to the parent.
        app.world_mut()
            .entity_mut(parent)
            .insert(SourceComponent("Parent Restored".to_string()));
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child_from_parent),
            Some(&TargetComponent("Parent Restored".to_string())),
            "Child's component should be re-added when parent's source component is re-added."
        );

        // --- 6. Test Failure (Invalid Ancestor) ---
        // Create a top-level builder (no parent) that asks for an ancestor.
        // The signal chain should terminate gracefully.
        let top_level_child = JonmoBuilder::new()
            .component_signal_from_ancestor(1, signal_factory)
            .spawn(app.world_mut());

        app.update();
        assert!(
            app.world().get::<TargetComponent>(top_level_child).is_none(),
            "Top-level child should not have target component when asking for a non-existent parent."
        );

        // --- 7. Test Automatic Cleanup ---
        // Despawn the child entity.
        app.world_mut().entity_mut(child_from_parent).despawn();
        app.update(); // Process the despawn command.

        // Verify it's gone.
        assert!(
            app.world().get_entity(child_from_parent).is_err(),
            "Child entity should be despawned."
        );

        // Mutate the parent's component again. If the signal was not cleaned up,
        // this could panic when trying to access the despawned child.
        app.world_mut().get_mut::<SourceComponent>(parent).unwrap().0 = "Post-Despawn Update".to_string();
        app.update();

        // The test passes if the previous update didn't panic.
    }

    #[test]
    fn test_component_signal_from_parent() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The factory closure defines the reusable signal logic.
        // It takes a signal of a parent's entity, gets its `SourceComponent`,
        // and maps it to an `Option<TargetComponent>`.
        let signal_factory = |parent_signal: crate::signal::Map<crate::signal::Source<Entity>, Entity>| {
            parent_signal
                // Map the parent entity to its `SourceComponent`.
                .map(|In(entity): In<Entity>, query: Query<&SourceComponent>| query.get(entity).ok().cloned())
                // Map the `Option<SourceComponent>` to `Option<TargetComponent>`.
                .map_in(|opt_source: Option<SourceComponent>| opt_source.map(|source| TargetComponent(source.0)))
                .dedupe()
        };

        // Create a parent-child hierarchy.
        let parent = app.world_mut().spawn(SourceComponent("Parent Data".to_string())).id();

        // --- 2. Test Basic Functionality ---
        let child = JonmoBuilder::new()
            .component_signal_from_parent(signal_factory)
            .spawn(app.world_mut());
        app.world_mut().entity_mut(parent).add_child(child);

        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child),
            Some(&TargetComponent("Parent Data".to_string())),
            "Child should get component from its direct parent."
        );

        // --- 3. Test Reactivity (Update, Remove, Re-add) ---
        // Update the parent's source component.
        app.world_mut().get_mut::<SourceComponent>(parent).unwrap().0 = "Parent Updated".to_string();
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child),
            Some(&TargetComponent("Parent Updated".to_string())),
            "Child's component should update when parent's component changes."
        );

        // Remove the source component from the parent.
        app.world_mut().entity_mut(parent).remove::<SourceComponent>();
        app.update();
        assert!(
            app.world().get::<TargetComponent>(child).is_none(),
            "Child's component should be removed when parent's source component is removed."
        );

        // Re-add the source component to the parent.
        app.world_mut()
            .entity_mut(parent)
            .insert(SourceComponent("Parent Restored".to_string()));
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(child),
            Some(&TargetComponent("Parent Restored".to_string())),
            "Child's component should be re-added when parent's source component is re-added."
        );

        // --- 4. Test Failure Case (No Parent) ---
        // Create a top-level builder (no parent). The signal should not run.
        let top_level_child = JonmoBuilder::new()
            .component_signal_from_parent(signal_factory)
            .spawn(app.world_mut());

        app.update();
        assert!(
            app.world().get::<TargetComponent>(top_level_child).is_none(),
            "Top-level child should not have target component as it has no parent."
        );

        // --- 5. Test Automatic Cleanup ---
        // Despawn the child entity.
        app.world_mut().entity_mut(child).despawn();
        app.update(); // Process the despawn command.

        // Verify it's gone.
        assert!(
            app.world().get_entity(child).is_err(),
            "Child entity should be despawned."
        );

        // Mutate the parent's component again. If the signal was not cleaned up,
        // this could panic when trying to access the despawned child.
        app.world_mut().get_mut::<SourceComponent>(parent).unwrap().0 = "Post-Despawn Update".to_string();
        app.update();

        // The test passes if the previous update didn't panic.
    }

    #[test]
    fn test_component_signal_from_component() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The component that acts as the data source for the signal.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct SourceComponent(i32);

        // The component that will be reactively managed by the signal's output.
        #[derive(Component, Clone, Debug, PartialEq)]
        struct TargetComponent(String);

        // The factory closure that defines the signal logic.
        // It receives a signal of the `SourceComponent` and transforms its value
        // into an `Option<TargetComponent>`.
        let signal_factory = |source_signal: crate::signal::Map<crate::signal::Source<Entity>, SourceComponent>| {
            source_signal
                // The signal from `component_signal_from_component` is already deduplicated
                // by the `Changed<C>` filter in its upstream `signal_from_component_changed`,
                // but adding one here is a good practice for robustness.
                .dedupe()
                // Map the source value to the target component's value.
                .map_in(|source: SourceComponent| TargetComponent(format!("Value is {}", source.0)))
                // The final signal must produce an `Option`.
                .map_in(Some)
        };

        // --- 2. Test Initial State & Reactivity ---
        let builder1 = JonmoBuilder::new()
            .insert(SourceComponent(10))
            .component_signal_from_component(signal_factory);

        let entity1 = builder1.spawn(app.world_mut());

        // Frame 1: The `SourceComponent` is newly added, which counts as `Changed`.
        // The signal should fire and insert the `TargetComponent`.
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Value is 10".to_string())),
            "TargetComponent should be inserted on initial spawn."
        );

        // Frame 2: No change to `SourceComponent`. The signal should not fire again.
        // We can test this by trying to remove the TargetComponent and seeing if it
        // gets re-added. It should not.
        app.world_mut().entity_mut(entity1).remove::<TargetComponent>();
        app.update();
        assert!(
            app.world().get::<TargetComponent>(entity1).is_none(),
            "TargetComponent should not be re-inserted on an unchanged frame."
        );

        // Frame 3: Mutate the `SourceComponent`. The signal should fire and update the
        // `TargetComponent`.
        app.world_mut().get_mut::<SourceComponent>(entity1).unwrap().0 = 20;
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Value is 20".to_string())),
            "TargetComponent should be updated after SourceComponent changes."
        );

        // --- 3. Test Failure Case: No SourceComponent ---
        // Spawn an entity with the signal but without the `SourceComponent`.
        // The signal chain should terminate gracefully and not insert the `TargetComponent`.
        let builder_no_source = JonmoBuilder::new().component_signal_from_component(signal_factory);
        let entity_no_source = builder_no_source.spawn(app.world_mut());

        app.update();
        assert!(
            app.world().get::<TargetComponent>(entity_no_source).is_none(),
            "TargetComponent should not be inserted if SourceComponent is missing."
        );

        // --- 4. Test Multi-Entity Independence ---
        let builder2 = JonmoBuilder::new()
            .insert(SourceComponent(100))
            .component_signal_from_component(signal_factory);
        let entity2 = builder2.spawn(app.world_mut());

        // Frame 4: Update both entities.
        app.world_mut().get_mut::<SourceComponent>(entity1).unwrap().0 = 30;
        // The update for entity2 happens implicitly on spawn.
        app.update();

        assert_eq!(
            app.world().get::<TargetComponent>(entity1),
            Some(&TargetComponent("Value is 30".to_string())),
            "Entity 1 failed to update in multi-entity test."
        );
        assert_eq!(
            app.world().get::<TargetComponent>(entity2),
            Some(&TargetComponent("Value is 100".to_string())),
            "Entity 2 failed to initialize in multi-entity test."
        );

        // --- 5. Test Automatic Cleanup ---
        app.world_mut().entity_mut(entity1).despawn();
        app.update(); // Process despawn command.

        // Mutate the `SourceComponent` of the remaining entity.
        app.world_mut().get_mut::<SourceComponent>(entity2).unwrap().0 = 200;
        app.update();

        // The test passes if the app doesn't panic. We also verify the state.
        assert!(
            app.world().get_entity(entity1).is_err(),
            "Despawned entity should not exist."
        );
        assert_eq!(
            app.world().get::<TargetComponent>(entity2),
            Some(&TargetComponent("Value is 200".to_string())),
            "Remaining entity failed to update after other was despawned."
        );
    }

    #[test]
    fn test_component_signal_from_component_option() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        // The factory closure defines the signal logic. It receives a signal that
        // *always* emits an `Option<SourceComponent>`. This is the key difference
        // from `component_signal_from_component`.
        let signal_factory =
            |source_opt_signal: crate::signal::Map<crate::signal::Source<Entity>, Option<SourceComponent>>| {
                source_opt_signal
                    .dedupe()
                    // Map the Option<SourceComponent> to an Option<TargetComponent>.
                    // This logic explicitly handles both the Some and None cases.
                    .map_in(|opt_source: Option<SourceComponent>| match opt_source {
                        // If the source exists, create a target component with its data.
                        Some(source) => Some(TargetComponent(format!("Source: {}", source.0))),
                        // If the source is missing, create a target component with a default value.
                        None => Some(TargetComponent("Source: None".to_string())),
                    })
            };

        // --- 2. Test Case: Source Component is PRESENT ---
        let builder_with_comp = JonmoBuilder::new()
            .insert(SourceComponent("Data A".to_string()))
            .component_signal_from_component_option(signal_factory);

        let entity_with_comp = builder_with_comp.spawn(app.world_mut());
        app.update();

        // The factory should have received `Some(SourceComponent)` and produced the corresponding target.
        assert_eq!(
            app.world().get::<TargetComponent>(entity_with_comp),
            Some(&TargetComponent("Source: Data A".to_string())),
            "TargetComponent should be created from the present SourceComponent."
        );

        // --- 3. Test Case: Source Component is MISSING ---
        // This is the primary test for this function's unique behavior.
        let builder_without_comp = JonmoBuilder::new().component_signal_from_component_option(signal_factory);

        let entity_without_comp = builder_without_comp.spawn(app.world_mut());
        app.update();

        // The factory should have received `None` and produced the default target.
        assert_eq!(
            app.world().get::<TargetComponent>(entity_without_comp),
            Some(&TargetComponent("Source: None".to_string())),
            "TargetComponent should be created from the missing SourceComponent (None path)."
        );

        // --- 4. Test Reactivity of Component PRESENCE ---
        // We will use `entity_with_comp` for this test.

        // Step 4a: Remove the source component. The signal should re-run and update the target.
        app.world_mut().entity_mut(entity_with_comp).remove::<SourceComponent>();
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(entity_with_comp),
            Some(&TargetComponent("Source: None".to_string())),
            "TargetComponent should update to 'None' state after source is removed."
        );

        // Step 4b: Re-add the source component. The signal should re-run again.
        app.world_mut()
            .entity_mut(entity_with_comp)
            .insert(SourceComponent("Data B".to_string()));
        app.update();
        assert_eq!(
            app.world().get::<TargetComponent>(entity_with_comp),
            Some(&TargetComponent("Source: Data B".to_string())),
            "TargetComponent should update after source is re-added."
        );

        // Step 4c: No change frame. Due to `.dedupe()`, the signal should not cause a write.
        // (This is harder to assert directly, but is an implicit part of the logic).
        app.update(); // Run again without changes.
        assert_eq!(
            app.world().get::<TargetComponent>(entity_with_comp),
            Some(&TargetComponent("Source: Data B".to_string())),
            "TargetComponent should remain unchanged on a no-op frame."
        );

        // --- 5. Test Automatic Cleanup ---
        app.world_mut().entity_mut(entity_with_comp).despawn();
        app.update();

        assert!(
            app.world().get_entity(entity_with_comp).is_err(),
            "Entity should be despawned."
        );

        // The test passes if the app doesn't panic on subsequent updates, proving
        // the signal isn't trying to access a despawned entity.
        app.update();
    }

    #[test]
    fn test_child() {
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
        let child_builder_simple = JonmoBuilder::new().insert((ChildCompA, Name::new("SimpleChild")));
        let parent_builder_simple = JonmoBuilder::new().insert(ParentComp).child(child_builder_simple);

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
        let parent_builder_chained = JonmoBuilder::new()
            .insert(ParentComp)
            .child(JonmoBuilder::new().insert(ChildCompA))
            .child(JonmoBuilder::new().insert(ChildCompB));

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
        let grandchild_builder = JonmoBuilder::new().insert(Name::new("Grandchild"));
        let child_builder_nested = JonmoBuilder::new().insert(Name::new("Child")).child(grandchild_builder);
        let grandparent_builder = JonmoBuilder::new()
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

        let child_a = JonmoBuilder::new().insert(ChildCompA);
        let child_b = JonmoBuilder::new().insert(ChildCompB);
        let child_c = JonmoBuilder::new().insert(ChildCompC);
        let child_d = JonmoBuilder::new().insert(ChildCompD);
        let child_e = JonmoBuilder::new().insert(ChildCompE);

        let parent_builder_mixed = JonmoBuilder::new()
            .insert(ParentComp)
            .child(child_a) // Block 0, offset 0
            .children([child_b.clone(), child_c.clone()]) // Block 1, offset 1
            .child(child_d) // Block 2, offset 3
            .child_signal(
                // Block 3, offset 4
                SignalBuilder::from_resource::<SignalTrigger>().map_in::<Option<JonmoBuilder>, _, _>(
                    move |trigger: SignalTrigger| if trigger.0 { Some(child_e.clone()) } else { None },
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
        let child_builder_with_signal = JonmoBuilder::new().component_signal_from_parent(|parent_signal| {
            parent_signal
                .component::<SourceComp>()
                .map_in(|source: SourceComp| Some(TargetComp(source.0 * 2)))
        });

        let parent_builder_with_signal = JonmoBuilder::new()
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
        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalSource>();

        // A factory function to create reactive child builders based on a number
        let reactive_child_factory = |id: u32| JonmoBuilder::new().insert(ReactiveChild(id));

        // The signal that maps the resource to an Option<JonmoBuilder>
        let source_signal = SignalBuilder::from_resource::<SignalSource>()
            .dedupe()
            .map_in(move |source: SignalSource| source.0.map(reactive_child_factory));

        // --- 2. Build the Parent Entity ---
        // This builder has static children sandwiching the reactive one to test ordering.
        let parent_builder = JonmoBuilder::new()
            .insert(ParentComp)
            .child(JonmoBuilder::new().insert(StaticChildBefore)) // Child in block 0
            .child_signal(source_signal) // Child in block 1
            .child(JonmoBuilder::new().insert(StaticChildAfter)); // Child in block 2

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

    #[test]
    fn test_children() {
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
                JonmoBuilder::new().insert(ChildCompA),
                JonmoBuilder::new().insert(ChildCompB),
                JonmoBuilder::new().insert(ChildCompC),
            ];
            let parent_builder = JonmoBuilder::new().insert(ParentComp).children(child_builders);

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
            let parent_builder_empty = JonmoBuilder::new()
                .insert(ParentComp)
                .children(vec![] as Vec<JonmoBuilder>);
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
            let complex_child_builder = JonmoBuilder::new()
                .insert(ChildCompD)
                .child(JonmoBuilder::new().insert(GrandchildComp)) // Nested child
                .component_signal_from_parent(|parent_signal| {
                    // Signal reading from parent
                    parent_signal
                        .component::<SourceComp>()
                        .map_in(|source: SourceComp| Some(TargetComp(source.0 * 10)))
                });

            let parent_builder_complex = JonmoBuilder::new()
                .insert((ParentComp, SourceComp(5)))
                .children(vec![complex_child_builder]);

            let parent_entity_complex = parent_builder_complex.spawn(app.world_mut());
            app.update();

            let complex_children = app.world().get::<Children>(parent_entity_complex).unwrap();
            let complex_child_entity = complex_children[0];

            // Verify signal ran correctly
            assert_eq!(
                app.world().get::<TargetComp>(complex_child_entity),
                Some(&TargetComp(50))
            );

            // Verify nested hierarchy
            let grandchild_entity = app.world().get::<Children>(complex_child_entity).unwrap()[0];
            assert!(app.world().entity(grandchild_entity).contains::<GrandchildComp>());

            // Test reactivity
            app.world_mut().get_mut::<SourceComp>(parent_entity_complex).unwrap().0 = 7;
            app.update();
            assert_eq!(
                app.world().get::<TargetComp>(complex_child_entity),
                Some(&TargetComp(70)),
                "Signal did not react to parent's component change."
            );

            // Test cleanup
            app.world_mut().entity_mut(parent_entity_complex).despawn();
            app.update();

            assert!(
                app.world().get_entity(complex_child_entity).is_err(),
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

            let parent_builder = JonmoBuilder::new()
                .insert(ParentComp)
                .child(JonmoBuilder::new().insert(ChildCompA)) // Block 0, size 1
                .children([
                    // Block 1, size 2
                    JonmoBuilder::new().insert(ChildCompB),
                    JonmoBuilder::new().insert(ChildCompC),
                ])
                .child_signal(
                    // Block 2, size 0 -> 1
                    SignalBuilder::from_resource::<SignalTrigger>().map_in(|trigger: SignalTrigger| {
                        if trigger.0 {
                            Some(JonmoBuilder::new().insert(ChildCompD))
                        } else {
                            None
                        }
                    }),
                )
                .children(vec![
                    // Block 3, size 3
                    JonmoBuilder::new().insert(ChildCompE),
                    JonmoBuilder::new().insert(ChildCompF),
                    JonmoBuilder::new().insert(ChildCompG),
                ])
                .child(JonmoBuilder::new().insert(ChildCompH)); // Block 4, size 1

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
        // --- 1. SETUP ---
        let mut app = create_test_app();
        let source_vec = MutableVec::from((app.world_mut(), [10u32, 20u32]));

        // A factory function to create a simple JonmoBuilder for a reactive child.
        let child_builder_factory = |id: u32| JonmoBuilder::new().insert(ReactiveChild(id));

        // The SignalVec that will drive the children.
        let children_signal = source_vec.signal_vec().map_in(child_builder_factory);

        // The parent builder, with static children sandwiching the reactive ones to test ordering.
        let parent_builder = JonmoBuilder::new()
            .insert(ParentComp)
            .child(JonmoBuilder::new().insert(StaticChildBefore))
            .children_signal_vec(children_signal)
            .child(JonmoBuilder::new().insert(StaticChildAfter));

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
}
