use crate::tree::*;
use bevy_ecs::{prelude::*, system::SystemId};
use bevy_log::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, Typed};
use std::{marker::PhantomData, sync::Arc};

// --- Type Aliases for Boxed Registration Functions ---

/// Type alias for the boxed source registration function.
type SourceRegisterFn<O> =
    dyn Fn(&mut World) -> SystemId<In<()>, Option<O>> + Send + Sync + 'static;

/// Type alias for the boxed combine registration function.
type CombineRegisterFn<O1, O2> = dyn Fn(
        &mut World,
        UntypedSystemId,
        UntypedSystemId,
    ) -> (
        SystemId<In<(Option<O1>, Option<O2>)>, Option<(O1, O2)>>,
        Vec<UntypedSystemId>,
    ) + Send
    + Sync
    + 'static;

/// Type alias for the boxed map registration function.
type MapRegisterFn = dyn Fn(
        &mut World,
        UntypedSystemId, // Previous system's entity ID
    ) -> UntypedSystemId // Registered map system's entity ID
    + Send
    + Sync
    + 'static;

/// Internal trait handling the registration logic for different signal node types.
/// **Note:** This trait is intended for internal use only.
pub trait SignalBuilderInternal: Send + Sync + 'static {
    /// The logical output type of this signal node.
    type Item: Send + Sync + 'static;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a `Vec<UntypedSystemId>` containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    fn register(&self, world: &mut World) -> Vec<UntypedSystemId>;
}

/// Represents a value that changes over time.
///
/// Signals are the core building block for reactive data flow. They are typically
/// created using methods on the [`SignalBuilder`] struct (e.g., [`SignalBuilder::from_component`])
/// and then transformed or combined using methods from the [`SignalExt`] trait.
pub trait Signal: Send + Sync + 'static {
    /// The type of value produced by this signal.
    type Item: Send + Sync + 'static;
}

/// Struct representing a source node in the signal chain definition. Implements [`Signal`].
#[derive(Clone)]
pub struct Source<O>
where
    O: Send + Sync + 'static,
{
    /// The type-erased function responsible for registering the source system.
    pub(crate) register_fn: Arc<SourceRegisterFn<O>>,
    _marker: PhantomData<O>,
}

// Implement internal registration logic
impl<O> SignalBuilderInternal for Source<O>
where
    O: Send + Sync + 'static,
{
    type Item = O;

    fn register(&self, world: &mut World) -> Vec<UntypedSystemId> {
        let system_id = (self.register_fn)(world);
        vec![system_id.entity()]
    }
}

// Implement the public Signal trait
impl<O> Signal for Source<O>
where
    O: Send + Sync + 'static,
{
    type Item = O;
}

/// Struct representing a map node in the signal chain definition. Implements [`Signal`].
/// Generic only over the previous signal (`Prev`) and the output type (`U`).
pub struct Map<Prev, U>
where
    Prev: SignalBuilderInternal,
    U: Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: Send + Sync + 'static,
{
    pub(crate) prev_signal: Prev,
    pub(crate) register_fn: Arc<MapRegisterFn>,
    _marker: PhantomData<U>,
}

// Add Clone implementation for Map
impl<Prev, U> Clone for Map<Prev, U>
where
    Prev: SignalBuilderInternal + Clone,
    U: Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            prev_signal: self.prev_signal.clone(),
            register_fn: self.register_fn.clone(),
            _marker: PhantomData,
        }
    }
}

// Implement internal registration logic for Map<Prev, U>
impl<Prev, U> SignalBuilderInternal for Map<Prev, U>
where
    Prev: SignalBuilderInternal,
    U: FromReflect + Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: FromReflect + Send + Sync + 'static,
{
    type Item = U;

    fn register(&self, world: &mut World) -> Vec<UntypedSystemId> {
        let mut prev_ids = self.prev_signal.register(world);
        if let Some(&prev_last_id_entity) = prev_ids.last() {
            let new_system_entity = (self.register_fn)(world, prev_last_id_entity);
            prev_ids.push(new_system_entity);
        } else {
            error!("Map signal parent registration returned empty ID list.");
        }
        prev_ids
    }
}

// Implement the public Signal trait for Map<Prev, U>
impl<Prev, U> Signal for Map<Prev, U>
where
    Prev: SignalBuilderInternal,
    U: Send + Sync + 'static,
    <Prev as SignalBuilderInternal>::Item: Send + Sync + 'static,
    Prev: Signal<Item = <Prev as SignalBuilderInternal>::Item>,
{
    type Item = U;
}

/// Struct representing a combine node in the signal chain definition. Implements [`Signal`].
pub struct Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal,
    Right: Signal + SignalBuilderInternal,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
{
    pub(crate) left_signal: Left,
    pub(crate) right_signal: Right,
    pub(crate) register_fn: Arc<CombineRegisterFn<<Left as Signal>::Item, <Right as Signal>::Item>>,
    _marker: PhantomData<(<Left as Signal>::Item, <Right as Signal>::Item)>,
}

// Add Clone implementation for Combine
impl<Left, Right> Clone for Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal + Clone,
    Right: Signal + SignalBuilderInternal + Clone,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
    Arc<CombineRegisterFn<<Left as Signal>::Item, <Right as Signal>::Item>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            left_signal: self.left_signal.clone(),
            right_signal: self.right_signal.clone(),
            register_fn: self.register_fn.clone(),
            _marker: PhantomData,
        }
    }
}

// Implement internal registration logic
impl<Left, Right> SignalBuilderInternal for Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal,
    Right: Signal + SignalBuilderInternal,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
{
    type Item = (<Left as Signal>::Item, <Right as Signal>::Item);

    fn register(&self, world: &mut World) -> Vec<UntypedSystemId> {
        let mut left_ids = self.left_signal.register(world);
        let mut right_ids = self.right_signal.register(world);

        let combined_ids = if let (Some(&left_last_id), Some(&right_last_id)) =
            (left_ids.last(), right_ids.last())
        {
            let (_combine_system_id, combine_node_ids) =
                (self.register_fn)(world, left_last_id, right_last_id);
            combine_node_ids
        } else {
            error!("CombineSignal parent registration returned empty ID list(s).");
            Vec::new()
        };

        left_ids.append(&mut right_ids);
        left_ids.extend(combined_ids);
        left_ids
    }
}

// Implement the public Signal trait
impl<Left, Right> Signal for Combine<Left, Right>
where
    Left: Signal + SignalBuilderInternal,
    Right: Signal + SignalBuilderInternal,
    <Left as Signal>::Item: Send + Sync + 'static,
    <Right as Signal>::Item: Send + Sync + 'static,
{
    type Item = (<Left as Signal>::Item, <Right as Signal>::Item);
}

/// Handle returned by [`SignalExt::register`] used for cleaning up the registered signal chain.
///
/// Contains the list of all system entities created by the specific `register` call
/// that produced this handle. Dropping the handle does *not* automatically clean up.
/// Use the [`cleanup`](SignalHandle::cleanup) method for explicit cleanup.
#[derive(Clone, Debug)]
pub struct SignalHandle(Vec<UntypedSystemId>);

impl SignalHandle {
    /// Creates a new SignalHandle.
    /// This is crate-public to allow construction from other modules.
    pub(crate) fn new(ids: Vec<UntypedSystemId>) -> Self {
        Self(ids)
    }

    /// Decrements the reference count for each system associated with this handle.
    /// If a system's reference count reaches zero, it removes the system's node
    /// from the internal signal graph and despawns its associated entity from the Bevy `World`.
    pub fn cleanup(self, world: &mut World) {
        if let Some(mut propagator) = world.remove_resource::<SignalPropagator>() {
            let mut nodes_to_remove = Vec::new();
            for system_entity in &self.0 {
                if let Some(metadata) = world.get::<SignalNodeMetadata>(*system_entity) {
                    if metadata.decrement() == 1 {
                        nodes_to_remove.push(*system_entity);
                    }
                } else {
                    warn!(
                        "SignalNodeMetadata not found for system {:?} during cleanup.",
                        system_entity
                    );
                }
            }

            for entity_to_remove in nodes_to_remove {
                propagator.remove_graph_node(world, entity_to_remove);
            }

            world.insert_resource(propagator);
        } else {
            warn!(
                "SignalPropagator not found during cleanup. Cannot decrement reference counts or remove nodes."
            );
            // Fallback: Try to despawn entities directly if propagator is missing
            for system_entity in self.0 {
                if let Ok(entity_mut) = world.get_entity_mut(system_entity) {
                    entity_mut.despawn();
                }
            }
        }
    }
}

/// Helper to create a source signal node. Wraps the registration function
/// in the `Source` struct, boxing the function.
pub(crate) fn create_source<F, O>(register_fn: F) -> Source<O>
where
    O: Send + Sync + 'static,
    F: Fn(&mut World) -> SystemId<In<()>, Option<O>> + Send + Sync + 'static,
{
    Source {
        register_fn: Arc::new(register_fn),
        _marker: std::marker::PhantomData,
    }
}

/// Provides static methods for creating new signal chains (source signals).
/// Use methods like [`SignalBuilder::from_component`] or [`SignalBuilder::from_system`]
/// to start building a signal chain.
pub struct SignalBuilder;

// Static methods to start signal chains, now associated with SignalBuilder struct
impl SignalBuilder {
    /// Creates a signal chain starting from a custom Bevy system.
    ///
    /// The provided system should take `In<()>` and return `Option<O>`.
    /// This system will be registered as a root node for signal propagation.
    /// The system `F` must be `Clone` as it's captured for registration.
    pub fn from_system<O, M, F>(system: F) -> Source<O>
    where
        O: FromReflect + Send + Sync + 'static,
        F: IntoSystem<In<()>, Option<O>, M> + Send + Sync + Clone + 'static,
        M: Send + Sync + 'static,
    {
        let register_fn = move |world: &mut World| {
            let system_id = register_signal::<(), O, M>(world, system.clone());
            mark_signal_root(world, system_id.entity());
            system_id
        };
        create_source(register_fn)
    }

    /// Creates a signal chain starting from a specific entity.
    ///
    /// The signal will emit the `Entity` ID whenever the propagation starts from this source.
    /// Useful for chains that operate on or react to changes related to this entity.
    /// Internally uses [`SignalBuilder::from_system`] with [`entity_root`].
    pub fn from_entity(entity: Entity) -> Source<Entity> {
        Self::from_system(entity_root(entity))
    }

    /// Creates a signal chain that starts by observing changes to a specific component `C`
    /// on a given `entity`.
    ///
    /// The signal emits the new value of the component `C` whenever it changes on the entity.
    /// Requires the component `C` to implement `Component`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    /// Internally uses [`SignalBuilder::from_system`].
    pub fn from_component<C>(entity: Entity) -> Source<C>
    where
        C: Component + FromReflect + Clone + Send + Sync + 'static,
    {
        let component_query_system =
            move |_: In<()>, query: Query<&'static C, Changed<C>>| query.get(entity).ok().cloned();
        Self::from_system(component_query_system)
    }

    /// Creates a signal chain that starts by observing changes to a specific resource `R`.
    ///
    /// The signal emits the new value of the resource `R` whenever it changes.
    /// Requires the resource `R` to implement `Resource`, `FromReflect`, `Clone`, `Send`, `Sync`, and `'static`.
    /// Internally uses [`SignalBuilder::from_system`].
    pub fn from_resource<R>() -> Source<R>
    where
        R: Resource + FromReflect + Clone + Send + Sync + 'static,
    {
        let resource_query_system = move |_: In<()>, res: Res<R>| {
            if res.is_changed() {
                Some(res.clone())
            } else {
                None
            }
        };
        Self::from_system(resource_query_system)
    }
}

/// Extension trait providing combinator methods for types implementing [`Signal`],
/// [`SignalBuilderInternal`], and [`Clone`].
pub trait SignalExt: Signal + SignalBuilderInternal + Clone {
    /// Appends a transformation step to the signal chain using a Bevy system.
    ///
    /// The provided `system` takes the output `Item` of the previous step (wrapped in `In<Item>`)
    /// and returns an `Option<U>`. If it returns `Some(U)`, `U` is propagated to the next step.
    /// If it returns `None` (or [`TERMINATE`]), propagation along this branch stops for the frame.
    ///
    /// The system `F` must be `Clone` as it's captured for registration.
    /// Returns a [`Map`] signal node.
    fn map<U, M, F>(self, system: F) -> Map<Self, U>
    where
        Self: Sized,
        <Self as SignalBuilderInternal>::Item: FromReflect + Send + Sync + 'static,
        U: FromReflect + Send + Sync + 'static,
        F: IntoSystem<In<<Self as SignalBuilderInternal>::Item>, Option<U>, M>
            + Send
            + Sync
            + Clone // <-- Re-added Clone bound
            + 'static,
        M: Send + Sync + 'static;

    /// Combines this signal with another signal (`other`), producing a new signal that emits
    /// a tuple `(Self::Item, S2::Item)` of the outputs of both signals.
    ///
    /// The new signal emits a value only when *both* input signals have emitted at least one
    /// value since the last combined emission. It caches the latest value from each input.
    /// Both `self` and `other` must implement `Clone`.
    /// Returns a [`Combine`] signal node.
    fn combine_with<S2>(self, other: S2) -> Combine<Self, S2>
    where
        Self: Sized,
        S2: Signal + SignalBuilderInternal + Clone,
        <Self as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
        <S2 as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug;

    /// Registers all the systems defined in this signal chain into the Bevy `World`.
    ///
    /// This activates the signal chain. It traverses the internal representation (calling
    /// [`SignalBuilderInternal::register`] recursively), registers each required Bevy system
    /// (or increments its reference count if already registered), connects them in the
    /// [`SignalPropagator`], and marks the source system(s) as roots.
    ///
    /// Returns a [`SignalHandle`] which can be used later to [`cleanup`](SignalHandle::cleanup)
    /// the systems created or referenced specifically by *this* `register` call.
    fn register(&self, world: &mut World) -> SignalHandle;
}

// Implement SignalExt for any type T that implements Signal + SignalBuilderInternal + Clone
impl<T> SignalExt for T
where
    T: Signal + SignalBuilderInternal<Item = <T as Signal>::Item> + Clone,
{
    fn map<U, M, F>(self, system: F) -> Map<Self, U>
    where
        <T as SignalBuilderInternal>::Item: Send + Sync + 'static,
        <T as SignalBuilderInternal>::Item: FromReflect + Send + Sync + 'static,
        U: FromReflect + Send + Sync + 'static,
        F: IntoSystem<In<<T as SignalBuilderInternal>::Item>, Option<U>, M>
            + Send
            + Sync
            + Clone // <-- Re-added Clone bound
            + 'static,
        M: Send + Sync + 'static,
    {
        let system_clone = system.clone(); // <-- Clone the system here

        let register_fn = Arc::new(
            // Closure doesn't need to be move anymore
            move |world: &mut World, prev_last_id_entity: UntypedSystemId| -> UntypedSystemId {
                // Clone the system again for registration
                let system_id = register_signal::<<T as SignalBuilderInternal>::Item, U, M>(
                    world,
                    system_clone.clone(), // Pass the cloned system
                );
                let system_entity = system_id.entity();

                pipe_signal(world, prev_last_id_entity, system_entity);

                system_entity
            },
        );

        Map {
            prev_signal: self,
            register_fn,
            _marker: PhantomData,
        }
    }

    fn combine_with<S2>(self, other: S2) -> Combine<Self, S2>
    where
        S2: Signal + SignalBuilderInternal + Clone,
        <Self as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
        <S2 as Signal>::Item: FromReflect
            + GetTypeRegistration
            + Typed
            + Send
            + Sync
            + Clone
            + 'static
            + std::fmt::Debug,
    {
        let register_fn = move |world: &mut World,
                                left_id_entity: UntypedSystemId,
                                right_id_entity: UntypedSystemId| {
            combine_signal::<<Self as Signal>::Item, <S2 as Signal>::Item>(
                world,
                left_id_entity,
                right_id_entity,
            )
        };

        Combine {
            left_signal: self,
            right_signal: other,
            register_fn: Arc::new(register_fn)
                as Arc<CombineRegisterFn<<Self as Signal>::Item, <S2 as Signal>::Item>>,
            _marker: PhantomData,
        }
    }

    fn register(&self, world: &mut World) -> SignalHandle {
        let all_system_ids = <T as SignalBuilderInternal>::register(self, world);
        SignalHandle(all_system_ids)
    }
}
