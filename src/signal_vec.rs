use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{change_detection::Mut, prelude::*, system::SystemId};
use bevy_log::debug;
use bevy_platform::{
    prelude::*,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use core::{any::Any, cmp::Ordering, fmt, marker::PhantomData, ops::Deref};

use super::{signal::*, tree::*, utils::*};

/// Describes the changes to a `Vec`.
///
/// This is used by [`SignalVec`] to efficiently represent changes.
pub enum VecDiff<T> {
    /// Replaces the entire contents of the `Vec`.
    Replace {
        /// The new values for the vector.
        values: Vec<T>,
    },
    /// Inserts a new item at the `index`.
    InsertAt {
        /// The index where the value should be inserted.
        index: usize,
        /// The value to insert.
        value: T,
    },
    /// Updates the item at the `index`.
    UpdateAt {
        /// The index of the value to update.
        index: usize,
        /// The new value.
        value: T,
    },
    /// Removes the item at the `index`.
    RemoveAt {
        /// The index of the value to remove.
        index: usize,
    },
    /// Moves the item at `old_index` to `new_index`.
    Move {
        /// The original index of the item.
        old_index: usize,
        /// The new index for the item.
        new_index: usize,
    },
    /// Appends a new item to the end of the `Vec`.
    Push {
        /// The value to append.
        value: T,
    },
    /// Removes the last item from the `Vec`.
    Pop,
    /// Removes all items from the `Vec`.
    Clear,
}

impl<T> Clone for VecDiff<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Replace { values } => Self::Replace {
                values: values.clone(),
            },
            Self::InsertAt { index, value } => Self::InsertAt {
                index: *index,
                value: value.clone(),
            },
            Self::UpdateAt { index, value } => Self::UpdateAt {
                index: *index,
                value: value.clone(),
            },
            Self::RemoveAt { index } => Self::RemoveAt { index: *index },
            Self::Move {
                old_index,
                new_index,
            } => Self::Move {
                old_index: *old_index,
                new_index: *new_index,
            },
            Self::Push { value } => Self::Push {
                value: value.clone(),
            },
            Self::Pop => Self::Pop,
            Self::Clear => Self::Clear,
        }
    }
}

impl<T> fmt::Debug for VecDiff<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Replace { values } => f.debug_struct("Replace").field("values", values).finish(),
            Self::InsertAt { index, value } => f
                .debug_struct("InsertAt")
                .field("index", index)
                .field("value", value)
                .finish(),
            Self::UpdateAt { index, value } => f
                .debug_struct("UpdateAt")
                .field("index", index)
                .field("value", value)
                .finish(),
            Self::RemoveAt { index } => f.debug_struct("RemoveAt").field("index", index).finish(),
            Self::Move {
                old_index,
                new_index,
            } => f
                .debug_struct("Move")
                .field("old_index", old_index)
                .field("new_index", new_index)
                .finish(),
            Self::Push { value } => f.debug_struct("Push").field("value", value).finish(),
            Self::Pop => f.debug_struct("Pop").finish(),
            Self::Clear => f.debug_struct("Clear").finish(),
        }
    }
}

/// Represents a `Vec` that changes over time, yielding [`VecDiff<T>`] and handling registration.
///
/// Instead of yielding the entire `Vec` with each change, it yields [`VecDiff<T>`]
/// describing the change. This trait combines the public concept with internal registration.
pub trait SignalVec: SSs {
    /// The type of items in the vector.
    type Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a [`SignalHandle`] containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    /// **Note:** This method is intended for internal use by the signal combinators and registration process.
    fn register_signal_vec(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.boxed().register_boxed_signal_vec(world)
    }
}

impl<O: 'static> SignalVec for Box<dyn SignalVec<Item = O> + Send + Sync> {
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        let inner_box: Box<dyn SignalVec<Item = O> + Send + Sync> = *self;
        inner_box.register_boxed_signal_vec(world)
    }
}

/// A source node for a `SignalVec` chain. Holds the entity ID of the registered source system.
#[derive(Clone)]
pub struct Source<T> {
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> T>,
}

impl<T> SignalVec for Source<T>
where
    T: 'static,
{
    type Item = T;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        let signal = self.signal.register(world);
        if let Ok(mut entity) = world.get_entity_mut(*signal) {
            entity.insert(SignalVecLock);
        }
        signal.into()
    }
}

#[derive(Clone)]
pub struct ForEach<Upstream, O> {
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
}

impl<Upstream, O> SignalVec for ForEach<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        // TODO: dry logic with register_non_source_signal_vec
        if let Some(upstream) = get_direct_upstream(world, signal) {
            if let Ok(mut entity) = world.get_entity_mut(*upstream) {
                entity.remove::<SignalVecLock>();
            }
        }
        signal.into()
    }
}

fn get_direct_upstream(world: &mut World, signal: SignalSystem) -> Option<SignalSystem> {
    world
        .run_system_cached_with(
            move |In(signal): In<SignalSystem>, upstreams: Query<&Upstream>| {
                UpstreamIter::new(&upstreams, signal).next()
            },
            signal,
        )
        .ok()
        .flatten()
}

fn register_non_source_signal_vec(world: &mut World, signal: LazySignal) -> SignalHandle {
    let signal = signal.register(world);
    if let Some(upstream) = get_direct_upstream(world, signal) {
        if let Ok(mut entity) = world.get_entity_mut(*upstream) {
            entity.remove::<SignalVecLock>();
        }
    }
    signal.into()
}

/// A map node in a `SignalVec` chain.
#[derive(Clone)]
pub struct Map<Upstream, O> {
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> SignalVec for Map<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

struct MapSignalItem<S: Signal>
where
    S::Item: Send + Sync,
{
    signal: SignalHandle,
    current_value: S::Item,
}

#[derive(Component)]
struct MapSignalData<T, S: Signal>
where
    T: 'static,
    S::Item: Send + Sync,
{
    items: Vec<MapSignalItem<S>>,
    diffs: Vec<VecDiff<S::Item>>,
    system: SystemId<In<T>, S>,
}

fn with_map_signal_data<T: SSs, S: Signal, O>(
    world: &mut World,
    entity: Entity,
    f: impl FnOnce(Mut<MapSignalData<T, S>>) -> O,
) -> O
where
    S::Item: Send + Sync,
{
    // Added bound
    let data = world.get_mut::<MapSignalData<T, S>>(entity).unwrap();
    f(data)
}

#[derive(Component, Deref, DerefMut)]
struct MapSignalIndex(usize);

fn create_map_signal_processor<T: SSs, S: Signal>(
    parent: Entity,
    entity: LazyEntity,
) -> impl Fn(In<S::Item>, Query<&MapSignalIndex>, Query<&mut MapSignalData<T, S>>)
where
    S::Item: Clone + Send + Sync, // Added bound
{
    move |In(new_value): In<S::Item>,
          map_signal_indices: Query<&MapSignalIndex>,
          mut map_signal_datas: Query<&mut MapSignalData<T, S>>| {
        if let Ok(mut map_signal_data) = map_signal_datas.get_mut(parent) {
            if let Ok(&MapSignalIndex(index)) = map_signal_indices.get(entity.get()) {
                if let Some(item) = map_signal_data.items.get_mut(index) {
                    item.current_value = new_value.clone();
                    map_signal_data.diffs.push(VecDiff::UpdateAt {
                        index,
                        value: new_value,
                    });
                }
            }
        }
    }
}

// Corrected spawn_map_signal that manually pipes signals instead of using .map()
fn spawn_map_signal<T: SSs, S: Signal>(
    world: &mut World,
    index: usize,
    signal: S,
    parent: Entity,
) -> (SignalHandle, S::Item)
where
    S::Item: Clone + Send + Sync,
{
    let entity = LazyEntity::new();

    // 1. Create the processor closure.
    let processor = create_map_signal_processor::<T, S>(parent, entity.clone());

    // 2. Use `.map()` to attach the processor. This is the correct pattern.
    // The output of the processor is (), so the new mapped signal is a Signal<Item = ()>.
    let mapped_signal = signal.map(processor);

    // 3. Register the new mapped_signal. The handle now refers to the processor node.
    let processor_handle = mapped_signal.register(world);
    entity.set(**processor_handle);
    world
        .entity_mut(**processor_handle)
        .insert(MapSignalIndex(index));

    // 4. To get the initial value, we need to poll the *upstream* of our new processor node.
    // The upstream is the original inner signal (`signal`).
    let upstream_handle = world
        .get::<Upstream>(**processor_handle)
        .unwrap()
        .iter()
        .next()
        .cloned()
        .unwrap();

    let initial_value = poll_signal(world, upstream_handle)
        .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<S::Item>().ok())
        .map(|boxed| *boxed)
        .expect("map_signal's inner signal must emit an initial value upon registration");

    // 5. The handle for the entire inner chain is the processor handle. Cleaning it up
    // will correctly cascade to the original inner signal.
    (processor_handle, initial_value)
}

/// A node that maps items to signals and flattens the result.
#[derive(Clone)]
pub struct MapSignal<Upstream, S: Signal> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S: Signal> SignalVec for MapSignal<Upstream, S>
where
    Upstream: SignalVec,
{
    type Item = S::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

fn find_index<'a>(indices: impl Iterator<Item = &'a bool>, index: usize) -> usize {
    indices.take(index).filter(|x| **x).count()
}

fn filter_helper<T, O, O2>(
    world: &mut World,
    diffs: Vec<VecDiff<T::Inner<'static>>>,
    system: SystemId<T, O>,
    f: impl Fn(T::Inner<'static>, O) -> Option<O2>,
    indices: &mut Vec<bool>,
) -> Option<Vec<VecDiff<O2>>>
where
    T: SystemInput + 'static,
    T::Inner<'static>: Clone,
    O: 'static,
{
    let mut output = vec![];
    for diff in diffs {
        let diff_option = match diff {
            VecDiff::Replace { values } => {
                *indices = Vec::with_capacity(values.len());
                let mut output = Vec::with_capacity(values.len());
                for input in values {
                    let value = world
                        .run_system_with(system, input.clone())
                        .ok()
                        .and_then(|output| f(input, output));
                    indices.push(value.is_some());
                    if let Some(value) = value {
                        output.push(value);
                    }
                }
                Some(VecDiff::Replace { values: output })
            }

            VecDiff::InsertAt { index, value } => {
                if let Some(value) = world
                    .run_system_with(system, value.clone())
                    .ok()
                    .and_then(|output| f(value, output))
                {
                    indices.insert(index, true);
                    Some(VecDiff::InsertAt {
                        index: find_index(indices.iter(), index),
                        value,
                    })
                } else {
                    indices.insert(index, false);
                    None
                }
            }

            VecDiff::UpdateAt { index, value } => {
                if let Some(value) = world
                    .run_system_with(system, value.clone())
                    .ok()
                    .and_then(|output| f(value, output))
                {
                    if indices[index] {
                        Some(VecDiff::UpdateAt {
                            index: find_index(indices.iter(), index),
                            value,
                        })
                    } else {
                        indices[index] = true;
                        Some(VecDiff::InsertAt {
                            index: find_index(indices.iter(), index),
                            value,
                        })
                    }
                } else {
                    if indices[index] {
                        indices[index] = false;
                        Some(VecDiff::RemoveAt {
                            index: find_index(indices.iter(), index),
                        })
                    } else {
                        None
                    }
                }
            }

            VecDiff::Move {
                old_index,
                new_index,
            } => {
                if indices.remove(old_index) {
                    indices.insert(new_index, true);

                    Some(VecDiff::Move {
                        old_index: find_index(indices.iter(), old_index),
                        new_index: find_index(indices.iter(), new_index),
                    })
                } else {
                    indices.insert(new_index, false);
                    None
                }
            }

            VecDiff::RemoveAt { index } => {
                if indices.remove(index) {
                    Some(VecDiff::RemoveAt {
                        index: find_index(indices.iter(), index),
                    })
                } else {
                    None
                }
            }

            VecDiff::Push { value } => {
                if let Some(value) = world
                    .run_system_with(system, value.clone())
                    .ok()
                    .and_then(|output| f(value, output))
                {
                    indices.push(true);
                    Some(VecDiff::Push { value })
                } else {
                    indices.push(false);
                    None
                }
            }

            VecDiff::Pop {} => {
                if indices.pop().expect("can't pop from empty vec") {
                    Some(VecDiff::Pop {})
                } else {
                    None
                }
            }

            VecDiff::Clear {} => {
                indices.clear();
                Some(VecDiff::Clear {})
            }
        };

        if let Some(diff) = diff_option {
            output.push(diff);
        }
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

#[derive(Clone)]
pub struct Filter<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for Filter<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

#[derive(Clone)]
pub struct FilterMap<Upstream, O>
where
    Upstream: SignalVec,
{
    signal: LazySignal,

    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> SignalVec for FilterMap<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

struct FilterSignalItem<T> {
    signal: SignalHandle,
    value: T,
    filtered: bool,
}

#[derive(Component)]
struct FilterSignalData<T> {
    items: Vec<FilterSignalItem<T>>,
    diffs: Vec<VecDiff<T>>,
}

fn with_filter_signal_data<T: SSs, O>(
    world: &mut World,
    entity: Entity,
    f: impl FnOnce(Mut<FilterSignalData<T>>) -> O,
) -> O {
    let data = world.get_mut::<FilterSignalData<T>>(entity).unwrap();
    f(data)
}

fn find_filter_signal_index<T>(filter_signal_items: &Vec<FilterSignalItem<T>>, i: usize) -> usize {
    find_index(filter_signal_items.iter().map(|signal| &signal.filtered), i)
}

#[derive(Component, Deref, DerefMut)]
struct FilterSignalIndex(usize);

fn create_filter_signal_processor<T: Clone + SSs>(
    parent: Entity,
    entity: LazyEntity,
) -> impl Fn(In<bool>, Query<&FilterSignalIndex>, Query<&mut FilterSignalData<T>>) {
    move |In(filter),
          filter_signal_indices: Query<&FilterSignalIndex>,
          mut filter_signal_datas: Query<&mut FilterSignalData<T>>| {
        let mut filter_signal_data = filter_signal_datas.get_mut(parent).unwrap();
        let index = filter_signal_indices.get(entity.get()).unwrap().0;
        let mut new = None;
        if let Some(signal) = filter_signal_data.items.get(index) {
            if signal.filtered != filter {
                let filtered_index = find_filter_signal_index(&filter_signal_data.items, index);
                if filter {
                    new = Some((filtered_index, Some(signal.value.clone())))
                } else {
                    new = Some((filtered_index, None))
                }
            };
        }
        if let Some((filtered_index, value_option)) = new {
            if let Some(signal) = filter_signal_data.items.get_mut(index) {
                signal.filtered = filter;
            }
            filter_signal_data
                .diffs
                .push(if let Some(value) = value_option {
                    VecDiff::InsertAt {
                        index: filtered_index,
                        value,
                    }
                } else {
                    VecDiff::RemoveAt {
                        index: filtered_index,
                    }
                });
        }
    }
}

fn spawn_filter_signal<T: Clone + SSs>(
    world: &mut World,
    index: usize,
    signal: impl Signal<Item = bool> + 'static,
    parent: Entity,
) -> (SignalHandle, bool) {
    let entity = LazyEntity::new();
    let processor = create_filter_signal_processor::<T>(parent, entity.clone());

    // Use .map() to attach the processor. The dedupe is still important.
    let mapped_signal = signal.dedupe().map(processor);

    let handle = mapped_signal.register(world);
    entity.set(**handle);
    world.entity_mut(**handle).insert(FilterSignalIndex(index));

    // To get the initial value, we must poll the original signal before the map/dedupe.
    // The upstream of the `map` node is the `dedupe` node.
    let dedupe_handle = world
        .get::<Upstream>(**handle)
        .unwrap()
        .iter()
        .next()
        .cloned()
        .unwrap();
    // The upstream of the `dedupe` node is the original signal.
    let original_signal_handle = world
        .get::<Upstream>(*dedupe_handle)
        .unwrap()
        .iter()
        .next()
        .cloned()
        .unwrap();

    let initial_value = poll_signal(world, original_signal_handle)
        .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<bool>().ok())
        .map(|boxed| *boxed)
        .expect("filter_signal's inner signal must emit an initial value upon registration");

    (handle, initial_value)
}

#[derive(Clone)]
pub struct FilterSignal<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for FilterSignal<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

#[derive(Component, Deref, DerefMut, Clone)]
pub struct EnumeratedIndex(usize);

fn index_signal_from_entity(
    entity: Entity,
) -> Dedupe<super::signal::Map<super::signal::Source<Option<EnumeratedIndex>>, Option<usize>>> {
    SignalBuilder::from_component_option::<EnumeratedIndex>(entity).map(|In(opt): In<Option<EnumeratedIndex>>| opt.map(|c| c.0)).dedupe()
}

#[derive(Clone)]
pub struct Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<
        Upstream,
        Vec<VecDiff<(Dedupe<super::signal::Map<super::signal::Source<Option<EnumeratedIndex>>, Option<usize>>>, Upstream::Item)>>,
    >,
}

impl<Upstream> SignalVec for Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    type Item = (Dedupe<super::signal::Map<super::signal::Source<Option<EnumeratedIndex>>, Option<usize>>>, Upstream::Item);

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct ToSignal<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<Upstream, Vec<Upstream::Item>>,
}

impl<Upstream> Signal for ToSignal<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Vec<Upstream::Item>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct IsEmpty<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    signal: super::signal::Map<ToSignal<Upstream>, bool>,
}

impl<Upstream> Signal for IsEmpty<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    type Item = bool;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct Len<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    signal: super::signal::Map<ToSignal<Upstream>, usize>,
}

impl<Upstream> Signal for Len<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    type Item = usize;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct Sum<Upstream>
where
    Upstream: SignalVec,
{
    signal: super::signal::Map<ToSignal<Upstream>, Upstream::Item>,
}

impl<Upstream> Signal for Sum<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
enum LrDiff<T> {
    Left(Vec<VecDiff<T>>),
    Right(Vec<VecDiff<T>>),
}

/// A node that chains two `SignalVec`s together.
#[derive(Clone)]
pub struct Chain<Left, Right>
where
    Left: SignalVec,
    Right: SignalVec<Item = Left::Item>,
{
    left_wrapper: ForEach<Left, LrDiff<Left::Item>>,
    right_wrapper: ForEach<Right, LrDiff<Left::Item>>,
    signal: LazySignal,
}

impl<Left, Right> SignalVec for Chain<Left, Right>
where
    Left: SignalVec,
    Right: SignalVec<Item = Left::Item>,
{
    type Item = Left::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        let SignalHandle(left_upstream) = self.left_wrapper.register_signal_vec(world);
        let SignalHandle(right_upstream) = self.right_wrapper.register_signal_vec(world);

        let signal = self.signal.register(world);

        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);

        signal.into()
    }
}

/// A node that intersperses a separator between adjacent items of a `SignalVec`.
#[derive(Clone)]
pub struct Intersperse<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<Upstream, Vec<VecDiff<Upstream::Item>>>,
}

impl<Upstream> SignalVec for Intersperse<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// A node that intersperses a separator generated by a system between adjacent items of a `SignalVec`.
#[derive(Clone)]
pub struct IntersperseWith<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for IntersperseWith<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

/// A node that sorts the items of a `SignalVec` using a comparison system.
#[derive(Clone)]
pub struct SortBy<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for SortBy<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

/// A node that sorts the items of a `SignalVec` using an extracted key.
#[derive(Clone)]
pub struct SortByKey<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for SortByKey<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

#[derive(Clone)]
struct FlattenItem<O: SSs + Clone> {
    processor_handle: SignalHandle,
    values: Vec<O>,
    _marker: PhantomData<fn() -> O>,
}

#[derive(Component)]
struct FlattenData<O: SSs + Clone> {
    items: Vec<FlattenItem<O>>,
    diffs: Vec<VecDiff<O>>,
}

fn with_flatten_data<O: SSs + Clone, R>(
    world: &mut World,
    parent: Entity,
    f: impl FnOnce(Mut<FlattenData<O>>) -> R,
) -> R {
    f(world.get_mut::<FlattenData<O>>(parent).unwrap())
}

#[derive(Component, Deref, DerefMut)]
struct FlattenInnerIndex(usize);

fn create_flatten_processor<O: Clone + SSs>(
    parent: Entity,
) -> impl Fn(In<Vec<VecDiff<O>>>, Query<&FlattenInnerIndex>, Query<&mut FlattenData<O>>) {
    move |In(diffs): In<Vec<VecDiff<O>>>,
          query_self_index: Query<&FlattenInnerIndex>,
          mut query_parent_data: Query<&mut FlattenData<O>>| {
        if let Ok(mut parent_data) = query_parent_data.get_mut(parent) {
            // Use .get_single().ok()
            if let Ok(&FlattenInnerIndex(self_index)) = query_self_index.single() {
                if self_index >= parent_data.items.len() {
                    return;
                }
                let offset: usize = parent_data.items[..self_index]
                    .iter()
                    .map(|item| item.values.len())
                    .sum();

                for diff in diffs {
                    let apply_and_queue = |pd: &mut Mut<FlattenData<O>>, d: VecDiff<O>| {
                        let translated_diff = match d {
                            VecDiff::Replace { values } => {
                                let old_len = pd.items[self_index].values.len();
                                for _ in 0..old_len {
                                    pd.diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                                for (i, v) in values.iter().enumerate() {
                                    pd.diffs.push(VecDiff::InsertAt {
                                        index: offset + i,
                                        value: v.clone(),
                                    });
                                }
                                pd.items[self_index].values = values;
                                return;
                            }
                            VecDiff::InsertAt { index, value } => {
                                pd.items[self_index].values.insert(index, value.clone());
                                VecDiff::InsertAt {
                                    index: index + offset,
                                    value,
                                }
                            }
                            VecDiff::UpdateAt { index, value } => {
                                pd.items[self_index].values[index] = value.clone();
                                VecDiff::UpdateAt {
                                    index: index + offset,
                                    value,
                                }
                            }
                            VecDiff::RemoveAt { index } => {
                                pd.items[self_index].values.remove(index);
                                VecDiff::RemoveAt {
                                    index: index + offset,
                                }
                            }
                            VecDiff::Move {
                                old_index,
                                new_index,
                            } => {
                                let val = pd.items[self_index].values.remove(old_index);
                                pd.items[self_index].values.insert(new_index, val);
                                VecDiff::Move {
                                    old_index: old_index + offset,
                                    new_index: new_index + offset,
                                }
                            }
                            VecDiff::Push { value } => {
                                let old_len = pd.items[self_index].values.len();
                                pd.items[self_index].values.push(value.clone());
                                VecDiff::InsertAt {
                                    index: offset + old_len,
                                    value,
                                }
                            }
                            VecDiff::Pop => {
                                pd.items[self_index].values.pop();
                                VecDiff::RemoveAt {
                                    index: offset + pd.items[self_index].values.len(),
                                }
                            }
                            VecDiff::Clear {} => {
                                let old_len = pd.items[self_index].values.len();
                                pd.items[self_index].values.clear();
                                for _ in 0..old_len {
                                    pd.diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                                return;
                            }
                        };
                        pd.diffs.push(translated_diff);
                    };
                    apply_and_queue(&mut parent_data, diff);
                }
            }
        }
    }
}

fn spawn_flatten_item<O: Clone + SSs>(
    world: &mut World,
    parent: Entity,
    index: usize,
    // Fix: Add `Clone` bound to allow using the signal twice.
    inner_signal: impl SignalVec<Item = O> + 'static + Clone,
) -> (FlattenItem<O>, Vec<O>) {
    // Use the cloned signal for the temporary state getter.
    let temp_get_state_signal = inner_signal.clone().to_signal().first();
    let handle = temp_get_state_signal.register(world);
    let initial_values = poll_signal(world, *handle)
        .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<Vec<O>>().ok())
        .map(|b| *b)
        .unwrap_or_default();
    handle.cleanup(world);

    // Use the original signal for the persistent processor.
    let processor_system = create_flatten_processor(parent);
    let processor_handle = inner_signal.for_each(processor_system).register(world);
    world
        .entity_mut(**processor_handle)
        .insert(FlattenInnerIndex(index)); // Correct deref

    let item = FlattenItem {
        processor_handle,
        values: initial_values.clone(),
        _marker: PhantomData,
    };
    (item, initial_values)
}

/// A node that flattens a `SignalVec` of `SignalVec`s.
#[derive(Clone)]
pub struct Flatten<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: SignalVec + 'static,
    <Upstream::Item as SignalVec>::Item: Clone + SSs,
{
    type Item = <Upstream::Item as SignalVec>::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

#[derive(Clone)]
pub struct Debug<Upstream>
where
    Upstream: SignalVec,
{
    pub(crate) signal: ForEach<Upstream, Vec<VecDiff<Upstream::Item>>>,
}

impl<Upstream> SignalVec for Debug<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Extension trait providing combinator methods for types implementing [`SignalVec`] and [`Clone`].
pub trait SignalVecExt: SignalVec {
    /// Registers a system that runs for each batch of `VecDiff`s emitted by this signal.
    ///
    /// The provided system `F` takes `In<Vec<VecDiff<Self::Item>>>` and returns `()`.
    /// This method consumes the signal stream at this point; no further signals are propagated.
    ///
    /// Returns a [`ForEachVec`] node representing this terminal operation.
    /// Call `.register(world)` on the result to activate the chain and get a [`SignalHandle`].
    fn for_each<O, IOO, F, M>(self, system: F) -> ForEach<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Vec<VecDiff<Self::Item>>>, IOO, M> + SSs,
    {
        ForEach {
            upstream: self,
            signal: lazy_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    /// Creates a new `SignalVec` which maps the items within the output diffs of this `SignalVec`
    /// using the given Bevy system `F: IntoSystem<In<Self::Item>, Option<U>, M>`.
    ///
    /// The provided system `F` is run for each relevant item within the incoming `VecDiff<Self::Item>`
    /// (e.g., for `Push`, `InsertAt`, `UpdateAt`, `Replace`). If the system `F` returns `None` for an item,
    /// that item is effectively filtered out from the resulting `VecDiff<U>`. The structure of the diff
    /// (like `RemoveAt`, `Move`, `Pop`, `Clear`) is preserved.
    ///
    /// The system `F` must be `Clone`, `Send`, `Sync`, and `'static`.
    fn map<O, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        F: IntoSystem<In<Self::Item>, O, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(system);
            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<O>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                        let mut output: Vec<VecDiff<O>> = Vec::new();

                        for diff in diffs {
                            let diff_option: Option<VecDiff<O>> = match diff {
                                VecDiff::Replace { values } => {
                                    let mapped_values: Vec<O> = values
                                        .into_iter()
                                        .filter_map(|v| world.run_system_with(system, v).ok())
                                        .collect();
                                    Some(VecDiff::Replace {
                                        values: mapped_values,
                                    })
                                }
                                VecDiff::InsertAt { index, value } => world
                                    .run_system_with(system, value)
                                    .ok()
                                    .map(|mapped_value| VecDiff::InsertAt {
                                        index,
                                        value: mapped_value,
                                    }),
                                VecDiff::UpdateAt { index, value } => world
                                    .run_system_with(system, value)
                                    .ok()
                                    .map(|mapped_value| VecDiff::UpdateAt {
                                        index,
                                        value: mapped_value,
                                    }),
                                VecDiff::Push { value } => world
                                    .run_system_with(system, value)
                                    .ok()
                                    .map(|mapped_value| VecDiff::Push {
                                        value: mapped_value,
                                    }),
                                VecDiff::RemoveAt { index } => Some(VecDiff::RemoveAt { index }),
                                VecDiff::Move {
                                    old_index,
                                    new_index,
                                } => Some(VecDiff::Move {
                                    old_index,
                                    new_index,
                                }),
                                VecDiff::Pop => Some(VecDiff::Pop),
                                VecDiff::Clear => Some(VecDiff::Clear),
                            };

                            if let Some(diff) = diff_option {
                                output.push(diff);
                            }
                        }

                        if output.is_empty() {
                            None
                        } else {
                            Some(output)
                        }
                    },
                )
                .register(world);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal.into()
        });

        Map {
            signal,
            _marker: PhantomData,
        }
    }

    fn map_in<O, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        F: FnMut(Self::Item) -> O + SSs,
    {
        self.map(move |In(item)| function(item))
    }

    fn map_in_ref<O, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        F: FnMut(&Self::Item) -> O + SSs,
    {
        self.map(move |In(item)| function(&item))
    }

    /// Creates a new `SignalVec` by mapping each item of the source `SignalVec` to a `Signal`.
    ///
    /// For each item in the source vector, the provided `system` is run to produce an
    /// inner `Signal`. The output vector's value at a given index becomes the latest
    /// value emitted by the corresponding inner `Signal`.
    ///
    /// This is useful for creating dynamic lists where each element has its own
    /// independent, reactive state.
    fn map_signal<S, F, M>(self, system: F) -> MapSignal<Self, S>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        S: Signal + 'static,
        S::Item: Clone + Send + Sync,
        F: IntoSystem<In<Self::Item>, S, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let parent_entity = LazyEntity::new();
            let system_id = world.register_system(system);

            let SignalHandle(output_signal) = self
                .for_each::<Vec<VecDiff<S::Item>>, _, _, _>(
                    clone!((parent_entity) move |In(mut diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                        let parent = parent_entity.get();
                        let mut new_diffs = vec![];

                        for diff in diffs.drain(..) {
                            match diff {
                                VecDiff::Replace { values } => {
                                    let system_id = with_map_signal_data(world, parent, |data: Mut<MapSignalData<Self::Item, S>>| data.system);
                                    let old_items = with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| data.items.drain(..).collect::<Vec<_>>());

                                    for item in old_items {
                                        item.signal.cleanup(world);
                                    }

                                    let mut new_items = Vec::with_capacity(values.len());
                                    let mut new_values = Vec::with_capacity(values.len());

                                    for (i, value) in values.into_iter().enumerate() {
                                        if let Ok(signal) = world.run_system_with(system_id, value) {
                                            let (handle, initial_value) = spawn_map_signal::<Self::Item, S>(world, i, signal, parent);
                                            new_items.push(MapSignalItem {
                                                signal: handle,
                                                current_value: initial_value.clone(),
                                            });
                                            new_values.push(initial_value);
                                        }
                                    }
                                    with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| {
                                        data.items = new_items;
                                    });
                                    new_diffs.push(VecDiff::Replace { values: new_values });
                                }
                                VecDiff::InsertAt { index, value } => {
                                    let system_id = with_map_signal_data(world, parent, |data: Mut<MapSignalData<Self::Item, S>>| data.system);
                                    if let Ok(signal) = world.run_system_with(system_id, value) {
                                        let (handle, initial_value) = spawn_map_signal::<Self::Item, S>(world, index, signal, parent);
                                        with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| {
                                            data.items.insert(index, MapSignalItem {
                                                signal: handle,
                                                current_value: initial_value.clone(),
                                            });
                                        });
                                        new_diffs.push(VecDiff::InsertAt { index, value: initial_value });
                                    }
                                }
                                VecDiff::UpdateAt { index, value } => {
                                    let (old_item, system_id) = with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| (data.items.remove(index), data.system));
                                    old_item.signal.cleanup(world);
                                    if let Ok(signal) = world.run_system_with(system_id, value) {
                                        let (handle, initial_value) = spawn_map_signal::<Self::Item, S>(world, index, signal, parent);
                                        with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| {
                                            data.items.insert(index, MapSignalItem {
                                                signal: handle,
                                                current_value: initial_value.clone(),
                                            });
                                        });
                                        new_diffs.push(VecDiff::UpdateAt { index, value: initial_value });
                                    }
                                }
                                VecDiff::RemoveAt { index } => {
                                    let item = with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| data.items.remove(index));
                                    item.signal.cleanup(world);
                                    new_diffs.push(VecDiff::RemoveAt { index });
                                }
                                VecDiff::Push { value } => {
                                    let system_id = with_map_signal_data(world, parent, |data: Mut<MapSignalData<Self::Item, S>>| data.system);
                                    if let Ok(signal) = world.run_system_with(system_id, value) {
                                        let index = with_map_signal_data(world, parent, |data: Mut<MapSignalData<Self::Item, S>>| data.items.len());
                                        let (handle, initial_value) = spawn_map_signal::<Self::Item, S>(world, index, signal, parent);
                                        with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| {
                                            data.items.push(MapSignalItem {
                                                signal: handle,
                                                current_value: initial_value.clone(),
                                            });
                                        });
                                        new_diffs.push(VecDiff::Push { value: initial_value });
                                    }
                                }
                                VecDiff::Pop {} => {
                                    let item = with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| data.items.pop().unwrap());
                                    item.signal.cleanup(world);
                                    new_diffs.push(VecDiff::Pop {});
                                }
                                VecDiff::Move { old_index, new_index } => {
                                    with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| {
                                        let item = data.items.remove(old_index);
                                        data.items.insert(new_index, item);
                                    });
                                    new_diffs.push(VecDiff::Move { old_index, new_index });
                                }
                                VecDiff::Clear {} => {
                                    let old_items = with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| data.items.drain(..).collect::<Vec<_>>());
                                    for item in old_items {
                                        item.signal.cleanup(world);
                                    }
                                    new_diffs.push(VecDiff::Clear {});
                                }
                            }
                        }

                        let queued_diffs = with_map_signal_data(world, parent, |mut data: Mut<MapSignalData<Self::Item, S>>| {
                            data.diffs.drain(..).collect::<Vec<_>>()
                        });
                        new_diffs.extend(queued_diffs);

                        if new_diffs.is_empty() { None } else { Some(new_diffs) }
                    }),
                ).register(world);

            parent_entity.set(*output_signal);

            let SignalHandle(flusher) = SignalBuilder::from_entity(*output_signal)
                .map::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    |In(entity), data: Query<&MapSignalData<Self::Item, S>>| {
                        data.get(entity).ok().and_then(|data| {
                            if !data.diffs.is_empty() {
                                Some(vec![])
                            } else {
                                None
                            }
                        })
                    },
                )
                .register(world);

            pipe_signal(world, flusher, output_signal);

            world
                .entity_mut(*output_signal)
                .insert(MapSignalData::<Self::Item, S> {
                    items: vec![],
                    diffs: vec![],
                    system: system_id,
                })
                .add_child(system_id.entity());

            output_signal
        });

        MapSignal {
            signal,
            _marker: PhantomData,
        }
    }

    fn filter<F, M>(self, system: F) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
        F: IntoSystem<In<Self::Item>, bool, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(system);
            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut indices: Local<Vec<bool>>| {
                        filter_helper(
                            world,
                            diffs,
                            system,
                            |item, include| {
                                if include { Some(item) } else { None }
                            },
                            &mut indices,
                        )
                    },
                )
                .register(world);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal.into()
        });
        Filter {
            signal,
            _marker: PhantomData,
        }
    }

    fn filter_map<O, F, M>(self, system: F) -> FilterMap<Self, Self::Item>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
        O: Clone + 'static,
        F: IntoSystem<In<Self::Item>, Option<O>, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(system);
            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<O>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut indices: Local<Vec<bool>>| {
                        filter_helper(world, diffs, system, |_, mapped| mapped, &mut indices)
                    },
                )
                .register(world);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal.into()
        });
        FilterMap {
            signal,
            _marker: PhantomData,
        }
    }

    fn filter_signal<F, S, M>(self, system: F) -> FilterSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        S: Signal<Item = bool> + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let parent_entity = LazyEntity::new();
            let system_id = world.register_system(system);

            let SignalHandle(output_signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(clone!((parent_entity) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                    let parent = parent_entity.get();
                    let mut new_diffs = vec![];

                    for diff in diffs.into_iter() {
                        match diff {
                            VecDiff::Replace { values } => {
                                let old_signals = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    data.items.drain(..).map(|item| item.signal).collect::<Vec<_>>()
                                });
                                for signal in old_signals {
                                    signal.cleanup(world);
                                }

                                let mut new_items = Vec::with_capacity(values.len());
                                let mut new_values = vec![];
                                for (i, value) in values.into_iter().enumerate() {
                                    if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                        let (handle, filtered) = spawn_filter_signal::<Self::Item>(world, i, signal, parent);
                                        if filtered {
                                            new_values.push(value.clone());
                                        }
                                        new_items.push(FilterSignalItem { signal: handle, value, filtered });
                                    }
                                }
                                with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    data.items = new_items;
                                    data.diffs.clear();
                                });
                                new_diffs.push(VecDiff::Replace { values: new_values });
                            },
                            VecDiff::InsertAt { index, value } => {
                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                    let (handle, filtered) = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                    let (new_filtered_index, signals_to_update) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                        data.items.insert(index, FilterSignalItem { signal: handle, value: value.clone(), filtered });
                                        (find_filter_signal_index(&data.items, index), data.items.get((index + 1)..).map(|items| items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>()))
                                    });
                                    if let Some(to_update) = signals_to_update {
                                        for signal in to_update {
                                            if let Some(mut signal_index) = world.get_mut::<FilterSignalIndex>(**signal) {
                                                **signal_index += 1;
                                            }
                                        }
                                    }
                                    if filtered {
                                        new_diffs.push(VecDiff::InsertAt { index: new_filtered_index, value });
                                    }
                                }
                            },
                            VecDiff::UpdateAt { index, value } => {
                                let (old_signal, old_filtered, filtered_index) = with_filter_signal_data(world, parent, |data: Mut<FilterSignalData<Self::Item>>| {
                                    let filtered_index = find_filter_signal_index(&data.items, index);
                                    let item = &data.items[index];
                                    (item.signal.clone(), item.filtered, filtered_index)
                                });

                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                    let (new_handle, new_filtered) = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                    old_signal.cleanup(world);
                                    with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                        let item = &mut data.items[index];
                                        item.signal = new_handle;
                                        item.value = value.clone();
                                        item.filtered = new_filtered;
                                    });
                                    if new_filtered {
                                        if old_filtered {
                                            new_diffs.push(VecDiff::UpdateAt { index: filtered_index, value });
                                        } else {
                                            new_diffs.push(VecDiff::InsertAt { index: filtered_index, value });
                                        }
                                    } else if old_filtered {
                                        new_diffs.push(VecDiff::RemoveAt { index: filtered_index });
                                    }
                                }
                            },
                            VecDiff::RemoveAt { index } => {
                                let (signal, filtered, filtered_index, signals_to_update) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    let filtered_index = find_filter_signal_index(&data.items, index);
                                    let item = data.items.remove(index);
                                    let signals_to_update = data.items.get(index..).map(|items| items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>());
                                    (item.signal, item.filtered, filtered_index, signals_to_update)
                                });
                                signal.cleanup(world);
                                if let Some(to_update) = signals_to_update {
                                    for signal in to_update {
                                        if let Some(mut signal_index) = world.get_mut::<FilterSignalIndex>(**signal) {
                                            **signal_index -= 1;
                                        }
                                    }
                                }
                                if filtered {
                                    new_diffs.push(VecDiff::RemoveAt { index: filtered_index });
                                }
                            },
                            VecDiff::Push { value } => {
                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                    let index = with_filter_signal_data(world, parent, |data: Mut<FilterSignalData<Self::Item>>| data.items.len());
                                    let (handle, filtered) = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                    with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                        data.items.push(FilterSignalItem { signal: handle, value: value.clone(), filtered });
                                    });
                                    if filtered {
                                        new_diffs.push(VecDiff::Push { value });
                                    }
                                }
                            },
                            VecDiff::Pop {} => {
                                let (signal, filtered) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    let item = data.items.pop().expect("can't pop from empty vec");
                                    (item.signal, item.filtered)
                                });
                                signal.cleanup(world);
                                if filtered {
                                    new_diffs.push(VecDiff::Pop {});
                                }
                            },
                            VecDiff::Move { old_index, new_index } => {
                                let (filtered, old_filtered_index, new_filtered_index, items_to_update) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    let old_filtered_index = find_filter_signal_index(&data.items, old_index);
                                    let item = data.items.remove(old_index);
                                    let filtered = item.filtered;
                                    data.items.insert(new_index, item);
                                    let new_filtered_index = find_filter_signal_index(&data.items, new_index);
                                    // Collect the signal handles that need their indices updated.
                                    let items_to_update = data.items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>();
                                    (filtered, old_filtered_index, new_filtered_index, items_to_update)
                                });

                                // Now that the borrow from with_filter_signal_data is over, we can mutably access world.
                                for (i, signal) in items_to_update.into_iter().enumerate() {
                                    if let Some(mut signal_index) = world.get_mut::<FilterSignalIndex>(**signal) {
                                        **signal_index = i;
                                    }
                                }

                                if filtered {
                                    new_diffs.push(VecDiff::Move { old_index: old_filtered_index, new_index: new_filtered_index });
                                }
                            },
                            VecDiff::Clear {} => {
                                let signals = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    data.diffs.clear();
                                    data.items.drain(..).map(|item| item.signal).collect::<Vec<_>>()
                                });
                                for signal in signals {
                                    signal.cleanup(world);
                                }
                                new_diffs.push(VecDiff::Clear {});
                            },
                        }
                    }
                    let mut diffs = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| data.diffs.drain(..).collect::<Vec<_>>());
                    diffs.extend(new_diffs);
                    if diffs.is_empty() { None } else { Some(diffs) }
                }))
                .register(world);

            parent_entity.set(*output_signal);
            let SignalHandle(flusher) = SignalBuilder::from_entity(*output_signal)
                .map::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    |In(entity), filter_signal_datas: Query<&FilterSignalData<Self::Item>>| {
                        if let Ok(data) = filter_signal_datas.get(entity) {
                            if !data.diffs.is_empty() {
                                Some(vec![])
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    },
                )
                .register(world);
            pipe_signal(world, flusher, output_signal);
            world
                .entity_mut(*output_signal)
                .insert(FilterSignalData::<Self::Item> {
                    items: vec![],
                    diffs: vec![],
                })
                .add_child(system_id.entity());
            output_signal
        });
        FilterSignal {
            signal,
            _marker: PhantomData,
        }
    }

    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
    {
        // The inner system that manages the index entities.
        let signal = self.for_each(
            |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                world: &mut World,
                mut index_entities: Local<Vec<Entity>>|
            {
                let mut out_diffs = vec![];

                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values } => {
                            // Despawn all old entities.
                            for entity in index_entities.drain(..) {
                                if let Ok(entity) = world.get_entity_mut(entity) {
                                    entity.despawn();
                                }
                            }
                            
                            let mut new_values = Vec::with_capacity(values.len());
                            for (i, value) in values.into_iter().enumerate() {
                                let entity = world.spawn(EnumeratedIndex(i)).id();
                                index_entities.push(entity);
                                new_values.push((index_signal_from_entity(entity), value));
                            }
                            out_diffs.push(VecDiff::Replace { values: new_values });
                        }
                        VecDiff::InsertAt { index, value } => {
                            let entity = world.spawn(EnumeratedIndex(index)).id();
                            index_entities.insert(index, entity);

                            // Increment indices of all subsequent items.
                            for (i, &entity_to_update) in index_entities.iter().enumerate().skip(index + 1) {
                                if let Some(mut component) = world.get_mut::<EnumeratedIndex>(entity_to_update) {
                                    component.0 = i;
                                }
                            }

                            out_diffs.push(VecDiff::InsertAt { index, value: (index_signal_from_entity(entity), value) });
                        }
                        VecDiff::UpdateAt { index, value } => {
                            let entity = index_entities[index];
                            out_diffs.push(VecDiff::UpdateAt { index, value: (index_signal_from_entity(entity), value) });
                        }
                        VecDiff::RemoveAt { index } => {
                            let entity = index_entities.remove(index);
                            if let Ok(entity) = world.get_entity_mut(entity) {
                                entity.despawn();
                            }
                            
                            // Decrement indices of all subsequent items.
                            for (i, &entity_to_update) in index_entities.iter().enumerate().skip(index) {
                                if let Some(mut component) = world.get_mut::<EnumeratedIndex>(entity_to_update) {
                                    component.0 = i;
                                }
                            }
                            out_diffs.push(VecDiff::RemoveAt { index });
                        }
                        VecDiff::Move { old_index, new_index } => {
                            let entity = index_entities.remove(old_index);
                            index_entities.insert(new_index, entity);

                            // Update all indices between the move points.
                            let start = old_index.min(new_index);
                            let end = old_index.max(new_index) + 1;
                            for (i, &entity_to_update) in index_entities.iter().enumerate().skip(start).take(end - start) {
                                if let Some(mut component) = world.get_mut::<EnumeratedIndex>(entity_to_update) {
                                    component.0 = i;
                                }
                            }
                            out_diffs.push(VecDiff::Move { old_index, new_index });
                        }
                        VecDiff::Push { value } => {
                            let index = index_entities.len();
                            let entity = world.spawn(EnumeratedIndex(index)).id();
                            index_entities.push(entity);                                
                            out_diffs.push(VecDiff::Push { value: (index_signal_from_entity(entity), value) });
                        }
                        VecDiff::Pop {} => {
                            if let Some(entity) = index_entities.pop() {
                                if let Ok(entity) = world.get_entity_mut(entity) {
                                    entity.despawn();
                                }
                                out_diffs.push(VecDiff::Pop {});
                            }
                        }
                        VecDiff::Clear {} => {
                            for entity in index_entities.drain(..) {
                                if let Ok(entity) = world.get_entity_mut(entity) {
                                    entity.despawn();
                                }
                            }
                            out_diffs.push(VecDiff::Clear {});
                        }
                    }
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            }
        );
        Enumerate { signal }
    }

    fn to_signal(self) -> ToSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + 'static,
    {
        let signal = self.for_each(|In(diffs), mut values: Local<Vec<Self::Item>>| {
            for diff in diffs {
                match diff {
                    VecDiff::Replace { values: new_values } => {
                        *values = new_values;
                    }
                    VecDiff::InsertAt { index, value } => {
                        values.insert(index, value);
                    }
                    VecDiff::UpdateAt { index, value } => {
                        values[index] = value;
                    }
                    VecDiff::RemoveAt { index } => {
                        values.remove(index);
                    }
                    VecDiff::Move {
                        old_index,
                        new_index,
                    } => {
                        let old = values.remove(old_index);
                        values.insert(new_index, old);
                    }
                    VecDiff::Push { value } => {
                        values.push(value);
                    }
                    VecDiff::Pop {} => {
                        values.pop().expect("can't pop from empty vec");
                    }
                    VecDiff::Clear {} => {
                        values.clear();
                    }
                }
            }
            values.clone()
        });
        ToSignal { signal }
    }

    fn is_empty(self) -> IsEmpty<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + 'static,
    {
        IsEmpty {
            signal: self
                .to_signal()
                .map(|In(v): In<Vec<Self::Item>>| v.is_empty()),
        }
    }

    fn len(self) -> Len<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + 'static,
    {
        Len {
            signal: self.to_signal().map(|In(v): In<Vec<Self::Item>>| v.len()),
        }
    }

    fn sum(self) -> Sum<Self>
    where
        Self: Sized,
        Self::Item: for<'a> core::iter::Sum<&'a Self::Item> + Clone + Send + 'static,
    {
        Sum {
            signal: self
                .to_signal()
                .map(|In(v): In<Vec<Self::Item>>| v.iter().sum::<Self::Item>()),
        }
    }

    /// Chains two `SignalVec`s together.
    ///
    /// The output `SignalVec` will contain all of the items in `self`,
    /// followed by all of the items in `other`.
    ///
    /// # Performance
    ///
    /// This is a fairly efficient method: it is *guaranteed* constant time, regardless of how big `self` or `other` are.
    ///
    /// The only exception is when `self` or `other` notifies with `VecDiff::Replace` or `VecDiff::Clear`,
    /// in which case it is linear time.
    fn chain<S>(self, other: S) -> Chain<Self, S>
    where
        S: SignalVec<Item = Self::Item>,
        Self: Sized,
        Self::Item: SSs + Clone,
    {
        let left_wrapper = self.for_each(|In(diffs)| LrDiff::Left(diffs));
        let right_wrapper = other.for_each(|In(diffs)| LrDiff::Right(diffs));

        let signal = lazy_signal_from_system::<_, Vec<VecDiff<Self::Item>>, _, _, _>(
            move |In(lr_diff): In<LrDiff<Self::Item>>,
                  mut left_len: Local<usize>,
                  mut right_len: Local<usize>| {
                let mut out_diffs = Vec::new();

                let (is_left, diffs) = match lr_diff {
                    LrDiff::Left(diffs) => (true, diffs),
                    LrDiff::Right(diffs) => (false, diffs),
                };

                for diff in diffs {
                    if is_left {
                        match diff {
                            VecDiff::Replace { values } => {
                                let removing = *left_len;
                                *left_len = values.len();
                                if *right_len == 0 {
                                    out_diffs.push(VecDiff::Replace { values });
                                } else {
                                    for i in (0..removing).rev() {
                                        out_diffs.push(VecDiff::RemoveAt { index: i });
                                    }
                                    for (i, value) in values.into_iter().enumerate() {
                                        out_diffs.push(VecDiff::InsertAt { index: i, value });
                                    }
                                }
                            }
                            VecDiff::InsertAt { index, value } => {
                                *left_len += 1;
                                out_diffs.push(VecDiff::InsertAt { index, value });
                            }
                            VecDiff::UpdateAt { index, value } => {
                                out_diffs.push(VecDiff::UpdateAt { index, value });
                            }
                            VecDiff::Move {
                                old_index,
                                new_index,
                            } => {
                                out_diffs.push(VecDiff::Move {
                                    old_index,
                                    new_index,
                                });
                            }
                            VecDiff::RemoveAt { index } => {
                                *left_len -= 1;
                                out_diffs.push(VecDiff::RemoveAt { index });
                            }
                            VecDiff::Push { value } => {
                                let index = *left_len;
                                *left_len += 1;
                                if *right_len == 0 {
                                    out_diffs.push(VecDiff::Push { value });
                                } else {
                                    out_diffs.push(VecDiff::InsertAt { index, value });
                                }
                            }
                            VecDiff::Pop {} => {
                                *left_len -= 1;
                                if *right_len == 0 {
                                    out_diffs.push(VecDiff::Pop {});
                                } else {
                                    out_diffs.push(VecDiff::RemoveAt { index: *left_len });
                                }
                            }
                            VecDiff::Clear {} => {
                                let removing = *left_len;
                                *left_len = 0;
                                if *right_len == 0 {
                                    out_diffs.push(VecDiff::Clear {});
                                } else {
                                    for i in (0..removing).rev() {
                                        out_diffs.push(VecDiff::RemoveAt { index: i });
                                    }
                                }
                            }
                        }
                    } else {
                        // is_right
                        match diff {
                            VecDiff::Replace { values } => {
                                let removing = *right_len;
                                *right_len = values.len();
                                if *left_len == 0 {
                                    out_diffs.push(VecDiff::Replace { values });
                                } else {
                                    for _ in 0..removing {
                                        out_diffs.push(VecDiff::Pop {});
                                    }
                                    for value in values {
                                        out_diffs.push(VecDiff::Push { value });
                                    }
                                }
                            }
                            VecDiff::InsertAt { index, value } => {
                                *right_len += 1;
                                out_diffs.push(VecDiff::InsertAt {
                                    index: index + *left_len,
                                    value,
                                });
                            }
                            VecDiff::UpdateAt { index, value } => {
                                out_diffs.push(VecDiff::UpdateAt {
                                    index: index + *left_len,
                                    value,
                                });
                            }
                            VecDiff::Move {
                                old_index,
                                new_index,
                            } => {
                                out_diffs.push(VecDiff::Move {
                                    old_index: old_index + *left_len,
                                    new_index: new_index + *left_len,
                                });
                            }
                            VecDiff::RemoveAt { index } => {
                                *right_len -= 1;
                                out_diffs.push(VecDiff::RemoveAt {
                                    index: index + *left_len,
                                });
                            }
                            VecDiff::Push { value } => {
                                *right_len += 1;
                                out_diffs.push(VecDiff::Push { value });
                            }
                            VecDiff::Pop {} => {
                                *right_len -= 1;
                                out_diffs.push(VecDiff::Pop {});
                            }
                            VecDiff::Clear {} => {
                                let removing = *right_len;
                                *right_len = 0;
                                if *left_len == 0 {
                                    out_diffs.push(VecDiff::Clear {});
                                } else {
                                    for _ in 0..removing {
                                        out_diffs.push(VecDiff::Pop {});
                                    }
                                }
                            }
                        }
                    }
                }

                if out_diffs.is_empty() {
                    None
                } else {
                    Some(out_diffs)
                }
            },
        );

        Chain {
            left_wrapper,
            right_wrapper,
            signal,
        }
    }

    /// Creates a new `SignalVec` which places a copy of `separator` between adjacent
    /// items of the original `SignalVec`.
    ///
    /// This behaves like `Iterator::intersperse`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use bevy::prelude::*;
    /// # use jonmo::prelude::*;
    /// let numbers = MutableVec::from([1, 2, 3]);
    /// let interspersed_signal = numbers.signal_vec().intersperse(0);
    /// // The resulting vector signal will represent: [1, 0, 2, 0, 3]
    /// ```
    fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
    {
        let signal = self.for_each(
            move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                  mut local_values: Local<Vec<Self::Item>>| {
                let mut out_diffs = Vec::new();

                for diff in diffs {
                    let old_len = local_values.len();
                    match diff {
                        VecDiff::Replace { mut values } => {
                            let new_len = values.len();

                            // Correctly set the new state *first*.
                            local_values.clear();
                            local_values.append(&mut values);

                            let mut interspersed = Vec::new();
                            if new_len > 0 {
                                interspersed.reserve(2 * new_len - 1);
                                // Iterate over our now-correct local state.
                                let mut iter = local_values.iter().cloned();
                                interspersed.push(iter.next().unwrap());
                                for item in iter {
                                    interspersed.push(separator.clone());
                                    interspersed.push(item);
                                }
                            }
                            out_diffs.push(VecDiff::Replace {
                                values: interspersed,
                            });

                            // The buggy line `*local_values = values;` is removed.
                            // `local_values` now correctly holds the state for the next diff.
                        }
                        VecDiff::InsertAt { index, value } => {
                            local_values.insert(index, value.clone());
                            if old_len == 0 {
                                out_diffs.push(VecDiff::Push { value });
                            } else {
                                out_diffs.push(VecDiff::InsertAt {
                                    index: 2 * index,
                                    value,
                                });
                                // Insert separator *after* unless the item is at the very end.
                                if index < old_len {
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: 2 * index + 1,
                                        value: separator.clone(),
                                    });
                                }
                            }
                        }
                        VecDiff::UpdateAt { index, value } => {
                            local_values[index] = value.clone();
                            out_diffs.push(VecDiff::UpdateAt {
                                index: 2 * index,
                                value,
                            });
                        }
                        VecDiff::RemoveAt { index } => {
                            local_values.remove(index);
                            if old_len == 1 {
                                out_diffs.push(VecDiff::RemoveAt { index: 0 });
                            } else if index == old_len - 1 {
                                // Removing the last item: remove the item and the separator *before* it.
                                out_diffs.push(VecDiff::Pop {}); // item
                                out_diffs.push(VecDiff::Pop {}); // separator
                            } else {
                                // Removing from start/middle: remove the item and the separator *after* it.
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // item
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // separator
                            }
                        }
                        VecDiff::Move {
                            old_index,
                            new_index,
                        } => {
                            let value = local_values.remove(old_index);
                            local_values.insert(new_index, value.clone());

                            // Decompose move into remove + insert for robustness.
                            let mut temp_out = Vec::new();
                            // 1. Generate remove diffs based on old position
                            if old_len == 1 { // Nothing to remove
                            } else if old_index == old_len - 1 {
                                temp_out.push(VecDiff::Pop {});
                                temp_out.push(VecDiff::Pop {});
                            } else {
                                temp_out.push(VecDiff::RemoveAt {
                                    index: 2 * old_index,
                                });
                                temp_out.push(VecDiff::RemoveAt {
                                    index: 2 * old_index,
                                });
                            }

                            // 2. Generate insert diffs based on new position
                            let current_len = local_values.len();
                            if current_len == 1 {
                                temp_out.push(VecDiff::Push { value });
                            } else {
                                temp_out.push(VecDiff::InsertAt {
                                    index: 2 * new_index,
                                    value,
                                });
                                if new_index < current_len - 1 {
                                    temp_out.push(VecDiff::InsertAt {
                                        index: 2 * new_index + 1,
                                        value: separator.clone(),
                                    });
                                }
                            }
                            out_diffs.extend(temp_out);
                        }
                        VecDiff::Push { value } => {
                            local_values.push(value.clone());
                            if old_len > 0 {
                                out_diffs.push(VecDiff::Push {
                                    value: separator.clone(),
                                });
                            }
                            out_diffs.push(VecDiff::Push { value });
                        }
                        VecDiff::Pop {} => {
                            local_values.pop();
                            // This is guaranteed to be removing the last item.
                            if old_len > 0 {
                                out_diffs.push(VecDiff::Pop {}); // The item
                            }
                            if old_len > 1 {
                                out_diffs.push(VecDiff::Pop {}); // The separator before it
                            }
                        }
                        VecDiff::Clear {} => {
                            local_values.clear();
                            out_diffs.push(VecDiff::Clear {});
                        }
                    }
                }

                if out_diffs.is_empty() {
                    None
                } else {
                    Some(out_diffs)
                }
            },
        );
        Intersperse { signal }
    }

    /// Creates a new `SignalVec` which places an item generated by a `separator_system`
    /// between adjacent items of the original `SignalVec`.
    ///
    /// The provided system will be run each time a separator is needed. It receives the
    /// logical `usize` index of the separator as an `In` parameter and must return
    /// a value of the same type as the vector's items.
    ///
    /// The lifecycle of the provided system is tied to the lifecycle of the returned `SignalVec`.
    fn intersperse_with<F, M>(self, separator_system: F) -> IntersperseWith<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        F: IntoSystem<In<usize>, Self::Item, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let separator_system_id = world.register_system(separator_system);

            let SignalHandle(processor_signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut local_values: Local<Vec<Self::Item>>| {
                        let mut out_diffs = Vec::new();

                        let mut separator_fn = |index: usize| {
                            world
                                .run_system_with(separator_system_id, index)
                                .unwrap_or_else(|_| {
                                    panic!("Separator system failed to run for index {}", index)
                                })
                        };

                        for diff in diffs {
                            let old_len = local_values.len();
                            match diff {
                                VecDiff::Replace { mut values } => {
                                    let new_len = values.len();
                                    local_values.clear();
                                    local_values.append(&mut values);

                                    let mut interspersed = Vec::new();
                                    if new_len > 0 {
                                        interspersed.reserve(2 * new_len - 1);
                                        let mut iter = local_values.iter().cloned();
                                        interspersed.push(iter.next().unwrap());
                                        for (i, item) in iter.enumerate() {
                                            interspersed.push(separator_fn(i));
                                            interspersed.push(item);
                                        }
                                    }
                                    out_diffs.push(VecDiff::Replace {
                                        values: interspersed,
                                    });
                                }
                                VecDiff::InsertAt { index, value } => {
                                    local_values.insert(index, value.clone());
                                    if old_len == 0 {
                                        out_diffs.push(VecDiff::Push { value });
                                    } else {
                                        out_diffs.push(VecDiff::InsertAt {
                                            index: 2 * index,
                                            value,
                                        });
                                        if index < old_len {
                                            out_diffs.push(VecDiff::InsertAt {
                                                index: 2 * index + 1,
                                                value: separator_fn(index),
                                            });
                                        }
                                    }
                                }
                                VecDiff::UpdateAt { index, value } => {
                                    local_values[index] = value.clone();
                                    out_diffs.push(VecDiff::UpdateAt {
                                        index: 2 * index,
                                        value,
                                    });
                                }
                                VecDiff::RemoveAt { index } => {
                                    local_values.remove(index);
                                    if old_len == 1 {
                                        out_diffs.push(VecDiff::RemoveAt { index: 0 });
                                    } else if index == old_len - 1 {
                                        out_diffs.push(VecDiff::Pop {});
                                        out_diffs.push(VecDiff::Pop {});
                                    } else {
                                        out_diffs.push(VecDiff::RemoveAt { index: 2 * index });
                                        out_diffs.push(VecDiff::RemoveAt { index: 2 * index });
                                    }
                                }
                                VecDiff::Move {
                                    old_index,
                                    new_index,
                                } => {
                                    let value = local_values.remove(old_index);
                                    local_values.insert(new_index, value.clone());

                                    let mut temp_out = Vec::new();
                                    if old_len > 1 {
                                        if old_index == old_len - 1 {
                                            temp_out.push(VecDiff::Pop {});
                                            temp_out.push(VecDiff::Pop {});
                                        } else {
                                            temp_out.push(VecDiff::RemoveAt {
                                                index: 2 * old_index,
                                            });
                                            temp_out.push(VecDiff::RemoveAt {
                                                index: 2 * old_index,
                                            });
                                        }
                                    }

                                    let current_len = local_values.len();
                                    if current_len == 1 {
                                        temp_out.push(VecDiff::Push { value });
                                    } else {
                                        temp_out.push(VecDiff::InsertAt {
                                            index: 2 * new_index,
                                            value,
                                        });
                                        if new_index < current_len - 1 {
                                            temp_out.push(VecDiff::InsertAt {
                                                index: 2 * new_index + 1,
                                                value: separator_fn(new_index),
                                            });
                                        }
                                    }
                                    out_diffs.extend(temp_out);
                                }
                                VecDiff::Push { value } => {
                                    local_values.push(value.clone());
                                    if old_len > 0 {
                                        out_diffs.push(VecDiff::Push {
                                            value: separator_fn(old_len - 1),
                                        });
                                    }
                                    out_diffs.push(VecDiff::Push { value });
                                }
                                VecDiff::Pop {} => {
                                    local_values.pop();
                                    if old_len > 0 {
                                        out_diffs.push(VecDiff::Pop {});
                                    }
                                    if old_len > 1 {
                                        out_diffs.push(VecDiff::Pop {});
                                    }
                                }
                                VecDiff::Clear {} => {
                                    local_values.clear();
                                    out_diffs.push(VecDiff::Clear {});
                                }
                            }
                        }
                        if out_diffs.is_empty() {
                            None
                        } else {
                            Some(out_diffs)
                        }
                    },
                )
                .register(world);

            // Tie the separator system's lifecycle to the main processor's lifecycle.
            world
                .entity_mut(*processor_signal)
                .add_child(separator_system_id.entity());

            processor_signal
        });

        IntersperseWith {
            signal,
            _marker: PhantomData,
        }
    }

    /// Creates a new `SignalVec` which keeps its items sorted according to a comparison system.
    ///
    /// The provided `compare_system` takes a tuple of two items `In<(Self::Item, Self::Item)>`
    /// and must return a `std::cmp::Ordering`. The output `SignalVec` will always contain the
    /// same items as the source, but reordered based on the system's logic.
    ///
    /// The sort is stable: if the comparison system returns `Ordering::Equal`, the original
    /// order of the equal items is preserved.
    fn sort_by<F, M>(self, compare_system: F) -> SortBy<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        F: IntoSystem<In<(Self::Item, Self::Item)>, Ordering, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let compare_system_id = world.register_system(compare_system);

            struct SortState<T> {
                values: Vec<T>,
                sorted_indices: Vec<usize>,
            }

            impl<T> Default for SortState<T> {
                fn default() -> Self {
                    Self {
                        values: Vec::new(),
                        sorted_indices: Vec::new(),
                    }
                }
            }

            fn search<T: Clone + SSs>(
                world: &mut World,
                system_id: SystemId<In<(T, T)>, Ordering>,
                state: &SortState<T>,
                item_idx: usize,
            ) -> Result<usize, usize> {
                state.sorted_indices.binary_search_by(|&probe_idx| {
                    let a = state.values[probe_idx].clone();
                    let b = state.values[item_idx].clone();
                    world
                        .run_system_with(system_id, (a, b))
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| probe_idx.cmp(&item_idx))
                })
            }

            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut state: Local<SortState<Self::Item>>|
                          -> Option<Vec<VecDiff<Self::Item>>> {
                        let mut out_diffs = Vec::new();

                        for diff in diffs {
                            match diff {
                                VecDiff::Replace { values: new_values } => {
                                    state.values = new_values;
                                    state.sorted_indices = (0..state.values.len()).collect();

                                    // Fix: Clone the values *before* the sort to avoid borrow conflict.
                                    let values_for_sort = state.values.clone();
                                    state.sorted_indices.sort_unstable_by(|&a, &b| {
                                        let val_a = values_for_sort[a].clone();
                                        let val_b = values_for_sort[b].clone();
                                        world
                                            .run_system_with(compare_system_id, (val_a, val_b))
                                            .unwrap()
                                    });

                                    let sorted_values = state
                                        .sorted_indices
                                        .iter()
                                        .map(|&i| state.values[i].clone())
                                        .collect();
                                    out_diffs.push(VecDiff::Replace {
                                        values: sorted_values,
                                    });
                                }
                                VecDiff::Push { value } => {
                                    let new_idx = state.values.len();
                                    state.values.push(value.clone());
                                    let insert_pos =
                                        search(world, compare_system_id, &*state, new_idx)
                                            .unwrap_err();
                                    state.sorted_indices.insert(insert_pos, new_idx);
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: insert_pos,
                                        value,
                                    });
                                }
                                VecDiff::InsertAt { index, value } => {
                                    state.values.insert(index, value.clone());
                                    for i in state.sorted_indices.iter_mut() {
                                        if *i >= index {
                                            *i += 1;
                                        }
                                    }
                                    let insert_pos =
                                        search(world, compare_system_id, &*state, index)
                                            .unwrap_err();
                                    state.sorted_indices.insert(insert_pos, index);
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: insert_pos,
                                        value,
                                    });
                                }
                                VecDiff::RemoveAt { index } => {
                                    // Must search *before* removing from values.
                                    let remove_pos =
                                        search(world, compare_system_id, &*state, index).unwrap();
                                    state.sorted_indices.remove(remove_pos);
                                    state.values.remove(index);
                                    for i in state.sorted_indices.iter_mut() {
                                        if *i > index {
                                            *i -= 1;
                                        }
                                    }
                                    out_diffs.push(VecDiff::RemoveAt { index: remove_pos });
                                }
                                VecDiff::UpdateAt { index, value } => {
                                    let old_pos =
                                        search(world, compare_system_id, &*state, index).unwrap();
                                    state.sorted_indices.remove(old_pos);
                                    state.values[index] = value.clone();
                                    let new_pos = search(world, compare_system_id, &*state, index)
                                        .unwrap_err();
                                    state.sorted_indices.insert(new_pos, index);

                                    if old_pos == new_pos {
                                        out_diffs.push(VecDiff::UpdateAt {
                                            index: old_pos,
                                            value,
                                        });
                                    } else {
                                        out_diffs.push(VecDiff::Move {
                                            old_index: old_pos,
                                            new_index: new_pos,
                                        });
                                    }
                                }
                                VecDiff::Move {
                                    old_index,
                                    new_index,
                                } => {
                                    let val = state.values.remove(old_index);
                                    state.values.insert(new_index, val);
                                    for i in state.sorted_indices.iter_mut() {
                                        if *i == old_index {
                                            *i = new_index;
                                        } else if *i > old_index && *i <= new_index {
                                            *i -= 1;
                                        } else if *i < old_index && *i >= new_index {
                                            *i += 1;
                                        }
                                    }
                                }
                                VecDiff::Pop {} => {
                                    let index = state.values.len() - 1;
                                    let remove_pos =
                                        search(world, compare_system_id, &*state, index).unwrap();
                                    state.sorted_indices.remove(remove_pos);
                                    state.values.pop();
                                    out_diffs.push(VecDiff::RemoveAt { index: remove_pos });
                                }
                                VecDiff::Clear {} => {
                                    state.values.clear();
                                    state.sorted_indices.clear();
                                    out_diffs.push(VecDiff::Clear {});
                                }
                            }
                        }
                        if out_diffs.is_empty() {
                            None
                        } else {
                            Some(out_diffs)
                        }
                    },
                )
                .register(world);

            world
                .entity_mut(*signal)
                .add_child(compare_system_id.entity());
            signal
        });

        SortBy {
            signal,
            _marker: PhantomData,
        }
    }

    // /// Sorts the `SignalVec` with a key extraction system, preserving the initial order of equal elements.
    // ///
    // /// The provided `key_extraction_system` takes an item `In<Self::Item>` and must return a key
    // /// `K` that implements `Ord`. The output `SignalVec` will be sorted based on this key.
    // ///
    // /// This is generally more efficient than `sort_by` if the key extraction is a non-trivial operation,
    // /// as the system is run only once per item change, rather than on every comparison.
    fn sort_by_key<K, F, M>(self, key_extraction_system: F) -> SortByKey<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        K: Ord + Clone + SSs,
        F: IntoSystem<In<Self::Item>, K, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let key_system_id = world.register_system(key_extraction_system);

            struct SortState<T, K> {
                values: Vec<T>,
                keys: Vec<K>,
                sorted_indices: Vec<usize>,
            }

            impl<T, K> Default for SortState<T, K> {
                fn default() -> Self {
                    Self {
                        values: Vec::new(),
                        keys: Vec::new(),
                        sorted_indices: Vec::new(),
                    }
                }
            }

            fn search<T, K: Ord>(state: &SortState<T, K>, item_idx: usize) -> Result<usize, usize> {
                state.sorted_indices.binary_search_by(|&probe_idx| {
                    state.keys[probe_idx]
                        .cmp(&state.keys[item_idx])
                        .then_with(|| probe_idx.cmp(&item_idx))
                })
            }

            let SignalHandle(signal) = self
                // Fix 1: Add turbofish to specify the generic type for `for_each`.
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut state: Local<SortState<Self::Item, K>>|
                          -> Option<Vec<VecDiff<Self::Item>>> {
                        let mut out_diffs = Vec::new();

                        let get_key = |world: &mut World, value: Self::Item| {
                            world.run_system_with(key_system_id, value).unwrap()
                        };

                        for diff in diffs {
                            match diff {
                                VecDiff::Replace { values: new_values } => {
                                    state.values = new_values;
                                    state.keys = state
                                        .values
                                        .iter()
                                        .map(|v| get_key(world, v.clone()))
                                        .collect();
                                    state.sorted_indices = (0..state.values.len()).collect();

                                    // Fix 2: Clone keys before sorting to avoid mutable/immutable borrow conflict.
                                    let keys_for_sort = state.keys.clone();
                                    state.sorted_indices.sort_unstable_by(|&a, &b| {
                                        keys_for_sort[a].cmp(&keys_for_sort[b])
                                    });

                                    let sorted_values = state
                                        .sorted_indices
                                        .iter()
                                        .map(|&i| state.values[i].clone())
                                        .collect();
                                    out_diffs.push(VecDiff::Replace {
                                        values: sorted_values,
                                    });
                                }
                                VecDiff::Push { value } => {
                                    let new_idx = state.values.len();
                                    let key = get_key(world, value.clone());
                                    state.values.push(value.clone());
                                    state.keys.push(key);
                                    let insert_pos = search(&*state, new_idx).unwrap_err();
                                    state.sorted_indices.insert(insert_pos, new_idx);
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: insert_pos,
                                        value,
                                    });
                                }
                                VecDiff::InsertAt { index, value } => {
                                    let key = get_key(world, value.clone());
                                    state.values.insert(index, value.clone());
                                    state.keys.insert(index, key);
                                    for i in state.sorted_indices.iter_mut() {
                                        if *i >= index {
                                            *i += 1;
                                        }
                                    }
                                    let insert_pos = search(&*state, index).unwrap_err();
                                    state.sorted_indices.insert(insert_pos, index);
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: insert_pos,
                                        value,
                                    });
                                }
                                VecDiff::RemoveAt { index } => {
                                    let remove_pos = search(&*state, index).unwrap();
                                    state.sorted_indices.remove(remove_pos);
                                    state.values.remove(index);
                                    state.keys.remove(index);
                                    for i in state.sorted_indices.iter_mut() {
                                        if *i > index {
                                            *i -= 1;
                                        }
                                    }
                                    out_diffs.push(VecDiff::RemoveAt { index: remove_pos });
                                }
                                VecDiff::UpdateAt { index, value } => {
                                    let old_pos = search(&*state, index).unwrap();
                                    state.sorted_indices.remove(old_pos);
                                    let new_key = get_key(world, value.clone());
                                    state.values[index] = value.clone();
                                    state.keys[index] = new_key;
                                    let new_pos = search(&*state, index).unwrap_err();
                                    state.sorted_indices.insert(new_pos, index);

                                    if old_pos == new_pos {
                                        out_diffs.push(VecDiff::UpdateAt {
                                            index: old_pos,
                                            value,
                                        });
                                    } else {
                                        out_diffs.push(VecDiff::Move {
                                            old_index: old_pos,
                                            new_index: new_pos,
                                        });
                                    }
                                }
                                VecDiff::Move {
                                    old_index,
                                    new_index,
                                } => {
                                    let val = state.values.remove(old_index);
                                    state.values.insert(new_index, val);
                                    let key = state.keys.remove(old_index);
                                    state.keys.insert(new_index, key);

                                    for i in state.sorted_indices.iter_mut() {
                                        if *i == old_index {
                                            *i = new_index;
                                        } else if *i > old_index && *i <= new_index {
                                            *i -= 1;
                                        } else if *i < old_index && *i >= new_index {
                                            *i += 1;
                                        }
                                    }
                                }
                                VecDiff::Pop {} => {
                                    let index = state.values.len() - 1;
                                    let remove_pos = search(&*state, index).unwrap();
                                    state.sorted_indices.remove(remove_pos);
                                    state.values.pop();
                                    state.keys.pop();
                                    out_diffs.push(VecDiff::RemoveAt { index: remove_pos });
                                }
                                VecDiff::Clear {} => {
                                    state.values.clear();
                                    state.keys.clear();
                                    state.sorted_indices.clear();
                                    out_diffs.push(VecDiff::Clear {});
                                }
                            }
                        }
                        if out_diffs.is_empty() {
                            None
                        } else {
                            Some(out_diffs)
                        }
                    },
                )
                .register(world);

            world.entity_mut(*signal).add_child(key_system_id.entity());
            signal
        });

        SortByKey {
            signal,
            _marker: PhantomData,
        }
    }

    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: SignalVec + 'static + Clone,
        <Self::Item as SignalVec>::Item: Clone + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let parent_entity = LazyEntity::new();

            let SignalHandle(output_signal) = self
                .for_each::<Vec<VecDiff<<Self::Item as SignalVec>::Item>>, _, _, _>(clone!((parent_entity) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                    let parent = parent_entity.get();
                    let mut new_diffs = vec![];

                    for diff in diffs {
                        match diff {
                            VecDiff::Push { value: inner_signal } => {
                                let index = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items.len());
                                let (item, initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
                                let offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items.iter().map(|i| i.values.len()).sum());
                                with_flatten_data(world, parent, |mut data| data.items.push(item));

                                for (i, value) in initial_values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
                                }
                            }
                            VecDiff::InsertAt { index, value: inner_signal } => {
                                let (item, initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
                                let offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items[..index].iter().map(|i| i.values.len()).sum());
                                with_flatten_data(world, parent, |mut data| data.items.insert(index, item));
                                let signals_to_update = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items.iter().map(|i| i.processor_handle.clone()).collect::<Vec<_>>());
                                for (i, handle) in signals_to_update.into_iter().enumerate() {
                                    if let Some(mut idx) = world.get_mut::<FlattenInnerIndex>(**handle) {
                                        idx.0 = i;
                                    }
                                }
                                for (i, value) in initial_values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
                                }
                            }
                            VecDiff::RemoveAt { index } => {
                                let (item, offset) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let offset = data.items[..index].iter().map(|i| i.values.len()).sum();
                                    (data.items.remove(index), offset)
                                });
                                item.processor_handle.cleanup(world);
                                for _ in 0..item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                            }
                            VecDiff::Pop {} => {
                                let (item, offset) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let offset = data.items.iter().map(|i| i.values.len()).sum::<usize>().saturating_sub(data.items.last().map_or(0, |i| i.values.len()));
                                    (data.items.pop().unwrap(), offset)
                                });
                                item.processor_handle.cleanup(world);
                                for _ in 0..item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                            }
                            VecDiff::Clear {} => {
                                let old_items = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.items.drain(..).collect::<Vec<_>>());
                                for item in old_items {
                                    item.processor_handle.cleanup(world);
                                }
                                new_diffs.push(VecDiff::Clear {});
                            }
                            VecDiff::Replace { values } => {
                                let old_items = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.items.drain(..).collect::<Vec<_>>());
                                for item in old_items {
                                    item.processor_handle.cleanup(world);
                                }
                                new_diffs.push(VecDiff::Clear {});

                                for (i, inner_signal) in values.into_iter().enumerate() {
                                    let (item, initial_values) = spawn_flatten_item(world, parent, i, inner_signal);
                                    with_flatten_data(world, parent, |mut data| data.items.push(item));
                                    for value in initial_values {
                                        new_diffs.push(VecDiff::Push { value });
                                    }
                                }
                            }
                            VecDiff::UpdateAt { index, value: inner_signal } => {
                                let (old_item, offset) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let offset = data.items[..index].iter().map(|i| i.values.len()).sum();
                                    (data.items.remove(index), offset)
                                });
                                old_item.processor_handle.cleanup(world);
                                for _ in 0..old_item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: offset });
                                }

                                let (new_item, initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
                                with_flatten_data(world, parent, |mut data| data.items.insert(index, new_item));
                                for (i, value) in initial_values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
                                }
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let (old_offset, moved_item, signals_to_update) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let old_offset = data.items[..old_index].iter().map(|i| i.values.len()).sum();
                                    let moved_item = data.items.remove(old_index);
                                    data.items.insert(new_index, moved_item);
                                    let signals_to_update = data.items.iter().map(|i| i.processor_handle.clone()).collect::<Vec<_>>();
                                    (old_offset, data.items[new_index].clone(), signals_to_update)
                                });

                                for (i, handle) in signals_to_update.into_iter().enumerate() {
                                    if let Some(mut fi_index) = world.get_mut::<FlattenInnerIndex>(**handle) {
                                        fi_index.0 = i;
                                    }
                                }

                                for _ in 0..moved_item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: old_offset });
                                }

                                let new_offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items[..new_index].iter().map(|i| i.values.len()).sum());

                                for (i, value) in moved_item.values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: new_offset + i, value });
                                }
                            }
                        }
                    }

                    let queued_diffs = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.diffs.drain(..).collect::<Vec<_>>());
                    new_diffs.extend(queued_diffs);

                    if new_diffs.is_empty() { None } else { Some(new_diffs) }
                }))
                .register(world);

            parent_entity.set(*output_signal);

            let SignalHandle(flusher) = SignalBuilder::from_entity(*output_signal)
                .map::<Vec<VecDiff<<Self::Item as SignalVec>::Item>>, _, _, _>(
                    move |In(_), data: Query<&FlattenData<<Self::Item as SignalVec>::Item>>| {
                        if let Ok(data) = data.get(parent_entity.get()) {
                            if !data.diffs.is_empty() {
                                return Some(vec![]);
                            }
                        }
                        None
                    },
                )
                .register(world);

            pipe_signal(world, flusher, output_signal);

            world.entity_mut(*output_signal).insert(
                FlattenData::<<Self::Item as SignalVec>::Item> {
                    items: vec![],
                    diffs: vec![],
                },
            );
            output_signal
        });

        Flatten {
            signal,
            _marker: PhantomData,
        }
    }

    fn debug(self) -> Debug<Self>
    where
        Self: Sized,
        Self::Item: fmt::Debug + Clone + 'static,
    {
        let location = core::panic::Location::caller();
        Debug {
            signal: self.for_each(move |In(item)| {
                debug!("[{}] {:#?}", location, item);
                item
            }),
        }
    }

    fn boxed(self) -> Box<dyn SignalVec<Item = Self::Item>>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Registers all the systems defined in this `SignalVec` chain into the Bevy `World`.
    ///
    /// Returns a [`SignalHandle`] for potential cleanup.
    fn register(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.register_signal_vec(world)
    }
}

impl<T: ?Sized> SignalVecExt for T where T: SignalVec {}

//-------------------------------------------------------------------------------------------------
// Lock Guards
//-------------------------------------------------------------------------------------------------

/// A read guard for `MutableVec`, providing immutable access to the underlying `Vec`.
/// The guard holds the read lock, ensuring safe access.
pub struct MutableVecReadGuard<'a, T>
where
    T: Clone,
{
    guard: RwLockReadGuard<'a, MutableVecState<T>>,
}

impl<'a, T> Deref for MutableVecReadGuard<'a, T>
where
    T: Clone,
{
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: The guard ensures the data is valid for the lifetime 'a.
        &self.guard.vec
    }
}

/// A write guard for `MutableVec`, providing mutable access to the underlying `Vec`.
/// The guard holds the write lock. Changes made through this guard automatically queue
/// the corresponding `VecDiff`s.
pub struct MutableVecWriteGuard<'a, T>
where
    T: Clone,
{
    guard: RwLockWriteGuard<'a, MutableVecState<T>>,
}

impl<'a, T> MutableVecWriteGuard<'a, T>
where
    T: Clone,
{
    /// Pushes a value to the end of the vector and queues a `VecDiff::Push`.
    pub fn push(&mut self, value: T) {
        self.guard.vec.push(value.clone());
        self.guard.pending_diffs.push(VecDiff::Push { value });
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    /// Queues a `VecDiff::Pop` if an element was removed.
    pub fn pop(&mut self) -> Option<T> {
        let result = self.guard.vec.pop();
        if result.is_some() {
            self.guard.pending_diffs.push(VecDiff::Pop);
        }
        result
    }

    /// Inserts an element at `index` within the vector, shifting all elements after it to the right.
    /// Queues a `VecDiff::InsertAt`.
    /// # Panics
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, value: T) {
        self.guard.vec.insert(index, value.clone());
        self.guard
            .pending_diffs
            .push(VecDiff::InsertAt { index, value });
    }

    /// Removes and returns the element at `index` within the vector, shifting all elements after it to the left.
    /// Queues a `VecDiff::RemoveAt`.
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        let value = self.guard.vec.remove(index);
        self.guard.pending_diffs.push(VecDiff::RemoveAt { index });
        value
    }

    /// Removes all elements from the vector.
    /// Queues a `VecDiff::Clear` if the vector was not empty.
    pub fn clear(&mut self) {
        if !self.guard.vec.is_empty() {
            self.guard.vec.clear();
            self.guard.pending_diffs.push(VecDiff::Clear);
        }
    }

    /// Updates the element at `index` with a new `value`.
    /// Queues a `VecDiff::UpdateAt`.
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        let len = self.guard.vec.len();
        if index < len {
            self.guard.vec[index] = value.clone();
            self.guard
                .pending_diffs
                .push(VecDiff::UpdateAt { index, value });
        } else {
            panic!(
                "MutableVecWriteGuard::set: index {} out of bounds for len {}",
                index, len
            );
        }
    }

    /// Moves an item from `old_index` to `new_index`.
    /// Queues a `VecDiff::Move` if the indices are different and valid.
    /// # Panics
    /// Panics if `old_index` or `new_index` are out of bounds.
    pub fn move_item(&mut self, old_index: usize, new_index: usize) {
        let len = self.guard.vec.len();
        if old_index >= len || new_index >= len {
            panic!(
                "MutableVecWriteGuard::move_item: index out of bounds (len: {}, old: {}, new: {})",
                len, old_index, new_index
            );
        }

        if old_index != new_index {
            let value = self.guard.vec.remove(old_index);
            self.guard.vec.insert(new_index, value);
            self.guard.pending_diffs.push(VecDiff::Move {
                old_index,
                new_index,
            });
        }
    }

    /// Replaces the entire contents of the vector with the provided `values`.
    /// Queues a `VecDiff::Replace`.
    pub fn replace(&mut self, values: Vec<T>) {
        self.guard.vec = values.clone();
        self.guard.pending_diffs.push(VecDiff::Replace { values });
    }

    // --- Provide immutable access via Deref ---
    // This allows reading the state even with a write guard.
}

impl<'a, T> Deref for MutableVecWriteGuard<'a, T>
where
    T: Clone,
{
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: The guard ensures the data is valid for the lifetime 'a.
        &self.guard.vec
    }
}

//-------------------------------------------------------------------------------------------------
// MutableVec - Updated to use Guards
//-------------------------------------------------------------------------------------------------

/// Internal state for `MutableVec`, allowing `MutableVec` to be `Clone`.
struct MutableVecState<T>
where
    T: Clone,
{
    vec: Vec<T>,
    pending_diffs: Vec<VecDiff<T>>,
    signals: Vec<SignalSystem>,
}

/// A mutable vector that tracks changes as `VecDiff`s and sends them as a batch on `flush`.
/// This struct is `Clone`able, sharing the underlying state.
#[derive(Clone)]
pub struct MutableVec<T>
where
    T: Clone,
{
    state: Arc<RwLock<MutableVecState<T>>>,
}

#[derive(Component)]
struct QueuedVecDiffs<T>(Vec<VecDiff<T>>);

impl<T, A: Clone> From<T> for MutableVec<A>
where
    Vec<A>: From<T>,
{
    #[inline]
    fn from(values: T) -> Self {
        MutableVec {
            state: Arc::new(RwLock::new(MutableVecState {
                vec: values.into(),
                pending_diffs: Vec::new(),
                signals: Vec::new(),
            })),
        }
    }
}

// TODO: this doesn't work for just a straight up vec of jomnobuilders, actually yes it do ??
/// Marks that a a [`Source`] [`SignalVec`] is ready to be flushed because it at least one downstream signal has been registered.
#[derive(Component)]
struct SignalVecLock;

impl<T> MutableVec<T>
where
    T: Clone,
{
    /// Creates a new, empty `MutableVec`.
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableVecState {
                vec: Vec::new(),
                pending_diffs: Vec::new(),
                signals: Vec::new(),
            })),
        }
    }

    /// Creates a new `MutableVec` initialized with the given values.
    pub fn with_values(values: Vec<T>) -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableVecState {
                pending_diffs: Vec::new(), // Start with empty diffs
                vec: values,
                signals: Vec::new(),
            })),
        }
    }

    /// Acquires a read lock, returning a guard that provides immutable access.
    #[inline]
    pub fn read(&self) -> MutableVecReadGuard<'_, T> {
        MutableVecReadGuard {
            guard: self.state.read().unwrap(),
        }
    }

    /// Acquires a write lock, returning a guard that provides mutable access
    /// and automatically queues diffs for modifications.
    #[inline]
    pub fn write(&self) -> MutableVecWriteGuard<'_, T> {
        MutableVecWriteGuard {
            guard: self.state.write().unwrap(),
        }
    }

    /// Creates a [`SourceVec<T>`] signal linked to this `MutableVec`.
    pub fn signal_vec(&self) -> Source<T>
    where
        T: SSs,
    {
        let signal = LazySignal::new(clone!((self.state => state) move |world: &mut World| {
            let entity = LazyEntity::new();
            let signal = register_signal::<_, Vec<VecDiff<T>>, _, _, _>(world, clone!((entity) move |_: In<()>, world: &mut World| {
                if !world.entity(entity.get()).contains::<SignalVecLock>() {
                    world
                        .get_entity_mut(entity.get())
                        .ok()
                        .and_then(|mut entity: EntityWorldMut<'_>| {
                            entity
                                .get_mut::<QueuedVecDiffs<T>>()
                                .map(|mut queued_diffs| queued_diffs.0.drain(..).collect())
                                .and_then(
                                    |diffs: Vec<VecDiff<T>>| {
                                        if diffs.is_empty() { None } else { Some(diffs) }
                                    },
                                )
                        })
                } else {
                    None
                }
            }));
            entity.set(*signal);
            let mut queued = vec![];
            let init = state.read().unwrap().vec.clone();
            if !init.is_empty() {
                queued.push(VecDiff::Replace { values: init });
            }
            world
                .entity_mut(*signal)
                .insert(QueuedVecDiffs(queued));
            // TODOTODO: these are never cleaned up !!
            state.write().unwrap().signals.push(signal);
            signal
        }));

        Source {
            signal,
            _marker: PhantomData,
        }
    }

    pub fn flush_into_world(&self, world: &mut World)
    where
        T: SSs,
    {
        let mut state: RwLockWriteGuard<'_, MutableVecState<T>> = self.state.write().unwrap();
        if !state.pending_diffs.is_empty() {
            for &signal in &state.signals {
                if let Ok(mut entity) = world.get_entity_mut(*signal) {
                    if let Some(mut queued_diffs) = entity.get_mut::<QueuedVecDiffs<T>>() {
                        queued_diffs.0.extend(state.pending_diffs.clone());
                    }
                }
            }
            state.pending_diffs.clear();
        }
    }

    /// Sends any pending `VecDiff`s accumulated since the last flush to the signal system.
    pub fn flush(&self) -> impl Command
    where
        T: SSs,
    {
        let self_ = self.clone();
        move |world: &mut World| self_.flush_into_world(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JonmoPlugin;
    use bevy::prelude::*;

    // Helper component and resource for testing (similar to signal.rs tests)
    #[derive(Component, Clone, Debug, PartialEq, Default)]
    struct TestItem(i32);

    #[derive(Resource, Default)]
    struct SignalVecOutput<T: Clone + fmt::Debug>(Vec<VecDiff<T>>);

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app
    }

    // Helper system to capture signal_vec output
    fn capture_signal_vec_output<T>(
        In(diffs): In<Vec<VecDiff<T>>>,
        mut output: ResMut<SignalVecOutput<T>>,
    ) where
        T: SSs + Clone + fmt::Debug,
    {
        debug!(
            "Capture SignalVec Output: Received {:?}, extending resource from {:?} with new diffs",
            diffs, output.0
        );
        output.0.extend(diffs);
    }

    fn get_signal_vec_output<T: SSs + Clone + fmt::Debug>(world: &World) -> Vec<VecDiff<T>> {
        world.resource::<SignalVecOutput<T>>().0.clone()
    }

    fn clear_signal_vec_output<T: SSs + Clone + fmt::Debug>(world: &mut World) {
        if let Some(mut output) = world.get_resource_mut::<SignalVecOutput<T>>() {
            output.0.clear();
        }
    }

    impl<T: SSs + PartialEq + fmt::Debug> PartialEq for VecDiff<T> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Replace { values: l_values }, Self::Replace { values: r_values }) => {
                    l_values == r_values
                }
                (
                    Self::InsertAt {
                        index: l_index,
                        value: l_value,
                    },
                    Self::InsertAt {
                        index: r_index,
                        value: r_value,
                    },
                ) => l_index == r_index && l_value == r_value,
                (
                    Self::UpdateAt {
                        index: l_index,
                        value: l_value,
                    },
                    Self::UpdateAt {
                        index: r_index,
                        value: r_value,
                    },
                ) => l_index == r_index && l_value == r_value,
                (Self::RemoveAt { index: l_index }, Self::RemoveAt { index: r_index }) => {
                    l_index == r_index
                }
                (
                    Self::Move {
                        old_index: l_old_index,
                        new_index: l_new_index,
                    },
                    Self::Move {
                        old_index: r_old_index,
                        new_index: r_new_index,
                    },
                ) => l_old_index == r_old_index && l_new_index == r_new_index,
                (Self::Push { value: l_value }, Self::Push { value: r_value }) => {
                    l_value == r_value
                }
                (Self::Pop, Self::Pop) => true,
                (Self::Clear, Self::Clear) => true,
                _ => false,
            }
        }
    }

    #[test]
    fn test_mutable_vec_push_and_flush() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<u32>>();

        let mutable_vec = MutableVec::new();
        let signal_handle = mutable_vec
            .signal_vec()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        app.update(); // Initial flush (Replace with empty)
        let initial_output = get_signal_vec_output::<u32>(app.world());
        assert_eq!(initial_output.len(), 0);

        mutable_vec.write().push(1u32);
        mutable_vec.flush_into_world(app.world_mut());
        app.update();

        let output = get_signal_vec_output::<u32>(app.world());
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], VecDiff::Push { value: 1 });
        assert_eq!(mutable_vec.read().as_ref(), &[1]);

        signal_handle.cleanup(app.world_mut());
    }

    // #[test]
    // fn test_mutable_vec_pop() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(2)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update(); // Initial Replace
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     let popped = mutable_vec.pop();
    //     assert_eq!(popped, Some(TestItem(2)));
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::Pop]);
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(1)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_mutable_vec_insert() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(3)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec.insert(1, TestItem(2));
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::InsertAt { index: 1, value: TestItem(2) }]);
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(1), TestItem(2), TestItem(3)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_mutable_vec_remove() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(2), TestItem(3)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     let removed = mutable_vec.remove(1);
    //     assert_eq!(removed, TestItem(2));
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::RemoveAt { index: 1 }]);
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(1), TestItem(3)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_mutable_vec_clear() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(2)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec.clear();
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::Clear]);
    //     assert!(mutable_vec.is_empty());
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_mutable_vec_set() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(2)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec.set(1, TestItem(5));
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::UpdateAt { index: 1, value: TestItem(5) }]);
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(1), TestItem(5)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_mutable_vec_move_item() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(2), TestItem(3)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec.move_item(0, 2);
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::Move { old_index: 0, new_index: 2 }]);
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(2), TestItem(3), TestItem(1)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_mutable_vec_replace() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::with_values(vec![TestItem(1), TestItem(2)]);
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec.replace(vec![TestItem(10), TestItem(20), TestItem(30)]);
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::Replace { values: vec![TestItem(10), TestItem(20), TestItem(30)] }]);
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(10), TestItem(20), TestItem(30)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_signal_vec_map() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>(); // Output will be TestItem

    //     let mutable_vec_i32 = MutableVec::<i32>::new();
    //     let handle = mutable_vec_i32
    //         .signal_vec()
    //         .map(|In(x): In<i32>| TestItem(x * 2)) // Map i32 to TestItem
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update(); // Initial Replace
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec_i32.push(5);
    //     mutable_vec_i32.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![VecDiff::Push { value: TestItem(10) }]);

    //     clear_signal_vec_output::<TestItem>(app.world_mut());
    //     mutable_vec_i32.insert(0, 1);
    //     mutable_vec_i32.flush().apply(app.world_mut());
    //     app.update();
    //     let output_insert = get_signal_vec_output(app.world());
    //     assert_eq!(output_insert, vec![VecDiff::InsertAt { index: 0, value: TestItem(2)}]);

    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_signal_vec_map_filters_out_none() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();

    //     let mutable_vec_i32 = MutableVec::<i32>::new();

    //     // This map function will be wrapped by the `map` combinator to return Option<TestItem>
    //     // The combinator itself handles the Option part.
    //     // Here, we define a system that returns TestItem directly for valid inputs.
    //     // The map combinator expects a system F: IntoSystem<In<Self::Item>, O, M>
    //     // where O is the output type (TestItem in this case).
    //     // The internal wrapper of the `map` combinator will handle the Option for filtering.
    //     // Let's adjust the test to reflect how `map` is actually used if it were to filter.
    //     // The current `map` implementation for `SignalVecExt` doesn't filter if the mapping system returns `None`.
    //     // It maps values, and if a mapping returns `None` (which the provided system signature doesn't allow directly,
    //     // as it expects `O` not `Option<O>`), it would skip that item in `Replace`, `InsertAt`, `UpdateAt`, `Push`.
    //     // For `RemoveAt`, `Move`, `Pop`, `Clear`, it passes them through.

    //     // To test filtering, the mapping system itself would need to return Option<O>.
    //     // The current `SignalVecExt::map` signature is:
    //     // F: IntoSystem<In<Self::Item>, O, M>
    //     // This means the system `F` must produce `O`, not `Option<O>`.
    //     // The filtering logic is: `world.run_system_with(system, v).ok()`
    //     // If `run_system_with` returns `Err` (e.g. system panics or has unmet dependencies), it's filtered.
    //     // It does NOT filter based on `Option<O>` from the user's system.

    //     // Let's test the existing behavior: mapping values.
    //     // If we want to test filtering, we'd need a `filter_map` or change `map`'s signature.

    //     let handle = mutable_vec_i32
    //         .signal_vec()
    //         .map(|In(x): In<i32>| TestItem(x * 10)) // Simple map
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update();
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec_i32.push(1);
    //     mutable_vec_i32.push(2); // This would be filtered if map returned Option and it was None
    //     mutable_vec_i32.push(3);
    //     mutable_vec_i32.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     // Based on current map (no Option-based filtering from user system):
    //     assert_eq!(output, vec![
    //         VecDiff::Push { value: TestItem(10) },
    //         VecDiff::Push { value: TestItem(20) },
    //         VecDiff::Push { value: TestItem(30) },
    //     ]);

    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_multiple_diffs_batched() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<TestItem>>();
    //     let mutable_vec = MutableVec::<TestItem>::new();
    //     let handle = mutable_vec
    //         .signal_vec()
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update(); // Initial Replace
    //     clear_signal_vec_output::<TestItem>(app.world_mut());

    //     mutable_vec.push(TestItem(1));
    //     mutable_vec.push(TestItem(2));
    //     mutable_vec.insert(0, TestItem(0));
    //     mutable_vec.pop(); // Removes TestItem(2)

    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();

    //     let output = get_signal_vec_output(app.world());
    //     assert_eq!(output, vec![
    //         VecDiff::Push { value: TestItem(1) },
    //         VecDiff::Push { value: TestItem(2) },
    //         VecDiff::InsertAt { index: 0, value: TestItem(0) },
    //         VecDiff::Pop,
    //     ]);
    //     // Expected state: [TestItem(0), TestItem(1)]
    //     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(0), TestItem(1)]);
    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_signal_vec_for_each_runs() {
    //     let mut app = create_test_app();
    //     let counter = Arc::new(Mutex::new(0));

    //     let mutable_vec = MutableVec::<TestItem>::new();
    //     let signal_handle = mutable_vec
    //         .signal_vec()
    //         .for_each(
    //             clone!((counter) move |In(diffs): In<Vec<VecDiff<TestItem>>>| {
    //                 *counter.lock().unwrap() += diffs.len();
    //             }),
    //         )
    //         .register(app.world_mut());

    //     app.update(); // Initial replace diff
    //     assert_eq!(*counter.lock().unwrap(), 1, "Initial replace diff");

    //     mutable_vec.push(TestItem(1));
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(*counter.lock().unwrap(), 2, "After one push diff");

    //     mutable_vec.push(TestItem(2));
    //     mutable_vec.push(TestItem(3));
    //     mutable_vec.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(*counter.lock().unwrap(), 4, "After two more push diffs (batched)");

    //     signal_handle.cleanup(app.world_mut());
    // }

    // // Test cleanup behavior similar to signal.rs tests
    // #[test]
    // fn signal_vec_source_cleanup() {
    //     let mut app = create_test_app();
    //     let mutable_vec = MutableVec::<TestItem>::new();

    //     let source_signal_vec_struct = mutable_vec.signal_vec();
    //     let handle = source_signal_vec_struct.clone().register(app.world_mut());
    //     let system_entity = handle.0.entity();

    //     assert!(app.world().get_entity(system_entity).is_some());
    //     assert_eq!(
    //         **app.world().get::<SignalRegistrationCount>(system_entity).unwrap(),
    //         1
    //     );

    //     handle.cleanup(app.world_mut());
    //     assert_eq!(
    //         **app.world().get::<SignalRegistrationCount>(system_entity).unwrap(),
    //         0
    //     );
    //     // Unlike simple LazySignal, MutableVec's signal source involves a LazyEntity
    //     // and the LazySignalHolder might persist if the MutableVec itself (and its Arc<State>)
    //     // is still alive, as it holds a reference to the signal system.
    //     // The crucial part is that the SignalRegistrationCount is 0.
    //     // The actual despawning logic for systems tied to MutableVec might be more complex
    //     // due to the shared state and the QueuedVecDiffs component.

    //     // For this test, we primarily care that cleanup decrements the count.
    //     // The system might not be immediately despawned if MutableVec still holds a reference.
    //     // Let's check if the system is still there (it might be, this is okay)
    //     // assert!(app.world().get_entity(system_entity).is_some());

    //     drop(source_signal_vec_struct);
    //     // Dropping the original struct clone might trigger further cleanup if it was the last
    //     // structural reference to the LazySignal within.
    //     // The MutableVec itself also needs to be considered.
    //     drop(mutable_vec); // Ensure MutableVec is dropped
    //     app.update(); // Allow cleanup systems to run

    //     // Now, the system should ideally be gone if all references are dropped and cleanup ran.
    //     // This depends on the exact implementation of LazySignal::drop and how it interacts
    //     // with the CLEANUP_SIGNALS queue when its reference count (internal to LazySignalState)
    //     // drops to 1 (meaning only the LazySignalHolder has it).
    //     // And also how MutableVecState.signals are cleaned up.

    //     // A more robust test would be to ensure no panic and counts are correct.
    //     // The entity might persist if the LazySignalHolder is still there due to
    //     // the MutableVec's own Arc<RwLock<MutableVecState<T>>>.
    //     // The key is that new registrations/cleanups behave correctly.
    //     // For now, let's assert the system is gone after everything is dropped and updated.
    //     // This might require careful handling of the `signals` vec in `MutableVecState`.
    //     // If `MutableVec::drop` doesn't explicitly clean up its registered signal systems,
    //     // they might linger if their `LazySignalHolder`'s `LazySignal` doesn't get dropped
    //     // to the point of queuing for cleanup.

    //     // Given the current `MutableVec` structure, `state.signals` are just `SignalSystem` (Entity).
    //     // They are not automatically cleaned up when `MutableVec` is dropped.
    //     // The `SignalHandle::cleanup` is the primary mechanism.
    //     // If `source_signal_vec_struct` is dropped, its `LazySignal`'s strong_count decreases.
    //     // The `LazySignal` inside `LazySignalHolder` on the system entity is another.
    //     // If these are the only two, dropping `source_signal_vec_struct` makes the holder's copy the last one.
    //     // When `LazySignalHolder` is eventually dropped (e.g., because `SignalRegistrationCount` is 0),
    //     // its `LazySignal::drop` will queue the system for cleanup.
    //     app.update(); // Process potential cleanup queue
    //     assert!(
    //         app.world().get_entity(system_entity).is_err(),
    //         "System entity should be despawned after handle cleanup and struct drop"
    //     );
    // }

    // #[test]
    // fn test_map_vec_diff_propagation() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<String>>(); // Output will be String

    //     let mutable_vec_int = MutableVec::<i32>::new();
    //     let handle = mutable_vec_int
    //         .signal_vec()
    //         .map(|In(x): In<i32>| format!("Item: {}", x)) // Map i32 to String
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     app.update(); // Initial Replace
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // Push
    //     mutable_vec_int.push(10);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Push { value: "Item: 10".to_string() }]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // Pop
    //     mutable_vec_int.pop();
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Pop]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // InsertAt
    //     mutable_vec_int.insert(0, 20);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::InsertAt { index: 0, value: "Item: 20".to_string() }]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // UpdateAt
    //     mutable_vec_int.set(0, 30);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::UpdateAt { index: 0, value: "Item: 30".to_string() }]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // RemoveAt
    //     mutable_vec_int.remove(0);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::RemoveAt { index: 0 }]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // Setup for Move: [0, 1, 2] -> map to ["Item: 0", "Item: 1", "Item: 2"]
    //     mutable_vec_int.push(0);
    //     mutable_vec_int.push(1);
    //     mutable_vec_int.push(2);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     clear_signal_vec_output::<String>(app.world_mut()); // Clear the push diffs

    //     // Move
    //     mutable_vec_int.move_item(0, 2); // state: [1, 2, 0]
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Move { old_index: 0, new_index: 2 }]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // Clear
    //     mutable_vec_int.clear();
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Clear]);
    //     clear_signal_vec_output::<String>(app.world_mut());

    //     // Replace
    //     mutable_vec_int.replace(vec![100, 200]);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Replace { values: vec!["Item: 100".to_string(), "Item: 200".to_string()] }]);

    //     handle.cleanup(app.world_mut());
    // }
}
