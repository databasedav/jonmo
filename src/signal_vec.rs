use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{change_detection::Mut, prelude::*, system::SystemId};
use bevy_platform::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use core::{fmt, marker::PhantomData, ops::Deref};

use super::{signal::*, tree::*, utils::*};

/// Describes the changes to a `Vec`.
///
/// This is used by [`SignalVec`] to efficiently represent changes.
pub enum VecDiff<T> {
    // Add PartialEq bound
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
    // NOTE: futures-signals has Truncate, but it's less common and can be represented by multiple RemoveAt/Pop.
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

impl<T> std::fmt::Debug for VecDiff<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a [`SignalHandle`] containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    /// **Note:** This method is intended for internal use by the signal combinators and registration process.
    fn register_signal_vec(self, world: &mut World) -> SignalHandle;
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

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
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

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
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

/// A map node in a `SignalVec` chain.
#[derive(Clone)]
pub struct Map<Upstream, O> {
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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

impl<Upstream, O> SignalVec for Map<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
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

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
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

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        register_non_source_signal_vec(world, self.signal)
    }
}

#[derive(Clone)]
pub struct Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<
        Upstream,
        Vec<VecDiff<(Dedupe<super::signal::Source<Option<usize>>>, Upstream::Item)>>,
    >,
}

impl<Upstream> SignalVec for Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    type Item = (Dedupe<super::signal::Source<Option<usize>>>, Upstream::Item);

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
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

    fn register_signal(self, world: &mut World) -> SignalHandle {
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

    fn register_signal(self, world: &mut World) -> SignalHandle {
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

    fn register_signal(self, world: &mut World) -> SignalHandle {
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

    fn register_signal(self, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
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

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
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

fn index_signal_from_index(
    index: Arc<Mutex<Option<usize>>>,
) -> Dedupe<super::signal::Source<Option<usize>>> {
    SignalBuilder::from_system(move |_: In<_>| *index.lock().unwrap()).dedupe()
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
) -> impl Fn(
    In<bool>,
    Query<&FilterSignalIndex>,
    Query<&mut FilterSignalData<T>>,
    Query<&SignalRegistrationCount>,
) -> bool {
    move |In(filter),
          filter_signal_indices: Query<&FilterSignalIndex>,
          mut filter_signal_datas: Query<&mut FilterSignalData<T>>,
          signals: Query<&SignalRegistrationCount>| {
        println!("here {}: {}", signals.iter().len(), filter);
        let mut filter_signal_data = filter_signal_datas.get_mut(parent).unwrap();
        let index = filter_signal_indices.get(entity.get()).unwrap().0;
        println!("index: {}, filter: {}", index, filter);

        // First, check if we need to do anything and gather info with immutable access
        let (should_insert, should_remove, filtered_index, value) = {
            if let Some(signal) = filter_signal_data.items.get(index) {
                if signal.filtered != filter {
                    let old_filtered = signal.filtered;

                    if filter && !old_filtered {
                        // Item becoming visible - calculate insertion index based on current state
                        let filtered_index =
                            find_filter_signal_index(&filter_signal_data.items, index);
                        (true, false, filtered_index, Some(signal.value.clone()))
                    } else if !filter && old_filtered {
                        // Item becoming hidden - calculate removal index based on current state
                        let filtered_index = filter_signal_data
                            .items
                            .iter()
                            .take(index)
                            .filter(|item| item.filtered)
                            .count();
                        (false, true, filtered_index, None)
                    } else {
                        (false, false, 0, None)
                    }
                } else {
                    (false, false, 0, None)
                }
            } else {
                (false, false, 0, None)
            }
        };

        // Now update the filtered field with mutable access
        if should_insert || should_remove {
            if let Some(signal) = filter_signal_data.items.get_mut(index) {
                signal.filtered = filter;
            }
        }

        // Finally, push the diffs
        if should_insert {
            println!("filter_signal: InsertAt({})", filtered_index);
            filter_signal_data.diffs.push(VecDiff::InsertAt {
                index: filtered_index,
                value: value.unwrap(),
            });
        } else if should_remove {
            println!("filter_signal: RemoveAt({})", filtered_index);
            filter_signal_data.diffs.push(VecDiff::RemoveAt {
                index: filtered_index,
            });
        }

        filter
    }
}

fn spawn_filter_signal<T: Clone + SSs>(
    world: &mut World,
    index: usize,
    signal: impl Signal<Item = bool>,
    parent: Entity,
) -> SignalHandle {
    let entity = LazyEntity::new();
    let signal = signal
        .dedupe()
        .map(create_filter_signal_processor::<T>(parent, entity.clone()))
        .register(world);
    entity.set(**signal);
    world.entity_mut(**signal).insert(FilterSignalIndex(index));
    signal
}

fn poll_filter_signal(world: &mut World, signal: SignalSystem) -> bool {
    poll_signal(world, signal)
        .map(|output| output.downcast::<bool>().ok().as_deref().copied())
        .flatten()
        .unwrap_or(false)
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

    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
    {
        fn increment_indices(range: &[Arc<Mutex<Option<usize>>>]) {
            for index in range {
                let mut guard = index.lock().unwrap();
                *guard = guard.as_ref().map(|value| value + 1)
            }
        }

        fn decrement_indices(range: &[Arc<Mutex<Option<usize>>>]) {
            for index in range {
                let mut guard = index.lock().unwrap();
                *guard = guard.as_ref().map(|value| value - 1)
            }
        }
        let signal = self.for_each(
            |In(diffs), mut indices: Local<Vec<Arc<Mutex<Option<usize>>>>>| {
                let mut output = vec![];
                for diff in diffs {
                    let diff = match diff {
                        VecDiff::Replace { values } => {
                            for signal in indices.drain(..) {
                                signal.lock().unwrap().take();
                            }
                            *indices = Vec::with_capacity(values.len());
                            VecDiff::Replace {
                                values: values
                                    .into_iter()
                                    .enumerate()
                                    .map(|(index, value)| {
                                        let index = Arc::new(Mutex::new(Some(index)));
                                        indices.push(index.clone());
                                        (index_signal_from_index(index), value)
                                    })
                                    .collect(),
                            }
                        }
                        VecDiff::InsertAt { index: i, value } => {
                            let index = Arc::new(Mutex::new(Some(i)));
                            indices.insert(i, index.clone());
                            increment_indices(&indices[(i + 1)..]);
                            VecDiff::InsertAt {
                                index: i,
                                value: (index_signal_from_index(index), value),
                            }
                        }
                        VecDiff::UpdateAt { index, value } => VecDiff::UpdateAt {
                            index,
                            value: (index_signal_from_index(indices[index].clone()), value),
                        },
                        VecDiff::Push { value } => {
                            let index = Arc::new(Mutex::new(Some(indices.len())));
                            indices.push(index.clone());
                            VecDiff::Push {
                                value: (index_signal_from_index(index), value),
                            }
                        }
                        VecDiff::Move {
                            old_index,
                            new_index,
                        } => {
                            let index = indices.remove(old_index);
                            indices.insert(new_index, index.clone());
                            if old_index < new_index {
                                decrement_indices(&indices[old_index..new_index]);
                            } else if new_index < old_index {
                                increment_indices(&indices[(new_index + 1)..(old_index + 1)]);
                            }
                            *index.lock().unwrap() = Some(new_index);
                            VecDiff::Move {
                                old_index,
                                new_index,
                            }
                        }
                        VecDiff::RemoveAt { index: i } => {
                            let index = indices.remove(i);
                            decrement_indices(&indices[i..]);
                            *index.lock().unwrap() = None;
                            VecDiff::RemoveAt { index: i }
                        }
                        VecDiff::Pop {} => {
                            let index = indices.pop().expect("can't pop from empty vec");
                            *index.lock().unwrap() = None;
                            VecDiff::Pop {}
                        }

                        VecDiff::Clear {} => {
                            for index in indices.drain(..) {
                                *index.lock().unwrap() = None;
                            }
                            VecDiff::Clear {}
                        }
                    };
                    output.push(diff);
                }
                if output.is_empty() {
                    None
                } else {
                    Some(output)
                }
            },
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
        Self::Item: for<'a> std::iter::Sum<&'a Self::Item> + Clone + Send + 'static,
    {
        Sum {
            signal: self
                .to_signal()
                .map(|In(v): In<Vec<Self::Item>>| v.iter().sum::<Self::Item>()),
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
            let entity = LazyEntity::new();
            let system = world.register_system(system);
            let SignalHandle(signal) = self.for_each(clone!((entity) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                let parent = entity.get();
                let mut new_diffs = vec![];
                for diff in diffs.into_iter() {
                    let diff_option = match diff {
                        VecDiff::Replace { values } => {
                            println!("filter_signal: Replace: {}", values.len());
                            let mut items = Vec::with_capacity(values.len());
                            let mut new_values = vec![];
                            for (i, value) in values.into_iter().enumerate() {
                                if let Ok(signal) = world.run_system_with(system, value.clone()) {
                                    let signal = spawn_filter_signal::<Self::Item>(world, i, signal, parent);
                                    let filtered = poll_filter_signal(world, *signal);
                                    println!("filter_signal: {}", filtered);
                                    if filtered {
                                        new_values.push(value.clone());
                                    }
                                    items.push(FilterSignalItem {
                                        signal,
                                        value,
                                        filtered,
                                    });
                                }
                            }
                            let old_signals = with_filter_signal_data(world, parent, |mut data| {
                                let old_signals = data.items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>();
                                data.items = items;
                                data.diffs.clear();
                                old_signals
                            });
                            for signal in old_signals {
                                signal.cleanup(world);
                            }
                            Some(VecDiff::Replace { values: new_values })
                        },
                        VecDiff::InsertAt { index, value } => {
                            if let Ok(signal) = world.run_system_with(system, value.clone()) {
                                let signal = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                let filtered = poll_filter_signal(world, *signal);
                                let (index, increment_option) = with_filter_signal_data(world, parent, |mut data| {
                                    data.items.insert(index, FilterSignalItem {
                                        signal,
                                        value: value.clone(),
                                        filtered,
                                    });
                                    (find_filter_signal_index(&data.items, index), data.items.get(index+1..).map(|items| items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>()))
                                });
                                if let Some(increment) = increment_option {
                                    for signal in increment {
                                        let mut index = world.get_mut::<FilterSignalIndex>(**signal).unwrap();
                                        **index += 1;
                                    }
                                }
                                if filtered {
                                    Some(VecDiff::InsertAt { index, value })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        },
                        VecDiff::UpdateAt { index, value } => {
                            if let Ok(signal) = world.run_system_with(system, value.clone()) {
                                let signal = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                let filtered = poll_filter_signal(world, *signal);
                                let (old_signal, old_filtered, index) = with_filter_signal_data(world, parent, |mut data| {
                                    let item = &mut data.items[index];
                                    let old_signal = item.signal.clone();
                                    let filtered = item.filtered;
                                    item.signal = signal;
                                    item.value = value.clone();
                                    item.filtered = filtered;
                                    (old_signal, filtered, find_filter_signal_index(&data.items, index))
                                });
                                old_signal.cleanup(world);
                                if filtered {
                                    if old_filtered {
                                        Some(VecDiff::UpdateAt { index, value })
                                    } else {
                                        Some(VecDiff::InsertAt { index, value })
                                    }
                                } else {
                                    if old_filtered {
                                        Some(VecDiff::RemoveAt { index })
                                    } else {
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        },
                        VecDiff::Push { value } => {
                            if let Ok(signal) = world.run_system_with(system, value.clone()) {
                                let index = with_filter_signal_data::<Self::Item, _>(world, parent, |data| data.items.len());
                                let signal = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                let filtered = poll_filter_signal(world, *signal);
                                with_filter_signal_data(world, parent, |mut data| {
                                    data.items.push(FilterSignalItem {
                                        signal,
                                        value: value.clone(),
                                        filtered,
                                    });
                                });
                                if filtered {
                                    Some(VecDiff::Push { value })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        },
                        VecDiff::Move { old_index, new_index } => {
                            if old_index != new_index {
                                let (index_syncs, filtered, old_index, new_index) = with_filter_signal_data::<Self::Item, _>(world, parent, |mut data| {
                                    let item = data.items.remove(old_index);
                                    let filtered = item.filtered;
                                    data.items.insert(new_index, item);
                                    (data.items[old_index.min(new_index)..old_index.max(new_index)].iter().map(|item| item.signal.clone()).collect::<Vec<_>>(), filtered, find_filter_signal_index(&data.items, old_index), find_filter_signal_index(&data.items, new_index))
                                });
                                for (signal, new_index) in index_syncs.into_iter().zip(old_index..=new_index) {
                                    let mut index = world.get_mut::<FilterSignalIndex>(**signal).unwrap();
                                    **index = new_index;
                                }
                                if filtered {
                                    Some(VecDiff::Move { old_index, new_index })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        },
                        VecDiff::RemoveAt { index } => {
                            let (signal, filtered, index) = with_filter_signal_data::<Self::Item, _>(world, parent, |mut data| {
                                let index = find_filter_signal_index(&data.items, index);
                                let item = data.items.remove(index);
                                let signal = item.signal;
                                let filtered = item.filtered;
                                (signal, filtered, index)
                            });
                            signal.cleanup(world);
                            if filtered {
                                Some(VecDiff::RemoveAt { index })
                            } else {
                                None
                            }
                        },
                        VecDiff::Pop {} => {
                            let (signal, filtered) = with_filter_signal_data::<Self::Item, _>(world, parent, |mut data| {
                                let item = data.items.pop().expect("can't pop from empty vec");
                                let signal = item.signal;
                                let filtered = item.filtered;
                                (signal, filtered)
                            });
                            signal.cleanup(world);
                            if filtered {
                                Some(VecDiff::Pop {})
                            } else {
                                None
                            }
                        },
                        VecDiff::Clear {} => {
                            let signals = with_filter_signal_data::<Self::Item, _>(world, parent, |mut data| {
                                let signals = data.items.drain(..).map(|item| item.signal).collect::<Vec<_>>();
                                data.diffs.clear();
                                signals
                            });
                            for signal in signals {
                                signal.cleanup(world);
                            }
                            Some(VecDiff::Clear {})
                        },
                    };
                    if let Some(diff) = diff_option {
                        new_diffs.push(diff);
                    };
                }
                let mut diffs = with_filter_signal_data(world, parent, |mut data| {
                    data.diffs.drain(..).collect::<Vec<_>>()
                });
                diffs.extend(new_diffs);
                diffs
            })).register(world);
            entity.set(*signal);
            world
                .entity_mut(*signal)
                .insert(FilterSignalData::<Self::Item> {
                    items: vec![],
                    diffs: vec![],
                })
                .add_child(system.entity());
            signal
        });
        FilterSignal {
            signal,
            _marker: PhantomData,
        }
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

impl<T> SignalVecExt for T where T: SignalVec {}

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

// TODO: this doesn't work for just a straight up vec of jomnobuilders
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
    use std::sync::{Arc, Mutex};

    // Helper component and resource for testing (similar to signal.rs tests)
    #[derive(Component, Clone, Debug, PartialEq, Reflect, Default)]
    #[reflect(Clone)]
    struct TestItem(i32);

    #[derive(Resource, Default)]
    struct SignalVecOutput<T: Clone + std::fmt::Debug>(Vec<VecDiff<T>>);

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app.register_type::<TestItem>();
        app.register_type::<VecDiff<TestItem>>();
        app
    }

    // Helper system to capture signal_vec output
    fn capture_signal_vec_output<T>(
        In(diffs): In<Vec<VecDiff<T>>>,
        mut output: ResMut<SignalVecOutput<T>>,
    ) where
        T: SSs + Clone + std::fmt::Debug,
    {
        debug!(
            "Capture SignalVec Output: Received {:?}, extending resource from {:?} with new diffs",
            diffs, output.0
        );
        output.0.extend(diffs);
    }

    fn get_signal_vec_output<T: SSs + Clone + std::fmt::Debug>(
        world: &World,
    ) -> Vec<VecDiff<T>> {
        world.resource::<SignalVecOutput<T>>().0.clone()
    }

    fn clear_signal_vec_output<T: SSs + Clone + std::fmt::Debug>(world: &mut World) {
        if let Some(mut output) = world.get_resource_mut::<SignalVecOutput<T>>() {
            output.0.clear();
        }
    }

    impl<T: SSs + PartialEq + std::fmt::Debug> PartialEq for VecDiff<T> {
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

        mutable_vec.push(1u32);
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
