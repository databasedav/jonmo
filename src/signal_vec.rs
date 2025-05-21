use bevy_ecs::{prelude::*, system::SystemId};
use bevy_reflect::{FromReflect, GetTypeRegistration, Typed, prelude::*};
use std::{
    convert::identity, fmt, marker::PhantomData, ops::Deref, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}
};

use super::{tree::*, utils::*};

/// Describes the changes to a `Vec`.
///
/// This is used by [`SignalVec`] to efficiently represent changes.
#[derive(Reflect)] // Removed PartialEq, removed #[reflect(PartialEq)]
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
    T: Clone
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
    T: std::fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Replace { values } => write!(f, "Replace({:?})", values),
            Self::InsertAt { index, value } => write!(f, "InsertAt({}, {:?})", index, value),
            Self::UpdateAt { index, value } => write!(f, "UpdateAt({}, {:?})", index, value),
            Self::RemoveAt { index } => write!(f, "RemoveAt({})", index),
            Self::Move {
                old_index,
                new_index,
            } => write!(f, "Move({}, {})", old_index, new_index),
            Self::Push { value } => write!(f, "Push({:?})", value),
            Self::Pop => write!(f, "Pop"),
            Self::Clear => write!(f, "Clear"),
        }
    }
}

/// Represents a `Vec` that changes over time, yielding [`VecDiff<T>`] and handling registration.
///
/// Instead of yielding the entire `Vec` with each change, it yields [`VecDiff<T>`]
/// describing the change. This trait combines the public concept with internal registration.
pub trait SignalVec: SSs {
    /// The type of items in the vector.
    type Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs;

    /// Registers the systems associated with this node and its predecessors in the `World`.
    /// Returns a [`SignalHandle`] containing the entities of *all* systems
    /// registered or reference-counted during this specific registration call instance.
    /// **Note:** This method is intended for internal use by the signal combinators and registration process.
    fn register_signal_vec(self, world: &mut World) -> SignalHandle;
}

/// A source node for a `SignalVec` chain. Holds the entity ID of the registered source system.
#[derive(Clone)]
pub struct Source<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    pub(crate) signal: LazySignal,
    _marker: PhantomData<T>,
}

impl<T> SignalVec for Source<T>
where
    T: Clone + Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = T;

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        SignalHandle::new(self.signal.register(world))
    }
}

/// A map node in a `SignalVec` chain.
#[derive(Clone, Reflect)]
pub struct Map<Upstream, U>
where
    Upstream: SignalVec, // Use consolidated SignalVec trait
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
    U: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
    _marker: PhantomData<U>,
}

// Implement SignalVec for MapVec<Upstream, U>
impl<Upstream, U> SignalVec for Map<Upstream, U>
where
    Upstream: SignalVec, // Use consolidated SignalVec trait
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
    U: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = U;

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register_signal_vec(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

/// A terminal node in a `SignalVec` chain that executes a system for each batch.
#[derive(Clone, Reflect)]
pub struct ForEach<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
}

impl<Upstream> SignalVec for ForEach<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = ();

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register_signal_vec(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

pub struct FilterMap<Upstream, U>
where
    Upstream: SignalVec,
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
    U: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<U>,
}

impl<Upstream, U> SignalVec for FilterMap<Upstream, U>
where
    Upstream: SignalVec,
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
    U: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = U;

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register_signal_vec(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

pub struct Filter<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<Upstream>,
}

impl<Upstream> SignalVec for Filter<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
{
    type Item = Upstream::Item;

    fn register_signal_vec(self, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register_signal_vec(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        SignalHandle::new(signal)
    }
}

fn find_index(indices: &[bool], index: usize) -> usize {
    indices[0..index].into_iter().filter(|x| **x).count()
}

fn filter_helper<T, O, O2>(
    world: &mut World,
    diffs: Vec<VecDiff<T::Inner<'static>>>,
    system: SystemId<T, O>,
    f: impl Fn(T::Inner<'static>, O) -> Option<O2>,
    indices: &mut Vec<bool>
) -> Option<Vec<VecDiff<O2>>>
where
    T: SystemInput + 'static,
    T::Inner<'static>: Clone,
    O: 'static
{
    let mut output = vec![];

    for diff in diffs {
        let diff_option = match diff {
            VecDiff::Replace { values } => {
                *indices = Vec::with_capacity(values.len());
                let mut output = Vec::with_capacity(values.len());
                for input in values {
                    let value = world.run_system_with(system, input.clone()).ok().and_then(|output| f(input, output));
                    indices.push(value.is_some());
                    if let Some(value) = value {
                        output.push(value);
                    }
                }
                Some(VecDiff::Replace { values: output })
            }

            VecDiff::InsertAt { index, value } => {
                if let Some(value) = world.run_system_with(system, value.clone()).ok().and_then(|output| f(value, output))
                {
                    indices.insert(index, true);
                    Some(VecDiff::InsertAt {
                        index: find_index(&indices, index),
                        value,
                    })
                } else {
                    indices.insert(index, false);
                    continue;
                }
            }

            VecDiff::UpdateAt { index, value } => {
                if let Some(value) = world.run_system_with(system, value.clone()).ok().and_then(|output| f(value, output))
                {
                    if indices[index] {
                        Some(VecDiff::UpdateAt {
                            index: find_index(&indices, index),
                            value,
                        })
                    } else {
                        indices[index] = true;
                        Some(VecDiff::InsertAt {
                            index: find_index(&indices, index),
                            value,
                        })
                    }
                } else {
                    if indices[index] {
                        indices[index] = false;
                        Some(VecDiff::RemoveAt {
                            index: find_index(indices, index),
                        })
                    } else {
                        continue;
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
                        old_index: find_index(indices, old_index),
                        new_index: find_index(indices, new_index),
                    })
                } else {
                    indices.insert(new_index, false);
                    continue;
                }
            }

            VecDiff::RemoveAt { index } => {
                if indices.remove(index) {
                    Some(VecDiff::RemoveAt {
                        index: find_index(&indices, index),
                    })
                } else {
                    continue;
                }
            }

            VecDiff::Push { value } => {
                if let Some(value) = world.run_system_with(system, value.clone()).ok().and_then(|output| f(value, output))
                {
                    indices.push(true);
                    Some(VecDiff::Push { value })
                } else {
                    indices.push(false);
                    continue;
                }
            }

            VecDiff::Pop {} => {
                if indices.pop().expect("Cannot pop from empty vec") {
                    Some(VecDiff::Pop {})
                } else {
                    continue;
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

/// Extension trait providing combinator methods for types implementing [`SignalVec`] and [`Clone`].
pub trait SignalVecExt: SignalVec {
    /// Registers a system that runs for each batch of `VecDiff`s emitted by this signal.
    ///
    /// The provided system `F` takes `In<Vec<VecDiff<Self::Item>>>` and returns `()`.
    /// This method consumes the signal stream at this point; no further signals are propagated.
    ///
    /// Returns a [`ForEachVec`] node representing this terminal operation.
    /// Call `.register(world)` on the result to activate the chain and get a [`SignalHandle`].
    fn for_each<O, F, M>(self, system: F) -> ForEach<Self>
    where
        Self: Sized,
        Self::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
        O: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
        F: IntoSystem<In<Vec<VecDiff<Self::Item>>>, O, M> + Send + Sync + Clone + 'static,
        M: SSs,
    {
        ForEach {
            upstream: self,
            signal: lazy_signal_from_system(system),
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
    // F is IntoSystem
    where
        Self: Sized,
        Self::Item: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
        O: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
        F: IntoSystem<In<Self::Item>, O, M> + Send + Sync + 'static,
        M: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| -> SignalSystem {
            let system = world.register_system(system);
            let wrapper_system = move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                                       world: &mut World|
                  -> Option<Vec<VecDiff<O>>> {
                let mut output: Vec<VecDiff<O>> = Vec::new();

                for diff in diffs {
                    let diff_option: Option<VecDiff<O>> =
                        match diff {
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
            };
            let signal = register_signal::<_, Vec<VecDiff<O>>, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal.into()
        });

        Map {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    fn filter_map<O, F, M>(self, system: F) -> FilterMap<Self, Self::Item>
    where
        Self: Sized,
        Self::Item: Reflect + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
        O: Reflect + FromReflect + GetTypeRegistration + Typed + SSs,
        F: IntoSystem<In<Self::Item>, Option<O>, M> + Send + Sync + Clone + 'static,
        M: SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| -> SignalSystem {
            let system = world.register_system(system);
            let wrapper_system = move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut indices: Local<Vec<bool>>| {
                filter_helper(
                    world,
                    diffs,
                    system,
                    |_, mapped| mapped,
                    &mut indices,
                )
            };
            let signal = register_signal::<_, Vec<VecDiff<O>>, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal.into()
        });
        FilterMap {
            upstream: self,
            signal,
            _marker: PhantomData,
        }
    }

    fn filter<F, M>(self, system: F) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Reflect + FromReflect + GetTypeRegistration + Typed + Clone + SSs,
        F: IntoSystem<In<Self::Item>, bool, M> + Send + Sync + Clone + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| -> SignalSystem {
            let system = world.register_system(system);
            let wrapper_system = move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut indices: Local<Vec<bool>>| {
                filter_helper(
                    world,
                    diffs,
                    system,
                    |item, include| {
                        if include {
                            Some(item)
                        } else {
                            None
                        }
                    },
                    &mut indices,
                )
            };
            let signal = register_signal::<_, Vec<VecDiff<Self::Item>>, _, _, _>(world, wrapper_system);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal.into()
        });
        Filter {
            upstream: self,
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
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
{
    guard: RwLockReadGuard<'a, MutableVecState<T>>,
}

impl<'a, T> Deref for MutableVecReadGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
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
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
{
    guard: RwLockWriteGuard<'a, MutableVecState<T>>,
}

impl<'a, T> MutableVecWriteGuard<'a, T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
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
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
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
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
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
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone,
{
    state: Arc<RwLock<MutableVecState<T>>>,
}

#[derive(Component)]
struct QueuedVecDiffs<T: FromReflect + GetTypeRegistration + Typed>(Vec<VecDiff<T>>);

impl<T, A: Clone + FromReflect + GetTypeRegistration + Typed + SSs> From<T> for MutableVec<A>
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

impl<T> MutableVec<T>
where
    T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone, // Removed PartialEq
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

    // --- Convenience methods (delegate to guards) ---

    /// Returns the number of elements in the vector (acquires read lock).
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Returns `true` if the vector contains no elements (acquires read lock).
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Returns a clone of the element at `index`, or `None` if out of bounds (acquires read lock).
    pub fn get(&self, index: usize) -> Option<T> {
        self.read().get(index).cloned()
    }

    /// Pushes a value to the end of the vector (acquires write lock).
    pub fn push(&self, value: T) {
        self.write().push(value);
    }

    /// Removes the last element from the vector and returns it (acquires write lock).
    pub fn pop(&self) -> Option<T> {
        self.write().pop()
    }

    /// Inserts an element at `index` (acquires write lock).
    pub fn insert(&self, index: usize, value: T) {
        self.write().insert(index, value);
    }

    /// Removes and returns the element at `index` (acquires write lock).
    pub fn remove(&self, index: usize) -> T {
        self.write().remove(index)
    }

    /// Removes all elements from the vector (acquires write lock).
    pub fn clear(&self) {
        self.write().clear();
    }

    /// Updates the element at `index` with a new `value` (acquires write lock).
    pub fn set(&self, index: usize, value: T) {
        self.write().set(index, value);
    }

    /// Moves an item from `old_index` to `new_index` (acquires write lock).
    pub fn move_item(&self, old_index: usize, new_index: usize) {
        self.write().move_item(old_index, new_index);
    }

    /// Replaces the entire contents of the vector (acquires write lock).
    pub fn replace(&self, values: Vec<T>) {
        self.write().replace(values);
    }

    /// Creates a [`SourceVec<T>`] signal linked to this `MutableVec`.
    pub fn signal_vec(&self) -> Source<T> {
        let signal = LazySignal::new(clone!((self.state => state) move |world: &mut World| {
            let entity = LazyEntity::new();
            let signal = register_signal::<_, Vec<VecDiff<T>>, _, _, _>(world, clone!((entity) move |_: In<()>, world: &mut World| {
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
            state.write().unwrap().signals.push(signal);
            signal
        }));

        Source {
            signal,
            _marker: PhantomData,
        }
    }

    pub fn flush_into_world(&self, world: &mut World) {
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
    pub fn flush(&self) -> impl Command {
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
    struct SignalVecOutput<
        T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone + std::fmt::Debug,
    >(Vec<VecDiff<T>>);

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
        T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone + std::fmt::Debug,
    {
        debug!(
            "Capture SignalVec Output: Received {:?}, extending resource from {:?} with new diffs",
            diffs, output.0
        );
        output.0.extend(diffs);
    }

    fn get_signal_vec_output<
        T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone + std::fmt::Debug,
    >(
        world: &World,
    ) -> Vec<VecDiff<T>> {
        world.resource::<SignalVecOutput<T>>().0.clone()
    }

    fn clear_signal_vec_output<
        T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Clone + std::fmt::Debug,
    >(
        world: &mut World,
    ) {
        if let Some(mut output) = world.get_resource_mut::<SignalVecOutput<T>>() {
            output.0.clear();
        }
    }

    impl<T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + PartialEq + std::fmt::Debug>
        PartialEq for VecDiff<T>
    {
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
    impl<T: Reflect + FromReflect + GetTypeRegistration + Typed + SSs + Eq + std::fmt::Debug> Eq
        for VecDiff<T>
    {
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
