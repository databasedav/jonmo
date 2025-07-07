//! A `Signal` that represents a map of keys to values.

use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::{cmp::Ord, fmt, marker::PhantomData, ops::Deref};

use bevy_ecs::{prelude::*, system::SystemId};
use bevy_log::debug;
use bevy_platform::sync::*;

use super::{
    graph::*,
    signal::{Signal, SignalBuilder, SignalExt},
    signal_vec::{SignalVec, VecDiff},
    utils::*,
};

/// Describes the changes to a map.
///
/// This is used by [`SignalMap`] to efficiently represent changes.
#[derive(Debug)]
pub enum MapDiff<K, V> {
    /// Replaces the entire contents of the map.
    Replace {
        /// The new entries for the map.
        entries: Vec<(K, V)>,
    },
    /// Inserts a new entry into the map.
    Insert {
        /// The key of the new entry.
        key: K,
        /// The value of the new entry.
        value: V,
    },
    /// Updates an existing entry in the map.
    Update {
        /// The key of the entry to update.
        key: K,
        /// The new value for the entry.
        value: V,
    },
    /// Removes an entry from the map.
    Remove {
        /// The key of the entry to remove.
        key: K,
    },
    /// Removes all entries from the map.
    Clear,
}

impl<K: Clone, V: Clone> Clone for MapDiff<K, V> {
    fn clone(&self) -> Self {
        match self {
            MapDiff::Replace { entries } => MapDiff::Replace {
                entries: entries.clone(),
            },
            MapDiff::Insert { key, value } => MapDiff::Insert {
                key: key.clone(),
                value: value.clone(),
            },
            MapDiff::Update { key, value } => MapDiff::Update {
                key: key.clone(),
                value: value.clone(),
            },
            MapDiff::Remove { key } => MapDiff::Remove { key: key.clone() },
            MapDiff::Clear => MapDiff::Clear,
        }
    }
}

impl<K, A> MapDiff<K, A> {
    /// Maps the values of the diff, preserving the keys and structure.
    pub fn map<B, F>(self, mut callback: F) -> MapDiff<K, B>
    where
        F: FnMut(A) -> B,
    {
        match self {
            MapDiff::Replace { entries } => MapDiff::Replace {
                entries: entries.into_iter().map(|(k, v)| (k, callback(v))).collect(),
            },
            MapDiff::Insert { key, value } => MapDiff::Insert {
                key,
                value: callback(value),
            },
            MapDiff::Update { key, value } => MapDiff::Update {
                key,
                value: callback(value),
            },
            MapDiff::Remove { key } => MapDiff::Remove { key },
            MapDiff::Clear => MapDiff::Clear,
        }
    }
}

/// A reactive map that yields [`MapDiff<K, V>`] to describe changes.
pub trait SignalMap: SSs {
    /// The key type of the map.
    type Key;
    /// The value type of the map.
    type Value;

    /// Registers the systems associated with this signal map, consuming its boxed form.
    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle;

    /// Registers the systems associated with this signal map.
    fn register_signal_map(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.boxed().register_boxed_signal_map(world)
    }
}

impl<K: 'static, V: 'static> SignalMap for Box<dyn SignalMap<Key = K, Value = V> + Send + Sync> {
    type Key = K;
    type Value = V;

    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        let inner_box: Box<dyn SignalMap<Key = K, Value = V> + Send + Sync> = *self;
        inner_box.register_boxed_signal_map(world)
    }
}

// --- Combinator Structs ---

/// A terminal node that runs a system for each batch of `MapDiff`s.
/// Created by the [`SignalMapExt::for_each`] method.
#[derive(Clone)]
pub struct ForEach<Upstream, O> {
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
}

impl<Upstream, O> Signal for ForEach<Upstream, O>
where
    Upstream: SignalMap,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// A node that maps the values of a `SignalMap`.
/// Created by the [`SignalMapExt::map_value`] method.
#[derive(Clone)]
pub struct MapValue<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> SignalMap for MapValue<Upstream, O>
where
    Upstream: SignalMap,
    O: 'static,
{
    type Key = Upstream::Key;
    type Value = O;

    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// A node that maps the values of a `SignalMap` to new `Signal`s.
/// Created by the [`SignalMapExt::map_value_signal`] method.
#[derive(Clone)]
pub struct MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    S: Signal,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S> SignalMap for MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    S: Signal,
{
    type Key = Upstream::Key;
    type Value = S::Item;

    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// A node that converts a `SignalMap` into a `SignalVec` of its keys.
/// Created by the [`SignalMapExt::to_signal_vec_keys`] method.
#[derive(Clone)]
pub struct MapKeys<Upstream>
where
    Upstream: SignalMap,
{
    signal: ForEach<Upstream, Vec<VecDiff<Upstream::Key>>>,
}

impl<Upstream> SignalVec for MapKeys<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: Ord + Clone,
{
    type Item = Upstream::Key;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// A node that converts a `SignalMap` into a `SignalVec` of its entries.
/// Created by the [`SignalMapExt::to_signal_vec_entries`] method.
#[derive(Clone)]
pub struct MapEntries<Upstream>
where
    Upstream: SignalMap,
{
    signal: ForEach<Upstream, Vec<VecDiff<(Upstream::Key, Upstream::Value)>>>,
}

impl<Upstream> SignalVec for MapEntries<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: Ord + Clone,
    Upstream::Value: Clone,
{
    type Item = (Upstream::Key, Upstream::Value);

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// A `Signal` that tracks the value of a single key in a `SignalMap`.
/// Created by the [`SignalMapExt::key_cloned`] method.
#[derive(Clone)]
pub struct Key<Upstream>
where
    Upstream: SignalMap,
{
    signal: ForEach<Upstream, Option<Upstream::Value>>,
}

impl<Upstream> Signal for Key<Upstream>
where
    Upstream: SignalMap,
{
    type Item = Option<Upstream::Value>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Extension trait for `SignalMap`.
pub trait SignalMapExt: SignalMap {
    /// Runs a system for each batch of `MapDiff`s. This is a terminal operation.
    fn for_each<O, IOO, F, M>(self, system: F) -> ForEach<Self, O>
    where
        Self: Sized,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Vec<MapDiff<Self::Key, Self::Value>>>, IOO, M> + SSs,
    {
        ForEach {
            upstream: self,
            signal: lazy_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    /// Creates a new `SignalMap` by mapping the values of this one.
    fn map_value<O, F, M>(self, system: F) -> MapValue<Self, O>
    where
        Self: Sized,
        Self::Key: Clone,
        O: Clone + 'static,
        F: IntoSystem<In<Self::Value>, O, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system_id = world.register_system(system);

            let processor = move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, world: &mut World| {
                let mut out_diffs = Vec::with_capacity(diffs.len());

                for diff in diffs {
                    let mapped_diff = match diff {
                        MapDiff::Replace { entries } => Some(MapDiff::Replace {
                            entries: entries
                                .into_iter()
                                .filter_map(|(k, v)| world.run_system_with(system_id, v).ok().map(|new_v| (k, new_v)))
                                .collect(),
                        }),
                        MapDiff::Insert { key, value } => world
                            .run_system_with(system_id, value)
                            .ok()
                            .map(|new_value| MapDiff::Insert { key, value: new_value }),
                        MapDiff::Update { key, value } => world
                            .run_system_with(system_id, value)
                            .ok()
                            .map(|new_value| MapDiff::Update { key, value: new_value }),
                        MapDiff::Remove { key } => Some(MapDiff::Remove { key }),
                        MapDiff::Clear => Some(MapDiff::Clear),
                    };

                    if let Some(diff) = mapped_diff {
                        out_diffs.push(diff);
                    }
                }

                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            };

            let SignalHandle(signal) = self
                .for_each::<Vec<MapDiff<Self::Key, O>>, _, _, _>(processor)
                .register(world);

            world.entity_mut(*signal).add_child(system_id.entity());

            signal
        });
        MapValue {
            signal,
            _marker: PhantomData,
        }
    }

    /// Creates a new `SignalMap` by mapping each value to a new `Signal`.
    ///
    /// For each item in the source map, the provided `system` is run to produce an
    /// inner `Signal`. The output map's value for a given key becomes the latest
    /// value emitted by the corresponding inner `Signal`.
    ///
    /// This is useful for creating dynamic maps where each value has its own
    /// independent, reactive state.
    fn map_value_signal<S, F, M>(self, system: F) -> MapValueSignal<Self, S>
    where
        Self: Sized,
        Self::Key: Ord + Clone + SSs,
        Self::Value: 'static,
        S: Signal + Clone + 'static,
        S::Item: Clone + SSs,
        F: IntoSystem<In<Self::Value>, S, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            // Helper function to spawn a processor for an inner signal.
            // It gets the signal's initial value and sets up a listener for future updates.
            fn spawn_inner_processor<K, S>(
                world: &mut World,
                output_system: SignalSystem,
                key: K,
                inner_signal: S,
            ) -> (SignalHandle, S::Item, SignalSystem)
            where
                K: Clone + SSs,
                S: Signal + Clone + 'static,
                S::Item: Clone + SSs,
            {
                use core::any::Any;

                // Get the canonical System ID for the signal type for memoization purposes.
                let temp_handle_for_id = inner_signal.clone().register(world);
                let inner_signal_id = *temp_handle_for_id;

                // Get initial value by polling a temporary signal chain that only emits once.
                let temp_handle_for_poll = inner_signal.clone().first().register(world);
                let initial_value = poll_signal(world, *temp_handle_for_poll)
                    .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<S::Item>().ok().map(|b| *b))
                    .expect("map_value_signal's inner signal must have an initial value");

                // Clean up temporary registrations used for ID and polling.
                temp_handle_for_id.cleanup(world);
                temp_handle_for_poll.cleanup(world);

                // Create the persistent processor that listens for future updates.
                // When the inner signal updates, it sends a `MapDiff::Update` to the main output.
                let processor_handle = inner_signal
                    .map(move |In(value): In<S::Item>, world: &mut World| {
                        let diffs = vec![MapDiff::Update {
                            key: key.clone(),
                            value,
                        }];
                        process_signals(world, [output_system], Box::new(diffs));
                        None::<()> // This system doesn't output to a further signal chain.
                    })
                    .register(world);

                (processor_handle, initial_value, inner_signal_id)
            }

            // The public-facing system that emits the final diffs.
            let output_system =
                lazy_signal_from_system(|In(diffs): In<Vec<MapDiff<Self::Key, S::Item>>>| diffs).register(world);
            world.entity_mut(*output_system).insert(Upstream(Default::default()));

            let manager_entity = LazyEntity::new();
            let manager_system_id = world.register_system(system);

            #[derive(Component)]
            struct ManagerState<K: Ord + Clone + SSs, S: Signal + Clone + 'static> {
                signal_processors: BTreeMap<K, SignalHandle>,
                inner_signal_ids: BTreeMap<K, SignalSystem>,
                _marker: PhantomData<S>,
            }

            // The main orchestration system that processes diffs from the upstream SignalMap.
            let manager_processor = clone!((manager_entity) move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, world: &mut World| {
                let mut out_diffs: Vec<MapDiff<Self::Key, S::Item>> = Vec::new();
                let manager = manager_entity.get();
                let mut state = world.get_mut::<ManagerState<Self::Key, S>>(manager).unwrap();

                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            for (_, handle) in state.signal_processors.drain(..) {
                                handle.cleanup(world);
                            }
                            state.inner_signal_ids.clear();

                            let new_entries: Vec<(Self::Key, S::Item)> = entries.into_iter().filter_map(|(key, value)| {
                                world.run_system_with(manager_system_id, value).ok().map(|inner_signal: S| {
                                    let (handle, initial_value, inner_signal_id) =
                                        spawn_inner_processor(world, output_system, key.clone(), inner_signal);
                                    state.signal_processors.insert(key.clone(), handle);
                                    state.inner_signal_ids.insert(key.clone(), inner_signal_id);
                                    (key, initial_value)
                                })
                            }).collect();
                            out_diffs.push(MapDiff::Replace { entries: new_entries });
                        },
                        MapDiff::Insert { key, value } => {
                            if let Ok(inner_signal) = world.run_system_with(manager_system_id, value) {
                                let (handle, initial_value, inner_signal_id) =
                                    spawn_inner_processor(world, output_system, key.clone(), inner_signal);
                                state.signal_processors.insert(key.clone(), handle);
                                state.inner_signal_ids.insert(key.clone(), inner_signal_id);
                                out_diffs.push(MapDiff::Insert { key, value: initial_value });
                            }
                        },
                        MapDiff::Update { key, value } => {
                            if let Ok(new_inner_signal) = world.run_system_with(manager_system_id, value) {
                                let temp_handle = new_inner_signal.clone().register(world);
                                let new_inner_signal_id = *temp_handle;
                                temp_handle.cleanup(world);

                                if Some(&new_inner_signal_id) == state.inner_signal_ids.get(&key) {
                                    continue; // The inner signal is the same, do nothing.
                                }

                                if let Some(old_handle) = state.signal_processors.remove(&key) {
                                    old_handle.cleanup(world);
                                }

                                let (handle, initial_value, _) =
                                    spawn_inner_processor(world, output_system, key.clone(), new_inner_signal);

                                state.signal_processors.insert(key.clone(), handle);
                                state.inner_signal_ids.insert(key.clone(), new_inner_signal_id);
                                out_diffs.push(MapDiff::Update { key, value: initial_value });
                            }
                        },
                        MapDiff::Remove { key } => {
                            if let Some(handle) = state.signal_processors.remove(&key) {
                                handle.cleanup(world);
                            }
                            state.inner_signal_ids.remove(&key);
                            out_diffs.push(MapDiff::Remove { key });
                        },
                        MapDiff::Clear => {
                            for (_, handle) in state.signal_processors.drain(..) {
                                handle.cleanup(world);
                            }
                            state.inner_signal_ids.clear();
                            out_diffs.push(MapDiff::Clear);
                        }
                    }
                }

                if !out_diffs.is_empty() {
                    process_signals(world, [output_system], Box::new(out_diffs));
                }
                None::<()>
            });

            // Register the main processor and set up its state and lifecycle.
            let manager_handle = self.for_each(manager_processor).register(world);
            manager_entity.set(*manager_handle);

            world.entity_mut(*manager_handle).insert(ManagerState::<Self::Key, S> {
                signal_processors: BTreeMap::new(),
                inner_signal_ids: BTreeMap::new(),
                _marker: PhantomData,
            });

            world
                .entity_mut(*output_system)
                .add_child(*manager_handle)
                .add_child(manager_system_id.entity());

            output_system
        });

        MapValueSignal {
            signal,
            _marker: PhantomData,
        }
    }

    /// Converts the `SignalMap` into a `SignalVec` of its keys, sorted.
    fn signal_vec_keys(self) -> MapKeys<Self>
    where
        Self: Sized,
        Self::Key: Ord + Clone + SSs,
        Self::Value: 'static,
    {
        let signal = self.for_each(
            move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, mut keys: Local<Vec<Self::Key>>| {
                let mut out = vec![];
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            *keys = entries.into_iter().map(|(k, _)| k).collect();
                            // Assume BTreeMap source is sorted.
                            out.push(VecDiff::Replace { values: keys.clone() });
                        }
                        MapDiff::Insert { key, .. } => {
                            let index = keys.binary_search(&key).unwrap_err();
                            keys.insert(index, key.clone());
                            out.push(VecDiff::InsertAt { index, value: key });
                        }
                        MapDiff::Update { .. } => {
                            // No change to keys vector
                        }
                        MapDiff::Remove { key } => {
                            if let Ok(index) = keys.binary_search(&key) {
                                keys.remove(index);
                                out.push(VecDiff::RemoveAt { index });
                            }
                        }
                        MapDiff::Clear => {
                            keys.clear();
                            out.push(VecDiff::Clear);
                        }
                    }
                }
                if out.is_empty() { None } else { Some(out) }
            },
        );
        MapKeys { signal }
    }

    /// Converts the `SignalMap` into a `SignalVec` of its `(key, value)` entries, sorted by key.
    fn signal_vec_entries(self) -> MapEntries<Self>
    where
        Self: Sized,
        Self::Key: Ord + Clone + SSs,
        Self::Value: Clone + SSs,
    {
        let signal = self.for_each(
            move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, mut keys: Local<Vec<Self::Key>>| {
                let mut out_diffs = Vec::new();
                for diff in diffs {
                    let vec_diff = match diff {
                        MapDiff::Replace { entries } => {
                            *keys = entries.iter().map(|(k, _)| k.clone()).collect();
                            VecDiff::Replace { values: entries }
                        }
                        MapDiff::Insert { key, value } => {
                            let index = keys.binary_search(&key).unwrap_err();
                            keys.insert(index, key.clone());
                            VecDiff::InsertAt {
                                index,
                                value: (key, value),
                            }
                        }
                        MapDiff::Update { key, value } => {
                            let index = keys.binary_search(&key).unwrap();
                            VecDiff::UpdateAt {
                                index,
                                value: (key, value),
                            }
                        }
                        MapDiff::Remove { key } => {
                            let index = keys.binary_search(&key).unwrap();
                            keys.remove(index);
                            VecDiff::RemoveAt { index }
                        }
                        MapDiff::Clear => {
                            keys.clear();
                            VecDiff::Clear
                        }
                    };
                    out_diffs.push(vec_diff);
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            },
        );
        MapEntries { signal }
    }

    /// Creates a `Signal` that tracks the value of a specific key.
    fn key(self, key: Self::Key) -> Key<Self>
    where
        Self: Sized,
        Self::Key: PartialEq + Clone + SSs,
        Self::Value: Clone + SSs,
    {
        Key {
            signal: self.for_each(move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>| {
                let mut changed = None;
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            changed = Some(entries.into_iter().find(|(k, _)| k == &key).map(|(_, v)| v));
                        }
                        MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                            if key == key {
                                changed = Some(Some(value));
                            }
                        }
                        MapDiff::Remove { key } => {
                            if key == key {
                                changed = Some(None);
                            }
                        }
                        MapDiff::Clear => {
                            changed = Some(None);
                        }
                    }
                }

                changed
            }),
        }
    }

    /// Creates a dynamic `SignalMap` that is Send + 'static.
    fn boxed(self) -> Box<dyn SignalMap<Key = Self::Key, Value = Self::Value>>
    where
        Self: Sized + Send,
    {
        Box::new(self)
    }

    /// Registers the `SignalMap` and returns a handle for cleanup.
    fn register(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.register_signal_map(world)
    }
}

impl<T: ?Sized> SignalMapExt for T where T: SignalMap {}

// --- MutableBTreeMap ---

/// Internal state for `MutableBTreeMap`.
struct MutableBTreeMapState<K, V> {
    map: BTreeMap<K, V>,
    pending_diffs: Vec<MapDiff<K, V>>,
    signal: Option<LazySignal>,
}

/// A read guard for `MutableBTreeMap`.
pub struct MutableBTreeMapReadGuard<'a, K, V> {
    guard: RwLockReadGuard<'a, MutableBTreeMapState<K, V>>,
}

impl<'a, K: Ord, V> Deref for MutableBTreeMapReadGuard<'a, K, V> {
    type Target = BTreeMap<K, V>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

/// A write guard for `MutableBTreeMap`. Changes automatically queue diffs.
pub struct MutableBTreeMapWriteGuard<'a, K: Ord, V> {
    guard: RwLockWriteGuard<'a, MutableBTreeMapState<K, V>>,
}

impl<'a, K: Ord + Clone, V> MutableBTreeMapWriteGuard<'a, K, V> {
    /// Inserts a key-value pair. If the key already existed, it's an update.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let old = self.guard.map.insert(key.clone(), value);
        if let Some(old_value) = old {
            self.guard.pending_diffs.push(MapDiff::Update {
                key,
                value: self.guard.map.get(&key).unwrap().clone(),
            });
            Some(old_value)
        } else {
            self.guard.pending_diffs.push(MapDiff::Insert {
                key,
                value: self.guard.map.get(&key).unwrap().clone(),
            });
            None
        }
    }

    /// Removes a key, returning the value at the key if the key was previously in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let result = self.guard.map.remove(key);
        if result.is_some() {
            self.guard.pending_diffs.push(MapDiff::Remove { key: key.clone() });
        }
        result
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&mut self) {
        if !self.guard.map.is_empty() {
            self.guard.map.clear();
            self.guard.pending_diffs.push(MapDiff::Clear);
        }
    }

    /// Replaces the entire map with new entries.
    pub fn replace(&mut self, entries: Vec<(K, V)>) {
        self.guard.map = entries.iter().cloned().collect();
        self.guard.pending_diffs.push(MapDiff::Replace { entries });
    }
}

// Immutable deref for write guard
impl<'a, K: Ord, V> Deref for MutableBTreeMapWriteGuard<'a, K, V> {
    type Target = BTreeMap<K, V>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

/// A mutable `BTreeMap` that tracks changes as `MapDiff`s.
#[derive(Clone)]
pub struct MutableBTreeMap<K, V> {
    state: Arc<RwLock<MutableBTreeMapState<K, V>>>,
}

#[derive(Component)]
struct QueuedMapDiffs<K, V>(Vec<MapDiff<K, V>>);

impl<K: Ord, V> Default for MutableBTreeMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, K: Ord, V: Clone> From<T> for MutableBTreeMap<K, V>
where
    BTreeMap<K, V>: From<T>,
{
    #[inline]
    fn from(values: T) -> Self {
        Self::with_values(values.into())
    }
}

impl<K: Ord, V> MutableBTreeMap<K, V> {
    /// Creates a new, empty `MutableBTreeMap`.
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableBTreeMapState {
                map: BTreeMap::new(),
                pending_diffs: Vec::new(),
                signal: None,
            })),
        }
    }

    /// Creates a new `MutableBTreeMap` initialized with the given values.
    pub fn with_values(map: BTreeMap<K, V>) -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableBTreeMapState {
                pending_diffs: Vec::new(),
                map,
                signal: None,
            })),
        }
    }

    /// Acquires a read lock for immutable access.
    pub fn read(&self) -> MutableBTreeMapReadGuard<'_, K, V> {
        MutableBTreeMapReadGuard {
            guard: self.state.read().unwrap(),
        }
    }

    /// Acquires a write lock for mutable access.
    pub fn write(&self) -> MutableBTreeMapWriteGuard<'_, K, V> {
        MutableBTreeMapWriteGuard {
            guard: self.state.write().unwrap(),
        }
    }

    /// Sends any pending `MapDiff`s to the signal system.
    pub fn flush_into_world(&self, world: &mut World)
    where
        K: SSs,
        V: SSs,
    {
        let mut state = self.state.write().unwrap();
        if state.pending_diffs.is_empty() {
            return;
        }

        if let Some(lazy_signal) = &state.signal {
            if let LazySystem::Registered(signal_system) = *lazy_signal.inner.system.read().unwrap() {
                if let Ok(mut entity) = world.get_entity_mut(*signal_system) {
                    if let Some(mut queued_diffs) = entity.get_mut::<QueuedMapDiffs<K, V>>() {
                        queued_diffs.0.append(&mut state.pending_diffs);
                    }
                }
            }
        }
        // Always clear diffs after attempting to flush
        state.pending_diffs.clear();
    }

    /// Returns a command to flush pending diffs in the Bevy command queue.
    pub fn flush(&self) -> impl Command
    where
        K: SSs,
        V: SSs,
    {
        let self_ = self.clone();
        move |world: &mut World| self_.flush_into_world(world)
    }
}

/// A source node for a `SignalMap` chain.
#[derive(Clone)]
pub struct Source<K, V> {
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> (K, V)>,
}

impl<K, V> SignalMap for Source<K, V>
where
    K: SSs,
    V: SSs,
{
    type Key = K;
    type Value = V;

    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

impl<K, V> MutableBTreeMap<K, V>
where
    K: Ord + Clone + SSs,
    V: Clone + SSs,
{
    /// Returns the canonical `SignalMap` source for this `MutableBTreeMap`.
    pub fn signal_map(&self) -> Source<K, V> {
        let mut state = self.state.write().unwrap();
        if let Some(lazy_signal) = &state.signal {
            return Source {
                signal: lazy_signal.clone(),
                _marker: PhantomData,
            };
        }

        let signal = LazySignal::new(clone!((self.state => state) move |world: &mut World| {
            let self_entity = LazyEntity::new();
            let source_system_logic = clone!((self_entity) move |_: In<()>, world: &mut World| {
                if let Ok(mut diffs) = world.get_mut::<QueuedMapDiffs<K, V>>(self_entity.get()) {
                    if diffs.0.is_empty() { None } else { Some(diffs.0.drain(..).collect()) }
                } else {
                    None
                }
            });

            let signal_system =
                register_signal::<(), Vec<MapDiff<K, V>>, _, _, _>(world, source_system_logic);
            self_entity.set(*signal_system);

            let initial_map = state.read().unwrap().map.clone();
            let initial_diffs = if !initial_map.is_empty() {
                vec![MapDiff::Replace {
                    entries: initial_map.into_iter().collect(),
                }]
            } else {
                vec![]
            };
            world
                .entity_mut(*signal_system)
                .insert(QueuedMapDiffs(initial_diffs));

            signal_system
        }));

        state.signal = Some(signal.clone());
        Source {
            signal,
            _marker: PhantomData,
        }
    }

    /// Convenience method to get a `SignalVec` of the map's keys.
    pub fn signal_vec_keys(&self) -> MapKeys<Source<K, V>> {
        self.signal_map().signal_vec_keys()
    }

    /// Convenience method to get a `SignalVec` of the map's entries.
    pub fn signal_vec_entries(&self) -> MapEntries<Source<K, V>> {
        self.signal_map().signal_vec_entries()
    }
}
