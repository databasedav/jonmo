//! Data structures and combinators for constructing reactive [`System`] dependency graphs on top of
//! [`BTreeMap`] mutations, see [`MutableBTreeMap`] and [`SignalMapExt`].

use crate::signal_vec::Replayable;

use super::{
    graph::*,
    signal::*,
    signal_vec::{SignalVec, VecDiff},
    utils::*,
};
use alloc::collections::BTreeMap;
use bevy_ecs::prelude::*;
use bevy_platform::{
    prelude::*,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use core::{fmt, marker::PhantomData, ops::Deref};

/// Describes the mutations made to the underlying [`MutableBTreeMap`] that are piped to
/// downstream [`SignalMap`]s.
#[allow(missing_docs)]
pub enum MapDiff<K, V> {
    Replace { entries: Vec<(K, V)> },
    Insert { key: K, value: V },
    Update { key: K, value: V },
    Remove { key: K },
    Clear,
}

impl<K, V> Clone for MapDiff<K, V>
where
    K: Clone,
    V: Clone,
{
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

impl<K, V> fmt::Debug for MapDiff<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapDiff::Replace { entries } => f.debug_struct("Replace").field("entries", entries).finish(),
            MapDiff::Insert { key, value } => f
                .debug_struct("Insert")
                .field("key", key)
                .field("value", value)
                .finish(),
            MapDiff::Update { key, value } => f
                .debug_struct("Update")
                .field("key", key)
                .field("value", value)
                .finish(),
            MapDiff::Remove { key } => f.debug_struct("Remove").field("key", key).finish(),
            MapDiff::Clear => f.debug_struct("Clear").finish(),
        }
    }
}

impl<K, V> MapDiff<K, V> {
    /// Maps the `value` part of the diff, preserving the key and diff type.
    pub fn map_value<O, F>(self, mut callback: F) -> MapDiff<K, O>
    where
        F: FnMut(V) -> O,
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
            MapDiff::Clear => MapDiff::Clear {},
        }
    }
}

/// Monadic registration facade for structs that encapsulate some [`System`] which is a valid member
/// of the signal graph downstream of some source [`MutableBTreeMap`]; this is similar to
/// [`Signal`] but critically requires that the [`System`] outputs [`Option<MapDiff<Self::Key,
/// Self::Value>>`].
pub trait SignalMap: SSs {
    #[allow(missing_docs)]
    type Key;
    #[allow(missing_docs)]
    type Value;

    /// Registers the [`System`]s associated with this [`SignalMap`] by consuming its boxed form.
    ///
    /// All concrete signal map types must implement this method.
    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle;

    /// Registers the [`System`]s associated with this [`SignalMap`].
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
        (*self).register_boxed_signal_map(world)
    }
}

/// Signal graph node which applies a [`System`] directly to the "raw" [`Vec<MapDiff>`]s of its
/// upstream, see [.for_each](SignalMapExt::for_each).
#[derive(Clone)]
pub struct ForEach<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
}

impl<Upstream, O> Signal for ForEach<Upstream, O>
where
    Upstream: SignalMap,
    O: 'static,
{
    type Item = O;
    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register_signal_map(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        signal.into()
    }
}

/// Signal graph node which applies a [`System`] to each [`Value`](SignalMap::Value) of its
/// upstream, see [`.map_value`](SignalMapExt::map_value).
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

/// Signal graph node which applies a [`System`] to each [`Value`](SignalMap::Value) of its
/// upstream, forwarding the output of each resulting [`Signal`], see
/// [`.map_value`](SignalMapExt::map_value).
#[derive(Clone)]
pub struct MapValueSignal<Upstream, S: Signal> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S: Signal> SignalMap for MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    S: Signal + 'static,
    S::Item: Clone + SSs,
{
    type Key = Upstream::Key;
    type Value = S::Item;

    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which maps its upstream [`SignalVec`] to a [`Key`](SignalMap::Key)-lookup
/// [`Signal`], see [`.key`](SignalMapExt::key).
#[derive(Clone)]
pub struct Key<Upstream>
where
    Upstream: SignalMap,
{
    inner: ForEach<Upstream, Option<Upstream::Value>>,
}

impl<Upstream> Signal for Key<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: 'static,
{
    type Item = Option<Upstream::Value>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.inner.register_signal(world)
    }
}

/// Signal graph node with no upstreams which outputs some [`MutableBTreeMap`]'s sorted
/// [`Key`](SignalMap::Key)s as a [`SignalVec`], see
/// [`.signal_vec_keys`](MutableBTreeMap::signal_vec_keys).
#[derive(Clone)]
pub struct SignalVecKeys<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for SignalVecKeys<Upstream>
where
    Upstream: SignalMap,
{
    type Item = Upstream::Key;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which maps its upstream [`MutableBTreeMap`] to a [`SignalVec`] of its sorted
/// `(key, value)`s, see [`.signal_vec_entries`](MutableBTreeMap::signal_vec_entries).
#[derive(Clone)]
pub struct SignalVecEntries<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for SignalVecEntries<Upstream>
where
    Upstream: SignalMap,
{
    type Item = (Upstream::Key, Upstream::Value);

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Extension trait providing combinator methods for [`SignalMap`]s.
pub trait SignalMapExt: SignalMap {
    /// Pass the "raw" [`Vec<MapDiff<Self::Key, Self::Value>>`] output of this [`SignalMap`] to a
    /// [`System`], continuing propagation if the [`System`] returns [`Some`] or terminating for the
    /// frame if it returns [`None`]. This transforms the `SignalMap` into a `Signal`. Unlike
    /// most other [`SignalMap`] methods, [`.for_each`](SignalMapExt::for_each), returns a
    /// [`Signal`], not a [`SignalMap`], since the output type need not be an
    /// [`Option<Vec<MapDiff>>`]. If the [`System`] logic is infallible, wrapping the result in
    /// an option is unnecessary.
    fn for_each<O, IOO, F, M>(self, system: F) -> ForEach<Self, O>
    where
        Self: Sized,
        Self::Key: 'static,
        Self::Value: 'static,
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

    /// Pass each [`Value`](SignalMap::Value) of this [`SignalMap`] to a [`System`], transforming
    /// it.
    ///
    /// # Example
    /// ```no_run
    /// MutableBTreeMap::from([(1, 2), (3, 4)]).signal_map().map_value(|In(x)| x * 2); // outputs `SignalMap -> {1: 2, 3: 4}`
    /// ```
    fn map_value<O, F, M>(self, system: F) -> MapValue<Self, O>
    where
        Self: Sized,
        Self::Key: Clone + 'static,
        Self::Value: 'static,
        O: Clone + 'static,
        F: IntoSystem<In<Self::Value>, O, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system_id = world.register_system(system);
            let upstream_handle = self.register_signal_map(world);

            let processor_logic = move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, world: &mut World| {
                let mut out_diffs = Vec::with_capacity(diffs.len());
                for diff in diffs {
                    let new_diff = diff.map_value(|v| world.run_system_with(system_id, v).unwrap());
                    out_diffs.push(new_diff);
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            };

            let processor_handle =
                lazy_signal_from_system::<_, Vec<MapDiff<Self::Key, O>>, _, _, _>(processor_logic).register(world);
            world.entity_mut(*processor_handle).add_child(system_id.entity());
            pipe_signal(world, *upstream_handle, processor_handle);
            processor_handle
        });

        MapValue {
            signal,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Value`](SignalMap::Value) of this [`SignalMap`] to a [`System`] that produces a
    /// [`Signal`], forwarding the output of each resulting [`Signal`].
    ///
    /// # Example
    /// ```no_run
    /// MutableBTreeMap::from([(1, 2), (3, 4)]).signal_map()
    ///     .map_value_signal(|In(x)|
    ///         SignalBuilder::from_system(move |_: In<()>| x * 2).dedupe()
    ///     ); // outputs `SignalMap -> {1: 4, 3: 8}`
    /// ```
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
            let factory_system_id = world.register_system(system);

            let state_and_queue_entity = world.spawn_empty().id();

            let output_system_handle = SignalBuilder::from_system::<Vec<MapDiff<Self::Key, S::Item>>, _, _, _>(
                move |_: In<()>, world: &mut World| {
                    if let Some(mut diffs) = world.get_mut::<QueuedMapDiffs<Self::Key, S::Item>>(state_and_queue_entity)
                    {
                        if diffs.0.is_empty() {
                            None
                        } else {
                            Some(diffs.0.drain(..).collect())
                        }
                    } else {
                        None
                    }
                },
            )
            .register(world);

            fn spawn_processor<K: Clone + SSs, V: Clone + SSs>(
                world: &mut World,
                queue_entity: Entity,
                output_system: SignalSystem,
                key: K,
                inner_signal: impl Signal<Item = V> + Clone + 'static,
            ) -> (SignalHandle, SignalSystem, V) {
                let inner_signal_id = inner_signal.clone().register(world);

                let temp_handle = inner_signal.clone().first().register(world);
                let initial_value = poll_signal(world, *temp_handle)
                    .and_then(downcast_any_clone::<V>)
                    .expect("map_value_signal's inner signal must emit an initial value");
                temp_handle.cleanup(world);

                let processor_handle = inner_signal
                    .map(move |In(value): In<V>, world: &mut World| {
                        if let Some(mut queue) = world.get_mut::<QueuedMapDiffs<K, V>>(queue_entity) {
                            queue.0.push(MapDiff::Update {
                                key: key.clone(),
                                value,
                            });
                            process_signals(world, [output_system], Box::new(()));
                        }
                    })
                    .register(world);

                (processor_handle, *inner_signal_id, initial_value)
            }

            #[derive(Component)]
            struct ManagerState<K, S: Signal> {
                signals: BTreeMap<K, (SignalHandle, SignalSystem)>,
                _phantom: PhantomData<S>,
            }

            let output_system_handle_clone = output_system_handle.clone();
            let manager_system_logic = move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, world: &mut World| {
                let mut new_map_diffs = Vec::new();

                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            let old_signals = {
                                let mut state = world
                                    .get_mut::<ManagerState<Self::Key, S>>(state_and_queue_entity)
                                    .unwrap();
                                core::mem::take(&mut state.signals)
                            };

                            for (_, (handle, _)) in old_signals {
                                handle.cleanup(world);
                            }

                            let mut new_signals = BTreeMap::new();
                            let mut new_entries_for_diff = Vec::with_capacity(entries.len());
                            for (key, value) in entries {
                                if let Ok(inner_signal) = world.run_system_with(factory_system_id, value) {
                                    let (handle, id, initial_value) = spawn_processor(
                                        world,
                                        state_and_queue_entity,
                                        *output_system_handle_clone,
                                        key.clone(),
                                        inner_signal,
                                    );
                                    new_signals.insert(key.clone(), (handle, id));
                                    new_entries_for_diff.push((key, initial_value));
                                }
                            }

                            world
                                .get_mut::<ManagerState<Self::Key, S>>(state_and_queue_entity)
                                .unwrap()
                                .signals = new_signals;

                            if !new_entries_for_diff.is_empty() {
                                new_map_diffs.push(MapDiff::Replace {
                                    entries: new_entries_for_diff,
                                });
                            }
                        }
                        MapDiff::Insert { key, value } => {
                            if let Ok(inner_signal) = world.run_system_with(factory_system_id, value) {
                                let (handle, id, initial_value) = spawn_processor(
                                    world,
                                    state_and_queue_entity,
                                    *output_system_handle_clone,
                                    key.clone(),
                                    inner_signal,
                                );

                                let old_handle = {
                                    let mut state = world
                                        .get_mut::<ManagerState<Self::Key, S>>(state_and_queue_entity)
                                        .unwrap();
                                    state.signals.insert(key.clone(), (handle, id))
                                };

                                if let Some((old_handle, _)) = old_handle {
                                    old_handle.cleanup(world);
                                }

                                new_map_diffs.push(MapDiff::Insert {
                                    key,
                                    value: initial_value,
                                });
                            }
                        }
                        MapDiff::Update { key, value } => {
                            if let Ok(new_inner_signal) = world.run_system_with(factory_system_id, value) {
                                let new_inner_id = new_inner_signal.clone().register(world);

                                let old_inner_id_opt = {
                                    let state =
                                        world.get::<ManagerState<Self::Key, S>>(state_and_queue_entity).unwrap();
                                    state.signals.get(&key).map(|(_, id)| *id)
                                };

                                if old_inner_id_opt == Some(*new_inner_id) {
                                    new_inner_id.cleanup(world);
                                    continue;
                                }

                                let (new_processor_handle, new_processor_id, initial_value) = spawn_processor(
                                    world,
                                    state_and_queue_entity,
                                    *output_system_handle_clone,
                                    key.clone(),
                                    new_inner_signal,
                                );

                                let old_processor_handle = {
                                    let mut state = world
                                        .get_mut::<ManagerState<Self::Key, S>>(state_and_queue_entity)
                                        .unwrap();
                                    state
                                        .signals
                                        .insert(key.clone(), (new_processor_handle, new_processor_id))
                                };

                                if let Some((old_handle, _)) = old_processor_handle {
                                    old_handle.cleanup(world);
                                }

                                new_inner_id.cleanup(world);

                                new_map_diffs.push(MapDiff::Update {
                                    key,
                                    value: initial_value,
                                });
                            }
                        }
                        MapDiff::Remove { key } => {
                            let old_handle = {
                                let mut state = world
                                    .get_mut::<ManagerState<Self::Key, S>>(state_and_queue_entity)
                                    .unwrap();
                                state.signals.remove(&key)
                            };
                            if let Some((handle, _)) = old_handle {
                                handle.cleanup(world);
                            }
                            new_map_diffs.push(MapDiff::Remove { key });
                        }
                        MapDiff::Clear => {
                            let old_signals = {
                                let mut state = world
                                    .get_mut::<ManagerState<Self::Key, S>>(state_and_queue_entity)
                                    .unwrap();
                                if state.signals.is_empty() {
                                    BTreeMap::new()
                                } else {
                                    core::mem::take(&mut state.signals)
                                }
                            };

                            if !old_signals.is_empty() {
                                for (_, (handle, _)) in old_signals {
                                    handle.cleanup(world);
                                }
                                new_map_diffs.push(MapDiff::Clear);
                            }
                        }
                    }
                }

                if !new_map_diffs.is_empty()
                    && let Some(mut queue) = world.get_mut::<QueuedMapDiffs<Self::Key, S::Item>>(state_and_queue_entity)
                {
                    queue.0.extend(new_map_diffs);
                }
                process_signals(world, [*output_system_handle_clone], Box::new(()));
            };

            let manager_handle = self.for_each(manager_system_logic).register(world);

            world
                .entity_mut(state_and_queue_entity)
                .insert((
                    ManagerState::<Self::Key, S> {
                        signals: BTreeMap::new(),
                        _phantom: PhantomData,
                    },
                    QueuedMapDiffs::<Self::Key, S::Item>(vec![]),
                ))
                .add_child(**manager_handle)
                .add_child(factory_system_id.entity())
                .add_child(**output_system_handle);

            *output_system_handle
        });

        MapValueSignal {
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps this [`SignalMap`] to a [`Key`]-lookup [`Signal`] which outputs [`Some<Value>`] if the
    /// [`Key`] is present and [`None`] otherwise.
    ///
    /// # Example
    /// ```no_run
    /// MutableBTreeMap::from([(1, 2), (3, 4)]).signal_map().key(1); // outputs `2`
    /// ```
    ///
    /// [`Key`]: SignalMap::Key
    fn key(self, key: Self::Key) -> Key<Self>
    where
        Self: Sized,
        Self::Key: PartialEq + Clone + SSs,
        Self::Value: Clone + Send + 'static,
    {
        Key {
            inner: self.for_each(
                move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, mut state: Local<Option<Self::Value>>| {
                    let mut changed = false;
                    let mut new_value = (*state).clone();

                    for diff in diffs {
                        match diff {
                            MapDiff::Replace { entries } => {
                                new_value = entries.into_iter().find(|(k, _)| *k == key).map(|(_, v)| v);
                                changed = true;
                            }
                            MapDiff::Insert { key: k, value } | MapDiff::Update { key: k, value } => {
                                if k == key {
                                    new_value = Some(value);
                                    changed = true;
                                }
                            }
                            MapDiff::Remove { key: k } => {
                                if k == key {
                                    new_value = None;
                                    changed = true;
                                }
                            }
                            MapDiff::Clear => {
                                new_value = None;
                                changed = true;
                            }
                        }
                    }

                    if changed {
                        *state = new_value.clone();
                        Some(new_value)
                    } else {
                        None
                    }
                },
            ),
        }
    }

    /// Erases the type of this [`SignalMap`], allowing it to be used in conjunction with
    /// [`SignalMap`]s of other concrete types.
    ///
    /// # Example
    /// ```no_run
    /// let signal = if condition {
    ///     MutableBTreeMap::from([(1, 2), (3, 4)]).map_value(...).boxed() // this is a `MapValue<Source<i32, i32>>`
    /// } else {
    ///     MutableBTreeMap::from([(1, 2), (3, 4)]).map_value_signal(...).boxed() // this is a `MapValueSignal<Source<i32, i32>>`
    /// } // without the `.boxed()`, the compiler would not allow this
    /// ```
    fn boxed(self) -> Box<dyn SignalMap<Key = Self::Key, Value = Self::Value>>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Activate this [`SignalMap`] and all its upstreams, causing them to be evaluated every
    /// frame until they are [`SignalHandle::cleanup`]-ed, see [`SignalHandle`].
    fn register(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.register_signal_map(world)
    }
}

impl<T: ?Sized> SignalMapExt for T where T: SignalMap {}

/// Provides immutable access to the underlying [`BTreeMap`].
pub struct MutableBTreeMapReadGuard<'a, K, V> {
    guard: RwLockReadGuard<'a, MutableBTreeMapState<K, V>>,
}

impl<'a, K, V> Deref for MutableBTreeMapReadGuard<'a, K, V> {
    type Target = BTreeMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

/// Provides limited mutable access to the underlying [`BTreeMap`].
pub struct MutableBTreeMapWriteGuard<'a, K, V> {
    guard: RwLockWriteGuard<'a, MutableBTreeMapState<K, V>>,
}

impl<'a, K, V> MutableBTreeMapWriteGuard<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Inserts a key-value pair into this [`MutableBTreeMap`], queueing a [`MapDiff::Update`] or
    /// [`MapDiff::Insert`] depending on whether the key was present.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let diff = if self.guard.map.contains_key(&key) {
            MapDiff::Update {
                key: key.clone(),
                value: value.clone(),
            }
        } else {
            MapDiff::Insert {
                key: key.clone(),
                value: value.clone(),
            }
        };
        let old = self.guard.map.insert(key, value);
        self.guard.pending_diffs.push(diff);
        old
    }

    /// Removes a key from this [`MutableBTreeMap`], queueing a [`MapDiff::Remove`] and returning
    /// the value at the key if the key was previously present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let old = self.guard.map.remove(key);
        if old.is_some() {
            self.guard.pending_diffs.push(MapDiff::Remove { key: key.clone() });
        }
        old
    }

    /// Clears this [`MutableBTreeMap`], removing all elements and queueing a [`MapDiff::Clear`] if
    /// any elements were present.
    pub fn clear(&mut self) {
        if !self.guard.map.is_empty() {
            self.guard.map.clear();
            self.guard.pending_diffs.push(MapDiff::Clear);
        }
    }

    /// Replaces the entire contents of this [`MutableBTreeMap`] with a new set of entries, queueing
    /// a [`MapDiff::Replace`].
    pub fn replace<T>(&mut self, entries: T)
    where
        BTreeMap<K, V>: From<T>,
    {
        self.guard.map = entries.into();
        let entries = self.guard.map.clone().into_iter().collect();
        self.guard.pending_diffs.push(MapDiff::Replace { entries });
    }
}

impl<'a, K, V> Deref for MutableBTreeMapWriteGuard<'a, K, V> {
    type Target = BTreeMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

struct MutableBTreeMapState<K, V> {
    map: BTreeMap<K, V>,
    pending_diffs: Vec<MapDiff<K, V>>,
    signal: Option<LazySignal>,
}

#[derive(Component)]
struct QueuedMapDiffs<K, V>(Vec<MapDiff<K, V>>);

/// Wrapper around a [`BTreeMap`] that tracks mutations as [`MapDiff`]s and emits them as a batch on
/// [`flush`](MutableBTreeMap::flush), enabling diff-less constant time reactive updates for
/// downstream [`SignalMap`]s.
#[derive(Clone)]
pub struct MutableBTreeMap<K, V> {
    state: Arc<RwLock<MutableBTreeMapState<K, V>>>,
}

impl<T, K, V> From<T> for MutableBTreeMap<K, V>
where
    BTreeMap<K, V>: From<T>,
{
    fn from(values: T) -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableBTreeMapState {
                map: values.into(),
                pending_diffs: Vec::new(),
                signal: None,
            })),
        }
    }
}

/// Signal graph node with no upstreams which forwards [`Vec<MapDiff<K, V>>`]s flushed from some
/// source [`MutableBTreeMap<K, V>`], see [`MutableBTreeMap::signal_map`].
#[derive(Clone)]
pub struct Source<K, V> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (K, V)>,
}

impl<K, V> SignalMap for Source<K, V>
where
    K: 'static,
    V: 'static,
{
    type Key = K;
    type Value = V;
    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

#[derive(Component)]
pub(crate) struct MapReplayTrigger(Box<dyn FnOnce(&mut World) + Send + Sync>);

impl Replayable for MapReplayTrigger {
    fn trigger(self) -> Box<dyn FnOnce(&mut World) + Send + Sync> {
        self.0
    }
}

impl<K, V> MutableBTreeMap<K, V> {
    /// Constructs a new, emtpy [`MutableBTreeMap<K, V>`].
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::from(BTreeMap::new())
    }

    /// Locks this [`MutableBTreeMap`] with shared read access, blocking the current thread until it
    /// can be acquired, see [`RwLock::read`].
    pub fn read(&self) -> MutableBTreeMapReadGuard<'_, K, V> {
        MutableBTreeMapReadGuard {
            guard: self.state.read().unwrap(),
        }
    }

    /// Locks this [`MutableBTreeMap`] with exclusive write access, blocking the current thread
    /// until it can be acquired, see [`RwLock::write`].
    pub fn write(&self) -> MutableBTreeMapWriteGuard<'_, K, V> {
        MutableBTreeMapWriteGuard {
            guard: self.state.write().unwrap(),
        }
    }

    fn get_or_create_broadcaster_signal(&self) -> LazySignal
    where
        K: Clone + SSs,
        V: Clone + SSs,
    {
        let mut state = self.state.write().unwrap();
        // If the signal already exists, just clone and return it.
        if let Some(lazy_signal) = &state.signal {
            return lazy_signal.clone();
        }

        // Otherwise, create the broadcaster signal for the first time.
        let broadcaster_lazy_signal = LazySignal::new(move |world: &mut World| {
            let self_entity = LazyEntity::new();

            // This is the system for the one-and-only broadcaster. It just drains
            // diffs that `flush` has put into its component.
            let source_system_logic = clone!((self_entity) move |_: In<()>, world: &mut World| {
                if let Some(mut diffs) = world.get_mut::<QueuedMapDiffs<K, V>>(self_entity.get()) {
                    if diffs.0.is_empty() { None } else { Some(diffs.0.drain(..).collect()) }
                } else {
                    None
                }
            });

            let signal_system = register_signal::<(), Vec<MapDiff<K, V>>, _, _, _>(world, source_system_logic);
            self_entity.set(*signal_system);

            // The broadcaster itself does not have an initial state to replay.
            // It just needs the component to receive flushed diffs.
            world.entity_mut(*signal_system).insert(QueuedMapDiffs::<K, V>(vec![]));

            signal_system
        });

        // Store it for future calls.
        state.signal = Some(broadcaster_lazy_signal.clone());
        broadcaster_lazy_signal
    }

    /// Returns a [`Source`] signal from this [`MutableBTreeMap`], always returning clones of the
    /// same underlying [`Signal`]; such [`SignalMap`]s only emit incremental updates so clones
    /// will not re-emit initial states.
    pub fn signal_map(&self) -> Source<K, V>
    where
        K: Clone + Ord + SSs,
        V: Clone + SSs,
    {
        let broadcaster_signal = self.get_or_create_broadcaster_signal();

        let replay_lazy_signal = LazySignal::new(clone!((self.state => state) move |world: &mut World| {
            let self_entity = LazyEntity::new();
            let broadcaster_system = broadcaster_signal.register(world);

            let replay_system_logic = clone!((self_entity) move |In(upstream_diffs): In<Vec<MapDiff<K, V>>>, world: &mut World| {
                // Get the locally queued diffs (the initial 'Replace' diff).
                let mut diffs = world
                    .get_entity_mut(self_entity.get()).ok()
                    .and_then(|mut entity| entity.take::<QueuedMapDiffs<K, V>>())
                    .map(|queued| queued.0)
                    .unwrap_or_default();

                // Add the diffs from upstream (the broadcaster).
                diffs.extend(upstream_diffs);

                // If there are any diffs to process, return them.
                if diffs.is_empty() { None } else { Some(diffs) }
            });

            // 1. Register the replay system.
            let replay_signal = register_signal::<_, Vec<MapDiff<K, V>>, _, _, _>(world, replay_system_logic);
            self_entity.set(*replay_signal);

            // This closure is the core of the replay trigger. It checks if the broadcaster is idle.
            let trigger = Box::new(move |world: &mut World| {
                // If the broadcaster's queue has no diffs, it's an idle frame, so we trigger a replay.
                if world.get::<QueuedMapDiffs<K, V>>(*broadcaster_system).is_some_and(|q| q.0.is_empty()) {
                    // Trigger with an empty vec, forcing the replay_system_logic to check its own queue.
                    process_signals(world, [replay_signal], Box::new(Vec::<MapDiff<K, V>>::new()));
                }
            });

            // 2. Queue the initial state for this new subscriber.
            let initial_map: BTreeMap<K, V> = state.read().unwrap().map.clone();
            let initial_diffs = if !initial_map.is_empty() {
                vec![MapDiff::Replace { entries: initial_map.into_iter().collect() }]
            } else {
                vec![]
            };
            world.entity_mut(*replay_signal).insert((QueuedMapDiffs(initial_diffs), MapReplayTrigger(trigger)));

            // 3. Pipe the broadcaster to the new replay node.
            pipe_signal(world, broadcaster_system, replay_signal);

            replay_signal
        }));

        Source {
            signal: replay_lazy_signal,
            _marker: PhantomData,
        }
    }

    /// Returns a [`SignalVec`] which outputs this [`MutableBTreeMap`]'s [`Key`](SignalMap::Key)s in
    /// sorted order.
    pub fn signal_vec_keys(&self) -> SignalVecKeys<Self>
    where
        K: Ord + Clone + SSs,
        V: Clone + SSs,
    {
        let upstream = self.signal_map();
        let lazy_signal = LazySignal::new(move |world: &mut World| {
            let upstream_handle = upstream.register_signal_map(world);
            let processor_logic = move |In(diffs): In<Vec<MapDiff<K, V>>>, mut keys: Local<Vec<K>>| {
                let mut out_diffs = Vec::new();
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            *keys = entries.into_iter().map(|(k, _)| k).collect();
                            out_diffs.push(VecDiff::Replace { values: keys.clone() });
                        }
                        MapDiff::Insert { key, .. } => {
                            let index = keys.binary_search(&key).unwrap_err();
                            keys.insert(index, key.clone());
                            out_diffs.push(VecDiff::InsertAt { index, value: key });
                        }
                        MapDiff::Update { .. } => {
                            // no change to keys
                        }
                        MapDiff::Remove { key } => {
                            if let Ok(index) = keys.binary_search(&key) {
                                keys.remove(index);
                                out_diffs.push(VecDiff::RemoveAt { index });
                            }
                        }
                        MapDiff::Clear => {
                            keys.clear();
                            out_diffs.push(VecDiff::Clear);
                        }
                    }
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            };
            let processor_handle =
                lazy_signal_from_system::<_, Vec<VecDiff<K>>, _, _, _>(processor_logic).register(world);
            pipe_signal(world, *upstream_handle, processor_handle);
            processor_handle
        });

        SignalVecKeys {
            signal: lazy_signal,
            _marker: PhantomData,
        }
    }

    /// Returns a [`SignalVec`] which outputs this [`MutableBTreeMap`]'s `(key, value)`s in sorted
    /// order.
    pub fn signal_vec_entries(&self) -> SignalVecEntries<Self>
    where
        K: Ord + Clone + SSs,
        V: Clone + SSs,
    {
        let upstream = self.signal_map();
        let lazy_signal = LazySignal::new(move |world: &mut World| {
            let upstream_handle = upstream.register_signal_map(world);
            let processor_logic = move |In(diffs): In<Vec<MapDiff<K, V>>>, mut keys: Local<Vec<K>>| {
                let mut out_diffs = Vec::new();
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            *keys = entries.iter().map(|(k, _)| k.clone()).collect();
                            out_diffs.push(VecDiff::Replace { values: entries });
                        }
                        MapDiff::Insert { key, value } => {
                            let index = keys.binary_search(&key).unwrap_err();
                            keys.insert(index, key.clone());
                            out_diffs.push(VecDiff::InsertAt {
                                index,
                                value: (key, value),
                            });
                        }
                        MapDiff::Update { key, value } => {
                            if let Ok(index) = keys.binary_search(&key) {
                                out_diffs.push(VecDiff::UpdateAt {
                                    index,
                                    value: (key, value),
                                });
                            }
                        }
                        MapDiff::Remove { key } => {
                            if let Ok(index) = keys.binary_search(&key) {
                                keys.remove(index);
                                out_diffs.push(VecDiff::RemoveAt { index });
                            }
                        }
                        MapDiff::Clear => {
                            keys.clear();
                            out_diffs.push(VecDiff::Clear);
                        }
                    }
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            };
            let processor_handle =
                lazy_signal_from_system::<_, Vec<VecDiff<(K, V)>>, _, _, _>(processor_logic).register(world);
            pipe_signal(world, *upstream_handle, processor_handle);
            processor_handle
        });

        SignalVecEntries {
            signal: lazy_signal,
            _marker: PhantomData,
        }
    }

    /// Emits any pending [`MapDiff`]s to downstream [`SignalMap`]s.
    pub fn flush_into_world(&self, world: &mut World)
    where
        K: SSs,
        V: SSs,
    {
        let mut state = self.state.write().unwrap();
        if state.pending_diffs.is_empty() {
            return;
        }

        let signal = if let Some(lazy_signal) = &state.signal
            && let LazySystem::Registered(signal_system) = *lazy_signal.inner.system.read().unwrap()
        {
            signal_system
        } else {
            return;
        };
        if let Ok(mut entity) = world.get_entity_mut(*signal)
            && let Some(mut queued_diffs) = entity.get_mut::<QueuedMapDiffs<K, V>>()
        {
            queued_diffs.0.append(&mut state.pending_diffs);
        }
    }

    /// Returns an `impl Command` that can be passed to [`Commands::queue`] to flush this
    /// [`MutableBTreeMap`]'s pending [`MapDiff`]s, see
    /// [`.flush_into_world`](Self::flush_into_world).
    pub fn flush(&self) -> impl Command
    where
        K: Clone + SSs,
        V: Clone + SSs,
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

    // Helper resource to capture the output diffs from a SignalMap for assertions.
    #[derive(Resource, Default, Debug)]
    struct SignalMapOutput<K, V>(Vec<MapDiff<K, V>>)
    where
        K: SSs + Clone + fmt::Debug,
        V: SSs + Clone + fmt::Debug;

    // Helper system that captures incoming diffs and stores them in the SignalMapOutput resource.
    fn capture_map_output<K, V>(In(diffs): In<Vec<MapDiff<K, V>>>, mut output: ResMut<SignalMapOutput<K, V>>)
    where
        K: SSs + Clone + fmt::Debug,
        V: SSs + Clone + fmt::Debug,
    {
        output.0.extend(diffs);
    }

    // Helper function to retrieve and clear the captured diffs from the world,
    // making it easy to assert against the output of a single frame's update.
    fn get_and_clear_map_output<K, V>(world: &mut World) -> Vec<MapDiff<K, V>>
    where
        K: SSs + Clone + fmt::Debug,
        V: SSs + Clone + fmt::Debug,
    {
        world
            .get_resource_mut::<SignalMapOutput<K, V>>()
            .map(|mut res| core::mem::take(&mut res.0))
            .unwrap_or_default()
    }

    // Helper to create a minimal Bevy App with the JonmoPlugin for testing.
    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app
    }

    // A custom PartialEq implementation for MapDiff to make test assertions cleaner.
    impl<K: PartialEq, V: PartialEq> PartialEq for MapDiff<K, V> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Replace { entries: l_entries }, Self::Replace { entries: r_entries }) => l_entries == r_entries,
                (
                    Self::Insert {
                        key: l_key,
                        value: l_value,
                    },
                    Self::Insert {
                        key: r_key,
                        value: r_value,
                    },
                ) => l_key == r_key && l_value == r_value,
                (
                    Self::Update {
                        key: l_key,
                        value: l_value,
                    },
                    Self::Update {
                        key: r_key,
                        value: r_value,
                    },
                ) => l_key == r_key && l_value == r_value,
                (Self::Remove { key: l_key }, Self::Remove { key: r_key }) => l_key == r_key,
                (Self::Clear, Self::Clear) => true,
                _ => false,
            }
        }
    }

    #[test]
    fn test_map_value() {
        let mut app = create_test_app();
        // The output map will have keys of `u32` and values of `String`.
        app.init_resource::<SignalMapOutput<u32, String>>();

        // The source map contains integer values.
        let source_map = MutableBTreeMap::from([(1, 10), (2, 20)]);

        // The mapping function transforms an i32 value into a String.
        let mapping_system = |In(val): In<i32>| format!("Val:{val}");

        // Apply `map_value` to create the derived signal.
        let mapped_signal = source_map.signal_map().map_value(mapping_system);

        let handle = mapped_signal
            .for_each(capture_map_output::<u32, String>)
            .register(app.world_mut());

        // Test 1: Initial State (Replace).
        // The first update should replay the initial state with mapped values.
        app.update();
        let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Initial state should produce one Replace diff");
        assert_eq!(
            diffs[0],
            MapDiff::Replace {
                entries: vec![(1, "Val:10".to_string()), (2, "Val:20".to_string())]
            },
            "Initial Replace diff has incorrect mapped values"
        );

        // Test 2: Insert.
        // A new entry in the source should result in an Insert diff with a mapped value.
        source_map.write().insert(3, 30);
        source_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Insert should produce one diff");
        assert_eq!(
            diffs[0],
            MapDiff::Insert {
                key: 3,
                value: "Val:30".to_string()
            },
            "Insert diff has incorrect mapped value"
        );

        // Test 3: Update.
        // Updating an existing entry should result in an Update diff with a mapped value.
        source_map.write().insert(1, 15); // `insert` on existing key is an update
        source_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Update should produce one diff");
        assert_eq!(
            diffs[0],
            MapDiff::Update {
                key: 1,
                value: "Val:15".to_string()
            },
            "Update diff has incorrect mapped value"
        );

        // Test 4: Remove.
        // Removing an entry should result in a Remove diff, which has no value to map.
        source_map.write().remove(&2);
        source_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Remove should produce one diff");
        assert_eq!(
            diffs[0],
            MapDiff::Remove { key: 2 },
            "Remove diff was not propagated correctly"
        );

        // Test 5: Clear.
        // Clearing the source map should result in a Clear diff.
        source_map.write().clear();
        source_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Clear should produce one diff");
        assert_eq!(diffs[0], MapDiff::Clear, "Clear diff was not propagated correctly");

        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_value_signal() {
        let mut app = create_test_app();
        // The output map will have keys of `u32` and values of `Name` components.
        app.init_resource::<SignalMapOutput<u32, Name>>();

        // Setup: Create entities with `Name` components that our signals will track.
        let entity_a = app.world_mut().spawn(Name::new("Alice")).id();
        let entity_b = app.world_mut().spawn(Name::new("Bob")).id();

        // The source map contains entities. The goal is to create a derived map
        // that contains the *names* of these entities.
        let entity_map = MutableBTreeMap::from([(1, entity_a), (2, entity_b)]);

        // This "factory" system takes an entity and creates a signal that tracks its `Name`.
        let factory_system = |In(entity): In<Entity>| SignalBuilder::from_component::<Name>(entity).dedupe();

        // Apply `map_value_signal` to transform the SignalMap<u32, Entity>
        // into a SignalMap<u32, Name>.
        let name_map_signal = entity_map.signal_map().map_value_signal(factory_system);

        let handle = name_map_signal
            .for_each(capture_map_output::<u32, Name>)
            .register(app.world_mut());

        // Test 1: Initial State.
        // The first update should replay the initial state of the map.
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Initial state should produce one Replace diff");
        assert_eq!(
            diffs[0],
            MapDiff::Replace {
                entries: vec![(1, Name::new("Alice")), (2, Name::new("Bob"))]
            },
            "Initial state is incorrect"
        );

        // Test 2: Inner Signal Update.
        // Change a component on a tracked entity. This should trigger an Update diff.
        *app.world_mut().get_mut::<Name>(entity_a).unwrap() = Name::new("Alicia");
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Name change should produce one Update diff");
        assert_eq!(
            diffs[0],
            MapDiff::Update {
                key: 1,
                value: Name::new("Alicia")
            },
            "Update diff is incorrect"
        );

        // Test 3: No change should produce no diff.
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert!(diffs.is_empty(), "No change should produce no diffs");

        // Test 4: Source Map Insertion.
        // Add a new entity to the source map. This should trigger an Insert diff.
        let entity_c = app.world_mut().spawn(Name::new("Charlie")).id();
        entity_map.write().insert(3, entity_c);
        entity_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Insert should produce one Insert diff");
        assert_eq!(
            diffs[0],
            MapDiff::Insert {
                key: 3,
                value: Name::new("Charlie")
            },
            "Insert diff is incorrect"
        );

        // Test 5: Source Map Removal.
        // Remove an entity from the source map. This should trigger a Remove diff.
        entity_map.write().remove(&2);
        entity_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Remove should produce one Remove diff");
        assert_eq!(diffs[0], MapDiff::Remove { key: 2 }, "Remove diff is incorrect");

        // Test 6: Source Map Update (switching the underlying signal).
        // Update a key to point to a new entity. This must tear down the old signal
        // and create a new one, resulting in an Update diff with the new value.
        let entity_d = app.world_mut().spawn(Name::new("David")).id();
        entity_map.write().insert(1, entity_d); // `insert` acts as update here.
        entity_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Source map update should produce one Update diff");
        assert_eq!(
            diffs[0],
            MapDiff::Update {
                key: 1,
                value: Name::new("David")
            },
            "Update-to-new-entity diff is incorrect"
        );

        // Verify that the old signal (for entity_a) is no longer tracked.
        *app.world_mut().get_mut::<Name>(entity_a).unwrap() = Name::new("Alicia-v2");
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert!(
            diffs.is_empty(),
            "Update on old, replaced entity should not produce a diff"
        );

        // Verify that the new signal (for entity_d) is now being tracked.
        *app.world_mut().get_mut::<Name>(entity_d).unwrap() = Name::new("Dave");
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Update on new entity should produce a diff");
        assert_eq!(
            diffs[0],
            MapDiff::Update {
                key: 1,
                value: Name::new("Dave")
            },
            "Update on new entity is incorrect"
        );

        // Test 7: Source Map Clear.
        // Clear the source map. This should trigger a Clear diff.
        entity_map.write().clear();
        entity_map.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Clear should produce one Clear diff");
        assert_eq!(diffs[0], MapDiff::Clear, "Clear diff is incorrect");

        handle.cleanup(app.world_mut());
    }

    // Helper resource to capture the output from a standard Signal for assertions.
    #[derive(Resource, Default, Debug)]
    struct SignalOutput<T>(Option<T>)
    where
        T: SSs + Clone + fmt::Debug;

    // Helper system that captures incoming values and stores them in the SignalOutput resource.
    fn capture_output<T>(In(value): In<T>, mut output: ResMut<SignalOutput<T>>)
    where
        T: SSs + Clone + fmt::Debug,
    {
        output.0 = Some(value);
    }

    // Helper to retrieve the last captured value.
    fn get_output<T: SSs + Clone + fmt::Debug>(world: &mut World) -> Option<T> {
        world.get_resource::<SignalOutput<T>>().and_then(|res| res.0.clone())
    }

    // Helper to clear the output, useful for testing no-emission cases.
    fn clear_output<T: SSs + Clone + fmt::Debug>(world: &mut World) {
        if let Some(mut res) = world.get_resource_mut::<SignalOutput<T>>() {
            res.0 = None;
        }
    }

    #[test]
    fn test_key() {
        let mut app = create_test_app();
        // The output is a Signal<Option<String>>.
        app.init_resource::<SignalOutput<Option<String>>>();

        let source_map = MutableBTreeMap::from([(1, "one".to_string()), (2, "two".to_string())]);

        // We will specifically track the value associated with key `2`.
        let key_to_track = 2;
        let key_signal = source_map.signal_map().key(key_to_track);

        let handle = key_signal
            .map(capture_output::<Option<String>>)
            .register(app.world_mut());

        // Test 1: Initial State (Key is Present).
        // The first update should emit the initial value for the key.
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            Some(Some("two".to_string())),
            "Initial value for present key is incorrect"
        );

        // Test 2: Update Tracked Key's Value.
        // This should cause the signal to emit the new value.
        source_map.write().insert(key_to_track, "two_v2".to_string());
        source_map.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            Some(Some("two_v2".to_string())),
            "Update to tracked key did not emit correctly"
        );

        // Test 3: Update a Different Key.
        // This should NOT cause the signal to emit, as our key's value hasn't changed.
        clear_output::<Option<String>>(app.world_mut());
        source_map.write().insert(1, "one_v2".to_string());
        source_map.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            None,
            "Signal emitted when a different key was updated"
        );

        // Test 4: Remove Tracked Key.
        // This should cause the signal to emit `None`.
        source_map.write().remove(&key_to_track);
        source_map.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            Some(None),
            "Removing the tracked key did not emit None"
        );

        // Test 5: No Change (Key is Absent).
        // Another update with the key still absent should not emit.
        clear_output::<Option<String>>(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            None,
            "Signal emitted when key remained absent"
        );

        // Test 6: Re-insert Tracked Key.
        // This should cause the signal to emit the new value.
        source_map.write().insert(key_to_track, "two_reborn".to_string());
        source_map.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            Some(Some("two_reborn".to_string())),
            "Re-inserting the tracked key did not emit its value"
        );

        // Test 7: Clear the map.
        // Since this removes the key, it should emit `None`.
        source_map.write().clear();
        source_map.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world_mut()),
            Some(None),
            "Clearing the map did not emit None for the tracked key"
        );

        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_for_each() {
        let mut app = create_test_app();
        // The output of our `for_each` system will be the full, reconstructed BTreeMap.
        app.init_resource::<SignalOutput<BTreeMap<u32, String>>>();

        let source_map = MutableBTreeMap::from([(1, "one".to_string()), (2, "two".to_string())]);

        // This system reconstructs the state of the map by applying the diffs it receives.
        // It then outputs the complete, current state of the map.
        // This allows us to verify that `for_each` is receiving the diffs correctly.
        let reconstructor_system = |In(diffs): In<Vec<MapDiff<u32, String>>>,
                                    mut state: Local<BTreeMap<u32, String>>| {
            for diff in diffs {
                match diff {
                    MapDiff::Replace { entries } => {
                        *state = entries.into_iter().collect();
                    }
                    MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                        state.insert(key, value);
                    }
                    MapDiff::Remove { key } => {
                        state.remove(&key);
                    }
                    MapDiff::Clear => {
                        state.clear();
                    }
                }
            }
            state.clone() // Output the current reconstructed state
        };

        let handle = source_map
            .signal_map()
            .for_each(reconstructor_system)
            .map(capture_output::<BTreeMap<u32, String>>)
            .register(app.world_mut());

        // Test 1: Initial State.
        // The initial `Replace` diff should be received and processed.
        app.update();
        let expected_initial_state: BTreeMap<_, _> =
            [(1, "one".to_string()), (2, "two".to_string())].into_iter().collect();
        assert_eq!(
            get_output::<BTreeMap<u32, String>>(app.world_mut()),
            Some(expected_initial_state.clone()),
            "Initial state was not reconstructed correctly"
        );

        // Test 2: Batched Mutations.
        // We'll perform multiple operations before flushing to test batch processing.
        {
            let mut writer = source_map.write();
            writer.insert(3, "three".to_string()); // Insert
            writer.insert(1, "one_v2".to_string()); // Update
            writer.remove(&2); // Remove
        }
        source_map.flush_into_world(app.world_mut());
        app.update();
        let expected_batched_state: BTreeMap<_, _> = [(1, "one_v2".to_string()), (3, "three".to_string())]
            .into_iter()
            .collect();
        assert_eq!(
            get_output::<BTreeMap<u32, String>>(app.world_mut()),
            Some(expected_batched_state.clone()),
            "State after batched mutations was not reconstructed correctly"
        );

        // Test 3: Clear.
        // The `Clear` diff should result in an empty map.
        source_map.write().clear();
        source_map.flush_into_world(app.world_mut());
        app.update();
        let expected_cleared_state: BTreeMap<u32, String> = BTreeMap::new();
        assert_eq!(
            get_output::<BTreeMap<u32, String>>(app.world_mut()),
            Some(expected_cleared_state.clone()),
            "State after Clear was not reconstructed correctly"
        );

        handle.cleanup(app.world_mut());
    }
}
