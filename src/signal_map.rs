//! Data structures and combinators for constructing reactive [`System`] dependency graphs on top
//! of [`BTreeMap`] mutations.
//!
//! This module provides [`SignalMap`], a collection-oriented signal trait with **diff-based
//! semantics**. Rather than forwarding entire maps each frame, [`SignalMap`] propagates
//! [`MapDiff`] values that describe incremental mutations (insert, update, remove, etc.). This
//! enables efficient, constant-time reactive updates for keyed collections of any size.
//!
//! See [`MutableBTreeMap`] for the primary source type and [`SignalMapExt`] for available
//! combinators. For the general signal graph runtime model and core concepts, see the [`Signal`]
//! trait documentation.
pub use super::signal::{OrdinaryBroadcast, ReplayableBroadcastMarker};
use super::{
    graph::SignalClaim,
    signal::{self, Signal, SignalExt},
    signal_vec::{CachedSignalVec, SignalVec, SignalVecExt, VecDiff},
};
use crate::{
    DynamicEmissions, DynamicManagerStep, DynamicManagerUpdate, DynamicSignalOperatorSpec, KeyedSlotCommand,
    MapSignalKind, MapSignalMaterializer, OutputDelivery, ReplayInitialization, SameSignalIdentity,
    SignalRegistrationError, cleanup::CleanupSender, registration::SignalMaterializerFactory,
};
use alloc::collections::BTreeMap;
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use bevy_platform::{
    prelude::*,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};
use core::{fmt, marker::PhantomData, ops::Deref};

/// Describes the mutations made to the underlying [`MutableBTreeMap`] that are piped to downstream
/// [`SignalMap`]s.
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

/// A lazy, composable chain specialized for diff-based [`BTreeMap`] reactivity.
///
/// Unlike [`Signal`] which forwards complete values, a [`SignalMap`] propagates [`MapDiff`]
/// values describing incremental mutations to an underlying keyed collection. This diff-based
/// approach enables **constant-time reactive updates** regardless of map size; only the changes
/// are transmitted and processed, not the entire map.
///
/// Downstream consumers receive a stream of [`MapDiff`] variants ([`Insert`](MapDiff::Insert),
/// [`Update`](MapDiff::Update), [`Remove`](MapDiff::Remove), etc.) that they can apply to
/// maintain a synchronized view of the source data.
///
/// A concrete chain is a lazy Rust value. Registration materializes its framework-created backing
/// into world-resident entities and returns one move-only direct claim on the terminal. Cloning a
/// cloneable chain shares its lazy identity but acquires no claim. External implementations remain
/// supported by returning a [`MapSignalMaterializer`]; they receive no graph-mutation authority.
///
/// For the general graph runtime and composition model, see [`Signal`].
///
/// Public implementations provide only materializer transfer; they cannot define or override a
/// world-taking registration method:
///
/// ```compile_fail,E0407
/// use bevy_ecs::world::World;
/// use jonmo::{MapSignalMaterializer, graph::SignalClaim, signal_map::SignalMap};
///
/// struct InvalidSignalMap;
///
/// impl SignalMap for InvalidSignalMap {
///     type Key = i32;
///     type Value = i32;
///
///     fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
///         unimplemented!()
///     }
///
///     fn register_signal_map(self, _world: &mut World) -> SignalClaim {
///         unimplemented!()
///     }
/// }
/// ```
///
/// Ordinary map chains are linear (`!Clone`); call [`SignalMapExt::broadcast`] for explicit
/// fan-out:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// fn assert_clone<T: Clone>(_: &T) {}
///
/// let mut world = World::new();
/// let chain = MutableBTreeMap::builder()
///     .values([(1, 2)])
///     .spawn(&mut world)
///     .signal_map();
/// assert_clone(&chain);
/// ```
///
/// Registration consumes the chain, so it cannot be registered again or have `.cache()` called on
/// the resulting [`SignalClaim`]:
///
/// ```compile_fail,E0382
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let chain = MutableBTreeMap::builder()
///     .values([(1, 2)])
///     .spawn(&mut world)
///     .signal_map();
/// let _first = chain.register(&mut world);
/// let _second = chain.register(&mut world);
/// ```
///
/// ```compile_fail,E0599
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let claim = MutableBTreeMap::builder()
///     .values([(1, 2)])
///     .spawn(&mut world)
///     .signal_map()
///     .register(&mut world);
/// let _ = claim.cache();
/// ```
pub trait SignalMap: Send + Sync + 'static {
    #[allow(missing_docs)]
    type Key;
    #[allow(missing_docs)]
    type Value;

    /// Transfers this concrete chain's framework-created, one-shot materializer backing.
    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value>;
}

impl<K: 'static, V: 'static> SignalMap for Box<dyn SignalMap<Key = K, Value = V> + Send + Sync> {
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        (*self).into_signal_map_materializer()
    }
}

/// An alias for a move-only boxed [`SignalMap`], returned by [`SignalMapExt::boxed`].
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// fn assert_clone<T: Clone>(_: &T) {}
///
/// let mut world = World::new();
/// let boxed = MutableBTreeMap::builder()
///     .values([(1, 2)])
///     .spawn(&mut world)
///     .signal_map()
///     .boxed();
/// assert_clone(&boxed);
/// ```
pub type BoxedSignalMap<K, V> = Box<dyn SignalMap<Key = K, Value = V> + Send + Sync>;

mod replayable_signal_map_sealed {
    pub trait Sealed {}
}

/// A map signal whose terminal replay contract is managed and verified by Jonmo.
///
/// New dependents receive targeted authoritative complete replacements with relationship-local
/// cursors, followed only by newer accepted revisions. `Warming` is distinct from a ready empty
/// map; an empty `Replace` is authoritative. One private maintenance claim keeps a retained replay
/// terminal current. Replay guarantees eventual handoff, not same-frame switching.
///
/// Stateful high-level switching accepts only this capability. Normalize a non-replayable map
/// chain with [`SignalMapExt::cache`] before selecting it.
pub trait ReplayableSignalMap: SignalMap + replayable_signal_map_sealed::Sealed {}

/// A type-erased, move-only replayable map signal.
pub struct BoxedReplayableSignalMap<K: 'static, V: 'static>(
    Box<dyn ReplayableSignalMap<Key = K, Value = V> + Send + Sync>,
);

impl<K: 'static, V: 'static> BoxedReplayableSignalMap<K, V> {
    /// Returns this already-verified replayable map signal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable map terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastMap<K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable map terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledMap<Sched, K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_schedule(self)
    }

    /// Returns this already-boxed replayable map signal without adding another allocation.
    pub fn boxed_replayable(self) -> Self {
        self
    }
}

impl<K: 'static, V: 'static> SignalMap for BoxedReplayableSignalMap<K, V> {
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.0.into_signal_map_materializer()
    }
}

impl<K: 'static, V: 'static> replayable_signal_map_sealed::Sealed for BoxedReplayableSignalMap<K, V> {}

impl<K, V> ReplayableSignalMap for BoxedReplayableSignalMap<K, V>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
}

/// Explicit cloneable fan-out wrapper created by [`SignalMapExt::broadcast`].
///
/// Cloning `BroadcastMap` shares one canonical lazy terminal identity without acquiring a graph
/// claim until a clone is registered or consumed by a downstream combinator. Downstream combinators
/// on a `BroadcastMap` clone return ordinary linear (`!Clone`) map chains unless `.broadcast()` is
/// called again at the new fan-out point.
pub struct BroadcastMap<K: 'static, V: 'static, Mode = OrdinaryBroadcast> {
    materializer: MapSignalMaterializer<K, V>,
    _mode: PhantomData<fn() -> Mode>,
}

/// Alias for a [`BroadcastMap`] handle backed by a verified [`ReplayableSignalMap`] terminal.
pub type ReplayableBroadcastMap<K, V> = BroadcastMap<K, V, ReplayableBroadcastMarker>;

impl<K: Clone + 'static, V: Clone + 'static, Mode> Clone for BroadcastMap<K, V, Mode> {
    fn clone(&self) -> Self {
        Self {
            materializer: self.materializer.shared_clone(),
            _mode: PhantomData,
        }
    }
}

impl<K: 'static, V: 'static> BroadcastMap<K, V, OrdinaryBroadcast> {
    /// Returns this already-broadcast handle without adding another graph entity.
    pub fn broadcast(self) -> Self {
        self
    }
}

impl<K: 'static, V: 'static> BroadcastMap<K, V, ReplayableBroadcastMarker> {
    /// Returns this already-verified replayable broadcast map without adding another cache entity,
    /// claim, or targeted snapshot transfer.
    pub fn cache(self) -> Self {
        self
    }

    /// Returns this already-broadcast replayable handle without adding another graph entity.
    pub fn broadcast(self) -> Self {
        self
    }

    /// Assigns a schedule to this verified replayable broadcast map while preserving its replay
    /// capability.
    ///
    /// Note that `.schedule::<Sched>()` returns a linear [`ReplayableScheduledMap`] handle and tags
    /// the shared canonical terminal when materialized. When multiple clones should share the
    /// scheduled terminal, prefer calling `.schedule::<Sched>().broadcast()` (rather than cloning
    /// before `.schedule::<Sched>()`), as assigning conflicting schedules to an already-registered
    /// canonical terminal fails registration.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledMap<Sched, K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_schedule(self)
    }
}

impl<K: 'static, V: 'static, Mode: Send + Sync + 'static> SignalMap for BroadcastMap<K, V, Mode> {
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

impl<K: 'static, V: 'static> replayable_signal_map_sealed::Sealed for BroadcastMap<K, V, ReplayableBroadcastMarker> {}

impl<K: Clone + Ord + Send + Sync + 'static, V: Clone + Send + Sync + 'static> ReplayableSignalMap
    for BroadcastMap<K, V, ReplayableBroadcastMarker>
{
}

pub(crate) fn replayable_map_broadcast<S>(signal: S) -> ReplayableBroadcastMap<S::Key, S::Value>
where
    S: ReplayableSignalMap,
    S::Key: Clone + Ord + Send + Sync + 'static,
    S::Value: Clone + Send + Sync + 'static,
{
    let upstream = Box::new(signal).into_signal_map_materializer();
    let materializer = MapSignalMaterializer::new(move |registrar| {
        let upstream = registrar.materialize_signal_map(upstream)?;
        registrar
            .map_broadcast_replayable_terminal(upstream)?
            .ok_or(SignalRegistrationError::ConflictingCanonicalState)
    });
    BroadcastMap {
        materializer,
        _mode: PhantomData,
    }
}

pub(crate) fn replayable_map_schedule<Sched, S>(signal: S) -> ReplayableScheduledMap<Sched, S::Key, S::Value>
where
    Sched: ScheduleLabel + Default + 'static,
    S: ReplayableSignalMap + 'static,
{
    let upstream = Box::new(signal).into_signal_map_materializer();
    let materializer = MapSignalMaterializer::new(move |registrar| {
        let upstream = registrar.materialize_signal_map(upstream)?;
        registrar.apply_schedule(upstream, Sched::default())
    });
    ScheduledMap {
        materializer,
        _marker: PhantomData,
    }
}

impl<Upstream, Switched> replayable_signal_map_sealed::Sealed for crate::signal::SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: ReplayableSignalMap,
    Switched::Key: Clone + Ord + Send + Sync + 'static,
    Switched::Value: Clone + Send + Sync + 'static,
{
}

impl<Upstream, Switched> ReplayableSignalMap for crate::signal::SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: ReplayableSignalMap,
    Switched::Key: Clone + Ord + Send + Sync + 'static,
    Switched::Value: Clone + Send + Sync + 'static,
{
}

/// A map cache wrapper created by [`SignalMapExt::cache`].
pub struct CachedSignalMap<Upstream: SignalMap> {
    materializer: MapSignalMaterializer<Upstream::Key, Upstream::Value>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream: SignalMap> CachedSignalMap<Upstream> {
    /// Returns this already-verified cache without adding another graph entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable cache terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastMap<Upstream::Key, Upstream::Value>
    where
        Upstream::Key: Clone + Ord + Send + Sync + 'static,
        Upstream::Value: Clone + Send + Sync + 'static,
    {
        replayable_map_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable cache terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> ReplayableScheduledMap<Sched, Upstream::Key, Upstream::Value>
    where
        Upstream: 'static,
        Upstream::Key: Clone + Ord + Send + Sync + 'static,
        Upstream::Value: Clone + Send + Sync + 'static,
    {
        replayable_map_schedule(self)
    }
}

impl<Upstream> SignalMap for CachedSignalMap<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: Clone + Ord + Send + Sync + 'static,
    Upstream::Value: Clone + Send + Sync + 'static,
{
    type Key = Upstream::Key;
    type Value = Upstream::Value;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_map_sealed::Sealed for CachedSignalMap<Upstream> where Upstream: SignalMap {}

impl<Upstream> ReplayableSignalMap for CachedSignalMap<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: Clone + Ord + Send + Sync + 'static,
    Upstream::Value: Clone + Send + Sync + 'static,
{
}

/// Signal graph node which applies a [`System`] directly to the "raw" [`Vec<MapDiff>`]s of its
/// upstream, see [.for_each](SignalMapExt::for_each).
pub struct ForEach<Upstream, O: 'static> {
    materializer: crate::ScalarSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for ForEach<Upstream, O>
where
    Upstream: SignalMap,
    O: 'static,
{
    type Item = O;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which applies a [`System`] to each [`Value`](SignalMap::Value) of its
/// upstream, see [`.map_value`](SignalMapExt::map_value).
pub struct MapValue<Upstream: SignalMap, O: 'static> {
    materializer: MapSignalMaterializer<Upstream::Key, O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> SignalMap for MapValue<Upstream, O>
where
    Upstream: SignalMap,
    O: 'static,
{
    type Key = Upstream::Key;
    type Value = O;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

/// Signal graph node which applies a [`System`] to each [`Value`](SignalMap::Value) of its
/// upstream, forwarding the output of each resulting [`Signal`], see
/// [`.map_value_signal`](SignalMapExt::map_value_signal).
pub struct MapValueSignal<Upstream, S: Signal>
where
    Upstream: SignalMap,
{
    materializer: MapSignalMaterializer<Upstream::Key, S::Item>,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S: Signal> MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    Upstream::Key: Clone + Ord + Send + Sync + 'static,
    S: super::signal::ReplayableSignal + 'static,
    S::Item: Clone + Send + Sync + 'static,
{
    /// Returns this already-verified replayable terminal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable terminal in-place to broadcast fan-out without adding another
    /// cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastMap<Upstream::Key, S::Item> {
        replayable_map_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> ReplayableScheduledMap<Sched, Upstream::Key, S::Item>
    where
        Upstream: 'static,
    {
        replayable_map_schedule(self)
    }
}

impl<Upstream, S: Signal> SignalMap for MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    S: super::signal::ReplayableSignal + 'static,
    S::Item: Clone + Send + Sync + 'static,
{
    type Key = Upstream::Key;
    type Value = S::Item;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

impl<Upstream, S> replayable_signal_map_sealed::Sealed for MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    Upstream::Key: Clone + Ord + Send + Sync + 'static,
    S: super::signal::ReplayableSignal,
    S::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream, S> ReplayableSignalMap for MapValueSignal<Upstream, S>
where
    Upstream: SignalMap,
    Upstream::Key: Clone + Ord + Send + Sync + 'static,
    S: super::signal::ReplayableSignal,
    S::Item: Clone + Send + Sync + 'static,
{
}

/// The verified replayable [`Signal`] returned by [`SignalMapExt::key`].
pub struct Key<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: 'static,
{
    inner: signal::CachedSignal<KeyTerminal<Upstream>>,
}

impl<Upstream> Key<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: 'static,
{
    /// Returns this already-verified replayable key lookup without adding another cache entity.
    pub fn cache(self) -> Self {
        Self {
            inner: self.inner.cache(),
        }
    }

    /// Normalizes this replayable key lookup terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> signal::ReplayableBroadcast<Option<Upstream::Value>>
    where
        Upstream::Value: Clone + Send + Sync + 'static,
    {
        self.inner.broadcast()
    }

    /// Assigns a schedule to this verified replayable key lookup terminal while preserving its
    /// replay capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> signal::ReplayableScheduled<Sched, Option<Upstream::Value>>
    where
        Upstream::Value: Clone + Send + Sync + 'static,
    {
        self.inner.schedule::<Sched>()
    }
}

impl<Upstream> Signal for Key<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: Clone + Send + Sync + 'static,
{
    type Item = Option<Upstream::Value>;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        Box::new(self.inner).into_signal_materializer()
    }
}

impl<Upstream> signal::replayable_signal_sealed::Sealed for Key<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: 'static,
{
}

impl<Upstream> signal::ReplayableSignal for Key<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: Clone + Send + Sync + 'static,
{
}

struct KeyTerminal<Upstream>
where
    Upstream: SignalMap,
    Upstream::Value: 'static,
{
    materializer: crate::ScalarSignalMaterializer<Option<Upstream::Value>>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> Signal for KeyTerminal<Upstream>
where
    Upstream: SignalMap,
    Upstream::Key: 'static,
    Upstream::Value: 'static,
{
    type Item = Option<Upstream::Value>;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "tracing")] {
        /// Signal graph node that debug logs its upstream's "raw" [`Vec<MapDiff>`]s, see
        /// [`.debug`](SignalMapExt::debug).
        pub struct Debug<Upstream>
        where
            Upstream: SignalMap,
        {
            materializer: MapSignalMaterializer<Upstream::Key, Upstream::Value>,
            _marker: PhantomData<fn() -> Upstream>,
        }

        impl<Upstream> SignalMap for Debug<Upstream>
        where
            Upstream: SignalMap,
        {
            type Key = Upstream::Key;
            type Value = Upstream::Value;

            fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
                self.materializer
            }
        }
    }
}

/// The verified replayable key vector returned by [`MutableBTreeMap::signal_vec_keys`].
pub struct SignalVecKeys<K: 'static> {
    inner: CachedSignalVec<SignalVecKeysTerminal<K>>,
}

impl<K: 'static> SignalVecKeys<K> {
    /// Returns this already-verified replayable key vector without adding another cache entity.
    pub fn cache(self) -> Self {
        Self {
            inner: self.inner.cache(),
        }
    }

    /// Normalizes this replayable key vector terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> crate::signal_vec::ReplayableBroadcastVec<K>
    where
        K: Clone + Send + Sync + 'static,
    {
        self.inner.broadcast()
    }

    /// Assigns a schedule to this verified replayable key vector terminal while preserving its
    /// replay capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> crate::signal_vec::ReplayableScheduledVec<Sched, K>
    where
        K: Clone + Send + Sync + 'static,
    {
        self.inner.schedule::<Sched>()
    }
}

impl<K> SignalVec for SignalVecKeys<K>
where
    K: Clone + Send + Sync + 'static,
{
    type Item = K;

    fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
        Box::new(self.inner).into_signal_vec_materializer()
    }
}

impl<K: 'static> crate::signal_vec::replayable_signal_vec_sealed::Sealed for SignalVecKeys<K> {}

impl<K> crate::signal_vec::ReplayableSignalVec for SignalVecKeys<K> where K: Clone + Send + Sync + 'static {}

struct SignalVecKeysTerminal<K: 'static> {
    materializer: crate::VecSignalMaterializer<K>,
}

impl<K> SignalVec for SignalVecKeysTerminal<K>
where
    K: 'static,
{
    type Item = K;

    fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// The verified replayable entry vector returned by [`MutableBTreeMap::signal_vec_entries`].
pub struct SignalVecEntries<K: 'static, V: 'static> {
    inner: CachedSignalVec<SignalVecEntriesTerminal<K, V>>,
}

impl<K: 'static, V: 'static> SignalVecEntries<K, V> {
    /// Returns this already-verified replayable entry vector without adding another cache entity.
    pub fn cache(self) -> Self {
        Self {
            inner: self.inner.cache(),
        }
    }

    /// Normalizes this replayable entry vector terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> crate::signal_vec::ReplayableBroadcastVec<(K, V)>
    where
        K: Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        self.inner.broadcast()
    }

    /// Assigns a schedule to this verified replayable entry vector terminal while preserving its
    /// replay capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> crate::signal_vec::ReplayableScheduledVec<Sched, (K, V)>
    where
        K: Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        self.inner.schedule::<Sched>()
    }
}

impl<K, V> SignalVec for SignalVecEntries<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    type Item = (K, V);

    fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
        Box::new(self.inner).into_signal_vec_materializer()
    }
}

impl<K: 'static, V: 'static> crate::signal_vec::replayable_signal_vec_sealed::Sealed for SignalVecEntries<K, V> {}

impl<K, V> crate::signal_vec::ReplayableSignalVec for SignalVecEntries<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
}

struct SignalVecEntriesTerminal<K: 'static, V: 'static> {
    materializer: crate::VecSignalMaterializer<(K, V)>,
}

impl<K, V> SignalVec for SignalVecEntriesTerminal<K, V>
where
    K: 'static,
    V: 'static,
{
    type Item = (K, V);

    fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Enables returning different concrete [`SignalMap`] types from branching logic without boxing,
/// although note that all [`SignalMap`]s are boxed internally regardless.
///
/// Inspired by <https://github.com/rayon-rs/either>.
#[allow(missing_docs)]
pub enum SignalMapEither<L, R>
where
    L: SignalMap,
    R: SignalMap,
{
    Left(L),
    Right(R),
}

impl<K, V, L, R> SignalMapEither<L, R>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    L: ReplayableSignalMap<Key = K, Value = V>,
    R: ReplayableSignalMap<Key = K, Value = V>,
{
    /// Returns this already-verified replayable branch without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes the selected replayable branch terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastMap<K, V> {
        replayable_map_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable branch while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledMap<Sched, K, V>
    where
        L: 'static,
        R: 'static,
    {
        replayable_map_schedule(self)
    }
}

impl<K: 'static, V: 'static, L, R> SignalMap for SignalMapEither<L, R>
where
    L: SignalMap<Key = K, Value = V>,
    R: SignalMap<Key = K, Value = V>,
{
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        match *self {
            SignalMapEither::Left(left) => Box::new(left).into_signal_map_materializer(),
            SignalMapEither::Right(right) => Box::new(right).into_signal_map_materializer(),
        }
    }
}

impl<K, V, L, R> replayable_signal_map_sealed::Sealed for SignalMapEither<L, R>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    L: ReplayableSignalMap<Key = K, Value = V>,
    R: ReplayableSignalMap<Key = K, Value = V>,
{
}

impl<K, V, L, R> ReplayableSignalMap for SignalMapEither<L, R>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    L: ReplayableSignalMap<Key = K, Value = V>,
    R: ReplayableSignalMap<Key = K, Value = V>,
{
}

/// Blanket trait for transforming [`SignalMap`]s into [`SignalMapEither::Left`] or
/// [`SignalMapEither::Right`].
pub trait IntoSignalMapEither: Sized
where
    Self: SignalMap,
{
    /// Wrap this [`SignalMap`] in the [`SignalMapEither::Left`] variant.
    ///
    /// Useful for conditional branching where different [`SignalMap`] types need to be returned
    /// from the same function or closure, particularly with
    /// [`.switch_signal_map`](SignalExt::switch_signal_map).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource)]
    /// struct DoubleValues(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(DoubleValues(true));
    ///
    /// let map = MutableBTreeMap::builder()
    ///     .values([(1, 10), (2, 20)])
    ///     .spawn(&mut world);
    ///
    /// let signal = signal::from_system(|In(_), res: Res<DoubleValues>| res.0).switch_signal_map(
    ///     move |In(double): In<bool>, world: &mut World| {
    ///         if double {
    ///             map.signal_map()
    ///                 .map_value(|In(v): In<i32>| v * 2)
    ///                 .cache()
    ///                 .left_either()
    ///         } else {
    ///             map.signal_map().right_either()
    ///         }
    ///     },
    /// );
    /// // both branches produce compatible SignalMapEither types
    /// ```
    fn left_either<R>(self) -> SignalMapEither<Self, R>
    where
        R: SignalMap,
    {
        SignalMapEither::Left(self)
    }

    /// Wrap this [`SignalMap`] in the [`SignalMapEither::Right`] variant.
    ///
    /// Useful for conditional branching where different [`SignalMap`] types need to be returned
    /// from the same function or closure, particularly with
    /// [`.switch_signal_map`](SignalExt::switch_signal_map).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource)]
    /// struct MapSelector(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(MapSelector(false));
    ///
    /// let map_a = MutableBTreeMap::builder()
    ///     .values([(1, 10), (2, 20)])
    ///     .spawn(&mut world);
    /// let map_b = MutableBTreeMap::builder()
    ///     .values([(1, 100), (2, 200), (3, 300)])
    ///     .spawn(&mut world);
    ///
    /// let signal = signal::from_system(|In(_), res: Res<MapSelector>| res.0).switch_signal_map(
    ///     move |In(use_a): In<bool>, world: &mut World| {
    ///         if use_a {
    ///             map_a
    ///                 .signal_map()
    ///                 .map_value(|In(v): In<i32>| v * 10)
    ///                 .cache()
    ///                 .left_either()
    ///         } else {
    ///             map_b.signal_map().right_either()
    ///         }
    ///     },
    /// );
    /// // both branches produce compatible SignalMapEither types
    /// ```
    fn right_either<L>(self) -> SignalMapEither<L, Self>
    where
        L: SignalMap,
    {
        SignalMapEither::Right(self)
    }
}

impl<T: SignalMap> IntoSignalMapEither for T {}

/// Extension trait providing combinator methods for [`SignalMap`]s.
pub trait SignalMapExt: SignalMap {
    /// Explicitly shares this map signal across multiple downstream consumers.
    ///
    /// Cloning the returned [`BroadcastMap`] handle shares one canonical lazy terminal identity
    /// without acquiring a graph claim until a clone is registered or consumed by a downstream
    /// combinator. Verified replayable terminals normalize in-place to broadcast delivery without
    /// adding an extra entity or maintenance claim; non-replayable inputs materialize one
    /// canonical identity broadcast entity.
    ///
    /// Maps with move-only (`!Clone`) keys or values cannot be broadcast:
    ///
    /// ```compile_fail,E0277
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// struct NonClone(i32);
    ///
    /// let mut world = World::new();
    /// let map = MutableBTreeMap::builder().values([(1, 2)]).spawn(&mut world);
    /// let _ = map.signal_map().map_value(|In(v): In<i32>| NonClone(v)).broadcast();
    /// ```
    fn broadcast(self) -> BroadcastMap<Self::Key, Self::Value>
    where
        Self: Sized,
        Self::Key: Clone + Send + Sync + 'static,
        Self::Value: Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_map_materializer();
        let materializer = MapSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(upstream)?;
            if let Some(upstream) = registrar.map_try_broadcast_terminal(upstream)? {
                return Ok(upstream);
            }
            registrar.signal_map_entity(
                upstream,
                |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>| Some(diffs),
                OutputDelivery::broadcast(),
            )
        });
        BroadcastMap {
            materializer,
            _mode: PhantomData,
        }
    }

    /// Normalizes this terminal to framework-managed replayability.
    ///
    /// A non-replayable terminal gains one cache entity and maintenance lifecycle. An already
    /// verified replayable terminal preserves its identity and adds neither another entity nor a
    /// second maintenance claim.
    ///
    /// Maps with move-only (`!Clone`) keys or values cannot be cached for replay:
    ///
    /// ```compile_fail,E0277
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// struct NonClone(i32);
    ///
    /// let mut world = World::new();
    /// let map = MutableBTreeMap::builder().values([(1, 2)]).spawn(&mut world);
    /// let _ = map.signal_map().map_value(|In(v): In<i32>| NonClone(v)).cache();
    /// ```
    fn cache(self) -> CachedSignalMap<Self>
    where
        Self: Sized,
        Self::Key: Clone + Ord + Send + Sync + 'static,
        Self::Value: Clone + Send + Sync + 'static,
    {
        let materializer = MapSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(self).into_signal_map_materializer())?;
            if let Some(upstream) = registrar.map_cache_terminal(upstream)? {
                return Ok(upstream);
            }
            registrar.replayable_signal_map_entity(
                upstream,
                |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>| Some(diffs),
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        CachedSignalMap {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass the "raw" [`Vec<MapDiff<Self::Key, Self::Value>>`] output of this [`SignalMap`] to a
    /// [`System`], continuing propagation if the [`System`] returns [`Some`] or terminating for the
    /// frame if it returns [`None`]. This transforms the `SignalMap` into a `Signal`. Unlike most
    /// other [`SignalMap`] methods, [`.for_each`](SignalMapExt::for_each), returns a [`Signal`],
    /// not a [`SignalMap`], since the output type need not be an [`Option<Vec<MapDiff>>`]. If the
    /// [`System`] logic is infallible, wrapping the result in an option is unnecessary.
    fn for_each<O, IOO, F, M>(self, system: F) -> ForEach<Self, O>
    where
        Self: Sized,
        Self::Key: Send + Sync + 'static,
        Self::Value: Send + Sync + 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Vec<MapDiff<Self::Key, Self::Value>>>, IOO, M> + Send + Sync + 'static,
    {
        let materializer = crate::ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(self).into_signal_map_materializer())?;
            registrar.signal_entity(upstream, system, OutputDelivery::move_only())
        });
        ForEach {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Value`](SignalMap::Value) of this [`SignalMap`] to a [`System`], transforming
    /// it linearly without installing a replay cache.
    ///
    /// Unchanged keys and mapped values are forwarded by move through [`MapDiff`], so neither
    /// `Self::Key` nor `O` is required to implement [`Clone`] or [`Ord`]. Compose
    /// [`.cache()`](SignalMapExt::cache) downstream when verified replayability is required (such
    /// as inside [`.switch_signal_map`](crate::signal::SignalExt::switch_signal_map) or
    /// [`.boxed_replayable`](SignalMapExt::boxed_replayable)).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableBTreeMap::builder()
    ///     .values([(1, 2), (3, 4)])
    ///     .spawn(&mut world)
    ///     .signal_map()
    ///     .map_value(|In(x)| x * 2); // outputs `SignalMap -> {1: 4, 3: 8}`
    /// ```
    ///
    /// Because `map_value` is a linear diff adapter rather than an implicit replay cache, add
    /// [`.cache()`](SignalMapExt::cache) before replay-only consumers:
    ///
    /// ```compile_fail,E0277
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let map = MutableBTreeMap::builder().values([(1, 2)]).spawn(&mut world);
    /// let _ = map.signal_map().map_value(|In(x): In<i32>| x * 2).boxed_replayable();
    /// ```
    fn map_value<O, F, M>(self, system: F) -> MapValue<Self, O>
    where
        Self: Sized,
        Self::Key: Send + Sync + 'static,
        Self::Value: Send + Sync + 'static,
        O: Send + Sync + 'static,
        F: IntoSystem<In<Self::Value>, O, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_map_materializer();
        let materializer = MapSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(upstream)?;
            let mut system = IntoSystem::into_system(system);
            let mut initialized = false;
            registrar.signal_map_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, world: &mut World| {
                    if !initialized {
                        system.initialize(world);
                        initialized = true;
                    }
                    let mut out_diffs = Vec::with_capacity(diffs.len());
                    for diff in diffs {
                        let new_diff = diff.map_value(|value| {
                            system.run(value, world).unwrap_or_else(|err| {
                                panic!(
                                    "SignalMap::map_value value system failed or skipped ({err}); collection item transforms must be total per entry (use map_value_signal for async/warming per-entry state)"
                                )
                            })
                        });
                        out_diffs.push(new_diff);
                    }
                    if out_diffs.is_empty() { None } else { Some(out_diffs) }
                },
                OutputDelivery::move_only(),
            )
        });
        MapValue {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Value`](SignalMap::Value) of this [`SignalMap`] to a [`System`] that produces a
    /// [`Signal`], forwarding the output of each resulting [`Signal`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableBTreeMap::builder().values([(1, 2), (3, 4)])
    ///     .spawn(&mut world)
    ///     .signal_map()
    ///     .map_value_signal(|In(x)|
    ///         signal::from_system(move |In(_)| x * 2).dedupe()
    ///     ); // outputs `SignalMap -> {1: 4, 3: 8}`
    /// ```
    fn map_value_signal<S, F, M>(self, system: F) -> MapValueSignal<Self, S>
    where
        Self: Sized,
        Self::Key: Ord + Clone + Send + Sync + 'static,
        Self::Value: Send + Sync + 'static,
        S: super::signal::ReplayableSignal + 'static,
        S::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Value>, S, M> + Send + Sync + 'static,
    {
        enum MapValueSignalQueueItem<K, V> {
            Replace { keys: Vec<K> },
            Insert { key: K },
            Remove { key: K },
            Clear,
            Value { key: K, value: V },
        }

        let upstream = Box::new(self).into_signal_map_materializer();
        let materializer = MapSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(upstream)?;
            let mut factory = IntoSystem::into_system(system);
            let mut initialized = false;
            let manager = registrar.signal_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>, world: &mut World| {
                    if !initialized {
                        factory.initialize(world);
                        initialized = true;
                    }
                    let mut steps = Vec::new();
                    for diff in diffs {
                        match diff {
                            MapDiff::Replace { entries } => {
                                let mut selected = Vec::with_capacity(entries.len());
                                let mut keys = Vec::with_capacity(entries.len());
                                for (key, value) in entries {
                                    let signal = factory
                                        .run(value, world)
                                        .expect("map_value_signal factory system failed to run");
                                    keys.push(key.clone());
                                    selected.push((key, signal));
                                }
                                steps.push(DynamicManagerStep::command(
                                    KeyedSlotCommand::ReplaceAll { entries: selected },
                                    DynamicEmissions::One(MapValueSignalQueueItem::Replace { keys }),
                                ));
                            }
                            MapDiff::Insert { key, value } => {
                                let signal = factory
                                    .run(value, world)
                                    .expect("map_value_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    KeyedSlotCommand::Insert {
                                        key: key.clone(),
                                        signal,
                                    },
                                    DynamicEmissions::One(MapValueSignalQueueItem::Insert { key }),
                                ));
                            }
                            MapDiff::Update { key, value } => {
                                let signal = factory
                                    .run(value, world)
                                    .expect("map_value_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    KeyedSlotCommand::Replace { key, signal },
                                    DynamicEmissions::None,
                                ));
                            }
                            MapDiff::Remove { key } => {
                                steps.push(DynamicManagerStep::command(
                                    KeyedSlotCommand::Remove { key: key.clone() },
                                    DynamicEmissions::One(MapValueSignalQueueItem::Remove { key }),
                                ));
                            }
                            MapDiff::Clear => {
                                steps.push(DynamicManagerStep::command(
                                    KeyedSlotCommand::Clear,
                                    DynamicEmissions::One(MapValueSignalQueueItem::Clear),
                                ));
                            }
                        }
                    }
                    Some(DynamicManagerUpdate::sequenced(steps))
                },
                OutputDelivery::move_only(),
            )?;
            let epoch = Arc::new(AtomicUsize::new(0));
            let mut entries = BTreeMap::<Self::Key, Option<S::Item>>::new();
            let reducer_epoch = Arc::clone(&epoch);
            let mut reducer_observed_epoch = 0usize;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<S::Item>()
                    .adapter(|key, value| DynamicEmissions::One(MapValueSignalQueueItem::Value { key, value }))
                    .replayable_signal_map_output(move |items| {
                        let current_epoch = reducer_epoch.load(AtomicOrdering::Acquire);
                        if reducer_observed_epoch != current_epoch {
                            reducer_observed_epoch = current_epoch;
                            entries.clear();
                        }
                        let mut output = Vec::new();
                        let mut replacing = false;
                        for item in items {
                            match item {
                                MapValueSignalQueueItem::Replace { keys } => {
                                    let mut replacement = BTreeMap::new();
                                    for key in keys {
                                        replacement
                                            .insert(key.clone(), entries.get(&key).and_then(|value| value.clone()));
                                    }
                                    entries = replacement;
                                    replacing = true;
                                }
                                MapValueSignalQueueItem::Insert { key } => {
                                    entries.insert(key, None);
                                }
                                MapValueSignalQueueItem::Remove { key } => {
                                    if entries.remove(&key).is_some() && !replacing {
                                        output.push(MapDiff::Remove { key });
                                    }
                                }
                                MapValueSignalQueueItem::Clear => {
                                    if !entries.is_empty() {
                                        entries.clear();
                                        if !replacing {
                                            output.push(MapDiff::Clear);
                                        }
                                    }
                                }
                                MapValueSignalQueueItem::Value { key, value } => {
                                    let previous = entries.insert(key.clone(), Some(value.clone()));
                                    if !replacing {
                                        output.push(match previous {
                                            Some(Some(_)) => MapDiff::Update { key, value },
                                            Some(None) | None => MapDiff::Insert { key, value },
                                        });
                                    }
                                }
                            }
                        }
                        if replacing {
                            let entries = entries
                                .iter()
                                .map(|(key, value)| {
                                    (
                                        key.clone(),
                                        value.clone().unwrap_or_else(|| {
                                            panic!(
                                                "map_value_signal replacement published before all values were ready"
                                            )
                                        }),
                                    )
                                })
                                .collect();
                            Some(vec![MapDiff::Replace { entries }])
                        } else {
                            (!output.is_empty()).then_some(output)
                        }
                    })
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_deactivate(move || {
                        epoch.fetch_add(1, AtomicOrdering::AcqRel);
                    }),
            )
        });
        MapValueSignal {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Maps this [`SignalMap`] to a [`Key`]-lookup [`Signal`] which outputs [`Some<Value>`] if the
    /// [`Key`] is present and [`None`] otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableBTreeMap::builder()
    ///     .values([(1, 2), (3, 4)])
    ///     .spawn(&mut world)
    ///     .signal_map()
    ///     .key(1); // outputs `2`
    /// ```
    fn key(self, key: Self::Key) -> Key<Self>
    where
        Self: Sized,
        Self::Key: PartialEq + Send + Sync + 'static,
        Self::Value: Clone + Send + Sync + 'static,
    {
        let materializer = crate::ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(self).into_signal_map_materializer())?;
            let mut initialized = false;
            let mut present = false;
            registrar.replayable_signal_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>| {
                    let mut changed = false;
                    let mut new_value = None;
                    for diff in diffs {
                        match diff {
                            MapDiff::Replace { entries } => {
                                new_value = entries
                                    .into_iter()
                                    .find(|(map_key, _)| *map_key == key)
                                    .map(|(_, value)| value);
                                initialized = true;
                                present = new_value.is_some();
                                changed = true;
                            }
                            MapDiff::Insert { key: map_key, value } | MapDiff::Update { key: map_key, value } => {
                                if map_key == key {
                                    initialized = true;
                                    present = true;
                                    new_value = Some(value);
                                    changed = true;
                                }
                            }
                            MapDiff::Remove { key: map_key } => {
                                if map_key == key && (present || !initialized) {
                                    initialized = true;
                                    present = false;
                                    new_value = None;
                                    changed = true;
                                }
                            }
                            MapDiff::Clear => {
                                if present || !initialized {
                                    initialized = true;
                                    present = false;
                                    new_value = None;
                                    changed = true;
                                }
                            }
                        }
                    }
                    changed.then_some(new_value)
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        Key {
            inner: SignalExt::cache(KeyTerminal {
                materializer,
                _marker: PhantomData,
            }),
        }
    }

    #[cfg(feature = "tracing")]
    #[track_caller]
    /// Adds debug logging to this [`SignalMap`]'s raw [`MapDiff`] outputs.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut map = MutableBTreeMap::builder()
    ///     .values([(1, 2), (3, 4)])
    ///     .spawn(&mut world);
    /// let signal = map.signal_map().debug();
    /// // `signal` logs `[ Replace { entries: [ (1, 2), (3, 4) ] } ]`
    /// map.write(&mut world).insert(5, 6);
    /// // `signal` logs `[ Insert { key: 5, value: 6 } ]` on next update
    /// ```
    fn debug(self) -> Debug<Self>
    where
        Self: Sized,
        Self::Key: fmt::Debug + Send + Sync + 'static,
        Self::Value: fmt::Debug + Send + Sync + 'static,
    {
        let location = core::panic::Location::caller();
        let materializer = MapSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(self).into_signal_map_materializer())?;
            registrar.signal_map_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<Self::Key, Self::Value>>>| {
                    bevy_log::debug!("[{}] {:#?}", location, diffs);
                    Some(diffs)
                },
                OutputDelivery::move_only(),
            )
        });
        Debug {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Erases the type of this [`SignalMap`], allowing it to be used in conjunction with
    /// [`SignalMap`]s of other concrete types.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let condition = true;
    /// let signal = if condition {
    ///     MutableBTreeMap::builder()
    ///         .values([(1, 2), (3, 4)])
    ///         .spawn(&mut world)
    ///         .signal_map()
    ///         .map_value(|In(x): In<i32>| x * 2)
    ///         .boxed() // this is a `MapValue<Source<i32, i32>>`
    /// } else {
    ///     MutableBTreeMap::builder()
    ///         .values([(1, 2), (3, 4)])
    ///         .spawn(&mut world)
    ///         .signal_map()
    ///         .map_value_signal(|In(x): In<i32>| signal::from_system(move |In(_)| x * 2).dedupe())
    ///         .boxed() // this is a `MapValueSignal<Source<i32, i32>>`
    /// }; // without the `.boxed()`, the compiler would not allow this
    /// ```
    fn boxed(self) -> BoxedSignalMap<Self::Key, Self::Value>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Erases this map signal's concrete type while preserving verified replay capability.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system(|In(_)| true).switch_signal_map(
    ///     |In(condition): In<bool>, world: &mut World| {
    ///         if condition {
    ///             MutableBTreeMap::builder()
    ///                 .values([(1, 2), (3, 4)])
    ///                 .spawn(world)
    ///                 .signal_map()
    ///                 .map_value(|In(x): In<i32>| x * 2)
    ///                 .cache()
    ///                 .boxed_replayable() // this is a cached `MapValue<Source<i32, i32>>`
    ///         } else {
    ///             MutableBTreeMap::builder()
    ///                 .values([(1, 2), (3, 4)])
    ///                 .spawn(world)
    ///                 .signal_map()
    ///                 .map_value_signal(|In(x): In<i32>| {
    ///                     signal::from_system(move |In(_)| x * 2).dedupe()
    ///                 })
    ///                 .boxed_replayable() // this is a `MapValueSignal<Source<i32, i32>>`
    ///         } // without `.boxed_replayable()`, the compiler would not allow this
    ///     },
    /// );
    /// ```
    fn boxed_replayable(self) -> BoxedReplayableSignalMap<Self::Key, Self::Value>
    where
        Self: Sized + ReplayableSignalMap,
        Self::Key: Clone + Ord + Send + Sync + 'static,
        Self::Value: Clone + Send + Sync + 'static,
    {
        BoxedReplayableSignalMap(Box::new(self))
    }

    /// Assign a schedule to this signal map chain, see [`SignalExt::schedule`].
    ///
    /// # Panics
    ///
    /// Panics if the schedule has not been registered with
    /// [`JonmoPlugin::with_schedule`](crate::JonmoPlugin::with_schedule).
    fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ScheduledMap<Sched, Self::Key, Self::Value>
    where
        Self: Sized + 'static,
    {
        let materializer = MapSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(self).into_signal_map_materializer())?;
            registrar.apply_schedule(upstream, Sched::default())
        });
        ScheduledMap {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Attempts to materialize this map signal through its framework-owned registrar and acquire
    /// one claim.
    ///
    /// Complete allocation-free preflight returns a structured [`SignalRegistrationError`] on
    /// detectable rejection without mutating graph or Bevy allocation/materialization state.
    /// Successful commit has no ordinary structured failure path, including through dynamic
    /// collection operators and external implementations.
    fn try_register(self, world: &mut World) -> Result<SignalClaim, SignalRegistrationError>
    where
        Self: Sized,
    {
        Box::new(self).into_signal_map_materializer().try_register(world)
    }

    /// Activate this [`SignalMap`] and all its upstreams, causing them to be evaluated every frame
    /// until they are [`SignalClaim::release`]-ed, see [`SignalClaim`].
    fn register(self, world: &mut World) -> SignalClaim
    where
        Self: Sized,
    {
        Box::new(self).into_signal_map_materializer().register(world)
    }
}

impl<T: ?Sized> SignalMapExt for T where T: SignalMap {}

/// Signal map node wrapper that assigns a schedule to a signal chain, see
/// [`.schedule`](SignalMapExt::schedule).
pub struct ScheduledMap<Sched, K: 'static, V: 'static, Mode = OrdinaryBroadcast> {
    materializer: MapSignalMaterializer<K, V>,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<fn() -> (Sched, Mode)>,
}

/// Alias for a [`ScheduledMap`] handle backed by a verified [`ReplayableSignalMap`] terminal.
pub type ReplayableScheduledMap<Sched, K, V> = ScheduledMap<Sched, K, V, ReplayableBroadcastMarker>;

impl<Sched: 'static, K: 'static, V: 'static> ScheduledMap<Sched, K, V, ReplayableBroadcastMarker> {
    /// Returns this already-verified replayable scheduled map without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable scheduled map terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastMap<K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_broadcast(self)
    }

    /// Reassigns this verified replayable map terminal's schedule while preserving its replay
    /// capability.
    pub fn schedule<NextSched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledMap<NextSched, K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_schedule(self)
    }
}

impl<Sched: 'static, K: 'static, V: 'static, Mode: Send + Sync + 'static> SignalMap
    for ScheduledMap<Sched, K, V, Mode>
{
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

impl<Sched: 'static, K: 'static, V: 'static> replayable_signal_map_sealed::Sealed
    for ScheduledMap<Sched, K, V, ReplayableBroadcastMarker>
{
}

impl<Sched: 'static, K: Clone + Ord + Send + Sync + 'static, V: Clone + Send + Sync + 'static> ReplayableSignalMap
    for ScheduledMap<Sched, K, V, ReplayableBroadcastMarker>
{
}

/// Provides immutable access to the underlying [`BTreeMap`].
pub struct MutableBTreeMapReadGuard<'s, K, V> {
    guard: &'s MutableBTreeMapData<K, V>,
}

impl<'s, K, V> Deref for MutableBTreeMapReadGuard<'s, K, V> {
    type Target = BTreeMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
}

/// Provides limited mutable access to the underlying [`BTreeMap`].
pub struct MutableBTreeMapWriteGuard<'s, K, V> {
    guard: Mut<'s, MutableBTreeMapData<K, V>>,
}

impl<'s, K, V> Deref for MutableBTreeMapWriteGuard<'s, K, V> {
    type Target = BTreeMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.guard.map
    }
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
    /// If the map did have this key present, the value is updated, and the old value is returned
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
        self.guard.queue_diff(diff);
        old
    }

    /// Removes a key from this [`MutableBTreeMap`], queueing a [`MapDiff::Remove`] and returning
    /// the value at the key if the key was previously present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let old = self.guard.map.remove(key);
        if old.is_some() {
            self.guard.queue_diff(MapDiff::Remove { key: key.clone() });
        }
        old
    }

    /// Clears this [`MutableBTreeMap`], removing all elements and queueing a [`MapDiff::Clear`] if
    /// any elements were present.
    pub fn clear(&mut self) {
        if !self.guard.map.is_empty() {
            self.guard.map.clear();

            self.guard.queue_diff(MapDiff::Clear);
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
        self.guard.queue_diff(MapDiff::Replace { entries });
    }
}

type MapRevision = u64;

#[derive(Clone)]
struct VersionedMapDiff<K, V> {
    revision: MapRevision,
    diff: MapDiff<K, V>,
}

/// [`Component`] that holds the actual state for a [`MutableBTreeMap`].
#[derive(Component)]
pub struct MutableBTreeMapData<K, V> {
    map: BTreeMap<K, V>,
    lifetime: Weak<MutableBTreeMapDataLifetime>,
    revision: MapRevision,
    pending_diffs: Vec<VersionedMapDiff<K, V>>,
}

impl<K, V> MutableBTreeMapData<K, V> {
    fn queue_diff(&mut self, diff: MapDiff<K, V>) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("MutableBTreeMap source revision overflow");

        self.pending_diffs.push(VersionedMapDiff {
            revision: self.revision,
            diff,
        });
    }
}

struct MutableBTreeMapDataLifetime {
    entity: Entity,
    cleanup: CleanupSender,
}

impl Drop for MutableBTreeMapDataLifetime {
    fn drop(&mut self) {
        self.cleanup.send_mutable_btree_map(self.entity);
    }
}

struct MutableBTreeMapOwner<K: 'static, V: 'static> {
    lifetime: Arc<MutableBTreeMapDataLifetime>,
    source: SignalMaterializerFactory<MapSignalKind<K, V>>,
}

/// Wrapper around a [`BTreeMap`] that emits mutations as [`MapDiff`]s, enabling diff-less
/// constant-time reactive updates for downstream [`SignalMap`]s.
pub struct MutableBTreeMap<K: 'static, V: 'static> {
    owner: Arc<MutableBTreeMapOwner<K, V>>,
    _marker: PhantomData<fn() -> (K, V)>,
}

impl<K, V> Clone for MutableBTreeMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            _marker: PhantomData,
        }
    }
}

/// Signal graph node with no upstreams which forwards [`Vec<MapDiff<K, V>>`]s flushed from some
/// source [`MutableBTreeMap<K, V>`], see [`MutableBTreeMap::signal_map`].
pub struct Source<K: 'static, V: 'static> {
    materializer: MapSignalMaterializer<K, V>,
}

impl<K, V> Source<K, V> {
    /// Returns this already-verified cache without adding another graph entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Returns a cloneable broadcast wrapper backed directly by this [`MutableBTreeMap`] source
    /// terminal without adding another entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastMap<K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_broadcast(self)
    }

    /// Assigns a schedule to this [`MutableBTreeMap`] source terminal while preserving its verified
    /// replay capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledMap<Sched, K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        replayable_map_schedule(self)
    }
}

impl<K, V> SignalMap for Source<K, V>
where
    K: 'static,
    V: 'static,
{
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

impl<K: 'static, V: 'static> replayable_signal_map_sealed::Sealed for Source<K, V> {}
impl<K: Clone + Ord + Send + Sync + 'static, V: Clone + Send + Sync + 'static> ReplayableSignalMap for Source<K, V> {}

fn new_mutable_btree_map_data<K, V>(
    map: BTreeMap<K, V>,
    lifetime: &Arc<MutableBTreeMapDataLifetime>,
) -> MutableBTreeMapData<K, V> {
    MutableBTreeMapData {
        lifetime: Arc::downgrade(lifetime),
        map,
        revision: 0,
        pending_diffs: Vec::new(),
    }
}

fn mutable_btree_map_from_parts<K, V>(lifetime: Arc<MutableBTreeMapDataLifetime>) -> MutableBTreeMap<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    let entity = lifetime.entity;
    let source_lifetime = lifetime.clone();
    let source = MapSignalMaterializer::new(move |registrar| {
        let data = registrar.preflight_component::<MutableBTreeMapData<K, V>>(entity)?;
        // Entity IDs can collide across worlds; the data must belong to this source's lifetime.
        if !Weak::ptr_eq(&data.lifetime, &Arc::downgrade(&source_lifetime)) {
            return Err(SignalRegistrationError::EntityUnavailable);
        }
        let current = data.map.clone();
        let initial_revision = data.revision;
        registrar.replayable_signal_map_source(
            current,
            move |In(()),
                  mut mutable_btree_map_datas: Query<&mut MutableBTreeMapData<K, V>>,
                  mut last_applied_revision: Local<Option<MapRevision>>| {
                let _source_lifetime = &source_lifetime;
                let mut data = mutable_btree_map_datas.get_mut(entity).ok()?;
                let cursor = last_applied_revision.get_or_insert(initial_revision);
                let mut public_diffs = Vec::new();
                for VersionedMapDiff { revision, diff } in core::mem::take(&mut data.pending_diffs) {
                    if revision > *cursor {
                        *cursor = revision;
                        public_diffs.push(diff);
                    }
                }
                if public_diffs.is_empty() {
                    None
                } else {
                    Some(public_diffs)
                }
            },
            OutputDelivery::broadcast(),
        )
    })
    .into_factory();
    MutableBTreeMap {
        owner: Arc::new(MutableBTreeMapOwner { lifetime, source }),
        _marker: PhantomData,
    }
}

fn spawn_mutable_btree_map<K, V>(world: &mut World, map: BTreeMap<K, V>) -> MutableBTreeMap<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    let cleanup = CleanupSender::bound(world);
    let entity = world.spawn_empty().id();
    let lifetime = Arc::new(MutableBTreeMapDataLifetime { entity, cleanup });
    world
        .entity_mut(entity)
        .insert(new_mutable_btree_map_data(map, &lifetime));
    mutable_btree_map_from_parts(lifetime)
}

fn spawn_mutable_btree_map_commands<K, V>(commands: &mut Commands, map: BTreeMap<K, V>) -> MutableBTreeMap<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    let (cleanup, binder) = CleanupSender::deferred();
    let entity = commands.spawn_empty().id();
    let lifetime = Arc::new(MutableBTreeMapDataLifetime { entity, cleanup });
    commands
        .entity(entity)
        .insert(new_mutable_btree_map_data(map, &lifetime));
    commands.queue(move |world: &mut World| binder.bind(world));
    mutable_btree_map_from_parts(lifetime)
}

impl<K, V> MutableBTreeMap<K, V> {
    /// Creates a [`MutableBTreeMapBuilder`] for constructing a [`MutableBTreeMap`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// // Empty MutableBTreeMap
    /// let map = MutableBTreeMap::<i32, &str>::builder().spawn(&mut world);
    /// // With initial values
    /// let map = MutableBTreeMap::builder()
    ///     .values([(1, "one"), (2, "two")])
    ///     .spawn(&mut world);
    /// // With custom initialization
    /// let map = MutableBTreeMap::builder()
    ///     .with_values(|m| {
    ///         m.insert(1, "one");
    ///     })
    ///     .spawn(&mut world);
    /// ```
    pub fn builder() -> MutableBTreeMapBuilder<K, V> {
        MutableBTreeMapBuilder::new()
    }

    fn entity(&self) -> Entity {
        self.owner.lifetime.entity
    }

    /// Provides read-only access to the underlying [`BTreeMap`] via either a `&World` or a
    /// `&Query<MutableBTreeMapData<K, V>>`.
    pub fn read<'s>(
        &self,
        mutable_btree_map_data_reader: impl ReadMutableBTreeMapData<'s, K, V>,
    ) -> MutableBTreeMapReadGuard<'s, K, V>
    where
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        MutableBTreeMapReadGuard {
            guard: mutable_btree_map_data_reader.read(self.entity()),
        }
    }

    /// Provides write access to the underlying [`BTreeMap`] via either a `&mut World` or a
    /// `&mut Query<&mut MutableBTreeMapData<K, V>>`.
    pub fn write<'w>(
        &self,
        mutable_btree_map_data_writer: impl WriteMutableBTreeMapData<'w, K, V>,
    ) -> MutableBTreeMapWriteGuard<'w, K, V>
    where
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        MutableBTreeMapWriteGuard {
            guard: mutable_btree_map_data_writer.write(self.entity()),
        }
    }

    /// Returns a [`Source`] signal from this [`MutableBTreeMap`].
    pub fn signal_map(&self) -> Source<K, V>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        Source {
            materializer: self.owner.source.materializer(),
        }
    }

    /// Returns a [`SignalVec`] which outputs this [`MutableBTreeMap`]'s [`Key`](SignalMap::Key)s in
    /// sorted order.
    pub fn signal_vec_keys(&self) -> SignalVecKeys<K>
    where
        K: Ord + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let upstream = self.signal_map();
        let materializer = crate::VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(upstream).into_signal_map_materializer())?;
            registrar.replayable_signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<K, V>>>, mut keys: Local<Vec<K>>| {
                    let mut out_diffs = Vec::new();
                    for diff in diffs {
                        match diff {
                            MapDiff::Replace { entries } => {
                                *keys = entries.into_iter().map(|(key, _)| key).collect();
                                out_diffs.push(VecDiff::Replace { values: keys.clone() });
                            }
                            MapDiff::Insert { key, .. } => {
                                if let Err(index) = keys.binary_search(&key) {
                                    keys.insert(index, key.clone());
                                    out_diffs.push(VecDiff::InsertAt { index, value: key });
                                }
                            }
                            MapDiff::Update { .. } => {}
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
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        SignalVecKeys {
            inner: SignalVecExt::cache(SignalVecKeysTerminal { materializer }),
        }
    }

    /// Returns a [`SignalVec`] which outputs this [`MutableBTreeMap`]'s `(key, value)`s in sorted
    /// order.
    pub fn signal_vec_entries(&self) -> SignalVecEntries<K, V>
    where
        K: Ord + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let upstream = self.signal_map();
        let materializer = crate::VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(upstream).into_signal_map_materializer())?;
            registrar.replayable_signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<K, V>>>, mut keys: Local<Vec<K>>| {
                    let mut out_diffs = Vec::new();
                    for diff in diffs {
                        match diff {
                            MapDiff::Replace { entries } => {
                                *keys = entries.iter().map(|(key, _)| key.clone()).collect();
                                out_diffs.push(VecDiff::Replace { values: entries });
                            }
                            MapDiff::Insert { key, value } => match keys.binary_search(&key) {
                                Err(index) => {
                                    keys.insert(index, key.clone());
                                    out_diffs.push(VecDiff::InsertAt {
                                        index,
                                        value: (key, value),
                                    });
                                }
                                Ok(index) => {
                                    out_diffs.push(VecDiff::UpdateAt {
                                        index,
                                        value: (key, value),
                                    });
                                }
                            },
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
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        SignalVecEntries {
            inner: SignalVecExt::cache(SignalVecEntriesTerminal { materializer }),
        }
    }
}

impl<K, V> From<&mut World> for MutableBTreeMap<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn from(world: &mut World) -> Self {
        spawn_mutable_btree_map(world, BTreeMap::new())
    }
}

impl<K, V> FromWorld for MutableBTreeMap<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn from_world(world: &mut World) -> Self {
        world.into()
    }
}

/// Builder for constructing a [`MutableBTreeMap`].
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let map = MutableBTreeMap::builder()
///     .values([(1, "one"), (2, "two")])
///     .spawn(&mut world);
/// ```
pub struct MutableBTreeMapBuilder<K, V>(BTreeMap<K, V>);

impl<K, V> Default for MutableBTreeMapBuilder<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> MutableBTreeMapBuilder<K, V> {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }
}

impl<K: Ord, V> MutableBTreeMapBuilder<K, V> {
    /// Sets the initial values of the [`MutableBTreeMap`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let map = MutableBTreeMap::builder()
    ///     .values([(1, "one")])
    ///     .spawn(&mut world);
    /// ```
    pub fn values(mut self, values: impl Into<BTreeMap<K, V>>) -> Self {
        self.0 = values.into();
        self
    }

    /// Mutably access the inner [`BTreeMap`] for custom initialization.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let map = MutableBTreeMap::builder()
    ///     .with_values(|m| {
    ///         m.insert(1, "one");
    ///         m.insert(2, "two");
    ///     })
    ///     .spawn(&mut world);
    /// ```
    pub fn with_values(mut self, f: impl FnOnce(&mut BTreeMap<K, V>)) -> Self {
        f(&mut self.0);
        self
    }
}

impl<K, V, A> From<A> for MutableBTreeMapBuilder<K, V>
where
    BTreeMap<K, V>: From<A>,
{
    fn from(value: A) -> Self {
        Self(value.into())
    }
}

impl<K, V> MutableBTreeMapBuilder<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Spawns a [`MutableBTreeMap`] using a `&mut World`.
    pub fn spawn(self, world: &mut World) -> MutableBTreeMap<K, V> {
        spawn_mutable_btree_map(world, self.0)
    }

    /// Spawns a [`MutableBTreeMap`] using a `&mut Commands`.
    pub fn spawnc(self, commands: &mut Commands) -> MutableBTreeMap<K, V> {
        spawn_mutable_btree_map_commands(commands, self.0)
    }
}

impl<K, V> From<&mut Commands<'_, '_>> for MutableBTreeMap<K, V>
where
    K: Ord + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn from(commands: &mut Commands) -> Self {
        spawn_mutable_btree_map_commands(commands, BTreeMap::new())
    }
}

/// Specifies read accessors for [`MutableBTreeMap`]s.
pub trait ReadMutableBTreeMapData<'s, K, V>
where
    K: Send + Sync,
    V: Send + Sync,
{
    #[allow(missing_docs)]
    fn read(self, entity: Entity) -> &'s MutableBTreeMapData<K, V>;
}

impl<'s, K, V> ReadMutableBTreeMapData<'s, K, V> for &'s Query<'_, 's, &MutableBTreeMapData<K, V>>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn read(self, entity: Entity) -> &'s MutableBTreeMapData<K, V> {
        self.get(entity).unwrap()
    }
}

impl<'s, K, V> ReadMutableBTreeMapData<'s, K, V> for &'s World
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn read(self, entity: Entity) -> &'s MutableBTreeMapData<K, V> {
        self.get(entity).unwrap()
    }
}

/// Specifies write accessors for [`MutableBTreeMap`]s.
pub trait WriteMutableBTreeMapData<'w, K, V>
where
    K: Send + Sync,
    V: Send + Sync,
{
    #[allow(missing_docs)]
    fn write(self, entity: Entity) -> Mut<'w, MutableBTreeMapData<K, V>>;
}

impl<'a, 'w, 's, K, V> WriteMutableBTreeMapData<'a, K, V> for &'a mut Query<'w, 's, &mut MutableBTreeMapData<K, V>>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn write(self, entity: Entity) -> Mut<'a, MutableBTreeMapData<K, V>> {
        self.get_mut(entity).unwrap()
    }
}

impl<'w, K, V> WriteMutableBTreeMapData<'w, K, V> for &'w mut World
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn write(self, entity: Entity) -> Mut<'w, MutableBTreeMapData<K, V>> {
        self.get_mut(entity).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JonmoPlugin, cleanup, graph::graph_snapshot, prelude::clone, signal_vec::SignalVecExt};
    use bevy::prelude::*;
    use bevy_ecs::world::CommandQueue;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MutableMapReplaySourceSnapshot {
        entity: Entity,
        exists: bool,
        current_len: Option<usize>,
        revision: Option<MapRevision>,
        pending_revisions: Vec<MapRevision>,
    }

    fn mutable_map_replay_source_snapshot<K, V>(world: &World, entity: Entity) -> MutableMapReplaySourceSnapshot
    where
        K: Ord + Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        let data = world.get::<MutableBTreeMapData<K, V>>(entity);
        MutableMapReplaySourceSnapshot {
            entity,
            exists: world.get_entity(entity).is_ok(),
            current_len: data.map(|data| data.map.len()),
            revision: data.map(|data| data.revision),
            pending_revisions: data
                .map(|data| data.pending_diffs.iter().map(|diff| diff.revision).collect())
                .unwrap_or_default(),
        }
    }

    #[test]
    fn static_map_terminals_are_verified_replayable_and_normalize_cache_identity() {
        fn assert_replayable_map<T: ReplayableSignalMap>(_signal: &T) {}
        fn assert_replayable_signal<T: crate::signal::ReplayableSignal>(_signal: &T) {}
        fn assert_replayable_vec<T: crate::signal_vec::ReplayableSignalVec>(_signal: &T) {}

        let mut app = create_test_app();
        let source = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());

        let map = source.signal_map();
        assert_replayable_map(&map);
        let map_claim = map.register(app.world_mut());
        let entities_before_cache = app.world().entities().len();
        let cached_map_claim = source.signal_map().cache().register(app.world_mut());
        assert_eq!(map_claim.entity(), cached_map_claim.entity());
        assert_eq!(app.world().entities().len(), entities_before_cache);

        let mapped = source.signal_map().map_value(|In(value): In<i32>| value * 2).cache();
        assert_replayable_map(&mapped);
        let mapped = mapped.broadcast();
        assert_replayable_map(&mapped);
        let mapped_claim = mapped.clone().register(app.world_mut());
        let entities_before_cache = app.world().entities().len();
        let cached_mapped_claim = mapped.cache().register(app.world_mut());
        assert_eq!(mapped_claim.entity(), cached_mapped_claim.entity());
        assert_eq!(app.world().entities().len(), entities_before_cache);

        let key = source.signal_map().key(1);
        assert_replayable_signal(&key);
        let key = key.broadcast();
        assert_replayable_signal(&key);
        let key_claim = key.clone().register(app.world_mut());
        let entities_before_cache = app.world().entities().len();
        let cached_key_claim = key.cache().register(app.world_mut());
        assert_eq!(key_claim.entity(), cached_key_claim.entity());
        assert_eq!(app.world().entities().len(), entities_before_cache);

        let keys = source.signal_vec_keys();
        assert_replayable_vec(&keys);
        let keys = keys.broadcast();
        assert_replayable_vec(&keys);
        let keys_claim = keys.clone().register(app.world_mut());
        let entities_before_cache = app.world().entities().len();
        let cached_keys_claim = keys.cache().register(app.world_mut());
        assert_eq!(keys_claim.entity(), cached_keys_claim.entity());
        assert_eq!(app.world().entities().len(), entities_before_cache);

        let entries = source.signal_vec_entries();
        assert_replayable_vec(&entries);
        let entries = entries.broadcast();
        assert_replayable_vec(&entries);
        let entries_claim = entries.clone().register(app.world_mut());
        let entities_before_cache = app.world().entities().len();
        let cached_entries_claim = entries.cache().register(app.world_mut());
        assert_eq!(entries_claim.entity(), cached_entries_claim.entity());
        assert_eq!(app.world().entities().len(), entities_before_cache);

        for claim in [
            map_claim,
            cached_map_claim,
            mapped_claim,
            cached_mapped_claim,
            key_claim,
            cached_key_claim,
            keys_claim,
            cached_keys_claim,
            entries_claim,
            cached_entries_claim,
        ] {
            claim.release(app.world_mut());
        }
    }

    #[test]
    fn mutable_map_replay_snapshot_registration_ownership_observability() {
        let mut world = World::new();
        let source = MutableBTreeMap::builder().values([(1, 10), (2, 20)]).spawn(&mut world);
        let entity = source.entity();

        assert_eq!(
            mutable_map_replay_source_snapshot::<i32, i32>(&world, entity),
            MutableMapReplaySourceSnapshot {
                entity,
                exists: true,
                current_len: Some(2),
                revision: Some(0),
                pending_revisions: Vec::new(),
            }
        );

        source.write(&mut world).insert(3, 30);
        source.write(&mut world).remove(&1);
        let first = mutable_map_replay_source_snapshot::<i32, i32>(&world, entity);
        let second = mutable_map_replay_source_snapshot::<i32, i32>(&world, entity);
        assert_eq!(
            first, second,
            "unchanged typed replay state must snapshot deterministically"
        );
        assert_eq!(
            first,
            MutableMapReplaySourceSnapshot {
                entity,
                exists: true,
                current_len: Some(2),
                revision: Some(2),
                pending_revisions: vec![1, 2],
            },
            "pending replay history must preserve source revision order"
        );
    }

    // Helper resource to capture the output diffs from a SignalMap for assertions.
    #[derive(Resource, Default, Debug)]
    struct SignalMapOutput<K, V>(Vec<MapDiff<K, V>>)
    where
        K: Send + Sync + 'static + Clone + fmt::Debug,
        V: Send + Sync + 'static + Clone + fmt::Debug;

    // Helper system that captures incoming diffs and stores them in the
    // SignalMapOutput resource.
    fn capture_map_output<K, V>(In(diffs): In<Vec<MapDiff<K, V>>>, mut output: ResMut<SignalMapOutput<K, V>>)
    where
        K: Send + Sync + 'static + Clone + fmt::Debug,
        V: Send + Sync + 'static + Clone + fmt::Debug,
    {
        output.0.extend(diffs);
    }

    // Helper function to retrieve and clear the captured diffs from the world, making
    // it easy to assert against the output of a single frame's update.
    fn get_and_clear_map_output<K, V>(world: &mut World) -> Vec<MapDiff<K, V>>
    where
        K: Send + Sync + 'static + Clone + fmt::Debug,
        V: Send + Sync + 'static + Clone + fmt::Debug,
    {
        world
            .get_resource_mut::<SignalMapOutput<K, V>>()
            .map(|mut res| core::mem::take(&mut res.0))
            .unwrap_or_default()
    }

    fn apply_map_diffs<K: Ord, V>(state: &mut BTreeMap<K, V>, diffs: impl IntoIterator<Item = MapDiff<K, V>>) {
        for diff in diffs {
            match diff {
                MapDiff::Replace { entries } => *state = entries.into_iter().collect(),
                MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                    state.insert(key, value);
                }
                MapDiff::Remove { key } => {
                    state.remove(&key);
                }
                MapDiff::Clear => state.clear(),
            }
        }
    }

    #[derive(Resource, Default, Debug)]
    struct CollectionCorrectnessMapCaptureA(Vec<MapDiff<i32, i32>>);

    #[derive(Resource, Default, Debug)]
    struct CollectionCorrectnessMapCaptureB(Vec<MapDiff<i32, i32>>);

    // Helper to create a minimal Bevy App with the JonmoPlugin for testing.
    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        app
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DynamicRegistrationObservation {
        count: Option<usize>,
        active_downstreams: Vec<Entity>,
    }

    fn dynamic_registration_observation(world: &mut World, entity: Entity) -> DynamicRegistrationObservation {
        graph_snapshot(world)
            .node(entity)
            .map(|node| DynamicRegistrationObservation {
                count: node.claim_count,
                active_downstreams: node.active_downstreams.clone(),
            })
            .unwrap_or(DynamicRegistrationObservation {
                count: None,
                active_downstreams: Vec::new(),
            })
    }

    #[test]
    fn map_value_signal_final_dynamic_release_registration_ownership_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();

        let inner_a = signal::from_system(|In(_)| 10).dedupe().broadcast();
        let inner_b = signal::from_system(|In(_)| 20).dedupe().broadcast();
        let inner_a_handle = inner_a.clone().register(app.world_mut());
        let inner_a_entity = inner_a_handle.entity().entity();
        inner_a_handle.release(app.world_mut());
        let inner_b_handle = inner_b.clone().register(app.world_mut());
        let inner_b_entity = inner_b_handle.entity().entity();
        inner_b_handle.release(app.world_mut());

        let selectors = MutableBTreeMap::builder().values([(1, false)]).spawn(app.world_mut());
        let mapped = selectors
            .signal_map()
            .map_value_signal(clone!((inner_a, inner_b) move |In(select_b)| {
                if select_b {
                    inner_b.clone().boxed_replayable()
                } else {
                    inner_a.clone().boxed_replayable()
                }
            }));
        let outer_handle = mapped
            .for_each(capture_map_output::<i32, i32>)
            .register(app.world_mut());

        app.update();
        get_and_clear_map_output::<i32, i32>(app.world_mut());
        let initial = (
            dynamic_registration_observation(app.world_mut(), inner_a_entity),
            dynamic_registration_observation(app.world_mut(), inner_b_entity),
        );
        assert_eq!(initial.0.active_downstreams.len(), 1);
        let inner_a_route = initial.0.active_downstreams.clone();

        selectors.write(app.world_mut()).insert(1, true);
        app.update();
        get_and_clear_map_output::<i32, i32>(app.world_mut());
        let replaced = (
            dynamic_registration_observation(app.world_mut(), inner_a_entity),
            dynamic_registration_observation(app.world_mut(), inner_b_entity),
        );
        assert_eq!(replaced.1.active_downstreams.len(), 1);
        let inner_b_route = replaced.1.active_downstreams.clone();

        outer_handle.release(app.world_mut());
        let cleaned = (
            dynamic_registration_observation(app.world_mut(), inner_a_entity),
            dynamic_registration_observation(app.world_mut(), inner_b_entity),
        );
        app.update();
        let output_after_final_cleanup = get_and_clear_map_output::<i32, i32>(app.world_mut());

        assert_eq!(
            (initial, replaced, cleaned, output_after_final_cleanup),
            (
                (
                    DynamicRegistrationObservation {
                        count: Some(2),
                        active_downstreams: inner_a_route,
                    },
                    DynamicRegistrationObservation {
                        count: Some(1),
                        active_downstreams: Vec::new(),
                    },
                ),
                (
                    DynamicRegistrationObservation {
                        count: Some(1),
                        active_downstreams: Vec::new(),
                    },
                    DynamicRegistrationObservation {
                        count: Some(2),
                        active_downstreams: inner_b_route,
                    },
                ),
                (
                    DynamicRegistrationObservation {
                        count: Some(1),
                        active_downstreams: Vec::new(),
                    },
                    DynamicRegistrationObservation {
                        count: Some(1),
                        active_downstreams: Vec::new(),
                    },
                ),
                Vec::new(),
            ),
            "map_value_signal must release identity-probe registrations, superseded processors, and the final active processor; final cleanup should match the state before the dynamic subscription existed"
        );
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
    fn mutable_btree_map_world_cleanup_owner_clones_emit_one_final_root() {
        let mut world = World::new();
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let neighbor: MutableBTreeMap<i32, i32> = (&mut world).into();
        let source_entity = source.entity();
        let neighbor_entity = neighbor.entity();
        let clone = source.clone();
        assert!(Arc::ptr_eq(&source.owner, &clone.owner));

        drop(clone);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        assert!(world.get_entity(source_entity).is_ok());

        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(source_entity).is_err());
        assert!(world.get_entity(neighbor_entity).is_ok());
        assert_eq!(cleanup::queued_message_count(&world), 0);

        drop(neighbor);
        cleanup::drain_world_cleanup(&mut world);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_builder_spawn_preserves_values_and_entity() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let source = MutableBTreeMap::builder().values([(1, 10), (2, 20)]).spawn(&mut world);
        let entity = source.entity();
        assert_eq!(source.read(&world).get(&1), Some(&10));
        source.write(&mut world).insert(3, 30);
        assert_eq!(source.read(&world).get(&3), Some(&30));

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_from_world_preserves_exact_entity() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let source = <MutableBTreeMap<i32, i32> as FromWorld>::from_world(&mut world);
        let entity = source.entity();
        assert!(source.read(&world).is_empty());

        drop(source);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_builder_commands_apply_before_drop() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source = {
            let mut commands = Commands::new(&mut queue, &world);
            MutableBTreeMap::builder()
                .values([(4, 40), (5, 50)])
                .spawnc(&mut commands)
        };
        let entity = source.entity();
        assert_eq!(source.owner.lifetime.cleanup.inbox_identity(), None);

        queue.apply(&mut world);
        assert_eq!(
            source.owner.lifetime.cleanup.inbox_identity(),
            cleanup::cleanup_inbox_identity(&world)
        );
        assert_eq!(source.read(&world).get(&4), Some(&40));
        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_builder_commands_drop_before_apply() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source = {
            let mut commands = Commands::new(&mut queue, &world);
            MutableBTreeMap::builder().values([(8, 80)]).spawnc(&mut commands)
        };
        let entity = source.entity();

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        queue.apply(&mut world);
        assert!(world.get::<MutableBTreeMapData<i32, i32>>(entity).is_some());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_from_commands_apply_before_drop() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source: MutableBTreeMap<i32, i32> = {
            let mut commands = Commands::new(&mut queue, &world);
            (&mut commands).into()
        };
        let entity = source.entity();

        queue.apply(&mut world);
        assert_eq!(
            source.owner.lifetime.cleanup.inbox_identity(),
            cleanup::cleanup_inbox_identity(&world)
        );
        source.write(&mut world).insert(3, 30);
        assert_eq!(source.read(&world).get(&3), Some(&30));
        drop(source);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_from_commands_drop_before_apply() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source: MutableBTreeMap<i32, i32> = {
            let mut commands = Commands::new(&mut queue, &world);
            (&mut commands).into()
        };
        let entity = source.entity();

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        queue.apply(&mut world);
        assert!(world.get::<MutableBTreeMapData<i32, i32>>(entity).is_some());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_discarded_commands_are_safe() {
        let reservation_world = World::new();
        let mut queue = CommandQueue::default();
        let source: MutableBTreeMap<i32, i32> = {
            let mut commands = Commands::new(&mut queue, &reservation_world);
            (&mut commands).into()
        };
        drop(source);
        drop(queue);
        drop(reservation_world);

        let mut unrelated_world = World::new();
        let unrelated: MutableBTreeMap<i32, i32> = (&mut unrelated_world).into();
        let unrelated_entity = unrelated.entity();
        cleanup::drain_world_cleanup(&mut unrelated_world);
        assert!(unrelated_world.get_entity(unrelated_entity).is_ok());
        drop(unrelated);
        cleanup::drain_world_cleanup(&mut unrelated_world);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_worlds_with_colliding_entities_are_isolated() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let source_a: MutableBTreeMap<i32, i32> = (&mut world_a).into();
        let source_b: MutableBTreeMap<i32, i32> = (&mut world_b).into();
        let entity_a = source_a.entity();
        let entity_b = source_b.entity();
        assert_eq!(entity_a, entity_b);

        drop(source_a);
        cleanup::drain_world_cleanup(&mut world_b);
        assert!(world_b.get_entity(entity_b).is_ok());
        assert!(world_a.get_entity(entity_a).is_ok());

        cleanup::drain_world_cleanup(&mut world_a);
        assert!(world_a.get_entity(entity_a).is_err());
        assert!(world_b.get_entity(entity_b).is_ok());
        drop(source_b);
        cleanup::drain_world_cleanup(&mut world_b);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_plugin_b_preserves_app_a_pending_root() {
        let mut app_a = App::new();
        app_a.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        let source: MutableBTreeMap<i32, i32> = app_a.world_mut().into();
        let entity = source.entity();
        drop(source);
        assert_eq!(cleanup::queued_message_count(app_a.world()), 1);

        let mut app_b = App::new();
        app_b.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        assert_eq!(cleanup::queued_message_count(app_a.world()), 1);
        app_b.update();
        assert!(app_a.world().get_entity(entity).is_ok());

        app_a.update();
        assert!(app_a.world().get_entity(entity).is_err());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_stale_generation_does_not_remove_replacement() {
        let mut world = World::new();
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let stale = source.entity();
        world.entity_mut(stale).despawn();
        let replacement = world.spawn_empty().id();
        assert_eq!(stale.index(), replacement.index());
        assert_ne!(stale.generation(), replacement.generation());

        drop(source);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(replacement).is_ok());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_unregistered_broadcaster_is_one_root() {
        let mut world = World::new();
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let entity = source.entity();
        world.entity_mut(entity).despawn();
        assert_eq!(cleanup::queued_message_count(&world), 0);

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert_eq!(cleanup::queued_message_count(&world), 0);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_registered_broadcaster_is_reaped_causally() {
        let mut world = World::new();
        world.insert_resource(crate::graph::SignalGraphState::default());
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let data_entity = source.entity();
        let broadcaster = source.signal_map().register(&mut world);
        let broadcaster_entity = broadcaster.entity().entity();
        assert_eq!(cleanup::queued_message_count(&world), 0);

        broadcaster.release(&mut world);
        assert!(world.get_entity(broadcaster_entity).is_ok());
        assert_eq!(cleanup::queued_message_count(&world), 0);

        let registered_again = source.signal_map().register(&mut world);
        assert_eq!(registered_again.entity().entity(), broadcaster_entity);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        registered_again.release(&mut world);
        assert!(world.get_entity(broadcaster_entity).is_ok());
        assert_eq!(cleanup::queued_message_count(&world), 0);

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(broadcaster_entity).is_err());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(data_entity).is_err());
        assert_eq!(cleanup::queued_message_count(&world), 0);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_missing_data_still_reaps_broadcaster() {
        let mut world = World::new();
        world.insert_resource(crate::graph::SignalGraphState::default());
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let data_entity = source.entity();
        let broadcaster = source.signal_map().register(&mut world);
        let broadcaster_entity = broadcaster.entity().entity();
        assert_eq!(cleanup::queued_message_count(&world), 0);

        broadcaster.release(&mut world);
        assert!(world.get_entity(broadcaster_entity).is_ok());
        assert_eq!(cleanup::queued_message_count(&world), 0);

        world.entity_mut(data_entity).despawn();
        assert_eq!(cleanup::queued_message_count(&world), 0);
        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(broadcaster_entity).is_err());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert_eq!(cleanup::queued_message_count(&world), 0);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_descriptions_have_no_owner_cycle() {
        let mut world = World::new();
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let entity = source.entity();
        let lifetime = Arc::downgrade(&source.owner.lifetime);
        let map_signal = source.signal_map();
        let keys_signal = source.signal_vec_keys();
        let entries_signal = source.signal_vec_entries();

        drop(source);
        drop(map_signal);
        drop(keys_signal);
        assert!(lifetime.upgrade().is_some());
        assert_eq!(cleanup::queued_message_count(&world), 0);
        drop(entries_signal);
        assert!(lifetime.upgrade().is_none());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
    }

    #[test]
    fn mutable_btree_map_world_cleanup_registered_source_releases_owner() {
        let mut world = World::new();
        world.insert_resource(crate::graph::SignalGraphState::default());
        let source: MutableBTreeMap<i32, i32> = (&mut world).into();
        let entity = source.entity();
        let lifetime = Arc::downgrade(&source.owner.lifetime);
        let handle = source
            .signal_map()
            .for_each(|In(_): In<Vec<MapDiff<i32, i32>>>| ())
            .register(&mut world);
        cleanup::drain_world_cleanup(&mut world);

        drop(source);
        assert!(lifetime.upgrade().is_some());
        handle.release(&mut world);
        assert!(lifetime.upgrade().is_none());
        assert_eq!(cleanup::queued_message_count(&world), 2);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert_eq!(cleanup::queued_message_count(&world), 0);
    }

    #[test]
    fn mutable_btree_map_world_cleanup_drop_after_world_destruction_is_safe() {
        let source = {
            let mut world = World::new();
            let source: MutableBTreeMap<i32, i32> = (&mut world).into();
            drop(world);
            source
        };

        drop(source);
    }

    #[test]
    fn test_for_each() {
        {
            let mut app = create_test_app();

            // The output of our `for_each` system will be the full, reconstructed BTreeMap.
            app.init_resource::<SignalOutput<BTreeMap<u32, String>>>();
            let source_map = (MutableBTreeMap::builder().values([(1, "one".to_string()), (2, "two".to_string())]))
                .spawn(app.world_mut());

            // This system reconstructs the state of the map by applying the diffs it
            // receives. It then outputs the complete, current state of the map. This allows
            // us to verify that `for_each` is receiving the diffs correctly.
            let reconstructor_system =
                |In(diffs): In<Vec<MapDiff<u32, String>>>, mut state: Local<BTreeMap<u32, String>>| {
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

                    // Output the current reconstructed state
                    state.clone()
                };
            let handle = source_map
                .signal_map()
                .for_each(reconstructor_system)
                .map(capture_output::<BTreeMap<u32, String>>)
                .register(app.world_mut());

            // Test 1: Initial State. The initial `Replace` diff should be received and
            // processed.
            app.update();
            let expected_initial_state: BTreeMap<_, _> =
                [(1, "one".to_string()), (2, "two".to_string())].into_iter().collect();
            assert_eq!(
                get_output::<BTreeMap<u32, String>>(app.world_mut()),
                Some(expected_initial_state.clone()),
                "Initial state was not reconstructed correctly"
            );

            // Test 2: Batched Mutations. We'll perform multiple operations before the next
            // update to test batch processing.
            {
                let mut writer = source_map.write(app.world_mut());

                // Insert
                writer.insert(3, "three".to_string());

                // Update
                writer.insert(1, "one_v2".to_string());

                // Remove
                writer.remove(&2);
            }
            app.update();
            let expected_batched_state: BTreeMap<_, _> = [(1, "one_v2".to_string()), (3, "three".to_string())]
                .into_iter()
                .collect();
            assert_eq!(
                get_output::<BTreeMap<u32, String>>(app.world_mut()),
                Some(expected_batched_state.clone()),
                "State after batched mutations was not reconstructed correctly"
            );

            // Test 3: Clear. The `Clear` diff should result in an empty map.
            source_map.write(app.world_mut()).clear();
            app.update();
            let expected_cleared_state: BTreeMap<u32, String> = BTreeMap::new();
            assert_eq!(
                get_output::<BTreeMap<u32, String>>(app.world_mut()),
                Some(expected_cleared_state.clone()),
                "State after Clear was not reconstructed correctly"
            );
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_map_value() {
        {
            let mut app = create_test_app();

            // The output map will have keys of `u32` and values of `String`.
            app.init_resource::<SignalMapOutput<u32, String>>();

            // The source map contains integer values.
            let source_map = (MutableBTreeMap::builder().values([(1, 10), (2, 20)])).spawn(app.world_mut());

            // The mapping function transforms an i32 value into a String.
            let mapping_system = |In(val): In<i32>| format!("Val:{val}");

            // Apply `map_value` to create the derived signal.
            let mapped_signal = source_map.signal_map().map_value(mapping_system);
            let handle = mapped_signal
                .for_each(capture_map_output::<u32, String>)
                .register(app.world_mut());

            // Test 1: Initial State (Replace). The first update should replay the initial
            // state with mapped values.
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

            // Test 2: Insert. A new entry in the source should result in an Insert diff with
            // a mapped value.
            source_map.write(app.world_mut()).insert(3, 30);
            app.update();
            let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Insert should produce one diff");
            assert_eq!(
                diffs[0],
                MapDiff::Insert {
                    key: 3,
                    value: "Val:30".to_string(),
                },
                "Insert diff has incorrect mapped value"
            );

            // Test 3: Update. Updating an existing entry should result in an Update diff with
            // a mapped value. `insert` on existing key is an update
            source_map.write(app.world_mut()).insert(1, 15);
            app.update();
            let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Update should produce one diff");
            assert_eq!(
                diffs[0],
                MapDiff::Update {
                    key: 1,
                    value: "Val:15".to_string(),
                },
                "Update diff has incorrect mapped value"
            );

            // Test 4: Remove. Removing an entry should result in a Remove diff, which has no
            // value to map.
            source_map.write(app.world_mut()).remove(&2);
            app.update();
            let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Remove should produce one diff");
            assert_eq!(
                diffs[0],
                MapDiff::Remove { key: 2 },
                "Remove diff was not propagated correctly"
            );

            // Test 5: Clear. Clearing the source map should result in a Clear diff.
            source_map.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_map_output::<u32, String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Clear should produce one diff");
            assert_eq!(diffs[0], MapDiff::Clear, "Clear diff was not propagated correctly");
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn map_value_signal_replace_empty_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<u32, Name>>();

        let retained_entity = app.world_mut().spawn(Name::new("Alice")).id();
        let other_entity = app.world_mut().spawn(Name::new("Bob")).id();
        let entity_map = MutableBTreeMap::builder()
            .values([(1, retained_entity), (2, other_entity)])
            .spawn(app.world_mut());
        let handle = entity_map
            .signal_map()
            .map_value_signal(|In(entity): In<Entity>| signal::from_component::<Name>(entity).dedupe())
            .for_each(capture_map_output::<u32, Name>)
            .register(app.world_mut());
        let mut downstream = BTreeMap::new();

        app.update();
        let initial_diffs = app.world().resource::<SignalMapOutput<u32, Name>>().0.clone();
        apply_map_diffs(&mut downstream, initial_diffs);
        assert_eq!(
            downstream,
            BTreeMap::from([(1, Name::new("Alice")), (2, Name::new("Bob"))]),
            "initial mapped collection did not match the source components"
        );
        app.world_mut().resource_mut::<SignalMapOutput<u32, Name>>().0.clear();

        entity_map
            .write(app.world_mut())
            .replace(BTreeMap::<u32, Entity>::new());
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        apply_map_diffs(&mut downstream, diffs.clone());
        let expected_diffs = vec![MapDiff::Replace { entries: vec![] }];
        let replacement_correct = diffs == expected_diffs && downstream.is_empty();

        *app.world_mut().get_mut::<Name>(retained_entity).unwrap() = Name::new("Alicia");
        app.update();
        let boundary_diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        let boundary_correct = boundary_diffs.is_empty();

        handle.release(app.world_mut());
        assert!(
            replacement_correct && boundary_correct,
            "empty source replacement was not represented exactly downstream; expected diffs: {expected_diffs:?}; actual diffs: {diffs:?}; actual state: {downstream:?}; old-inner boundary diffs: {boundary_diffs:?}"
        );
    }

    #[test]
    fn test_map_value_signal_populated_empty_populated_replace() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<u32, Name>>();

        let old_alice = app.world_mut().spawn(Name::new("Alice")).id();
        let old_bob = app.world_mut().spawn(Name::new("Bob")).id();
        let source = MutableBTreeMap::builder()
            .values([(1, old_alice), (2, old_bob)])
            .spawn(app.world_mut());
        let handle = source
            .signal_map()
            .map_value_signal(|In(entity): In<Entity>| signal::from_component::<Name>(entity).dedupe())
            .for_each(capture_map_output::<u32, Name>)
            .register(app.world_mut());
        let mut downstream = BTreeMap::new();

        app.update();
        let initial_diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        let expected_initial_diffs = vec![MapDiff::Replace {
            entries: vec![(1, Name::new("Alice")), (2, Name::new("Bob"))],
        }];
        apply_map_diffs(&mut downstream, initial_diffs.iter().cloned());
        assert_eq!(
            initial_diffs, expected_initial_diffs,
            "initial populated replacement emitted an unexpected stream"
        );
        assert_eq!(
            downstream,
            BTreeMap::from([(1, Name::new("Alice")), (2, Name::new("Bob"))]),
            "initial populated state was not reconstructed from emitted diffs"
        );

        source.write(app.world_mut()).replace(BTreeMap::<u32, Entity>::new());
        app.update();
        let empty_diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        let expected_empty_diffs = vec![MapDiff::Replace { entries: vec![] }];
        apply_map_diffs(&mut downstream, empty_diffs.iter().cloned());
        assert_eq!(
            empty_diffs, expected_empty_diffs,
            "empty replacement must emit exactly one authoritative Replace"
        );
        assert!(downstream.is_empty(), "empty replacement left stale downstream entries");

        let new_carol = app.world_mut().spawn(Name::new("Carol")).id();
        let new_dave = app.world_mut().spawn(Name::new("Dave")).id();
        source
            .write(app.world_mut())
            .replace(BTreeMap::from([(2, new_carol), (3, new_dave)]));
        app.update();
        let repopulated_diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        let expected_repopulated_diffs = vec![MapDiff::Replace {
            entries: vec![(2, Name::new("Carol")), (3, Name::new("Dave"))],
        }];
        apply_map_diffs(&mut downstream, repopulated_diffs.iter().cloned());
        let expected_repopulated_state = BTreeMap::from([(2, Name::new("Carol")), (3, Name::new("Dave"))]);
        let correct = repopulated_diffs == expected_repopulated_diffs && downstream == expected_repopulated_state;

        handle.release(app.world_mut());
        assert!(
            correct,
            "repopulation after empty replacement diverged: expected_diffs={expected_repopulated_diffs:?}, actual_diffs={repopulated_diffs:?}, expected_state={expected_repopulated_state:?}, actual_state={downstream:?}"
        );
    }

    #[test]
    fn test_map_value_signal() {
        {
            let mut app = create_test_app();

            // The output map will have keys of `u32` and values of `Name` components.
            app.init_resource::<SignalMapOutput<u32, Name>>();

            // Setup: Create entities with `Name` components that our signals will track.
            let entity_a = app.world_mut().spawn(Name::new("Alice")).id();
            let entity_b = app.world_mut().spawn(Name::new("Bob")).id();

            // The source map contains entities. The goal is to create a derived map that
            // contains the _names_ of these entities.
            let entity_map = (MutableBTreeMap::builder().values([(1, entity_a), (2, entity_b)])).spawn(app.world_mut());

            // This "factory" system takes an entity and creates a signal that tracks its
            // `Name`.
            let factory_system = |In(entity): In<Entity>| signal::from_component::<Name>(entity).dedupe();

            // Apply `map_value_signal` to transform the SignalMap<u32, Entity> into a
            // SignalMap<u32, Name>.
            let name_map_signal = entity_map.signal_map().map_value_signal(factory_system);
            let handle = name_map_signal
                .for_each(capture_map_output::<u32, Name>)
                .register(app.world_mut());

            // Test 1: Initial State. The first update should replay the initial state of the
            // map.
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

            // Test 2: Inner Signal Update. Change a component on a tracked entity. This
            // should trigger an Update diff.
            *app.world_mut().get_mut::<Name>(entity_a).unwrap() = Name::new("Alicia");
            app.update();
            let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Name change should produce one Update diff");
            assert_eq!(
                diffs[0],
                MapDiff::Update {
                    key: 1,
                    value: Name::new("Alicia"),
                },
                "Update diff is incorrect"
            );

            // Test 3: No change should produce no diff.
            app.update();
            let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
            assert!(diffs.is_empty(), "No change should produce no diffs");

            // Test 4: Source Map Insertion. Add a new entity to the source map. This should
            // trigger an Insert diff.
            let entity_c = app.world_mut().spawn(Name::new("Charlie")).id();
            entity_map.write(app.world_mut()).insert(3, entity_c);
            app.update();
            let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Insert should produce one Insert diff");
            assert_eq!(
                diffs[0],
                MapDiff::Insert {
                    key: 3,
                    value: Name::new("Charlie"),
                },
                "Insert diff is incorrect"
            );

            // Test 5: Source Map Removal. Remove an entity from the source map. This should
            // trigger a Remove diff.
            entity_map.write(app.world_mut()).remove(&2);
            app.update();
            let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Remove should produce one Remove diff");
            assert_eq!(diffs[0], MapDiff::Remove { key: 2 }, "Remove diff is incorrect");

            // Test 6: Source Map Update (switching the underlying signal). Update a key to
            // point to a new entity. This must tear down the old signal and create a new one,
            // resulting in an Update diff with the new value.
            let entity_d = app.world_mut().spawn(Name::new("David")).id();

            // `insert` acts as update here.
            entity_map.write(app.world_mut()).insert(1, entity_d);
            app.update();
            let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Source map update should produce one Update diff");
            assert_eq!(
                diffs[0],
                MapDiff::Update {
                    key: 1,
                    value: Name::new("David"),
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
                    value: Name::new("Dave"),
                },
                "Update on new entity is incorrect"
            );

            // Test 7: Source Map Clear. Clear the source map. This should trigger a Clear
            // diff.
            entity_map.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Clear should produce one Clear diff");
            assert_eq!(diffs[0], MapDiff::Clear, "Clear diff is incorrect");
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn map_value_signal_preserves_queued_mutable_map_diff_order() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<u32, i32>>();
        let source = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20)])
            .spawn(app.world_mut());
        let handle = source
            .signal_map()
            .map_value_signal(|In(value): In<i32>| signal::from_system(move |In(_)| value * 10).dedupe())
            .for_each(capture_map_output::<u32, i32>)
            .register(app.world_mut());

        app.update();
        let mut reconstructed = BTreeMap::new();
        apply_map_diffs(
            &mut reconstructed,
            get_and_clear_map_output::<u32, i32>(app.world_mut()),
        );

        {
            let mut source = source.write(app.world_mut());
            source.insert(3, 30);
            source.insert(3, 31);
            source.remove(&1);
            source.clear();
            source.insert(4, 40);
        }
        app.update();

        let diffs = get_and_clear_map_output::<u32, i32>(app.world_mut());
        assert_eq!(
            diffs,
            vec![
                MapDiff::Insert { key: 3, value: 300 },
                MapDiff::Update { key: 3, value: 310 },
                MapDiff::Remove { key: 1 },
                MapDiff::Clear,
                MapDiff::Insert { key: 4, value: 400 },
            ]
        );
        apply_map_diffs(&mut reconstructed, diffs);
        assert_eq!(reconstructed, BTreeMap::from([(4, 400)]));

        handle.release(app.world_mut());
    }

    // Helper resource to capture the output from a standard Signal for assertions.
    #[derive(Resource, Default, Debug)]
    struct SignalOutput<T>(Option<T>)
    where
        T: Send + Sync + 'static + Clone + fmt::Debug;

    // Helper system that captures incoming values and stores them in the SignalOutput
    // resource.
    fn capture_output<T>(In(value): In<T>, mut output: ResMut<SignalOutput<T>>)
    where
        T: Send + Sync + 'static + Clone + fmt::Debug,
    {
        output.0 = Some(value);
    }

    // Helper to retrieve the last captured value.
    fn get_output<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &mut World) -> Option<T> {
        world.get_resource::<SignalOutput<T>>().and_then(|res| res.0.clone())
    }

    // Helper to clear the output, useful for testing no-emission cases.
    fn clear_output<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &mut World) {
        if let Some(mut res) = world.get_resource_mut::<SignalOutput<T>>() {
            res.0 = None;
        }
    }

    #[test]
    fn test_key() {
        {
            let mut app = create_test_app();

            // The output is a Signal<Option`<String>`>.
            app.init_resource::<SignalOutput<Option<String>>>();
            let source_map = (MutableBTreeMap::builder().values([(1, "one".to_string()), (2, "two".to_string())]))
                .spawn(app.world_mut());

            // We will specifically track the value associated with key `2`.
            let key_to_track = 2;
            let key_signal = source_map.signal_map().key(key_to_track);
            let handle = key_signal
                .map(capture_output::<Option<String>>)
                .register(app.world_mut());

            // Test 1: Initial State (Key is Present). The first update should emit the
            // initial value for the key.
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                Some(Some("two".to_string())),
                "Initial value for present key is incorrect"
            );

            // Test 2: Update Tracked Key's Value. This should cause the signal to emit the
            // new value.
            source_map
                .write(app.world_mut())
                .insert(key_to_track, "two_v2".to_string());
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                Some(Some("two_v2".to_string())),
                "Update to tracked key did not emit correctly"
            );

            // Test 3: Update a Different Key. This should NOT cause the signal to emit, as
            // our key's value hasn't changed.
            clear_output::<Option<String>>(app.world_mut());
            source_map.write(app.world_mut()).insert(1, "one_v2".to_string());
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                None,
                "Signal emitted when a different key was updated"
            );

            // Test 4: Remove Tracked Key. This should cause the signal to emit `None`.
            source_map.write(app.world_mut()).remove(&key_to_track);
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                Some(None),
                "Removing the tracked key did not emit None"
            );

            // Test 5: No Change (Key is Absent). Another update with the key still absent
            // should not emit.
            clear_output::<Option<String>>(app.world_mut());
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                None,
                "Signal emitted when key remained absent"
            );

            // Test 6: Re-insert Tracked Key. This should cause the signal to emit the new
            // value.
            source_map
                .write(app.world_mut())
                .insert(key_to_track, "two_reborn".to_string());
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                Some(Some("two_reborn".to_string())),
                "Re-inserting the tracked key did not emit its value"
            );

            // Test 7: Clear the map. Since this removes the key, it should emit `None`.
            source_map.write(app.world_mut()).clear();
            app.update();
            assert_eq!(
                get_output::<Option<String>>(app.world_mut()),
                Some(None),
                "Clearing the map did not emit None for the tracked key"
            );
            handle.release(app.world_mut());
        }
    }

    #[derive(Resource, Default, Debug)]
    struct SignalVecOutput<T: Send + Sync + 'static + Clone + fmt::Debug>(Vec<VecDiff<T>>);

    fn capture_vec_output<T>(In(diffs): In<Vec<VecDiff<T>>>, mut output: ResMut<SignalVecOutput<T>>)
    where
        T: Send + Sync + 'static + Clone + fmt::Debug,
    {
        output.0.extend(diffs);
    }

    fn get_and_clear_vec_output<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &mut World) -> Vec<VecDiff<T>> {
        world
            .get_resource_mut::<SignalVecOutput<T>>()
            .map(|mut res| core::mem::take(&mut res.0))
            .unwrap_or_default()
    }

    fn apply_diffs_to_vec<T: Clone>(vec: &mut Vec<T>, diffs: Vec<VecDiff<T>>) {
        for diff in diffs {
            diff.apply_to_vec(vec);
        }
    }

    // ADD: The comprehensive unit test for `signal_vec_keys`.
    #[test]
    fn test_signal_vec_keys() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>(); // Keys are u32

            // Start with unsorted data to verify initial sort.
            let source_map = (MutableBTreeMap::builder().values([(3, 'c'), (1, 'a'), (4, 'd')])).spawn(app.world_mut());

            let keys_signal = source_map.signal_vec_keys();
            let handle = keys_signal
                .for_each(capture_vec_output::<u32>)
                .register(app.world_mut());

            // Local mirror of the key state for verification.
            let mut current_keys: Vec<u32> = vec![];

            // --- 2. Test Initial State ---
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace { values: vec![1, 3, 4] },
                "Initial state should be a Replace with sorted keys."
            );
            apply_diffs_to_vec(&mut current_keys, diffs);
            assert_eq!(current_keys, vec![1, 3, 4]);

            // --- 3. Test Insert ---
            // Insert a key that goes in the middle.
            source_map.write(app.world_mut()).insert(2, 'b');
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::InsertAt { index: 1, value: 2 });
            apply_diffs_to_vec(&mut current_keys, diffs);
            assert_eq!(current_keys, vec![1, 2, 3, 4]);

            // Insert a key at the beginning.
            source_map.write(app.world_mut()).insert(0, 'z');
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::InsertAt { index: 0, value: 0 });
            apply_diffs_to_vec(&mut current_keys, diffs);
            assert_eq!(current_keys, vec![0, 1, 2, 3, 4]);

            // --- 4. Test Update (No Key Change) ---
            // This should produce NO diffs for the keys vector.
            source_map.write(app.world_mut()).insert(3, 'C'); // Update value for key 3
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert!(diffs.is_empty(), "Updating a value should not produce a key diff.");
            assert_eq!(current_keys, vec![0, 1, 2, 3, 4]); // State unchanged

            // --- 5. Test Remove ---
            // Remove key '3' from the middle of the sorted list.
            source_map.write(app.world_mut()).remove(&3);
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 3 }); // '3' was at index 3
            apply_diffs_to_vec(&mut current_keys, diffs);
            assert_eq!(current_keys, vec![0, 1, 2, 4]);

            // --- 6. Test Batched Diffs ---
            {
                let mut writer = source_map.write(app.world_mut());
                writer.remove(&1); // current_keys should become [0, 2, 4]
                writer.insert(5, 'e'); // current_keys should become [0, 2, 4, 5]
            }
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::RemoveAt { index: 1 }, VecDiff::InsertAt { index: 3, value: 5 }],
                "Batched diffs were not processed correctly."
            );
            apply_diffs_to_vec(&mut current_keys, diffs);
            assert_eq!(
                current_keys,
                vec![0, 2, 4, 5],
                "State after batched diffs is incorrect."
            );

            // --- 7. Test Clear ---
            source_map.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_vec_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs_to_vec(&mut current_keys, diffs);
            assert!(current_keys.is_empty());

            // --- 8. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_signal_vec_entries() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<(u32, char)>>(); // Entries are (u32, char)

            // Start with unsorted data to verify initial sort.
            let source_map = (MutableBTreeMap::builder().values([(3, 'c'), (1, 'a'), (4, 'd')])).spawn(app.world_mut());

            let entries_signal = source_map.signal_vec_entries();
            let handle = entries_signal
                .for_each(capture_vec_output::<(u32, char)>)
                .register(app.world_mut());

            // Local mirror of the entry state for verification.
            let mut current_entries: Vec<(u32, char)> = vec![];

            // --- 2. Test Initial State ---
            app.update();
            let diffs = get_and_clear_vec_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec![(1, 'a'), (3, 'c'), (4, 'd')]
                },
                "Initial state should be a Replace with sorted entries."
            );
            apply_diffs_to_vec(&mut current_entries, diffs);
            assert_eq!(current_entries, vec![(1, 'a'), (3, 'c'), (4, 'd')]);

            // --- 3. Test Insert ---
            // Insert an entry that goes in the middle of the sorted list.
            source_map.write(app.world_mut()).insert(2, 'b');
            app.update();
            let diffs = get_and_clear_vec_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: (2, 'b')
                }
            );
            apply_diffs_to_vec(&mut current_entries, diffs);
            assert_eq!(current_entries, vec![(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]);

            // --- 4. Test Update ---
            // Update the value for an existing key. The index should remain the same.
            source_map.write(app.world_mut()).insert(3, 'C'); // Update value for key 3
            app.update();
            let diffs = get_and_clear_vec_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 2, // '3' is at index 2 in the sorted list
                    value: (3, 'C')
                }
            );
            apply_diffs_to_vec(&mut current_entries, diffs);
            assert_eq!(current_entries, vec![(1, 'a'), (2, 'b'), (3, 'C'), (4, 'd')]);

            // --- 5. Test Remove ---
            // Remove key '1' from the beginning of the sorted list.
            source_map.write(app.world_mut()).remove(&1);
            app.update();
            let diffs = get_and_clear_vec_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 }); // '1' was at index 0
            apply_diffs_to_vec(&mut current_entries, diffs);
            assert_eq!(current_entries, vec![(2, 'b'), (3, 'C'), (4, 'd')]);

            // --- 6. Test Batched Diffs ---
            {
                let mut writer = source_map.write(app.world_mut());
                writer.remove(&4); // current_entries should become [(2, 'b'), (3, 'C')]
                writer.insert(0, 'z'); // current_entries should become [(0, 'z'), (2, 'b'), (3, 'C')]
            }
            app.update();
            let diffs = get_and_clear_vec_output::<(u32, char)>(app.world_mut());
            assert_eq!(
                diffs,
                vec![
                    VecDiff::RemoveAt { index: 2 }, // '4' was at index 2
                    VecDiff::InsertAt {
                        index: 0,
                        value: (0, 'z')
                    }  // '0' is inserted at index 0
                ],
                "Batched diffs were not processed correctly."
            );
            apply_diffs_to_vec(&mut current_entries, diffs);
            assert_eq!(
                current_entries,
                vec![(0, 'z'), (2, 'b'), (3, 'C')],
                "State after batched diffs is incorrect."
            );

            // --- 7. Test Clear ---
            source_map.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_vec_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs_to_vec(&mut current_entries, diffs);
            assert!(current_entries.is_empty());

            // --- 8. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_empty_map_first_insert() {
        {
            // An insertion before the first replay is included in one authoritative snapshot
            // and is not replayed again as incremental history.

            let mut app = create_test_app();
            app.init_resource::<SignalMapOutput<String, i32>>();

            // Start with an empty map
            let source_map = MutableBTreeMap::from(app.world_mut());

            // Create a signal_map and register it
            let signal = source_map.signal_map();
            let handle = signal.for_each(capture_map_output).register(app.world_mut());
            let mut downstream = BTreeMap::new();

            // Insert the first entry
            source_map.write(app.world_mut()).insert("a".to_string(), 42);
            app.update();

            let diffs = get_and_clear_map_output::<String, i32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![MapDiff::Replace {
                    entries: vec![("a".to_string(), 42)],
                }],
                "The first replay must snapshot the inserted entry without a trailing Insert"
            );
            apply_map_diffs(&mut downstream, diffs);
            assert_eq!(downstream, BTreeMap::from([("a".to_string(), 42)]));

            // Insert another entry to verify normal operation continues
            source_map.write(app.world_mut()).insert("b".to_string(), 99);
            app.update();

            let diffs = get_and_clear_map_output::<String, i32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![MapDiff::Insert {
                    key: "b".to_string(),
                    value: 99,
                }]
            );
            apply_map_diffs(&mut downstream, diffs);
            assert_eq!(
                downstream,
                BTreeMap::from([("a".to_string(), 42), ("b".to_string(), 99)])
            );

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_map_replay_initially_empty_then_insert() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();
        let source = MutableBTreeMap::<i32, i32>::builder().spawn(app.world_mut());
        let handle = source
            .signal_map()
            .for_each(capture_map_output)
            .register(app.world_mut());
        let mut downstream = BTreeMap::new();

        app.update();
        let initial_diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
        assert_eq!(
            initial_diffs,
            vec![MapDiff::Replace { entries: vec![] }],
            "an initially empty replay must emit one authoritative empty snapshot"
        );
        apply_map_diffs(&mut downstream, initial_diffs);
        assert!(downstream.is_empty());

        source.write(app.world_mut()).insert(1, 10);
        app.update();
        let later_diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
        assert_eq!(
            later_diffs,
            vec![MapDiff::Insert { key: 1, value: 10 }],
            "an insertion after the snapshot must remain incremental"
        );
        apply_map_diffs(&mut downstream, later_diffs);
        assert_eq!(downstream, BTreeMap::from([(1, 10)]));

        handle.release(app.world_mut());
    }

    #[test]
    fn test_map_replay_filters_historical_prefix_and_forwards_multiple_newer_diffs() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();
        let source = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20), (3, 30)])
            .spawn(app.world_mut());

        source.write(app.world_mut()).remove(&1);
        let source_after_snapshot = source.clone();
        let handle = source
            .signal_map()
            .for_each(move |In(diffs): In<Vec<MapDiff<i32, i32>>>, world: &mut World| {
                world
                    .resource_mut::<SignalMapOutput<i32, i32>>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs.iter().any(|diff| {
                    matches!(
                        diff,
                        MapDiff::Replace { entries } if entries == &vec![(2, 20), (3, 30)]
                    )
                }) {
                    let mut source = source_after_snapshot.write(world);
                    source.insert(4, 40);
                    source.insert(5, 50);
                }
            })
            .register(app.world_mut());

        app.update();
        let mut actual_diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_map_output::<i32, i32>(app.world_mut()));
        let expected_diffs = vec![
            MapDiff::Replace {
                entries: vec![(2, 20), (3, 30)],
            },
            MapDiff::Insert { key: 4, value: 40 },
            MapDiff::Insert { key: 5, value: 50 },
        ];
        let mut downstream = BTreeMap::new();
        apply_map_diffs(&mut downstream, actual_diffs.iter().cloned());
        assert_eq!(
            actual_diffs, expected_diffs,
            "replay must skip the snapshotted Remove prefix and forward every newer diff in order"
        );
        assert_eq!(downstream, BTreeMap::from([(2, 20), (3, 30), (4, 40), (5, 50)]));

        handle.release(app.world_mut());
    }

    #[test]
    fn test_map_replay_snapshot_skips_pending_history_without_newer_diff() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();
        let source = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20)])
            .spawn(app.world_mut());

        source.write(app.world_mut()).remove(&1);
        let handle = source
            .signal_map()
            .for_each(capture_map_output)
            .register(app.world_mut());

        app.update();
        let mut actual_diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_map_output::<i32, i32>(app.world_mut()));
        let expected_diffs = vec![MapDiff::Replace { entries: vec![(2, 20)] }];
        let mut downstream = BTreeMap::new();
        apply_map_diffs(&mut downstream, actual_diffs.iter().cloned());
        assert_eq!(
            actual_diffs, expected_diffs,
            "history already represented by the snapshot must not be emitted again"
        );
        assert_eq!(downstream, BTreeMap::from([(2, 20)]));

        handle.release(app.world_mut());
    }

    #[test]
    fn test_nonempty_map_initial_replace() {
        {
            // Test that when a MutableBTreeMap starts with entries,
            // we get an initial Replace diff

            let mut app = create_test_app();
            app.init_resource::<SignalMapOutput<String, i32>>();

            // Start with a map containing initial entries
            let source_map = MutableBTreeMap::builder()
                .values([("x".to_string(), 1), ("y".to_string(), 2), ("z".to_string(), 3)])
                .spawn(app.world_mut());

            // Create a signal_map and register it
            let signal = source_map.signal_map();
            let handle = signal.for_each(capture_map_output).register(app.world_mut());

            // First update should produce a Replace with initial entries
            app.update();

            let diffs = get_and_clear_map_output::<String, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Expected exactly one diff for initial state");
            assert_eq!(
                diffs[0],
                MapDiff::Replace {
                    entries: vec![("x".to_string(), 1), ("y".to_string(), 2), ("z".to_string(), 3),]
                },
                "Expected a Replace diff with initial entries"
            );

            // Insert another entry to verify normal operation continues
            source_map.write(app.world_mut()).insert("w".to_string(), 4);
            app.update();

            let diffs = get_and_clear_map_output::<String, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                MapDiff::Insert {
                    key: "w".to_string(),
                    value: 4
                }
            );

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn map_replay_mutated_before_registration_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();
        let source = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20), (3, 30)])
            .spawn(app.world_mut());

        source.write(app.world_mut()).remove(&1);
        let source_after_snapshot = source.clone();
        let handle = source
            .signal_map()
            .for_each(move |In(diffs): In<Vec<MapDiff<i32, i32>>>, world: &mut World| {
                world
                    .resource_mut::<SignalMapOutput<i32, i32>>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs.iter().any(|diff| {
                    matches!(
                        diff,
                        MapDiff::Replace { entries } if entries == &vec![(2, 20), (3, 30)]
                    )
                }) {
                    source_after_snapshot.write(world).insert(4, 40);
                }
            })
            .register(app.world_mut());

        app.update();
        let mut actual_diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_map_output::<i32, i32>(app.world_mut()));
        let mut downstream = BTreeMap::new();
        apply_map_diffs(&mut downstream, actual_diffs.iter().cloned());
        let expected_diffs = vec![
            MapDiff::Replace {
                entries: vec![(2, 20), (3, 30)],
            },
            MapDiff::Insert { key: 4, value: 40 },
        ];
        let expected_state = BTreeMap::from([(2, 20), (3, 30), (4, 40)]);
        let correct = actual_diffs == expected_diffs && downstream == expected_state;

        handle.release(app.world_mut());
        assert!(
            correct,
            "map replay snapshot/history overlap before registration: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, expected_state={expected_state:?}, public_reconstructed_downstream={downstream:?}"
        );
    }

    #[test]
    fn map_replay_two_subscribers_pending_batch_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<CollectionCorrectnessMapCaptureA>();
        app.init_resource::<CollectionCorrectnessMapCaptureB>();
        let source = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20), (3, 30)])
            .spawn(app.world_mut());

        let handle_a = source
            .signal_map()
            .for_each(move |In(diffs): In<Vec<MapDiff<i32, i32>>>, world: &mut World| {
                world.resource_mut::<CollectionCorrectnessMapCaptureA>().0.extend(diffs);
            })
            .register(app.world_mut());
        app.update();
        let initial_a = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessMapCaptureA>().0);
        let mut downstream_a = BTreeMap::new();
        apply_map_diffs(&mut downstream_a, initial_a.iter().cloned());
        assert_eq!(
            downstream_a,
            BTreeMap::from([(1, 10), (2, 20), (3, 30)]),
            "subscriber A initial state was not reconstructed from emitted diffs; initial_a={initial_a:?}"
        );

        source.write(app.world_mut()).remove(&1);
        let source_after_b_snapshot = source.clone();
        let handle_b = source
            .signal_map()
            .for_each(move |In(diffs): In<Vec<MapDiff<i32, i32>>>, world: &mut World| {
                world
                    .resource_mut::<CollectionCorrectnessMapCaptureB>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs.iter().any(|diff| {
                    matches!(
                        diff,
                        MapDiff::Replace { entries } if entries == &vec![(2, 20), (3, 30)]
                    )
                }) {
                    source_after_b_snapshot.write(world).insert(4, 40);
                }
            })
            .register(app.world_mut());

        app.update();
        app.update();
        let actual_diffs_a = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessMapCaptureA>().0);
        let actual_diffs_b = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessMapCaptureB>().0);
        apply_map_diffs(&mut downstream_a, actual_diffs_a.iter().cloned());
        let mut downstream_b = BTreeMap::new();
        apply_map_diffs(&mut downstream_b, actual_diffs_b.iter().cloned());
        let expected_diffs_a = vec![MapDiff::Remove { key: 1 }, MapDiff::Insert { key: 4, value: 40 }];
        let expected_diffs_b = vec![
            MapDiff::Replace {
                entries: vec![(2, 20), (3, 30)],
            },
            MapDiff::Insert { key: 4, value: 40 },
        ];
        let expected_state = BTreeMap::from([(2, 20), (3, 30), (4, 40)]);
        let correct = actual_diffs_a == expected_diffs_a
            && actual_diffs_b == expected_diffs_b
            && downstream_a == expected_state
            && downstream_b == expected_state;

        handle_b.release(app.world_mut());
        handle_a.release(app.world_mut());
        assert!(
            correct,
            "per-subscriber map replay cursor divergence: initial_a={initial_a:?}, expected_subscriber_a_diffs={expected_diffs_a:?}, actual_subscriber_a_diffs={actual_diffs_a:?}, subscriber_a_downstream={downstream_a:?}, expected_subscriber_b_diffs={expected_diffs_b:?}, actual_subscriber_b_diffs={actual_diffs_b:?}, subscriber_b_downstream={downstream_b:?}, expected_state={expected_state:?}"
        );
    }

    fn take_collection_correctness_capture_a(world: &mut World) -> Vec<MapDiff<i32, i32>> {
        core::mem::take(&mut world.resource_mut::<CollectionCorrectnessMapCaptureA>().0)
    }

    fn take_collection_correctness_capture_b(world: &mut World) -> Vec<MapDiff<i32, i32>> {
        core::mem::take(&mut world.resource_mut::<CollectionCorrectnessMapCaptureB>().0)
    }

    fn settle_exact_map_collection_correctness_oracle(
        app: &mut App,
        downstream_a: &mut BTreeMap<i32, i32>,
        downstream_b: &mut BTreeMap<i32, i32>,
        expected_diffs: Vec<MapDiff<i32, i32>>,
        expected_state: &BTreeMap<i32, i32>,
        operation: &str,
    ) {
        app.update();
        let diffs_a = take_collection_correctness_capture_a(app.world_mut());
        let diffs_b = take_collection_correctness_capture_b(app.world_mut());
        assert_eq!(
            diffs_a, expected_diffs,
            "subscriber A emitted the wrong exact structure after {operation}"
        );
        assert_eq!(
            diffs_b, expected_diffs,
            "subscriber B emitted the wrong exact structure after {operation}"
        );
        apply_map_diffs(downstream_a, diffs_a);
        apply_map_diffs(downstream_b, diffs_b);
        assert_eq!(
            downstream_a, expected_state,
            "subscriber A public-diff reconstruction diverged after {operation}"
        );
        assert_eq!(
            downstream_b, expected_state,
            "subscriber B public-diff reconstruction diverged after {operation}"
        );
    }

    #[test]
    fn mutable_btree_map_signal_map_boundaries_collection_correctness_oracle() {
        {
            let mut app = create_test_app();
            app.init_resource::<SignalMapOutput<i32, i32>>();
            let source = MutableBTreeMap::<i32, i32>::builder().spawn(app.world_mut());
            let handle = source
                .signal_map()
                .for_each(capture_map_output::<i32, i32>)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![MapDiff::Replace { entries: vec![] }],
                "an empty initial collection must begin with one authoritative Replace"
            );
            let mut downstream = BTreeMap::from([(99, 99)]);
            apply_map_diffs(&mut downstream, diffs);
            assert!(
                downstream.is_empty(),
                "the empty initial Replace must authoritatively discard downstream state"
            );

            handle.release(app.world_mut());
        }

        let mut app = create_test_app();
        app.init_resource::<CollectionCorrectnessMapCaptureA>();
        app.init_resource::<CollectionCorrectnessMapCaptureB>();
        let source = MutableBTreeMap::builder()
            .values([(3, 30), (1, 10)])
            .spawn(app.world_mut());
        let mut model = BTreeMap::from([(1, 10), (3, 30)]);
        let handle_a = source
            .signal_map()
            .for_each(
                |In(diffs): In<Vec<MapDiff<i32, i32>>>, mut output: ResMut<CollectionCorrectnessMapCaptureA>| {
                    output.0.extend(diffs);
                },
            )
            .register(app.world_mut());

        source.write(app.world_mut()).insert(2, 20);
        model.insert(2, 20);
        app.update();
        let initial_a = take_collection_correctness_capture_a(app.world_mut());
        let expected_initial_a = vec![MapDiff::Replace {
            entries: vec![(1, 10), (2, 20), (3, 30)],
        }];
        assert_eq!(
            initial_a, expected_initial_a,
            "insert-before-snapshot must be folded into one ordered authoritative Replace"
        );
        let mut downstream_a = BTreeMap::from([(-1, -1)]);
        apply_map_diffs(&mut downstream_a, initial_a);
        assert_eq!(
            downstream_a, model,
            "subscriber A failed to reconstruct the populated initial state from public diffs"
        );

        {
            let mut writer = source.write(app.world_mut());
            assert_eq!(writer.insert(2, 22), Some(20));
            assert_eq!(writer.remove(&1), Some(10));
            assert_eq!(writer.remove(&99), None);
            assert_eq!(writer.insert(4, 40), None);
        }
        model.insert(2, 22);
        model.remove(&1);
        model.insert(4, 40);

        let handle_b = source
            .signal_map()
            .for_each(
                |In(diffs): In<Vec<MapDiff<i32, i32>>>, mut output: ResMut<CollectionCorrectnessMapCaptureB>| {
                    output.0.extend(diffs);
                },
            )
            .register(app.world_mut());
        app.update();
        let prefix_a = take_collection_correctness_capture_a(app.world_mut());
        let initial_b = take_collection_correctness_capture_b(app.world_mut());
        let expected_prefix_a = vec![
            MapDiff::Update { key: 2, value: 22 },
            MapDiff::Remove { key: 1 },
            MapDiff::Insert { key: 4, value: 40 },
        ];
        let expected_initial_b = vec![MapDiff::Replace {
            entries: vec![(2, 22), (3, 30), (4, 40)],
        }];
        assert_eq!(
            prefix_a, expected_prefix_a,
            "the established subscriber must receive the exact pending-history prefix"
        );
        assert_eq!(
            initial_b, expected_initial_b,
            "the later subscriber must replace with the ordered current state and filter the pending-history prefix"
        );
        apply_map_diffs(&mut downstream_a, prefix_a);
        let mut downstream_b = BTreeMap::from([(-2, -2)]);
        apply_map_diffs(&mut downstream_b, initial_b);
        assert_eq!(downstream_a, model, "subscriber A diverged after the exact prefix");
        assert_eq!(
            downstream_b, model,
            "subscriber B baseline diverged after the exact prefix"
        );

        {
            let mut writer = source.write(app.world_mut());
            assert_eq!(writer.remove(&3), Some(30));
            assert_eq!(writer.insert(5, 50), None);
            assert_eq!(writer.insert(4, 44), Some(40));
        }
        model.remove(&3);
        model.insert(5, 50);
        model.insert(4, 44);
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![
                MapDiff::Remove { key: 3 },
                MapDiff::Insert { key: 5, value: 50 },
                MapDiff::Update { key: 4, value: 44 },
            ],
            &model,
            "the post-registration batched suffix",
        );

        assert_eq!(source.write(app.world_mut()).remove(&99), None);
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![],
            &model,
            "a missing remove",
        );

        source.write(app.world_mut()).clear();
        model.clear();
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![MapDiff::Clear],
            &model,
            "a nonempty clear",
        );

        source.write(app.world_mut()).clear();
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![],
            &model,
            "an already-empty clear",
        );

        source.write(app.world_mut()).replace(BTreeMap::new());
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![MapDiff::Replace { entries: vec![] }],
            &model,
            "an empty Replace",
        );

        let replacement = BTreeMap::from([(8, 80), (6, 60), (7, 70)]);
        source.write(app.world_mut()).replace(replacement.clone());
        model = replacement;
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![MapDiff::Replace {
                entries: vec![(6, 60), (7, 70), (8, 80)],
            }],
            &model,
            "a nonempty ordered Replace",
        );

        assert_eq!(source.write(app.world_mut()).insert(9, 90), None);
        model.insert(9, 90);
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![MapDiff::Insert { key: 9, value: 90 }],
            &model,
            "a new-key insert",
        );

        assert_eq!(source.write(app.world_mut()).insert(7, 77), Some(70));
        model.insert(7, 77);
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![MapDiff::Update { key: 7, value: 77 }],
            &model,
            "an existing-key insert",
        );

        assert_eq!(source.write(app.world_mut()).remove(&6), Some(60));
        model.remove(&6);
        settle_exact_map_collection_correctness_oracle(
            &mut app,
            &mut downstream_a,
            &mut downstream_b,
            vec![MapDiff::Remove { key: 6 }],
            &model,
            "a successful remove",
        );

        handle_b.release(app.world_mut());
        handle_a.release(app.world_mut());
    }

    struct CollectionCorrectnessTinyLcg(u64);

    impl CollectionCorrectnessTinyLcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }

        fn key(&mut self) -> i32 {
            (self.next_u32() % 10) as i32
        }

        fn value(&mut self) -> i32 {
            (self.next_u32() % 2_001) as i32 - 1_000
        }
    }

    #[test]
    fn mutable_btree_map_signal_map_generated_collection_correctness_oracle() {
        const SEED: u64 = 0xC011_EC71_0A5E_2026;
        const STEPS: usize = 160;
        const SECOND_SUBSCRIBER_STEP: usize = 61;

        let mut rng = CollectionCorrectnessTinyLcg(SEED);
        let mut app = create_test_app();
        app.init_resource::<CollectionCorrectnessMapCaptureA>();
        app.init_resource::<CollectionCorrectnessMapCaptureB>();
        let source = MutableBTreeMap::builder()
            .values([(7, 70), (2, 20)])
            .spawn(app.world_mut());
        let mut model = BTreeMap::from([(2, 20), (7, 70)]);
        let handle_a = source
            .signal_map()
            .for_each(
                |In(diffs): In<Vec<MapDiff<i32, i32>>>, mut output: ResMut<CollectionCorrectnessMapCaptureA>| {
                    output.0.extend(diffs);
                },
            )
            .register(app.world_mut());
        let mut handle_b = None;
        let mut downstream_a = BTreeMap::new();
        let mut downstream_b = BTreeMap::new();
        let mut trace = vec![format!("initial={model:?}")];

        app.update();
        let initial_a = take_collection_correctness_capture_a(app.world_mut());
        let expected_initial = vec![MapDiff::Replace {
            entries: model.clone().into_iter().collect(),
        }];
        assert_eq!(
            initial_a, expected_initial,
            "seed={SEED:#x} step=initial current_op=register subscriber A trace={trace:#?}"
        );
        apply_map_diffs(&mut downstream_a, initial_a);
        assert_eq!(
            downstream_a, model,
            "seed={SEED:#x} step=initial current_op=apply authoritative Replace trace={trace:#?}"
        );

        for step in 0..STEPS {
            let operation = match rng.next_u32() % 9 {
                0..=3 => {
                    let key = rng.key();
                    let value = rng.value();
                    let old_model = model.insert(key, value);
                    let old_source = source.write(app.world_mut()).insert(key, value);
                    assert_eq!(
                        old_source, old_model,
                        "seed={SEED:#x} step={step} current_op=insert({key}, {value}) trace={trace:#?}"
                    );
                    format!("insert key={key} value={value} existed={}", old_model.is_some())
                }
                4 => {
                    let key = rng.key();
                    let old_model = model.remove(&key);
                    let old_source = source.write(app.world_mut()).remove(&key);
                    assert_eq!(
                        old_source, old_model,
                        "seed={SEED:#x} step={step} current_op=remove({key}) trace={trace:#?}"
                    );
                    format!("remove key={key} existed={}", old_model.is_some())
                }
                5 => {
                    let was_empty = model.is_empty();
                    model.clear();
                    source.write(app.world_mut()).clear();
                    format!("clear was_empty={was_empty}")
                }
                6 | 7 => {
                    let attempts = (rng.next_u32() % 11) as usize;
                    let mut replacement = BTreeMap::new();
                    for _ in 0..attempts {
                        replacement.insert(rng.key(), rng.value());
                    }
                    source.write(app.world_mut()).replace(replacement.clone());
                    model = replacement;
                    format!("replace entries={model:?}")
                }
                8 => {
                    let count = 2 + (rng.next_u32() % 3) as usize;
                    let mut details = Vec::with_capacity(count);
                    let mut writer = source.write(app.world_mut());
                    for _ in 0..count {
                        let key = rng.key();
                        if rng.next_u32().is_multiple_of(3) {
                            let existed = model.remove(&key).is_some();
                            writer.remove(&key);
                            details.push(format!("remove({key}) existed={existed}"));
                        } else {
                            let value = rng.value();
                            let existed = model.insert(key, value).is_some();
                            writer.insert(key, value);
                            details.push(format!("insert({key}, {value}) existed={existed}"));
                        }
                    }
                    format!("batch [{}]", details.join(", "))
                }
                _ => unreachable!(),
            };
            trace.push(format!("{step}: {operation}"));

            assert!(
                model.len() <= 10,
                "seed={SEED:#x} step={step} current_op={operation} map_size={} trace={trace:#?}",
                model.len()
            );

            if step == SECOND_SUBSCRIBER_STEP {
                handle_b = Some(
                    source
                        .signal_map()
                        .for_each(
                            |In(diffs): In<Vec<MapDiff<i32, i32>>>,
                             mut output: ResMut<CollectionCorrectnessMapCaptureB>| {
                                output.0.extend(diffs);
                            },
                        )
                        .register(app.world_mut()),
                );
            }

            app.update();
            let diffs_a = take_collection_correctness_capture_a(app.world_mut());
            apply_map_diffs(&mut downstream_a, diffs_a.iter().cloned());
            assert_eq!(
                downstream_a, model,
                "seed={SEED:#x} step={step} current_op={operation} subscriber=A diffs={diffs_a:?} trace={trace:#?}"
            );

            if step >= SECOND_SUBSCRIBER_STEP {
                let diffs_b = take_collection_correctness_capture_b(app.world_mut());
                if step == SECOND_SUBSCRIBER_STEP {
                    let expected_b = vec![MapDiff::Replace {
                        entries: model.clone().into_iter().collect(),
                    }];
                    assert_eq!(
                        diffs_b, expected_b,
                        "seed={SEED:#x} step={step} current_op={operation} subscriber B's first diff was not one ordered authoritative Replace around pending history trace={trace:#?}"
                    );
                }
                apply_map_diffs(&mut downstream_b, diffs_b.iter().cloned());
                assert_eq!(
                    downstream_b, model,
                    "seed={SEED:#x} step={step} current_op={operation} subscriber=B diffs={diffs_b:?} trace={trace:#?}"
                );
            }
        }

        handle_b.unwrap().release(app.world_mut());
        handle_a.release(app.world_mut());
    }

    fn settle_name_map_collection_correctness_oracle(
        app: &mut App,
        downstream: &mut BTreeMap<u32, Name>,
        expected_state: &BTreeMap<u32, Name>,
        operation: &str,
    ) -> Vec<MapDiff<u32, Name>> {
        app.update();
        let diffs = get_and_clear_map_output::<u32, Name>(app.world_mut());
        apply_map_diffs(downstream, diffs.iter().cloned());
        assert_eq!(
            downstream, expected_state,
            "dynamic map_value_signal public-diff reconstruction diverged after {operation}; diffs={diffs:?}"
        );
        diffs
    }

    #[test]
    fn map_value_signal_dynamic_lifecycle_collection_correctness_oracle() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<u32, Name>>();

        let old_alice = app.world_mut().spawn(Name::new("Alice")).id();
        let old_bob = app.world_mut().spawn(Name::new("Bob")).id();
        let source = MutableBTreeMap::builder()
            .values([(2, old_bob), (1, old_alice)])
            .spawn(app.world_mut());
        let handle = source
            .signal_map()
            .map_value_signal(|In(entity): In<Entity>| signal::from_component::<Name>(entity).dedupe())
            .for_each(capture_map_output::<u32, Name>)
            .register(app.world_mut());
        let mut logical_values = BTreeMap::from([(1, Name::new("Alice")), (2, Name::new("Bob"))]);
        let mut downstream = BTreeMap::from([(99, Name::new("stale"))]);

        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "the initial populated collection",
        );
        assert_eq!(
            diffs,
            vec![MapDiff::Replace {
                entries: vec![(1, Name::new("Alice")), (2, Name::new("Bob"))],
            }],
            "initial dynamic values must be one ordered authoritative Replace"
        );

        *app.world_mut().get_mut::<Name>(old_alice).unwrap() = Name::new("Alicia");
        logical_values.insert(1, Name::new("Alicia"));
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "an active inner update",
        );
        assert_eq!(
            diffs,
            vec![MapDiff::Update {
                key: 1,
                value: Name::new("Alicia"),
            }]
        );

        let old_carol = app.world_mut().spawn(Name::new("Carol")).id();
        source.write(app.world_mut()).insert(3, old_carol);
        logical_values.insert(3, Name::new("Carol"));
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "a new key and inner insertion",
        );
        assert_eq!(
            diffs,
            vec![MapDiff::Insert {
                key: 3,
                value: Name::new("Carol"),
            }]
        );

        let replacement_alice = app.world_mut().spawn(Name::new("Ada")).id();
        source.write(app.world_mut()).insert(1, replacement_alice);
        logical_values.insert(1, Name::new("Ada"));
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "replacement of an existing key's inner source",
        );
        assert_eq!(
            diffs,
            vec![MapDiff::Update {
                key: 1,
                value: Name::new("Ada"),
            }]
        );

        *app.world_mut().get_mut::<Name>(old_alice).unwrap() = Name::new("superseded Alice");
        let state_before_superseded_update = downstream.clone();
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "mutation of a superseded old inner source",
        );
        assert!(
            diffs.is_empty(),
            "a superseded old inner source emitted output: {diffs:?}"
        );
        assert_eq!(
            downstream, state_before_superseded_update,
            "a superseded old inner source changed downstream state"
        );

        source.write(app.world_mut()).remove(&2);
        logical_values.remove(&2);
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "removal of an active key",
        );
        assert_eq!(diffs, vec![MapDiff::Remove { key: 2 }]);

        *app.world_mut().get_mut::<Name>(old_alice).unwrap() = Name::new("superseded Alice again");
        *app.world_mut().get_mut::<Name>(old_bob).unwrap() = Name::new("removed Bob");
        let state_before_old_updates = downstream.clone();
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "mutations of replaced and removed retained old inner sources",
        );
        assert!(
            diffs.is_empty(),
            "retained superseded old inner sources emitted output: {diffs:?}"
        );
        assert_eq!(
            downstream, state_before_old_updates,
            "retained superseded old inner sources changed downstream state"
        );

        source.write(app.world_mut()).replace(BTreeMap::new());
        logical_values.clear();
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "a complete empty Replace",
        );
        assert_eq!(diffs, vec![MapDiff::Replace { entries: vec![] }]);

        *app.world_mut().get_mut::<Name>(replacement_alice).unwrap() = Name::new("superseded Ada");
        *app.world_mut().get_mut::<Name>(old_carol).unwrap() = Name::new("superseded Carol");
        let state_before_empty_boundary_updates = downstream.clone();
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "mutations of old inners after the empty replacement boundary",
        );
        assert!(
            diffs.is_empty(),
            "old inners crossed the empty replacement boundary: {diffs:?}"
        );
        assert_eq!(
            downstream, state_before_empty_boundary_updates,
            "old inners changed state across the empty replacement boundary"
        );

        let new_dora = app.world_mut().spawn(Name::new("Dora")).id();
        let new_eve = app.world_mut().spawn(Name::new("Eve")).id();
        source
            .write(app.world_mut())
            .replace(BTreeMap::from([(4, new_eve), (2, new_dora)]));
        logical_values = BTreeMap::from([(2, Name::new("Dora")), (4, Name::new("Eve"))]);
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "repopulation with new inners",
        );
        assert_eq!(
            diffs,
            vec![MapDiff::Replace {
                entries: vec![(2, Name::new("Dora")), (4, Name::new("Eve"))],
            }]
        );

        *app.world_mut().get_mut::<Name>(new_dora).unwrap() = Name::new("Dorothy");
        logical_values.insert(2, Name::new("Dorothy"));
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "an update from a repopulated new inner",
        );
        assert_eq!(
            diffs,
            vec![MapDiff::Update {
                key: 2,
                value: Name::new("Dorothy"),
            }]
        );

        source.write(app.world_mut()).clear();
        logical_values.clear();
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "the final clear",
        );
        assert_eq!(diffs, vec![MapDiff::Clear]);

        *app.world_mut().get_mut::<Name>(new_dora).unwrap() = Name::new("cleared Dora");
        *app.world_mut().get_mut::<Name>(new_eve).unwrap() = Name::new("cleared Eve");
        let state_before_final_old_updates = downstream.clone();
        let diffs = settle_name_map_collection_correctness_oracle(
            &mut app,
            &mut downstream,
            &logical_values,
            "mutations of retained inners after the final clear",
        );
        assert!(diffs.is_empty(), "cleared old inners emitted output: {diffs:?}");
        assert_eq!(
            downstream, state_before_final_old_updates,
            "cleared old inners changed downstream state"
        );

        handle.release(app.world_mut());
    }

    #[test]
    fn test_boxed_replayable_map_inherent_methods_preserve_replayable_with_prelude() {
        use crate::prelude::*;

        fn assert_replayable_map<S: ReplayableSignalMap>(_signal: &S) -> &S {
            _signal
        }
        fn assert_replayable_scalar<S: ReplayableSignal>(_signal: &S) -> &S {
            _signal
        }
        fn assert_replayable_vec<S: ReplayableSignalVec>(_signal: &S) -> &S {
            _signal
        }

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default().with_schedule::<Update>()));

        let src = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20)])
            .spawn(app.world_mut());

        let cached = src.signal_map().boxed_replayable().cache();
        assert_replayable_map(&cached);

        let broadcast = cached.boxed_replayable().broadcast();
        assert_replayable_map(&broadcast);

        let scheduled = broadcast.clone().boxed_replayable().schedule::<Update>();
        assert_replayable_map(&scheduled);
        let map_claim = scheduled.register(app.world_mut());

        // Also verify opaque `Key`, `SignalVecKeys`, and `SignalVecEntries` preserve replayable
        // capabilities across `.cache()`, `.broadcast()`, and `.schedule::<Update>()`.
        let key_scheduled = src.signal_map().key(1).cache().broadcast().schedule::<Update>();
        assert_replayable_scalar(&key_scheduled);
        let key_claim = key_scheduled.register(app.world_mut());

        let keys_scheduled = src.signal_vec_keys().cache().broadcast().schedule::<Update>();
        assert_replayable_vec(&keys_scheduled);
        let keys_claim = keys_scheduled.register(app.world_mut());

        let entries_scheduled = src.signal_vec_entries().cache().broadcast().schedule::<Update>();
        assert_replayable_vec(&entries_scheduled);
        let entries_claim = entries_scheduled.register(app.world_mut());

        for claim in [map_claim, key_claim, keys_claim, entries_claim] {
            claim.release(app.world_mut());
        }
    }

    #[test]
    fn test_key_clear_suppression_when_absent() {
        #[derive(Resource, Default)]
        struct KeyLog(Vec<Option<i32>>);

        let mut app = create_test_app();
        app.init_resource::<KeyLog>();

        // Start with key `1` present and watched key `99` absent.
        let src = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
        let handle = src
            .signal_map()
            .key(99)
            .map(|In(val): In<Option<i32>>, mut log: ResMut<KeyLog>| {
                log.0.push(val);
            })
            .register(app.world_mut());

        // Initial `Replace` must establish the baseline `None` once.
        app.update();
        assert_eq!(app.world().resource::<KeyLog>().0, vec![None]);
        app.world_mut().resource_mut::<KeyLog>().0.clear();

        // Clearing the map when key `99` was already absent must NOT re-emit `None`.
        src.write(app.world_mut()).clear();
        app.update();
        assert!(
            app.world().resource::<KeyLog>().0.is_empty(),
            "MapDiff::Clear must not emit a duplicate None when the watched key was already absent"
        );

        // Inserting and clearing another key still must NOT emit `None`.
        src.write(app.world_mut()).insert(2, 20);
        app.update();
        src.write(app.world_mut()).clear();
        app.update();
        assert!(app.world().resource::<KeyLog>().0.is_empty());

        // Inserting watched key `99` emits `Some(990)`, and clearing now DOES emit `None`.
        src.write(app.world_mut()).insert(99, 990);
        app.update();
        assert_eq!(app.world().resource::<KeyLog>().0, vec![Some(990)]);
        app.world_mut().resource_mut::<KeyLog>().0.clear();

        src.write(app.world_mut()).clear();
        app.update();
        assert_eq!(app.world().resource::<KeyLog>().0, vec![None]);

        handle.release(app.world_mut());
    }

    #[test]
    #[should_panic(expected = "SignalMap::map_value value system failed or skipped")]
    fn test_map_value_panics_on_skipped_value_system() {
        #[derive(Resource)]
        struct MissingFactor(i32);

        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();

        let src = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
        let _claim = src
            .signal_map()
            .map_value(|In(val): In<i32>, factor: Res<MissingFactor>| val * factor.0)
            .for_each(capture_map_output::<i32, i32>)
            .register(app.world_mut());

        app.update();
    }

    #[test]
    fn test_map_value_signal_reactivation_resets_entries() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();

        let src = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20)])
            .spawn(app.world_mut());
        let mapped = src
            .signal_map()
            .map_value_signal(|In(v): In<i32>| signal::from_system(move |In(())| v * 10).dedupe())
            .broadcast();

        let claim1 = mapped
            .clone()
            .for_each(capture_map_output::<i32, i32>)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_map_output::<i32, i32>(app.world_mut()),
            vec![MapDiff::Replace {
                entries: vec![(1, 100), (2, 200)],
            }]
        );

        // `1 -> 0` deactivation.
        claim1.release(app.world_mut());

        // Mutate source while deactivated (`0` claims).
        src.write(app.world_mut()).replace(BTreeMap::from([(3, 30)]));
        app.update();

        // `0 -> 1` reactivation: stale keys `(1, 2)` must not leak into `entries`.
        let claim2 = mapped
            .for_each(capture_map_output::<i32, i32>)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_map_output::<i32, i32>(app.world_mut()),
            vec![MapDiff::Replace {
                entries: vec![(3, 300)]
            }]
        );

        claim2.release(app.world_mut());
    }

    #[test]
    fn test_mutable_btree_map_despawned_entity_polls_cleanly() {
        let mut app = create_test_app();
        app.init_resource::<SignalMapOutput<i32, i32>>();

        let src = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
        let data_entity = src.entity();
        let claim = src
            .signal_map()
            .for_each(capture_map_output::<i32, i32>)
            .register(app.world_mut());

        app.update();
        assert_eq!(
            get_and_clear_map_output::<i32, i32>(app.world_mut()),
            vec![MapDiff::Replace { entries: vec![(1, 10)] }]
        );

        // Despawn the backing entity directly while `src` and `claim` are still alive: polling
        // must return `None` cleanly instead of panicking.
        app.world_mut().entity_mut(data_entity).despawn();
        app.update();
        assert!(get_and_clear_map_output::<i32, i32>(app.world_mut()).is_empty());

        claim.release(app.world_mut());
    }
}
