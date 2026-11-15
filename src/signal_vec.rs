//! Data structures and combinators for constructing reactive [`System`] dependency graphs on top
//! of [`Vec`] mutations.
//!
//! This module provides [`SignalVec`], a collection-oriented signal trait with **diff-based
//! semantics**. Rather than forwarding entire collections each frame, [`SignalVec`] propagates
//! [`VecDiff`] values that describe incremental mutations (insert, remove, update, move, etc.).
//! This enables efficient, constant-time reactive updates for collections of any size.
//!
//! See [`MutableVec`] for the primary source type and [`SignalVecExt`] for available combinators.
//! For the general signal graph runtime model and core concepts, see the [`Signal`] trait
//! documentation.
pub use super::signal::{OrdinaryBroadcast, ReplayableBroadcastMarker};
use super::{
    graph::SignalClaim,
    signal::{Broadcast, Signal, broadcast_from_materializer},
};
use crate::{
    DynamicEmissions, DynamicManagerStep, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand,
    OutputDelivery, ReplayInitialization, SameSignalIdentity, ScalarSignalMaterializer, SignalRegistrationError,
    VecSignalKind, VecSignalMaterializer, cleanup::CleanupSender, registration::SignalMaterializerFactory,
};

use bevy_ecs::{change_detection::Mut, prelude::*, schedule::ScheduleLabel};
use bevy_platform::{
    collections::HashMap,
    prelude::*,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};
use core::{cmp::Ordering, fmt, marker::PhantomData, ops::Deref};

#[derive(Component)]
struct IndexLookup {
    map: HashMap<usize, usize>,
    lifetime: Weak<IndexLookupLifetime>,
}

// Readers retain only data, not a graph claim on the producer.
struct IndexLookupLifetime {
    entity: Entity,
    cleanup: CleanupSender,
}

impl IndexLookupLifetime {
    fn new(world: &mut World) -> Arc<Self> {
        let entity = world.spawn_empty().id();
        Self::init_at(world, entity)
    }

    fn init_at(world: &mut World, entity: Entity) -> Arc<Self> {
        let cleanup = CleanupSender::bound(world);
        let lifetime = Arc::new(Self { entity, cleanup });
        world.entity_mut(entity).insert(IndexLookup {
            map: HashMap::default(),
            lifetime: Arc::downgrade(&lifetime),
        });
        lifetime
    }

    fn signal(self: &Arc<Self>, key: usize) -> Broadcast<Option<usize>> {
        let lifetime = self.clone();
        broadcast_from_materializer(ScalarSignalMaterializer::new(move |registrar| {
            let lookup = registrar.preflight_component::<IndexLookup>(lifetime.entity)?;
            if !Weak::ptr_eq(&lookup.lifetime, &Arc::downgrade(&lifetime)) {
                return Err(SignalRegistrationError::EntityUnavailable);
            }
            registrar.signal_entity(
                (),
                move |In(()), lookups: Query<&IndexLookup>| {
                    Some(
                        lookups
                            .get(lifetime.entity)
                            .ok()
                            .and_then(|lookup| lookup.map.get(&key).copied()),
                    )
                },
                OutputDelivery::broadcast(),
            )
        }))
    }

    fn synchronize(&self, world: &mut World, keys: &[usize]) {
        let mut lookup = world.get_mut::<IndexLookup>(self.entity).expect("live index lookup");
        lookup.map.clear();
        lookup
            .map
            .extend(keys.iter().copied().enumerate().map(|(index, key)| (key, index)));
    }
}

impl Drop for IndexLookupLifetime {
    fn drop(&mut self) {
        self.cleanup.send_index_lookup(self.entity);
    }
}

/// Describes the mutations made to the underlying [`MutableVec`] that are piped to downstream
/// [`SignalVec`]s.
#[allow(missing_docs)]
pub enum VecDiff<T> {
    Replace { values: Vec<T> },
    InsertAt { index: usize, value: T },
    UpdateAt { index: usize, value: T },
    RemoveAt { index: usize },
    Move { old_index: usize, new_index: usize },
    Push { value: T },
    Pop,
    Clear,
}

impl<T> Clone for VecDiff<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Replace { values } => Self::Replace { values: values.clone() },
            Self::InsertAt { index, value } => Self::InsertAt {
                index: *index,
                value: value.clone(),
            },
            Self::UpdateAt { index, value } => Self::UpdateAt {
                index: *index,
                value: value.clone(),
            },
            Self::RemoveAt { index } => Self::RemoveAt { index: *index },
            Self::Move { old_index, new_index } => Self::Move {
                old_index: *old_index,
                new_index: *new_index,
            },
            Self::Push { value } => Self::Push { value: value.clone() },
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
            Self::Move { old_index, new_index } => f
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

impl<T> VecDiff<T> {
    #[allow(missing_docs)]
    pub fn apply_to_vec(self, vec: &mut Vec<T>) {
        match self {
            VecDiff::Replace { values } => *vec = values,
            VecDiff::InsertAt { index, value } => vec.insert(index, value),
            VecDiff::UpdateAt { index, value } => {
                if let Some(elem) = vec.get_mut(index) {
                    *elem = value;
                }
            }
            VecDiff::RemoveAt { index } => {
                if index < vec.len() {
                    vec.remove(index);
                }
            }
            VecDiff::Move { old_index, new_index } => {
                if old_index < vec.len() {
                    let val = vec.remove(old_index);
                    vec.insert(new_index, val);
                }
            }
            VecDiff::Push { value } => vec.push(value),
            VecDiff::Pop => {
                vec.pop();
            }
            VecDiff::Clear => vec.clear(),
        }
    }
}

/// A lazy, composable chain specialized for diff-based [`Vec`] reactivity.
///
/// Unlike [`Signal`] which forwards complete values, a [`SignalVec`] propagates [`VecDiff`]
/// values describing incremental mutations to an underlying collection. This diff-based approach
/// enables **constant-time reactive updates** regardless of collection size; only the changes are
/// transmitted and processed, not the entire collection.
///
/// Downstream consumers receive a stream of [`VecDiff`] variants ([`InsertAt`](VecDiff::InsertAt),
/// [`RemoveAt`](VecDiff::RemoveAt), [`UpdateAt`](VecDiff::UpdateAt), [`Move`](VecDiff::Move),
/// etc.) that they can apply to maintain a synchronized view of the source data.
///
/// A concrete chain is a lazy Rust value. Registration materializes its framework-created backing
/// into world-resident entities and returns one move-only direct claim on the terminal. Cloning a
/// cloneable chain shares its lazy identity but acquires no claim. External implementations remain
/// supported by returning a [`VecSignalMaterializer`]; they receive no graph-mutation authority.
///
/// For the general graph runtime and composition model, see [`Signal`].
///
/// Public implementations provide only materializer transfer; they cannot define or override a
/// world-taking registration method:
///
/// ```compile_fail,E0407
/// use bevy_ecs::world::World;
/// use jonmo::{VecSignalMaterializer, graph::SignalClaim, signal_vec::SignalVec};
///
/// struct InvalidSignalVec;
///
/// impl SignalVec for InvalidSignalVec {
///     type Item = i32;
///
///     fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
///         unimplemented!()
///     }
///
///     fn register_signal_vec(self, _world: &mut World) -> SignalClaim {
///         unimplemented!()
///     }
/// }
/// ```
///
/// Ordinary vector chains are linear (`!Clone`); call [`SignalVecExt::broadcast`] for explicit
/// fan-out:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// fn assert_clone<T: Clone>(_: &T) {}
///
/// let mut world = World::new();
/// let chain = MutableVec::builder().values([1, 2, 3]).spawn(&mut world).signal_vec();
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
/// let chain = MutableVec::builder().values([1, 2, 3]).spawn(&mut world).signal_vec();
/// let _first = chain.register(&mut world);
/// let _second = chain.register(&mut world);
/// ```
///
/// ```compile_fail,E0599
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let claim = MutableVec::builder()
///     .values([1, 2, 3])
///     .spawn(&mut world)
///     .signal_vec()
///     .register(&mut world);
/// let _ = claim.cache();
/// ```
pub trait SignalVec: Send + Sync + 'static {
    /// Output type.
    type Item;

    /// Transfers this concrete chain's framework-created, one-shot materializer backing.
    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item>;
}

impl<O: 'static> SignalVec for Box<dyn SignalVec<Item = O> + Send + Sync> {
    type Item = O;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        (*self).into_signal_vec_materializer()
    }
}

/// An alias for a move-only boxed [`SignalVec`], returned by [`SignalVecExt::boxed`].
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// fn assert_clone<T: Clone>(_: &T) {}
///
/// let mut world = World::new();
/// let boxed = MutableVec::builder().values([1, 2]).spawn(&mut world).signal_vec().boxed();
/// assert_clone(&boxed);
/// ```
pub type BoxedSignalVec<T> = Box<dyn SignalVec<Item = T> + Send + Sync>;

pub(crate) mod replayable_signal_vec_sealed {
    pub trait Sealed {}
}

/// A vector signal whose terminal replay contract is managed and verified by Jonmo.
///
/// New dependents receive targeted authoritative complete replacements with relationship-local
/// cursors, followed only by newer accepted revisions. `Warming` is distinct from a ready empty
/// vector; an empty `Replace` is authoritative. One private maintenance claim keeps a retained
/// replay terminal current. Replay guarantees eventual handoff, not same-frame switching.
///
/// Stateful high-level switching accepts only this capability. Normalize a non-replayable vector
/// chain with [`SignalVecExt::cache`] before selecting it.
pub trait ReplayableSignalVec: SignalVec + replayable_signal_vec_sealed::Sealed {}

/// A type-erased, move-only replayable vector signal.
pub struct BoxedReplayableSignalVec<T: 'static>(Box<dyn ReplayableSignalVec<Item = T> + Send + Sync>);

impl<T: 'static> BoxedReplayableSignalVec<T> {
    /// Returns this already-verified replayable vector signal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable vector terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable vector terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_vec_schedule(self)
    }

    /// Returns this already-boxed replayable vector signal without adding another allocation.
    pub fn boxed_replayable(self) -> Self {
        self
    }
}

impl<T: 'static> SignalVec for BoxedReplayableSignalVec<T> {
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.0.into_signal_vec_materializer()
    }
}

impl<T: 'static> replayable_signal_vec_sealed::Sealed for BoxedReplayableSignalVec<T> {}

impl<T> ReplayableSignalVec for BoxedReplayableSignalVec<T> where T: Clone + Send + Sync + 'static {}

/// Explicit cloneable fan-out wrapper created by [`SignalVecExt::broadcast`].
///
/// Cloning `BroadcastVec` shares one canonical lazy terminal identity without acquiring a graph
/// claim until a clone is registered or consumed by a downstream combinator. Downstream combinators
/// on a `BroadcastVec` clone return ordinary linear (`!Clone`) vector chains unless `.broadcast()`
/// is called again at the new fan-out point.
pub struct BroadcastVec<T: 'static, Mode = OrdinaryBroadcast> {
    materializer: VecSignalMaterializer<T>,
    _mode: PhantomData<fn() -> Mode>,
}

/// Alias for a [`BroadcastVec`] handle backed by a verified [`ReplayableSignalVec`] terminal.
pub type ReplayableBroadcastVec<T> = BroadcastVec<T, ReplayableBroadcastMarker>;

impl<T: Clone + 'static, Mode> Clone for BroadcastVec<T, Mode> {
    fn clone(&self) -> Self {
        Self {
            materializer: self.materializer.shared_clone(),
            _mode: PhantomData,
        }
    }
}

impl<T: 'static> BroadcastVec<T, OrdinaryBroadcast> {
    /// Returns this already-broadcast handle without adding another graph entity.
    pub fn broadcast(self) -> Self {
        self
    }
}

impl<T: 'static> BroadcastVec<T, ReplayableBroadcastMarker> {
    /// Returns this already-verified replayable broadcast vector without adding another cache
    /// entity, claim, or targeted snapshot transfer.
    pub fn cache(self) -> Self {
        self
    }

    /// Returns this already-broadcast replayable handle without adding another graph entity.
    pub fn broadcast(self) -> Self {
        self
    }

    /// Assigns a schedule to this verified replayable broadcast vector while preserving its replay
    /// capability.
    ///
    /// Note that `.schedule::<Sched>()` returns a linear [`ReplayableScheduledVec`] handle and tags
    /// the shared canonical terminal when materialized. When multiple clones should share the
    /// scheduled terminal, prefer calling `.schedule::<Sched>().broadcast()` (rather than cloning
    /// before `.schedule::<Sched>()`), as assigning conflicting schedules to an already-registered
    /// canonical terminal fails registration.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<T: 'static, Mode: Send + Sync + 'static> SignalVec for BroadcastVec<T, Mode> {
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<T: 'static> replayable_signal_vec_sealed::Sealed for BroadcastVec<T, ReplayableBroadcastMarker> {}

impl<T: Clone + Send + Sync + 'static> ReplayableSignalVec for BroadcastVec<T, ReplayableBroadcastMarker> {}

pub(crate) fn replayable_vec_broadcast<S>(signal: S) -> ReplayableBroadcastVec<S::Item>
where
    S: ReplayableSignalVec,
    S::Item: Clone + Send + Sync + 'static,
{
    let upstream = Box::new(signal).into_signal_vec_materializer();
    let materializer = VecSignalMaterializer::new(move |registrar| {
        let upstream = registrar.materialize_signal_vec(upstream)?;
        registrar
            .vec_broadcast_replayable_terminal(upstream)?
            .ok_or(SignalRegistrationError::ConflictingCanonicalState)
    });
    BroadcastVec {
        materializer,
        _mode: PhantomData,
    }
}

pub(crate) fn replayable_vec_schedule<Sched, S>(signal: S) -> ReplayableScheduledVec<Sched, S::Item>
where
    Sched: ScheduleLabel + Default + 'static,
    S: ReplayableSignalVec + 'static,
{
    let upstream = Box::new(signal).into_signal_vec_materializer();
    let materializer = VecSignalMaterializer::new(move |registrar| {
        let upstream = registrar.materialize_signal_vec(upstream)?;
        registrar.apply_schedule(upstream, Sched::default())
    });
    ScheduledVec {
        materializer,
        _marker: PhantomData,
    }
}

impl<Upstream, Switched> replayable_signal_vec_sealed::Sealed for crate::signal::SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: ReplayableSignalVec,
    Switched::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream, Switched> ReplayableSignalVec for crate::signal::SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: ReplayableSignalVec,
    Switched::Item: Clone + Send + Sync + 'static,
{
}

/// A vector cache wrapper created by [`SignalVecExt::cache`].
pub struct CachedSignalVec<Upstream: SignalVec> {
    materializer: VecSignalMaterializer<Upstream::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream: SignalVec> CachedSignalVec<Upstream> {
    /// Returns this already-verified cache without adding another graph entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable cache terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<Upstream::Item>
    where
        Upstream::Item: Clone + Send + Sync + 'static,
    {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable cache terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, Upstream::Item>
    where
        Upstream: 'static,
        Upstream::Item: Clone + Send + Sync + 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<Upstream> SignalVec for CachedSignalVec<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone + Send + Sync + 'static,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_vec_sealed::Sealed for CachedSignalVec<Upstream> where Upstream: SignalVec {}

impl<Upstream> ReplayableSignalVec for CachedSignalVec<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone + Send + Sync + 'static,
{
}

/// Signal graph node with no upstreams which forwards [`Vec<VecDiff<T>>`]s flushed from some source
/// [`MutableVec<T>`], see [`MutableVec::signal_vec`].
pub struct Source<T: 'static> {
    materializer: VecSignalMaterializer<T>,
}

impl<T: 'static> Source<T> {
    /// Returns this already-verified source without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Returns a cloneable broadcast wrapper backed directly by this [`MutableVec`] source
    /// terminal without adding another entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this [`MutableVec`] source terminal while preserving its verified
    /// replay capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<T> SignalVec for Source<T>
where
    T: 'static,
{
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<T: 'static> replayable_signal_vec_sealed::Sealed for Source<T> {}

impl<T> ReplayableSignalVec for Source<T> where T: Clone + Send + Sync + 'static {}

/// Signal graph node which applies a [`System`] directly to the "raw" [`Vec<VecDiff>`]s of its
/// upstream, see [`.for_each`](SignalVecExt::for_each).
pub struct ForEach<Upstream, O: 'static> {
    materializer: crate::ScalarSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for ForEach<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which applies a [`System`] to each [`Item`](SignalVec::Item) of its upstream,
/// see [`.map`](SignalVecExt::map).
pub struct Map<Upstream, O: 'static> {
    materializer: VecSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> SignalVec for Map<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which applies a [`System`] to each [`Item`](SignalVec::Item) of its upstream,
/// forwarding the output of each resulting [`Signal`], see
/// [`.map_signal`](SignalVecExt::map_signal).
pub struct MapSignal<Upstream, S: Signal> {
    materializer: VecSignalMaterializer<S::Item>,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S: Signal> MapSignal<Upstream, S>
where
    Upstream: SignalVec,
    S: super::signal::ReplayableSignal + 'static,
    S::Item: Clone + Send + Sync + 'static,
{
    /// Returns this already-verified replayable terminal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable terminal in-place to broadcast fan-out without adding another
    /// cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<S::Item> {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, S::Item>
    where
        Upstream: 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<Upstream, S: Signal> SignalVec for MapSignal<Upstream, S>
where
    Upstream: SignalVec,
    S: super::signal::ReplayableSignal + 'static,
    S::Item: Clone + Send + Sync + 'static,
{
    type Item = S::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream, S> replayable_signal_vec_sealed::Sealed for MapSignal<Upstream, S>
where
    Upstream: SignalVec,
    S: super::signal::ReplayableSignal,
    S::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream, S> ReplayableSignalVec for MapSignal<Upstream, S>
where
    Upstream: SignalVec,
    S: super::signal::ReplayableSignal,
    S::Item: Clone + Send + Sync + 'static,
{
}

fn find_index<'a>(indices: impl Iterator<Item = &'a bool>, index: usize) -> usize {
    indices.take(index).filter(|x| **x).count()
}

/// Generic helper for filter/filter_map operations on VecDiff streams.
/// `process` takes a value and returns `Option<O>` - `Some` means include, `None` means exclude.
fn filter_generic_helper<T, O>(
    diffs: Vec<VecDiff<T>>,
    indices: &mut Vec<bool>,
    mut process: impl FnMut(T) -> Option<O>,
) -> Option<Vec<VecDiff<O>>> {
    let mut output = vec![];
    for diff in diffs {
        let diff_option = match diff {
            VecDiff::Replace { values } => {
                *indices = Vec::with_capacity(values.len());
                let mut out_values = Vec::with_capacity(values.len());
                for input in values {
                    let mapped = process(input);
                    indices.push(mapped.is_some());
                    if let Some(value) = mapped {
                        out_values.push(value);
                    }
                }
                Some(VecDiff::Replace { values: out_values })
            }
            VecDiff::InsertAt { index, value } => {
                if let Some(value) = process(value) {
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
                if let Some(value) = process(value) {
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
                } else if indices[index] {
                    indices[index] = false;
                    Some(VecDiff::RemoveAt {
                        index: find_index(indices.iter(), index),
                    })
                } else {
                    None
                }
            }
            VecDiff::Move { old_index, new_index } => {
                // Determine if the item being moved is part of the filtered set.
                let was_included = indices[old_index];

                // Calculate the filtered old and new indices BEFORE and AFTER mutating `indices`.
                let filtered_old_index = find_index(indices.iter(), old_index);
                let temp_val = indices.remove(old_index);
                indices.insert(new_index, temp_val);
                let filtered_new_index = find_index(indices.iter(), new_index);

                if was_included && filtered_old_index != filtered_new_index {
                    Some(VecDiff::Move {
                        old_index: filtered_old_index,
                        new_index: filtered_new_index,
                    })
                } else {
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
                if let Some(value) = process(value) {
                    indices.push(true);
                    Some(VecDiff::Push { value })
                } else {
                    indices.push(false);
                    None
                }
            }
            VecDiff::Pop => {
                if indices.pop().expect("can't pop from empty vec") {
                    Some(VecDiff::Pop)
                } else {
                    None
                }
            }
            VecDiff::Clear => {
                let had_passed = indices.iter().any(|&included| included);
                indices.clear();
                had_passed.then_some(VecDiff::Clear)
            }
        };
        if let Some(diff) = diff_option {
            output.push(diff);
        }
    }
    if output.is_empty() { None } else { Some(output) }
}

/// Signal graph node which selectively forwards upstream [`Item`](SignalVec::Item)s, see
/// [`.filter`](SignalVecExt::filter).
pub struct Filter<Upstream>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<Upstream::Item>,
}

impl<Upstream> SignalVec for Filter<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which transforms and selectively forwards upstream [`Item`](SignalVec::Item)s,
/// see [`.filter_map`](SignalVecExt::filter_map).
pub struct FilterMap<Upstream, O: 'static>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> SignalVec for FilterMap<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which selectively forwards upstream [`Item`](SignalVec::Item)s depending on a
/// [`Signal<Item = bool>`], see [`.filter_signal`](SignalVecExt::filter_signal).
pub struct FilterSignal<Upstream>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<Upstream::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> FilterSignal<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone + Send + Sync + 'static,
{
    /// Returns this already-verified replayable terminal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable terminal in-place to broadcast fan-out without adding another
    /// cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<Upstream::Item> {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, Upstream::Item>
    where
        Upstream: 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<Upstream> SignalVec for FilterSignal<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_vec_sealed::Sealed for FilterSignal<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream> ReplayableSignalVec for FilterSignal<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone + Send + Sync + 'static,
{
}

/// Signal graph node which prepends an index signal to each upstream [`Item`](SignalVec::Item), see
/// [`.enumerate`](SignalVecExt::enumerate).
pub struct Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    #[allow(clippy::type_complexity)]
    materializer: VecSignalMaterializer<(Broadcast<Option<usize>>, Upstream::Item)>,
}

impl<Upstream> SignalVec for Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    type Item = (Broadcast<Option<usize>>, Upstream::Item);

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which collects upstream [`Item`](SignalVec::Item)s into a single [`Vec`], see
/// [`.to_signal`](SignalVecExt::to_signal).
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

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

fn apply_vec_diffs_strict<T>(values: &mut Vec<T>, diffs: Vec<VecDiff<T>>) {
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
            VecDiff::Move { old_index, new_index } => {
                let old = values.remove(old_index);
                values.insert(new_index, old);
            }
            VecDiff::Push { value } => {
                values.push(value);
            }
            VecDiff::Pop => {
                values.pop().expect("can't pop from empty vec");
            }
            VecDiff::Clear => {
                values.clear();
            }
        }
    }
}

fn apply_len_diffs<T>(len: &mut usize, diffs: Vec<VecDiff<T>>) {
    for diff in diffs {
        match diff {
            VecDiff::Replace { values } => {
                *len = values.len();
            }
            VecDiff::InsertAt { index, .. } => {
                assert!(
                    index <= *len,
                    "insertion index (is {index}) should be <= len (is {})",
                    *len
                );
                *len += 1;
            }
            VecDiff::UpdateAt { index, .. } => {
                assert!(
                    index < *len,
                    "index out of bounds: the len is {} but the index is {index}",
                    *len
                );
            }
            VecDiff::RemoveAt { index } => {
                assert!(index < *len, "removal index (is {index}) should be < len (is {})", *len);
                *len -= 1;
            }
            VecDiff::Move { old_index, new_index } => {
                assert!(
                    old_index < *len,
                    "removal index (is {old_index}) should be < len (is {})",
                    *len
                );
                assert!(
                    new_index < *len,
                    "insertion index (is {new_index}) should be <= len (is {})",
                    *len - 1
                );
            }
            VecDiff::Push { .. } => {
                *len += 1;
            }
            VecDiff::Pop => {
                *len = len.checked_sub(1).expect("can't pop from empty vec");
            }
            VecDiff::Clear => {
                *len = 0;
            }
        }
    }
}

/// Signal graph node which outputs whether its upstream is populated, see
/// [`.is_empty`](SignalVecExt::is_empty).
pub struct IsEmpty<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<Upstream, bool>,
}

impl<Upstream> Signal for IsEmpty<Upstream>
where
    Upstream: SignalVec,
{
    type Item = bool;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which outputs its upstream's length, see [`.len`](SignalVecExt::len).
pub struct Len<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<Upstream, usize>,
}

impl<Upstream> Signal for Len<Upstream>
where
    Upstream: SignalVec,
{
    type Item = usize;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which outputs its upstream's sum, see [`.sum`](SignalVecExt::sum).
pub struct Sum<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for Sum<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> crate::ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

#[cfg_attr(test, derive(Clone))]
pub(crate) enum ChainInput<T> {
    Left(Vec<VecDiff<T>>),
    Right(Vec<VecDiff<T>>),
}

struct ChainState<T> {
    lengths: ChainLengths,
    published: Option<Vec<T>>,
    input_epoch: Option<u64>,
    repair_pending: bool,
}

impl<T> Default for ChainState<T> {
    fn default() -> Self {
        Self {
            lengths: ChainLengths::default(),
            published: None,
            input_epoch: None,
            repair_pending: false,
        }
    }
}

impl<T: Clone> ChainState<T> {
    fn transform(&mut self, input: ChainInput<T>, input_epoch: u64) -> Option<Vec<VecDiff<T>>> {
        if self
            .input_epoch
            .replace(input_epoch)
            .is_some_and(|previous| previous != input_epoch)
        {
            self.repair_pending = true;
        }
        let (is_left, diffs) = match input {
            ChainInput::Left(diffs) => (true, diffs),
            ChainInput::Right(diffs) => (false, diffs),
        };
        let mut output = self.lengths.transform(is_left, diffs)?;
        if self.repair_pending {
            // Deactivation may have canceled downstream delivery after these lengths advanced.
            // Restore their baseline, rather than translating against the downstream's older state.
            if let Some(published) = &self.published {
                output.insert(
                    0,
                    VecDiff::Replace {
                        values: published.clone(),
                    },
                );
            }
            self.repair_pending = false;
        }
        for diff in &output {
            match diff {
                VecDiff::Replace { values } => self.published = Some(values.clone()),
                diff => {
                    if let Some(published) = &mut self.published {
                        diff.clone().apply_to_vec(published);
                    }
                }
            }
        }
        Some(output)
    }
}

#[derive(Default)]
struct ChainLengths {
    left: usize,
    right: usize,
}

impl ChainLengths {
    fn transform<T>(&mut self, is_left: bool, diffs: Vec<VecDiff<T>>) -> Option<Vec<VecDiff<T>>> {
        let Self {
            left: left_len,
            right: right_len,
        } = self;
        let mut out_diffs = Vec::new();
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
                    VecDiff::Move { old_index, new_index } => {
                        out_diffs.push(VecDiff::Move { old_index, new_index });
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
                    VecDiff::Pop => {
                        *left_len -= 1;
                        if *right_len == 0 {
                            out_diffs.push(VecDiff::Pop);
                        } else {
                            out_diffs.push(VecDiff::RemoveAt { index: *left_len });
                        }
                    }
                    VecDiff::Clear => {
                        let removing = *left_len;
                        *left_len = 0;
                        if *right_len == 0 {
                            out_diffs.push(VecDiff::Clear);
                        } else {
                            for i in (0..removing).rev() {
                                out_diffs.push(VecDiff::RemoveAt { index: i });
                            }
                        }
                    }
                }
            } else {
                match diff {
                    VecDiff::Replace { values } => {
                        let removing = *right_len;
                        *right_len = values.len();
                        if *left_len == 0 {
                            out_diffs.push(VecDiff::Replace { values });
                        } else {
                            for _ in 0..removing {
                                out_diffs.push(VecDiff::Pop);
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
                    VecDiff::Move { old_index, new_index } => {
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
                    VecDiff::Pop => {
                        *right_len -= 1;
                        out_diffs.push(VecDiff::Pop);
                    }
                    VecDiff::Clear => {
                        let removing = *right_len;
                        *right_len = 0;
                        if *left_len == 0 {
                            out_diffs.push(VecDiff::Clear);
                        } else {
                            for _ in 0..removing {
                                out_diffs.push(VecDiff::Pop);
                            }
                        }
                    }
                }
            }
        }
        if out_diffs.is_empty() { None } else { Some(out_diffs) }
    }
}

/// Signal graph node which concatenates its upstreams, see [`.chain`](SignalVecExt::chain).
pub struct Chain<Left, Right>
where
    Left: SignalVec,
    Right: SignalVec<Item = Left::Item>,
{
    materializer: VecSignalMaterializer<Left::Item>,
    _marker: PhantomData<fn() -> (Left, Right)>,
}

impl<Left, Right> SignalVec for Chain<Left, Right>
where
    Left: SignalVec,
    Right: SignalVec<Item = Left::Item>,
{
    type Item = Left::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node that places a separator between each upstream [`Item`](SignalVec::Item), see
/// [`.intersperse`](SignalVecExt::intersperse).
pub struct Intersperse<Upstream>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<Upstream::Item>,
}

impl<Upstream> SignalVec for Intersperse<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node that places a [`System`]-dependent separator between each upstream
/// [`Item`](SignalVec::Item), see [`.intersperse_with`](SignalVecExt::intersperse_with).
pub struct IntersperseWith<Upstream>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<Upstream::Item>,
}

impl<Upstream> SignalVec for IntersperseWith<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node that sorts each upstream [`Item`](SignalVec::Item) based on a comparison
/// [`System`], see [`.sort_by`](SignalVecExt::sort_by).
pub struct SortBy<Upstream>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<Upstream::Item>,
}

impl<Upstream> SignalVec for SortBy<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node that sorts each upstream [`Item`](SignalVec::Item) based on a key extraction
/// [`System`], see [`.sort_by_key`](SignalVecExt::sort_by_key).
pub struct SortByKey<Upstream>
where
    Upstream: SignalVec,
{
    materializer: VecSignalMaterializer<Upstream::Item>,
}

impl<Upstream> SignalVec for SortByKey<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which flattens a [`SignalVec`] of replayable [`SignalVec`]s, see
/// [`.flatten`](SignalVecExt::flatten).
pub struct Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: SignalVec,
{
    materializer: VecSignalMaterializer<<Upstream::Item as SignalVec>::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: ReplayableSignalVec + 'static,
    <Upstream::Item as SignalVec>::Item: Clone + Send + Sync + 'static,
{
    /// Returns this already-verified replayable terminal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable terminal in-place to broadcast fan-out without adding another
    /// cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<<Upstream::Item as SignalVec>::Item> {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> ReplayableScheduledVec<Sched, <Upstream::Item as SignalVec>::Item>
    where
        Upstream: 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<Upstream> SignalVec for Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: ReplayableSignalVec + 'static,
    <Upstream::Item as SignalVec>::Item: Clone + Send + Sync + 'static,
{
    type Item = <Upstream::Item as SignalVec>::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_vec_sealed::Sealed for Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: ReplayableSignalVec + 'static,
    <Upstream::Item as SignalVec>::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream> ReplayableSignalVec for Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: ReplayableSignalVec + 'static,
    <Upstream::Item as SignalVec>::Item: Clone + Send + Sync + 'static,
{
}

cfg_if::cfg_if! {
    if #[cfg(feature = "tracing")] {
        /// Signal graph node that debug logs its upstream's "raw" [`Vec<VecDiff>`]s, see
        /// [`.debug`](SignalVecExt::debug).
        pub struct Debug<Upstream>
        where
            Upstream: SignalVec,
        {
            materializer: VecSignalMaterializer<Upstream::Item>,
        }

        impl<Upstream> SignalVec for Debug<Upstream>
        where
            Upstream: SignalVec,
        {
            type Item = Upstream::Item;

            fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
                self.materializer
            }
        }
    }
}

/// Enables returning different concrete [`SignalVec`] types from branching logic without boxing,
/// although note that all [`SignalVec`]s are boxed internally regardless.
///
/// Inspired by <https://github.com/rayon-rs/either>.
#[allow(missing_docs)]
pub enum SignalVecEither<L, R>
where
    L: SignalVec,
    R: SignalVec,
{
    Left(L),
    Right(R),
}

impl<T, L, R> SignalVecEither<L, R>
where
    T: Clone + Send + Sync + 'static,
    L: ReplayableSignalVec<Item = T>,
    R: ReplayableSignalVec<Item = T>,
{
    /// Returns this already-verified replayable branch without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes the selected replayable branch terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<T> {
        replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable branch while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<Sched, T>
    where
        L: 'static,
        R: 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<T: 'static, L, R> SignalVec for SignalVecEither<L, R>
where
    L: SignalVec<Item = T>,
    R: SignalVec<Item = T>,
{
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        match *self {
            SignalVecEither::Left(left) => Box::new(left).into_signal_vec_materializer(),
            SignalVecEither::Right(right) => Box::new(right).into_signal_vec_materializer(),
        }
    }
}

impl<T, L, R> replayable_signal_vec_sealed::Sealed for SignalVecEither<L, R>
where
    T: Clone + Send + Sync + 'static,
    L: ReplayableSignalVec<Item = T>,
    R: ReplayableSignalVec<Item = T>,
{
}

impl<T, L, R> ReplayableSignalVec for SignalVecEither<L, R>
where
    T: Clone + Send + Sync + 'static,
    L: ReplayableSignalVec<Item = T>,
    R: ReplayableSignalVec<Item = T>,
{
}

/// Blanket trait for transforming [`SignalVec`]s into [`SignalVecEither::Left`] or
/// [`SignalVecEither::Right`].
pub trait IntoSignalVecEither: Sized
where
    Self: SignalVec,
{
    /// Wrap this [`SignalVec`] in the [`SignalVecEither::Left`] variant.
    ///
    /// Useful for conditional branching where different [`SignalVec`] types need to be returned
    /// from the same function or closure, particularly with
    /// [`.switch_signal_vec`](crate::signal::SignalExt::switch_signal_vec).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource)]
    /// struct ShowFiltered(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(ShowFiltered(true));
    ///
    /// let list = MutableVec::builder()
    ///     .values([1, 2, 3, 4, 5])
    ///     .spawn(&mut world);
    ///
    /// let signal = signal::from_system(|In(_), res: Res<ShowFiltered>| res.0).switch_signal_vec(
    ///     move |In(show_filtered): In<bool>, world: &mut World| {
    ///         if show_filtered {
    ///             list.signal_vec()
    ///                 .filter(|In(x): In<i32>| x % 2 == 0)
    ///                 .cache()
    ///                 .left_either()
    ///         } else {
    ///             list.signal_vec().right_either()
    ///         }
    ///     },
    /// );
    /// // both branches produce compatible SignalVecEither types
    /// ```
    fn left_either<R>(self) -> SignalVecEither<Self, R>
    where
        R: SignalVec,
    {
        SignalVecEither::Left(self)
    }

    /// Wrap this [`SignalVec`] in the [`SignalVecEither::Right`] variant.
    ///
    /// Useful for conditional branching where different [`SignalVec`] types need to be returned
    /// from the same function or closure, particularly with
    /// [`.switch_signal_vec`](crate::signal::SignalExt::switch_signal_vec).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource)]
    /// struct ListMode(u8);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(ListMode(0));
    ///
    /// let list_a = MutableVec::builder().values([10, 20]).spawn(&mut world);
    /// let list_b = MutableVec::builder()
    ///     .values([100, 200, 300])
    ///     .spawn(&mut world);
    ///
    /// let signal = signal::from_system(|In(_), res: Res<ListMode>| res.0).switch_signal_vec(
    ///     move |In(mode): In<u8>, world: &mut World| {
    ///         if mode == 0 {
    ///             list_a
    ///                 .signal_vec()
    ///                 .map_in(|x: i32| x * 2)
    ///                 .cache()
    ///                 .left_either()
    ///         } else {
    ///             list_b.signal_vec().right_either()
    ///         }
    ///     },
    /// );
    /// // both branches produce compatible SignalVecEither types
    /// ```
    fn right_either<L>(self) -> SignalVecEither<L, Self>
    where
        L: SignalVec,
    {
        SignalVecEither::Right(self)
    }
}

impl<T: SignalVec> IntoSignalVecEither for T {}

// impl<T: SignalVec> IntoSignalEither for T {}
/// Extension trait providing combinator methods for [`SignalVec`]s.
pub trait SignalVecExt: SignalVec {
    /// Explicitly shares this vector signal across multiple downstream consumers.
    ///
    /// Cloning the returned [`BroadcastVec`] handle shares one canonical lazy terminal identity
    /// without acquiring a graph claim until a clone is registered or consumed by a downstream
    /// combinator. Verified replayable terminals normalize in-place to broadcast delivery without
    /// adding an extra entity or maintenance claim; non-replayable inputs materialize one
    /// canonical identity broadcast entity.
    ///
    /// Move-only (`!Clone`) items cannot be broadcast:
    ///
    /// ```compile_fail,E0277
    /// use jonmo::prelude::*;
    ///
    /// struct NonClone(usize);
    ///
    /// let _ = signal::once(vec![NonClone(1)]).to_signal_vec().broadcast();
    /// ```
    fn broadcast(self) -> BroadcastVec<Self::Item>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            if let Some(upstream) = registrar.vec_try_broadcast_terminal(upstream)? {
                return Ok(upstream);
            }
            registrar.signal_vec_entity(
                upstream,
                |In(diffs): In<Vec<VecDiff<Self::Item>>>| Some(diffs),
                OutputDelivery::broadcast(),
            )
        });
        BroadcastVec {
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
    /// Move-only (`!Clone`) items cannot be cached for replay:
    ///
    /// ```compile_fail,E0277
    /// use jonmo::prelude::*;
    ///
    /// struct NonClone(usize);
    ///
    /// let _ = signal::once(vec![NonClone(1)]).to_signal_vec().cache();
    /// ```
    fn cache(self) -> CachedSignalVec<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(Box::new(self).into_signal_vec_materializer())?;
            if let Some(upstream) = registrar.vec_cache_terminal(upstream)? {
                return Ok(upstream);
            }
            registrar.replayable_signal_vec_entity(
                upstream,
                |In(diffs): In<Vec<VecDiff<Self::Item>>>| Some(diffs),
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        CachedSignalVec {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass the "raw" [`Vec<VecDiff<Self::Item>>`] output of this [`SignalVec`] to a [`System`],
    /// continuing propagation if the [`System`] returns [`Some`] or terminating for the frame if it
    /// returns [`None`]. Unlike most other [`SignalVec`] methods,
    /// [`.for_each`](SignalVecExt::for_each), returns a [`Signal`], not a [`SignalVec`], since the
    /// output type need not be an [`Option<Vec<VecDiff>>`]. If the [`System`] logic is infallible,
    /// wrapping the result in an option is unnecessary.
    fn for_each<O, IOO, F, M>(self, system: F) -> ForEach<Self, O>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Vec<VecDiff<Self::Item>>>, IOO, M> + Send + Sync + 'static,
    {
        let materializer = crate::ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(Box::new(self).into_signal_vec_materializer())?;
            registrar.signal_entity(upstream, system, OutputDelivery::move_only())
        });
        ForEach {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`], transforming it.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .map(|In(x): In<i32>| x * 2); // outputs `SignalVec -> [2, 4, 6]`
    /// ```
    fn map<O, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
        O: Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, O, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut item_system = IntoSystem::into_system(system);
            let mut item_system_initialized = false;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                    let mut run_item_system = |value| {
                        if !item_system_initialized {
                            item_system.initialize(world);
                            item_system_initialized = true;
                        }
                        item_system.run(value, world).unwrap_or_else(|err| {
                            panic!(
                                "SignalVec::map item system failed or skipped ({err}); collection item transforms must be total per element (use map_signal for async/warming per-item state)"
                            )
                        })
                    };
                    let mut output = Vec::with_capacity(diffs.len());
                    for diff in diffs {
                        let diff = match diff {
                            VecDiff::Replace { values } => VecDiff::Replace {
                                values: values.into_iter().map(&mut run_item_system).collect(),
                            },
                            VecDiff::InsertAt { index, value } => VecDiff::InsertAt {
                                index,
                                value: run_item_system(value),
                            },
                            VecDiff::UpdateAt { index, value } => VecDiff::UpdateAt {
                                index,
                                value: run_item_system(value),
                            },
                            VecDiff::Push { value } => VecDiff::Push {
                                value: run_item_system(value),
                            },
                            VecDiff::RemoveAt { index } => VecDiff::RemoveAt { index },
                            VecDiff::Move { old_index, new_index } => VecDiff::Move { old_index, new_index },
                            VecDiff::Pop => VecDiff::Pop,
                            VecDiff::Clear => VecDiff::Clear,
                        };
                        output.push(diff);
                    }
                    (!output.is_empty()).then_some(output)
                },
                OutputDelivery::move_only(),
            )
        });
        Map {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to an [`FnMut`], transforming it.
    ///
    /// Convenient when additional [`SystemParam`](bevy_ecs::system::SystemParam)s aren't necessary.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .map_in(|x: i32| x * 2); // outputs `SignalVec -> [2, 4, 6]`
    /// ```
    fn map_in<O, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
        O: Send + Sync + 'static,
        F: FnMut(Self::Item) -> O + Send + Sync + 'static,
    {
        self.map(move |In(item)| function(item))
    }

    /// Pass a reference to each [`Item`](SignalVec::Item) of this [`SignalVec`] to an [`FnMut`],
    /// transforming it.
    ///
    /// Convenient when additional [`SystemParam`](bevy_ecs::system::SystemParam)s aren't necessary
    /// and the target function expects a reference.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .map_in_ref(ToString::to_string); // outputs `SignalVec -> ["1", "2", "3"]`
    /// ```
    fn map_in_ref<O, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
        O: Send + Sync + 'static,
        F: FnMut(&Self::Item) -> O + Send + Sync + 'static,
    {
        self.map(move |In(item)| function(&item))
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`] that produces a
    /// [`Signal`], forwarding the output of each resulting [`Signal`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder().values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .map_signal(|In(x): In<i32>|
    ///         signal::from_system(move |In(_)| x * 2).dedupe()
    ///     ); // outputs `SignalVec -> [2, 4, 6]`
    /// ```
    fn map_signal<S, F, M>(self, system: F) -> MapSignal<Self, S>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
        S: super::signal::ReplayableSignal + 'static,
        S::Item: Clone + Send + Sync,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        enum MapSignalQueueItem<T> {
            Replace { length: usize },
            Insert { index: usize },
            Push { index: usize },
            Remove { index: usize },
            Pop { index: usize },
            Move { old_index: usize, new_index: usize },
            Clear,
            Value { index: usize, value: T },
        }

        #[derive(Clone, Copy)]
        enum PendingInsertion {
            Insert,
            Push,
        }

        struct MappedEntry<T> {
            value: Option<T>,
            insertion: Option<PendingInsertion>,
        }

        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut factory = IntoSystem::into_system(system);
            let mut initialized = false;
            let epoch = Arc::new(AtomicUsize::new(0));
            let manager_epoch = Arc::clone(&epoch);
            let manager = registrar.signal_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                      world: &mut World,
                      mut slot_count: Local<usize>,
                      mut observed_epoch: Local<usize>| {
                    if !initialized {
                        factory.initialize(world);
                        initialized = true;
                    }
                    let current_epoch = manager_epoch.load(AtomicOrdering::Acquire);
                    if *observed_epoch != current_epoch {
                        *observed_epoch = current_epoch;
                        *slot_count = 0;
                    }
                    let mut steps = Vec::new();
                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values } => {
                                let mut signals = Vec::with_capacity(values.len());
                                for value in values {
                                    signals.push(
                                        factory
                                            .run(value, world)
                                            .expect("map_signal factory system failed to run"),
                                    );
                                }
                                *slot_count = signals.len();
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::ReplaceAll { signals },
                                    DynamicEmissions::One(MapSignalQueueItem::Replace { length: *slot_count }),
                                ));
                            }
                            VecDiff::InsertAt { index, value } => {
                                let signal = factory
                                    .run(value, world)
                                    .expect("map_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Insert { index, signal },
                                    DynamicEmissions::One(MapSignalQueueItem::Insert { index }),
                                ));
                                *slot_count += 1;
                            }
                            VecDiff::UpdateAt { index, value } => {
                                let signal = factory
                                    .run(value, world)
                                    .expect("map_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Replace { index, signal },
                                    DynamicEmissions::None,
                                ));
                            }
                            VecDiff::RemoveAt { index } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Remove { index },
                                    DynamicEmissions::One(MapSignalQueueItem::Remove { index }),
                                ));
                                *slot_count = slot_count.saturating_sub(1);
                            }
                            VecDiff::Move { old_index, new_index } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Move { old_index, new_index },
                                    DynamicEmissions::One(MapSignalQueueItem::Move { old_index, new_index }),
                                ));
                            }
                            VecDiff::Push { value } => {
                                let index = *slot_count;
                                let signal = factory
                                    .run(value, world)
                                    .expect("map_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Insert { index, signal },
                                    DynamicEmissions::One(MapSignalQueueItem::Push { index }),
                                ));
                                *slot_count += 1;
                            }
                            VecDiff::Pop => {
                                if let Some(index) = slot_count.checked_sub(1) {
                                    steps.push(DynamicManagerStep::command(
                                        IndexedSlotCommand::Remove { index },
                                        DynamicEmissions::One(MapSignalQueueItem::Pop { index }),
                                    ));
                                    *slot_count = index;
                                }
                            }
                            VecDiff::Clear => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Clear,
                                    DynamicEmissions::One(MapSignalQueueItem::Clear),
                                ));
                                *slot_count = 0;
                            }
                        }
                    }
                    Some(DynamicManagerUpdate::sequenced(steps))
                },
                OutputDelivery::move_only(),
            )?;
            let mut entries = Vec::<MappedEntry<S::Item>>::new();
            let reducer_epoch = Arc::clone(&epoch);
            let mut reducer_observed_epoch = 0usize;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<S::Item>()
                    .adapter(|index, value| DynamicEmissions::One(MapSignalQueueItem::Value { index, value }))
                    .replayable_signal_vec_output(move |items| {
                        let current_epoch = reducer_epoch.load(AtomicOrdering::Acquire);
                        if reducer_observed_epoch != current_epoch {
                            reducer_observed_epoch = current_epoch;
                            entries.clear();
                        }
                        let mut output = Vec::new();
                        let mut replacing = false;
                        for item in items {
                            match item {
                                MapSignalQueueItem::Replace { length } => {
                                    let mut replacement = Vec::with_capacity(length);
                                    for index in 0..length {
                                        replacement.push(MappedEntry {
                                            value: entries.get(index).and_then(|entry| entry.value.clone()),
                                            insertion: None,
                                        });
                                    }
                                    entries = replacement;
                                    replacing = true;
                                }
                                MapSignalQueueItem::Insert { index } => entries.insert(
                                    index,
                                    MappedEntry {
                                        value: None,
                                        insertion: Some(PendingInsertion::Insert),
                                    },
                                ),
                                MapSignalQueueItem::Push { index } => entries.insert(
                                    index,
                                    MappedEntry {
                                        value: None,
                                        insertion: Some(PendingInsertion::Push),
                                    },
                                ),
                                MapSignalQueueItem::Remove { index } => {
                                    if index < entries.len() {
                                        let output_index =
                                            entries.iter().take(index).filter(|entry| entry.value.is_some()).count();
                                        let removed = entries.remove(index);
                                        if removed.value.is_some() && !replacing {
                                            output.push(VecDiff::RemoveAt { index: output_index });
                                        }
                                    }
                                }
                                MapSignalQueueItem::Pop { index } => {
                                    if index < entries.len() {
                                        let removed = entries.remove(index);
                                        if removed.value.is_some() && !replacing {
                                            output.push(VecDiff::Pop);
                                        }
                                    }
                                }
                                MapSignalQueueItem::Move { old_index, new_index } => {
                                    if old_index < entries.len() && new_index < entries.len() {
                                        let old_output_index = entries
                                            .iter()
                                            .take(old_index)
                                            .filter(|entry| entry.value.is_some())
                                            .count();
                                        let entry = entries.remove(old_index);
                                        let published = entry.value.is_some();
                                        entries.insert(new_index, entry);
                                        let new_output_index = entries
                                            .iter()
                                            .take(new_index)
                                            .filter(|entry| entry.value.is_some())
                                            .count();
                                        if published && !replacing && old_output_index != new_output_index {
                                            output.push(VecDiff::Move {
                                                old_index: old_output_index,
                                                new_index: new_output_index,
                                            });
                                        }
                                    }
                                }
                                MapSignalQueueItem::Clear => {
                                    entries.clear();
                                    if !replacing {
                                        output.push(VecDiff::Clear);
                                    }
                                }
                                MapSignalQueueItem::Value { index, value } => {
                                    let entry = entries
                                        .get_mut(index)
                                        .unwrap_or_else(|| panic!("map_signal adapter emitted for a missing index"));
                                    let insertion = entry.insertion.take();
                                    let published = entry.value.is_some();
                                    entry.value = Some(value.clone());
                                    if !replacing {
                                        // Cancelled initialization leaves logical placeholders, not
                                        // output elements.
                                        let output_index =
                                            entries.iter().take(index).filter(|entry| entry.value.is_some()).count();
                                        output.push(if published {
                                            VecDiff::UpdateAt {
                                                index: output_index,
                                                value,
                                            }
                                        } else if matches!(insertion, Some(PendingInsertion::Push))
                                            && entries.iter().skip(index + 1).all(|entry| entry.value.is_none())
                                        {
                                            VecDiff::Push { value }
                                        } else {
                                            VecDiff::InsertAt {
                                                index: output_index,
                                                value,
                                            }
                                        });
                                    }
                                }
                            }
                        }
                        if replacing {
                            let values = entries
                                .iter()
                                .map(|entry| {
                                    entry.value.clone().unwrap_or_else(|| {
                                        panic!("map_signal replacement published before all values were ready")
                                    })
                                })
                                .collect();
                            Some(vec![VecDiff::Replace { values }])
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
        MapSignal {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`], only forwarding
    /// those which return `true`.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3, 4])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .filter(|In(x): In<i32>| x % 2 == 0); // outputs `SignalVec -> [2, 4]`
    /// ```
    fn filter<F, M>(self, predicate: F) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut predicate = IntoSystem::into_system(predicate);
            let mut predicate_initialized = false;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut indices: Local<Vec<bool>>| {
                    filter_generic_helper(diffs, &mut indices, |value| {
                        if !predicate_initialized {
                            predicate.initialize(world);
                            predicate_initialized = true;
                        }
                        if predicate.run(value.clone(), world).unwrap_or(false) {
                            Some(value)
                        } else {
                            None
                        }
                    })
                },
                OutputDelivery::move_only(),
            )
        });
        Filter { materializer }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`], transforming it
    /// and only forwarding those which return `Some`.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder().values(["1", "two", "NaN", "four", "5"])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .filter_map(|In(s): In<&'static str>| s.parse::<u32>().ok()); // outputs `SignalVec -> [1, 5]`
    /// ```
    fn filter_map<O, F, M>(self, system: F) -> FilterMap<Self, O>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
        O: Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, Option<O>, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut item_system = IntoSystem::into_system(system);
            let mut item_system_initialized = false;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut indices: Local<Vec<bool>>| {
                    filter_generic_helper(diffs, &mut indices, |value| {
                        if !item_system_initialized {
                            item_system.initialize(world);
                            item_system_initialized = true;
                        }
                        item_system.run(value, world).ok().flatten()
                    })
                },
                OutputDelivery::move_only(),
            )
        });
        FilterMap {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`] that produces a
    /// [`Signal<Item = bool>`], only forwarding those whose [`Signal`] outputs `true`.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone)]
    /// struct Even(bool);
    ///
    /// let mut world = World::new();
    /// MutableVec::builder().values([1, 2, 3, 4])
    ///      .spawn(&mut world)
    ///      .signal_vec()
    ///      .filter_signal(|In(x): In<i32>| {
    ///          signal::from_resource()
    ///              .map_in(move |even: Even| x % 2 == if even.0 { 0 } else { 1 })
    ///              .cache()
    ///      }); // outputs `SignalVec -> [2, 4]` when `Even(true)` and `SignalVec -> [1, 3]` when `Even(false)`
    /// ```
    fn filter_signal<F, S, M>(self, system: F) -> FilterSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
        S: super::signal::ReplayableSignal<Item = bool> + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        enum FilterSignalQueueItem<T> {
            Replace { values: Vec<T> },
            Insert { index: usize, value: T, push: bool },
            Update { index: usize, value: T },
            Remove { index: usize, pop: bool },
            Move { old_index: usize, new_index: usize },
            Clear,
            Predicate { index: usize, included: bool },
        }

        enum PendingFilterUpdate {
            None,
            Insert { push: bool },
            Update,
        }

        struct FilterEntry<T> {
            value: T,
            included: Option<bool>,
            pending: PendingFilterUpdate,
        }

        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut factory = IntoSystem::into_system(system);
            let mut initialized = false;
            let epoch = Arc::new(AtomicUsize::new(0));
            let manager_epoch = Arc::clone(&epoch);
            let manager = registrar.signal_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                      world: &mut World,
                      mut slot_count: Local<usize>,
                      mut observed_epoch: Local<usize>| {
                    if !initialized {
                        factory.initialize(world);
                        initialized = true;
                    }
                    let current_epoch = manager_epoch.load(AtomicOrdering::Acquire);
                    if *observed_epoch != current_epoch {
                        *observed_epoch = current_epoch;
                        *slot_count = 0;
                    }
                    let mut steps = Vec::new();
                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values } => {
                                let mut signals = Vec::with_capacity(values.len());
                                for value in values.iter().cloned() {
                                    signals.push(
                                        factory
                                            .run(value, world)
                                            .expect("filter_signal factory system failed to run"),
                                    );
                                }
                                *slot_count = signals.len();
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::ReplaceAll { signals },
                                    DynamicEmissions::One(FilterSignalQueueItem::Replace { values }),
                                ));
                            }
                            VecDiff::InsertAt { index, value } => {
                                let signal = factory
                                    .run(value.clone(), world)
                                    .expect("filter_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Insert { index, signal },
                                    DynamicEmissions::One(FilterSignalQueueItem::Insert {
                                        index,
                                        value,
                                        push: false,
                                    }),
                                ));
                                *slot_count += 1;
                            }
                            VecDiff::UpdateAt { index, value } => {
                                let signal = factory
                                    .run(value.clone(), world)
                                    .expect("filter_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Replace { index, signal },
                                    DynamicEmissions::One(FilterSignalQueueItem::Update { index, value }),
                                ));
                            }
                            VecDiff::RemoveAt { index } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Remove { index },
                                    DynamicEmissions::One(FilterSignalQueueItem::Remove { index, pop: false }),
                                ));
                                *slot_count = slot_count.saturating_sub(1);
                            }
                            VecDiff::Move { old_index, new_index } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Move { old_index, new_index },
                                    DynamicEmissions::One(FilterSignalQueueItem::Move { old_index, new_index }),
                                ));
                            }
                            VecDiff::Push { value } => {
                                let index = *slot_count;
                                let signal = factory
                                    .run(value.clone(), world)
                                    .expect("filter_signal factory system failed to run");
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Insert { index, signal },
                                    DynamicEmissions::One(FilterSignalQueueItem::Insert {
                                        index,
                                        value,
                                        push: true,
                                    }),
                                ));
                                *slot_count += 1;
                            }
                            VecDiff::Pop => {
                                if let Some(index) = slot_count.checked_sub(1) {
                                    steps.push(DynamicManagerStep::command(
                                        IndexedSlotCommand::Remove { index },
                                        DynamicEmissions::One(FilterSignalQueueItem::Remove { index, pop: true }),
                                    ));
                                    *slot_count = index;
                                }
                            }
                            VecDiff::Clear => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Clear,
                                    DynamicEmissions::One(FilterSignalQueueItem::Clear),
                                ));
                                *slot_count = 0;
                            }
                        }
                    }
                    Some(DynamicManagerUpdate::sequenced(steps))
                },
                OutputDelivery::move_only(),
            )?;
            let mut entries = Vec::<FilterEntry<Self::Item>>::new();
            let reducer_epoch = Arc::clone(&epoch);
            let mut reducer_observed_epoch = 0usize;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<bool>()
                    .adapter(|index, included| {
                        DynamicEmissions::One(FilterSignalQueueItem::Predicate { index, included })
                    })
                    .replayable_signal_vec_output(move |items| {
                        let current_epoch = reducer_epoch.load(AtomicOrdering::Acquire);
                        if reducer_observed_epoch != current_epoch {
                            reducer_observed_epoch = current_epoch;
                            entries.clear();
                        }
                        let mut output = Vec::new();
                        let mut replacing = false;
                        for item in items {
                            match item {
                                FilterSignalQueueItem::Replace { values } => {
                                    entries = values
                                        .into_iter()
                                        .map(|value| FilterEntry {
                                            value,
                                            included: None,
                                            pending: PendingFilterUpdate::None,
                                        })
                                        .collect();
                                    replacing = true;
                                }
                                FilterSignalQueueItem::Insert { index, value, push } => entries.insert(
                                    index,
                                    FilterEntry {
                                        value,
                                        included: None,
                                        pending: PendingFilterUpdate::Insert { push },
                                    },
                                ),
                                FilterSignalQueueItem::Update { index, value } => {
                                    let entry = entries
                                        .get_mut(index)
                                        .unwrap_or_else(|| panic!("filter_signal updated a missing index"));
                                    entry.value = value;
                                    // Membership describes the published value until the
                                    // replacement predicate arrives.
                                    if !matches!(entry.pending, PendingFilterUpdate::Insert { .. }) {
                                        entry.pending = PendingFilterUpdate::Update;
                                    }
                                }
                                FilterSignalQueueItem::Remove { index, pop } => {
                                    let filtered_index = entries
                                        .iter()
                                        .take(index)
                                        .filter(|entry| entry.included == Some(true))
                                        .count();
                                    if index < entries.len() {
                                        let removed = entries.remove(index);
                                        if removed.included == Some(true) && !replacing {
                                            output.push(if pop {
                                                VecDiff::Pop
                                            } else {
                                                VecDiff::RemoveAt { index: filtered_index }
                                            });
                                        }
                                    }
                                }
                                FilterSignalQueueItem::Move { old_index, new_index } => {
                                    if old_index < entries.len() && new_index < entries.len() {
                                        let old_filtered_index = entries
                                            .iter()
                                            .take(old_index)
                                            .filter(|entry| entry.included == Some(true))
                                            .count();
                                        let entry = entries.remove(old_index);
                                        let included = entry.included == Some(true);
                                        entries.insert(new_index, entry);
                                        let new_filtered_index = entries
                                            .iter()
                                            .take(new_index)
                                            .filter(|entry| entry.included == Some(true))
                                            .count();
                                        if included && !replacing && old_filtered_index != new_filtered_index {
                                            output.push(VecDiff::Move {
                                                old_index: old_filtered_index,
                                                new_index: new_filtered_index,
                                            });
                                        }
                                    }
                                }
                                FilterSignalQueueItem::Clear => {
                                    entries.clear();
                                    if !replacing {
                                        output.push(VecDiff::Clear);
                                    }
                                }
                                FilterSignalQueueItem::Predicate { index, included } => {
                                    let output_length =
                                        entries.iter().filter(|entry| entry.included == Some(true)).count();
                                    let filtered_index = entries
                                        .iter()
                                        .take(index)
                                        .filter(|entry| entry.included == Some(true))
                                        .count();
                                    let entry = entries.get_mut(index).unwrap_or_else(|| {
                                        panic!("filter_signal predicate emitted for a missing index")
                                    });
                                    let pending = core::mem::replace(&mut entry.pending, PendingFilterUpdate::None);
                                    let old_included = entry.included == Some(true);
                                    entry.included = Some(included);
                                    if replacing {
                                        continue;
                                    }
                                    match pending {
                                        PendingFilterUpdate::Insert { push } => {
                                            if included {
                                                output.push(if push && filtered_index == output_length {
                                                    VecDiff::Push {
                                                        value: entry.value.clone(),
                                                    }
                                                } else {
                                                    VecDiff::InsertAt {
                                                        index: filtered_index,
                                                        value: entry.value.clone(),
                                                    }
                                                });
                                            }
                                        }
                                        PendingFilterUpdate::Update => match (old_included, included) {
                                            (true, true) => output.push(VecDiff::UpdateAt {
                                                index: filtered_index,
                                                value: entry.value.clone(),
                                            }),
                                            (true, false) => output.push(VecDiff::RemoveAt { index: filtered_index }),
                                            (false, true) => output.push(VecDiff::InsertAt {
                                                index: filtered_index,
                                                value: entry.value.clone(),
                                            }),
                                            (false, false) => {}
                                        },
                                        PendingFilterUpdate::None if old_included != included => {
                                            if included {
                                                output.push(VecDiff::InsertAt {
                                                    index: filtered_index,
                                                    value: entry.value.clone(),
                                                });
                                            } else {
                                                output.push(VecDiff::RemoveAt { index: filtered_index });
                                            }
                                        }
                                        PendingFilterUpdate::None => {}
                                    }
                                }
                            }
                        }
                        if replacing {
                            Some(vec![VecDiff::Replace {
                                values: entries
                                    .iter()
                                    .filter(|entry| entry.included == Some(true))
                                    .map(|entry| entry.value.clone())
                                    .collect(),
                            }])
                        } else {
                            (!output.is_empty()).then_some(output)
                        }
                    })
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::Restart)
                    .on_deactivate(move || {
                        epoch.fetch_add(1, AtomicOrdering::AcqRel);
                    }),
            )
        });
        FilterSignal {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Transform each [`Item`](SignalVec::Item) into a tuple where the right item is the original
    /// item and the left item is a [`Signal<Item = Option<usize>>`] which outputs the item's index
    /// or [`None`] if it was removed. Note that the [`Signal<Item = Option<usize>>`]s are *not*
    /// deduped so consumers must call [`.dedupe`](crate::signal::SignalExt::dedupe) if resending
    /// the same index every frame would be spurious. Index signals must be registered and
    /// executed in the [`World`] that produced them.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::builder().values([1, 2, 5]).spawn(&mut world);
    /// let signal = vec.signal_vec().enumerate();
    /// // signal outputs `SignalVec -> [(Signal -> Some(0), 1), (Signal -> Some(1), 2), (Signal -> Some(2), 5)]`
    /// vec.write(&mut world).remove(1);
    /// // signal outputs `SignalVec -> [(Signal -> Some(0), 1), (Signal -> Some(1), 5)]`
    /// ```
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        #[derive(Default)]
        struct EnumerateState {
            lookup: Option<Arc<IndexLookupLifetime>>,
            ordered_keys: Vec<usize>,
            next_key: usize,
        }

        fn new_key(state: &mut EnumerateState, indices: &HashMap<usize, usize>) -> usize {
            loop {
                let key = state.next_key;
                state.next_key = state.next_key.wrapping_add(1);
                if !indices.contains_key(&key) {
                    return key;
                }
            }
        }

        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut state: Local<EnumerateState>| {
                    if diffs.is_empty() {
                        return None;
                    }
                    let lookup = state
                        .lookup
                        .get_or_insert_with(|| IndexLookupLifetime::new(world))
                        .clone();
                    let create_index_signal = |key| lookup.signal(key);
                    let mut output = Vec::new();
                    for diff in diffs {
                        let mut lookup_data = world.get_mut::<IndexLookup>(lookup.entity).expect("live index lookup");
                        let indices = &mut lookup_data.map;
                        match diff {
                            VecDiff::Replace { values } => {
                                indices.clear();
                                state.ordered_keys.clear();
                                let values = values
                                    .into_iter()
                                    .enumerate()
                                    .map(|(index, value)| {
                                        let key = new_key(&mut state, indices);
                                        state.ordered_keys.push(key);
                                        indices.insert(key, index);
                                        (create_index_signal(key), value)
                                    })
                                    .collect();
                                output.push(VecDiff::Replace { values });
                            }
                            VecDiff::InsertAt { index, value } => {
                                let key = new_key(&mut state, indices);
                                state.ordered_keys.insert(index, key);
                                for (index, key) in state.ordered_keys.iter().copied().enumerate().skip(index) {
                                    indices.insert(key, index);
                                }
                                output.push(VecDiff::InsertAt {
                                    index,
                                    value: (create_index_signal(key), value),
                                });
                            }
                            VecDiff::UpdateAt { index, value } => {
                                let key = state.ordered_keys[index];
                                output.push(VecDiff::UpdateAt {
                                    index,
                                    value: (create_index_signal(key), value),
                                });
                            }
                            VecDiff::RemoveAt { index } => {
                                let removed_key = state.ordered_keys.remove(index);
                                indices.remove(&removed_key);
                                for (index, key) in state.ordered_keys.iter().copied().enumerate().skip(index) {
                                    indices.insert(key, index);
                                }
                                output.push(VecDiff::RemoveAt { index });
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let key = state.ordered_keys.remove(old_index);
                                state.ordered_keys.insert(new_index, key);
                                let start = old_index.min(new_index);
                                let end = old_index.max(new_index) + 1;
                                for (index, key) in state
                                    .ordered_keys
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .skip(start)
                                    .take(end - start)
                                {
                                    indices.insert(key, index);
                                }
                                output.push(VecDiff::Move { old_index, new_index });
                            }
                            VecDiff::Push { value } => {
                                let key = new_key(&mut state, indices);
                                let index = state.ordered_keys.len();
                                state.ordered_keys.push(key);
                                indices.insert(key, index);
                                output.push(VecDiff::Push {
                                    value: (create_index_signal(key), value),
                                });
                            }
                            VecDiff::Pop => {
                                if let Some(key) = state.ordered_keys.pop() {
                                    indices.remove(&key);
                                    output.push(VecDiff::Pop);
                                }
                            }
                            VecDiff::Clear => {
                                state.ordered_keys.clear();
                                indices.clear();
                                output.push(VecDiff::Clear);
                            }
                        }
                    }
                    (!output.is_empty()).then_some(output)
                },
                OutputDelivery::move_only(),
            )
        });
        Enumerate { materializer }
    }

    /// Collect this [`SignalVec`]'s [`Item`](SignalVec::Item)s into a [`Vec`], transforming it into
    /// an [`Signal<Item = Vec<Item>>`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .to_signal()
    ///     .map_in(|vec: Vec<i32>| vec[1]); // outputs `2`
    /// ```
    fn to_signal(self) -> ToSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let signal = self.for_each(|In(diffs), mut values: Local<Vec<Self::Item>>| {
            apply_vec_diffs_strict(&mut values, diffs);
            values.clone()
        });
        ToSignal { signal }
    }

    /// Transform this [`SignalVec`] into a [`Signal<Item = bool>`] which outputs whether it's
    /// populated.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::builder().values([1]).spawn(&mut world);
    /// let signal = vec.signal_vec().is_empty();
    /// // `signal` outputs `false`
    /// vec.write(&mut world).pop();
    /// // `signal` outputs `true`
    /// ```
    #[allow(clippy::wrong_self_convention)]
    fn is_empty(self) -> IsEmpty<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        IsEmpty {
            signal: self.for_each(|In(diffs), mut len: Local<usize>| {
                apply_len_diffs(&mut len, diffs);
                *len == 0
            }),
        }
    }

    /// Transform this [`SignalVec`] into a [`Signal<Item = usize>`] which outputs its length.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::builder().values([1]).spawn(&mut world);
    /// let signal = vec.signal_vec().len();
    /// // `signal` outputs `1`
    /// vec.write(&mut world).pop();
    /// // `signal` outputs `0`
    /// ```
    fn len(self) -> Len<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        Len {
            signal: self.for_each(|In(diffs), mut len: Local<usize>| {
                apply_len_diffs(&mut len, diffs);
                *len
            }),
        }
    }

    /// Transform this [`SignalVec`] into a [`Signal<Item = Self::Item>`] which outputs its sum.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
    /// let signal = vec.signal_vec().sum();
    /// // `signal` outputs `6`
    /// vec.write(&mut world).push(4);
    /// // `signal` outputs `10`
    /// ```
    fn sum(self) -> Sum<Self>
    where
        Self: Sized,
        Self::Item: for<'a> core::iter::Sum<&'a Self::Item> + Send + Sync + 'static,
    {
        Sum {
            signal: self.for_each(|In(diffs), mut values: Local<Vec<Self::Item>>| {
                apply_vec_diffs_strict(&mut values, diffs);
                values.iter().sum::<Self::Item>()
            }),
        }
    }

    /// Chains this [`SignalVec`] with another [`SignalVec`], producing a [`SignalVec`] with all the
    /// [`Item`](SignalVec::Item)s of `self`, followed by all the [`Item`](SignalVec::Item)s of
    /// `other`.
    ///
    /// After this chain is deactivated and reactivated, its next produced batch restores its last
    /// authoritative published baseline before applying new diffs. This can repair output canceled
    /// before downstream consumption, but does not make the chain replayable: a silent source or a
    /// newly attached branch is not guaranteed an immediate snapshot. Incremental-only history
    /// cannot supply an authoritative baseline.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .chain(
    ///         MutableVec::builder()
    ///             .values([2, 4])
    ///             .spawn(&mut world)
    ///             .signal_vec(),
    ///     ); // outputs `SignalVec -> [1, 3, 2, 4]`
    /// ```
    fn chain<S>(self, other: S) -> Chain<Self, S>
    where
        S: SignalVec<Item = Self::Item>,
        Self: Sized,
        Self::Item: Send + Sync + 'static + Clone,
    {
        let left = Box::new(self).into_signal_vec_materializer();
        let right = Box::new(other).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let left = registrar.materialize_signal_vec(left)?;
            let right = registrar.materialize_signal_vec(right)?;
            let left =
                registrar.signal_entity(left, |In(diffs)| ChainInput::Left(diffs), OutputDelivery::move_only())?;
            let right =
                registrar.signal_entity(right, |In(diffs)| ChainInput::Right(diffs), OutputDelivery::move_only())?;
            let mut state = ChainState::default();
            registrar.ordered_chain_consumer(
                [left, right],
                move |input, input_epoch| state.transform(input, input_epoch),
                OutputDelivery::move_only(),
            )
        });

        Chain {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Places a [`Clone`] of `separator` between adjacent items of this [`SignalVec`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .intersperse(0); // outputs `SignalVec -> [1, 0, 2, 0, 3]`
    /// ```
    fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>, mut local_values: Local<Vec<Self::Item>>| {
                    let mut out_diffs = Vec::new();
                    for diff in diffs {
                        let old_len = local_values.len();
                        match diff {
                            VecDiff::Replace { values } => {
                                // Correctly set the new state first.
                                *local_values = values;

                                let mut interspersed = Vec::new();
                                if !local_values.is_empty() {
                                    interspersed.reserve(2 * local_values.len() - 1);

                                    let mut iter = local_values.iter().cloned();
                                    interspersed.push(iter.next().unwrap());
                                    for item in iter {
                                        interspersed.push(separator.clone());
                                        interspersed.push(item);
                                    }
                                }
                                out_diffs.push(VecDiff::Replace { values: interspersed });
                            }
                            VecDiff::InsertAt { index, value } => {
                                local_values.insert(index, value.clone());
                                if old_len == 0 {
                                    // Inserting the very first item.
                                    out_diffs.push(VecDiff::Push { value });
                                } else if index == old_len {
                                    // Appending to the end. This is a special case.
                                    out_diffs.push(VecDiff::Push {
                                        value: separator.clone(),
                                    });
                                    out_diffs.push(VecDiff::Push { value });
                                } else {
                                    // Inserting at the beginning or in the middle.
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: 2 * index,
                                        value,
                                    });
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: 2 * index + 1,
                                        value: separator.clone(),
                                    });
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
                                    // Removing the only item.
                                    out_diffs.push(VecDiff::RemoveAt { index: 0 });
                                } else if index == old_len - 1 {
                                    // Removing the last item: remove the item and the separator
                                    // before it.
                                    out_diffs.push(VecDiff::Pop); // item
                                    out_diffs.push(VecDiff::Pop); // separator
                                } else {
                                    // Removing from start/middle: remove the item and the separator
                                    // after it.
                                    out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // item
                                    out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // separator
                                }
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let value = local_values.remove(old_index);
                                local_values.insert(new_index, value);

                                // The most robust way to handle move is to recalculate via Replace,
                                // as managing indices of items and separators during a move is
                                // complex.
                                let mut interspersed = Vec::new();
                                if !local_values.is_empty() {
                                    let mut iter = local_values.iter().cloned();
                                    interspersed.push(iter.next().unwrap());
                                    for item in iter {
                                        interspersed.push(separator.clone());
                                        interspersed.push(item);
                                    }
                                }
                                out_diffs.push(VecDiff::Replace { values: interspersed });
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
                            VecDiff::Pop => {
                                local_values.pop();
                                if old_len > 0 {
                                    out_diffs.push(VecDiff::Pop); // The item
                                }
                                if old_len > 1 {
                                    out_diffs.push(VecDiff::Pop); // The separator before it
                                }
                            }
                            VecDiff::Clear => {
                                local_values.clear();
                                out_diffs.push(VecDiff::Clear);
                            }
                        }
                    }
                    if out_diffs.is_empty() { None } else { Some(out_diffs) }
                },
                OutputDelivery::move_only(),
            )
        });
        Intersperse { materializer }
    }

    /// Place the [`Item`](SignalVec::Item) output by the [`System`] between adjacent items of this
    /// [`SignalVec`]; the [`System`] takes [`In`] a [`Broadcast<Option<usize>>`] which outputs the
    /// separator's 0-based ordinal separator index (`0..len - 1`, where separator `i` sits between
    /// item `i` and item `i + 1` at interspersed output index `2 * i + 1`) or [`None`] if it has
    /// been removed. Note that the [`Broadcast<Option<usize>>`] signals are *not* deduped so
    /// consumers must call [`.dedupe`](crate::signal::SignalExt::dedupe) if resending the same
    /// index every frame would be spurious. Index signals must be registered and executed in the
    /// [`World`] that produced them.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([1, 2, 3])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .intersperse_with(|_: In<Broadcast<Option<usize>>>| 0); // outputs `SignalVec -> [1, 0, 2, 0, 3]`
    /// ```
    fn intersperse_with<F, M>(self, separator_system: F) -> IntersperseWith<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Broadcast<Option<usize>>>, Self::Item, M> + Send + Sync + 'static,
    {
        struct IntersperseState<T> {
            lookup: Option<Arc<IndexLookupLifetime>>,
            values: Vec<T>,
            separator_keys: Vec<usize>,
            separators: Vec<T>,
            next_key: usize,
        }

        impl<T> Default for IntersperseState<T> {
            fn default() -> Self {
                Self {
                    lookup: None,
                    values: Vec::new(),
                    separator_keys: Vec::new(),
                    separators: Vec::new(),
                    next_key: 0,
                }
            }
        }

        fn next_key<T>(state: &mut IntersperseState<T>) -> usize {
            let key = state.next_key;
            state.next_key = state.next_key.wrapping_add(1);
            key
        }

        fn interspersed_values<T: Clone>(state: &IntersperseState<T>) -> Vec<T> {
            let mut values = Vec::with_capacity(state.values.len().saturating_mul(2).saturating_sub(1));
            if let Some(first) = state.values.first().cloned() {
                values.push(first);
                for (separator, value) in state
                    .separators
                    .iter()
                    .cloned()
                    .zip(state.values.iter().skip(1).cloned())
                {
                    values.push(separator);
                    values.push(value);
                }
            }
            values
        }

        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;

            let mut separator_system = IntoSystem::into_system(separator_system);
            let mut separator_system_initialized = false;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                      world: &mut World,
                      mut state: Local<IntersperseState<Self::Item>>| {
                    if diffs.is_empty() {
                        return None;
                    }
                    let lookup = state
                        .lookup
                        .get_or_insert_with(|| IndexLookupLifetime::new(world))
                        .clone();
                    let mut create_separator = |key, world: &mut World| {
                        if !separator_system_initialized {
                            separator_system.initialize(world);
                            separator_system_initialized = true;
                        }
                        separator_system.run(lookup.signal(key), world).unwrap()
                    };
                    let mut output = Vec::new();

                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values } => {
                                state.values = values;
                                state.separator_keys.clear();
                                state.separators.clear();
                                let separator_count = state.values.len().saturating_sub(1);
                                for _ in 0..separator_count {
                                    let key = next_key(&mut state);
                                    state.separator_keys.push(key);
                                }
                                lookup.synchronize(world, &state.separator_keys);
                                for key in state.separator_keys.clone() {
                                    state.separators.push(create_separator(key, world));
                                }
                                output.push(VecDiff::Replace {
                                    values: interspersed_values(&state),
                                });
                            }
                            VecDiff::InsertAt { index, value } => {
                                let old_length = state.values.len();
                                state.values.insert(index, value.clone());
                                let separator_index = if old_length == 0 {
                                    None
                                } else if index == old_length {
                                    Some(old_length - 1)
                                } else {
                                    Some(index)
                                };
                                if let Some(separator_index) = separator_index {
                                    let key = next_key(&mut state);
                                    state.separator_keys.insert(separator_index, key);
                                    lookup.synchronize(world, &state.separator_keys);
                                    let separator = create_separator(key, world);
                                    state.separators.insert(separator_index, separator.clone());
                                    if index == old_length {
                                        output.push(VecDiff::Push { value: separator });
                                        output.push(VecDiff::Push { value });
                                    } else {
                                        output.push(VecDiff::InsertAt {
                                            index: 2 * index,
                                            value,
                                        });
                                        output.push(VecDiff::InsertAt {
                                            index: 2 * index + 1,
                                            value: separator,
                                        });
                                    }
                                } else {
                                    output.push(VecDiff::Push { value });
                                }
                            }
                            VecDiff::UpdateAt { index, value } => {
                                state.values[index] = value.clone();
                                output.push(VecDiff::UpdateAt {
                                    index: 2 * index,
                                    value,
                                });
                            }
                            VecDiff::RemoveAt { index } => {
                                let old_length = state.values.len();
                                state.values.remove(index);
                                if old_length > 1 {
                                    let separator_index = if index + 1 == old_length { index - 1 } else { index };
                                    state.separator_keys.remove(separator_index);
                                    state.separators.remove(separator_index);
                                    lookup.synchronize(world, &state.separator_keys);
                                }
                                if old_length == 1 {
                                    output.push(VecDiff::RemoveAt { index: 0 });
                                } else if index + 1 == old_length {
                                    output.push(VecDiff::Pop);
                                    output.push(VecDiff::Pop);
                                } else {
                                    output.push(VecDiff::RemoveAt { index: 2 * index });
                                    output.push(VecDiff::RemoveAt { index: 2 * index });
                                }
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let value = state.values.remove(old_index);
                                state.values.insert(new_index, value);
                                state.separator_keys.clear();
                                state.separators.clear();
                                let separator_count = state.values.len().saturating_sub(1);
                                for _ in 0..separator_count {
                                    let key = next_key(&mut state);
                                    state.separator_keys.push(key);
                                }
                                lookup.synchronize(world, &state.separator_keys);
                                for key in state.separator_keys.clone() {
                                    state.separators.push(create_separator(key, world));
                                }
                                output.push(VecDiff::Replace {
                                    values: interspersed_values(&state),
                                });
                            }
                            VecDiff::Push { value } => {
                                let old_length = state.values.len();
                                state.values.push(value.clone());
                                if old_length > 0 {
                                    let key = next_key(&mut state);
                                    state.separator_keys.push(key);
                                    lookup.synchronize(world, &state.separator_keys);
                                    let separator = create_separator(key, world);
                                    state.separators.push(separator.clone());
                                    output.push(VecDiff::Push { value: separator });
                                }
                                output.push(VecDiff::Push { value });
                            }
                            VecDiff::Pop => {
                                let old_length = state.values.len();
                                if state.values.pop().is_some() && old_length > 1 {
                                    state.separator_keys.pop();
                                    state.separators.pop();
                                    lookup.synchronize(world, &state.separator_keys);
                                }
                                if old_length > 0 {
                                    output.push(VecDiff::Pop);
                                }
                                if old_length > 1 {
                                    output.push(VecDiff::Pop);
                                }
                            }
                            VecDiff::Clear => {
                                state.values.clear();
                                state.separator_keys.clear();
                                state.separators.clear();
                                lookup.synchronize(world, &state.separator_keys);
                                output.push(VecDiff::Clear);
                            }
                        }
                    }

                    (!output.is_empty()).then_some(output)
                },
                OutputDelivery::move_only(),
            )
        });
        IntersperseWith { materializer }
    }

    /// Sort this [`SignalVec`] according to a [`System`] which takes [`In`] `(Self::Item,
    /// Self::Item)` and returns a [`core::cmp::Ordering`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder().values([3, 2, 1]).spawn(&mut world).signal_vec().sort_by(|In((left, right)): In<(i32, i32)>| left.cmp(&right)); // outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn sort_by<F, M>(self, compare_system: F) -> SortBy<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<(Self::Item, Self::Item)>, Ordering, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut compare_system = IntoSystem::into_system(compare_system);
            let mut compare_system_initialized = false;

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

            fn search<T, System>(
                world: &mut World,
                system: &mut System,
                state: &SortState<T>,
                item_index: usize,
            ) -> Result<usize, usize>
            where
                T: Clone + Send + Sync + 'static,
                System: bevy_ecs::system::System<In = In<(T, T)>, Out = Ordering>,
            {
                state.sorted_indices.binary_search_by(|&probe_index| {
                    let left = state.values[probe_index].clone();
                    let right = state.values[item_index].clone();
                    system
                        .run((left, right), world)
                        .unwrap_or_else(|err| panic!("SignalVec::sort_by comparator system failed or skipped ({err})"))
                        .then_with(|| probe_index.cmp(&item_index))
                })
            }

            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                      world: &mut World,
                      mut state: Local<SortState<Self::Item>>|
                      -> Option<Vec<VecDiff<Self::Item>>> {
                    if !compare_system_initialized {
                        compare_system.initialize(world);
                        compare_system_initialized = true;
                    }
                    let mut out_diffs = Vec::new();
                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values: new_values } => {
                                state.values = new_values;
                                state.sorted_indices = (0..state.values.len()).collect();

                                let values_for_sort = state.values.clone();
                                state.sorted_indices.sort_unstable_by(|&a, &b| {
                                    let val_a = values_for_sort[a].clone();
                                    let val_b = values_for_sort[b].clone();
                                    compare_system
                                        .run((val_a, val_b), world)
                                        .unwrap_or_else(|err| {
                                            panic!("SignalVec::sort_by comparator system failed or skipped ({err})")
                                        })
                                        .then_with(|| a.cmp(&b))
                                });
                                let sorted_values =
                                    state.sorted_indices.iter().map(|&i| state.values[i].clone()).collect();
                                out_diffs.push(VecDiff::Replace { values: sorted_values });
                            }
                            VecDiff::Push { value } => {
                                let new_idx = state.values.len();
                                state.values.push(value.clone());
                                let insert_pos = search(world, &mut compare_system, &*state, new_idx).unwrap_err();
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
                                let insert_pos = search(world, &mut compare_system, &*state, index).unwrap_err();
                                state.sorted_indices.insert(insert_pos, index);
                                out_diffs.push(VecDiff::InsertAt {
                                    index: insert_pos,
                                    value,
                                });
                            }
                            VecDiff::RemoveAt { index } => {
                                // Must search _before_ removing from values.
                                let remove_pos = search(world, &mut compare_system, &*state, index).unwrap();
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
                                // 1. Find the old sorted position before any changes.
                                let old_pos = search(world, &mut compare_system, &*state, index).unwrap();

                                // 2. Remove the old entry from sorted tracking.
                                state.sorted_indices.remove(old_pos);

                                // 3. Update the underlying value in our source-of-truth vec.
                                state.values[index] = value.clone();

                                // 4. Find where the *new* value should be inserted in the now N-1
                                //    element list.
                                let new_pos = search(world, &mut compare_system, &*state, index).unwrap_err();

                                // 5. Re-insert the item's index into our tracking at the new
                                //    position.
                                state.sorted_indices.insert(new_pos, index);

                                // 6. If the item's final sorted position is the same, emit a single
                                //    UpdateAt. Otherwise, emit a move as RemoveAt + InsertAt.
                                if new_pos == old_pos {
                                    out_diffs.push(VecDiff::UpdateAt { index: old_pos, value });
                                } else {
                                    out_diffs.push(VecDiff::RemoveAt { index: old_pos });
                                    out_diffs.push(VecDiff::InsertAt { index: new_pos, value });
                                }
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let old_sorted_pos = search(world, &mut compare_system, &*state, old_index).unwrap();

                                let val = state.values.remove(old_index);
                                state.values.insert(new_index, val);
                                for i in state.sorted_indices.iter_mut() {
                                    if *i == old_index {
                                        *i = new_index;
                                    } else if old_index < new_index && *i > old_index && *i <= new_index {
                                        *i -= 1;
                                    } else if new_index < old_index && *i >= new_index && *i < old_index {
                                        *i += 1;
                                    }
                                }

                                let moved_source_index = state.sorted_indices.remove(old_sorted_pos);
                                debug_assert_eq!(moved_source_index, new_index);
                                let new_sorted_pos =
                                    search(world, &mut compare_system, &*state, new_index).unwrap_err();
                                state.sorted_indices.insert(new_sorted_pos, moved_source_index);

                                if old_sorted_pos != new_sorted_pos {
                                    out_diffs.push(VecDiff::Move {
                                        old_index: old_sorted_pos,
                                        new_index: new_sorted_pos,
                                    });
                                }
                            }
                            VecDiff::Pop => {
                                let index = state.values.len() - 1;
                                let remove_pos = search(world, &mut compare_system, &*state, index).unwrap();
                                state.sorted_indices.remove(remove_pos);
                                state.values.pop();
                                out_diffs.push(VecDiff::RemoveAt { index: remove_pos });
                            }
                            VecDiff::Clear => {
                                state.values.clear();
                                state.sorted_indices.clear();
                                out_diffs.push(VecDiff::Clear);
                            }
                        }
                    }
                    if out_diffs.is_empty() { None } else { Some(out_diffs) }
                },
                OutputDelivery::move_only(),
            )
        });
        SortBy { materializer }
    }

    /// Sorts this [`SignalVec`] according to its [`Item`](SignalVec::Item)'s [`Ord`]
    /// implementation.
    ///
    /// This is a convenience method that is equivalent to `.sort_by(|In((a, b))| a.cmp(b))`.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([3, 2, 1])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .sort_by_cmp(); // outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn sort_by_cmp(self) -> SortBy<Self>
    where
        Self: Sized,
        Self::Item: Ord + Clone + Send + Sync + 'static,
    {
        self.sort_by(|In((left, right)): In<(Self::Item, Self::Item)>| left.cmp(&right))
    }

    /// Sorts this [`SignalVec`] with a key extraction [`System`] which takes [`In`] an
    /// [`Item`](SignalVec::Item) and returns a key `K` that implements [`Ord`] which the output
    /// will be sorted by.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// MutableVec::builder()
    ///     .values([3, 2, 1])
    ///     .spawn(&mut world)
    ///     .signal_vec()
    ///     .sort_by_key(|In(x): In<i32>| -x); // outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn sort_by_key<K, F, M>(self, system: F) -> SortByKey<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
        K: Ord + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, K, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let mut key_system = IntoSystem::into_system(system);
            let mut key_system_initialized = false;

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

            registrar.signal_vec_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                      world: &mut World,
                      mut state: Local<SortState<Self::Item, K>>|
                      -> Option<Vec<VecDiff<Self::Item>>> {
                    if !key_system_initialized {
                        key_system.initialize(world);
                        key_system_initialized = true;
                    }
                    let mut out_diffs = Vec::new();
                    let mut get_key = |world: &mut World, value: Self::Item| key_system.run(value, world).unwrap();
                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values: new_values } => {
                                state.values = new_values;
                                state.keys = state.values.iter().map(|v| get_key(world, v.clone())).collect();
                                state.sorted_indices = (0..state.values.len()).collect();

                                let SortState {
                                    values,
                                    keys,
                                    sorted_indices,
                                } = &mut *state;
                                sorted_indices.sort_unstable_by(|&a, &b| keys[a].cmp(&keys[b]).then_with(|| a.cmp(&b)));
                                let sorted_values = sorted_indices.iter().map(|&i| values[i].clone()).collect();
                                out_diffs.push(VecDiff::Replace { values: sorted_values });
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
                                // 1. Find the old sorted position before any changes.
                                let old_pos = search(&*state, index).unwrap();

                                // 2. Remove the old entry from sorted tracking.
                                state.sorted_indices.remove(old_pos);

                                // 3. Update the underlying value and its key in our source-of-truth
                                //    vecs.
                                let new_key = get_key(world, value.clone());
                                state.values[index] = value.clone();
                                state.keys[index] = new_key;

                                // 4. Find where the *new* value should be inserted in the now N-1
                                //    element list.
                                let new_pos = search(&*state, index).unwrap_err();

                                // 5. Re-insert the item's index into our tracking at the new
                                //    position.
                                state.sorted_indices.insert(new_pos, index);

                                // 6. If the item's final sorted position is the same, emit a single
                                //    UpdateAt. Otherwise, emit a move as RemoveAt + InsertAt.
                                if old_pos == new_pos {
                                    out_diffs.push(VecDiff::UpdateAt { index: old_pos, value });
                                } else {
                                    out_diffs.push(VecDiff::RemoveAt { index: old_pos });
                                    out_diffs.push(VecDiff::InsertAt { index: new_pos, value });
                                }
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let old_sorted_pos = search(&*state, old_index).unwrap();

                                let val = state.values.remove(old_index);
                                state.values.insert(new_index, val);
                                let key = state.keys.remove(old_index);
                                state.keys.insert(new_index, key);
                                for i in state.sorted_indices.iter_mut() {
                                    if *i == old_index {
                                        *i = new_index;
                                    } else if old_index < new_index && *i > old_index && *i <= new_index {
                                        *i -= 1;
                                    } else if new_index < old_index && *i >= new_index && *i < old_index {
                                        *i += 1;
                                    }
                                }

                                let moved_source_index = state.sorted_indices.remove(old_sorted_pos);
                                debug_assert_eq!(moved_source_index, new_index);
                                let new_sorted_pos = search(&*state, new_index).unwrap_err();
                                state.sorted_indices.insert(new_sorted_pos, moved_source_index);

                                if old_sorted_pos != new_sorted_pos {
                                    out_diffs.push(VecDiff::Move {
                                        old_index: old_sorted_pos,
                                        new_index: new_sorted_pos,
                                    });
                                }
                            }
                            VecDiff::Pop => {
                                let index = state.values.len() - 1;
                                let remove_pos = search(&*state, index).unwrap();
                                state.sorted_indices.remove(remove_pos);
                                state.values.pop();
                                state.keys.pop();
                                out_diffs.push(VecDiff::RemoveAt { index: remove_pos });
                            }
                            VecDiff::Clear => {
                                state.values.clear();
                                state.keys.clear();
                                state.sorted_indices.clear();
                                out_diffs.push(VecDiff::Clear);
                            }
                        }
                    }
                    if out_diffs.is_empty() { None } else { Some(out_diffs) }
                },
                OutputDelivery::move_only(),
            )
        });
        SortByKey { materializer }
    }

    /// Flatten a [`SignalVec`] of replayable [`SignalVec`]s into a single contiguous [`SignalVec`],
    /// translating both outer structural mutations and inner vector mutations into exact
    /// prefix-offset [`VecDiff`]s.
    ///
    /// Each inner vector must implement [`ReplayableSignalVec`]; normalize non-replayable inner
    /// chains with [`.cache()`](SignalVecExt::cache) first. Re-selecting an inner vector with the
    /// same canonical identity preserves its active subscription without re-emitting a redundant
    /// snapshot.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_app::prelude::*;
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut app = App::new();
    /// app.add_plugins(JonmoPlugin::default());
    ///
    /// #[derive(Resource, Default)]
    /// struct Log(Vec<i32>);
    /// app.init_resource::<Log>();
    ///
    /// let a = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    /// let b = MutableVec::builder().values([3]).spawn(app.world_mut());
    /// let outer = MutableVec::builder()
    ///     .values([a.clone(), b.clone()])
    ///     .spawn(app.world_mut());
    ///
    /// let _claim = outer
    ///     .signal_vec()
    ///     .map_in(|inner: MutableVec<i32>| inner.signal_vec())
    ///     .flatten()
    ///     .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, mut log: ResMut<Log>| {
    ///         for diff in diffs {
    ///             diff.apply_to_vec(&mut log.0);
    ///         }
    ///     })
    ///     .register(app.world_mut());
    ///
    /// app.update();
    /// assert_eq!(app.world().resource::<Log>().0, vec![1, 2, 3]);
    /// ```
    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: ReplayableSignalVec + 'static,
        <Self::Item as SignalVec>::Item: Clone + Send + Sync + 'static,
    {
        type InnerItem<Upstream> = <<Upstream as SignalVec>::Item as SignalVec>::Item;

        enum FlattenQueueItem<T> {
            Replace { length: usize },
            Insert { index: usize },
            Remove { index: usize },
            Move { old_index: usize, new_index: usize },
            Clear,
            InnerDiffs { index: usize, diffs: Vec<VecDiff<T>> },
        }

        struct FlattenSlot<T> {
            values: Vec<T>,
            published: bool,
        }

        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            let epoch = Arc::new(AtomicUsize::new(0));
            let manager_epoch = Arc::clone(&epoch);
            let manager = registrar.signal_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                      mut slot_count: Local<usize>,
                      mut observed_epoch: Local<usize>| {
                    let current_epoch = manager_epoch.load(AtomicOrdering::Acquire);
                    if *observed_epoch != current_epoch {
                        *observed_epoch = current_epoch;
                        *slot_count = 0;
                    }
                    let mut steps = Vec::new();
                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values: signals } => {
                                *slot_count = signals.len();
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::ReplaceAll { signals },
                                    DynamicEmissions::One(FlattenQueueItem::Replace { length: *slot_count }),
                                ));
                            }
                            VecDiff::InsertAt { index, value: signal } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Insert { index, signal },
                                    DynamicEmissions::One(FlattenQueueItem::Insert { index }),
                                ));
                                *slot_count += 1;
                            }
                            VecDiff::UpdateAt { index, value: signal } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Replace { index, signal },
                                    DynamicEmissions::None,
                                ));
                            }
                            VecDiff::RemoveAt { index } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Remove { index },
                                    DynamicEmissions::One(FlattenQueueItem::Remove { index }),
                                ));
                                *slot_count = slot_count.saturating_sub(1);
                            }
                            VecDiff::Move { old_index, new_index } => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Move { old_index, new_index },
                                    DynamicEmissions::One(FlattenQueueItem::Move { old_index, new_index }),
                                ));
                            }
                            VecDiff::Push { value: signal } => {
                                let index = *slot_count;
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Insert { index, signal },
                                    DynamicEmissions::One(FlattenQueueItem::Insert { index }),
                                ));
                                *slot_count += 1;
                            }
                            VecDiff::Pop => {
                                if let Some(index) = slot_count.checked_sub(1) {
                                    steps.push(DynamicManagerStep::command(
                                        IndexedSlotCommand::Remove { index },
                                        DynamicEmissions::One(FlattenQueueItem::Remove { index }),
                                    ));
                                    *slot_count = index;
                                }
                            }
                            VecDiff::Clear => {
                                steps.push(DynamicManagerStep::command(
                                    IndexedSlotCommand::Clear,
                                    DynamicEmissions::One(FlattenQueueItem::Clear),
                                ));
                                *slot_count = 0;
                            }
                        }
                    }
                    Some(DynamicManagerUpdate::sequenced(steps))
                },
                OutputDelivery::move_only(),
            )?;
            let mut entries = Vec::<FlattenSlot<InnerItem<Self>>>::new();
            let reducer_epoch = Arc::clone(&epoch);
            let mut reducer_observed_epoch = 0usize;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_signal_vec::<InnerItem<Self>>()
                    .adapter(|index, diffs| DynamicEmissions::One(FlattenQueueItem::InnerDiffs { index, diffs }))
                    .replayable_signal_vec_output(move |items| {
                        let current_epoch = reducer_epoch.load(AtomicOrdering::Acquire);
                        if reducer_observed_epoch != current_epoch {
                            reducer_observed_epoch = current_epoch;
                            entries.clear();
                        }
                        let mut output = Vec::new();
                        let mut replacing = false;
                        for item in items {
                            match item {
                                FlattenQueueItem::Replace { length } => {
                                    let mut replacement = Vec::with_capacity(length);
                                    for index in 0..length {
                                        replacement.push(match entries.get(index) {
                                            Some(existing) if existing.published => FlattenSlot {
                                                values: existing.values.clone(),
                                                published: true,
                                            },
                                            _ => FlattenSlot {
                                                values: Vec::new(),
                                                published: false,
                                            },
                                        });
                                    }
                                    entries = replacement;
                                    replacing = true;
                                }
                                FlattenQueueItem::Insert { index } => {
                                    entries.insert(
                                        index,
                                        FlattenSlot {
                                            values: Vec::new(),
                                            published: false,
                                        },
                                    );
                                }
                                FlattenQueueItem::Remove { index } => {
                                    if index < entries.len() {
                                        let base: usize = entries[..index].iter().map(|slot| slot.values.len()).sum();
                                        let removed = entries.remove(index);
                                        if !replacing {
                                            for _ in 0..removed.values.len() {
                                                output.push(VecDiff::RemoveAt { index: base });
                                            }
                                        }
                                    }
                                }
                                FlattenQueueItem::Move { old_index, new_index } => {
                                    if old_index < entries.len() && new_index < entries.len() {
                                        let old_base: usize =
                                            entries[..old_index].iter().map(|slot| slot.values.len()).sum();
                                        let slot = entries.remove(old_index);
                                        let slot_len = slot.values.len();
                                        entries.insert(new_index, slot);
                                        let new_base: usize =
                                            entries[..new_index].iter().map(|slot| slot.values.len()).sum();
                                        if !replacing && slot_len > 0 && old_base != new_base {
                                            if old_base < new_base {
                                                for _ in 0..slot_len {
                                                    output.push(VecDiff::Move {
                                                        old_index: old_base,
                                                        new_index: new_base + slot_len - 1,
                                                    });
                                                }
                                            } else {
                                                for offset in 0..slot_len {
                                                    output.push(VecDiff::Move {
                                                        old_index: old_base + offset,
                                                        new_index: new_base + offset,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                FlattenQueueItem::Clear => {
                                    let had_items = entries.iter().any(|slot| !slot.values.is_empty());
                                    entries.clear();
                                    if !replacing && had_items {
                                        output.push(VecDiff::Clear);
                                    }
                                }
                                FlattenQueueItem::InnerDiffs { index, diffs } => {
                                    let base: usize = entries[..index].iter().map(|slot| slot.values.len()).sum();
                                    let slot = entries
                                        .get_mut(index)
                                        .unwrap_or_else(|| panic!("flatten adapter emitted for a missing index"));
                                    for diff in diffs {
                                        match diff {
                                            VecDiff::Replace { values } => {
                                                if replacing {
                                                    slot.values = values;
                                                    slot.published = true;
                                                } else if !slot.published {
                                                    for (offset, value) in values.iter().cloned().enumerate() {
                                                        output.push(VecDiff::InsertAt {
                                                            index: base + offset,
                                                            value,
                                                        });
                                                    }
                                                    slot.values = values;
                                                    slot.published = true;
                                                } else {
                                                    let old_len = slot.values.len();
                                                    let new_len = values.len();
                                                    let common = old_len.min(new_len);
                                                    for (offset, value) in
                                                        values.iter().take(common).cloned().enumerate()
                                                    {
                                                        output.push(VecDiff::UpdateAt {
                                                            index: base + offset,
                                                            value,
                                                        });
                                                    }
                                                    if new_len > old_len {
                                                        for (offset, value) in values.iter().enumerate().skip(common) {
                                                            output.push(VecDiff::InsertAt {
                                                                index: base + offset,
                                                                value: value.clone(),
                                                            });
                                                        }
                                                    } else if old_len > new_len {
                                                        for _ in common..old_len {
                                                            output.push(VecDiff::RemoveAt { index: base + common });
                                                        }
                                                    }
                                                    slot.values = values;
                                                }
                                            }
                                            VecDiff::InsertAt {
                                                index: inner_index,
                                                value,
                                            } => {
                                                slot.values.insert(inner_index, value.clone());
                                                if !replacing {
                                                    output.push(VecDiff::InsertAt {
                                                        index: base + inner_index,
                                                        value,
                                                    });
                                                }
                                            }
                                            VecDiff::Push { value } => {
                                                let inner_index = slot.values.len();
                                                slot.values.push(value.clone());
                                                if !replacing {
                                                    output.push(VecDiff::InsertAt {
                                                        index: base + inner_index,
                                                        value,
                                                    });
                                                }
                                            }
                                            VecDiff::UpdateAt {
                                                index: inner_index,
                                                value,
                                            } => {
                                                slot.values[inner_index] = value.clone();
                                                if !replacing {
                                                    output.push(VecDiff::UpdateAt {
                                                        index: base + inner_index,
                                                        value,
                                                    });
                                                }
                                            }
                                            VecDiff::RemoveAt { index: inner_index } => {
                                                slot.values.remove(inner_index);
                                                if !replacing {
                                                    output.push(VecDiff::RemoveAt {
                                                        index: base + inner_index,
                                                    });
                                                }
                                            }
                                            VecDiff::Pop => {
                                                if slot.values.pop().is_some() && !replacing {
                                                    let inner_index = slot.values.len();
                                                    output.push(VecDiff::RemoveAt {
                                                        index: base + inner_index,
                                                    });
                                                }
                                            }
                                            VecDiff::Move {
                                                old_index: inner_old,
                                                new_index: inner_new,
                                            } => {
                                                let value = slot.values.remove(inner_old);
                                                slot.values.insert(inner_new, value);
                                                if !replacing && inner_old != inner_new {
                                                    output.push(VecDiff::Move {
                                                        old_index: base + inner_old,
                                                        new_index: base + inner_new,
                                                    });
                                                }
                                            }
                                            VecDiff::Clear => {
                                                let old_len = slot.values.len();
                                                slot.values.clear();
                                                if !replacing {
                                                    for _ in 0..old_len {
                                                        output.push(VecDiff::RemoveAt { index: base });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if replacing {
                            let values = entries
                                .iter()
                                .flat_map(|slot| {
                                    assert!(
                                        slot.published,
                                        "flatten replacement published before all inner vectors were ready"
                                    );
                                    slot.values.iter().cloned()
                                })
                                .collect();
                            Some(vec![VecDiff::Replace { values }])
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
        Flatten {
            materializer,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "tracing")]
    #[track_caller]
    /// Adds debug logging to this [`SignalVec`]'s raw [`VecDiff`] outputs.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
    /// let signal = vec.signal_vec().debug();
    /// // signal logs `[ Replace { values: [ 1, 2, 3 ] } ]`
    /// vec.write(&mut world).push(4);
    /// // signal logs `[ Push { value: 4 } ]`
    /// ```
    fn debug(self) -> Debug<Self>
    where
        Self: Sized,
        Self::Item: fmt::Debug + Send + Sync + 'static,
    {
        let location = core::panic::Location::caller();
        let upstream = Box::new(self).into_signal_vec_materializer();
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(upstream)?;
            registrar.signal_vec_entity(
                upstream,
                move |In(diffs)| {
                    bevy_log::debug!("[{}] {:#?}", location, diffs);
                    diffs
                },
                OutputDelivery::move_only(),
            )
        });
        Debug { materializer }
    }

    /// Erases the type of this [`SignalVec`], allowing it to be used in conjunction with
    /// [`SignalVec`]s of other concrete types.
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
    ///     MutableVec::builder()
    ///         .values([1, 2, 3])
    ///         .spawn(&mut world)
    ///         .signal_vec()
    ///         .map_in(|x: i32| x * 2)
    ///         .boxed() // this is a `Map<Source<i32>>`
    /// } else {
    ///     MutableVec::builder()
    ///         .values([1, 2, 3])
    ///         .spawn(&mut world)
    ///         .signal_vec()
    ///         .filter(|In(x): In<i32>| x % 2 == 0)
    ///         .boxed() // this is a `Filter<Source<i32>>`
    /// }; // without the `.boxed()`, the compiler would not allow this
    /// ```
    fn boxed(self) -> BoxedSignalVec<Self::Item>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Erases this vector signal's concrete type while preserving verified replay capability.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system(|In(_)| true).switch_signal_vec(
    ///     |In(condition): In<bool>, world: &mut World| {
    ///         if condition {
    ///             MutableVec::builder()
    ///                 .values([1, 2, 3])
    ///                 .spawn(world)
    ///                 .signal_vec()
    ///                 .map_in(|x: i32| x * 2)
    ///                 .cache()
    ///                 .boxed_replayable() // this is a cached `Map<Source<i32>>`
    ///         } else {
    ///             MutableVec::builder()
    ///                 .values([1, 2, 3])
    ///                 .spawn(world)
    ///                 .signal_vec()
    ///                 .filter(|In(x): In<i32>| x % 2 == 0)
    ///                 .cache()
    ///                 .boxed_replayable() // this is a cached `Filter<Source<i32>>`
    ///         } // without `.boxed_replayable()`, the compiler would not allow this
    ///     },
    /// );
    /// ```
    fn boxed_replayable(self) -> BoxedReplayableSignalVec<Self::Item>
    where
        Self: Sized + ReplayableSignalVec,
        Self::Item: Clone + Send + Sync + 'static,
    {
        BoxedReplayableSignalVec(Box::new(self))
    }

    /// Assign a schedule to this signal vec chain, see [`crate::signal::SignalExt::schedule`].
    ///
    /// # Panics
    ///
    /// Panics if the schedule has not been registered with
    /// [`JonmoPlugin::with_schedule`](crate::JonmoPlugin::with_schedule).
    fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ScheduledVec<Sched, Self::Item>
    where
        Self: Sized + 'static,
    {
        let materializer = VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal_vec(Box::new(self).into_signal_vec_materializer())?;
            registrar.apply_schedule(upstream, Sched::default())
        });
        ScheduledVec {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Attempts to materialize this vector signal through its framework-owned registrar and acquire
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
        Box::new(self).into_signal_vec_materializer().try_register(world)
    }

    /// Activate this [`SignalVec`] and all its upstreams, causing them to be evaluated every frame
    /// until they are [`SignalClaim::release`]-ed, see [`SignalClaim`].
    fn register(self, world: &mut World) -> SignalClaim
    where
        Self: Sized,
    {
        Box::new(self).into_signal_vec_materializer().register(world)
    }
}

impl<T: ?Sized> SignalVecExt for T where T: SignalVec {}

/// Signal vec node wrapper that assigns a schedule to a signal chain, see
/// [`.schedule`](SignalVecExt::schedule).
pub struct ScheduledVec<Sched, I: 'static, Mode = OrdinaryBroadcast> {
    materializer: VecSignalMaterializer<I>,
    _marker: PhantomData<fn() -> (Sched, Mode)>,
}

/// Alias for a [`ScheduledVec`] handle backed by a verified [`ReplayableSignalVec`] terminal.
pub type ReplayableScheduledVec<Sched, I> = ScheduledVec<Sched, I, ReplayableBroadcastMarker>;

impl<Sched: 'static, I: 'static> ScheduledVec<Sched, I, ReplayableBroadcastMarker> {
    /// Returns this already-verified replayable scheduled vector without adding another cache
    /// entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable scheduled vector terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcastVec<I>
    where
        I: Clone + Send + Sync + 'static,
    {
        replayable_vec_broadcast(self)
    }

    /// Reassigns this verified replayable vector terminal's schedule while preserving its replay
    /// capability.
    pub fn schedule<NextSched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduledVec<NextSched, I>
    where
        I: Clone + Send + Sync + 'static,
    {
        replayable_vec_schedule(self)
    }
}

impl<Sched: 'static, I: 'static, Mode: Send + Sync + 'static> SignalVec for ScheduledVec<Sched, I, Mode> {
    type Item = I;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Sched: 'static, I: 'static> replayable_signal_vec_sealed::Sealed
    for ScheduledVec<Sched, I, ReplayableBroadcastMarker>
{
}

impl<Sched: 'static, I: Clone + Send + Sync + 'static> ReplayableSignalVec
    for ScheduledVec<Sched, I, ReplayableBroadcastMarker>
{
}

/// Provides immutable access to the underlying [`Vec`].
pub struct MutableVecReadGuard<'s, T> {
    guard: &'s MutableVecData<T>,
}

impl<'s, T> Deref for MutableVecReadGuard<'s, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.guard.vec
    }
}

/// Provides limited mutable access to the underlying [`Vec`].
pub struct MutableVecWriteGuard<'s, T> {
    guard: Mut<'s, MutableVecData<T>>,
}

impl<'s, T> Deref for MutableVecWriteGuard<'s, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.guard.vec
    }
}

impl<'a, T> MutableVecWriteGuard<'a, T>
where
    T: Clone,
{
    /// Appends an element to the back of this [`MutableVec`], queueing a [`VecDiff::Push`].
    pub fn push(&mut self, value: T) {
        self.guard.vec.push(value.clone());
        self.guard.queue_diff(VecDiff::Push { value });
    }

    /// If this [`MutableVec`] is not empty, removes the last element and returns it, queueing a
    /// [`VecDiff::Pop`], otherwise returns [`None`].
    pub fn pop(&mut self) -> Option<T> {
        let result = self.guard.vec.pop();
        if result.is_some() {
            self.guard.queue_diff(VecDiff::Pop);
        }
        result
    }

    /// Inserts an element at `index` within the vector, shifting all elements after it to the
    /// right, queueing a [`VecDiff::InsertAt`].
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, value: T) {
        self.guard.vec.insert(index, value.clone());
        self.guard.queue_diff(VecDiff::InsertAt { index, value });
    }

    /// Removes and returns the element at position `index` within the vector, shifting all elements
    /// after it to the left, queueing a [`VecDiff::RemoveAt`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        let value = self.guard.vec.remove(index);
        self.guard.queue_diff(VecDiff::RemoveAt { index });
        value
    }

    /// Clears this [`MutableVec`], removing all values, queueing a [`VecDiff::Clear`] if it was not
    /// empty.
    pub fn clear(&mut self) {
        if !self.guard.vec.is_empty() {
            self.guard.vec.clear();
            self.guard.queue_diff(VecDiff::Clear);
        }
    }

    /// Updates the element at `index` with a new `value`, queueing a [`VecDiff::UpdateAt`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        let len = self.guard.vec.len();
        if index < len {
            self.guard.vec[index] = value.clone();
            self.guard.queue_diff(VecDiff::UpdateAt { index, value });
        } else {
            panic!(
                "MutableVecWriteGuard::set: index {} out of bounds for len {}",
                index, len
            );
        }
    }

    /// Moves an item from `old_index` to `new_index`. queueing a [`VecDiff::Move`] if the indices
    /// are different and valid.
    ///
    /// # Panics
    ///
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
            self.guard.queue_diff(VecDiff::Move { old_index, new_index });
        }
    }

    /// Replaces the entire contents of this [`MutableVec`] with the provided `values`, queueing a
    /// [`VecDiff::Replace`].
    pub fn replace<A>(&mut self, values: A)
    where
        Vec<T>: From<A>,
    {
        let values: Vec<T> = values.into();
        self.guard.vec = values.clone();
        self.guard.queue_diff(VecDiff::Replace { values });
    }
}

type VecRevision = u64;

#[derive(Clone)]
struct VersionedVecDiff<T> {
    revision: VecRevision,
    diff: VecDiff<T>,
}

/// [`Component`] that holds the actual state for a [`MutableVec`].
#[derive(Component)]
pub struct MutableVecData<T> {
    vec: Vec<T>,
    revision: VecRevision,
    pending_diffs: Vec<VersionedVecDiff<T>>,
    lifetime: Weak<MutableVecDataLifetime>,
}

impl<T> MutableVecData<T> {
    fn queue_diff(&mut self, diff: VecDiff<T>) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("MutableVec source revision overflow");

        self.pending_diffs.push(VersionedVecDiff {
            revision: self.revision,
            diff,
        });
    }
}

struct MutableVecDataLifetime {
    entity: Entity,
    cleanup: CleanupSender,
}

impl Drop for MutableVecDataLifetime {
    fn drop(&mut self) {
        self.cleanup.send_mutable_vec(self.entity);
    }
}

struct MutableVecOwner<T: 'static> {
    lifetime: Arc<MutableVecDataLifetime>,
    source: SignalMaterializerFactory<VecSignalKind<T>>,
}

/// Wrapper around a [`Vec`] that emits mutations as [`VecDiff`]s, enabling diff-less constant-time
/// reactive updates for downstream [`SignalVec`]s.
pub struct MutableVec<T: 'static> {
    owner: Arc<MutableVecOwner<T>>,
}

impl<T: 'static> Clone for MutableVec<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
        }
    }
}

impl<T: 'static> MutableVec<T> {
    /// Creates a [`MutableVecBuilder`] for constructing a [`MutableVec`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// // Empty MutableVec
    /// let vec = MutableVec::<i32>::builder().spawn(&mut world);
    /// // With initial values
    /// let vec = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
    /// // With custom initialization
    /// let vec = MutableVec::builder()
    ///     .with_values(|v| v.extend(0..10))
    ///     .spawn(&mut world);
    /// ```
    pub fn builder() -> MutableVecBuilder<T> {
        MutableVecBuilder::new()
    }

    fn entity(&self) -> Entity {
        self.owner.lifetime.entity
    }

    /// Provides read-only access to the underlying [`Vec`] via either a `&World` or a
    /// `&Query<MutableVecData<T>>`.
    pub fn read<'s>(&self, mutable_vec_data_reader: impl ReadMutableVecData<'s, T>) -> MutableVecReadGuard<'s, T>
    where
        T: Send + Sync + 'static,
    {
        MutableVecReadGuard {
            guard: mutable_vec_data_reader.read(self.entity()),
        }
    }
    /// Provides write access to the underlying [`Vec`] via either a `&mut World` or a
    /// `&mut Query<&mut MutableVecData<T>>`.
    pub fn write<'w>(&self, mutable_vec_data_writer: impl WriteMutableVecData<'w, T>) -> MutableVecWriteGuard<'w, T>
    where
        T: Send + Sync + 'static,
    {
        MutableVecWriteGuard {
            guard: mutable_vec_data_writer.write(self.entity()),
        }
    }

    /// Returns a [`Source`] signal from this [`MutableVec`].
    pub fn signal_vec(&self) -> Source<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        Source {
            materializer: self.owner.source.materializer(),
        }
    }
}

fn new_mutable_vec_source<T>(values: Vec<T>, lifetime: &Arc<MutableVecDataLifetime>) -> MutableVecData<T> {
    MutableVecData {
        vec: values,
        revision: 0,
        pending_diffs: Vec::new(),
        lifetime: Arc::downgrade(lifetime),
    }
}

fn mutable_vec_from_parts<T>(lifetime: Arc<MutableVecDataLifetime>) -> MutableVec<T>
where
    T: Clone + Send + Sync + 'static,
{
    let entity = lifetime.entity;
    let source_lifetime = lifetime.clone();
    let source = VecSignalMaterializer::new(move |registrar| {
        let data = registrar.preflight_component::<MutableVecData<T>>(entity)?;
        // Entity IDs can collide across worlds; the data must belong to this source's lifetime.
        if !Weak::ptr_eq(&data.lifetime, &Arc::downgrade(&source_lifetime)) {
            return Err(SignalRegistrationError::EntityUnavailable);
        }
        let current = data.vec.clone();
        let initial_revision = data.revision;
        registrar.replayable_signal_vec_source(
            current,
            move |In(()),
                  mut mutable_vec_datas: Query<&mut MutableVecData<T>>,
                  mut cursor: Local<Option<VecRevision>>| {
                let _source_lifetime = &source_lifetime;
                let mut data = mutable_vec_datas.get_mut(entity).ok()?;
                let cursor = cursor.get_or_insert(initial_revision);
                let mut public_diffs = Vec::new();
                for VersionedVecDiff { revision, diff } in core::mem::take(&mut data.pending_diffs) {
                    if revision > *cursor {
                        *cursor = revision;
                        public_diffs.push(diff);
                    }
                }
                (!public_diffs.is_empty()).then_some(public_diffs)
            },
            OutputDelivery::broadcast(),
        )
    })
    .into_factory();
    MutableVec {
        owner: Arc::new(MutableVecOwner { lifetime, source }),
    }
}

fn spawn_mutable_vec<T>(world: &mut World, values: Vec<T>) -> MutableVec<T>
where
    T: Clone + Send + Sync + 'static,
{
    let cleanup = CleanupSender::bound(world);
    let entity = world.spawn_empty().id();
    let lifetime = Arc::new(MutableVecDataLifetime { entity, cleanup });
    world
        .entity_mut(entity)
        .insert(new_mutable_vec_source(values, &lifetime));
    mutable_vec_from_parts(lifetime)
}

fn spawn_mutable_vec_commands<T>(commands: &mut Commands, values: Vec<T>) -> MutableVec<T>
where
    T: Clone + Send + Sync + 'static,
{
    let (cleanup, binder) = CleanupSender::deferred();
    let entity = commands.spawn_empty().id();
    let lifetime = Arc::new(MutableVecDataLifetime { entity, cleanup });
    commands
        .entity(entity)
        .insert(new_mutable_vec_source(values, &lifetime));
    commands.queue(move |world: &mut World| binder.bind(world));
    mutable_vec_from_parts(lifetime)
}

impl<T> From<&mut World> for MutableVec<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn from(world: &mut World) -> Self {
        spawn_mutable_vec(world, Vec::new())
    }
}

impl<T> FromWorld for MutableVec<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn from_world(world: &mut World) -> Self {
        world.into()
    }
}

/// Builder for constructing a [`MutableVec`].
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let vec = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
/// ```
pub struct MutableVecBuilder<T>(Vec<T>);

impl<T> Default for MutableVecBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MutableVecBuilder<T> {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Sets the initial values of the [`MutableVec`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let vec = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
    /// ```
    pub fn values(mut self, values: impl Into<Vec<T>>) -> Self {
        self.0 = values.into();
        self
    }

    /// Mutably access the inner [`Vec`] for custom initialization.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let vec = MutableVec::builder()
    ///     .with_values(|v| {
    ///         v.extend(0..10);
    ///         v.push(100);
    ///     })
    ///     .spawn(&mut world);
    /// ```
    pub fn with_values(mut self, f: impl FnOnce(&mut Vec<T>)) -> Self {
        f(&mut self.0);
        self
    }
}

impl<T, A> From<A> for MutableVecBuilder<T>
where
    Vec<T>: From<A>,
{
    fn from(value: A) -> Self {
        Self(value.into())
    }
}

impl<T: Send + Sync + 'static + Clone> MutableVecBuilder<T> {
    /// Spawns a [`MutableVec`] using a `&mut World`.
    pub fn spawn(self, world: &mut World) -> MutableVec<T> {
        spawn_mutable_vec(world, self.0)
    }

    /// Spawns a [`MutableVec`] using a `&mut Commands`.
    pub fn spawnc(self, commands: &mut Commands) -> MutableVec<T> {
        spawn_mutable_vec_commands(commands, self.0)
    }
}

impl<T> From<&mut Commands<'_, '_>> for MutableVec<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn from(commands: &mut Commands) -> Self {
        spawn_mutable_vec_commands(commands, Vec::new())
    }
}

/// Specifies mutation accessors for [`MutableVec`]s.
pub trait ReadMutableVecData<'s, T>
where
    T: Send + Sync,
{
    #[allow(missing_docs)]
    fn read(self, entity: Entity) -> &'s MutableVecData<T>;
}

impl<'s, T> ReadMutableVecData<'s, T> for &'s Query<'_, 's, &MutableVecData<T>>
where
    T: Send + Sync + 'static,
{
    fn read(self, entity: Entity) -> &'s MutableVecData<T> {
        self.get(entity).unwrap()
    }
}

impl<'s, T> ReadMutableVecData<'s, T> for &'s World
where
    T: Send + Sync + 'static,
{
    fn read(self, entity: Entity) -> &'s MutableVecData<T> {
        self.get(entity).unwrap()
    }
}

/// Specifies mutation accessors for [`MutableVec`]s.
pub trait WriteMutableVecData<'w, T>
where
    T: Send + Sync,
{
    #[allow(missing_docs)]
    fn write(self, entity: Entity) -> Mut<'w, MutableVecData<T>>;
}

impl<'a, 'w, 's, T> WriteMutableVecData<'a, T> for &'a mut Query<'w, 's, &mut MutableVecData<T>>
where
    T: Send + Sync + 'static,
{
    fn write(self, entity: Entity) -> Mut<'a, MutableVecData<T>> {
        self.get_mut(entity).unwrap()
    }
}

impl<'w, T> WriteMutableVecData<'w, T> for &'w mut World
where
    T: Send + Sync + 'static,
{
    fn write(self, entity: Entity) -> Mut<'w, MutableVecData<T>> {
        self.get_mut(entity).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        JonmoPlugin, cleanup,
        graph::{SignalClaim, downcast_erased_signal_value, run_signal_subgraph_once_internal},
        prelude::clone,
        signal::{self, SignalExt},
    };
    use bevy::prelude::*;
    use bevy_ecs::world::CommandQueue;
    use bevy_platform::sync::RwLock;
    use test_log::test;

    #[derive(Resource, Default)]
    struct RetainedIndexReaders(Vec<Broadcast<Option<usize>>>);

    fn index_lookup_entities(world: &mut World) -> Vec<Entity> {
        let mut entities = world
            .query_filtered::<Entity, With<IndexLookup>>()
            .iter(world)
            .collect::<Vec<_>>();
        entities.sort();
        entities
    }

    fn register_index_lookup_producer(world: &mut World, intersperse: bool) -> (MutableVec<i32>, SignalClaim) {
        type IndexedValue = (Broadcast<Option<usize>>, i32);

        world.init_resource::<RetainedIndexReaders>();
        let source = MutableVec::builder().values([10, 20, 30]).spawn(world);
        let claim = if intersperse {
            source
                .signal_vec()
                .intersperse_with(
                    |In(index): In<Broadcast<Option<usize>>>, mut readers: ResMut<RetainedIndexReaders>| {
                        readers.0.push(index);
                        0
                    },
                )
                .for_each(|_: In<Vec<VecDiff<i32>>>| {})
                .register(world)
        } else {
            source
                .signal_vec()
                .enumerate()
                .for_each(
                    |In(diffs): In<Vec<VecDiff<IndexedValue>>>, mut readers: ResMut<RetainedIndexReaders>| {
                        for diff in diffs {
                            if let VecDiff::Replace { values } = diff {
                                readers.0.extend(values.into_iter().map(|(index, _)| index));
                            }
                        }
                    },
                )
                .register(world)
        };
        (source, claim)
    }

    fn assert_index_lookup_retained_readers(intersperse: bool) {
        let mut app = create_test_app();
        let (source, producer) = register_index_lookup_producer(app.world_mut(), intersperse);
        let producer_entity = producer.entity().entity();
        assert!(
            index_lookup_entities(app.world_mut()).is_empty(),
            "registration must not allocate lookup data"
        );
        app.update();
        let entities = index_lookup_entities(app.world_mut());
        assert_eq!(entities.len(), 1);
        let lookup_entity = entities[0];
        let reader_count = if intersperse { 2 } else { 3 };
        assert_eq!(
            app.world().get::<IndexLookup>(lookup_entity).unwrap().map.len(),
            reader_count
        );
        assert_eq!(app.world().resource::<RetainedIndexReaders>().0.len(), reader_count);
        let retained = app.world().resource::<RetainedIndexReaders>().0[0].clone();
        let removed = app.world().resource::<RetainedIndexReaders>().0.last().unwrap().clone();
        let reader = retained.register(app.world_mut());

        source.write(app.world_mut()).pop();
        app.update();
        assert_eq!(index_lookup_entities(app.world_mut()), entities);
        assert_eq!(
            app.world().get::<IndexLookup>(lookup_entity).unwrap().map.len(),
            reader_count - 1
        );
        producer.release(app.world_mut());
        assert!(app.world().get_entity(producer_entity).is_err());
        let graph = crate::graph::graph_snapshot(app.world_mut());
        assert_eq!(
            graph.nodes.len(),
            2,
            "only the reader and the retained MutableVec source may remain"
        );
        let reader_node = graph
            .nodes
            .iter()
            .find(|node| node.entity == reader.entity().entity())
            .unwrap();
        assert_eq!(reader_node.canonical_static_dependencies, Some(Vec::new()));
        let source_node = graph
            .nodes
            .iter()
            .find(|node| node.entity != reader.entity().entity())
            .unwrap();
        assert!(source_node.replay_maintenance_armed);
        assert!(source_node.materialized_dependents.is_empty());
        assert!(source_node.active_downstreams.is_empty());
        assert!(graph.nodes.iter().all(|node| node.entity != lookup_entity));
        assert_eq!(poll_index(app.world_mut(), &reader), Some(0));
        let removed_reader = removed.clone().register(app.world_mut());
        assert_eq!(
            run_signal_subgraph_once_internal(app.world_mut(), removed_reader.entity())
                .and_then(downcast_erased_signal_value::<Option<usize>>),
            Some(None),
            "removed keys must stay absent after producer reaping"
        );
        removed_reader.release(app.world_mut());
        drop(app.world_mut().remove_resource::<RetainedIndexReaders>().unwrap());
        reader.release(app.world_mut());
        assert_eq!(cleanup::queued_message_count(app.world()), 1);
        cleanup::drain_world_cleanup(app.world_mut());
        assert_eq!(cleanup::queued_message_count(app.world()), 0);
        assert_eq!(index_lookup_entities(app.world_mut()), entities);
        std::thread::spawn(move || drop(removed)).join().unwrap();
        assert_eq!(cleanup::queued_message_count(app.world()), 1);
        assert!(
            app.world().get_entity(lookup_entity).is_ok(),
            "off-thread drop only queues cleanup"
        );
        cleanup::drain_world_cleanup(app.world_mut());
        // Reaping the cached reader drops the final data token after this drain's boundary.
        assert_eq!(cleanup::queued_message_count(app.world()), 1);
        assert_eq!(index_lookup_entities(app.world_mut()), entities);
        cleanup::drain_world_cleanup(app.world_mut());
        assert!(index_lookup_entities(app.world_mut()).is_empty());
        assert!(app.world().get_entity(lookup_entity).is_err());
        assert!(app.world().get_entity(source.entity()).is_ok());
        assert_eq!(cleanup::queued_message_count(app.world()), 0);
    }

    #[test]
    fn enumerate_index_lookup_retained_readers() {
        assert_index_lookup_retained_readers(false);
    }

    #[test]
    fn intersperse_index_lookup_retained_readers() {
        assert_index_lookup_retained_readers(true);
    }

    #[test]
    fn enumerate_intersperse_index_lookup_final_producer_loss() {
        for intersperse in [false, true] {
            let mut app = create_test_app();
            let (_source, producer) = register_index_lookup_producer(app.world_mut(), intersperse);
            app.update();
            let entities = index_lookup_entities(app.world_mut());
            assert_eq!(entities.len(), 1);
            drop(app.world_mut().remove_resource::<RetainedIndexReaders>().unwrap());
            assert_eq!(cleanup::queued_message_count(app.world()), 0);
            producer.release(app.world_mut());
            assert_eq!(cleanup::queued_message_count(app.world()), 1);
            assert_eq!(index_lookup_entities(app.world_mut()), entities);
            cleanup::drain_world_cleanup(app.world_mut());
            assert!(index_lookup_entities(app.world_mut()).is_empty());
            let graph = crate::graph::graph_snapshot(app.world_mut());
            assert_eq!(graph.nodes.len(), 1, "only the retained MutableVec source may remain");
            assert!(graph.nodes[0].replay_maintenance_armed);
            assert!(graph.nodes[0].materialized_dependents.is_empty());
            assert_eq!(cleanup::queued_message_count(app.world()), 0);
        }
    }

    #[test]
    fn enumerate_intersperse_index_lookup_unexecuted_producers_allocate_nothing() {
        for intersperse in [false, true] {
            let mut app = create_test_app();
            let (_source, producer) = register_index_lookup_producer(app.world_mut(), intersperse);
            producer.release(app.world_mut());
            assert!(index_lookup_entities(app.world_mut()).is_empty());
            assert_eq!(cleanup::queued_message_count(app.world()), 0);
        }
    }

    #[test]
    fn enumerate_intersperse_index_lookup_final_token_and_exact_inbox() {
        let mut world = World::new();
        let lifetime = IndexLookupLifetime::new(&mut world);
        let entity = lifetime.entity;
        let neighbor = IndexLookupLifetime::new(&mut world);
        assert_eq!(
            lifetime.cleanup.inbox_identity(),
            cleanup::cleanup_inbox_identity(&world)
        );
        let weak = Arc::downgrade(&lifetime);
        let reader = lifetime.signal(0);
        let reader_clone = reader.clone();
        drop(lifetime);
        drop(reader);
        assert!(weak.upgrade().is_some());
        assert_eq!(cleanup::queued_message_count(&world), 0);
        std::thread::spawn(move || drop(reader_clone)).join().unwrap();
        assert!(weak.upgrade().is_none());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert_eq!(index_lookup_entities(&mut world), [neighbor.entity]);
        assert!(crate::graph::graph_snapshot(&mut world).nodes.is_empty());
        drop(neighbor);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(index_lookup_entities(&mut world).is_empty());
    }

    #[test]
    fn enumerate_intersperse_index_lookup_stale_generation() {
        let mut app = create_test_app();
        let world = app.world_mut();
        let lifetime = IndexLookupLifetime::new(world);
        let stale_entity = lifetime.entity;
        let reader = lifetime.signal(0).register(world);
        let next_gen = world.despawn_no_free(stale_entity).unwrap();
        let replacement_entity = world.spawn_empty_at(next_gen).unwrap().id();
        let replacement = IndexLookupLifetime::init_at(world, replacement_entity);
        assert_eq!(stale_entity.index(), replacement.entity.index());
        assert_ne!(stale_entity.generation(), replacement.entity.generation());
        replacement.synchronize(world, &[0]);
        assert!(matches!(
            Box::new(lifetime.signal(0))
                .into_signal_materializer()
                .try_register(world),
            Err(SignalRegistrationError::EntityUnavailable)
        ));
        assert_eq!(
            run_signal_subgraph_once_internal(world, reader.entity())
                .and_then(downcast_erased_signal_value::<Option<usize>>),
            Some(None)
        );
        reader.release(world);
        drop(lifetime);
        assert_eq!(cleanup::queued_message_count(world), 1);
        cleanup::drain_world_cleanup(world);
        assert_eq!(index_lookup_entities(world), [replacement.entity]);
        assert_eq!(
            world.get::<IndexLookup>(replacement.entity).unwrap().map.get(&0),
            Some(&0)
        );
    }

    #[test]
    fn enumerate_intersperse_index_lookup_foreign_world_collision() {
        let mut app_a = create_test_app();
        let mut app_b = create_test_app();
        let lifetime_a = IndexLookupLifetime::new(app_a.world_mut());
        let lifetime_b = IndexLookupLifetime::new(app_b.world_mut());
        assert_eq!(lifetime_a.entity, lifetime_b.entity);
        assert!(matches!(
            Box::new(lifetime_a.signal(0))
                .into_signal_materializer()
                .try_register(app_b.world_mut()),
            Err(SignalRegistrationError::EntityUnavailable)
        ));
    }

    #[test]
    fn enumerate_intersperse_index_lookup_drop_after_world() {
        let mut world = World::new();
        let lifetime = IndexLookupLifetime::new(&mut world);
        let weak = Arc::downgrade(&lifetime);
        let reader = lifetime.signal(0);
        drop(lifetime);
        drop(world);
        assert!(weak.upgrade().is_some());
        std::thread::spawn(move || drop(reader)).join().unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn intersperse_index_lookup_synchronous_callback_publication() {
        #[derive(Resource, Default)]
        struct Observations {
            readers: Vec<Broadcast<Option<usize>>>,
            indices: Vec<Vec<Option<usize>>>,
            published_indices: Vec<Vec<usize>>,
        }

        let mut app = create_test_app();
        app.init_resource::<Observations>();
        let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());
        let producer = source
            .signal_vec()
            .intersperse_with(|In(index): In<Broadcast<Option<usize>>>, world: &mut World| {
                let readers = {
                    let mut observations = world.resource_mut::<Observations>();
                    observations.readers.push(index);
                    observations.readers.clone()
                };
                let indices = readers
                    .into_iter()
                    .map(|reader| {
                        let claim = reader.register(world);
                        let index = run_signal_subgraph_once_internal(world, claim.entity())
                            .and_then(downcast_erased_signal_value::<Option<usize>>)
                            .expect("reader must run synchronously inside the separator callback");
                        claim.release(world);
                        index
                    })
                    .collect();
                let entities = index_lookup_entities(world);
                assert_eq!(entities.len(), 1);
                let mut published = world
                    .get::<IndexLookup>(entities[0])
                    .unwrap()
                    .map
                    .values()
                    .copied()
                    .collect::<Vec<_>>();
                published.sort();
                let mut observations = world.resource_mut::<Observations>();
                observations.indices.push(indices);
                observations.published_indices.push(published);
                0
            })
            .for_each(|_: In<Vec<VecDiff<i32>>>| {})
            .register(app.world_mut());
        app.update();
        let lookup_entities = index_lookup_entities(app.world_mut());
        {
            let mut values = source.write(app.world_mut());
            values.insert(0, 5);
            values.remove(1);
            values.push(40);
            values.move_item(0, 2);
            values.clear();
            values.push(9);
            values.push(10);
            values.pop();
            values.replace([1, 2, 3]);
        }
        app.update();
        assert_eq!(index_lookup_entities(app.world_mut()), lookup_entities);
        let observations = app.world().resource::<Observations>();
        assert_eq!(
            observations.indices,
            vec![
                vec![Some(0)],
                vec![Some(0), Some(1)],
                vec![Some(1), Some(2), Some(0)],
                vec![None, Some(1), Some(0), Some(2)],
                vec![None, None, None, None, Some(0)],
                vec![None, None, None, None, Some(0), Some(1)],
                vec![None, None, None, None, Some(0), Some(1), Some(2)],
                vec![None, None, None, None, None, None, None, Some(0)],
                vec![None, None, None, None, None, None, None, None, Some(0)],
                vec![None, None, None, None, None, None, None, None, Some(0), Some(1)],
            ]
        );
        assert_eq!(
            observations.published_indices,
            vec![
                vec![0, 1],
                vec![0, 1],
                vec![0, 1, 2],
                vec![0, 1, 2],
                vec![0, 1, 2],
                vec![0, 1, 2],
                vec![0, 1, 2],
                vec![0],
                vec![0, 1],
                vec![0, 1],
            ]
        );
        producer.release(app.world_mut());
        assert_eq!(index_lookup_entities(app.world_mut()), lookup_entities);
        drop(app.world_mut().remove_resource::<Observations>().unwrap());
        assert_eq!(cleanup::queued_message_count(app.world()), 10);
        cleanup::drain_world_cleanup(app.world_mut());
        assert_eq!(cleanup::queued_message_count(app.world()), 1);
        assert_eq!(index_lookup_entities(app.world_mut()), lookup_entities);
        cleanup::drain_world_cleanup(app.world_mut());
        assert!(index_lookup_entities(app.world_mut()).is_empty());
        assert_eq!(cleanup::queued_message_count(app.world()), 0);
    }

    #[derive(Resource, Default, Debug)]
    struct SignalVecOutput<T: Clone + fmt::Debug>(Vec<VecDiff<T>>);

    #[derive(Resource, Default)]
    struct ReadySignalValues(Vec<i32>);

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MutableVecReplaySourceSnapshot {
        entity: Entity,
        exists: bool,
        current_len: Option<usize>,
        revision: Option<VecRevision>,
        pending_revisions: Vec<VecRevision>,
    }

    fn mutable_vec_replay_source_snapshot<T: Send + Sync + 'static>(
        world: &World,
        entity: Entity,
    ) -> MutableVecReplaySourceSnapshot {
        let data = world.get::<MutableVecData<T>>(entity);
        MutableVecReplaySourceSnapshot {
            entity,
            exists: world.get_entity(entity).is_ok(),
            current_len: data.map(|data| data.vec.len()),
            revision: data.map(|data| data.revision),
            pending_revisions: data
                .map(|data| data.pending_diffs.iter().map(|diff| diff.revision).collect())
                .unwrap_or_default(),
        }
    }

    #[test]
    fn mutable_vec_replay_snapshot_registration_ownership_observability() {
        let mut world = World::new();
        let source = MutableVec::builder().values([10, 20]).spawn(&mut world);
        let entity = source.entity();

        assert_eq!(
            mutable_vec_replay_source_snapshot::<i32>(&world, entity),
            MutableVecReplaySourceSnapshot {
                entity,
                exists: true,
                current_len: Some(2),
                revision: Some(0),
                pending_revisions: Vec::new(),
            }
        );

        source.write(&mut world).push(30);
        source.write(&mut world).insert(0, 5);
        let first = mutable_vec_replay_source_snapshot::<i32>(&world, entity);
        let second = mutable_vec_replay_source_snapshot::<i32>(&world, entity);
        assert_eq!(
            first, second,
            "unchanged typed replay state must snapshot deterministically"
        );
        assert_eq!(
            first,
            MutableVecReplaySourceSnapshot {
                entity,
                exists: true,
                current_len: Some(4),
                revision: Some(2),
                pending_revisions: vec![1, 2],
            },
            "pending replay history must preserve source revision order"
        );
    }

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        app
    }

    // Helper system to capture signal_vec output
    fn capture_signal_vec_output<T>(In(diffs): In<Vec<VecDiff<T>>>, mut output: ResMut<SignalVecOutput<T>>)
    where
        T: Send + Sync + 'static + Clone + fmt::Debug,
    {
        #[cfg(feature = "tracing")]
        bevy_log::debug!(
            "Capture SignalVec Output: Received {:?}, extending resource from {:?} with new diffs",
            diffs,
            output.0
        );
        output.0.extend(diffs);
    }

    // Helper to make assertions cleaner
    fn get_and_clear_output<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &mut World) -> Vec<VecDiff<T>> {
        let output = world.resource::<SignalVecOutput<T>>().0.clone();
        world.resource_mut::<SignalVecOutput<T>>().0.clear();
        output
    }

    // Add this PartialEq implementation for easier testing
    impl<T: Send + Sync + 'static + PartialEq + fmt::Debug> PartialEq for VecDiff<T> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Replace { values: l_values }, Self::Replace { values: r_values }) => l_values == r_values,
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
                (Self::RemoveAt { index: l_index }, Self::RemoveAt { index: r_index }) => l_index == r_index,
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
                (Self::Push { value: l_value }, Self::Push { value: r_value }) => l_value == r_value,
                (Self::Pop, Self::Pop) => true,
                (Self::Clear, Self::Clear) => true,
                _ => false,
            }
        }
    }

    // Helper function to apply a series of diffs to a vector to check the final state.
    fn apply_diffs<T: Clone>(initial: &mut Vec<T>, diffs: Vec<VecDiff<T>>) {
        for diff in diffs {
            diff.apply_to_vec(initial);
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct TaggedItem {
        key: i32,
        id: u32,
    }

    #[derive(Resource, Default, Debug)]
    struct CollectionCorrectnessCaptureA(Vec<VecDiff<i32>>);

    #[derive(Resource, Default, Debug)]
    struct CollectionCorrectnessCaptureB(Vec<VecDiff<i32>>);

    #[derive(Clone, Debug)]
    enum CollectionCorrectnessOp {
        Push(TaggedItem),
        Pop,
        Insert { index: usize, value: TaggedItem },
        Remove { index: usize },
        Update { index: usize, value: TaggedItem },
        Move { old_index: usize, new_index: usize },
        Clear,
        Replace(Vec<TaggedItem>),
    }

    impl CollectionCorrectnessOp {
        fn apply(&self, source: &MutableVec<TaggedItem>, world: &mut World, plain_source: &mut Vec<TaggedItem>) {
            match self {
                Self::Push(value) => {
                    source.write(&mut *world).push(value.clone());
                    plain_source.push(value.clone());
                }
                Self::Pop => {
                    source.write(&mut *world).pop();
                    plain_source.pop();
                }
                Self::Insert { index, value } => {
                    source.write(&mut *world).insert(*index, value.clone());
                    plain_source.insert(*index, value.clone());
                }
                Self::Remove { index } => {
                    source.write(&mut *world).remove(*index);
                    plain_source.remove(*index);
                }
                Self::Update { index, value } => {
                    source.write(&mut *world).set(*index, value.clone());
                    plain_source[*index] = value.clone();
                }
                Self::Move { old_index, new_index } => {
                    source.write(&mut *world).move_item(*old_index, *new_index);
                    let value = plain_source.remove(*old_index);
                    plain_source.insert(*new_index, value);
                }
                Self::Clear => {
                    source.write(&mut *world).clear();
                    plain_source.clear();
                }
                Self::Replace(values) => {
                    source.write(&mut *world).replace(values.clone());
                    *plain_source = values.clone();
                }
            }
        }
    }

    #[derive(Clone, Debug)]
    struct CollectionCorrectnessCase {
        label: &'static str,
        before: Vec<TaggedItem>,
        operation: CollectionCorrectnessOp,
        expect_silent: bool,
    }

    fn tagged(key: i32, id: u32) -> TaggedItem {
        TaggedItem { key, id }
    }

    fn sort_tagged_by_comparator(values: &mut [TaggedItem]) {
        values.sort_by_key(|item| item.key);
    }

    fn sort_tagged_by_key(values: &mut [TaggedItem]) {
        values.sort_by_key(|item| item.key);
    }

    fn collection_correctness_boundary_cases() -> Vec<CollectionCorrectnessCase> {
        vec![
            CollectionCorrectnessCase {
                label: "push into len 0",
                before: vec![],
                operation: CollectionCorrectnessOp::Push(tagged(2, 1)),
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "push len 3 to len 4",
                before: vec![tagged(3, 2), tagged(1, 3), tagged(2, 4)],
                operation: CollectionCorrectnessOp::Push(tagged(1, 5)),
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "pop only item",
                before: vec![tagged(1, 6)],
                operation: CollectionCorrectnessOp::Pop,
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "pop len 4 back boundary",
                before: vec![tagged(2, 7), tagged(1, 8), tagged(2, 9), tagged(0, 10)],
                operation: CollectionCorrectnessOp::Pop,
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "insert into len 0 at 0",
                before: vec![],
                operation: CollectionCorrectnessOp::Insert {
                    index: 0,
                    value: tagged(1, 11),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "insert front boundary",
                before: vec![tagged(2, 12), tagged(1, 13), tagged(3, 14)],
                operation: CollectionCorrectnessOp::Insert {
                    index: 0,
                    value: tagged(1, 15),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "insert middle boundary",
                before: vec![tagged(2, 16), tagged(1, 17), tagged(3, 18)],
                operation: CollectionCorrectnessOp::Insert {
                    index: 1,
                    value: tagged(1, 19),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "insert back boundary",
                before: vec![tagged(2, 20), tagged(1, 21), tagged(3, 22)],
                operation: CollectionCorrectnessOp::Insert {
                    index: 3,
                    value: tagged(1, 23),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "remove only item",
                before: vec![tagged(1, 24)],
                operation: CollectionCorrectnessOp::Remove { index: 0 },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "remove front boundary",
                before: vec![tagged(2, 25), tagged(1, 26), tagged(2, 27), tagged(0, 28)],
                operation: CollectionCorrectnessOp::Remove { index: 0 },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "remove middle boundary",
                before: vec![tagged(2, 29), tagged(1, 30), tagged(2, 31), tagged(0, 32)],
                operation: CollectionCorrectnessOp::Remove { index: 2 },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "remove back boundary",
                before: vec![tagged(2, 33), tagged(1, 34), tagged(2, 35), tagged(0, 36)],
                operation: CollectionCorrectnessOp::Remove { index: 3 },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "update only item",
                before: vec![tagged(1, 37)],
                operation: CollectionCorrectnessOp::Update {
                    index: 0,
                    value: tagged(2, 37),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "update front boundary",
                before: vec![tagged(2, 38), tagged(1, 39), tagged(2, 40), tagged(0, 41)],
                operation: CollectionCorrectnessOp::Update {
                    index: 0,
                    value: tagged(-1, 38),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "update middle boundary",
                before: vec![tagged(2, 42), tagged(1, 43), tagged(2, 44), tagged(0, 45)],
                operation: CollectionCorrectnessOp::Update {
                    index: 2,
                    value: tagged(1, 44),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "update back boundary",
                before: vec![tagged(2, 46), tagged(1, 47), tagged(2, 48), tagged(0, 49)],
                operation: CollectionCorrectnessOp::Update {
                    index: 3,
                    value: tagged(3, 49),
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "move only item onto itself",
                before: vec![tagged(1, 50)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 0,
                    new_index: 0,
                },
                expect_silent: true,
            },
            CollectionCorrectnessCase {
                label: "equal-key adjacent forward move",
                before: vec![tagged(1, 51), tagged(1, 52)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 0,
                    new_index: 1,
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "equal-key adjacent backward move",
                before: vec![tagged(1, 53), tagged(1, 54)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 1,
                    new_index: 0,
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "equal-key nonadjacent forward move",
                before: vec![tagged(1, 55), tagged(2, 56), tagged(1, 57), tagged(3, 58)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 0,
                    new_index: 2,
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "equal-key nonadjacent backward move",
                before: vec![tagged(1, 59), tagged(2, 60), tagged(1, 61), tagged(3, 62)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 2,
                    new_index: 0,
                },
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "unequal-key forward move is silent",
                before: vec![tagged(3, 63), tagged(1, 64), tagged(2, 65)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 0,
                    new_index: 2,
                },
                expect_silent: true,
            },
            CollectionCorrectnessCase {
                label: "unequal-key backward move is silent",
                before: vec![tagged(1, 66), tagged(2, 67), tagged(3, 68)],
                operation: CollectionCorrectnessOp::Move {
                    old_index: 2,
                    new_index: 0,
                },
                expect_silent: true,
            },
            CollectionCorrectnessCase {
                label: "clear len 0 is silent",
                before: vec![],
                operation: CollectionCorrectnessOp::Clear,
                expect_silent: true,
            },
            CollectionCorrectnessCase {
                label: "clear len 4",
                before: vec![tagged(2, 69), tagged(1, 70), tagged(2, 71), tagged(0, 72)],
                operation: CollectionCorrectnessOp::Clear,
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "replace empty with len 4",
                before: vec![],
                operation: CollectionCorrectnessOp::Replace(vec![
                    tagged(2, 73),
                    tagged(1, 74),
                    tagged(2, 75),
                    tagged(0, 76),
                ]),
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "replace len 4 with empty",
                before: vec![tagged(2, 77), tagged(1, 78), tagged(2, 79), tagged(0, 80)],
                operation: CollectionCorrectnessOp::Replace(vec![]),
                expect_silent: false,
            },
            CollectionCorrectnessCase {
                label: "replace len 2 with len 1",
                before: vec![tagged(1, 81), tagged(1, 82)],
                operation: CollectionCorrectnessOp::Replace(vec![tagged(0, 83)]),
                expect_silent: false,
            },
        ]
    }

    fn assert_collection_correctness_sorted_state(
        app: &mut App,
        downstream: &mut Vec<TaggedItem>,
        plain_source: &[TaggedItem],
        oracle_sort: fn(&mut [TaggedItem]),
        context: &str,
    ) -> Vec<VecDiff<TaggedItem>> {
        app.update();
        let diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(downstream, diffs.clone());
        let mut expected = plain_source.to_vec();
        oracle_sort(&mut expected);
        assert_eq!(
            *downstream, expected,
            "{context}; emitted_diffs={diffs:?}, downstream={downstream:?}, expected={expected:?}, plain_source={plain_source:?}"
        );
        diffs
    }

    fn run_collection_correctness_boundary_table(
        app: &mut App,
        source: &MutableVec<TaggedItem>,
        oracle_sort: fn(&mut [TaggedItem]),
    ) {
        let mut plain_source = Vec::new();
        let mut downstream = Vec::new();
        let initial_diffs = assert_collection_correctness_sorted_state(
            app,
            &mut downstream,
            &plain_source,
            oracle_sort,
            "initial replay",
        );
        assert!(
            matches!(initial_diffs.first(), Some(VecDiff::Replace { values }) if values.is_empty()),
            "initial replay must begin with an empty Replace; actual={initial_diffs:?}"
        );

        for (case_index, case) in collection_correctness_boundary_cases().into_iter().enumerate() {
            let setup = CollectionCorrectnessOp::Replace(case.before);
            setup.apply(source, app.world_mut(), &mut plain_source);
            let setup_context = format!("case={case_index}, label={}, setup={setup:?}", case.label);
            assert_collection_correctness_sorted_state(
                app,
                &mut downstream,
                &plain_source,
                oracle_sort,
                &setup_context,
            );

            case.operation.apply(source, app.world_mut(), &mut plain_source);
            let operation_context = format!(
                "case={case_index}, label={}, operation={:?}",
                case.label, case.operation
            );
            let diffs = assert_collection_correctness_sorted_state(
                app,
                &mut downstream,
                &plain_source,
                oracle_sort,
                &operation_context,
            );
            if case.expect_silent {
                assert!(
                    diffs.is_empty(),
                    "silent-order control emitted diffs; {operation_context}, emitted_diffs={diffs:?}, downstream={downstream:?}, plain_source={plain_source:?}"
                );
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct CollectionCorrectnessLcg(u64);

    impl CollectionCorrectnessLcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }

        fn index(&mut self, exclusive_upper_bound: usize) -> usize {
            (self.next_u32() as usize) % exclusive_upper_bound
        }
    }

    fn generated_tagged(rng: &mut CollectionCorrectnessLcg, next_id: &mut u32) -> TaggedItem {
        let value = tagged((rng.next_u32() % 5) as i32 - 2, *next_id);
        *next_id += 1;
        value
    }

    fn generated_collection_correctness_op(
        step: usize,
        rng: &mut CollectionCorrectnessLcg,
        plain_source: &[TaggedItem],
        next_id: &mut u32,
    ) -> CollectionCorrectnessOp {
        match step {
            0 => CollectionCorrectnessOp::Replace(vec![tagged(1, 1), tagged(0, 2), tagged(1, 3)]),
            1 => CollectionCorrectnessOp::Push(generated_tagged(rng, next_id)),
            2 => CollectionCorrectnessOp::Insert {
                index: 0,
                value: generated_tagged(rng, next_id),
            },
            3 => CollectionCorrectnessOp::Update {
                index: 2,
                value: generated_tagged(rng, next_id),
            },
            4 => CollectionCorrectnessOp::Move {
                old_index: 0,
                new_index: 4,
            },
            5 => CollectionCorrectnessOp::Remove { index: 2 },
            6 => CollectionCorrectnessOp::Pop,
            7 => CollectionCorrectnessOp::Clear,
            _ => match rng.index(8) {
                0 if plain_source.len() < 12 => CollectionCorrectnessOp::Push(generated_tagged(rng, next_id)),
                0 => CollectionCorrectnessOp::Pop,
                1 if !plain_source.is_empty() => CollectionCorrectnessOp::Pop,
                1 => CollectionCorrectnessOp::Push(generated_tagged(rng, next_id)),
                2 if plain_source.len() < 12 => CollectionCorrectnessOp::Insert {
                    index: rng.index(plain_source.len() + 1),
                    value: generated_tagged(rng, next_id),
                },
                2 => CollectionCorrectnessOp::Remove {
                    index: rng.index(plain_source.len()),
                },
                3 if !plain_source.is_empty() => CollectionCorrectnessOp::Remove {
                    index: rng.index(plain_source.len()),
                },
                3 => CollectionCorrectnessOp::Insert {
                    index: 0,
                    value: generated_tagged(rng, next_id),
                },
                4 if !plain_source.is_empty() => CollectionCorrectnessOp::Update {
                    index: rng.index(plain_source.len()),
                    value: generated_tagged(rng, next_id),
                },
                4 => CollectionCorrectnessOp::Push(generated_tagged(rng, next_id)),
                5 if !plain_source.is_empty() => CollectionCorrectnessOp::Move {
                    old_index: rng.index(plain_source.len()),
                    new_index: rng.index(plain_source.len()),
                },
                5 => CollectionCorrectnessOp::Push(generated_tagged(rng, next_id)),
                6 => CollectionCorrectnessOp::Clear,
                _ => {
                    let len = rng.index(13);
                    CollectionCorrectnessOp::Replace((0..len).map(|_| generated_tagged(rng, next_id)).collect())
                }
            },
        }
    }

    fn run_generated_collection_correctness_oracle(
        app: &mut App,
        source: &MutableVec<TaggedItem>,
        oracle_sort: fn(&mut [TaggedItem]),
        seed: u64,
    ) {
        const STEPS: usize = 160;
        let mut rng = CollectionCorrectnessLcg(seed);
        let mut next_id = 10_000;
        let mut plain_source = Vec::new();
        let mut downstream = Vec::new();
        let mut trace = Vec::with_capacity(STEPS);

        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        assert_eq!(
            downstream,
            Vec::<TaggedItem>::new(),
            "seed={seed:#018x}, step=initial, op=InitialReplay, emitted_diffs={initial_diffs:?}, downstream={downstream:?}, expected=[], operation_trace={trace:#?}"
        );
        assert!(
            matches!(initial_diffs.first(), Some(VecDiff::Replace { values }) if values.is_empty()),
            "seed={seed:#018x}, step=initial, op=InitialReplay did not begin with Replace; emitted_diffs={initial_diffs:?}, operation_trace={trace:#?}"
        );

        for step in 0..STEPS {
            let operation = generated_collection_correctness_op(step, &mut rng, &plain_source, &mut next_id);
            operation.apply(source, app.world_mut(), &mut plain_source);
            assert!(
                plain_source.len() <= 12,
                "seed={seed:#018x}, step={step}, op={operation:?} exceeded max collection size; plain_source={plain_source:?}, operation_trace={trace:#?}"
            );
            trace.push(operation.clone());

            app.update();
            let diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
            apply_diffs(&mut downstream, diffs.clone());
            let mut expected = plain_source.clone();
            oracle_sort(&mut expected);
            assert_eq!(
                downstream, expected,
                "seed={seed:#018x}, step={step}, op={operation:?}, emitted_diffs={diffs:?}, downstream={downstream:?}, expected={expected:?}, plain_source={plain_source:?}, operation_trace={trace:#?}"
            );
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the test oracle keeps both subscribers' state explicit at each scenario step"
    )]
    fn settle_vector_replay_collection_correctness_oracle(
        app: &mut App,
        expected: &[i32],
        active_a: bool,
        downstream_a: &mut Vec<i32>,
        history_a: &mut Vec<VecDiff<i32>>,
        active_b: bool,
        downstream_b: &mut Vec<i32>,
        history_b: &mut Vec<VecDiff<i32>>,
        label: &str,
    ) -> (Vec<VecDiff<i32>>, Vec<VecDiff<i32>>) {
        app.update();
        let diffs_a = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessCaptureA>().0);
        let diffs_b = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessCaptureB>().0);

        if active_a {
            apply_diffs(downstream_a, diffs_a.clone());
            history_a.extend(diffs_a.iter().cloned());
            assert_eq!(
                downstream_a, expected,
                "{label}: subscriber A diverged; diffs={diffs_a:?}, downstream={downstream_a:?}, expected={expected:?}, history={history_a:?}"
            );
        } else {
            assert!(
                diffs_a.is_empty(),
                "{label}: inactive subscriber A captured {diffs_a:?}"
            );
        }

        if active_b {
            apply_diffs(downstream_b, diffs_b.clone());
            history_b.extend(diffs_b.iter().cloned());
            assert_eq!(
                downstream_b, expected,
                "{label}: subscriber B diverged; diffs={diffs_b:?}, downstream={downstream_b:?}, expected={expected:?}, history={history_b:?}"
            );
        } else {
            assert!(
                diffs_b.is_empty(),
                "{label}: inactive subscriber B captured {diffs_b:?}"
            );
        }

        (diffs_a, diffs_b)
    }

    #[derive(Resource, Default, Debug)]
    struct FinalSignalOutput<T: Send + Sync + 'static + Clone + fmt::Debug>(Option<T>);

    fn capture_final_output<T>(In(value): In<T>, mut output: ResMut<FinalSignalOutput<T>>)
    where
        T: Send + Sync + 'static + Clone + fmt::Debug,
    {
        output.0 = Some(value);
    }

    fn get_and_clear_final_output<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &mut World) -> Option<T> {
        world
            .get_resource_mut::<FinalSignalOutput<T>>()
            .and_then(|mut res| res.0.take())
    }

    #[test]
    fn mutable_vec_world_cleanup_owner_clones_emit_one_final_root() {
        let mut world = World::new();
        let source: MutableVec<i32> = (&mut world).into();
        let neighbor: MutableVec<i32> = (&mut world).into();
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
    fn mutable_vec_world_cleanup_builder_spawn_preserves_values_and_entity() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let source = MutableVec::builder().values([1, 2, 3]).spawn(&mut world);
        let entity = source.entity();
        assert_eq!(&*source.read(&world), &[1, 2, 3]);
        source.write(&mut world).push(4);
        assert_eq!(&*source.read(&world), &[1, 2, 3, 4]);

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_from_world_preserves_exact_entity() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let source = <MutableVec<i32> as FromWorld>::from_world(&mut world);
        let entity = source.entity();
        assert!(source.read(&world).is_empty());

        drop(source);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_builder_commands_apply_before_drop() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source = {
            let mut commands = Commands::new(&mut queue, &world);
            MutableVec::builder().values([4, 5]).spawnc(&mut commands)
        };
        let entity = source.entity();
        assert_eq!(source.owner.lifetime.cleanup.inbox_identity(), None);

        queue.apply(&mut world);
        assert_eq!(
            source.owner.lifetime.cleanup.inbox_identity(),
            cleanup::cleanup_inbox_identity(&world)
        );
        assert_eq!(&*source.read(&world), &[4, 5]);
        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_builder_commands_drop_before_apply() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source = {
            let mut commands = Commands::new(&mut queue, &world);
            MutableVec::builder().values([8, 9]).spawnc(&mut commands)
        };
        let entity = source.entity();

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        queue.apply(&mut world);
        assert!(world.get::<MutableVecData<i32>>(entity).is_some());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_from_commands_apply_before_drop() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source: MutableVec<i32> = {
            let mut commands = Commands::new(&mut queue, &world);
            (&mut commands).into()
        };
        let entity = source.entity();

        queue.apply(&mut world);
        assert_eq!(
            source.owner.lifetime.cleanup.inbox_identity(),
            cleanup::cleanup_inbox_identity(&world)
        );
        source.write(&mut world).push(3);
        assert_eq!(&*source.read(&world), &[3]);
        drop(source);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_from_commands_drop_before_apply() {
        let mut world = World::new();
        let neighbor = world.spawn_empty().id();
        let mut queue = CommandQueue::default();
        let source: MutableVec<i32> = {
            let mut commands = Commands::new(&mut queue, &world);
            (&mut commands).into()
        };
        let entity = source.entity();

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 0);
        queue.apply(&mut world);
        assert!(world.get::<MutableVecData<i32>>(entity).is_some());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
        assert!(world.get_entity(neighbor).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_discarded_commands_are_safe() {
        let reservation_world = World::new();
        let mut queue = CommandQueue::default();
        let source: MutableVec<i32> = {
            let mut commands = Commands::new(&mut queue, &reservation_world);
            (&mut commands).into()
        };
        drop(source);
        drop(queue);
        drop(reservation_world);

        let mut unrelated_world = World::new();
        let unrelated: MutableVec<i32> = (&mut unrelated_world).into();
        let unrelated_entity = unrelated.entity();
        cleanup::drain_world_cleanup(&mut unrelated_world);
        assert!(unrelated_world.get_entity(unrelated_entity).is_ok());
        drop(unrelated);
        cleanup::drain_world_cleanup(&mut unrelated_world);
    }

    #[test]
    fn mutable_vec_world_cleanup_worlds_with_colliding_entities_are_isolated() {
        let mut world_a = World::new();
        let mut world_b = World::new();
        let source_a: MutableVec<i32> = (&mut world_a).into();
        let source_b: MutableVec<i32> = (&mut world_b).into();
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
    fn mutable_vec_world_cleanup_plugin_b_preserves_app_a_pending_root() {
        let mut app_a = App::new();
        app_a.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        let source: MutableVec<i32> = app_a.world_mut().into();
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
    fn mutable_vec_world_cleanup_stale_generation_does_not_remove_replacement() {
        let mut world = World::new();
        let source: MutableVec<i32> = (&mut world).into();
        let stale = source.entity();
        let next_gen = world.despawn_no_free(stale).unwrap();
        let replacement = world.spawn_empty_at(next_gen).unwrap().id();
        assert_eq!(stale.index(), replacement.index());
        assert_ne!(stale.generation(), replacement.generation());

        drop(source);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(replacement).is_ok());
    }

    #[test]
    fn mutable_vec_world_cleanup_unregistered_source_is_one_root() {
        let mut world = World::new();
        let source: MutableVec<i32> = (&mut world).into();
        let entity = source.entity();
        world.entity_mut(entity).despawn();
        assert_eq!(cleanup::queued_message_count(&world), 0);

        drop(source);
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert_eq!(cleanup::queued_message_count(&world), 0);
    }

    #[test]
    fn mutable_vec_source_cache_normalization_preserves_identity() {
        fn assert_replayable<Signal: ReplayableSignalVec>(_signal: &Signal) {}

        let mut world = World::new();
        world.insert_resource(crate::graph::SignalGraphState::default());
        let source: MutableVec<i32> = (&mut world).into();
        let signal = source.signal_vec();
        assert_replayable(&signal);
        let cached = source.signal_vec().cache();
        let direct_claim = signal.register(&mut world);
        let direct_entity = direct_claim.entity().entity();
        let cached_claim = cached.register(&mut world);
        assert_eq!(cached_claim.entity().entity(), direct_entity);

        cached_claim.release(&mut world);
        direct_claim.release(&mut world);
    }

    #[test]
    fn mutable_vec_world_cleanup_signal_source_has_no_owner_cycle() {
        let mut world = World::new();
        let source: MutableVec<i32> = (&mut world).into();
        let entity = source.entity();
        let owner = Arc::downgrade(&source.owner);
        let lifetime = Arc::downgrade(&source.owner.lifetime);
        let signal = source.signal_vec();
        assert_eq!(Arc::strong_count(&source.owner), 1);

        drop(source);
        assert!(owner.upgrade().is_none());
        assert!(lifetime.upgrade().is_some());
        assert_eq!(cleanup::queued_message_count(&world), 0);
        drop(signal);
        assert!(lifetime.upgrade().is_none());
        assert_eq!(cleanup::queued_message_count(&world), 1);
        cleanup::drain_world_cleanup(&mut world);
        assert!(world.get_entity(entity).is_err());
    }

    #[test]
    fn mutable_vec_world_cleanup_drop_after_world_destruction_is_safe() {
        let source = {
            let mut world = World::new();
            let source: MutableVec<i32> = (&mut world).into();
            drop(world);
            source
        };

        drop(source);
    }

    #[test]
    fn test_for_each() {
        {
            // A comprehensive test for `SignalVecExt::for_each`. This test verifies that
            // the combinator correctly receives a batch of `VecDiff`s and transforms the
            // `SignalVec` into a standard `Signal`.

            // --- 1. Setup ---
            let mut app = create_test_app();

            // The output of our `for_each` system will be the full, reconstructed vector.
            app.init_resource::<FinalSignalOutput<Vec<String>>>();
            let source_vec = MutableVec::builder()
                .values(["a".to_string(), "b".to_string()])
                .spawn(app.world_mut());

            // This system reconstructs the state of the vec by applying the diffs it
            // receives. It then outputs the complete, current state. This allows us to
            // verify that `for_each` receives the diffs correctly and transforms the stream.
            let reconstructor_system = |In(diffs): In<Vec<VecDiff<String>>>, mut state: Local<Vec<String>>| {
                // Only emit an output if there were actual changes.
                if diffs.is_empty() {
                    return None;
                }
                for diff in diffs {
                    diff.apply_to_vec(&mut state);
                }
                // Output the new, complete state.
                Some(state.clone())
            };

            let handle = source_vec
                .signal_vec()
                .for_each(reconstructor_system)
                .map(capture_final_output::<Vec<String>>)
                .register(app.world_mut());

            // --- 2. Test Initial State ---
            // The initial `Replace` diff should be received and processed.
            app.update();
            let expected_initial_state = vec!["a".to_string(), "b".to_string()];
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(expected_initial_state),
                "Initial state was not reconstructed correctly by for_each"
            );

            // --- 3. Test Batched Mutations ---
            // Perform multiple operations before flushing to test batch processing.
            {
                let mut writer = source_vec.write(app.world_mut());
                writer.push("c".to_string()); // source: ["a", "b", "c"]
                writer.set(0, "A".to_string()); // source: ["A", "b", "c"]
                writer.remove(1); // source: ["A", "c"]
            }
            app.update();
            let expected_batched_state = vec!["A".to_string(), "c".to_string()];
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(expected_batched_state),
                "State after batched mutations was not reconstructed correctly"
            );

            // --- 4. Test No-Op Frame ---
            // If the source doesn't flush, the `for_each` system should not run,
            // and the output signal should not emit a new value.
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                None,
                "Signal emitted on a no-op frame"
            );

            // --- 5. Test Clear Operation ---
            // The `Clear` diff should result in an empty vector.
            source_vec.write(app.world_mut()).clear();
            app.update();
            let expected_cleared_state: Vec<String> = Vec::new();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(expected_cleared_state),
                "State after Clear was not reconstructed correctly"
            );

            // --- 6. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_empty_vec_first_push() {
        {
            // A push before the first replay is included in one authoritative snapshot
            // and is not replayed again as incremental history.

            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // Start with an empty vec
            let source_vec = MutableVec::from(app.world_mut());

            // Create a signal_vec and register it
            let signal = source_vec.signal_vec();
            let handle = signal.for_each(capture_signal_vec_output).register(app.world_mut());

            // Push the first element
            source_vec.write(app.world_mut()).push(42);
            app.update();

            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Replace { values: vec![42] }],
                "The first replay must snapshot the pushed value without a trailing Push"
            );

            // Push another element to verify normal operation continues
            source_vec.write(app.world_mut()).push(99);
            app.update();

            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Push { value: 99 });

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_vec_replay_initially_empty_then_push() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::<i32>::builder().spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());
        let mut downstream = Vec::new();

        app.update();
        let initial_diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(
            initial_diffs,
            vec![VecDiff::Replace { values: vec![] }],
            "an initially empty replay must emit one authoritative empty snapshot"
        );
        apply_diffs(&mut downstream, initial_diffs);
        assert!(downstream.is_empty());

        source.write(app.world_mut()).push(7);
        app.update();
        let later_diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(
            later_diffs,
            vec![VecDiff::Push { value: 7 }],
            "a mutation after the snapshot must remain an incremental diff"
        );
        apply_diffs(&mut downstream, later_diffs);
        assert_eq!(downstream, vec![7]);

        handle.release(app.world_mut());
    }

    #[test]
    fn test_vec_replay_filters_historical_prefix_and_forwards_multiple_newer_diffs() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());

        source.write(app.world_mut()).remove(0);
        let source_after_snapshot = source.clone();
        let handle = source
            .signal_vec()
            .for_each(move |In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
                world
                    .resource_mut::<SignalVecOutput<i32>>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::Replace { values } if values == &vec![20, 30]))
                {
                    let mut source = source_after_snapshot.write(world);
                    source.insert(1, 25);
                    source.push(40);
                }
            })
            .register(app.world_mut());

        app.update();
        let mut actual_diffs = get_and_clear_output::<i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_output::<i32>(app.world_mut()));
        let expected_diffs = vec![
            VecDiff::Replace { values: vec![20, 30] },
            VecDiff::InsertAt { index: 1, value: 25 },
            VecDiff::Push { value: 40 },
        ];
        let mut downstream = Vec::new();
        apply_diffs(&mut downstream, actual_diffs.clone());
        assert_eq!(
            actual_diffs, expected_diffs,
            "replay must skip the snapshotted RemoveAt prefix and forward every newer diff in order"
        );
        assert_eq!(downstream, vec![20, 25, 30, 40]);

        handle.release(app.world_mut());
    }

    #[test]
    fn test_vec_replay_snapshot_skips_pending_history_without_newer_diff() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::builder().values([10, 20]).spawn(app.world_mut());

        source.write(app.world_mut()).remove(0);
        let handle = source
            .signal_vec()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        app.update();
        let mut actual_diffs = get_and_clear_output::<i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_output::<i32>(app.world_mut()));
        let expected_diffs = vec![VecDiff::Replace { values: vec![20] }];
        let mut downstream = Vec::new();
        apply_diffs(&mut downstream, actual_diffs.clone());
        assert_eq!(
            actual_diffs, expected_diffs,
            "history already represented by the snapshot must not be emitted again"
        );
        assert_eq!(downstream, vec![20]);

        handle.release(app.world_mut());
    }

    #[test]
    fn test_nonempty_vec_initial_replace() {
        {
            // Test that when a MutableVec starts with values,
            // we get an initial Replace diff

            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // Start with a vec containing initial values
            let source_vec = MutableVec::builder().values([1, 2, 3]).spawn(app.world_mut());

            // Create a signal_vec and register it
            let signal = source_vec.signal_vec();
            let handle = signal.for_each(capture_signal_vec_output).register(app.world_mut());

            // First update should produce a Replace with initial values
            app.update();

            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Expected exactly one diff for initial state");
            assert_eq!(
                diffs[0],
                VecDiff::Replace { values: vec![1, 2, 3] },
                "Expected a Replace diff with initial values"
            );

            // Push another element to verify normal operation continues
            source_vec.write(app.world_mut()).push(4);
            app.update();

            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Push { value: 4 });

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn sort_by_forward_update_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::builder().values([1, 2, 3]).spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(i32, i32)>| left.cmp(&right))
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<i32>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        assert_eq!(
            downstream,
            vec![1, 2, 3],
            "initial downstream must be reconstructed solely from emitted diffs; initial_diffs={initial_diffs:?}, downstream={downstream:?}"
        );

        source.write(app.world_mut()).set(0, 4);
        app.update();
        let actual_diffs = get_and_clear_output::<i32>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::RemoveAt { index: 0 }, VecDiff::InsertAt { index: 2, value: 4 }];
        let expected_downstream = vec![2, 3, 4];
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by emitted/public reconstructed divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, expected_downstream={expected_downstream:?}, actual_downstream={downstream:?}"
        );
    }

    #[test]
    fn sort_by_key_forward_update_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let source = MutableVec::builder()
            .values([
                TaggedItem { key: 1, id: 10 },
                TaggedItem { key: 2, id: 20 },
                TaggedItem { key: 3, id: 30 },
            ])
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        let expected_initial = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 2, id: 20 },
            TaggedItem { key: 3, id: 30 },
        ];
        assert_eq!(
            downstream, expected_initial,
            "initial downstream must be reconstructed solely from emitted diffs; initial_diffs={initial_diffs:?}, downstream={downstream:?}"
        );

        let updated = TaggedItem { key: 4, id: 10 };
        source.write(app.world_mut()).set(0, updated.clone());
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![
            VecDiff::RemoveAt { index: 0 },
            VecDiff::InsertAt {
                index: 2,
                value: updated.clone(),
            },
        ];
        let expected_downstream = vec![TaggedItem { key: 2, id: 20 }, TaggedItem { key: 3, id: 30 }, updated];
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by_key emitted/public reconstructed divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, expected_downstream={expected_downstream:?}, actual_downstream={downstream:?}"
        );
    }

    #[test]
    fn sort_by_adjacent_forward_update_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<f32>>();
        let source = MutableVec::builder().values([1.0_f32, 2.0, 3.0]).spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(f32, f32)>| left.partial_cmp(&right).unwrap())
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<f32>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        assert_eq!(
            downstream,
            vec![1.0, 2.0, 3.0],
            "initial downstream must be reconstructed solely from emitted diffs; initial_diffs={initial_diffs:?}, downstream={downstream:?}"
        );

        source.write(app.world_mut()).set(0, 2.5);
        app.update();
        let actual_diffs = get_and_clear_output::<f32>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![
            VecDiff::RemoveAt { index: 0 },
            VecDiff::InsertAt { index: 1, value: 2.5 },
        ];
        let expected_downstream = vec![2.0, 2.5, 3.0];
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by adjacent-forward emitted/public reconstructed divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, expected_downstream={expected_downstream:?}, actual_downstream={downstream:?}"
        );
    }

    #[test]
    fn sort_by_key_adjacent_forward_update_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let source = MutableVec::builder()
            .values([
                TaggedItem { key: 10, id: 10 },
                TaggedItem { key: 20, id: 20 },
                TaggedItem { key: 30, id: 30 },
            ])
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        assert_eq!(
            downstream,
            vec![
                TaggedItem { key: 10, id: 10 },
                TaggedItem { key: 20, id: 20 },
                TaggedItem { key: 30, id: 30 },
            ],
            "initial downstream must be reconstructed solely from emitted diffs; initial_diffs={initial_diffs:?}, downstream={downstream:?}"
        );

        let updated = TaggedItem { key: 25, id: 10 };
        source.write(app.world_mut()).set(0, updated.clone());
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![
            VecDiff::RemoveAt { index: 0 },
            VecDiff::InsertAt {
                index: 1,
                value: updated.clone(),
            },
        ];
        let expected_downstream = vec![TaggedItem { key: 20, id: 20 }, updated, TaggedItem { key: 30, id: 30 }];
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by_key adjacent-forward emitted/public reconstructed divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, expected_downstream={expected_downstream:?}, actual_downstream={downstream:?}"
        );
    }

    #[test]
    fn sort_by_stable_equal_key_move_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let mut plain_source = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 2, id: 20 },
            TaggedItem { key: 1, id: 11 },
        ];
        let source = MutableVec::builder()
            .values(plain_source.clone())
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(TaggedItem, TaggedItem)>| left.key.cmp(&right.key))
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());

        source.write(app.world_mut()).move_item(1, 2);
        let unequal = plain_source.remove(1);
        plain_source.insert(2, unequal);
        app.update();
        let unequal_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        let mut unequal_oracle = plain_source.clone();
        unequal_oracle.sort_by_key(|item| item.key);
        let unequal_correct = unequal_diffs.is_empty() && downstream == unequal_oracle;
        assert!(
            unequal_correct,
            "sort_by unequal-key Move control diverged: expected_diffs=[], actual_diffs={unequal_diffs:?}, expected_downstream={unequal_oracle:?}, actual_downstream={downstream:?}, initial_diffs={initial_diffs:?}"
        );

        source.write(app.world_mut()).move_item(0, 1);
        let equal = plain_source.remove(0);
        plain_source.insert(1, equal);
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Move {
            old_index: 0,
            new_index: 1,
        }];
        let mut expected_downstream = plain_source.clone();
        expected_downstream.sort_by_key(|item| item.key);
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by stable equal-key Move emitted/public reconstructed divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, independently_stable_sorted_source={expected_downstream:?}, actual_downstream={downstream:?}, current_plain_source={plain_source:?}"
        );
    }

    #[test]
    fn sort_by_key_stable_equal_key_move_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let mut plain_source = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 2, id: 20 },
            TaggedItem { key: 1, id: 11 },
        ];
        let source = MutableVec::builder()
            .values(plain_source.clone())
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());

        source.write(app.world_mut()).move_item(1, 2);
        let unequal = plain_source.remove(1);
        plain_source.insert(2, unequal);
        app.update();
        let unequal_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        let mut unequal_oracle = plain_source.clone();
        unequal_oracle.sort_by_key(|item| item.key);
        let unequal_correct = unequal_diffs.is_empty() && downstream == unequal_oracle;
        assert!(
            unequal_correct,
            "sort_by_key unequal-key Move control diverged: expected_diffs=[], actual_diffs={unequal_diffs:?}, expected_downstream={unequal_oracle:?}, actual_downstream={downstream:?}, initial_diffs={initial_diffs:?}"
        );

        source.write(app.world_mut()).move_item(0, 1);
        let equal = plain_source.remove(0);
        plain_source.insert(1, equal);
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Move {
            old_index: 0,
            new_index: 1,
        }];
        let mut expected_downstream = plain_source.clone();
        expected_downstream.sort_by_key(|item| item.key);
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by_key stable equal-key Move emitted/public reconstructed divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, independently_stable_sorted_source={expected_downstream:?}, actual_downstream={downstream:?}, current_plain_source={plain_source:?}"
        );
    }

    #[test]
    fn sort_by_backward_equal_key_move_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let mut plain_source = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 1, id: 11 },
            TaggedItem { key: 2, id: 20 },
        ];
        let source = MutableVec::builder()
            .values(plain_source.clone())
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(TaggedItem, TaggedItem)>| left.key.cmp(&right.key))
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        assert_eq!(
            downstream, plain_source,
            "initial stable sort_by state was not reconstructed from emitted diffs; initial_diffs={initial_diffs:?}"
        );

        source.write(app.world_mut()).move_item(1, 0);
        let moved = plain_source.remove(1);
        plain_source.insert(0, moved);
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Move {
            old_index: 1,
            new_index: 0,
        }];
        let mut expected_downstream = plain_source.clone();
        expected_downstream.sort_by_key(|item| item.key);
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by backward equal-key Move divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, independently_stable_sorted_source={expected_downstream:?}, actual_downstream={downstream:?}, current_plain_source={plain_source:?}"
        );
    }

    #[test]
    fn sort_by_key_backward_equal_key_move_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let mut plain_source = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 1, id: 11 },
            TaggedItem { key: 2, id: 20 },
        ];
        let source = MutableVec::builder()
            .values(plain_source.clone())
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        let initial_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, initial_diffs.clone());
        assert_eq!(
            downstream, plain_source,
            "initial stable sort_by_key state was not reconstructed from emitted diffs; initial_diffs={initial_diffs:?}"
        );

        source.write(app.world_mut()).move_item(1, 0);
        let moved = plain_source.remove(1);
        plain_source.insert(0, moved);
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Move {
            old_index: 1,
            new_index: 0,
        }];
        let mut expected_downstream = plain_source.clone();
        expected_downstream.sort_by_key(|item| item.key);
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by_key backward equal-key Move divergence: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, independently_stable_sorted_source={expected_downstream:?}, actual_downstream={downstream:?}, current_plain_source={plain_source:?}"
        );
    }

    #[test]
    fn sort_by_post_equal_key_move_update_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let mut plain_source = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 1, id: 11 },
            TaggedItem { key: 2, id: 20 },
        ];
        let source = MutableVec::builder()
            .values(plain_source.clone())
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(TaggedItem, TaggedItem)>| left.key.cmp(&right.key))
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        apply_diffs(&mut downstream, get_and_clear_output::<TaggedItem>(app.world_mut()));

        source.write(app.world_mut()).move_item(0, 1);
        let moved = plain_source.remove(0);
        plain_source.insert(1, moved);
        app.update();
        let move_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, move_diffs.clone());
        assert_eq!(
            move_diffs,
            vec![VecDiff::Move {
                old_index: 0,
                new_index: 1,
            }],
            "equal-key setup Move must use stable downstream coordinates"
        );

        let updated = TaggedItem { key: 3, id: 10 };
        source.write(app.world_mut()).set(1, updated.clone());
        plain_source[1] = updated.clone();
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![
            VecDiff::RemoveAt { index: 1 },
            VecDiff::InsertAt {
                index: 2,
                value: updated,
            },
        ];
        let mut expected_downstream = plain_source.clone();
        expected_downstream.sort_by_key(|item| item.key);
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by tracking invalid after equal-key Move and later UpdateAt: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, independently_stable_sorted_source={expected_downstream:?}, actual_downstream={downstream:?}, current_plain_source={plain_source:?}"
        );
    }

    #[test]
    fn sort_by_key_post_equal_key_move_update_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let mut plain_source = vec![
            TaggedItem { key: 1, id: 10 },
            TaggedItem { key: 1, id: 11 },
            TaggedItem { key: 2, id: 20 },
        ];
        let source = MutableVec::builder()
            .values(plain_source.clone())
            .spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut downstream = Vec::new();
        app.update();
        apply_diffs(&mut downstream, get_and_clear_output::<TaggedItem>(app.world_mut()));

        source.write(app.world_mut()).move_item(0, 1);
        let moved = plain_source.remove(0);
        plain_source.insert(1, moved);
        app.update();
        let move_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, move_diffs.clone());
        assert_eq!(
            move_diffs,
            vec![VecDiff::Move {
                old_index: 0,
                new_index: 1,
            }],
            "equal-key setup Move must use stable downstream coordinates"
        );

        let updated = TaggedItem { key: 3, id: 10 };
        source.write(app.world_mut()).set(1, updated.clone());
        plain_source[1] = updated.clone();
        app.update();
        let actual_diffs = get_and_clear_output::<TaggedItem>(app.world_mut());
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![
            VecDiff::RemoveAt { index: 1 },
            VecDiff::InsertAt {
                index: 2,
                value: updated,
            },
        ];
        let mut expected_downstream = plain_source.clone();
        expected_downstream.sort_by_key(|item| item.key);
        let correct = actual_diffs == expected_diffs && downstream == expected_downstream;

        handle.release(app.world_mut());
        assert!(
            correct,
            "sort_by_key tracking invalid after equal-key Move and later UpdateAt: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, independently_stable_sorted_source={expected_downstream:?}, actual_downstream={downstream:?}, current_plain_source={plain_source:?}"
        );
    }

    #[test]
    fn replay_mutated_before_registration_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());

        source.write(app.world_mut()).remove(0);
        let source_after_snapshot = source.clone();
        let subscriber_handle = source
            .signal_vec()
            .for_each(move |In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
                world
                    .resource_mut::<SignalVecOutput<i32>>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::Replace { values } if values == &vec![20, 30]))
                {
                    source_after_snapshot.write(world).push(40);
                }
            })
            .register(app.world_mut());

        app.update();
        let mut actual_diffs = get_and_clear_output::<i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_output::<i32>(app.world_mut()));
        let mut downstream = Vec::new();
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Replace { values: vec![20, 30] }, VecDiff::Push { value: 40 }];
        let authoritative_plain_source = vec![20, 30, 40];
        let correct = actual_diffs == expected_diffs && downstream == authoritative_plain_source;

        subscriber_handle.release(app.world_mut());
        assert!(
            correct,
            "replay snapshot/history overlap after positional mutation before registration: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, authoritative_plain_source={authoritative_plain_source:?}, public_reconstructed_downstream={downstream:?}"
        );
    }

    #[test]
    fn replay_nonempty_mutated_after_registration_before_first_replay_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());

        let source_after_snapshot = source.clone();
        let subscriber_handle = source
            .signal_vec()
            .for_each(move |In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
                world
                    .resource_mut::<SignalVecOutput<i32>>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::Replace { values } if values == &vec![20, 30]))
                {
                    source_after_snapshot.write(world).push(40);
                }
            })
            .register(app.world_mut());
        app.world_mut().resource_mut::<SignalVecOutput<i32>>().0.clear();
        source.write(app.world_mut()).remove(0);

        app.update();
        let mut actual_diffs = get_and_clear_output::<i32>(app.world_mut());
        app.update();
        actual_diffs.extend(get_and_clear_output::<i32>(app.world_mut()));
        let mut downstream = Vec::new();
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Replace { values: vec![20, 30] }, VecDiff::Push { value: 40 }];
        let authoritative_plain_source = vec![20, 30, 40];
        let correct = actual_diffs == expected_diffs && downstream == authoritative_plain_source;

        subscriber_handle.release(app.world_mut());
        assert!(
            correct,
            "nonempty replay snapshot/history overlap after mutation post-registration and pre-first-replay: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, authoritative_plain_source={authoritative_plain_source:?}, public_reconstructed_downstream={downstream:?}"
        );
    }

    #[test]
    fn replay_empty_mutated_after_registration_before_first_replay_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source = MutableVec::<i32>::builder().spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        app.world_mut().resource_mut::<SignalVecOutput<i32>>().0.clear();
        source.write(app.world_mut()).push(7);
        app.update();
        let actual_diffs = get_and_clear_output::<i32>(app.world_mut());
        let mut downstream = Vec::new();
        apply_diffs(&mut downstream, actual_diffs.clone());
        let expected_diffs = vec![VecDiff::Replace { values: vec![7] }];
        let authoritative_plain_source = vec![7];
        let correct = actual_diffs == expected_diffs && downstream == authoritative_plain_source;

        handle.release(app.world_mut());
        assert!(
            correct,
            "initially-empty replay must emit one authoritative snapshot with no trailing history: expected_diffs={expected_diffs:?}, actual_diffs={actual_diffs:?}, authoritative_plain_source={authoritative_plain_source:?}, public_reconstructed_downstream={downstream:?}"
        );
    }

    #[test]
    fn replay_two_subscribers_pending_batch_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<CollectionCorrectnessCaptureA>();
        app.init_resource::<CollectionCorrectnessCaptureB>();
        let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());

        let handle_a = source
            .signal_vec()
            .for_each(move |In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
                world.resource_mut::<CollectionCorrectnessCaptureA>().0.extend(diffs);
            })
            .register(app.world_mut());
        app.update();
        let initial_a = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessCaptureA>().0);
        let mut downstream_a = Vec::new();
        apply_diffs(&mut downstream_a, initial_a.clone());
        assert_eq!(
            downstream_a,
            vec![10, 20, 30],
            "subscriber A initial state was not reconstructed from emitted diffs; initial_a={initial_a:?}"
        );

        app.world_mut()
            .resource_mut::<CollectionCorrectnessCaptureA>()
            .0
            .clear();
        app.world_mut()
            .resource_mut::<CollectionCorrectnessCaptureB>()
            .0
            .clear();
        source.write(app.world_mut()).remove(0);
        let source_after_b_snapshot = source.clone();
        let handle_b = source
            .signal_vec()
            .for_each(move |In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
                world
                    .resource_mut::<CollectionCorrectnessCaptureB>()
                    .0
                    .extend(diffs.iter().cloned());
                if diffs
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::Replace { values } if values == &vec![20, 30]))
                {
                    source_after_b_snapshot.write(world).push(40);
                }
            })
            .register(app.world_mut());

        app.update();
        app.update();
        let actual_diffs_a = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessCaptureA>().0);
        let actual_diffs_b = core::mem::take(&mut app.world_mut().resource_mut::<CollectionCorrectnessCaptureB>().0);
        apply_diffs(&mut downstream_a, actual_diffs_a.clone());
        let mut downstream_b = Vec::new();
        apply_diffs(&mut downstream_b, actual_diffs_b.clone());
        let expected_diffs_a = vec![VecDiff::RemoveAt { index: 0 }, VecDiff::Push { value: 40 }];
        let expected_diffs_b = vec![VecDiff::Replace { values: vec![20, 30] }, VecDiff::Push { value: 40 }];
        let authoritative_plain_source = vec![20, 30, 40];
        let correct = actual_diffs_a == expected_diffs_a
            && actual_diffs_b == expected_diffs_b
            && downstream_a == authoritative_plain_source
            && downstream_b == authoritative_plain_source;

        handle_b.release(app.world_mut());
        handle_a.release(app.world_mut());
        assert!(
            correct,
            "per-subscriber replay cursor divergence: initial_a={initial_a:?}, expected_subscriber_a_diffs={expected_diffs_a:?}, actual_subscriber_a_diffs={actual_diffs_a:?}, subscriber_a_downstream={downstream_a:?}, expected_subscriber_b_diffs={expected_diffs_b:?}, actual_subscriber_b_diffs={actual_diffs_b:?}, subscriber_b_downstream={downstream_b:?}, authoritative_plain_source={authoritative_plain_source:?}"
        );
    }

    #[test]
    fn sort_by_boundary_table_collection_correctness_oracle() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let source = MutableVec::<TaggedItem>::builder().spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(TaggedItem, TaggedItem)>| left.key.cmp(&right.key))
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        run_collection_correctness_boundary_table(&mut app, &source, sort_tagged_by_comparator);

        handle.release(app.world_mut());
    }

    #[test]
    fn sort_by_key_boundary_table_collection_correctness_oracle() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let source = MutableVec::<TaggedItem>::builder().spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        run_collection_correctness_boundary_table(&mut app, &source, sort_tagged_by_key);

        handle.release(app.world_mut());
    }

    #[test]
    fn sort_by_fixed_seed_stream_collection_correctness_oracle() {
        const SEED: u64 = 0x5eed_c0de_51ab_1e55;
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let source = MutableVec::<TaggedItem>::builder().spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by(|In((left, right)): In<(TaggedItem, TaggedItem)>| left.key.cmp(&right.key))
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        run_generated_collection_correctness_oracle(&mut app, &source, sort_tagged_by_comparator, SEED);

        handle.release(app.world_mut());
    }

    #[test]
    fn sort_by_key_fixed_seed_stream_collection_correctness_oracle() {
        const SEED: u64 = 0x5eed_c0de_7e57_0a11;
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<TaggedItem>>();
        let source = MutableVec::<TaggedItem>::builder().spawn(app.world_mut());
        let handle = source
            .signal_vec()
            .sort_by_key(|In(item): In<TaggedItem>| item.key)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        run_generated_collection_correctness_oracle(&mut app, &source, sort_tagged_by_key, SEED);

        handle.release(app.world_mut());
    }

    #[test]
    fn vector_authoritative_replay_collection_correctness_oracle() {
        {
            let mut app = create_test_app();
            app.init_resource::<CollectionCorrectnessCaptureA>();
            app.init_resource::<CollectionCorrectnessCaptureB>();
            let source = MutableVec::<i32>::builder().spawn(app.world_mut());
            let handle = source
                .signal_vec()
                .for_each(
                    |In(diffs): In<Vec<VecDiff<i32>>>, mut capture: ResMut<CollectionCorrectnessCaptureA>| {
                        capture.0.extend(diffs);
                    },
                )
                .register(app.world_mut());

            let mut plain_source = Vec::new();
            source.write(app.world_mut()).push(7);
            plain_source.push(7);
            source.write(app.world_mut()).insert(0, 6);
            plain_source.insert(0, 6);

            let mut downstream_a = Vec::new();
            let mut history_a = Vec::new();
            let mut downstream_b = Vec::new();
            let mut history_b = Vec::new();
            let (first_a, _) = settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                false,
                &mut downstream_b,
                &mut history_b,
                "initially empty source mutated before first replay",
            );
            assert_eq!(
                first_a,
                vec![VecDiff::Replace {
                    values: plain_source.clone(),
                }],
                "initially empty subscriber must receive one authoritative Replace and no snapshotted incremental suffix"
            );

            handle.release(app.world_mut());
        }

        {
            let mut app = create_test_app();
            app.init_resource::<CollectionCorrectnessCaptureA>();
            app.init_resource::<CollectionCorrectnessCaptureB>();
            let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());
            let mut plain_source = vec![10, 20, 30];

            source.write(app.world_mut()).remove(0);
            plain_source.remove(0);
            let handle_a = source
                .signal_vec()
                .for_each(
                    |In(diffs): In<Vec<VecDiff<i32>>>, mut capture: ResMut<CollectionCorrectnessCaptureA>| {
                        capture.0.extend(diffs);
                    },
                )
                .register(app.world_mut());
            source.write(app.world_mut()).insert(1, 25);
            plain_source.insert(1, 25);

            let mut downstream_a = Vec::new();
            let mut history_a = Vec::new();
            let mut downstream_b = Vec::new();
            let mut history_b = Vec::new();
            let (first_a, _) = settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                false,
                &mut downstream_b,
                &mut history_b,
                "populated source with history around subscriber A registration",
            );
            assert_eq!(
                first_a,
                vec![VecDiff::Replace {
                    values: plain_source.clone(),
                }],
                "subscriber A first diff must be exactly one authoritative Replace"
            );

            {
                let mut writer = source.write(app.world_mut());
                writer.push(40);
                writer.set(0, 21);
                writer.move_item(1, 2);
            }
            plain_source.push(40);
            plain_source[0] = 21;
            let moved = plain_source.remove(1);
            plain_source.insert(2, moved);
            settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                false,
                &mut downstream_b,
                &mut history_b,
                "subscriber A incremental batch",
            );
            let a_prefix_before_b_registration = history_a.clone();

            source.write(app.world_mut()).remove(3);
            plain_source.remove(3);
            let handle_b = source
                .signal_vec()
                .for_each(
                    |In(diffs): In<Vec<VecDiff<i32>>>, mut capture: ResMut<CollectionCorrectnessCaptureB>| {
                        capture.0.extend(diffs);
                    },
                )
                .register(app.world_mut());
            source.write(app.world_mut()).insert(0, 5);
            plain_source.insert(0, 5);

            let (crossing_a, first_b) = settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                true,
                &mut downstream_b,
                &mut history_b,
                "subscriber B registration around pending history",
            );
            assert_eq!(
                crossing_a,
                vec![VecDiff::RemoveAt { index: 3 }, VecDiff::InsertAt { index: 0, value: 5 }],
                "subscriber A must retain its cursor and receive the pending-history crossing as incrementals"
            );
            assert_eq!(
                first_b,
                vec![VecDiff::Replace {
                    values: plain_source.clone(),
                }],
                "subscriber B must start at an independent authoritative Replace baseline"
            );
            assert_eq!(
                &history_a[..a_prefix_before_b_registration.len()],
                a_prefix_before_b_registration.as_slice(),
                "registering subscriber B changed subscriber A's established public-diff prefix"
            );
            assert!(matches!(history_a.first(), Some(VecDiff::Replace { .. })));
            assert!(matches!(history_b.first(), Some(VecDiff::Replace { .. })));

            let a_common_suffix_start = history_a.len();
            let b_common_suffix_start = history_b.len();

            source.write(app.world_mut()).replace(Vec::<i32>::new());
            plain_source.clear();
            settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                true,
                &mut downstream_b,
                &mut history_b,
                "active subscribers empty Replace",
            );

            source.write(app.world_mut()).replace(vec![3, 1, 2]);
            plain_source = vec![3, 1, 2];
            settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                true,
                &mut downstream_b,
                &mut history_b,
                "active subscribers nonempty Replace",
            );

            {
                let mut writer = source.write(app.world_mut());
                writer.push(4);
                writer.pop();
                writer.insert(0, 0);
                writer.remove(2);
                writer.set(1, 30);
                writer.move_item(2, 0);
            }
            plain_source.push(4);
            plain_source.pop();
            plain_source.insert(0, 0);
            plain_source.remove(2);
            plain_source[1] = 30;
            let moved = plain_source.remove(2);
            plain_source.insert(0, moved);
            settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                true,
                &mut downstream_b,
                &mut history_b,
                "active subscribers all-incremental-types batch",
            );

            source.write(app.world_mut()).clear();
            plain_source.clear();
            settle_vector_replay_collection_correctness_oracle(
                &mut app,
                &plain_source,
                true,
                &mut downstream_a,
                &mut history_a,
                true,
                &mut downstream_b,
                &mut history_b,
                "active subscribers Clear",
            );

            let common_suffix_a = &history_a[a_common_suffix_start..];
            let common_suffix_b = &history_b[b_common_suffix_start..];
            assert_eq!(
                common_suffix_a, common_suffix_b,
                "active subscribers with different baselines/cursors must receive the same settled public-diff suffix; history_a={history_a:?}, history_b={history_b:?}"
            );
            assert!(
                matches!(common_suffix_a.first(), Some(VecDiff::Replace { values }) if values.is_empty()),
                "common suffix must begin with the empty Replace; suffix={common_suffix_a:?}"
            );
            assert!(
                common_suffix_a
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::Replace { values } if !values.is_empty())),
                "common suffix omitted nonempty Replace; suffix={common_suffix_a:?}"
            );
            assert!(common_suffix_a.iter().any(|diff| matches!(diff, VecDiff::Push { .. })));
            assert!(common_suffix_a.iter().any(|diff| matches!(diff, VecDiff::Pop)));
            assert!(
                common_suffix_a
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::InsertAt { .. }))
            );
            assert!(
                common_suffix_a
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::RemoveAt { .. }))
            );
            assert!(
                common_suffix_a
                    .iter()
                    .any(|diff| matches!(diff, VecDiff::UpdateAt { .. }))
            );
            assert!(common_suffix_a.iter().any(|diff| matches!(diff, VecDiff::Move { .. })));
            assert!(common_suffix_a.iter().any(|diff| matches!(diff, VecDiff::Clear)));

            handle_b.release(app.world_mut());
            handle_a.release(app.world_mut());
        }
    }

    #[test]
    fn test_map() {
        {
            // A comprehensive test for `SignalVecExt::map`, covering all `VecDiff`
            // types in both success and system-failure scenarios.

            // --- 1. Setup ---
            let mut app = create_test_app();

            // The output `SignalVec` will contain `String`s.
            app.init_resource::<SignalVecOutput<String>>();

            // The source vector that we will mutate.
            let source_vec = MutableVec::builder().values([1, 10]).spawn(app.world_mut());

            // Define a resource that the mapping system will depend on. This allows us to
            // test system failure by removing the resource.
            #[derive(Resource)]
            struct MapFactor(String);

            // The mapping system: appends a factor from the resource to the number.
            let mapping_system = |In(val): In<i32>, factor: Res<MapFactor>| format!("{}{}", val, factor.0);

            // The signal chain under test.
            let mapped_signal = source_vec.signal_vec().map(mapping_system);

            // Register the final signal to a system that captures its output diffs.
            let handle = mapped_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // A local copy of the mapped vector's state to verify against.
            let mut current_state: Vec<String> = vec![];

            // --- 2. Test Success Scenarios ---
            // First, test all diff types when the mapping system is guaranteed to succeed.
            app.insert_resource(MapFactor("x".to_string()));

            // Test: Initial `Replace`
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec!["1x".to_string(), "10x".to_string()]
                },
                "Initial `Replace` was incorrect."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["1x".to_string(), "10x".to_string()]);

            // Test: `Push`
            source_vec.write(app.world_mut()).push(4);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Push {
                    value: "4x".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["1x".to_string(), "10x".to_string(), "4x".to_string()]
            );

            // Test: `InsertAt`
            source_vec.write(app.world_mut()).insert(1, 5);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: "5x".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["1x".to_string(), "5x".to_string(), "10x".to_string(), "4x".to_string()]
            );

            // Test: `UpdateAt`
            source_vec.write(app.world_mut()).set(0, 99);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: "99x".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["99x".to_string(), "5x".to_string(), "10x".to_string(), "4x".to_string()]
            );

            // Test: `RemoveAt` (passes through without mapping)
            source_vec.write(app.world_mut()).remove(2); // removes 10
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["99x".to_string(), "5x".to_string(), "4x".to_string()]
            );

            // Test: `Pop` (passes through without mapping)
            source_vec.write(app.world_mut()).pop(); // removes 4
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Pop);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["99x".to_string(), "5x".to_string()]);

            // Test: `Move` (passes through without mapping)
            source_vec.write(app.world_mut()).move_item(1, 0); // moves 5 to front
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Move {
                    old_index: 1,
                    new_index: 0
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["5x".to_string(), "99x".to_string()]);

            // Test: `Clear` (passes through without mapping)
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 3. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    #[should_panic(expected = "SignalVec::map item system failed or skipped")]
    fn test_map_panics_on_skipped_item_system() {
        #[derive(Resource)]
        struct MissingFactor;

        let mut app = create_test_app();
        let source_vec = MutableVec::builder().values([1i32]).spawn(app.world_mut());
        let _handle = source_vec
            .signal_vec()
            .map(|In(x): In<i32>, _factor: Res<MissingFactor>| x)
            .register(app.world_mut());
        app.update();
    }

    #[test]
    #[should_panic(expected = "SignalVec::sort_by comparator system failed or skipped")]
    fn test_sort_by_panics_on_skipped_comparator_system() {
        #[derive(Resource)]
        struct MissingOrder;

        let mut app = create_test_app();
        app.insert_resource(MissingOrder);
        let source_vec = MutableVec::builder().values([1i32, 3]).spawn(app.world_mut());
        let _handle = source_vec
            .signal_vec()
            .sort_by(|In((a, b)): In<(i32, i32)>, _ord: Res<MissingOrder>| a.cmp(&b))
            .register(app.world_mut());
        app.update();

        app.world_mut().remove_resource::<MissingOrder>();
        source_vec.write(app.world_mut()).push(2);
        app.update();
    }

    #[test]
    fn test_map_in() {
        {
            // A comprehensive test for `SignalVecExt::map_in`, covering all `VecDiff`
            // types and verifying it correctly uses state captured by the closure.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<String>>();

            let source_vec = MutableVec::builder().values([1, 10, 100]).spawn(app.world_mut());

            // A variable to be captured by the `map_in` closure.
            // We'll use this to demonstrate that the closure's captured state is respected
            // and can change over time.
            let mapping_factor = Arc::new(RwLock::new("a".to_string()));

            // The signal chain under test, using `map_in` with a `move` closure.
            let mapped_signal = source_vec.signal_vec().map_in(clone!((mapping_factor) move |val: i32| {
                let factor = mapping_factor.read().unwrap();
                format!("{}{}", val, *factor)
            }));

            let handle = mapped_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // A local copy of the mapped vector's state to verify against.
            let mut current_state: Vec<String> = vec![];

            // --- 2. Test Scenarios ---

            // Test: Initial `Replace`
            // The first update should replay the initial state of the source vector,
            // mapped with the initial factor "a".
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec!["1a".to_string(), "10a".to_string(), "100a".to_string()]
                },
                "Initial `Replace` was incorrect."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["1a".to_string(), "10a".to_string(), "100a".to_string()]
            );

            // --- Change the captured state ---
            // Modify the captured factor. Subsequent mapping operations should use this
            // new value.
            *mapping_factor.write().unwrap() = "b".to_string();

            // Test: `Push` using the new factor "b"
            source_vec.write(app.world_mut()).push(4);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Push {
                    value: "4b".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![
                    "1a".to_string(),
                    "10a".to_string(),
                    "100a".to_string(),
                    "4b".to_string()
                ]
            );

            // Test: `InsertAt` using the new factor "b"
            source_vec.write(app.world_mut()).insert(1, 5);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: "5b".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![
                    "1a".to_string(),
                    "5b".to_string(),
                    "10a".to_string(),
                    "100a".to_string(),
                    "4b".to_string()
                ]
            );

            // Test: `UpdateAt` using the new factor "b"
            source_vec.write(app.world_mut()).set(0, 99);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: "99b".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![
                    "99b".to_string(),
                    "5b".to_string(),
                    "10a".to_string(),
                    "100a".to_string(),
                    "4b".to_string()
                ]
            );

            // --- Test pass-through diffs ---
            // These diffs don't involve mapping values, so they are simply forwarded.

            // Test: `RemoveAt`
            source_vec.write(app.world_mut()).remove(2); // removes 10
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![
                    "99b".to_string(),
                    "5b".to_string(),
                    "100a".to_string(),
                    "4b".to_string()
                ]
            );

            // Test: `Pop`
            source_vec.write(app.world_mut()).pop(); // removes 4
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Pop);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["99b".to_string(), "5b".to_string(), "100a".to_string()]
            );

            // Test: `Move`
            source_vec.write(app.world_mut()).move_item(1, 0); // moves 5 to front
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Move {
                    old_index: 1,
                    new_index: 0
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["5b".to_string(), "99b".to_string(), "100a".to_string()]
            );

            // Test: `Clear`
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 3. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_map_in_ref() {
        {
            // A comprehensive test for `SignalVecExt::map_in_ref`, covering all `VecDiff`
            // types and verifying it correctly uses a closure that takes a reference.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<String>>();

            // The source vector contains integers.
            let source_vec = MutableVec::builder().values([10, 20]).spawn(app.world_mut());

            // The signal chain under test, using `map_in_ref` with a function pointer
            // that operates on a reference (`&i32`). This is an idiomatic use case.
            let mapped_signal = source_vec.signal_vec().map_in_ref(ToString::to_string);

            let handle = mapped_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // A local copy of the mapped vector's state to verify against.
            let mut current_state: Vec<String> = vec![];

            // --- 2. Test Scenarios ---

            // Test: Initial `Replace`
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec!["10".to_string(), "20".to_string()]
                },
                "Initial `Replace` was incorrect."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["10".to_string(), "20".to_string()]);

            // Test: `Push`
            source_vec.write(app.world_mut()).push(30);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Push {
                    value: "30".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["10".to_string(), "20".to_string(), "30".to_string()]
            );

            // Test: `InsertAt`
            source_vec.write(app.world_mut()).insert(1, 5);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: "5".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["10".to_string(), "5".to_string(), "20".to_string(), "30".to_string()]
            );

            // Test: `UpdateAt`
            source_vec.write(app.world_mut()).set(0, 99);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: "99".to_string()
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec!["99".to_string(), "5".to_string(), "20".to_string(), "30".to_string()]
            );

            // --- Test pass-through diffs ---
            // These diffs don't involve mapping values, so they are simply forwarded.

            // Test: `RemoveAt`
            source_vec.write(app.world_mut()).remove(2); // removes 20
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["99".to_string(), "5".to_string(), "30".to_string()]);

            // Test: `Pop`
            source_vec.write(app.world_mut()).pop(); // removes 30
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Pop);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["99".to_string(), "5".to_string()]);

            // Test: `Move`
            source_vec.write(app.world_mut()).move_item(1, 0); // moves 5 to front
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Move {
                    old_index: 1,
                    new_index: 0
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["5".to_string(), "99".to_string()]);

            // Test: `Clear`
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 3. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    /// This test provides comprehensive coverage for `SignalVecExt::map_signal`.
    ///
    /// It verifies that `map_signal` correctly transforms a `SignalVec<A>` into a
    /// `SignalVec<B>` by creating an inner signal for each item. The test ensures:
    ///
    /// 1. **Initial State**: A `Replace` diff is correctly produced with the initial values from
    ///    all newly created inner signals.
    ///
    /// 2. **Inner Signal Updates**: An `UpdateAt` diff is emitted when an inner signal's value
    ///    changes.
    ///
    /// 3. **Source `VecDiff` Handling**: All `VecDiff` types from the source `SignalVec` (`Push`,
    ///    `Pop`, `InsertAt`, `RemoveAt`, `UpdateAt`, `Move`, `Clear`) are translated correctly into
    ///    the output `SignalVec`. This includes creating new inner signals for new items and
    ///    cleaning up signals for removed items.
    ///
    /// 4. **Signal Switching**: When an item in the source vector is updated (via `set`), the old
    ///    inner signal is properly cleaned up, and a new one is created for the new item, with the
    ///    output vector reflecting this change.
    ///
    /// 5. **Cleanup Verification**: After an item is removed or replaced, the test confirms that
    ///    updates from the old, now-orphaned inner signal are no longer propagated.
    #[test]
    fn test_map_signal() {
        {
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<Name>>();

            // Setup: Create entities with `Name` components that our signals will track.
            let entity_a = app.world_mut().spawn(Name::new("Alice")).id();
            let entity_b = app.world_mut().spawn(Name::new("Bob")).id();

            // The source vec contains entities. The goal is to create a derived vec that
            // contains the _names_ of these entities.
            let entity_vec = MutableVec::builder()
                .values([entity_a, entity_b])
                .spawn(app.world_mut());

            // This "factory" system takes an entity and creates a signal that tracks its
            // `Name`.
            let factory_system = |In(entity): In<Entity>| signal::from_component::<Name>(entity).dedupe();

            // Apply `map_signal` to transform the SignalVec`<Entity>` into a
            // SignalVec`<Name>`.
            let name_vec_signal = entity_vec.signal_vec().map_signal(factory_system);
            let handle = name_vec_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // Test 1: Initial State.
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial state should produce one Replace diff");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec![Name::new("Alice"), Name::new("Bob")]
                },
                "Initial state is incorrect"
            );

            // Test 2: Inner Signal Update.
            *app.world_mut().get_mut::<Name>(entity_a).unwrap() = Name::new("Alicia");
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Name change should produce one Update diff");
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: Name::new("Alicia"),
                },
                "Update diff is incorrect"
            );

            // Test 3: No change should produce no diff.
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert!(diffs.is_empty(), "No change should produce no diffs");

            // Test 4: Source SignalVec Push.
            let entity_c = app.world_mut().spawn(Name::new("Charlie")).id();
            entity_vec.write(app.world_mut()).push(entity_c);
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Push should produce one diff");
            assert_eq!(
                diffs[0],
                VecDiff::Push {
                    value: Name::new("Charlie")
                },
                "Push diff is incorrect"
            );

            // Test 5: Source SignalVec InsertAt.
            let entity_d = app.world_mut().spawn(Name::new("David")).id();
            entity_vec.write(app.world_mut()).insert(1, entity_d);
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "InsertAt should produce one diff");
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: Name::new("David"),
                },
                "InsertAt diff is incorrect"
            );

            // Test 6: Source SignalVec RemoveAt. Remove Bob
            entity_vec.write(app.world_mut()).remove(2);
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "RemoveAt should produce one diff");
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 }, "RemoveAt diff is incorrect");

            // Verify cleanup by ensuring updates to the removed entity's signal are ignored.
            *app.world_mut().get_mut::<Name>(entity_b).unwrap() = Name::new("Robert");
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert!(diffs.is_empty(), "Update on removed entity should not produce a diff");

            // Test 7: Source SignalVec UpdateAt (switching the underlying signal).
            let entity_e = app.world_mut().spawn(Name::new("Eve")).id();

            // Replace Alicia with Eve
            entity_vec.write(app.world_mut()).set(0, entity_e);
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Set/UpdateAt should produce one diff");
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: Name::new("Eve"),
                },
                "Update-to-new-entity diff is incorrect"
            );

            // Verify cleanup of old signal
            *app.world_mut().get_mut::<Name>(entity_a).unwrap() = Name::new("Alicia_v2");
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert!(
                diffs.is_empty(),
                "Update on old, replaced entity should not produce a diff"
            );

            // Verify new signal is active
            *app.world_mut().get_mut::<Name>(entity_e).unwrap() = Name::new("Evelyn");
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Update on new entity should produce a diff");
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: Name::new("Evelyn"),
                },
                "Update on new entity is incorrect"
            );

            // Test 8: Source SignalVec Move. Move Charlie to front
            entity_vec.write(app.world_mut()).move_item(2, 0);
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Move should produce one diff");
            assert_eq!(
                diffs[0],
                VecDiff::Move {
                    old_index: 2,
                    new_index: 0,
                },
                "Move diff is incorrect"
            );

            // Verify tracking of moved item
            *app.world_mut().get_mut::<Name>(entity_c).unwrap() = Name::new("Charles");
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Update on moved entity should produce a diff");
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 0,
                    value: Name::new("Charles"),
                },
                "Update on moved entity is incorrect"
            );

            // Test 9: Source SignalVec Pop. Removes David
            entity_vec.write(app.world_mut()).pop();
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Pop should produce one diff");
            assert_eq!(diffs[0], VecDiff::Pop, "Pop diff is incorrect");

            // Test 10: Source SignalVec Clear.
            entity_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<Name>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Clear should produce one diff");
            assert_eq!(diffs[0], VecDiff::Clear, "Clear diff is incorrect");
            assert!(entity_vec.read(app.world()).is_empty());
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn map_signal_preserves_queued_diff_order() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        app.init_resource::<ReadySignalValues>();

        let source = MutableVec::builder().spawn(app.world_mut());
        let signal = source.signal_vec().map_signal(|In(value): In<i32>| {
            signal::from_system(move |In(_), ready_values: Res<ReadySignalValues>| {
                ready_values.0.contains(&value).then_some(value)
            })
            .dedupe()
        });
        let handle = signal
            .for_each(capture_signal_vec_output::<i32>)
            .register(app.world_mut());
        let mut state = Vec::new();

        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);
        apply_diffs(&mut state, diffs);

        source.write(app.world_mut()).insert(0, 10);
        source.write(app.world_mut()).insert(0, 20);
        app.world_mut().resource_mut::<ReadySignalValues>().0 = vec![20];
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());

        app.world_mut().resource_mut::<ReadySignalValues>().0 = vec![20, 10];
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(
            diffs,
            vec![
                VecDiff::InsertAt { index: 0, value: 10 },
                VecDiff::InsertAt { index: 0, value: 20 },
            ]
        );
        apply_diffs(&mut state, diffs);
        assert_eq!(state, vec![20, 10]);

        app.world_mut().resource_mut::<ReadySignalValues>().0 = vec![30, 40, 50, 60, 70];
        source.write(app.world_mut()).insert(1, 30);
        source.write(app.world_mut()).replace(vec![40, 50]);
        source.write(app.world_mut()).remove(0);
        source.write(app.world_mut()).push(60);
        source.write(app.world_mut()).clear();
        source.write(app.world_mut()).push(70);
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Replace { values: vec![70] }]);
        apply_diffs(&mut state, diffs);
        assert_eq!(state, vec![70]);

        handle.release(app.world_mut());
    }

    #[test]
    fn map_signal_cancelled_placeholders_preserve_move_pop_and_ready_transients() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        app.insert_resource(ReadySignalValues(vec![2, 4, 6]));
        let source = MutableVec::builder().values([2, 4, 6]).spawn(app.world_mut());
        let claim = source
            .signal_vec()
            .map_signal(|In(value): In<i32>| {
                signal::from_system(move |In(()), ready: Res<ReadySignalValues>| {
                    ready.0.contains(&value).then_some(value)
                })
                .dedupe()
            })
            .for_each(capture_signal_vec_output::<i32>)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<i32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![2, 4, 6] }]
        );
        source.write(app.world_mut()).insert(0, 10);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).move_item(3, 0);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).remove(1);
        app.update();
        assert_eq!(
            get_and_clear_output::<i32>(app.world_mut()),
            vec![VecDiff::Move {
                old_index: 2,
                new_index: 0
            }]
        );
        source.write(app.world_mut()).push(20);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).pop();
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        app.world_mut().resource_mut::<ReadySignalValues>().0.extend([10, 20]);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).insert(0, 10);
        source.write(app.world_mut()).remove(0);
        app.update();
        assert_eq!(
            get_and_clear_output::<i32>(app.world_mut()),
            vec![
                VecDiff::InsertAt { index: 0, value: 10 },
                VecDiff::RemoveAt { index: 0 }
            ]
        );
        claim.release(app.world_mut());
    }

    #[test]
    fn test_filter() {
        {
            // A comprehensive test for `SignalVecExt::filter`, covering all `VecDiff`
            // types and edge cases like items changing their filter status.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // The source vector contains a mix of values to be filtered.
            let source_vec = MutableVec::builder().values([1, 2, 3, 4]).spawn(app.world_mut());

            // The filter predicate: only allow even numbers.
            let is_even = |In(x): In<i32>| x % 2 == 0;

            // The signal chain under test.
            let filtered_signal = source_vec.signal_vec().filter(is_even);

            let handle = filtered_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // A local copy of the filtered vector's state to verify against.
            let mut current_state: Vec<i32> = vec![];

            // --- 2. Test Initial State (`Replace`) ---
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace { values: vec![2, 4] },
                "Initial `Replace` did not filter correctly."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![2, 4]);

            // --- 3. Test `Push` ---
            // Push a value that passes the filter (6).
            source_vec.write(app.world_mut()).push(6);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Push { value: 6 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![2, 4, 6]);

            // Push a value that fails the filter (7).
            source_vec.write(app.world_mut()).push(7);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert!(diffs.is_empty(), "Pushing a filtered-out value should produce no diff.");
            assert_eq!(current_state, vec![2, 4, 6]); // State should be unchanged.

            // --- 4. Test `UpdateAt` (Status Change) ---
            // Update an item from filtered-out to included (3 -> 8).
            // Source is [1, 2, 3, 4, 6, 7]. Update index 2.
            // It should be inserted into the output at index 1 (after 2).
            source_vec.write(app.world_mut()).set(2, 8);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt { index: 1, value: 8 },
                "Update from filtered to included failed"
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![2, 8, 4, 6]);

            // Update an item from included to filtered-out (4 -> 9).
            // Source is now [1, 2, 8, 4, 6, 7]. Update index 3.
            // It should be removed from the output at index 2 (where 4 was).
            source_vec.write(app.world_mut()).set(3, 9);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::RemoveAt { index: 2 },
                "Update from included to filtered failed"
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![2, 8, 6]);

            // --- 5. Test `RemoveAt` ---
            // Remove an included item (2) from the source at index 1.
            source_vec.write(app.world_mut()).remove(1);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 }); // It was at the start of the output
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![8, 6]);

            // --- 6. Test `Move` ---
            // Source is now [1, 8, 9, 6, 7]. Filtered is [8, 6].
            // Move 8 (source index 1) to after 6 (source index 3).
            // Source becomes [1, 9, 6, 8, 7].
            // This should move index 0 (value 8) to index 1 in the output.
            source_vec.write(app.world_mut()).move_item(1, 3);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Move {
                    old_index: 0,
                    new_index: 1
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![6, 8]);

            // --- 7. Test `Clear` ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 8. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_filter_map() {
        {
            // A comprehensive test for `SignalVecExt::filter_map`, covering all `VecDiff`
            // types and the three status-change scenarios for `UpdateAt`.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            // The source vector contains strings, some of which are parsable as numbers.
            let source_vec = MutableVec::builder()
                .values(["10", "foo", "20", "bar"])
                .spawn(app.world_mut());

            // The filter_map predicate: parse strings to u32, returning Some on success
            // and None on failure.
            let parse_u32 = |In(s): In<&'static str>| s.parse::<u32>().ok();

            // The signal chain under test.
            let filtered_mapped_signal = source_vec.signal_vec().filter_map(parse_u32);

            let handle = filtered_mapped_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // A local copy of the filtered/mapped vector's state.
            let mut current_state: Vec<u32> = vec![];

            // --- 2. Test Initial State (`Replace`) ---
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace { values: vec![10, 20] },
                "Initial `Replace` did not filter and map correctly."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 20]);

            // --- 3. Test `Push` ---
            // Push a value that passes the filter_map ("30").
            source_vec.write(app.world_mut()).push("30");
            // drop(source_vec)
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Push { value: 30 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 20, 30]);

            // Push a value that fails the filter_map ("baz").
            source_vec.write(app.world_mut()).push("baz");
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert!(diffs.is_empty(), "Pushing a filtered-out value should produce no diff.");

            // --- 4. Test `UpdateAt` Status Changes ---
            // Case 1: None -> Some (Update "foo" at source index 1 to "40").
            source_vec.write(app.world_mut()).set(1, "40");
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::InsertAt { index: 1, value: 40 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 40, 20, 30]);

            // Case 2: Some -> None (Update "20" at source index 2 to "qux").
            source_vec.write(app.world_mut()).set(2, "qux");
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 40, 30]);

            // Case 3: Some -> Some (Update "10" at source index 0 to "50").
            source_vec.write(app.world_mut()).set(0, "50");
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::UpdateAt { index: 0, value: 50 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![50, 40, 30]);

            // --- 5. Test `RemoveAt` ---
            // Source is now: ["50", "40", "qux", "bar", "30", "baz"]
            // Filtered is: [50, 40, 30]
            // Remove "40" from source at index 1. This is at output index 1.
            source_vec.write(app.world_mut()).remove(1);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 1 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![50, 30]);

            // --- 6. Test `Move` ---
            // Source is now: ["50", "qux", "bar", "30", "baz"]
            // Filtered is: [50, 30]
            // Move "30" (source index 3) to the front (source index 0).
            // This moves output index 1 to output index 0.
            source_vec.write(app.world_mut()).move_item(3, 0);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::Move {
                    old_index: 1,
                    new_index: 0
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![30, 50]);

            // --- 7. Test `Clear` ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(diffs[0], VecDiff::Clear);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 8. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    /// This test provides comprehensive coverage for a single
    /// `SignalVecExt::filter_signal` node.
    ///
    /// It verifies correct behavior under two primary conditions:
    ///
    /// 1. **Filter State Change**: When the external condition (a Bevy `Resource`) that the filter
    ///    signals depend on changes, the test ensures that the output `SignalVec` correctly updates
    ///    by inserting newly-matched items and removing items that no longer match.
    ///
    /// 2. **Source Vector Changes**: The test verifies that all `VecDiff` types from the source
    ///    `SignalVec` (`Push`, `RemoveAt`, `UpdateAt`, `Clear`) are correctly processed. New items
    ///    are filtered according to the _current_ filter state, and their corresponding signals are
    ///    created. Removed items have their signals cleaned up. Updated items can be added to or
    ///    removed from the filtered list based on their new value.
    #[test]
    fn test_filter_signal() {
        {
            // A resource to control the filter's behavior.
            #[derive(Resource, Clone, PartialEq, Debug)]
            struct FilterMode(bool);

            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // --- Setup --- Initially, filter for even numbers.
            app.insert_resource(FilterMode(true));
            let source_vec = MutableVec::builder().values([1, 2, 3, 4]).spawn(app.world_mut());
            let filtered_signal = source_vec.signal_vec().filter_signal(|In(val): In<i32>| {
                signal::from_resource::<FilterMode>()
                    .map(move |In(mode): In<FilterMode>| {
                        // even or odd
                        if mode.0 { val % 2 == 0 } else { val % 2 != 0 }
                    })
                    .cache()
            });
            let handle = filtered_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // --- 1. Initial State (Even) ---
            app.update();
            let mut current_state = vec![];
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(current_state, vec![2, 4], "Initial state (even) is incorrect.");

            // --- 2. Change Filter (to Odd) ---
            *app.world_mut().resource_mut::<FilterMode>() = FilterMode(false);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(current_state, vec![1, 3], "State after switching to odd is incorrect.");

            // --- 3. Source Vec Push (respecting Odd filter) --- Push an odd number, should
            // appear.
            source_vec.write(app.world_mut()).push(5);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(
                current_state,
                vec![1, 3, 5],
                "State after pushing odd number is incorrect."
            );

            // Push an even number, should be filtered out.
            source_vec.write(app.world_mut()).push(6);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert!(
                diffs.is_empty(),
                "Pushing a filtered-out number should produce no diffs."
            );
            assert_eq!(
                current_state,
                vec![1, 3, 5],
                "State should be unchanged after pushing even number."
            );

            // --- 4. Change Filter (back to Even) ---
            *app.world_mut().resource_mut::<FilterMode>() = FilterMode(true);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));

            // Source vec is now [1, 2, 3, 4, 5, 6]. Evens are [2, 4, 6].
            assert_eq!(
                current_state,
                vec![2, 4, 6],
                "State after switching back to even is incorrect."
            );

            // --- 5. Source Vec RemoveAt --- Remove `4` (even) from source vec (at index 3).
            source_vec.write(app.world_mut()).remove(3);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(current_state, vec![2, 6], "State after removing '4' is incorrect.");

            // --- 6. Source Vec UpdateAt --- Update `3` (odd, at index 2) to `8` (even). This
            // should insert `8` into the filtered list.
            source_vec.write(app.world_mut()).set(2, 8);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));

            // Source is now [1, 2, 8, 5, 6]. Evens are [2, 8, 6].
            assert_eq!(
                current_state,
                vec![2, 8, 6],
                "State after updating 3 to 8 is incorrect."
            );

            // Update `6` (even, at index 4) to `7` (odd). This should remove `6` from the
            // filtered list.
            source_vec.write(app.world_mut()).set(4, 7);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));

            // Source is now [1, 2, 8, 5, 7]. Evens are [2, 8].
            assert_eq!(current_state, vec![2, 8], "State after updating 6 to 7 is incorrect.");

            // --- 7. Source Vec Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Clear should produce one diff.");
            assert_eq!(diffs[0], VecDiff::Clear, "Expected a Clear diff.");
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty(), "State after clear should be empty.");

            // --- Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn filter_signal_cancelled_replacements_preserve_membership_through_moves_and_repeated_updates() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        app.insert_resource(ReadySignalValues(vec![2, 4, 6, 8]));
        let source = MutableVec::builder().values([2, 4, 6]).spawn(app.world_mut());
        let claim = source
            .signal_vec()
            .filter_signal(|In(value): In<i32>| {
                signal::from_system(move |In(()), ready: Res<ReadySignalValues>| {
                    ready.0.contains(&value).then_some(value % 2 == 0)
                })
                .dedupe()
            })
            .for_each(capture_signal_vec_output::<i32>)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<i32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![2, 4, 6] }]
        );
        for value in [7, 9] {
            source.write(app.world_mut()).set(1, value);
            app.update();
            assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        }
        source.write(app.world_mut()).insert(0, 8);
        source.write(app.world_mut()).move_item(2, 3);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).pop();
        app.update();
        assert_eq!(
            get_and_clear_output::<i32>(app.world_mut()),
            vec![
                VecDiff::InsertAt { index: 0, value: 8 },
                VecDiff::Move {
                    old_index: 2,
                    new_index: 3
                },
                VecDiff::Pop,
            ]
        );
        app.world_mut().resource_mut::<ReadySignalValues>().0.extend([7, 9]);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).push(10);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).move_item(3, 0);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        source.write(app.world_mut()).set(0, 12);
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());
        app.world_mut().resource_mut::<ReadySignalValues>().0.push(12);
        app.update();
        assert_eq!(
            get_and_clear_output::<i32>(app.world_mut()),
            vec![VecDiff::InsertAt { index: 0, value: 12 }]
        );
        claim.release(app.world_mut());
    }

    #[test]
    fn filter_signal_preserves_queued_diff_order() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        app.init_resource::<ReadySignalValues>();

        let source = MutableVec::builder().spawn(app.world_mut());
        let signal = source.signal_vec().filter_signal(|In(value): In<i32>| {
            signal::from_system(move |In(_), ready_values: Res<ReadySignalValues>| {
                ready_values.0.contains(&value).then_some(value % 2 == 0)
            })
            .dedupe()
        });
        let handle = signal.for_each(capture_signal_vec_output).register(app.world_mut());
        let mut state = Vec::new();

        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);
        apply_diffs(&mut state, diffs);

        source.write(app.world_mut()).insert(0, 2);
        source.write(app.world_mut()).insert(0, 4);
        app.world_mut().resource_mut::<ReadySignalValues>().0 = vec![4];
        app.update();
        assert!(get_and_clear_output::<i32>(app.world_mut()).is_empty());

        app.world_mut().resource_mut::<ReadySignalValues>().0 = vec![4, 2];
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(
            diffs,
            vec![
                VecDiff::InsertAt { index: 0, value: 2 },
                VecDiff::InsertAt { index: 0, value: 4 },
            ]
        );
        apply_diffs(&mut state, diffs);
        assert_eq!(state, vec![4, 2]);

        app.world_mut().resource_mut::<ReadySignalValues>().0 = vec![3, 5, 6, 7, 8];
        source.write(app.world_mut()).insert(1, 3);
        source.write(app.world_mut()).replace(vec![5, 6]);
        source.write(app.world_mut()).remove(0);
        source.write(app.world_mut()).push(7);
        source.write(app.world_mut()).clear();
        source.write(app.world_mut()).push(8);
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Replace { values: vec![8] }]);
        apply_diffs(&mut state, diffs);
        assert_eq!(state, vec![8]);

        handle.release(app.world_mut());
    }

    /// This test verifies the behavior of chaining multiple `.filter_signal()` calls.
    ///
    /// It ensures that the filters behave conjunctively (i.e., with AND logic). The
    /// core scenario tested is when an item is initially blocked by an early filter
    /// (`filter_a`) but would be allowed by a later one (`filter_b`). When
    /// `filter_a`'s condition flips to allow the item, it must correctly propagate an
    /// `InsertAt` diff to `filter_b`, which then allows the item to appear in the
    /// final output. This confirms that the state of each item is re-evaluated
    /// correctly as filter conditions change throughout the chain.
    #[test]
    fn test_filter_signal_chaining() {
        {
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // --- Setup --- Resources to control the two filters
            #[derive(Resource, Clone, PartialEq, Debug)]
            struct IsPositive(bool);

            #[derive(Resource, Clone, PartialEq, Debug)]
            struct IsEven(bool);

            // Initially, we only want non-positive numbers
            app.insert_resource(IsPositive(false));

            // And we want even numbers
            app.insert_resource(IsEven(true));

            // The source vector
            let source_vec = MutableVec::builder().values([-2, -1, 0, 1, 2]).spawn(app.world_mut());

            // --- The Signal Chain ---
            let final_signal = source_vec
                .signal_vec()
                // Filter 1: controlled by IsPositive resource.
                .filter_signal(|In(val): In<i32>| {
                    signal::from_resource::<IsPositive>()
                        .map(move |In(res): In<IsPositive>| if res.0 { val > 0 } else { val <= 0 })
                        .cache()
                })
                // Filter 2: controlled by IsEven resource.
                .filter_signal(|In(val): In<i32>| {
                    signal::from_resource::<IsEven>()
                        .map(move |In(res): In<IsEven>| if res.0 { val % 2 == 0 } else { val % 2 != 0 })
                        .cache()
                });
            let handle = final_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // --- 1. Initial State --- IsPositive(false) -> pass [-2, -1, 0] IsEven(true)
            //  -> from the above, pass [-2, 0]
            app.update();
            let mut current_state = vec![];
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(current_state, vec![-2, 0], "Initial state is incorrect.");

            // --- 2. Change the first filter's condition --- Now IsPositive(true) -> pass [1,
            // 2] The second filter IsEven(true) is unchanged -> from the new set, pass [2]
            // The combined effect should be to remove [-2, 0] and insert [2].
            *app.world_mut().resource_mut::<IsPositive>() = IsPositive(true);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(
                current_state,
                vec![2],
                "State after flipping first filter is incorrect."
            );

            // --- 3. Change the second filter's condition --- IsPositive(true) is unchanged
            // -> pass [1, 2] IsEven(false) -> from the above, pass [1] The combined effect
            // should be to remove [2] and insert [1].
            *app.world_mut().resource_mut::<IsEven>() = IsEven(false);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(
                current_state,
                vec![1],
                "State after flipping second filter is incorrect."
            );

            // --- 4. Change the first filter back --- IsPositive(false) -> pass [-2, -1, 0]
            // IsEven(false) -> from the above, pass [-1] The combined effect should be to
            // remove [1] and insert [-1].
            *app.world_mut().resource_mut::<IsPositive>() = IsPositive(false);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(
                current_state,
                vec![-1],
                "State after flipping first filter back is incorrect."
            );

            // --- 5. Test source vec modification --- Add an item that should pass both
            // current filters: IsPositive(false) and IsEven(false) -> Add -3. It should be
            // pushed to the end of the filtered list.
            source_vec.write(app.world_mut()).push(-3);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<i32>(app.world_mut()));
            assert_eq!(current_state, vec![-1, -3], "State after pushing -3 is incorrect.");

            // --- Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_to_signal() {
        {
            // A comprehensive test for `SignalVecExt::to_signal`, verifying that it
            // correctly reconstructs the full vector state from all `VecDiff` types.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<FinalSignalOutput<Vec<String>>>();
            let source_vec = MutableVec::builder()
                .values(["a".to_string(), "b".to_string()])
                .spawn(app.world_mut());

            // The signal chain under test.
            let state_signal = source_vec.signal_vec().to_signal();
            let handle = state_signal
                .map(capture_final_output::<Vec<String>>)
                .register(app.world_mut());

            // --- 2. Test Initial State ---
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["a".to_string(), "b".to_string()]),
                "Initial state was not correctly emitted."
            );

            // --- 3. Test Individual Diffs ---
            // Test Push
            source_vec.write(app.world_mut()).push("c".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
                "State after Push is incorrect."
            );

            // Test RemoveAt
            source_vec.write(app.world_mut()).remove(1); // removes "b"
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["a".to_string(), "c".to_string()]),
                "State after RemoveAt is incorrect."
            );

            // Test InsertAt
            source_vec.write(app.world_mut()).insert(0, "x".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["x".to_string(), "a".to_string(), "c".to_string()]),
                "State after InsertAt is incorrect."
            );

            // Test UpdateAt
            source_vec.write(app.world_mut()).set(2, "C".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["x".to_string(), "a".to_string(), "C".to_string()]),
                "State after UpdateAt is incorrect."
            );

            // Test Move
            source_vec.write(app.world_mut()).move_item(2, 0); // moves "C" to the front
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["C".to_string(), "x".to_string(), "a".to_string()]),
                "State after Move is incorrect."
            );

            // Test Pop
            source_vec.write(app.world_mut()).pop(); // removes "a"
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["C".to_string(), "x".to_string()]),
                "State after Pop is incorrect."
            );

            // --- 4. Test No-Op Frame ---
            // An update without any source changes should not emit a new state.
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                None,
                "Signal emitted on a no-op frame."
            );

            // --- 5. Test Batched Mutations ---
            {
                let mut writer = source_vec.write(app.world_mut());
                writer.push("y".to_string()); // State: ["C", "x", "y"]
                writer.remove(0); // State: ["x", "y"]
            }
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec!["x".to_string(), "y".to_string()]),
                "State after batched mutations is incorrect."
            );

            // --- 6. Test Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            assert_eq!(
                get_and_clear_final_output::<Vec<String>>(app.world_mut()),
                Some(vec![]),
                "State after Clear is incorrect."
            );

            // --- 7. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_is_empty() {
        {
            // A comprehensive test for `SignalVecExt::is_empty`, verifying that it
            // correctly emits true/false and re-emits on every source change, as intended.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<FinalSignalOutput<bool>>();
            // Start with a non-empty vec to test the `false` state first.
            let source_vec = MutableVec::builder()
                .values(["a".to_string(), "b".to_string()])
                .spawn(app.world_mut());

            let is_empty_signal = source_vec.signal_vec().is_empty();
            let handle = is_empty_signal
                .map(capture_final_output::<bool>)
                .register(app.world_mut());

            // --- 2. Test Initial State (Non-Empty) ---
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Initial state for non-empty vec should be `false`."
            );

            // --- 3. Test Operations that REMAIN Non-Empty ---
            // These should re-emit `false` because the underlying vector changed, even if
            // the emptiness state did not.

            // Push: size 2 -> 3. Still not empty.
            source_vec.write(app.world_mut()).push("c".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Pushing to a non-empty vec should re-emit `false`."
            );

            // UpdateAt: size 3 -> 3. Still not empty.
            source_vec.write(app.world_mut()).set(0, "A".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Updating a non-empty vec should re-emit `false`."
            );

            // Move: size 3 -> 3. Still not empty.
            source_vec.write(app.world_mut()).move_item(0, 2);
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Moving within a non-empty vec should re-emit `false`."
            );

            // --- 4. Test Transition to Empty ---
            // Current size is 3. We pop three times.
            source_vec.write(app.world_mut()).pop(); // size 3 -> 2
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Popping to size 2 should re-emit `false`."
            );

            source_vec.write(app.world_mut()).pop(); // size 2 -> 1
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Popping to size 1 should re-emit `false`."
            );

            // This is the pop that makes it empty. The state changes from false to true.
            source_vec.write(app.world_mut()).pop(); // size 1 -> 0
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(true),
                "Popping the last item should emit `true`."
            );

            // --- 5. Test True No-Op Frame ---
            // An update without any source diffs should not cause an emit.
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                None,
                "Signal should not emit on a true no-op frame."
            );

            // --- 6. Test Transition to Non-Empty ---
            // The state changes from true to false.
            source_vec.write(app.world_mut()).push("z".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(false),
                "Pushing to an empty vec should emit `false`."
            );

            // --- 7. Test Clear Operation ---
            // The state changes from false to true.
            source_vec.write(app.world_mut()).clear();
            app.update();
            assert_eq!(
                get_and_clear_final_output::<bool>(app.world_mut()),
                Some(true),
                "Clear on a non-empty vec should emit `true`."
            );

            // --- 8. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_len() {
        {
            // A comprehensive test for `SignalVecExt::len`, verifying that it correctly
            // reports the vector's length and re-emits on every source change.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<FinalSignalOutput<usize>>();
            // Start with a non-empty vec.
            let source_vec = MutableVec::builder()
                .values(["a".to_string(), "b".to_string()])
                .spawn(app.world_mut());

            let len_signal = source_vec.signal_vec().len();
            let handle = len_signal.map(capture_final_output::<usize>).register(app.world_mut());

            // --- 2. Test Initial State ---
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(2),
                "Initial length should be 2."
            );

            // --- 3. Test Length-Changing Operations ---
            // Push: size 2 -> 3
            source_vec.write(app.world_mut()).push("c".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(3),
                "Length after Push should be 3."
            );

            // InsertAt: size 3 -> 4
            source_vec.write(app.world_mut()).insert(1, "x".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(4),
                "Length after InsertAt should be 4."
            );

            // RemoveAt: size 4 -> 3
            source_vec.write(app.world_mut()).remove(2); // removes "b"
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(3),
                "Length after RemoveAt should be 3."
            );

            // Pop: size 3 -> 2
            source_vec.write(app.world_mut()).pop(); // removes "c"
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(2),
                "Length after Pop should be 2."
            );

            // --- 4. Test Length-Preserving Operations ---
            // These should still re-emit the current length because the underlying vec changed.

            // UpdateAt: size 2 -> 2
            source_vec.write(app.world_mut()).set(0, "A".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(2),
                "UpdateAt should re-emit length 2."
            );

            // Move: size 2 -> 2
            source_vec.write(app.world_mut()).move_item(0, 1);
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(2),
                "Move should re-emit length 2."
            );

            // --- 5. Test True No-Op Frame ---
            // An update without any source changes should not emit a new value.
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                None,
                "Signal emitted on a no-op frame."
            );

            // --- 6. Test Batched Mutations (Net Zero Length Change) ---
            {
                let mut writer = source_vec.write(app.world_mut());
                writer.push("y".to_string()); // len -> 3
                writer.remove(0); // len -> 2
            }
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(2),
                "Batched mutations with net-zero length change should emit final length."
            );

            // --- 7. Test Clear Operation ---
            // State changes from 2 to 0.
            source_vec.write(app.world_mut()).clear();
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(0),
                "Length after Clear should be 0."
            );

            // --- 8. Test Transition from Empty ---
            source_vec.write(app.world_mut()).push("z".to_string());
            app.update();
            assert_eq!(
                get_and_clear_final_output::<usize>(app.world_mut()),
                Some(1),
                "Length after pushing to an empty vec should be 1."
            );

            // --- 9. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_sum() {
        {
            // A comprehensive test for `SignalVecExt::sum`, verifying that it correctly
            // calculates the sum and re-emits on every source change, as intended.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<FinalSignalOutput<i32>>();
            let source_vec = MutableVec::builder().values([10, 20]).spawn(app.world_mut());

            let sum_signal = source_vec.signal_vec().sum();
            let handle = sum_signal.map(capture_final_output::<i32>).register(app.world_mut());

            // --- 2. Test Initial State ---
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(30),
                "Initial sum should be 30."
            );

            // --- 3. Test Value-Changing Operations ---
            // Push: 30 -> 35
            source_vec.write(app.world_mut()).push(5);
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(35),
                "Sum after Push should be 35."
            );

            // UpdateAt: 35 -> 45 (updates 10 to 20)
            source_vec.write(app.world_mut()).set(0, 20);
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(45),
                "Sum after UpdateAt should be 45."
            );

            // RemoveAt: 45 -> 25 (removes 20 at index 1)
            source_vec.write(app.world_mut()).remove(1);
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(25),
                "Sum after RemoveAt should be 25."
            );

            // Pop: 25 -> 5 (removes 20 at the end)
            source_vec.write(app.world_mut()).pop(); // vec is now [20]
            source_vec.write(app.world_mut()).pop(); // vec is now []
            source_vec.write(app.world_mut()).push(5); // vec is now [5]
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(5),
                "Sum after multiple operations should be 5."
            );

            // --- 4. Test Sum-Preserving Operation (`Move`) ---
            // Add another element to make move meaningful
            source_vec.write(app.world_mut()).push(15); // vec is now [5, 15], sum is 20
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(20),
                "Sum after adding item for move test should be 20."
            );

            // Move does not change the sum, but should still re-emit the value.
            source_vec.write(app.world_mut()).move_item(0, 1); // vec is now [15, 5], sum is still 20
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(20),
                "Move operation should re-emit the unchanged sum."
            );

            // --- 5. Test True No-Op Frame ---
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                None,
                "Signal should not emit on a true no-op frame."
            );

            // --- 6. Test Clear Operation ---
            // State changes from 20 to 0.
            source_vec.write(app.world_mut()).clear();
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(0),
                "Sum after Clear should be 0."
            );

            // --- 7. Test Transition from Empty ---
            source_vec.write(app.world_mut()).push(-10);
            app.update();
            assert_eq!(
                get_and_clear_final_output::<i32>(app.world_mut()),
                Some(-10),
                "Sum after pushing to an empty vec should be -10."
            );

            // --- 8. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_chain() {
        {
            // A comprehensive test for `SignalVecExt::chain`, covering all `VecDiff`
            // types on both the left and right sides, as well as edge cases like empty
            // vectors.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<String>>();

            let left_vec = MutableVec::builder()
                .values(["L1".to_string(), "L2".to_string()])
                .spawn(app.world_mut());
            let right_vec = MutableVec::builder()
                .values(["R1".to_string(), "R2".to_string()])
                .spawn(app.world_mut());

            let chained_signal = left_vec.signal_vec().chain(right_vec.signal_vec());
            let handle = chained_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            let mut current_state: Vec<String> = vec![];

            // --- 2. Initial State ---
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L1", "L2", "R1", "R2"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>(),
                "Initial state was not correctly concatenated."
            );
            let left_len = |world: &World| left_vec.read(world).len();

            // --- 3. Left-Side Mutations ---

            // Push
            left_vec.write(app.world_mut()).push("L3".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::InsertAt {
                    index: 2,
                    value: "L3".to_string()
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L1", "L2", "L3", "R1", "R2"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // UpdateAt
            left_vec.write(app.world_mut()).set(0, "L1-new".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::UpdateAt {
                    index: 0,
                    value: "L1-new".to_string()
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L1-new", "L2", "L3", "R1", "R2"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Move
            left_vec.write(app.world_mut()).move_item(2, 0); // "L3" to front
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Move {
                    old_index: 2,
                    new_index: 0
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L3", "L1-new", "L2", "R1", "R2"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Pop
            left_vec.write(app.world_mut()).pop(); // removes "L2"
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::RemoveAt { index: 2 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L3", "L1-new", "R1", "R2"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // --- 4. Right-Side Mutations --- (left_len() is now 2)

            // InsertAt
            right_vec.write(app.world_mut()).insert(1, "R-insert".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::InsertAt {
                    index: left_len(app.world()) + 1,
                    value: "R-insert".to_string()
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L3", "L1-new", "R1", "R-insert", "R2"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Move
            right_vec.write(app.world_mut()).move_item(0, 2); // "R1" to end
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Move {
                    old_index: left_len(app.world()),
                    new_index: left_len(app.world()) + 2
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L3", "L1-new", "R-insert", "R2", "R1"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // --- 5. `Replace` and `Clear` ---

            // Replace Left
            left_vec.write(app.world_mut()).replace(vec!["LX".to_string()]);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs); // apply diffs to check final state
            assert_eq!(
                current_state,
                ["LX", "R-insert", "R2", "R1"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>(),
                "State after replacing left is incorrect"
            );

            // Replace Right
            right_vec
                .write(app.world_mut())
                .replace(vec!["RY".to_string(), "RZ".to_string()]);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["LX", "RY", "RZ"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                "State after replacing right is incorrect"
            );

            // Clear Left
            left_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["RY", "RZ"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                "State after clearing left is incorrect"
            );
            assert_eq!(left_len(app.world()), 0);

            // Push to Left when it's empty
            left_vec.write(app.world_mut()).push("L-again".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::InsertAt {
                    index: 0,
                    value: "L-again".to_string()
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L-again", "RY", "RZ"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Clear Right
            right_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L-again"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                "State after clearing right is incorrect"
            );
            assert_eq!(right_vec.read(app.world()).len(), 0);

            // Push to Right when it's empty
            right_vec.write(app.world_mut()).push("R-again".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Push {
                    value: "R-again".to_string()
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["L-again", "R-again"].iter().map(|s| s.to_string()).collect::<Vec<_>>()
            );

            // --- 6. Batched Diffs ---
            left_vec.write(app.world_mut()).pop(); // L-again is gone
            right_vec.write(app.world_mut()).push("R-batch".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["R-again", "R-batch"].iter().map(|s| s.to_string()).collect::<Vec<_>>()
            );

            // --- 7. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_intersperse() {
        {
            // A comprehensive test for `SignalVecExt::intersperse`, covering all `VecDiff`
            // types and various edge cases.

            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<String>>();

            let separator = "sep".to_string();
            let source_vec = MutableVec::from(app.world_mut()); // Start empty for initial tests

            let interspersed_signal = source_vec.signal_vec().intersperse(separator.clone());
            let handle = interspersed_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            let mut current_state: Vec<String> = vec![];

            // --- 2. Test Initial and Edge Cases ---

            // Initial state is authoritatively empty.
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // Replace with empty
            source_vec.write(app.world_mut()).replace(Vec::<String>::new());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // Replace with one item (no separators)
            source_vec.write(app.world_mut()).replace(vec!["A".to_string()]);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Replace {
                    values: vec!["A".to_string()]
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["A".to_string()]);

            // Replace with multiple items
            source_vec
                .write(app.world_mut())
                .replace(vec!["A".to_string(), "B".to_string()]);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Replace {
                    values: vec!["A".to_string(), separator.clone(), "B".to_string()]
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["A".to_string(), separator.clone(), "B".to_string()]);

            // --- 3. Test Push & Pop ---

            // Push to multi-item vec
            source_vec.write(app.world_mut()).push("C".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["A", "sep", "B", "sep", "C"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Pop from multi-item vec
            source_vec.write(app.world_mut()).pop();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["A", "sep", "B"].iter().map(|&s| s.to_string()).collect::<Vec<_>>()
            );

            // Pop until one item left
            source_vec.write(app.world_mut()).pop();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["A".to_string()]);

            // Pop the last item
            source_vec.write(app.world_mut()).pop();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // Push to empty vec
            source_vec.write(app.world_mut()).push("Z".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec!["Z".to_string()]);

            // --- 4. Test Insert & Remove ---
            source_vec
                .write(app.world_mut())
                .replace(vec!["A".to_string(), "C".to_string()]);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<String>(app.world_mut()));
            assert_eq!(
                current_state,
                ["A", "sep", "C"].iter().map(|&s| s.to_string()).collect::<Vec<_>>()
            );

            // Insert in middle
            source_vec.write(app.world_mut()).insert(1, "B".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["A", "sep", "B", "sep", "C"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Insert at beginning
            source_vec.write(app.world_mut()).insert(0, "S".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["S", "sep", "A", "sep", "B", "sep", "C"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Insert at end (should behave like push)
            source_vec.write(app.world_mut()).insert(4, "E".to_string());
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["S", "sep", "A", "sep", "B", "sep", "C", "sep", "E"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>(),
                "InsertAt at end failed"
            );

            // Remove from middle ("A")
            source_vec.write(app.world_mut()).remove(1);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["S", "sep", "B", "sep", "C", "sep", "E"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Remove from end ("E")
            source_vec.write(app.world_mut()).remove(3);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["S", "sep", "B", "sep", "C"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // --- 5. Test Update & Move ---
            // Update
            source_vec.write(app.world_mut()).set(1, "Beta".to_string()); // "B" -> "Beta"
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::UpdateAt {
                    index: 2,
                    value: "Beta".to_string()
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["S", "sep", "Beta", "sep", "C"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
            );

            // Move
            // Source: ["S", "Beta", "C"] -> move 0 to 2 -> ["Beta", "C", "S"]
            // Interspersed: ["S", "sep", "Beta", "sep", "C"] -> ["Beta", "sep", "C", "sep", "S"]
            source_vec.write(app.world_mut()).move_item(0, 2);
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            // The exact diffs for Move can be complex; we verify the final state.
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                ["Beta", "sep", "C", "sep", "S"]
                    .iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>(),
                "Move failed"
            );

            // --- 6. Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<String>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Clear]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 7. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    /// A helper enum to represent the items in our final interspersed vector,
    /// allowing us to distinguish between original items and generated separators.
    enum ItemOrSep {
        Item(u32),
        // The separator variant holds the *unregistered* index signal.
        Sep(crate::signal::Broadcast<Option<usize>>),
    }

    impl Clone for ItemOrSep {
        fn clone(&self) -> Self {
            match self {
                Self::Item(i) => Self::Item(*i),
                Self::Sep(s) => Self::Sep(s.clone()),
            }
        }
    }

    // Manual `Debug` implementation is required because `Source` does not implement it.
    impl fmt::Debug for ItemOrSep {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ItemOrSep::Item(i) => f.debug_tuple("Item").field(i).finish(),
                ItemOrSep::Sep(_) => f.debug_tuple("Sep").field(&"...").finish(),
            }
        }
    }

    // A helper to make assertions cleaner by comparing the structure without the signal data.
    impl PartialEq for ItemOrSep {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Item(l), Self::Item(r)) => l == r,
                // For testing, we just care that it's a separator.
                (Self::Sep(_), Self::Sep(_)) => true,
                _ => false,
            }
        }
    }

    /// Helper to imperatively evaluate the current value of a separator's index signal.
    fn poll_index(world: &mut World, claim: &SignalClaim) -> Option<usize> {
        run_signal_subgraph_once_internal(world, claim.entity())
            .and_then(downcast_erased_signal_value::<Option<usize>>)
            .flatten()
    }

    /// This test provides comprehensive coverage for `SignalVecExt::intersperse_with`.
    ///
    /// It verifies that the combinator correctly:
    ///
    /// 1. **Builds Initial State**: Creates a correctly interspersed list from a source vector,
    ///    calling the separator factory for each gap and producing the correct `VecDiff::Replace`.
    ///
    /// 2. **Handles All `VecDiff` Types**: Correctly processes `Push`, `Pop`, `InsertAt`,
    ///    `RemoveAt`, `Move`, and `Clear` from the source, translating them into the appropriate
    ///    diffs for the final interspersed list.
    ///
    /// 3. **Provides Reactive Index Signals**: The key feature is that the factory receives a
    ///    reactive `Signal` for the separator's index. This test captures the *unregistered*
    ///    signal, registers it ephemerally during assertions, and steps its subgraph to prove that
    ///    the separator's perceived index updates correctly after structural changes (`RemoveAt`,
    ///    `Move`, etc.) occur in the source vector.
    #[test]
    fn test_intersperse_with() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();
            // Use `insert_resource` with an explicit value, as `ItemOrSep` doesn't have a Default.
            app.insert_resource(SignalVecOutput::<ItemOrSep>(Vec::new()));

            let source_vec = MutableVec::builder().values([10u32, 20, 30]).spawn(app.world_mut());

            let separator_factory = |In(index_signal): In<Broadcast<Option<usize>>>| ItemOrSep::Sep(index_signal);

            let interspersed_signal = source_vec
                .signal_vec()
                .map_in(ItemOrSep::Item) // Map source items into our test enum
                .intersperse_with(separator_factory);

            let handle = interspersed_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // This local vec will mirror the state of the final interspersed signal.
            let mut current_state: Vec<ItemOrSep> = vec![];

            // --- 2. Test Initial State ---
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<ItemOrSep>(app.world_mut()));

            let placeholder_sep = ItemOrSep::Sep(crate::signal::from_system(|In(_)| None::<usize>).broadcast());
            assert_eq!(
                current_state,
                vec![
                    ItemOrSep::Item(10),
                    placeholder_sep.clone(),
                    ItemOrSep::Item(20),
                    placeholder_sep.clone(),
                    ItemOrSep::Item(30),
                ],
                "Initial structure is incorrect."
            );

            // Verify the indices of the separators by registering and polling their signals.
            let mut handles_to_clean = Vec::new();
            let sep_signals: Vec<_> = current_state
                .iter()
                .filter_map(|x| match x {
                    ItemOrSep::Sep(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();

            assert_eq!(sep_signals.len(), 2);
            let h1 = sep_signals[0].clone().register(app.world_mut());
            let h2 = sep_signals[1].clone().register(app.world_mut());
            assert_eq!(poll_index(app.world_mut(), &h1), Some(0));
            assert_eq!(poll_index(app.world_mut(), &h2), Some(1));
            handles_to_clean.push(h1);
            handles_to_clean.push(h2);

            // --- 3. Test Push ---
            source_vec.write(app.world_mut()).push(40);
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<ItemOrSep>(app.world_mut()));

            assert_eq!(current_state.len(), 7, "State length after push is incorrect.");
            assert_eq!(current_state[6], ItemOrSep::Item(40));

            if let ItemOrSep::Sep(signal) = &current_state[5] {
                let h = signal.clone().register(app.world_mut());
                assert_eq!(
                    poll_index(app.world_mut(), &h),
                    Some(2),
                    "New separator's index is incorrect."
                );
                handles_to_clean.push(h);
            } else {
                panic!("Expected a separator at index 5");
            }

            // --- 4. Test RemoveAt (from middle) and Index Reactivity ---
            source_vec.write(app.world_mut()).remove(1); // Removes 20
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<ItemOrSep>(app.world_mut()));

            assert_eq!(current_state.len(), 5, "State length after RemoveAt is incorrect.");
            assert_eq!(
                current_state,
                vec![
                    ItemOrSep::Item(10),
                    placeholder_sep.clone(),
                    ItemOrSep::Item(30),
                    placeholder_sep.clone(),
                    ItemOrSep::Item(40),
                ],
                "Structure after RemoveAt is incorrect."
            );

            // The crucial test: the separator that was after `30` (original index 2) should
            // now have its index signal updated to `1`.
            let sep_signals_after_remove: Vec<_> = current_state
                .iter()
                .filter_map(|x| {
                    if let ItemOrSep::Sep(s) = x {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(sep_signals_after_remove.len(), 2);
            let h_rem_1 = sep_signals_after_remove[0].clone().register(app.world_mut());
            let h_rem_2 = sep_signals_after_remove[1].clone().register(app.world_mut());
            assert_eq!(poll_index(app.world_mut(), &h_rem_1), Some(0));
            assert_eq!(
                poll_index(app.world_mut(), &h_rem_2),
                Some(1),
                "Separator index did not reactively update after RemoveAt."
            );
            handles_to_clean.push(h_rem_1);
            handles_to_clean.push(h_rem_2);

            // --- 5. Test Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<ItemOrSep>(app.world_mut()));
            assert!(current_state.is_empty(), "State after Clear should be empty.");

            // --- 6. Final Cleanup ---
            handle.release(app.world_mut());
            for h in handles_to_clean {
                h.release(app.world_mut());
            }
            app.update(); // Run one last time to process cleanup commands.
        }
    }

    #[test]
    fn test_sort_by() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<(u32, char)>>();
            let source_vec = MutableVec::builder()
                .values([(3, 'c'), (1, 'a'), (4, 'd')])
                .spawn(app.world_mut());
            let compare_system = |In((left, right)): In<((u32, char), (u32, char))>| left.0.cmp(&right.0);
            let sorted_signal = source_vec.signal_vec().sort_by(compare_system);
            let handle = sorted_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());
            let mut current_state: Vec<(u32, char)> = vec![];

            // --- 2. Test Initial State ---
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![(1, 'a'), (3, 'c'), (4, 'd')],
                "Initial state is not correctly sorted."
            );

            // --- 3. Test Push ---
            source_vec.write(app.world_mut()).push((2, 'b'));
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: (2, 'b')
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]);

            // --- 4. Test RemoveAt ---
            source_vec.write(app.world_mut()).remove(1); // Removes (1, 'a')
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 });
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![(2, 'b'), (3, 'c'), (4, 'd')]);

            // --- 5. Test UpdateAt ---
            // Case A: Update without changing sort order.
            source_vec.write(app.world_mut()).set(0, (3, 'C')); // Was (3, 'c')
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 1);
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 1,
                    value: (3, 'C')
                }
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![(2, 'b'), (3, 'C'), (4, 'd')]);

            // Case B: Update that changes sort order.
            // Source is now [(3, 'C'), (4, 'd'), (2, 'b')]. Sorted is [(2, 'b'), (3, 'C'), (4,
            // 'd')] Update (4, 'd') at source index 1 to (0, 'z').
            // The old item was at sorted index 2. The new item should be at sorted index 0.
            source_vec.write(app.world_mut()).set(1, (0, 'z'));
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs.len(), 2, "Update with sort change should be Remove+Insert.");
            assert_eq!(
                diffs,
                vec![
                    VecDiff::RemoveAt { index: 2 }, // Remove (4, 'd') from old sorted pos
                    VecDiff::InsertAt {
                        index: 0,
                        value: (0, 'z')
                    }  // Insert (0, 'z') at new sorted pos
                ],
                "Updating key did not produce correct move diffs."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![(0, 'z'), (2, 'b'), (3, 'C')],
                "State after moving update is incorrect."
            );

            // --- 6. Test Move (in source) ---
            // Source: [(3, 'C'), (0, 'z'), (2, 'b')] -> move 0 to 2 -> [(0, 'z'), (2, 'b'), (3,
            // 'C')] Sorted output should not change at all.
            source_vec.write(app.world_mut()).move_item(0, 2);
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert!(diffs.is_empty(), "Moving in source should not affect sorted output.");
            assert_eq!(current_state, vec![(0, 'z'), (2, 'b'), (3, 'C')]);

            // --- 7. Test Stability (Duplicate Keys) ---
            source_vec.write(app.world_mut()).push((3, 'c'));
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<(u32, char)>(app.world_mut()));
            assert_eq!(
                current_state,
                vec![(0, 'z'), (2, 'b'), (3, 'C'), (3, 'c')],
                "Stability with duplicate keys failed."
            );

            // --- 8. Test Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Clear]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 9. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_sort_by_cmp() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // The source vector, intentionally unsorted.
            let source_vec = MutableVec::builder().values([5, 1, 4, -2, 3]).spawn(app.world_mut());

            // The signal chain under test, using the convenience method.
            let sorted_signal = source_vec.signal_vec().sort_by_cmp();

            let handle = sorted_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            let mut current_state: Vec<i32> = vec![];

            // --- 2. Test Initial State ---
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![-2, 1, 3, 4, 5],
                "Initial state is not correctly sorted by cmp."
            );

            // --- 3. Test Push ---
            // Push an item that should be inserted in the middle.
            source_vec.write(app.world_mut()).push(2);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Push should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt { index: 2, value: 2 },
                "Pushing 2 should insert at sorted index 2."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![-2, 1, 2, 3, 4, 5], "State after Push is incorrect.");

            // --- 4. Test RemoveAt ---
            // Remove '4' from the source vec.
            // Source is now: `[5, 1, 4, -2, 3, 2]` -> remove at index 2 (`4`)
            // Source becomes: `[5, 1, -2, 3, 2]`
            // Sorted output before was: `[-2, 1, 2, 3, 4, 5]`
            // The item '4' is at sorted index 4.
            source_vec.write(app.world_mut()).remove(2);
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "RemoveAt should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::RemoveAt { index: 4 },
                "Removing 4 should remove at sorted index 4."
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![-2, 1, 2, 3, 5],
                "State after RemoveAt is incorrect."
            );

            // --- 5. Test Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<i32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Clear]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty(), "State after clear should be empty.");

            // --- 6. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_sort_by_key() {
        {
            // --- 1. Setup ---
            let mut app = create_test_app();

            #[derive(Clone, Debug, PartialEq, Default)]
            struct DataItem {
                id: i32,
                name: &'static str,
            }

            app.init_resource::<SignalVecOutput<DataItem>>();

            let source_vec = MutableVec::builder()
                .values([
                    DataItem { id: 3, name: "C" },
                    DataItem { id: 1, name: "A" },
                    DataItem { id: 4, name: "D" },
                ])
                .spawn(app.world_mut());

            let key_system = |In(item): In<DataItem>| item.id;
            let sorted_signal = source_vec.signal_vec().sort_by_key(key_system);
            let handle = sorted_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            let mut current_state: Vec<DataItem> = vec![];

            // --- 2. Test Initial State ---
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            apply_diffs(&mut current_state, diffs);
            assert_eq!(
                current_state,
                vec![
                    DataItem { id: 1, name: "A" },
                    DataItem { id: 3, name: "C" },
                    DataItem { id: 4, name: "D" }
                ],
                "Initial state is not correctly sorted by key."
            );

            // --- 3. Test Push ---
            source_vec.write(app.world_mut()).push(DataItem { id: 2, name: "B" });
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Push should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::InsertAt {
                    index: 1,
                    value: DataItem { id: 2, name: "B" }
                }
            );
            apply_diffs(&mut current_state, diffs);

            // --- 4. Test RemoveAt ---
            source_vec.write(app.world_mut()).remove(1); // Removes (1, 'A')
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert_eq!(diffs.len(), 1, "RemoveAt should produce one diff.");
            assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 });
            apply_diffs(&mut current_state, diffs);

            // --- 5. Test UpdateAt ---
            // Case A: Update without changing the key.
            source_vec
                .write(app.world_mut())
                .set(0, DataItem { id: 3, name: "C-new" });
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Update without key change should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::UpdateAt {
                    index: 1,
                    value: DataItem { id: 3, name: "C-new" }
                }
            );
            apply_diffs(&mut current_state, diffs);

            // Case B: Update that changes the key and sort order.
            source_vec.write(app.world_mut()).set(1, DataItem { id: 0, name: "Z" }); // was (4, 'D')
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert_eq!(diffs.len(), 2, "Update with key change should be Remove+Insert.");
            assert_eq!(
                diffs,
                vec![
                    VecDiff::RemoveAt { index: 2 },
                    VecDiff::InsertAt {
                        index: 0,
                        value: DataItem { id: 0, name: "Z" }
                    }
                ],
                "Updating key did not produce correct move diffs."
            );
            apply_diffs(&mut current_state, diffs);

            // --- 6. Test Move (in source) ---
            source_vec.write(app.world_mut()).move_item(0, 2);
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert!(diffs.is_empty(), "Moving in source should not affect sorted output.");

            // --- 7. Test Stability (Duplicate Keys) ---
            source_vec
                .write(app.world_mut())
                .push(DataItem { id: 3, name: "c-dup" });
            app.update();
            apply_diffs(&mut current_state, get_and_clear_output::<DataItem>(app.world_mut()));
            assert_eq!(
                current_state,
                vec![
                    DataItem { id: 0, name: "Z" },
                    DataItem { id: 2, name: "B" },
                    DataItem { id: 3, name: "C-new" },
                    DataItem { id: 3, name: "c-dup" }
                ],
                "Stability with duplicate keys failed."
            );

            // --- 8. Test Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_and_clear_output::<DataItem>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Clear]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 9. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_enumerate() {
        {
            // A comprehensive test for `SignalVecExt::enumerate`, covering all `VecDiff`
            // types. Since enumerate wraps each value with an index signal, we test
            // that the structure is correct rather than trying to evaluate the signals.

            // --- 1. Setup ---
            let mut app = create_test_app();

            type EnumeratedItem = (Broadcast<Option<usize>>, i32);

            #[derive(Resource, Default)]
            struct EnumOutput(Vec<VecDiff<EnumeratedItem>>);

            app.init_resource::<EnumOutput>();

            let source_vec = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());

            let enumerated = source_vec.signal_vec().enumerate();

            // Custom capture system that doesn't require Debug
            let capture = |In(diffs): In<Vec<VecDiff<EnumeratedItem>>>, mut output: ResMut<EnumOutput>| {
                output.0.extend(diffs);
            };

            let handle = enumerated.for_each(capture).register(app.world_mut());

            // Helper to get and clear diffs
            let get_diffs = |world: &mut World| -> Vec<VecDiff<EnumeratedItem>> {
                let output = world.resource::<EnumOutput>().0.clone();
                world.resource_mut::<EnumOutput>().0.clear();
                output
            };

            // Track all items with their index signals throughout the test
            let mut tracked_items: Vec<(Broadcast<Option<usize>>, i32)>;

            // Helper to verify index signals match their positions
            let verify_indices =
                |world: &mut World, items: &[(Broadcast<Option<usize>>, i32)], expected: &[(usize, i32)]| {
                    assert_eq!(items.len(), expected.len(), "Item count mismatch");
                    for (i, ((index_signal, value), (expected_index, expected_value))) in
                        items.iter().zip(expected.iter()).enumerate()
                    {
                        assert_eq!(*value, *expected_value, "Value mismatch at position {}", i);

                        // Register and poll the index signal to get its current value
                        let signal_claim = index_signal.clone().register(world);
                        let index = run_signal_subgraph_once_internal(world, signal_claim.entity())
                            .and_then(downcast_erased_signal_value::<Option<usize>>)
                            .flatten();
                        signal_claim.release(world);

                        assert_eq!(
                            index,
                            Some(*expected_index),
                            "Index signal at position {} should be Some({}), but got {:?}",
                            i,
                            expected_index,
                            index
                        );
                    }
                };

            // --- 2. Test Initial State (Replace) ---
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
            match &diffs[0] {
                VecDiff::Replace { values } => {
                    assert_eq!(values.len(), 3, "Should have 3 items initially.");
                    assert_eq!(values[0].1, 10, "First value should be 10.");
                    assert_eq!(values[1].1, 20, "Second value should be 20.");
                    assert_eq!(values[2].1, 30, "Third value should be 30.");

                    // Verify index signals
                    verify_indices(app.world_mut(), values, &[(0, 10), (1, 20), (2, 30)]);

                    // Track these items
                    tracked_items = values.to_vec();
                }
                _ => panic!("Expected Replace diff"),
            }

            // --- 3. Test Push ---
            source_vec.write(app.world_mut()).push(40);
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "Push should produce one diff.");
            match &diffs[0] {
                VecDiff::Push { value } => {
                    assert_eq!(value.1, 40, "Pushed value should be 40.");
                    // Verify the pushed item has index 3
                    verify_indices(app.world_mut(), core::slice::from_ref(value), &[(3, 40)]);

                    // Add to tracked items
                    tracked_items.push(value.clone());
                }
                _ => panic!("Expected Push diff"),
            }

            // Verify all existing items still have correct indices
            verify_indices(app.world_mut(), &tracked_items, &[(0, 10), (1, 20), (2, 30), (3, 40)]);

            // At this point we have: [10, 20, 30, 40] at indices [0, 1, 2, 3]

            // --- 4. Test InsertAt ---
            source_vec.write(app.world_mut()).insert(1, 15);
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "InsertAt should produce one diff.");
            match &diffs[0] {
                VecDiff::InsertAt { index, value } => {
                    assert_eq!(*index, 1, "Insert index should be 1.");
                    assert_eq!(value.1, 15, "Inserted value should be 15.");
                    // New item at index 1 should have index signal of 1
                    verify_indices(app.world_mut(), core::slice::from_ref(value), &[(1, 15)]);

                    // Insert into tracked items
                    tracked_items.insert(1, value.clone());
                }
                _ => panic!("Expected InsertAt diff"),
            }

            // After insert at 1, the vec is: [10@0, 15@1, 20@2, 30@3, 40@4]
            // CRITICAL: Verify that existing items' index signals have been updated!
            app.update(); // Allow index signals to propagate
            verify_indices(
                app.world_mut(),
                &tracked_items,
                &[(0, 10), (1, 15), (2, 20), (3, 30), (4, 40)],
            );

            // --- 5. Test UpdateAt ---
            source_vec.write(app.world_mut()).set(2, 25);
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "UpdateAt should produce one diff.");
            match &diffs[0] {
                VecDiff::UpdateAt { index, value } => {
                    assert_eq!(*index, 2, "Update index should be 2.");
                    assert_eq!(value.1, 25, "Updated value should be 25.");
                    // Updated item should keep index 2
                    verify_indices(app.world_mut(), core::slice::from_ref(value), &[(2, 25)]);

                    // Update tracked item
                    tracked_items[2] = value.clone();
                }
                _ => panic!("Expected UpdateAt diff"),
            }

            // Verify all items still have correct indices
            verify_indices(
                app.world_mut(),
                &tracked_items,
                &[(0, 10), (1, 15), (2, 25), (3, 30), (4, 40)],
            );

            // --- 6. Test RemoveAt ---
            source_vec.write(app.world_mut()).remove(1);
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "RemoveAt should produce one diff.");
            match &diffs[0] {
                VecDiff::RemoveAt { index } => {
                    assert_eq!(*index, 1, "Remove index should be 1.");

                    // Remove from tracked items
                    tracked_items.remove(1);
                }
                _ => panic!("Expected RemoveAt diff"),
            }

            // After removal, indices should be updated: [10@0, 25@1, 30@2, 40@3]
            // CRITICAL: Verify that remaining items' index signals have been updated!
            app.update(); // Allow index signals to propagate
            verify_indices(app.world_mut(), &tracked_items, &[(0, 10), (1, 25), (2, 30), (3, 40)]);

            // --- 7. Test Move ---
            let current_len = source_vec.read(app.world()).len();
            if current_len > 1 {
                let old_idx = current_len - 1;
                let new_idx = 0;
                source_vec.write(app.world_mut()).move_item(old_idx, new_idx);
                app.update();
                let diffs = get_diffs(app.world_mut());
                assert_eq!(diffs.len(), 1, "Move should produce one diff.");
                match &diffs[0] {
                    VecDiff::Move { old_index, new_index } => {
                        assert_eq!(*old_index, old_idx, "Old index should match.");
                        assert_eq!(*new_index, new_idx, "New index should match.");

                        // Update tracked items
                        let item = tracked_items.remove(old_idx);
                        tracked_items.insert(new_idx, item);
                    }
                    _ => panic!("Expected Move diff"),
                }
                // After move, indices should be updated
                // Vec is now: [40@0, 10@1, 25@2, 30@3]
                // CRITICAL: Verify that all items' index signals have been updated!
                app.update(); // Allow index signals to propagate
                verify_indices(app.world_mut(), &tracked_items, &[(0, 40), (1, 10), (2, 25), (3, 30)]);
            }

            // --- 8. Test Pop ---
            source_vec.write(app.world_mut()).pop();
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "Pop should produce one diff.");
            assert!(matches!(diffs[0], VecDiff::Pop), "Expected Pop diff.");

            // Remove from tracked items
            tracked_items.pop();
            // Verify remaining items still have correct indices: [40@0, 10@1, 25@2]
            verify_indices(app.world_mut(), &tracked_items, &[(0, 40), (1, 10), (2, 25)]);

            // --- 9. Test Clear ---
            source_vec.write(app.world_mut()).clear();
            app.update();
            let diffs = get_diffs(app.world_mut());
            assert_eq!(diffs.len(), 1, "Clear should produce one diff.");
            assert!(matches!(diffs[0], VecDiff::Clear), "Expected Clear diff.");

            // Clear tracked items
            tracked_items.clear();

            // --- 10. Test Re-population after clear ---
            source_vec.write(app.world_mut()).push(100);
            source_vec.write(app.world_mut()).push(200);
            app.update();

            let diffs = get_diffs(app.world_mut());
            // Should get Push diffs for the new items
            assert_eq!(diffs.len(), 2, "Should have 2 Push diffs.");
            match (&diffs[0], &diffs[1]) {
                (VecDiff::Push { value: v1 }, VecDiff::Push { value: v2 }) => {
                    assert_eq!(v1.1, 100, "First pushed value should be 100.");
                    assert_eq!(v2.1, 200, "Second pushed value should be 200.");
                    // Verify indices start from 0 again after clear
                    verify_indices(app.world_mut(), core::slice::from_ref(v1), &[(0, 100)]);
                    verify_indices(app.world_mut(), core::slice::from_ref(v2), &[(1, 200)]);
                }
                _ => panic!("Expected two Push diffs"),
            }

            // --- 11. Cleanup ---
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_flatten() {
        {
            // Comprehensive test for `SignalVecExt::flatten`:
            // - Initial state
            // - Inner vec changes (push, pop, update, remove, move, clear, replace)
            // - Outer vec changes (push, pop, insert, remove, move, clear, replace, update)
            // - Cleanup verification (removed inner vecs are ignored)
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32, 11]).spawn(app.world_mut());
            let vec_b = MutableVec::builder().values([20u32]).spawn(app.world_mut());
            let vec_c = MutableVec::<u32>::builder().spawn(app.world_mut());

            let source_of_vecs = MutableVec::builder()
                .values([
                    vec_a.signal_vec().broadcast(),
                    vec_b.signal_vec().broadcast(),
                    vec_c.signal_vec().broadcast(),
                ])
                .spawn(app.world_mut());

            let flattened_signal = source_of_vecs.signal_vec().flatten();

            let handle = flattened_signal
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            // --- 1. Initial State ---
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Replace {
                    values: vec![10, 11, 20]
                }]
            );
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20]);

            // --- 2. Inner Vec Change: Push to `vec_a` ---
            vec_a.write(app.world_mut()).push(12);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::InsertAt { index: 2, value: 12 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 12, 20]);

            // --- 3. Inner Vec Change: Push to `vec_b` ---
            vec_b.write(app.world_mut()).push(21);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::InsertAt { index: 4, value: 21 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 12, 20, 21]);

            // --- 4. Inner Vec Change: UpdateAt on `vec_a` ---
            vec_a.write(app.world_mut()).set(1, 111);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::UpdateAt { index: 1, value: 111 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 111, 12, 20, 21]);

            // --- 5. Inner Vec Change: RemoveAt on `vec_a` ---
            vec_a.write(app.world_mut()).remove(0);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::RemoveAt { index: 0 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![111, 12, 20, 21]);

            // --- 6. Inner Vec Change: Pop on `vec_b` ---
            vec_b.write(app.world_mut()).pop();
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::RemoveAt { index: 3 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![111, 12, 20]);

            // --- 7. Outer Vec Change: Remove `vec_b` ---
            source_of_vecs.write(app.world_mut()).remove(1);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::RemoveAt { index: 2 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![111, 12]);

            // --- 8. Cleanup Verification ---
            vec_b.write(app.world_mut()).push(99);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert!(diffs.is_empty(), "Should ignore diffs from the removed vec_b.");

            // --- 9. Outer Vec Change: Push a new vector ---
            let vec_d = MutableVec::builder().values([40u32, 41]).spawn(app.world_mut());
            source_of_vecs
                .write(app.world_mut())
                .push(vec_d.signal_vec().broadcast());
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 2, "Pushing new vec_d should produce 2 diffs.");
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![111, 12, 40, 41]);

            // --- 10. Inner Vec Change: Push to empty `vec_c` ---
            vec_c.write(app.world_mut()).push(30);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::InsertAt { index: 2, value: 30 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![111, 12, 30, 40, 41]);

            // --- 11. Outer Vec Change: Move ---
            // Current outer vec: [vec_a, vec_c, vec_d]. Move vec_d (index 2) to front (index 0).
            source_of_vecs.write(app.world_mut()).move_item(2, 0);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![40, 41, 111, 12, 30]);

            // --- 12. Outer Vec Change: UpdateAt (replace `vec_c` with `vec_e`) ---
            let vec_e = MutableVec::builder().values([50u32, 51, 52]).spawn(app.world_mut());
            source_of_vecs
                .write(app.world_mut())
                .set(2, vec_e.signal_vec().broadcast());
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![
                    VecDiff::UpdateAt { index: 4, value: 50 },
                    VecDiff::InsertAt { index: 5, value: 51 },
                    VecDiff::InsertAt { index: 6, value: 52 },
                ]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![40, 41, 111, 12, 50, 51, 52]);

            // --- 13. Outer Vec Change: Pop ---
            source_of_vecs.write(app.world_mut()).pop();
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![40, 41, 111, 12]);

            // --- 14. Outer Vec Change: Clear ---
            source_of_vecs.write(app.world_mut()).clear();
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Clear]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            // --- 15. Outer Vec Change: Replace (re-populate after clear) ---
            let vec_f = MutableVec::builder().values([60u32]).spawn(app.world_mut());
            let vec_g = MutableVec::builder().values([70u32, 71]).spawn(app.world_mut());
            source_of_vecs
                .write(app.world_mut())
                .replace(vec![vec_f.signal_vec().broadcast(), vec_g.signal_vec().broadcast()]);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Replace {
                    values: vec![60, 70, 71]
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![60, 70, 71]);

            // --- 16. Inner Vec Change: Move within `vec_g` ---
            vec_g.write(app.world_mut()).move_item(0, 1);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Move {
                    old_index: 1,
                    new_index: 2,
                }]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![60, 71, 70]);

            // --- 17. Inner Vec Change: Clear on `vec_g` ---
            vec_g.write(app.world_mut()).clear();
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 2, "Clear on vec_g should produce 2 RemoveAt diffs.");
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![60]);

            // --- 18. Inner Vec Change: Replace on `vec_f` ---
            vec_f.write(app.world_mut()).replace(vec![61, 62, 63]);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![61, 62, 63]);

            // Also verify inner Replace shrinking (3 items -> 1 item) emits UpdateAt + RemoveAt.
            vec_f.write(app.world_mut()).replace(vec![64]);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![
                    VecDiff::UpdateAt { index: 0, value: 64 },
                    VecDiff::RemoveAt { index: 1 },
                    VecDiff::RemoveAt { index: 1 },
                ]
            );
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![64]);

            // Also verify inner InsertAt.
            vec_f.write(app.world_mut()).insert(0, 59);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::InsertAt { index: 0, value: 59 }]);
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![59, 64]);

            handle.release(app.world_mut());
        }

        {
            // --- 19. InsertAt on outer vec (insert in middle) ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32, 11]).spawn(app.world_mut());
            let vec_b = MutableVec::builder().values([30u32, 31]).spawn(app.world_mut());

            let source = MutableVec::builder()
                .values([vec_a.signal_vec().broadcast(), vec_b.signal_vec().broadcast()])
                .spawn(app.world_mut());

            let handle = source
                .signal_vec()
                .flatten()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 30, 31]);

            let vec_mid = MutableVec::builder().values([20u32, 21, 22]).spawn(app.world_mut());
            source
                .write(app.world_mut())
                .insert(1, vec_mid.signal_vec().broadcast());
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20, 21, 22, 30, 31]);

            vec_mid.write(app.world_mut()).push(23);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20, 21, 22, 23, 30, 31]);

            vec_b.write(app.world_mut()).push(32);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20, 21, 22, 23, 30, 31, 32]);

            handle.release(app.world_mut());
        }

        {
            // --- 20. Multiple simultaneous inner vec changes ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32]).spawn(app.world_mut());
            let vec_b = MutableVec::builder().values([20u32]).spawn(app.world_mut());

            let source = MutableVec::builder()
                .values([vec_a.signal_vec().broadcast(), vec_b.signal_vec().broadcast()])
                .spawn(app.world_mut());

            let handle = source
                .signal_vec()
                .flatten()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 20]);

            vec_a.write(app.world_mut()).push(11);
            vec_b.write(app.world_mut()).push(21);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20, 21]);

            handle.release(app.world_mut());
        }

        {
            // --- 21. Empty outer vec initially (must establish Ready(vec![]) via Replace { values:
            // vec![] }) ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let source = MutableVec::<MutableVec<u32>>::builder().spawn(app.world_mut());

            let flattened = source
                .signal_vec()
                .map_in(|inner: MutableVec<u32>| inner.signal_vec())
                .flatten()
                .broadcast();
            let handle = flattened
                .clone()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);

            // Check that a late subscriber immediately receives the Ready(vec![]) replay!
            #[derive(Resource, Default)]
            struct LateLog(Vec<VecDiff<u32>>);
            app.init_resource::<LateLog>();
            let late_handle = flattened
                .for_each(|In(diffs): In<Vec<VecDiff<u32>>>, mut log: ResMut<LateLog>| {
                    log.0.extend(diffs);
                })
                .register(app.world_mut());
            app.update();
            assert_eq!(
                app.world().resource::<LateLog>().0,
                vec![VecDiff::Replace { values: vec![] }]
            );
            late_handle.release(app.world_mut());

            let mut current_state: Vec<u32> = vec![];
            let vec_a = MutableVec::builder().values([10u32, 11]).spawn(app.world_mut());
            source.write(app.world_mut()).push(vec_a);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11]);

            let vec_b = MutableVec::builder().values([20u32]).spawn(app.world_mut());
            source.write(app.world_mut()).push(vec_b);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20]);

            handle.release(app.world_mut());
        }

        {
            // --- 21b. Replace outer vec with empty (emits Replace { values: vec![] }) ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32, 11]).spawn(app.world_mut());
            let vec_b = MutableVec::builder().values([20u32]).spawn(app.world_mut());

            let source = MutableVec::builder()
                .values([vec_a.signal_vec().broadcast(), vec_b.signal_vec().broadcast()])
                .spawn(app.world_mut());

            let handle = source
                .signal_vec()
                .flatten()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20]);

            source
                .write(app.world_mut())
                .replace(Vec::<ReplayableBroadcastVec<u32>>::new());
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);
            apply_diffs(&mut current_state, diffs);
            assert!(current_state.is_empty());

            handle.release(app.world_mut());
        }

        {
            // --- 22. Move forward on outer vec (index 0 -> 2) ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32, 11]).spawn(app.world_mut());
            let vec_b = MutableVec::builder().values([20u32]).spawn(app.world_mut());
            let vec_c = MutableVec::builder().values([30u32, 31, 32]).spawn(app.world_mut());

            let source = MutableVec::builder()
                .values([
                    vec_a.signal_vec().broadcast(),
                    vec_b.signal_vec().broadcast(),
                    vec_c.signal_vec().broadcast(),
                ])
                .spawn(app.world_mut());

            let handle = source
                .signal_vec()
                .flatten()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20, 30, 31, 32]);

            source.write(app.world_mut()).move_item(0, 2);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![20, 30, 31, 32, 10, 11]);

            vec_a.write(app.world_mut()).push(12);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![20, 30, 31, 32, 10, 11, 12]);

            handle.release(app.world_mut());
        }

        {
            // --- 23. Moving an empty inner vec ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32]).spawn(app.world_mut());
            let vec_empty = MutableVec::<u32>::builder().spawn(app.world_mut());
            let vec_c = MutableVec::builder().values([30u32]).spawn(app.world_mut());

            let source = MutableVec::builder()
                .values([
                    vec_a.signal_vec().broadcast(),
                    vec_empty.signal_vec().broadcast(),
                    vec_c.signal_vec().broadcast(),
                ])
                .spawn(app.world_mut());

            let handle = source
                .signal_vec()
                .flatten()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 30]);

            source.write(app.world_mut()).move_item(1, 2);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert!(diffs.is_empty(), "Moving empty inner vec should produce no diffs");
            assert_eq!(current_state, vec![10, 30]);

            vec_empty.write(app.world_mut()).push(20);
            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 30, 20]);

            handle.release(app.world_mut());
        }

        {
            // --- 24. Moving a non-empty inner vec vs empty (verify Move diffs emitted) ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<u32>>();

            let vec_a = MutableVec::builder().values([10u32, 11]).spawn(app.world_mut());
            let vec_b = MutableVec::builder().values([20u32]).spawn(app.world_mut());

            let source = MutableVec::builder()
                .values([vec_a.signal_vec().broadcast(), vec_b.signal_vec().broadcast()])
                .spawn(app.world_mut());

            let handle = source
                .signal_vec()
                .flatten()
                .for_each(capture_signal_vec_output)
                .register(app.world_mut());

            app.update();
            let diffs = get_and_clear_output::<u32>(app.world_mut());
            let mut current_state: Vec<u32> = vec![];
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![10, 11, 20]);

            source.write(app.world_mut()).move_item(0, 1);
            app.update();

            let diffs = get_and_clear_output::<u32>(app.world_mut());
            assert_eq!(diffs.len(), 2, "Moving vec with 2 items should produce 2 Move diffs");
            apply_diffs(&mut current_state, diffs);
            assert_eq!(current_state, vec![20, 10, 11]);

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_flatten_warming_keep_current_and_reactivation() {
        #[derive(Resource)]
        struct Gate(bool);

        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<u32>>();

        let raw_a = MutableVec::builder().values([1u32, 2]).spawn(app.world_mut());
        let raw_b = MutableVec::builder().values([10u32]).spawn(app.world_mut());

        // Inner `CachedSignalVec` gated on `Res<Gate>` starts in `Warming` until `Gate(true)`
        // exists.
        let warming_a = {
            use crate::signal::SignalExt;
            signal::from_system({
                let raw_a = raw_a.clone();
                move |In(()), world: &World| {
                    let open = world.get_resource::<Gate>().is_some_and(|g| g.0);
                    open.then(|| raw_a.read(world).to_vec())
                }
            })
            .dedupe()
            .to_signal_vec()
            .cache()
            .broadcast()
        };
        let ready_b = raw_b.signal_vec().broadcast();

        let selectors = MutableVec::builder().values([0u8, 1u8]).spawn(app.world_mut());
        let flattened = selectors
            .signal_vec()
            .map_in(clone!((warming_a, ready_b) move |id: u8| {
                if id == 0 {
                    warming_a.clone().boxed_replayable()
                } else {
                    ready_b.clone().boxed_replayable()
                }
            }))
            .flatten()
            .broadcast();

        let claim = flattened
            .clone()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        // Frame 1: `warming_a` has not warmed yet, so `Replace` is gated and nothing publishes yet.
        app.update();
        assert!(get_and_clear_output::<u32>(app.world_mut()).is_empty());

        // Frame 2: Enable `Gate(true)` and trigger `raw_a` so `warming_a` warms!
        app.insert_resource(Gate(true));
        raw_a.write(app.world_mut()).push(3);
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        let mut state = Vec::new();
        apply_diffs(&mut state, diffs);
        assert_eq!(state, vec![1, 2, 3, 10]);

        // SameSignalIdentity::KeepCurrent: setting slot 1 to the same `ready_b` identity emits no
        // diffs!
        selectors.write(app.world_mut()).set(1, 1);
        app.update();
        assert!(
            get_and_clear_output::<u32>(app.world_mut()).is_empty(),
            "Re-selecting the same inner signal identity must keep current without redundant diffs"
        );

        // `1 -> 0` deactivation: release the only claim on `flattened` (and its downstream
        // `for_each`).
        claim.release(app.world_mut());

        // Mutate while deactivated (`0` claims).
        raw_a.write(app.world_mut()).replace(vec![100, 200]);
        raw_b.write(app.world_mut()).replace(vec![300]);
        selectors.write(app.world_mut()).replace(vec![1u8, 0u8]);
        app.update();

        // `0 -> 1` reactivation: re-registering `flattened` must start cleanly from current state
        // without stale pre-deactivation `entries` or `slot_count`!
        let claim2 = flattened.for_each(capture_signal_vec_output).register(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        let mut reactivated_state = Vec::new();
        apply_diffs(&mut reactivated_state, diffs);
        assert_eq!(reactivated_state, vec![300, 100, 200]);

        // Also verify `1 -> 0 -> 1` reactivation for `MapSignal` and `FilterSignal`.
        let map_src = MutableVec::builder().values([2u32, 4]).spawn(app.world_mut());
        let map_sig = map_src
            .signal_vec()
            .map_signal(|In(x): In<u32>| signal::from_system(move |In(())| x * 10).dedupe())
            .broadcast();
        let map_claim = map_sig
            .clone()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![20, 40] }]
        );
        map_claim.release(app.world_mut());
        map_src.write(app.world_mut()).replace(vec![5u32]);
        app.update();
        let map_claim2 = map_sig.for_each(capture_signal_vec_output).register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![50] }]
        );
        map_claim2.release(app.world_mut());

        let filter_src = MutableVec::builder().values([1u32, 2, 3]).spawn(app.world_mut());
        let filter_sig = filter_src
            .signal_vec()
            .filter_signal(|In(x): In<u32>| signal::from_system(move |In(())| x % 2 == 0).dedupe())
            .broadcast();
        let filter_claim = filter_sig
            .clone()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![2] }]
        );
        filter_claim.release(app.world_mut());
        filter_src.write(app.world_mut()).replace(vec![4u32, 5, 6]);
        app.update();
        let filter_claim2 = filter_sig.for_each(capture_signal_vec_output).register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![4, 6] }]
        );
        filter_claim2.release(app.world_mut());

        claim2.release(app.world_mut());
    }

    #[test]
    fn test_degenerate_move_and_clear_suppression() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<u32>>();

        // 1. `filter`: moving a passing item across only filtered-out items must NOT emit a
        //    degenerate `Move { old_index: 0, new_index: 0 }`, and `Clear` when 0 items pass must
        //    NOT emit `VecDiff::Clear`.
        let src = MutableVec::builder().values([2u32, 1, 3]).spawn(app.world_mut());
        let handle = src
            .signal_vec()
            .filter(|In(x): In<u32>| x % 2 == 0)
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![2] }]
        );

        // Move `2` from index 0 to index 2 across odd items `1` and `3`: output index stays 0!
        src.write(app.world_mut()).move_item(0, 2);
        app.update();
        assert!(
            get_and_clear_output::<u32>(app.world_mut()).is_empty(),
            "filter must suppress degenerate Move when output index is unchanged"
        );

        // Remove the only even item (`2`, now at index 2), leaving `[1, 3]` (0 passing items).
        src.write(app.world_mut()).remove(2);
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::RemoveAt { index: 0 }]
        );

        // Now clear `[1, 3]`: downstream is already empty, so no `VecDiff::Clear` should fire!
        src.write(app.world_mut()).clear();
        app.update();
        assert!(
            get_and_clear_output::<u32>(app.world_mut()).is_empty(),
            "filter must suppress degenerate Clear when zero items were passing"
        );
        handle.release(app.world_mut());

        // 2. `filter_signal`: moving a passing item across non-passing items suppresses degenerate
        //    Move.
        let src_fs = MutableVec::builder().values([2u32, 1, 3]).spawn(app.world_mut());
        let handle_fs = src_fs
            .signal_vec()
            .filter_signal(|In(x): In<u32>| signal::from_system(move |In(())| x % 2 == 0).dedupe())
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![2] }]
        );
        src_fs.write(app.world_mut()).move_item(0, 2);
        app.update();
        assert!(
            get_and_clear_output::<u32>(app.world_mut()).is_empty(),
            "filter_signal must suppress degenerate Move when output index is unchanged"
        );
        handle_fs.release(app.world_mut());

        // 3. `map_signal`: moving a ready slot across warming slots suppresses degenerate Move.
        #[derive(Resource)]
        struct WarmOdd;
        let src_ms2 = MutableVec::builder().values([2u32]).spawn(app.world_mut());
        let handle_ms2 = src_ms2
            .signal_vec()
            .map_signal(|In(x): In<u32>| {
                use crate::signal::SignalExt;
                signal::from_system(move |In(()), warm: Option<Res<WarmOdd>>| {
                    (x % 2 == 0 || warm.is_some()).then_some(x * 10)
                })
                .dedupe()
                .cache()
            })
            .for_each(capture_signal_vec_output::<u32>)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_output::<u32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![20] }]
        );
        src_ms2.write(app.world_mut()).push(1);
        src_ms2.write(app.world_mut()).push(3);
        app.update();
        assert!(get_and_clear_output::<u32>(app.world_mut()).is_empty());

        src_ms2.write(app.world_mut()).move_item(0, 2);
        app.update();
        assert!(
            get_and_clear_output::<u32>(app.world_mut()).is_empty(),
            "map_signal must suppress degenerate Move across warming slots"
        );
        handle_ms2.release(app.world_mut());
    }

    #[test]
    fn test_boxed_replayable_vec_inherent_methods_preserve_replayable_with_prelude() {
        use crate::prelude::*;

        fn assert_replayable<S: ReplayableSignalVec>(_signal: &S) -> &S {
            _signal
        }

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default().with_schedule::<Update>()));

        let src = MutableVec::builder().values([1, 2, 3]).spawn(app.world_mut());

        let cached = src.signal_vec().boxed_replayable().cache();
        assert_replayable(&cached);

        let broadcast = cached.boxed_replayable().broadcast();
        assert_replayable(&broadcast);

        let scheduled = broadcast.clone().boxed_replayable().schedule::<Update>();
        assert_replayable(&scheduled);

        let claim = scheduled.register(app.world_mut());
        claim.release(app.world_mut());
    }
}
