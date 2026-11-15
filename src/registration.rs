//! Preflighted signal-entity registration and custom combinator-authoring APIs.
//!
//! This module provides [`SignalRegistrar`], [`SignalMaterializer`], branded [`SignalEntityRef`]
//! tokens, [`OutputDelivery`], [`ReplayInitialization`], and [`SignalRegistrationError`] for
//! authoring move-first static and dynamic signal combinators. Graph mutation begins only after
//! the complete candidate, including dynamic adapters and connections, has passed allocation-free
//! preflight.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet, VecDeque},
    vec::Vec,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    schedule::{InternedScheduleLabel, ScheduleLabel},
    system::{BoxedSystem, In, IntoSystem, SystemId},
    world::World,
};
use bevy_platform::sync::{
    Arc, Mutex, OnceLock, Weak,
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
};
use core::{
    any::{Any, TypeId, type_name},
    cmp::Ordering,
    fmt,
    marker::PhantomData,
};

use crate::{
    cleanup::{self, CleanupSender},
    dynamic::{DynamicAdapterBinding, DynamicSignalConnection, DynamicStableSlot},
    graph::{ErasedSignalValue, ReleasedSignalCandidate, SignalClaim, SignalEntity, erased_type_id},
    replay::{
        MapReplayCache, ReplayCacheReadiness, ReplayCacheSnapshot, ReplayCacheUpdate, ReplayRevision,
        ScalarReplayCache, VecReplayCache,
    },
    signal_map::MapDiff,
    signal_vec::{ChainInput, VecDiff},
};

/// A detectable failure while planning or materializing a signal entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignalRegistrationError {
    /// An existing identity was redeclared with different canonical static state.
    ConflictingCanonicalState,
    /// A move-only signal entity would have more than one distinct active or pending dependent.
    ///
    /// Place `.broadcast()` (or [`OutputDelivery::broadcast`]) upstream of the branch before
    /// attaching multiple distinct consumers.
    MoveOnlyFanout,
    /// The combined existing and pending static dependency graph contains a cycle.
    CycleDetected,
    /// A requested schedule was not registered with Jonmo.
    ScheduleNotRegistered,
    /// A planned direct claim count would overflow.
    ClaimCountOverflow,
    /// A projected runnable graph level would exceed the representable graph depth.
    GraphDepthOverflow,
    /// A referenced existing entity is no longer an available signal entity.
    EntityUnavailable,
    /// An existing signal entity's claim count disagrees with its committed runtime projection.
    InconsistentExistingRuntimeState {
        /// The existing signal entity whose runtime projection is inconsistent.
        entity: SignalEntity,
    },

    /// A shared materializer is still initializing in another nested materialization attempt.
    SharedMaterializerInitializing,
    /// A shared one-shot materializer was permanently rejected by an earlier attempt.
    SharedMaterializerRejected,
    /// A shared one-shot materializer's initializer or commit previously panicked.
    SharedMaterializerPanicked,
    /// A homogeneous registrar input collection was empty; use `()` for a scheduler-invoked root.
    EmptyRegistrarInputs,
    /// A dynamic slot command was invalid for the current logical layout.
    InvalidDynamicCommand,
    /// A dynamic route conflicts with an existing static or dynamic route.
    DynamicRouteConflict,
    /// A selected input did not provide the declared framework replay contract.
    DynamicReplayContractMismatch,
    /// A dynamic slot or cohort generation would overflow.
    DynamicGenerationOverflow,
}

impl fmt::Display for SignalRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingCanonicalState => {
                formatter.write_str("signal entity redeclared with conflicting canonical state")
            }
            Self::MoveOnlyFanout => formatter.write_str(
                "move-only signal entity cannot feed more than one distinct active dependent; place `.broadcast()` upstream of the branch",
            ),
            Self::CycleDetected => formatter.write_str("signal dependency graph contains a cycle"),
            Self::ScheduleNotRegistered => formatter.write_str("signal schedule is not registered with JonmoPlugin"),
            Self::ClaimCountOverflow => formatter.write_str("signal claim count would overflow"),
            Self::GraphDepthOverflow => formatter.write_str("signal graph depth would overflow"),
            Self::EntityUnavailable => formatter.write_str("referenced signal entity is unavailable"),
            Self::InconsistentExistingRuntimeState { entity } => {
                write!(formatter, "existing signal entity {entity:?} has inconsistent runtime state")
            }
            Self::SharedMaterializerInitializing => {
                formatter.write_str("shared signal materializer is already initializing")
            }
            Self::SharedMaterializerRejected => {
                formatter.write_str("shared signal materializer was previously rejected")
            }
            Self::SharedMaterializerPanicked => formatter.write_str("shared signal materializer previously panicked"),
            Self::EmptyRegistrarInputs => {
                formatter.write_str("homogeneous registrar inputs were empty; use `()` for a root signal entity")
            }
            Self::InvalidDynamicCommand => formatter.write_str("invalid dynamic slot command"),
            Self::DynamicRouteConflict => formatter.write_str("dynamic route conflicts with an existing route"),
            Self::DynamicReplayContractMismatch => {
                formatter.write_str("selected dynamic input does not satisfy the declared replay contract")
            }
            Self::DynamicGenerationOverflow => formatter.write_str("dynamic generation counter would overflow"),
        }
    }
}

impl core::error::Error for SignalRegistrationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputDeliveryKind {
    Move,
    Broadcast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputDeliveryKey {
    pub(crate) kind: OutputDeliveryKind,
    pub(crate) output_type: TypeId,
}

pub(crate) type ErasedOutputClone = fn(&dyn Any) -> ErasedSignalValue;

fn clone_erased_output<T: Clone + Send + Sync + 'static>(value: &dyn Any) -> ErasedSignalValue {
    let typed = value.downcast_ref::<T>().unwrap_or_else(|| {
        panic!(
            "broadcast output delivery received unexpected value type; expected {}",
            type_name::<T>()
        )
    });
    Box::new(typed.clone())
}

#[derive(Component, Clone, Copy)]
pub(crate) struct StoredOutputDelivery {
    key: OutputDeliveryKey,
    clone_value: Option<ErasedOutputClone>,
}

impl StoredOutputDelivery {
    pub(crate) fn new_move(output_type: TypeId) -> Self {
        Self {
            key: OutputDeliveryKey {
                kind: OutputDeliveryKind::Move,
                output_type,
            },
            clone_value: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_incoherent(key: OutputDeliveryKey, clone_value: Option<ErasedOutputClone>) -> Self {
        Self { key, clone_value }
    }

    pub(crate) fn key(self) -> OutputDeliveryKey {
        self.key
    }

    pub(crate) fn kind(self) -> OutputDeliveryKind {
        self.key.kind
    }

    pub(crate) fn clone_value(self) -> Option<ErasedOutputClone> {
        self.clone_value
    }

    pub(crate) fn is_coherent(self, expected_output_type: TypeId) -> bool {
        self.key.output_type == expected_output_type
            && matches!(
                (self.key.kind, self.clone_value),
                (OutputDeliveryKind::Move, None) | (OutputDeliveryKind::Broadcast, Some(_))
            )
    }
}

impl fmt::Debug for StoredOutputDelivery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredOutputDelivery")
            .field("key", &self.key)
            .field("has_clone_fn", &self.clone_value.is_some())
            .finish()
    }
}

/// Explicit output-delivery policy capability for a materialized signal entity.
///
/// Move-only entities ([`OutputDelivery::move_only`]) may feed at most one distinct active or
/// pending dependent and forward outputs by move without cloning. Broadcast entities
/// ([`OutputDelivery::broadcast`]) require [`Clone`] outputs, permit multiple distinct active
/// dependents, clone `N - 1` times for `N` live recipients, and move the original value to the
/// final deterministic recipient.
///
/// Non-cloneable types cannot construct a broadcast policy:
///
/// ```compile_fail,E0599
/// use jonmo::OutputDelivery;
///
/// struct NonClone(String);
///
/// let _ = OutputDelivery::<NonClone>::broadcast();
/// ```
pub struct OutputDelivery<T> {
    stored: StoredOutputDelivery,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Copy for OutputDelivery<T> {}

impl<T> Clone for OutputDelivery<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for OutputDelivery<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.stored.kind() {
            OutputDeliveryKind::Move => formatter.write_str("OutputDelivery::MoveOnly"),
            OutputDeliveryKind::Broadcast => formatter.write_str("OutputDelivery::Broadcast"),
        }
    }
}

impl<T> OutputDelivery<T>
where
    T: Send + Sync + 'static,
{
    /// Permits at most one distinct active dependent and forwards values by move without cloning.
    pub fn move_only() -> Self {
        Self {
            stored: StoredOutputDelivery::new_move(TypeId::of::<T>()),
            _marker: PhantomData,
        }
    }
}

impl<T> OutputDelivery<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Permits multiple distinct active dependents using framework-managed cloning.
    pub fn broadcast() -> Self {
        Self {
            stored: StoredOutputDelivery {
                key: OutputDeliveryKey {
                    kind: OutputDeliveryKind::Broadcast,
                    output_type: TypeId::of::<T>(),
                },
                clone_value: Some(clone_erased_output::<T>),
            },
            _marker: PhantomData,
        }
    }
}

impl<T: 'static> OutputDelivery<T> {
    pub(crate) fn into_stored(self) -> StoredOutputDelivery {
        self.stored
    }

    pub(crate) fn from_stored(stored: StoredOutputDelivery) -> Self {
        assert!(
            stored.is_coherent(TypeId::of::<T>()),
            "stored output delivery policy did not match expected output type {}",
            type_name::<T>()
        );
        Self {
            stored,
            _marker: PhantomData,
        }
    }
}

#[allow(private_interfaces)]
mod sealed {
    pub trait SignalKind {}

    pub trait RegistrarInputs<'materialization> {
        fn allows_empty(&self) -> bool {
            false
        }

        fn into_planned_inputs(self) -> alloc::vec::Vec<super::PlannedSignalEntityRef>;
    }
}

/// A framework-controlled marker describing one family of signal output.
#[allow(private_bounds)]
pub trait SignalKind: sealed::SignalKind + Send + Sync + 'static {
    /// The event forwarded by an entity of this kind.
    type Event: 'static;
}

/// The kind marker for a scalar signal whose current event is `T`.
pub struct ScalarSignalKind<T>(PhantomData<fn() -> T>);

/// The kind marker for a vector signal whose current event is a batch of vector diffs.
pub struct VecSignalKind<T>(PhantomData<fn() -> T>);

/// The kind marker for a map signal whose current event is a batch of map diffs.
pub struct MapSignalKind<K, V>(PhantomData<fn() -> (K, V)>);

impl<T> sealed::SignalKind for ScalarSignalKind<T> {}
impl<T> sealed::SignalKind for VecSignalKind<T> {}
impl<K, V> sealed::SignalKind for MapSignalKind<K, V> {}

impl<T: 'static> SignalKind for ScalarSignalKind<T> {
    type Event = T;
}

impl<T: 'static> SignalKind for VecSignalKind<T> {
    type Event = Vec<VecDiff<T>>;
}

impl<K: 'static, V: 'static> SignalKind for MapSignalKind<K, V> {
    type Event = Vec<MapDiff<K, V>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MaterializationPlanId(usize);

impl MaterializationPlanId {
    fn next() -> Self {
        static NEXT_PLAN_ID: AtomicUsize = AtomicUsize::new(1);

        let id = cleanup::update_atomic_usize(
            &NEXT_PLAN_ID,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            |current| current.checked_add(1),
        )
        .unwrap_or_else(|_| panic!("signal materialization plan identity overflowed"));
        Self(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlannedSignalEntityRef {
    Existing(SignalEntity),
    Pending {
        plan_id: MaterializationPlanId,
        index: usize,
    },
}

impl PlannedSignalEntityRef {
    pub(crate) fn ordering_key(self) -> (u8, u64, u64) {
        match self {
            Self::Existing(entity) => (0, entity.entity().to_bits(), 0),
            Self::Pending { plan_id, index } => (1, plan_id.0 as u64, index as u64),
        }
    }
}

impl PartialOrd for PlannedSignalEntityRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PlannedSignalEntityRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ordering_key().cmp(&other.ordering_key())
    }
}

/// A verified existing-or-pending signal capability scoped to one materialization attempt.
///
/// This is not an ECS borrow and does not expose a raw identity. Copying it creates no claim,
/// route, fan-out permission, or dependency multiplicity. Its invariant lifetime brand binds it to
/// the registrar scope that produced it, and its fields cannot be constructed or inspected outside
/// Jonmo:
///
/// ```compile_fail,E0599
/// use jonmo::{ScalarSignalEntityRef, graph::SignalEntity};
///
/// fn expose(reference: ScalarSignalEntityRef<'_, i32>) -> SignalEntity {
///     reference.entity()
/// }
/// ```
///
/// ```compile_fail,E0451
/// use jonmo::{ScalarSignalKind, SignalEntityRef};
///
/// fn inspect(reference: SignalEntityRef<'_, ScalarSignalKind<i32>>) {
///     let SignalEntityRef {
///         planned,
///         invariant_brand,
///         kind,
///     } = reference;
/// }
/// ```
///
/// The invariant brand prevents a reference from escaping its materialization scope:
///
/// ```compile_fail
/// use bevy_ecs::prelude::*;
/// use jonmo::{OutputDelivery, ScalarSignalEntityRef, SignalRegistrar};
///
/// fn escape<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> ScalarSignalEntityRef<'static, i32> {
///     registrar
///         .signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
///         .unwrap()
/// }
/// ```
///
/// Two materialization scopes cannot exchange references:
///
/// ```compile_fail
/// use bevy_ecs::prelude::*;
/// use jonmo::{OutputDelivery, ScalarSignalEntityRef, SignalRegistrar, SignalRegistrationError};
///
/// fn cross<'world, 'left: 'right, 'right>(
///     left: &mut SignalRegistrar<'world, 'left>,
///     right: &mut SignalRegistrar<'world, 'right>,
/// ) -> Result<ScalarSignalEntityRef<'right, i32>, SignalRegistrationError> {
///     let source = left.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
///     right.signal_entity(source, |In(value): In<i32>| Some(value), OutputDelivery::move_only())
/// }
/// ```
///
/// Generic lifetime rebinding is likewise unavailable:
///
/// ```compile_fail
/// use jonmo::ScalarSignalEntityRef;
///
/// fn rebind<'left: 'right, 'right>(
///     reference: ScalarSignalEntityRef<'left, i32>,
/// ) -> ScalarSignalEntityRef<'right, i32> {
///     reference
/// }
/// ```
///
/// Ordinary explicitly named materialization lifetimes remain usable:
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{OutputDelivery, ScalarSignalEntityRef, SignalRegistrar, SignalRegistrationError};
///
/// fn unary<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
///     input: ScalarSignalEntityRef<'materialization, i32>,
/// ) -> Result<ScalarSignalEntityRef<'materialization, i32>, SignalRegistrationError> {
///     registrar.signal_entity(
///         input,
///         |In(value): In<i32>| Some(value + 1),
///         OutputDelivery::move_only(),
///     )
/// }
/// ```
///
/// A scoped identity cannot be converted into cleanup authority:
///
/// ```compile_fail
/// use jonmo::{ScalarSignalEntityRef, graph::SignalClaim};
///
/// fn fabricate(reference: ScalarSignalEntityRef<'_, i32>) -> SignalClaim {
///     reference.into()
/// }
/// ```
pub struct SignalEntityRef<'materialization, Kind: SignalKind> {
    planned: PlannedSignalEntityRef,
    invariant_brand: PhantomData<fn(&'materialization mut ()) -> &'materialization mut ()>,
    kind: PhantomData<fn() -> Kind>,
}

impl<Kind: SignalKind> Copy for SignalEntityRef<'_, Kind> {}

impl<Kind: SignalKind> Clone for SignalEntityRef<'_, Kind> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Kind: SignalKind> fmt::Debug for SignalEntityRef<'_, Kind> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SignalEntityRef { .. }")
    }
}

impl<'materialization, Kind: SignalKind> SignalEntityRef<'materialization, Kind> {
    fn new(planned: PlannedSignalEntityRef) -> Self {
        Self {
            planned,
            invariant_brand: PhantomData,
            kind: PhantomData,
        }
    }
}

/// A scoped scalar signal-entity reference.
pub type ScalarSignalEntityRef<'materialization, T> = SignalEntityRef<'materialization, ScalarSignalKind<T>>;
/// A scoped vector signal-entity reference.
pub type VecSignalEntityRef<'materialization, T> = SignalEntityRef<'materialization, VecSignalKind<T>>;
/// A scoped map signal-entity reference.
pub type MapSignalEntityRef<'materialization, K, V> = SignalEntityRef<'materialization, MapSignalKind<K, V>>;

/// A framework-controlled homogeneous dependency shape for registrar constructors.
///
/// External implementations are rejected by the private sealed supertrait:
///
/// ```compile_fail
/// use jonmo::RegistrarInputs;
///
/// struct CounterfeitInputs;
///
/// impl<'materialization> RegistrarInputs<'materialization> for CounterfeitInputs {
///     type Event = ();
/// }
/// ```
///
/// Signal kinds cannot be mixed in one homogeneous collection:
///
/// ```compile_fail
/// use bevy_ecs::prelude::*;
/// use jonmo::{OutputDelivery, ReplayInitialization, ScalarSignalMaterializer};
///
/// let materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
///     let scalar = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
///     let vector = registrar.replayable_signal_vec_entity(
///         (),
///         |In(())| Some(Vec::<jonmo::signal_vec::VecDiff<i32>>::new()),
///         ReplayInitialization::Warming,
///         OutputDelivery::move_only(),
///     )?;
///     registrar.signal_entity(
///         [scalar, vector],
///         |In(value): In<i32>| Some(value),
///         OutputDelivery::move_only(),
///     )
/// });
/// ```
#[allow(private_bounds)]
pub trait RegistrarInputs<'materialization>: sealed::RegistrarInputs<'materialization> {
    /// The common event type emitted by every dependency in this shape.
    type Event: 'static;
}

#[allow(private_interfaces)]
impl<'materialization> sealed::RegistrarInputs<'materialization> for () {
    fn allows_empty(&self) -> bool {
        true
    }

    fn into_planned_inputs(self) -> Vec<PlannedSignalEntityRef> {
        Vec::new()
    }
}

impl RegistrarInputs<'_> for () {
    type Event = ();
}

#[allow(private_interfaces)]
impl<'materialization, Kind: SignalKind> sealed::RegistrarInputs<'materialization>
    for SignalEntityRef<'materialization, Kind>
{
    fn into_planned_inputs(self) -> Vec<PlannedSignalEntityRef> {
        Vec::from([self.planned])
    }
}

impl<'materialization, Kind: SignalKind> RegistrarInputs<'materialization> for SignalEntityRef<'materialization, Kind> {
    type Event = Kind::Event;
}

#[allow(private_interfaces)]
impl<'materialization, Kind: SignalKind, const LENGTH: usize> sealed::RegistrarInputs<'materialization>
    for [SignalEntityRef<'materialization, Kind>; LENGTH]
{
    fn into_planned_inputs(self) -> Vec<PlannedSignalEntityRef> {
        self.into_iter().map(|input| input.planned).collect()
    }
}

impl<'materialization, Kind: SignalKind, const LENGTH: usize> RegistrarInputs<'materialization>
    for [SignalEntityRef<'materialization, Kind>; LENGTH]
{
    type Event = Kind::Event;
}

#[allow(private_interfaces)]
impl<'materialization, Kind: SignalKind> sealed::RegistrarInputs<'materialization>
    for Vec<SignalEntityRef<'materialization, Kind>>
{
    fn into_planned_inputs(self) -> Vec<PlannedSignalEntityRef> {
        self.into_iter().map(|input| input.planned).collect()
    }
}

impl<'materialization, Kind: SignalKind> RegistrarInputs<'materialization>
    for Vec<SignalEntityRef<'materialization, Kind>>
{
    type Event = Kind::Event;
}

#[allow(private_interfaces)]
impl<'materialization, Kind: SignalKind> sealed::RegistrarInputs<'materialization>
    for Box<[SignalEntityRef<'materialization, Kind>]>
{
    fn into_planned_inputs(self) -> Vec<PlannedSignalEntityRef> {
        self.into_vec().into_iter().map(|input| input.planned).collect()
    }
}

impl<'materialization, Kind: SignalKind> RegistrarInputs<'materialization>
    for Box<[SignalEntityRef<'materialization, Kind>]>
{
    type Event = Kind::Event;
}

/// Initial state for an ordinary derived framework-managed replayable entity.
///
/// Arbitrary ready state cannot be attached to a derived entity:
///
/// ```compile_fail,E0599
/// use bevy_ecs::prelude::*;
/// use jonmo::{OutputDelivery, ReplayInitialization, ScalarSignalMaterializer};
///
/// let materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
///     let source = registrar.signal_entity((), |In(())| 1, OutputDelivery::move_only())?;
///     registrar.replayable_signal_entity(
///         source,
///         |In(value): In<i32>| value,
///         ReplayInitialization::Ready(99),
///         OutputDelivery::move_only(),
///     )
/// });
/// ```
#[non_exhaustive]
pub enum ReplayInitialization {
    /// The entity has no authoritative current state yet.
    Warming,
}

pub(crate) enum ErasedReplaySnapshot {
    Warming,
    Ready {
        snapshot: ErasedSignalValue,
        revision: ReplayRevision,
    },
}

#[derive(Clone)]
pub(crate) enum PlannedReplayInspection {
    Warming,
    Ready {
        make_snapshot: Arc<dyn Fn() -> ErasedSignalValue + Send + Sync>,
        revision: ReplayRevision,
    },
}

impl PlannedReplayInspection {
    pub(crate) fn snapshot(&self) -> ErasedReplaySnapshot {
        match self {
            Self::Warming => ErasedReplaySnapshot::Warming,
            Self::Ready {
                make_snapshot,
                revision,
            } => ErasedReplaySnapshot::Ready {
                snapshot: make_snapshot(),
                revision: *revision,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReplayRuntimeKey {
    cache_type: TypeId,
    event_type: TypeId,
}

impl ReplayRuntimeKey {
    pub(crate) fn new<Cache: 'static, Event: 'static>() -> Self {
        Self {
            cache_type: TypeId::of::<Cache>(),
            event_type: TypeId::of::<Event>(),
        }
    }

    pub(crate) fn cache_type(self) -> TypeId {
        self.cache_type
    }

    pub(crate) fn event_type(self) -> TypeId {
        self.event_type
    }
}

pub(crate) trait ReplayRuntimeContract: Send + Sync {
    fn cache_type(&self) -> TypeId;
    fn event_type(&self) -> TypeId;
    fn coalesces_pending_snapshot(&self) -> bool {
        false
    }
    fn readiness_and_event_type(&self, world: &World, entity: SignalEntity) -> Option<(ReplayCacheReadiness, TypeId)> {
        match self.inspect(world, entity)? {
            ErasedReplaySnapshot::Warming => Some((ReplayCacheReadiness::Warming, self.event_type())),
            ErasedReplaySnapshot::Ready { snapshot, revision } => {
                Some((ReplayCacheReadiness::Ready { revision }, erased_type_id(&snapshot)))
            }
        }
    }
    fn inspect(&self, world: &World, entity: SignalEntity) -> Option<ErasedReplaySnapshot>;
    fn clone_event(&self, _value: &dyn Any) -> ErasedSignalValue {
        panic!("replay contract does not support event cloning")
    }
    fn accept_output(
        &self,
        world: &mut World,
        entity: SignalEntity,
        output: ErasedSignalValue,
    ) -> Option<AcceptedSignalOutput>;
}

struct ScalarReplayRuntimeContract<T>(PhantomData<fn() -> T>);

impl<T: Clone + Send + Sync + 'static> ReplayRuntimeContract for ScalarReplayRuntimeContract<T> {
    fn cache_type(&self) -> TypeId {
        TypeId::of::<ScalarReplayCache<T>>()
    }

    fn event_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn coalesces_pending_snapshot(&self) -> bool {
        false
    }

    fn readiness_and_event_type(&self, world: &World, entity: SignalEntity) -> Option<(ReplayCacheReadiness, TypeId)> {
        Some((
            world.get::<ScalarReplayCache<T>>(*entity)?.readiness(),
            TypeId::of::<T>(),
        ))
    }

    fn inspect(&self, world: &World, entity: SignalEntity) -> Option<ErasedReplaySnapshot> {
        match world.get::<ScalarReplayCache<T>>(*entity)?.snapshot() {
            ReplayCacheSnapshot::Warming => Some(ErasedReplaySnapshot::Warming),
            ReplayCacheSnapshot::Ready { current, revision } => Some(ErasedReplaySnapshot::Ready {
                snapshot: Box::new(current.clone()),
                revision,
            }),
        }
    }

    fn clone_event(&self, value: &dyn Any) -> ErasedSignalValue {
        clone_erased_output::<T>(value)
    }

    fn accept_output(
        &self,
        world: &mut World,
        entity: SignalEntity,
        output: ErasedSignalValue,
    ) -> Option<AcceptedSignalOutput> {
        let output = *output
            .downcast::<T>()
            .unwrap_or_else(|_| panic!("managed scalar replay entity {entity:?} received the wrong output type"));
        let update = world
            .get_mut::<ScalarReplayCache<T>>(*entity)
            .unwrap_or_else(|| panic!("managed scalar replay entity {entity:?} lost its cache"))
            .accept(output)
            .unwrap_or_else(|_| panic!("managed scalar replay revision overflowed"));
        let ReplayCacheUpdate::Accepted { output, revision } = update else {
            unreachable!("scalar replay accepts every output");
        };
        Some(AcceptedSignalOutput {
            output: Box::new(output),
            replay_revision: Some(revision),
        })
    }
}

struct VecReplayRuntimeContract<T>(PhantomData<fn() -> T>);

impl<T: Clone + Send + Sync + 'static> ReplayRuntimeContract for VecReplayRuntimeContract<T> {
    fn cache_type(&self) -> TypeId {
        TypeId::of::<VecReplayCache<T>>()
    }

    fn event_type(&self) -> TypeId {
        TypeId::of::<Vec<VecDiff<T>>>()
    }

    fn coalesces_pending_snapshot(&self) -> bool {
        true
    }

    fn readiness_and_event_type(&self, world: &World, entity: SignalEntity) -> Option<(ReplayCacheReadiness, TypeId)> {
        Some((
            world.get::<VecReplayCache<T>>(*entity)?.readiness(),
            TypeId::of::<Vec<VecDiff<T>>>(),
        ))
    }

    fn inspect(&self, world: &World, entity: SignalEntity) -> Option<ErasedReplaySnapshot> {
        match world.get::<VecReplayCache<T>>(*entity)?.snapshot() {
            ReplayCacheSnapshot::Warming => Some(ErasedReplaySnapshot::Warming),
            ReplayCacheSnapshot::Ready { current, revision } => Some(ErasedReplaySnapshot::Ready {
                snapshot: Box::new(Vec::from([VecDiff::Replace {
                    values: current.clone(),
                }])),
                revision,
            }),
        }
    }

    fn clone_event(&self, value: &dyn Any) -> ErasedSignalValue {
        clone_erased_output::<Vec<VecDiff<T>>>(value)
    }

    fn accept_output(
        &self,
        world: &mut World,
        entity: SignalEntity,
        output: ErasedSignalValue,
    ) -> Option<AcceptedSignalOutput> {
        let output = *output
            .downcast::<Vec<VecDiff<T>>>()
            .unwrap_or_else(|_| panic!("managed vector replay entity {entity:?} received the wrong output type"));
        let update = world
            .get_mut::<VecReplayCache<T>>(*entity)
            .unwrap_or_else(|| panic!("managed vector replay entity {entity:?} lost its cache"))
            .accept(output)
            .unwrap_or_else(|_| panic!("managed vector replay revision overflowed"));
        match update {
            ReplayCacheUpdate::Suppressed => None,
            ReplayCacheUpdate::Accepted { output, revision } => Some(AcceptedSignalOutput {
                output: Box::new(output),
                replay_revision: Some(revision),
            }),
        }
    }
}

struct MapReplayRuntimeContract<K, V>(PhantomData<fn() -> (K, V)>);

impl<K, V> ReplayRuntimeContract for MapReplayRuntimeContract<K, V>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn cache_type(&self) -> TypeId {
        TypeId::of::<MapReplayCache<K, V>>()
    }

    fn event_type(&self) -> TypeId {
        TypeId::of::<Vec<MapDiff<K, V>>>()
    }

    fn coalesces_pending_snapshot(&self) -> bool {
        true
    }

    fn readiness_and_event_type(&self, world: &World, entity: SignalEntity) -> Option<(ReplayCacheReadiness, TypeId)> {
        Some((
            world.get::<MapReplayCache<K, V>>(*entity)?.readiness(),
            TypeId::of::<Vec<MapDiff<K, V>>>(),
        ))
    }

    fn inspect(&self, world: &World, entity: SignalEntity) -> Option<ErasedReplaySnapshot> {
        match world.get::<MapReplayCache<K, V>>(*entity)?.snapshot() {
            ReplayCacheSnapshot::Warming => Some(ErasedReplaySnapshot::Warming),
            ReplayCacheSnapshot::Ready { current, revision } => Some(ErasedReplaySnapshot::Ready {
                snapshot: Box::new(Vec::from([MapDiff::Replace {
                    entries: current
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                }])),
                revision,
            }),
        }
    }

    fn clone_event(&self, value: &dyn Any) -> ErasedSignalValue {
        clone_erased_output::<Vec<MapDiff<K, V>>>(value)
    }

    fn accept_output(
        &self,
        world: &mut World,
        entity: SignalEntity,
        output: ErasedSignalValue,
    ) -> Option<AcceptedSignalOutput> {
        let output = *output
            .downcast::<Vec<MapDiff<K, V>>>()
            .unwrap_or_else(|_| panic!("managed map replay entity {entity:?} received the wrong output type"));
        let update = world
            .get_mut::<MapReplayCache<K, V>>(*entity)
            .unwrap_or_else(|| panic!("managed map replay entity {entity:?} lost its cache"))
            .accept(output)
            .unwrap_or_else(|_| panic!("managed map replay revision overflowed"));
        match update {
            ReplayCacheUpdate::Suppressed => None,
            ReplayCacheUpdate::Accepted { output, revision } => Some(AcceptedSignalOutput {
                output: Box::new(output),
                replay_revision: Some(revision),
            }),
        }
    }
}

pub(crate) fn scalar_replay_runtime_contract<T: Clone + Send + Sync + 'static>() -> Arc<dyn ReplayRuntimeContract> {
    Arc::new(ScalarReplayRuntimeContract::<T>(PhantomData))
}

pub(crate) fn vec_replay_runtime_contract<T: Clone + Send + Sync + 'static>() -> Arc<dyn ReplayRuntimeContract> {
    Arc::new(VecReplayRuntimeContract::<T>(PhantomData))
}

pub(crate) fn map_replay_runtime_contract<K, V>() -> Arc<dyn ReplayRuntimeContract>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    Arc::new(MapReplayRuntimeContract::<K, V>(PhantomData))
}

pub(crate) enum PlannedReplayState {
    NotReplayable,
    Replayable {
        key: ReplayRuntimeKey,
        contract: Arc<dyn ReplayRuntimeContract>,
        initial_inspection: PlannedReplayInspection,
    },
}

impl PlannedReplayState {
    fn is_replayable(&self) -> bool {
        !matches!(self, Self::NotReplayable)
    }

    fn key(&self) -> Option<ReplayRuntimeKey> {
        match self {
            Self::NotReplayable => None,
            Self::Replayable { key, .. } => Some(*key),
        }
    }

    fn contract(&self) -> Option<&Arc<dyn ReplayRuntimeContract>> {
        match self {
            Self::NotReplayable => None,
            Self::Replayable { contract, .. } => Some(contract),
        }
    }

    fn preflight(&self) -> Option<RegistrationPlannedReplayPreflight> {
        match self {
            Self::NotReplayable => None,
            Self::Replayable {
                key,
                contract,
                initial_inspection,
            } => Some(RegistrationPlannedReplayPreflight {
                key: *key,
                contract: contract.clone(),
                inspection: initial_inspection.clone(),
            }),
        }
    }
}

pub(crate) struct RegistrationPlannedReplayPreflight {
    pub(crate) key: ReplayRuntimeKey,
    pub(crate) contract: Arc<dyn ReplayRuntimeContract>,
    pub(crate) inspection: PlannedReplayInspection,
}

pub(crate) struct AcceptedSignalOutput {
    pub(crate) output: ErasedSignalValue,
    pub(crate) replay_revision: Option<ReplayRevision>,
}

pub(crate) type CollectionPollAppend = fn(&mut ErasedSignalValue, ErasedSignalValue);

trait SignalOutputExecution: SignalKind {
    fn collection_poll_append() -> Option<CollectionPollAppend>;
}

impl<T: 'static> SignalOutputExecution for ScalarSignalKind<T> {
    fn collection_poll_append() -> Option<CollectionPollAppend> {
        None
    }
}

impl<T: Send + Sync + 'static> SignalOutputExecution for VecSignalKind<T> {
    fn collection_poll_append() -> Option<CollectionPollAppend> {
        Some(append_collection_poll_batch::<VecDiff<T>>)
    }
}

impl<K: Send + Sync + 'static, V: Send + Sync + 'static> SignalOutputExecution for MapSignalKind<K, V> {
    fn collection_poll_append() -> Option<CollectionPollAppend> {
        Some(append_collection_poll_batch::<MapDiff<K, V>>)
    }
}

fn append_collection_poll_batch<T: 'static>(history: &mut ErasedSignalValue, batch: ErasedSignalValue) {
    let history = history
        .downcast_mut::<Vec<T>>()
        .expect("collection poll history must match the registered signal kind");
    let batch = batch
        .downcast::<Vec<T>>()
        .expect("collection poll batch must match the registered signal kind");
    history.extend(*batch);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderedInputConsumption {
    ManagerBridge,
    Chain { input_type: TypeId, output_type: TypeId },
}

impl OrderedInputConsumption {
    fn chain<T: 'static>() -> Self {
        Self::Chain {
            input_type: TypeId::of::<ChainInput<T>>(),
            output_type: TypeId::of::<Vec<VecDiff<T>>>(),
        }
    }

    pub(crate) fn accepts_shape(
        self,
        dependency_count: usize,
        input_type: TypeId,
        output_type: TypeId,
        replayable: bool,
    ) -> bool {
        if replayable {
            return false;
        }
        match self {
            Self::ManagerBridge => dependency_count == 1 && output_type == TypeId::of::<()>(),
            Self::Chain {
                input_type: expected_input,
                output_type: expected_output,
            } => dependency_count == 2 && input_type == expected_input && output_type == expected_output,
        }
    }

    pub(crate) fn accepts_dependency(
        self,
        output_type: TypeId,
        replayable: bool,
        ordered_input_consumption: Option<Self>,
    ) -> bool {
        match self {
            Self::ManagerBridge => true,
            Self::Chain { input_type, .. } => {
                output_type == input_type && !replayable && ordered_input_consumption.is_none()
            }
        }
    }
}

/// A type-erased runtime adapter installed by the graph after a pending system is registered.
#[derive(Clone)]
pub(crate) struct RegistrationSignalRuntime {
    runner: Arc<RegistrationRunner>,
    input_type: TypeId,
    output_type: TypeId,
    output_delivery: StoredOutputDelivery,
    collection_poll_append: Option<CollectionPollAppend>,
    ordered_input_consumption: Option<OrderedInputConsumption>,
    replay_key: Option<ReplayRuntimeKey>,
    replay_contract: Option<Arc<dyn ReplayRuntimeContract>>,
}

pub(crate) type RegistrationRunner = dyn Fn(&mut World, ErasedSignalValue) -> Result<Option<ErasedSignalValue>, RegistrationSystemRunError>
    + Send
    + Sync;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegistrationSystemRunError {
    InputTypeMismatch { expected: &'static str },
    SystemUnavailable,
}

impl RegistrationSignalRuntime {
    pub(crate) fn runner(&self) -> &Arc<RegistrationRunner> {
        &self.runner
    }

    pub(crate) fn input_type(&self) -> TypeId {
        self.input_type
    }

    pub(crate) fn output_type(&self) -> TypeId {
        self.output_type
    }

    pub(crate) fn output_delivery(&self) -> StoredOutputDelivery {
        self.output_delivery
    }

    pub(crate) fn collection_poll_append(&self) -> Option<CollectionPollAppend> {
        self.collection_poll_append
    }

    pub(crate) fn ordered_input_consumption(&self) -> Option<OrderedInputConsumption> {
        self.ordered_input_consumption
    }

    pub(crate) fn replay_key(&self) -> Option<ReplayRuntimeKey> {
        self.replay_key
    }

    pub(crate) fn replay_contract(&self) -> Option<&Arc<dyn ReplayRuntimeContract>> {
        self.replay_contract.as_ref()
    }
}

type PendingEntityInitializer = Box<dyn FnOnce(&mut World) -> SignalEntity + Send + Sync + 'static>;
type PendingEntityInstaller = Box<dyn FnOnce(&mut World, SignalEntity, &[SignalEntity]) + Send + Sync + 'static>;
pub(crate) type ReplayInstaller = Box<dyn FnOnce(&mut World, SignalEntity) + Send + Sync + 'static>;

struct PendingSignalEntity {
    reference: PlannedSignalEntityRef,
    dependencies: Box<[PlannedSignalEntityRef]>,
    initializer: Option<PendingEntityInitializer>,
    entity_installer: Option<PendingEntityInstaller>,
    runtime: RegistrationSignalRuntime,
    replay: PlannedReplayState,
    replay_installer: Option<ReplayInstaller>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExistingSignalEntityRegistration {
    pub(crate) ordered_input_consumption: Option<OrderedInputConsumption>,
    pub(crate) dependencies: Box<[SignalEntity]>,
    pub(crate) replayable: bool,
    pub(crate) input_type: TypeId,
    pub(crate) output_type: TypeId,
    pub(crate) output_delivery: Option<OutputDeliveryKey>,
}

pub(crate) struct RegistrationEntityPreflight {
    pub(crate) ordered_input_consumption: Option<OrderedInputConsumption>,
    pub(crate) entity: PlannedSignalEntityRef,
    pub(crate) dependencies: Box<[PlannedSignalEntityRef]>,
    pub(crate) input_type: TypeId,
    pub(crate) output_type: TypeId,
    pub(crate) pending_output_delivery: Option<StoredOutputDelivery>,
    pub(crate) expected_output_delivery: Option<OutputDeliveryKey>,
    pub(crate) replayable: bool,
    pub(crate) planned_replay: Option<RegistrationPlannedReplayPreflight>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistrationSchedulePreflight {
    pub(crate) entity: PlannedSignalEntityRef,
    pub(crate) tag: Option<InternedScheduleLabel>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistrationClaimDelta {
    pub(crate) entity: PlannedSignalEntityRef,
    pub(crate) increment: usize,
}

pub(crate) enum RegistrationRouteDisposition {
    OrderedInput {
        bootstrap: Option<ErasedSignalValue>,
    },
    Ordinary,
    ReplayPending,
    ReplayReady {
        snapshot: ErasedSignalValue,
        revision: ReplayRevision,
        event_type: TypeId,
    },
}

pub(crate) struct RegistrationActiveRoutePreflight {
    pub(crate) dependency: PlannedSignalEntityRef,
    pub(crate) dependent: PlannedSignalEntityRef,
    pub(crate) disposition: RegistrationRouteDisposition,
}

pub(crate) struct RegistrationActivationPreflight {
    pub(crate) count_increments: Box<[RegistrationClaimDelta]>,
    pub(crate) active_routes: Box<[RegistrationActiveRoutePreflight]>,
    pub(crate) effective_membership: Box<[RegistrationEffectiveMembershipPreflight]>,
    pub(crate) activated: Box<[PlannedSignalEntityRef]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegistrationEffectiveMembershipPreflight {
    pub(crate) entity: PlannedSignalEntityRef,
    pub(crate) schedule: InternedScheduleLabel,
    pub(crate) level: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegistrationDynamicConnectionPreflight {
    pub(crate) source: PlannedSignalEntityRef,
    pub(crate) target: PlannedSignalEntityRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegistrationDynamicConnectionCommit {
    pub(crate) source: SignalEntity,
    pub(crate) target: SignalEntity,
}

pub(crate) struct RegistrationPreflight {
    pub(crate) entities: Box<[RegistrationEntityPreflight]>,
    pub(crate) schedules: Box<[RegistrationSchedulePreflight]>,
    pub(crate) claim_deltas: Box<[RegistrationClaimDelta]>,
    pub(crate) dynamic_connections: Box<[RegistrationDynamicConnectionPreflight]>,
}

pub(crate) struct RegistrationEntityCommit {
    pub(crate) entity: SignalEntity,
    pub(crate) dependencies: Box<[SignalEntity]>,
    pub(crate) runtime: RegistrationSignalRuntime,
}

pub(crate) struct RegistrationReplayCommit {
    pub(crate) entity: SignalEntity,
    pub(crate) replay: PlannedReplayState,
    pub(crate) installer: Option<ReplayInstaller>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistrationScheduleCommit {
    pub(crate) entity: SignalEntity,
    pub(crate) tag: Option<InternedScheduleLabel>,
    pub(crate) hint: Option<InternedScheduleLabel>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistrationMaterializedDependencyCommit {
    pub(crate) dependency: SignalEntity,
    pub(crate) dependent: SignalEntity,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistrationClaimCommit {
    pub(crate) entity: SignalEntity,
    pub(crate) increment: usize,
}

pub(crate) struct RegistrationActiveRouteCommit {
    pub(crate) dependency: SignalEntity,
    pub(crate) dependent: SignalEntity,
    pub(crate) disposition: RegistrationRouteDisposition,
}

pub(crate) struct RegistrationActivationCommit {
    pub(crate) count_increments: Box<[RegistrationClaimCommit]>,
    pub(crate) active_routes: Box<[RegistrationActiveRouteCommit]>,
    pub(crate) effective_membership: Box<[RegistrationEffectiveMembershipCommit]>,
    pub(crate) activated: Box<[SignalEntity]>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistrationEffectiveMembershipCommit {
    pub(crate) entity: SignalEntity,
    pub(crate) schedule: InternedScheduleLabel,
    pub(crate) level: u32,
}

pub(crate) struct RegistrationTerminalRetainerCommit {
    pub(crate) terminal: SignalEntity,
    pub(crate) retainer: SignalTerminalRetainer,
}

/// The graph-side interface required by this module.
///
/// `registration_preflight_*` methods must be read-only. Every `registration_commit_*` method is
/// called only after all local and graph preflight methods have succeeded; commit methods therefore
/// have no ordinary structured failure path.
pub(crate) trait SignalRegistrationGraphHooks {
    fn registration_world(&mut self) -> &mut World;

    fn registration_existing_signal(&self, entity: SignalEntity) -> Option<ExistingSignalEntityRegistration>;

    fn registration_preflight_existing_runtimes(
        &self,
        entities: &[SignalEntity],
    ) -> Result<(), SignalRegistrationError>;

    fn registration_schedule_is_registered(&self, schedule: InternedScheduleLabel) -> bool;

    fn registration_existing_schedule(
        &self,
        entity: SignalEntity,
    ) -> Result<(Option<InternedScheduleLabel>, Option<InternedScheduleLabel>), SignalRegistrationError>;

    fn registration_claim_count(&self, entity: SignalEntity) -> Option<usize>;

    fn registration_replay_maintenance_armed(&self, entity: SignalEntity) -> Option<bool>;

    fn registration_preflight_canonical_state(
        &self,
        entity: &RegistrationEntityPreflight,
    ) -> Result<(), SignalRegistrationError>;

    fn registration_preflight_reserved_output_policy(
        &self,
        entity: &RegistrationEntityPreflight,
    ) -> Result<(), SignalRegistrationError>;

    fn registration_preflight_dynamic_connections(
        &self,
        plan: &RegistrationPreflight,
    ) -> Result<(), SignalRegistrationError>;

    fn registration_preflight_activation(
        &self,
        plan: &RegistrationPreflight,
    ) -> Result<RegistrationActivationPreflight, SignalRegistrationError>;

    fn registration_commit_signal_entity(&mut self, entity: RegistrationEntityCommit);

    fn registration_commit_replay(&mut self, replay: RegistrationReplayCommit);

    fn registration_commit_materialized_dependency(&mut self, link: RegistrationMaterializedDependencyCommit);

    fn registration_commit_schedule(&mut self, schedule: RegistrationScheduleCommit);

    fn registration_commit_activation(&mut self, activation: RegistrationActivationCommit);

    fn registration_commit_dynamic_connection(
        &mut self,
        connection: RegistrationDynamicConnectionCommit,
    ) -> DynamicSignalConnection;

    fn registration_commit_replay_maintenance(&mut self, terminal: SignalEntity);

    fn registration_commit_terminal_retainer(&mut self, retainer: RegistrationTerminalRetainerCommit);

    fn registration_publish_initial_claim(&mut self, terminal: SignalEntity) -> SignalClaim;
}

#[derive(Clone, Copy, Default)]
struct PlannedScheduleState {
    tag: Option<InternedScheduleLabel>,
    hint: Option<InternedScheduleLabel>,
}

#[derive(Clone, Copy)]
struct PlannedClaimDelta {
    entity: PlannedSignalEntityRef,
    increment: usize,
}

struct MaterializerPublication {
    state: Arc<SignalMaterializerState>,
    terminal: PlannedSignalEntityRef,
}

struct MaterializerMaintenanceCandidate {
    state: Arc<SignalMaterializerState>,
    terminal: PlannedSignalEntityRef,
}

#[derive(Clone)]
struct ExistingCanonicalExpectation {
    entity: SignalEntity,
    dependencies: Option<Box<[SignalEntity]>>,
    replayable: Option<bool>,
    output_delivery: Option<OutputDeliveryKey>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlanOutcome {
    Active,
    Rejected,
    Committed,
}

struct PendingSignalMaterialization<'world> {
    graph: &'world mut dyn SignalRegistrationGraphHooks,
    plan_id: MaterializationPlanId,
    next_pending_index: usize,
    entities: Vec<PendingSignalEntity>,

    schedule_projection: BTreeMap<PlannedSignalEntityRef, PlannedScheduleState>,
    claim_deltas: Vec<PlannedClaimDelta>,
    dynamic_connections: Vec<RegistrationDynamicConnectionPreflight>,
    existing_expectations: Vec<ExistingCanonicalExpectation>,
    publications: Vec<MaterializerPublication>,
    maintenance_candidates: Vec<MaterializerMaintenanceCandidate>,
    local_candidates: Vec<ReleasedSignalCandidate>,
    retained_entities: Vec<SignalEntity>,
    uncommitted_entities: VecDeque<SignalEntity>,
    committed_pending_entities: Vec<SignalEntity>,

    maintenance_claims: BTreeSet<PlannedSignalEntityRef>,
    private_claims: Vec<PlannedSignalEntityRef>,
    activation_projection: Option<RegistrationActivationPreflight>,
    outcome: PlanOutcome,
}

impl<'world> PendingSignalMaterialization<'world> {
    fn new(graph: &'world mut dyn SignalRegistrationGraphHooks) -> Self {
        Self {
            graph,
            plan_id: MaterializationPlanId::next(),
            next_pending_index: 0,
            entities: Vec::new(),

            schedule_projection: BTreeMap::new(),
            claim_deltas: Vec::new(),
            dynamic_connections: Vec::new(),
            existing_expectations: Vec::new(),
            publications: Vec::new(),
            maintenance_candidates: Vec::new(),
            local_candidates: Vec::new(),
            retained_entities: Vec::new(),
            uncommitted_entities: VecDeque::new(),
            committed_pending_entities: Vec::new(),

            maintenance_claims: BTreeSet::new(),
            private_claims: Vec::new(),
            activation_projection: None,
            outcome: PlanOutcome::Active,
        }
    }

    fn id(&self) -> MaterializationPlanId {
        self.plan_id
    }

    fn allocate_pending_reference(&mut self) -> PlannedSignalEntityRef {
        let index = self.next_pending_index;
        self.next_pending_index = self
            .next_pending_index
            .checked_add(1)
            .unwrap_or_else(|| panic!("signal materialization pending-reference identity overflowed"));
        PlannedSignalEntityRef::Pending {
            plan_id: self.plan_id,
            index,
        }
    }

    fn pending_reference_exists(&self, reference: PlannedSignalEntityRef) -> bool {
        matches!(reference, PlannedSignalEntityRef::Pending { plan_id, .. } if plan_id == self.plan_id)
            && self.entities.iter().any(|entity| entity.reference == reference)
    }

    fn reject(&mut self, error: SignalRegistrationError) {
        if self.outcome != PlanOutcome::Active {
            return;
        }
        self.outcome = PlanOutcome::Rejected;
        for publication in &self.publications {
            let mut state = lock_materializer_state(&publication.state);
            if matches!(
                &*state,
                SharedMaterializerState::Initializing { plan_id, .. } if *plan_id == self.id()
            ) {
                *state = SharedMaterializerState::Rejected(error);
            }
        }
        self.process_local_candidates();
    }

    fn process_local_candidates(&mut self) {
        let candidates = core::mem::take(&mut self.local_candidates);
        if !candidates.is_empty() {
            cleanup::process_signal_candidates(self.graph.registration_world(), candidates);
        }
    }

    fn preflight(
        &mut self,
        terminal: PlannedSignalEntityRef,
        retain_activation_projection: bool,
    ) -> Result<RegistrationPreflight, SignalRegistrationError> {
        let existing = self.collect_existing_entities(terminal)?;
        self.preflight_cycles(&existing)?;

        let mut entities = Vec::with_capacity(self.entities.len() + self.existing_expectations.len());
        for pending in &self.entities {
            let output_delivery = pending.runtime.output_delivery();
            let entity = RegistrationEntityPreflight {
                entity: pending.reference,
                dependencies: pending.dependencies.clone(),
                ordered_input_consumption: pending.runtime.ordered_input_consumption(),
                input_type: pending.runtime.input_type(),
                output_type: pending.runtime.output_type(),
                pending_output_delivery: Some(output_delivery),
                expected_output_delivery: Some(output_delivery.key()),
                replayable: pending.replay.is_replayable(),
                planned_replay: pending.replay.preflight(),
            };
            self.graph.registration_preflight_canonical_state(&entity)?;
            self.graph.registration_preflight_reserved_output_policy(&entity)?;
            entities.push(entity);
        }

        for expectation in &self.existing_expectations {
            let reference = PlannedSignalEntityRef::Existing(expectation.entity);
            let existing = existing
                .get(&reference)
                .ok_or(SignalRegistrationError::EntityUnavailable)?;

            let dependencies = expectation
                .dependencies
                .as_deref()
                .unwrap_or(existing.dependencies.as_ref())
                .iter()
                .copied()
                .map(PlannedSignalEntityRef::Existing)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let entity = RegistrationEntityPreflight {
                entity: reference,
                dependencies,
                ordered_input_consumption: existing.ordered_input_consumption,
                input_type: existing.input_type,
                output_type: existing.output_type,
                pending_output_delivery: None,
                expected_output_delivery: expectation.output_delivery,
                replayable: expectation.replayable.unwrap_or(existing.replayable),
                planned_replay: None,
            };
            self.graph.registration_preflight_canonical_state(&entity)?;
            self.graph.registration_preflight_reserved_output_policy(&entity)?;
            entities.push(entity);
        }

        for entity in &entities {
            let Some(consumption) = entity.ordered_input_consumption else {
                continue;
            };
            for dependency in &entity.dependencies {
                let (output_type, replayable, ordered_input_consumption) = match dependency {
                    PlannedSignalEntityRef::Pending { .. } => {
                        let dependency = entities
                            .iter()
                            .find(|entity| entity.entity == *dependency)
                            .ok_or(SignalRegistrationError::EntityUnavailable)?;
                        (
                            dependency.output_type,
                            dependency.replayable,
                            dependency.ordered_input_consumption,
                        )
                    }
                    PlannedSignalEntityRef::Existing(_) => {
                        let dependency = existing
                            .get(dependency)
                            .ok_or(SignalRegistrationError::EntityUnavailable)?;
                        (
                            dependency.output_type,
                            dependency.replayable,
                            dependency.ordered_input_consumption,
                        )
                    }
                };
                if !consumption.accepts_dependency(output_type, replayable, ordered_input_consumption) {
                    return Err(SignalRegistrationError::ConflictingCanonicalState);
                }
            }
        }

        let mut schedules = Vec::with_capacity(self.schedule_projection.len());
        for (entity, state) in &self.schedule_projection {
            for schedule in [state.tag, state.hint].into_iter().flatten() {
                if !self.graph.registration_schedule_is_registered(schedule) {
                    return Err(SignalRegistrationError::ScheduleNotRegistered);
                }
            }
            schedules.push(RegistrationSchedulePreflight {
                entity: *entity,
                tag: state.tag,
            });
        }

        let claim_deltas = self.preflight_claim_counts()?;
        let preflight = RegistrationPreflight {
            entities: entities.into_boxed_slice(),
            schedules: schedules.into_boxed_slice(),
            claim_deltas,
            dynamic_connections: self.dynamic_connections.clone().into_boxed_slice(),
        };
        self.graph.registration_preflight_dynamic_connections(&preflight)?;
        let activation = self.graph.registration_preflight_activation(&preflight)?;
        let existing_entities = existing
            .keys()
            .filter_map(|reference| match reference {
                PlannedSignalEntityRef::Existing(entity) => Some(*entity),
                PlannedSignalEntityRef::Pending { .. } => None,
            })
            .collect::<Vec<_>>();
        self.graph
            .registration_preflight_existing_runtimes(&existing_entities)?;
        let known_references = self
            .plan_references(terminal)
            .into_iter()
            .chain(existing.keys().copied())
            .collect::<BTreeSet<_>>();
        for reference in Self::activation_references(&activation) {
            if let PlannedSignalEntityRef::Pending { .. } = reference
                && !self.pending_reference_exists(reference)
            {
                return Err(SignalRegistrationError::EntityUnavailable);
            }
            if !known_references.contains(&reference) {
                return Err(SignalRegistrationError::EntityUnavailable);
            }
        }
        if retain_activation_projection {
            self.activation_projection = Some(activation);
        }
        Ok(preflight)
    }

    fn plan_references(&self, terminal: PlannedSignalEntityRef) -> BTreeSet<PlannedSignalEntityRef> {
        let mut references = BTreeSet::from([terminal]);
        references.extend(self.entities.iter().map(|entity| entity.reference));

        references.extend(
            self.entities
                .iter()
                .flat_map(|entity| entity.dependencies.iter().copied()),
        );
        for expectation in &self.existing_expectations {
            references.insert(PlannedSignalEntityRef::Existing(expectation.entity));
            references.extend(
                expectation
                    .dependencies
                    .iter()
                    .flatten()
                    .copied()
                    .map(PlannedSignalEntityRef::Existing),
            );
        }
        references.extend(self.schedule_projection.keys().copied());
        references.extend(self.claim_deltas.iter().map(|delta| delta.entity));
        references.extend(
            self.dynamic_connections
                .iter()
                .flat_map(|connection| [connection.source, connection.target]),
        );
        references.extend(self.publications.iter().map(|publication| publication.terminal));
        references.extend(self.maintenance_candidates.iter().map(|candidate| candidate.terminal));
        references.extend(self.maintenance_claims.iter().copied());
        references.extend(self.private_claims.iter().copied());
        references
    }

    fn activation_references(activation: &RegistrationActivationPreflight) -> BTreeSet<PlannedSignalEntityRef> {
        let mut references = BTreeSet::new();
        references.extend(activation.count_increments.iter().map(|delta| delta.entity));
        references.extend(
            activation
                .active_routes
                .iter()
                .flat_map(|route| [route.dependency, route.dependent]),
        );
        references.extend(
            activation
                .effective_membership
                .iter()
                .map(|membership| membership.entity),
        );
        references.extend(activation.activated.iter().copied());
        references
    }

    fn collect_existing_entities(
        &self,
        terminal: PlannedSignalEntityRef,
    ) -> Result<BTreeMap<PlannedSignalEntityRef, ExistingSignalEntityRegistration>, SignalRegistrationError> {
        let mut existing = BTreeMap::new();
        let mut pending = self.plan_references(terminal);
        while let Some(reference) = pending.iter().next().copied() {
            pending.remove(&reference);
            match reference {
                PlannedSignalEntityRef::Pending { .. } => {
                    if !self.pending_reference_exists(reference) {
                        return Err(SignalRegistrationError::EntityUnavailable);
                    }
                }
                PlannedSignalEntityRef::Existing(entity) => {
                    if existing.contains_key(&reference) {
                        continue;
                    }
                    let state = self
                        .graph
                        .registration_existing_signal(entity)
                        .ok_or(SignalRegistrationError::EntityUnavailable)?;
                    pending.extend(state.dependencies.iter().copied().map(PlannedSignalEntityRef::Existing));
                    existing.insert(reference, state);
                }
            }
        }
        Ok(existing)
    }

    fn dependencies_for<'plan>(
        &'plan self,
        entity: PlannedSignalEntityRef,
        existing: &'plan BTreeMap<PlannedSignalEntityRef, ExistingSignalEntityRegistration>,
    ) -> Result<Vec<PlannedSignalEntityRef>, SignalRegistrationError> {
        match entity {
            PlannedSignalEntityRef::Existing(_) => existing
                .get(&entity)
                .map(|state| {
                    state
                        .dependencies
                        .iter()
                        .copied()
                        .map(PlannedSignalEntityRef::Existing)
                        .collect()
                })
                .ok_or(SignalRegistrationError::EntityUnavailable),
            reference @ PlannedSignalEntityRef::Pending { .. } => self
                .entities
                .iter()
                .find(|entity| entity.reference == reference)
                .map(|state| state.dependencies.to_vec())
                .ok_or(SignalRegistrationError::EntityUnavailable),
        }
    }

    fn preflight_cycles(
        &self,
        existing: &BTreeMap<PlannedSignalEntityRef, ExistingSignalEntityRegistration>,
    ) -> Result<(), SignalRegistrationError> {
        let mut roots = self.entities.iter().map(|entity| entity.reference).collect::<Vec<_>>();

        roots.extend(existing.keys().copied());
        roots.sort_unstable();
        roots.dedup();

        let mut colors = BTreeMap::<PlannedSignalEntityRef, u8>::new();
        for root in roots {
            if colors.get(&root) == Some(&2) {
                continue;
            }
            let mut stack = Vec::from([(root, false)]);
            while let Some((entity, exiting)) = stack.pop() {
                if exiting {
                    colors.insert(entity, 2);
                    continue;
                }
                match colors.get(&entity).copied() {
                    Some(1) => return Err(SignalRegistrationError::CycleDetected),
                    Some(2) => continue,
                    _ => {}
                }
                colors.insert(entity, 1);
                stack.push((entity, true));
                let mut dependencies = self.dependencies_for(entity, existing)?;
                dependencies.sort_unstable();
                dependencies.dedup();
                for dependency in dependencies.into_iter().rev() {
                    if colors.get(&dependency) == Some(&1) {
                        return Err(SignalRegistrationError::CycleDetected);
                    }
                    if colors.get(&dependency) != Some(&2) {
                        stack.push((dependency, false));
                    }
                }
            }
        }
        Ok(())
    }

    fn plan_replay_maintenance(&mut self) -> Result<bool, SignalRegistrationError> {
        let mut maintenance_claims = BTreeSet::new();
        let retained_terminals = self
            .maintenance_candidates
            .iter()
            .filter_map(|candidate| {
                (candidate.state.live_chains.load(AtomicOrdering::Acquire) != 0).then_some(candidate.terminal)
            })
            .collect::<Vec<_>>();
        for terminal in retained_terminals {
            let (replayable, maintenance_armed) = match terminal {
                reference @ PlannedSignalEntityRef::Pending { .. } => self
                    .entities
                    .iter()
                    .find(|entity| entity.reference == reference)
                    .map(|pending| (pending.replay.is_replayable(), false))
                    .ok_or(SignalRegistrationError::EntityUnavailable)?,
                PlannedSignalEntityRef::Existing(entity) => {
                    let existing = self
                        .graph
                        .registration_existing_signal(entity)
                        .ok_or(SignalRegistrationError::EntityUnavailable)?;
                    let maintenance_armed = self
                        .graph
                        .registration_replay_maintenance_armed(entity)
                        .ok_or(SignalRegistrationError::EntityUnavailable)?;
                    (existing.replayable, maintenance_armed)
                }
            };
            if replayable && !maintenance_armed {
                maintenance_claims.insert(terminal);
            }
        }
        let changed = maintenance_claims != self.maintenance_claims;
        self.maintenance_claims = maintenance_claims;
        Ok(changed)
    }

    fn preflight_claim_counts(&self) -> Result<Box<[RegistrationClaimDelta]>, SignalRegistrationError> {
        let mut combined = BTreeMap::<PlannedSignalEntityRef, usize>::new();
        for delta in self.claim_deltas.iter().copied().chain(
            self.maintenance_claims
                .iter()
                .map(|&entity| PlannedClaimDelta { entity, increment: 1 }),
        ) {
            let current = combined.get(&delta.entity).copied().unwrap_or(0);
            let increment = current
                .checked_add(delta.increment)
                .ok_or(SignalRegistrationError::ClaimCountOverflow)?;
            combined.insert(delta.entity, increment);
        }

        let mut result = Vec::with_capacity(combined.len());
        for (entity, increment) in combined {
            if let PlannedSignalEntityRef::Existing(existing) = entity {
                let count = self
                    .graph
                    .registration_claim_count(existing)
                    .ok_or(SignalRegistrationError::EntityUnavailable)?;
                count
                    .checked_add(increment)
                    .ok_or(SignalRegistrationError::ClaimCountOverflow)?;
            }
            result.push(RegistrationClaimDelta { entity, increment });
        }
        Ok(result.into_boxed_slice())
    }

    fn resolve(
        plan_id: MaterializationPlanId,
        reference: PlannedSignalEntityRef,
        pending: &BTreeMap<PlannedSignalEntityRef, SignalEntity>,
    ) -> SignalEntity {
        match reference {
            PlannedSignalEntityRef::Existing(entity) => entity,
            reference @ PlannedSignalEntityRef::Pending {
                plan_id: reference_plan_id,
                ..
            } if reference_plan_id == plan_id => pending.get(&reference).copied().unwrap_or_else(|| {
                panic!("validated pending signal reference {reference:?} disappeared during commit")
            }),
            PlannedSignalEntityRef::Pending { .. } => {
                panic!("a pending signal reference from another materialization reached commit")
            }
        }
    }

    fn commit(
        mut self,
        terminal: PlannedSignalEntityRef,
        acquire_initial_claim: bool,
        dynamic_owner: Option<(crate::dynamic::DynamicOwnerEpoch, RegistrationPreflight)>,
    ) -> CommittedMaterialization {
        let allocating = !self.entities.is_empty();
        if dynamic_owner.is_some() && allocating {
            // Retaining each referenced root also retains its immutable static dependencies.
            // Recollecting and pinning the transitive closure adds no lifetime protection.
            self.retained_entities = self
                .plan_references(terminal)
                .into_iter()
                .filter_map(|reference| match reference {
                    PlannedSignalEntityRef::Existing(entity) => Some(entity),
                    PlannedSignalEntityRef::Pending { .. } => None,
                })
                .collect();
            crate::graph::retain_registration_entities(self.graph.registration_world(), &self.retained_entities);
        }
        let mut resolved_pending = BTreeMap::new();
        let mut identities = BTreeSet::new();
        for pending in &mut self.entities {
            let initializer = pending
                .initializer
                .take()
                .unwrap_or_else(|| panic!("validated pending signal initializer disappeared during commit"));
            let entity = initializer(self.graph.registration_world());
            self.uncommitted_entities.push_back(entity);
            if !identities.insert(entity.entity().to_bits()) {
                panic!("distinct pending signal entities resolved to the same Bevy entity");
            }
            resolved_pending.insert(pending.reference, entity);
        }

        let defer_cleanup = dynamic_owner.is_some();
        let mut retired = false;
        if let Some((owner, mut preflight)) = dynamic_owner.filter(|_| allocating) {
            let activation_is_pending_only = self.activation_projection.as_ref().is_some_and(|activation| {
                activation
                    .count_increments
                    .iter()
                    .all(|claim| matches!(claim.entity, PlannedSignalEntityRef::Pending { .. }))
            }) && preflight
                .dynamic_connections
                .iter()
                .all(|connection| matches!(connection.target, PlannedSignalEntityRef::Existing(_)));
            self.graph.registration_world().flush();
            retired = !owner.is_current(self.graph.registration_world());
            if retired {
                // Invocation retirement is not rejection of independently retained materializers.
                // Finish their validated materialization, but never attach to the retired owner.
                self.dynamic_connections.clear();
                preflight.dynamic_connections = Box::default();
            }
            // Retirement may release an existing terminal's maintenance before a new retained
            // alias is published. Recompute missing maintenance separately from other claims so
            // an already-armed claim is never duplicated, including after reentrant registration.
            let maintenance_changed = self
                .plan_replay_maintenance()
                .expect("lifecycle callback invalidated validated replay maintenance topology");
            // A wholly pending activation has no live counts, replay cursors, or execution
            // dependencies to rebase. Connections to existing owners do not claim those owners.
            // Existing activation state still needs rebasing even when this owner did not retire.
            if maintenance_changed || !activation_is_pending_only {
                preflight.claim_deltas = self
                    .preflight_claim_counts()
                    .expect("lifecycle callback invalidated validated registration claim counts");
                self.activation_projection = Some(
                    self.graph
                        .registration_preflight_activation(&preflight)
                        .expect("lifecycle callback invalidated validated registration topology"),
                );
            }
        }

        let mut replay_commits = Vec::new();
        for mut pending in self.entities.drain(..) {
            let entity = resolved_pending
                .get(&pending.reference)
                .copied()
                .unwrap_or_else(|| panic!("validated pending signal entity disappeared during commit"));

            let mut dependencies: Vec<_> = pending
                .dependencies
                .iter()
                .copied()
                .map(|dependency| Self::resolve(self.plan_id, dependency, &resolved_pending))
                .collect();
            dependencies.sort_unstable_by_key(|dependency| dependency.entity().to_bits());
            dependencies.dedup_by_key(|dependency| dependency.entity().to_bits());
            self.graph.registration_commit_signal_entity(RegistrationEntityCommit {
                entity,
                dependencies: dependencies.clone().into_boxed_slice(),
                runtime: pending.runtime,
            });
            let popped = self.uncommitted_entities.pop_front();
            debug_assert_eq!(popped, Some(entity));
            self.committed_pending_entities.push(entity);
            if let Some(installer) = pending.entity_installer.take() {
                installer(self.graph.registration_world(), entity, &dependencies);
            }
            replay_commits.push(RegistrationReplayCommit {
                entity,
                replay: pending.replay,
                installer: pending.replay_installer,
            });
            for dependency in dependencies {
                self.graph
                    .registration_commit_materialized_dependency(RegistrationMaterializedDependencyCommit {
                        dependency,
                        dependent: entity,
                    });
            }
        }

        let activation = self
            .activation_projection
            .take()
            .unwrap_or_else(|| panic!("signal materialization commit began without an activation projection"));
        let activation = RegistrationActivationCommit {
            count_increments: activation
                .count_increments
                .iter()
                .map(|delta| RegistrationClaimCommit {
                    entity: Self::resolve(self.plan_id, delta.entity, &resolved_pending),
                    increment: delta.increment,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            active_routes: activation
                .active_routes
                .into_vec()
                .into_iter()
                .map(|route| RegistrationActiveRouteCommit {
                    dependency: Self::resolve(self.plan_id, route.dependency, &resolved_pending),
                    dependent: Self::resolve(self.plan_id, route.dependent, &resolved_pending),
                    disposition: route.disposition,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            effective_membership: activation
                .effective_membership
                .iter()
                .map(|membership| RegistrationEffectiveMembershipCommit {
                    entity: Self::resolve(self.plan_id, membership.entity, &resolved_pending),
                    schedule: membership.schedule,
                    level: membership.level,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            activated: activation
                .activated
                .iter()
                .map(|entity| Self::resolve(self.plan_id, *entity, &resolved_pending))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let terminal = Self::resolve(self.plan_id, terminal, &resolved_pending);
        for publication in &self.publications {
            let publication_terminal = Self::resolve(self.plan_id, publication.terminal, &resolved_pending);
            self.graph
                .registration_commit_terminal_retainer(RegistrationTerminalRetainerCommit {
                    terminal: publication_terminal,
                    retainer: SignalTerminalRetainer(Arc::downgrade(&publication.state)),
                });
        }

        for (entity, state) in &self.schedule_projection {
            self.graph.registration_commit_schedule(RegistrationScheduleCommit {
                entity: Self::resolve(self.plan_id, *entity, &resolved_pending),
                tag: state.tag,
                hint: state.hint,
            });
        }

        for replay in replay_commits {
            self.graph.registration_commit_replay(replay);
        }
        for terminal in &self.maintenance_claims {
            self.graph.registration_commit_replay_maintenance(Self::resolve(
                self.plan_id,
                *terminal,
                &resolved_pending,
            ));
        }
        self.graph.registration_commit_activation(activation);
        let dynamic_connections = self
            .dynamic_connections
            .drain(..)
            .map(|connection| {
                self.graph
                    .registration_commit_dynamic_connection(RegistrationDynamicConnectionCommit {
                        source: Self::resolve(self.plan_id, connection.source, &resolved_pending),
                        target: Self::resolve(self.plan_id, connection.target, &resolved_pending),
                    })
            })
            .collect();
        let initial_claim = acquire_initial_claim.then(|| self.graph.registration_publish_initial_claim(terminal));
        let private_claims = self
            .private_claims
            .iter()
            .map(|claim| {
                self.graph
                    .registration_publish_initial_claim(Self::resolve(self.plan_id, *claim, &resolved_pending))
            })
            .collect();

        for publication in &self.publications {
            let publication_terminal = Self::resolve(self.plan_id, publication.terminal, &resolved_pending);
            let output_delivery = self
                .graph
                .registration_existing_signal(publication_terminal)
                .and_then(|existing| existing.output_delivery)
                .unwrap_or_else(|| {
                    panic!("committed terminal {publication_terminal:?} was missing canonical output delivery policy")
                });
            let cleanup = CleanupSender::bound(self.graph.registration_world());
            let publish_candidate_locally = {
                let mut state = lock_materializer_state(&publication.state);
                let terminal_candidate_pending = match &*state {
                    SharedMaterializerState::Initializing {
                        plan_id,
                        terminal_candidate_pending,
                        ..
                    } if *plan_id == self.plan_id => *terminal_candidate_pending,
                    _ => panic!("materializer publication state changed before ready publication"),
                };
                let terminal_candidate_published =
                    terminal_candidate_pending || publication.state.live_chains.load(AtomicOrdering::Acquire) == 0;
                *state = SharedMaterializerState::Ready {
                    cache: SignalMaterializationCache {
                        terminal: publication_terminal,
                        cleanup,
                    },
                    output_delivery,
                    terminal_candidate_published,
                };
                terminal_candidate_published
            };
            if publish_candidate_locally {
                self.local_candidates
                    .push(ReleasedSignalCandidate::new(publication_terminal));
            }
        }
        self.outcome = PlanOutcome::Committed;
        self.committed_pending_entities.clear();
        // Raw registrar entities have no materializer whose final drop can nominate them for
        // cleanup.
        let materializer_terminals = self
            .publications
            .iter()
            .map(|publication| publication.terminal)
            .collect::<BTreeSet<_>>();
        for (reference, entity) in resolved_pending {
            if !materializer_terminals.contains(&reference) {
                self.local_candidates.push(ReleasedSignalCandidate::new(entity));
            }
        }
        let cleanup = RegistrationCleanup {
            candidates: core::mem::take(&mut self.local_candidates),
            retained_entities: core::mem::take(&mut self.retained_entities),
        };
        let cleanup = if defer_cleanup {
            cleanup
        } else {
            cleanup.finish(self.graph.registration_world());
            RegistrationCleanup::default()
        };
        CommittedMaterialization {
            #[cfg(test)]
            terminal,
            initial_claim,
            private_claims,
            dynamic_connections,
            retired,
            cleanup,
        }
    }
}

impl Drop for PendingSignalMaterialization<'_> {
    fn drop(&mut self) {
        if self.outcome != PlanOutcome::Active {
            return;
        }
        for entity in core::mem::take(&mut self.uncommitted_entities) {
            if let Ok(entity_mut) = self.graph.registration_world().get_entity_mut(*entity) {
                entity_mut.despawn();
            }
        }
        self.local_candidates.extend(
            core::mem::take(&mut self.committed_pending_entities)
                .into_iter()
                .map(ReleasedSignalCandidate::new),
        );
        let plan_id = self.id();
        for publication in &self.publications {
            let mut state = lock_materializer_state(&publication.state);
            if matches!(
                &*state,
                SharedMaterializerState::Initializing {
                    plan_id: initializing_plan,
                    ..
                } if *initializing_plan == plan_id
            ) {
                *state = SharedMaterializerState::Panicked;
            }
        }
        self.local_candidates
            .extend(crate::graph::release_registration_entities(
                self.graph.registration_world(),
                core::mem::take(&mut self.retained_entities),
            ));
        self.process_local_candidates();
    }
}

#[derive(Default)]
pub(crate) struct RegistrationCleanup {
    candidates: Vec<ReleasedSignalCandidate>,
    retained_entities: Vec<SignalEntity>,
}

impl RegistrationCleanup {
    pub(crate) fn is_empty(&self) -> bool {
        self.candidates.is_empty() && self.retained_entities.is_empty()
    }

    pub(crate) fn finish(mut self, world: &mut World) {
        if self.is_empty() {
            return;
        }
        self.candidates.extend(crate::graph::release_registration_entities(
            world,
            self.retained_entities,
        ));
        cleanup::process_signal_candidates(world, self.candidates);
    }
}

pub(crate) struct CommittedMaterialization {
    #[cfg(test)]
    pub(crate) terminal: SignalEntity,
    pub(crate) initial_claim: Option<SignalClaim>,
    pub(crate) private_claims: Vec<SignalClaim>,
    pub(crate) dynamic_connections: Vec<DynamicSignalConnection>,
    retired: bool,
    cleanup: RegistrationCleanup,
}

/// Safe graph-planning capability for one world and one materialization scope.
///
/// Jonmo creates the registrar and combines every declaration made through it into one complete,
/// allocation-free preflight. Structured rejection occurs before Bevy entities or systems are
/// allocated, and a successful commit has no ordinary failure path.
///
/// Returned [`SignalEntityRef`] values are typed dependency capabilities, not claims. `()` means no
/// dependency; a scalar reference whose event type is `()` is still a real dependency. Arrays,
/// vectors, and boxed slices represent homogeneous inputs and duplicate exact identities are
/// canonicalized. Use typed unary adapters to combine heterogeneous semantic roles.
///
/// Ordinary entity constructors are non-replayable. Managed replay constructors and
/// [`dynamic_operator`](Self::dynamic_operator) declare replay and dynamic semantics while the
/// framework privately owns claims, routes, connections, slots, generations, cursors, cohorts, and
/// cleanup. No raw graph-mutation or ownership token is exposed.
pub struct SignalRegistrar<'world, 'materialization> {
    plan: &'materialization mut PendingSignalMaterialization<'world>,
    invariant_brand: PhantomData<fn(&'materialization mut ()) -> &'materialization mut ()>,
}

impl<'world, 'materialization> SignalRegistrar<'world, 'materialization> {
    fn new(plan: &'materialization mut PendingSignalMaterialization<'world>) -> Self {
        Self {
            plan,
            invariant_brand: PhantomData,
        }
    }

    pub(crate) fn preflight_component<C: Component>(&mut self, entity: Entity) -> Result<&C, SignalRegistrationError> {
        self.plan
            .graph
            .registration_world()
            .get::<C>(entity)
            .ok_or(SignalRegistrationError::EntityUnavailable)
    }

    /// Adds an existing scalar materializer to this plan without acquiring a direct claim.
    pub fn materialize_signal<T: 'static>(
        &mut self,
        materializer: ScalarSignalMaterializer<T>,
    ) -> Result<ScalarSignalEntityRef<'materialization, T>, SignalRegistrationError> {
        materializer.materialize_consuming_into(self)
    }

    /// Adds an existing vector materializer to this plan without acquiring a direct claim.
    pub fn materialize_signal_vec<T: 'static>(
        &mut self,
        materializer: VecSignalMaterializer<T>,
    ) -> Result<VecSignalEntityRef<'materialization, T>, SignalRegistrationError> {
        materializer.materialize_consuming_into(self)
    }

    /// Adds an existing map materializer to this plan without acquiring a direct claim.
    pub fn materialize_signal_map<K: 'static, V: 'static>(
        &mut self,
        materializer: MapSignalMaterializer<K, V>,
    ) -> Result<MapSignalEntityRef<'materialization, K, V>, SignalRegistrationError> {
        materializer.materialize_consuming_into(self)
    }

    /// Plans an ordinary scalar signal entity with explicit output delivery policy.
    pub fn signal_entity<Inputs, T, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        delivery: OutputDelivery<T>,
    ) -> Result<ScalarSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        T: Send + Sync + 'static,
        SystemOutput: Into<Option<T>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_system_entity::<Inputs, ScalarSignalKind<T>, SystemOutput, System, Marker>(
            inputs,
            system,
            delivery,
            PlannedReplayState::NotReplayable,
            None,
        )
    }

    pub(crate) fn ordered_manager_bridge<T, System, Marker>(
        &mut self,
        manager: ScalarSignalEntityRef<'materialization, T>,
        system: System,
    ) -> Result<ScalarSignalEntityRef<'materialization, ()>, SignalRegistrationError>
    where
        T: Send + Sync + 'static,
        System: IntoSystem<In<T>, Option<()>, Marker> + Send + Sync + 'static,
    {
        let bridge = self.signal_entity(manager, system, OutputDelivery::move_only())?;
        let pending = self
            .plan
            .entities
            .iter_mut()
            .find(|pending| pending.reference == bridge.planned)
            .expect("new manager bridge must belong to the pending plan");
        pending.runtime.ordered_input_consumption = Some(OrderedInputConsumption::ManagerBridge);
        Ok(bridge)
    }

    /// Plans an ordinary vector signal entity with explicit output delivery policy.
    pub fn signal_vec_entity<Inputs, T, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        delivery: OutputDelivery<Vec<VecDiff<T>>>,
    ) -> Result<VecSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        T: Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<VecDiff<T>>>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_system_entity::<Inputs, VecSignalKind<T>, SystemOutput, System, Marker>(
            inputs,
            system,
            delivery,
            PlannedReplayState::NotReplayable,
            None,
        )
    }

    pub(crate) fn ordered_chain_consumer<T, Transform>(
        &mut self,
        inputs: [ScalarSignalEntityRef<'materialization, ChainInput<T>>; 2],
        mut transform: Transform,
        delivery: OutputDelivery<Vec<VecDiff<T>>>,
    ) -> Result<VecSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        T: Send + Sync + 'static,
        Transform: FnMut(ChainInput<T>, u64) -> Option<Vec<VecDiff<T>>> + Send + Sync + 'static,
    {
        let binding = Arc::new(OnceLock::new());
        let consumer = self.signal_vec_entity(
            inputs,
            {
                let binding = binding.clone();
                move |In(input): In<ChainInput<T>>, world: &mut World| {
                    let entity = *binding
                        .get()
                        .expect("chain consumer identity must be published before execution");
                    transform(input, crate::graph::signal_input_epoch(world, entity))
                }
            },
            delivery,
        )?;
        self.install_entity_component(consumer, move |_world, entity, _dependencies| {
            binding
                .set(entity)
                .expect("chain consumer identity must be published only once");
        })?;
        let pending = self
            .plan
            .entities
            .iter_mut()
            .find(|pending| pending.reference == consumer.planned)
            .expect("new chain consumer must belong to the pending plan");
        pending.runtime.ordered_input_consumption = Some(OrderedInputConsumption::chain::<T>());
        Ok(consumer)
    }

    /// Plans an ordinary map signal entity with explicit output delivery policy.
    pub fn signal_map_entity<Inputs, K, V, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        delivery: OutputDelivery<Vec<MapDiff<K, V>>>,
    ) -> Result<MapSignalEntityRef<'materialization, K, V>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<MapDiff<K, V>>>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_system_entity::<Inputs, MapSignalKind<K, V>, SystemOutput, System, Marker>(
            inputs,
            system,
            delivery,
            PlannedReplayState::NotReplayable,
            None,
        )
    }

    /// Plans a derived framework-managed replayable scalar entity in `Warming` state.
    ///
    /// The entity becomes ready only when its system accepts an actual output.
    pub fn replayable_signal_entity<Inputs, T, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        _initialization: ReplayInitialization,
        delivery: OutputDelivery<T>,
    ) -> Result<ScalarSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        T: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<T>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_replayable_signal(inputs, system, ScalarReplayCache::warming(), delivery)
    }

    /// Plans a ready graph-independent replayable scalar root with authoritative `current` state.
    ///
    /// `current` may itself be `None`; for `T = Option<U>`, `Ready(None)` is authoritative.
    pub fn replayable_signal_source<T, SystemOutput, System, Marker>(
        &mut self,
        current: T,
        system: System,
        delivery: OutputDelivery<T>,
    ) -> Result<ScalarSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        T: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<T>> + 'static,
        System: IntoSystem<In<()>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_replayable_signal((), system, ScalarReplayCache::ready(current), delivery)
    }

    fn plan_replayable_signal<Inputs, T, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        cache: ScalarReplayCache<T>,
        delivery: OutputDelivery<T>,
    ) -> Result<ScalarSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        T: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<T>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        let initial_inspection = match cache.snapshot() {
            ReplayCacheSnapshot::Warming => PlannedReplayInspection::Warming,
            ReplayCacheSnapshot::Ready { current, revision } => {
                let current = current.clone();
                PlannedReplayInspection::Ready {
                    make_snapshot: Arc::new(move || Box::new(current.clone())),
                    revision,
                }
            }
        };
        let installer = Box::new(move |world: &mut World, entity: SignalEntity| {
            world.entity_mut(*entity).insert(cache);
        });
        self.plan_system_entity::<Inputs, ScalarSignalKind<T>, SystemOutput, System, Marker>(
            inputs,
            system,
            delivery,
            PlannedReplayState::Replayable {
                key: ReplayRuntimeKey::new::<ScalarReplayCache<T>, T>(),
                contract: scalar_replay_runtime_contract::<T>(),
                initial_inspection,
            },
            Some(installer),
        )
    }

    /// Plans a derived framework-managed replayable vector entity in `Warming` state.
    pub fn replayable_signal_vec_entity<Inputs, T, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        _initialization: ReplayInitialization,
        delivery: OutputDelivery<Vec<VecDiff<T>>>,
    ) -> Result<VecSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        T: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<VecDiff<T>>>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_replayable_signal_vec(inputs, system, VecReplayCache::warming(), delivery)
    }

    /// Plans a ready graph-independent replayable vector root.
    ///
    /// An empty `current` vector is an authoritative ready snapshot, not warming.
    pub fn replayable_signal_vec_source<T, SystemOutput, System, Marker>(
        &mut self,
        current: Vec<T>,
        system: System,
        delivery: OutputDelivery<Vec<VecDiff<T>>>,
    ) -> Result<VecSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        T: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<VecDiff<T>>>> + 'static,
        System: IntoSystem<In<()>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_replayable_signal_vec((), system, VecReplayCache::ready(current), delivery)
    }

    fn plan_replayable_signal_vec<Inputs, T, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        cache: VecReplayCache<T>,
        delivery: OutputDelivery<Vec<VecDiff<T>>>,
    ) -> Result<VecSignalEntityRef<'materialization, T>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        T: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<VecDiff<T>>>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        let initial_inspection = match cache.snapshot() {
            ReplayCacheSnapshot::Warming => PlannedReplayInspection::Warming,
            ReplayCacheSnapshot::Ready { current, revision } => {
                let current = current.clone();
                PlannedReplayInspection::Ready {
                    make_snapshot: Arc::new(move || {
                        Box::new(Vec::from([VecDiff::Replace {
                            values: current.clone(),
                        }]))
                    }),
                    revision,
                }
            }
        };
        let installer = Box::new(move |world: &mut World, entity: SignalEntity| {
            world.entity_mut(*entity).insert(cache);
        });
        self.plan_system_entity::<Inputs, VecSignalKind<T>, SystemOutput, System, Marker>(
            inputs,
            system,
            delivery,
            PlannedReplayState::Replayable {
                key: ReplayRuntimeKey::new::<VecReplayCache<T>, Vec<VecDiff<T>>>(),
                contract: vec_replay_runtime_contract::<T>(),
                initial_inspection,
            },
            Some(installer),
        )
    }

    /// Plans a derived framework-managed replayable map entity in `Warming` state.
    pub fn replayable_signal_map_entity<K, V, Inputs, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        _initialization: ReplayInitialization,
        delivery: OutputDelivery<Vec<MapDiff<K, V>>>,
    ) -> Result<MapSignalEntityRef<'materialization, K, V>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<MapDiff<K, V>>>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_replayable_signal_map(inputs, system, MapReplayCache::warming(), delivery)
    }

    /// Plans a ready graph-independent replayable map root.
    ///
    /// An empty `current` map is an authoritative ready snapshot, not warming.
    pub fn replayable_signal_map_source<K, V, SystemOutput, System, Marker>(
        &mut self,
        current: BTreeMap<K, V>,
        system: System,
        delivery: OutputDelivery<Vec<MapDiff<K, V>>>,
    ) -> Result<MapSignalEntityRef<'materialization, K, V>, SignalRegistrationError>
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<MapDiff<K, V>>>> + 'static,
        System: IntoSystem<In<()>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        self.plan_replayable_signal_map((), system, MapReplayCache::ready(current), delivery)
    }

    fn plan_replayable_signal_map<K, V, Inputs, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        cache: MapReplayCache<K, V>,
        delivery: OutputDelivery<Vec<MapDiff<K, V>>>,
    ) -> Result<MapSignalEntityRef<'materialization, K, V>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
        SystemOutput: Into<Option<Vec<MapDiff<K, V>>>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        let initial_inspection = match cache.snapshot() {
            ReplayCacheSnapshot::Warming => PlannedReplayInspection::Warming,
            ReplayCacheSnapshot::Ready { current, revision } => {
                let entries: Vec<(K, V)> = current
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                PlannedReplayInspection::Ready {
                    make_snapshot: Arc::new(move || {
                        Box::new(Vec::from([MapDiff::Replace {
                            entries: entries.clone(),
                        }]))
                    }),
                    revision,
                }
            }
        };
        let installer = Box::new(move |world: &mut World, entity: SignalEntity| {
            world.entity_mut(*entity).insert(cache);
        });
        self.plan_system_entity::<Inputs, MapSignalKind<K, V>, SystemOutput, System, Marker>(
            inputs,
            system,
            delivery,
            PlannedReplayState::Replayable {
                key: ReplayRuntimeKey::new::<MapReplayCache<K, V>, Vec<MapDiff<K, V>>>(),
                contract: map_replay_runtime_contract::<K, V>(),
                initial_inspection,
            },
            Some(installer),
        )
    }

    pub(crate) fn scalar_cache_terminal<T: 'static>(
        &mut self,
        input: ScalarSignalEntityRef<'materialization, T>,
    ) -> Result<Option<ScalarSignalEntityRef<'materialization, T>>, SignalRegistrationError> {
        self.planned_replayable(input.planned)
            .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn vec_cache_terminal<T: 'static>(
        &mut self,
        input: VecSignalEntityRef<'materialization, T>,
    ) -> Result<Option<VecSignalEntityRef<'materialization, T>>, SignalRegistrationError> {
        self.planned_replayable(input.planned)
            .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn map_cache_terminal<K: 'static, V: 'static>(
        &mut self,
        input: MapSignalEntityRef<'materialization, K, V>,
    ) -> Result<Option<MapSignalEntityRef<'materialization, K, V>>, SignalRegistrationError> {
        self.planned_replayable(input.planned)
            .map(|replayable| replayable.then_some(input))
    }

    fn planned_replayable(&self, entity: PlannedSignalEntityRef) -> Result<bool, SignalRegistrationError> {
        match entity {
            reference @ PlannedSignalEntityRef::Pending { .. } => self
                .plan
                .entities
                .iter()
                .find(|pending| pending.reference == reference)
                .map(|pending| pending.replay.is_replayable())
                .ok_or(SignalRegistrationError::EntityUnavailable),
            PlannedSignalEntityRef::Existing(entity) => self
                .plan
                .graph
                .registration_existing_signal(entity)
                .map(|existing| existing.replayable)
                .ok_or(SignalRegistrationError::EntityUnavailable),
        }
    }

    pub(crate) fn scalar_broadcast_replayable_terminal<T: Clone + Send + Sync + 'static>(
        &mut self,
        input: ScalarSignalEntityRef<'materialization, T>,
    ) -> Result<Option<ScalarSignalEntityRef<'materialization, T>>, SignalRegistrationError> {
        self.planned_broadcast_replayable(input.planned, OutputDelivery::<T>::broadcast().into_stored(), true)
            .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn scalar_try_broadcast_terminal<T: Clone + Send + Sync + 'static>(
        &mut self,
        input: ScalarSignalEntityRef<'materialization, T>,
    ) -> Result<Option<ScalarSignalEntityRef<'materialization, T>>, SignalRegistrationError> {
        self.planned_broadcast_replayable(input.planned, OutputDelivery::<T>::broadcast().into_stored(), false)
            .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn vec_broadcast_replayable_terminal<T: Clone + Send + Sync + 'static>(
        &mut self,
        input: VecSignalEntityRef<'materialization, T>,
    ) -> Result<Option<VecSignalEntityRef<'materialization, T>>, SignalRegistrationError> {
        self.planned_broadcast_replayable(
            input.planned,
            OutputDelivery::<Vec<VecDiff<T>>>::broadcast().into_stored(),
            true,
        )
        .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn vec_try_broadcast_terminal<T: Clone + Send + Sync + 'static>(
        &mut self,
        input: VecSignalEntityRef<'materialization, T>,
    ) -> Result<Option<VecSignalEntityRef<'materialization, T>>, SignalRegistrationError> {
        self.planned_broadcast_replayable(
            input.planned,
            OutputDelivery::<Vec<VecDiff<T>>>::broadcast().into_stored(),
            false,
        )
        .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn map_broadcast_replayable_terminal<
        K: Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    >(
        &mut self,
        input: MapSignalEntityRef<'materialization, K, V>,
    ) -> Result<Option<MapSignalEntityRef<'materialization, K, V>>, SignalRegistrationError> {
        self.planned_broadcast_replayable(
            input.planned,
            OutputDelivery::<Vec<MapDiff<K, V>>>::broadcast().into_stored(),
            true,
        )
        .map(|replayable| replayable.then_some(input))
    }

    pub(crate) fn map_try_broadcast_terminal<K: Clone + Send + Sync + 'static, V: Clone + Send + Sync + 'static>(
        &mut self,
        input: MapSignalEntityRef<'materialization, K, V>,
    ) -> Result<Option<MapSignalEntityRef<'materialization, K, V>>, SignalRegistrationError> {
        self.planned_broadcast_replayable(
            input.planned,
            OutputDelivery::<Vec<MapDiff<K, V>>>::broadcast().into_stored(),
            false,
        )
        .map(|replayable| replayable.then_some(input))
    }

    fn planned_broadcast_replayable(
        &mut self,
        entity: PlannedSignalEntityRef,
        broadcast_delivery: StoredOutputDelivery,
        require_in_place: bool,
    ) -> Result<bool, SignalRegistrationError> {
        match entity {
            reference @ PlannedSignalEntityRef::Pending { .. } => {
                let pending = self
                    .plan
                    .entities
                    .iter_mut()
                    .find(|pending| pending.reference == reference)
                    .ok_or(SignalRegistrationError::EntityUnavailable)?;
                if !pending.replay.is_replayable() {
                    return Ok(false);
                }
                pending.runtime.output_delivery = broadcast_delivery;
                Ok(true)
            }
            PlannedSignalEntityRef::Existing(entity) => {
                let existing = self
                    .plan
                    .graph
                    .registration_existing_signal(entity)
                    .ok_or(SignalRegistrationError::EntityUnavailable)?;
                if !existing.replayable {
                    return Ok(false);
                }
                if !require_in_place && existing.output_delivery != Some(broadcast_delivery.key()) {
                    return Ok(false);
                }
                self.plan.existing_expectations.push(ExistingCanonicalExpectation {
                    entity,
                    dependencies: None,
                    replayable: Some(true),
                    output_delivery: Some(broadcast_delivery.key()),
                });
                Ok(true)
            }
        }
    }

    /// Plans one fully declared framework-managed dynamic operator.
    ///
    /// The public specification supplies only semantic commands, adaptation, reduction, replay
    /// policy, output kind, same-identity behavior, and optional candidate-error handling. Jonmo
    /// owns all dynamic topology and replacement cohorts.
    pub fn dynamic_operator<Spec>(
        &mut self,
        spec: Spec,
    ) -> Result<SignalEntityRef<'materialization, Spec::OutputKind>, SignalRegistrationError>
    where
        Spec: crate::dynamic::CompleteDynamicSignalOperatorSpec<'materialization>,
        <Spec::OutputKind as SignalKind>::Event: Send + Sync,
    {
        <Spec as crate::dynamic::sealed::CompleteDynamicSignalOperatorSpec<'materialization>>::plan(spec, self)
    }

    /// Records an ordered value-based schedule operation and returns the same scoped reference.
    ///
    /// Schedule operations participate in the same preflight and follow declaration order.
    pub fn apply_schedule<Kind, Schedule>(
        &mut self,
        input: SignalEntityRef<'materialization, Kind>,
        schedule: Schedule,
    ) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError>
    where
        Kind: SignalKind,
        Schedule: ScheduleLabel,
    {
        let schedule = schedule.intern();
        if !self.plan.graph.registration_schedule_is_registered(schedule) {
            return Err(SignalRegistrationError::ScheduleNotRegistered);
        }
        self.apply_planned_schedule(input.planned, schedule)?;
        Ok(input)
    }

    fn plan_system_entity<Inputs, Kind, SystemOutput, System, Marker>(
        &mut self,
        inputs: Inputs,
        system: System,
        delivery: OutputDelivery<Kind::Event>,
        replay: PlannedReplayState,
        replay_installer: Option<ReplayInstaller>,
    ) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError>
    where
        Inputs: RegistrarInputs<'materialization>,
        Kind: SignalOutputExecution,
        Kind::Event: Send + Sync,
        SystemOutput: Into<Option<Kind::Event>> + 'static,
        System: IntoSystem<In<Inputs::Event>, SystemOutput, Marker> + Send + Sync + 'static,
    {
        let output_delivery = delivery.into_stored();
        assert!(
            output_delivery.is_coherent(TypeId::of::<Kind::Event>()),
            "registrar output delivery policy was incoherent with signal event type {}",
            type_name::<Kind::Event>()
        );
        let allows_empty = sealed::RegistrarInputs::allows_empty(&inputs);
        let mut dependencies = sealed::RegistrarInputs::into_planned_inputs(inputs);
        if dependencies.is_empty() && !allows_empty {
            return Err(SignalRegistrationError::EmptyRegistrarInputs);
        }
        dependencies.sort_unstable();
        dependencies.dedup();
        let inherited_schedule = dependencies.iter().copied().find_map(|dependency| {
            self.planned_schedule_state(dependency)
                .ok()
                .and_then(|state| state.hint)
        });

        let boxed_system: BoxedSystem<In<Inputs::Event>, SystemOutput> = Box::new(IntoSystem::into_system(system));
        let system_id = Arc::new(OnceLock::<SystemId<In<Inputs::Event>, SystemOutput>>::new());
        let initializer_system_id = system_id.clone();
        let initializer = Box::new(move |world: &mut World| {
            let registered = world.register_boxed_system(boxed_system);
            if initializer_system_id.set(registered).is_err() {
                panic!("pending signal system was registered more than once");
            }
            SignalEntity::from(registered)
        });

        let runner = Arc::new(move |world: &mut World, input: ErasedSignalValue| {
            let input =
                input
                    .downcast::<Inputs::Event>()
                    .map_err(|_| RegistrationSystemRunError::InputTypeMismatch {
                        expected: type_name::<Inputs::Event>(),
                    })?;
            let registered = system_id
                .get()
                .copied()
                .ok_or(RegistrationSystemRunError::SystemUnavailable)?;
            let output = world
                .run_system_with(registered, *input)
                .map_err(|_| RegistrationSystemRunError::SystemUnavailable)?;
            let Some(event) = output.into() else {
                return Ok(None);
            };
            Ok(Some(Box::new(event) as ErasedSignalValue))
        });
        let replay_key = replay.key();
        let replay_contract = replay.contract().cloned();
        match (replay_key, &replay_contract) {
            (None, None) => {}
            (Some(key), Some(contract)) => {
                assert_eq!(
                    key.cache_type(),
                    contract.cache_type(),
                    "registrar replay contract cache type disagreed with its fixed runtime key"
                );
                assert_eq!(
                    key.event_type(),
                    TypeId::of::<Kind::Event>(),
                    "registrar replay runtime key event type disagreed with its signal kind"
                );
                assert_eq!(
                    key.event_type(),
                    contract.event_type(),
                    "registrar replay contract event type disagreed with its fixed runtime key"
                );
            }
            _ => panic!("registrar replay runtime key and contract presence diverged"),
        }

        let pending_reference = self.plan.allocate_pending_reference();
        let dependencies = dependencies.into_boxed_slice();
        self.plan.entities.push(PendingSignalEntity {
            reference: pending_reference,
            dependencies,
            initializer: Some(initializer),
            entity_installer: None,
            runtime: RegistrationSignalRuntime {
                runner,
                input_type: TypeId::of::<Inputs::Event>(),
                output_type: TypeId::of::<Kind::Event>(),
                output_delivery,
                collection_poll_append: Kind::collection_poll_append(),
                ordered_input_consumption: None,
                replay_key,
                replay_contract,
            },
            replay,
            replay_installer,
        });

        if let Some(schedule) = inherited_schedule {
            self.plan.schedule_projection.insert(
                pending_reference,
                PlannedScheduleState {
                    tag: Some(schedule),
                    hint: Some(schedule),
                },
            );
        }
        Ok(SignalEntityRef::new(pending_reference))
    }

    pub(crate) fn install_entity_component<Kind: SignalKind>(
        &mut self,
        entity: SignalEntityRef<'materialization, Kind>,
        installer: impl FnOnce(&mut World, SignalEntity, &[SignalEntity]) + Send + Sync + 'static,
    ) -> Result<(), SignalRegistrationError> {
        let pending = self
            .plan
            .entities
            .iter_mut()
            .find(|pending| pending.reference == entity.planned)
            .ok_or(SignalRegistrationError::EntityUnavailable)?;
        if pending.entity_installer.is_some() {
            return Err(SignalRegistrationError::ConflictingCanonicalState);
        }
        pending.entity_installer = Some(Box::new(installer));
        Ok(())
    }

    pub(crate) fn plan_dynamic_connection<SourceKind: SignalKind, TargetKind: SignalKind>(
        &mut self,
        source: SignalEntityRef<'materialization, SourceKind>,
        target: SignalEntityRef<'materialization, TargetKind>,
    ) {
        self.plan
            .dynamic_connections
            .push(RegistrationDynamicConnectionPreflight {
                source: source.planned,
                target: target.planned,
            });
    }

    fn planned_schedule_state(
        &mut self,
        entity: PlannedSignalEntityRef,
    ) -> Result<PlannedScheduleState, SignalRegistrationError> {
        if let Some(state) = self.plan.schedule_projection.get(&entity).copied() {
            return Ok(state);
        }
        let state = match entity {
            reference @ PlannedSignalEntityRef::Pending { .. } => self
                .plan
                .pending_reference_exists(reference)
                .then(PlannedScheduleState::default)
                .ok_or(SignalRegistrationError::EntityUnavailable)?,
            PlannedSignalEntityRef::Existing(entity) => {
                let (tag, hint) = self.plan.graph.registration_existing_schedule(entity)?;
                PlannedScheduleState { tag, hint }
            }
        };
        self.plan.schedule_projection.insert(entity, state);
        Ok(state)
    }

    fn planned_dependencies(
        &self,
        entity: PlannedSignalEntityRef,
    ) -> Result<Vec<PlannedSignalEntityRef>, SignalRegistrationError> {
        match entity {
            reference @ PlannedSignalEntityRef::Pending { .. } => self
                .plan
                .entities
                .iter()
                .find(|pending| pending.reference == reference)
                .map(|pending| pending.dependencies.to_vec())
                .ok_or(SignalRegistrationError::EntityUnavailable),
            PlannedSignalEntityRef::Existing(entity) => self
                .plan
                .graph
                .registration_existing_signal(entity)
                .map(|existing| {
                    existing
                        .dependencies
                        .iter()
                        .copied()
                        .map(PlannedSignalEntityRef::Existing)
                        .collect()
                })
                .ok_or(SignalRegistrationError::EntityUnavailable),
        }
    }

    fn apply_planned_schedule(
        &mut self,
        entity: PlannedSignalEntityRef,
        schedule: InternedScheduleLabel,
    ) -> Result<(), SignalRegistrationError> {
        let mut root = self.planned_schedule_state(entity)?;
        root.tag = Some(schedule);
        root.hint = Some(schedule);
        self.plan.schedule_projection.insert(entity, root);

        let mut stack = self.planned_dependencies(entity)?;
        let mut visited = BTreeSet::new();
        while let Some(upstream) = stack.pop() {
            if !visited.insert(upstream) {
                continue;
            }
            let mut state = self.planned_schedule_state(upstream)?;
            if state.tag.is_none() {
                state.tag = Some(schedule);
                self.plan.schedule_projection.insert(upstream, state);
            }
            stack.extend(self.planned_dependencies(upstream)?);
        }
        Ok(())
    }
}

type MaterializerInitializer<Kind> = Box<
    dyn for<'world, 'materialization> FnOnce(
            &mut SignalRegistrar<'world, 'materialization>,
        )
            -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError>
        + Send
        + 'static,
>;

struct SignalMaterializationCache {
    terminal: SignalEntity,
    cleanup: CleanupSender,
}

enum SharedMaterializerState {
    Uninitialized(Option<Box<dyn AnyMaterializerInitializer>>),
    Initializing {
        plan_id: MaterializationPlanId,
        terminal: Option<PlannedSignalEntityRef>,
        recursive_rejection: bool,
        terminal_candidate_pending: bool,
    },
    Ready {
        cache: SignalMaterializationCache,
        output_delivery: OutputDeliveryKey,
        terminal_candidate_published: bool,
    },
    Rejected(SignalRegistrationError),
    Panicked,
}

trait AnyMaterializerInitializer: Send {
    fn kind_type(&self) -> TypeId;
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;
}

struct TypedMaterializerInitializer<Kind: SignalKind>(MaterializerInitializer<Kind>);

impl<Kind: SignalKind> AnyMaterializerInitializer for TypedMaterializerInitializer<Kind> {
    fn kind_type(&self) -> TypeId {
        TypeId::of::<Kind>()
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        Box::new(self.0)
    }
}

struct SignalMaterializerState {
    live_chains: AtomicUsize,
    state: Mutex<SharedMaterializerState>,
    #[cfg(test)]
    after_final_live_chain_decrement: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

pub(crate) struct SignalTerminalRetainer(Weak<SignalMaterializerState>);

impl SignalTerminalRetainer {
    pub(crate) fn is_live(&self) -> bool {
        self.0
            .upgrade()
            .is_some_and(|state| state.live_chains.load(AtomicOrdering::Acquire) != 0)
    }
}

fn lock_materializer_state(
    state: &SignalMaterializerState,
) -> bevy_platform::sync::MutexGuard<'_, SharedMaterializerState> {
    state.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Framework-owned one-shot lazy backing for one terminal signal identity.
///
/// A materializer is not a claim or a reusable factory. External [`Signal`](crate::signal::Signal),
/// [`SignalVec`](crate::signal_vec::SignalVec), and [`SignalMap`](crate::signal_map::SignalMap)
/// implementations create it through [`new`](Self::new) and transfer it to Jonmo. Its higher-ranked
/// initializer receives only a registrar-scoped planning capability.
///
/// Concrete chain clones may share the private backing, but retention applies only to the terminal.
/// A retained replayable terminal owns one private maintenance claim. Registration consumes this
/// materializer, including on structured rejection; construct a fresh chain/materializer to retry.
/// Complete preflight leaves graph and Bevy allocation/materialization state unchanged on
/// detectable rejection. After materialization, registrations and releases must use the originating
/// world.
///
/// ```compile_fail,E0277
/// use jonmo::ScalarSignalMaterializer;
/// fn require_clone<T: Clone>() {}
/// require_clone::<ScalarSignalMaterializer<i32>>();
/// ```
///
/// ```compile_fail,E0277
/// use jonmo::ScalarSignalMaterializer;
/// fn require_copy<T: Copy>() {}
/// require_copy::<ScalarSignalMaterializer<i32>>();
/// ```
///
/// ```compile_fail,E0624
/// use bevy_ecs::prelude::In;
/// use jonmo::{OutputDelivery, ScalarSignalMaterializer};
/// let materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
///     registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
/// });
/// let _shared = materializer.shared_clone();
/// ```
pub struct SignalMaterializer<Kind: SignalKind> {
    state: Arc<SignalMaterializerState>,
    armed: bool,
    kind: PhantomData<fn() -> Kind>,
}

pub(crate) struct SignalMaterializerFactory<Kind: SignalKind> {
    state: Arc<SignalMaterializerState>,
    kind: PhantomData<fn() -> Kind>,
}

impl<Kind: SignalKind> SignalMaterializerFactory<Kind> {
    pub(crate) fn materializer(&self) -> SignalMaterializer<Kind> {
        cleanup::update_atomic_usize(
            &self.state.live_chains,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
            |current| current.checked_add(1),
        )
        .unwrap_or_else(|_| panic!("signal materializer live-chain count overflowed"));
        SignalMaterializer {
            state: self.state.clone(),
            armed: true,
            kind: PhantomData,
        }
    }
}

impl<Kind: SignalKind> Drop for SignalMaterializerFactory<Kind> {
    fn drop(&mut self) {
        drop(SignalMaterializer {
            state: self.state.clone(),
            armed: true,
            kind: PhantomData::<fn() -> Kind>,
        });
    }
}

/// A scalar signal materializer.
pub type ScalarSignalMaterializer<T> = SignalMaterializer<ScalarSignalKind<T>>;
/// A vector signal materializer.
pub type VecSignalMaterializer<T> = SignalMaterializer<VecSignalKind<T>>;
/// A map signal materializer.
pub type MapSignalMaterializer<K, V> = SignalMaterializer<MapSignalKind<K, V>>;

impl<Kind: SignalKind> SignalMaterializer<Kind> {
    /// Creates one lazy materializer from a higher-ranked registrar-scoped initializer.
    pub fn new<Initializer>(initializer: Initializer) -> Self
    where
        Initializer: for<'world, 'materialization> FnOnce(
                &mut SignalRegistrar<'world, 'materialization>,
            ) -> Result<
                SignalEntityRef<'materialization, Kind>,
                SignalRegistrationError,
            > + Send
            + 'static,
    {
        let initializer: MaterializerInitializer<Kind> = Box::new(initializer);
        Self {
            state: Arc::new(SignalMaterializerState {
                live_chains: AtomicUsize::new(1),
                state: Mutex::new(SharedMaterializerState::Uninitialized(Some(Box::new(
                    TypedMaterializerInitializer(initializer),
                )))),
                #[cfg(test)]
                after_final_live_chain_decrement: Mutex::new(None),
            }),
            armed: true,
            kind: PhantomData,
        }
    }

    /// Consumes this materializer, then attempts to materialize its terminal and acquire one direct
    /// claim.
    ///
    /// On structured rejection, preflight performs no graph or Bevy allocation/materialization
    /// mutation. The consumed materializer is not returned; construct a fresh chain to retry.
    pub fn try_register(self, world: &mut World) -> Result<SignalClaim, SignalRegistrationError> {
        self.materialize_and_claim_with_graph(world)
    }

    /// Consumes this materializer, materializes its terminal, and acquires one direct claim.
    ///
    /// # Panics
    ///
    /// Panics if complete registration preflight rejects the plan. Use
    /// [`try_register`](Self::try_register) for structured error handling.
    #[track_caller]
    pub fn register(self, world: &mut World) -> SignalClaim {
        self.try_register(world)
            .unwrap_or_else(|error| panic!("signal registration rejected before commit ({error:?}): {error}"))
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn test_after_final_live_chain_decrement(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        *self
            .state
            .after_final_live_chain_decrement
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(hook);
    }

    pub(crate) fn shared_clone(&self) -> Self {
        cleanup::update_atomic_usize(
            &self.state.live_chains,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
            |current| current.checked_add(1),
        )
        .unwrap_or_else(|_| panic!("signal materializer live-chain count overflowed"));
        Self {
            state: self.state.clone(),
            armed: true,
            kind: PhantomData,
        }
    }

    pub(crate) fn into_factory(mut self) -> SignalMaterializerFactory<Kind> {
        assert!(
            matches!(
                &*lock_materializer_state(&self.state),
                SharedMaterializerState::Uninitialized(_)
            ),
            "only an uninitialized materializer can become shared source backing"
        );
        assert_eq!(
            self.state.live_chains.load(AtomicOrdering::Acquire),
            1,
            "shared source backing requires exactly one initial materializer owner"
        );
        self.armed = false;
        SignalMaterializerFactory {
            state: self.state.clone(),
            kind: PhantomData,
        }
    }

    fn release_live_chain(&mut self) -> bool {
        assert!(self.armed, "signal materializer owner was consumed more than once");
        let previous = cleanup::update_atomic_usize(
            &self.state.live_chains,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
            |current| current.checked_sub(1),
        )
        .unwrap_or_else(|_| panic!("signal materializer live-chain count underflowed"));
        self.armed = false;
        let released_final = previous == 1;

        #[cfg(test)]
        if released_final {
            let hook = self
                .state
                .after_final_live_chain_decrement
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            if let Some(hook) = hook {
                hook();
            }
        }
        released_final
    }

    fn transfer_owner_to_plan(&mut self, plan: &mut PendingSignalMaterialization<'_>) {
        if !self.release_live_chain() {
            return;
        }

        let candidate = {
            let mut state = lock_materializer_state(&self.state);
            match &mut *state {
                SharedMaterializerState::Initializing {
                    terminal_candidate_pending,
                    ..
                } => {
                    *terminal_candidate_pending = true;
                    None
                }
                SharedMaterializerState::Ready {
                    cache,
                    terminal_candidate_published,
                    ..
                } if !*terminal_candidate_published => {
                    *terminal_candidate_published = true;
                    Some(ReleasedSignalCandidate::new(cache.terminal))
                }
                SharedMaterializerState::Uninitialized(_)
                | SharedMaterializerState::Ready { .. }
                | SharedMaterializerState::Rejected(_)
                | SharedMaterializerState::Panicked => None,
            }
        };
        if let Some(candidate) = candidate {
            plan.local_candidates.push(candidate);
        }
    }

    fn materialize_consuming_into<'world, 'materialization>(
        mut self,
        registrar: &mut SignalRegistrar<'world, 'materialization>,
    ) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError> {
        self.transfer_owner_to_plan(registrar.plan);
        self.materialize_into(registrar)
    }

    fn materialize_into<'world, 'materialization>(
        &self,
        registrar: &mut SignalRegistrar<'world, 'materialization>,
    ) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError> {
        let plan_id = registrar.plan.id();
        let erased_initializer = {
            let mut state = lock_materializer_state(&self.state);
            match &mut *state {
                SharedMaterializerState::Uninitialized(initializer) => {
                    let initializer = initializer
                        .take()
                        .unwrap_or_else(|| panic!("uninitialized materializer lost its one-shot initializer"));
                    *state = SharedMaterializerState::Initializing {
                        plan_id,
                        terminal: None,
                        recursive_rejection: false,
                        terminal_candidate_pending: false,
                    };
                    initializer
                }
                SharedMaterializerState::Initializing {
                    plan_id: initializing_plan,
                    terminal: Some(terminal),
                    ..
                } if *initializing_plan == plan_id => {
                    registrar
                        .plan
                        .maintenance_candidates
                        .push(MaterializerMaintenanceCandidate {
                            state: self.state.clone(),
                            terminal: *terminal,
                        });
                    return Ok(SignalEntityRef::new(*terminal));
                }
                SharedMaterializerState::Initializing {
                    plan_id: initializing_plan,
                    recursive_rejection,
                    ..
                } if *initializing_plan == plan_id => {
                    *recursive_rejection = true;
                    return Err(SignalRegistrationError::CycleDetected);
                }
                SharedMaterializerState::Initializing { .. } => {
                    return Err(SignalRegistrationError::SharedMaterializerInitializing);
                }
                SharedMaterializerState::Ready {
                    cache, output_delivery, ..
                } => {
                    let terminal = PlannedSignalEntityRef::Existing(cache.terminal);
                    registrar.plan.existing_expectations.push(ExistingCanonicalExpectation {
                        entity: cache.terminal,
                        dependencies: None,
                        replayable: None,
                        output_delivery: Some(*output_delivery),
                    });
                    registrar
                        .plan
                        .maintenance_candidates
                        .push(MaterializerMaintenanceCandidate {
                            state: self.state.clone(),
                            terminal,
                        });
                    return Ok(SignalEntityRef::new(terminal));
                }
                SharedMaterializerState::Rejected(previous) => {
                    let _historical_error = *previous;
                    return Err(SignalRegistrationError::SharedMaterializerRejected);
                }
                SharedMaterializerState::Panicked => {
                    return Err(SignalRegistrationError::SharedMaterializerPanicked);
                }
            }
        };

        let initializer = if erased_initializer.kind_type() == TypeId::of::<Kind>() {
            erased_initializer
                .into_any()
                .downcast::<MaterializerInitializer<Kind>>()
                .unwrap_or_else(|_| panic!("materializer initializer kind erasure was inconsistent"))
        } else {
            panic!("materializer initializer kind did not match its materializer")
        };

        let mut panic_guard = MaterializerPanicGuard {
            state: self.state.clone(),
            plan_id,
            armed: true,
        };
        let terminal = match initializer(registrar) {
            Ok(terminal) => terminal,
            Err(error) => {
                let mut state = lock_materializer_state(&self.state);
                *state = SharedMaterializerState::Rejected(error);
                panic_guard.armed = false;
                return Err(error);
            }
        };

        {
            let mut state = lock_materializer_state(&self.state);
            match &mut *state {
                SharedMaterializerState::Initializing {
                    plan_id: initializing_plan,
                    terminal: planned_terminal,
                    recursive_rejection,
                    ..
                } if *initializing_plan == plan_id => {
                    if *recursive_rejection {
                        *state = SharedMaterializerState::Rejected(SignalRegistrationError::CycleDetected);
                        panic_guard.armed = false;
                        return Err(SignalRegistrationError::CycleDetected);
                    }
                    *planned_terminal = Some(terminal.planned);
                }
                _ => panic!("materializer state changed while its initializer was running"),
            }
        }
        registrar.plan.publications.push(MaterializerPublication {
            state: self.state.clone(),
            terminal: terminal.planned,
        });
        registrar
            .plan
            .maintenance_candidates
            .push(MaterializerMaintenanceCandidate {
                state: self.state.clone(),
                terminal: terminal.planned,
            });
        panic_guard.armed = false;
        Ok(terminal)
    }

    #[cfg(test)]
    pub(crate) fn materialize_with_graph(
        self,
        graph: &mut dyn SignalRegistrationGraphHooks,
    ) -> Result<SignalEntity, SignalRegistrationError> {
        self.materialize_committed(graph, false)
            .map(|committed| committed.terminal)
    }

    pub(crate) fn materialize_and_claim_with_graph(
        self,
        graph: &mut dyn SignalRegistrationGraphHooks,
    ) -> Result<SignalClaim, SignalRegistrationError> {
        let committed = self.materialize_committed(graph, true)?;
        committed
            .initial_claim
            .ok_or(SignalRegistrationError::EntityUnavailable)
    }

    fn materialize_committed(
        self,
        graph: &mut dyn SignalRegistrationGraphHooks,
        acquire_initial_claim: bool,
    ) -> Result<CommittedMaterialization, SignalRegistrationError> {
        let mut plan = PendingSignalMaterialization::new(graph);
        let terminal = {
            let mut registrar = SignalRegistrar::new(&mut plan);
            match self.materialize_consuming_into(&mut registrar) {
                Ok(terminal) => terminal.planned,
                Err(error) => {
                    plan.reject(error);
                    return Err(error);
                }
            }
        };
        if acquire_initial_claim {
            plan.claim_deltas.push(PlannedClaimDelta {
                entity: terminal,
                increment: 1,
            });
        }
        if let Err(error) = plan.plan_replay_maintenance() {
            plan.reject(error);
            return Err(error);
        }

        if let Err(error) = plan.preflight(terminal, true) {
            plan.reject(error);
            return Err(error);
        }
        Ok(plan.commit(terminal, acquire_initial_claim, None))
    }
}

impl<Kind: SignalKind> Drop for SignalMaterializer<Kind> {
    fn drop(&mut self) {
        if !self.armed || !self.release_live_chain() {
            return;
        }

        let release = {
            let mut state = lock_materializer_state(&self.state);
            match &mut *state {
                SharedMaterializerState::Initializing {
                    terminal_candidate_pending,
                    ..
                } => {
                    *terminal_candidate_pending = true;
                    None
                }
                SharedMaterializerState::Ready {
                    cache,
                    terminal_candidate_published,
                    ..
                } if !*terminal_candidate_published => {
                    *terminal_candidate_published = true;
                    Some((cache.terminal, cache.cleanup.clone()))
                }
                SharedMaterializerState::Ready { .. } => None,
                SharedMaterializerState::Uninitialized(_)
                | SharedMaterializerState::Rejected(_)
                | SharedMaterializerState::Panicked => None,
            }
        };
        if let Some((terminal, cleanup)) = release {
            cleanup.send_stale_signal(ReleasedSignalCandidate::new(terminal));
        }
    }
}

pub(crate) enum DynamicCandidateMaterialization {
    Live {
        candidates: Vec<CommittedDynamicCandidate>,
        cleanup: RegistrationCleanup,
    },
    Retired {
        claims: Vec<SignalClaim>,
        cleanup: RegistrationCleanup,
    },
}

pub(crate) enum CommittedDynamicCandidate {
    KeptCurrent,
    New {
        selected: SignalEntity,
        slot: DynamicStableSlot,
        adapter_claim: SignalClaim,
        connection: DynamicSignalConnection,
    },
}

pub(crate) struct DynamicCandidateRequest<SelectedKind: SignalKind> {
    pub(crate) selected: SignalMaterializer<SelectedKind>,
    pub(crate) slot: DynamicStableSlot,
    pub(crate) current_selected: Option<SignalEntity>,
}

pub(crate) fn materialize_dynamic_candidates<SelectedKind, SystemOutput, System, Marker>(
    graph: &mut dyn SignalRegistrationGraphHooks,
    candidates: Vec<DynamicCandidateRequest<SelectedKind>>,
    owner: SignalEntity,
    same_identity: crate::dynamic::SameSignalIdentity,
    mut make_adapter: impl FnMut(Arc<OnceLock<DynamicAdapterBinding>>) -> System,
) -> Result<DynamicCandidateMaterialization, SignalRegistrationError>
where
    SelectedKind: SignalKind,
    SelectedKind::Event: Send + Sync,
    SystemOutput: Into<Option<()>> + 'static,
    System: IntoSystem<In<SelectedKind::Event>, SystemOutput, Marker> + Send + Sync + 'static,
{
    struct PlannedCandidate {
        binding: Option<Arc<OnceLock<DynamicAdapterBinding>>>,
        adapter: Option<PlannedSignalEntityRef>,
    }

    let owner_epoch = crate::dynamic::DynamicOwnerEpoch::capture(graph.registration_world(), owner);
    let mut plan = PendingSignalMaterialization::new(graph);
    let planning = (|| {
        let mut planned = Vec::with_capacity(candidates.len());
        let mut registrar = SignalRegistrar::new(&mut plan);
        let (owner_tag, _) = registrar.plan.graph.registration_existing_schedule(owner)?;
        for candidate in candidates {
            let selected = candidate.selected.materialize_consuming_into(&mut registrar)?;
            if !registrar.planned_replayable(selected.planned)? {
                return Err(SignalRegistrationError::DynamicReplayContractMismatch);
            }
            if same_identity == crate::dynamic::SameSignalIdentity::KeepCurrent
                && matches!(selected.planned, PlannedSignalEntityRef::Existing(entity) if Some(entity) == candidate.current_selected)
            {
                planned.push(PlannedCandidate {
                    binding: None,
                    adapter: None,
                });
                continue;
            }

            let binding = Arc::new(OnceLock::new());
            let adapter =
                registrar.signal_entity(selected, make_adapter(binding.clone()), OutputDelivery::move_only())?;
            if let Some(schedule) = owner_tag {
                registrar.apply_planned_schedule(adapter.planned, schedule)?;
            }
            registrar.install_entity_component(adapter, {
                let binding = binding.clone();
                let slot = candidate.slot;
                move |_world, adapter, dependencies| {
                    let [selected] = dependencies else {
                        panic!("dynamic adapter committed without exactly one canonical selected dependency");
                    };
                    binding
                        .set(DynamicAdapterBinding {
                            owner,
                            selected: *selected,
                            adapter,
                            slot,
                        })
                        .unwrap_or_else(|_| panic!("dynamic adapter binding was published more than once"));
                }
            })?;
            registrar.plan_dynamic_connection(
                adapter,
                SignalEntityRef::<ScalarSignalKind<()>>::new(PlannedSignalEntityRef::Existing(owner)),
            );
            registrar.plan.claim_deltas.push(PlannedClaimDelta {
                entity: adapter.planned,
                increment: 1,
            });
            registrar.plan.private_claims.push(adapter.planned);
            planned.push(PlannedCandidate {
                binding: Some(binding),
                adapter: Some(adapter.planned),
            });
        }
        Ok(planned)
    })();
    let planned = match planning {
        Ok(planned) => planned,
        Err(error) => {
            plan.reject(error);
            return Err(error);
        }
    };
    if let Err(error) = plan.plan_replay_maintenance() {
        plan.reject(error);
        return Err(error);
    }
    let terminal = planned
        .iter()
        .find_map(|candidate| candidate.adapter)
        .unwrap_or(PlannedSignalEntityRef::Existing(owner));
    let preflight = match plan.preflight(terminal, true) {
        Ok(preflight) => preflight,
        Err(error) => {
            plan.reject(error);
            return Err(error);
        }
    };
    // KeepCurrent skips adapter ownership, not materializer publication or other validated plan
    // effects.
    let committed = plan.commit(terminal, false, owner_epoch.map(|epoch| (epoch, preflight)));
    if committed.retired {
        assert!(
            committed.dynamic_connections.is_empty(),
            "retired invocation acquired a dynamic connection"
        );
        return Ok(DynamicCandidateMaterialization::Retired {
            claims: committed.private_claims,
            cleanup: committed.cleanup,
        });
    }
    let mut claims = committed.private_claims.into_iter();
    let mut connections = committed.dynamic_connections.into_iter();
    let mut result = Vec::with_capacity(planned.len());
    for candidate in planned {
        let Some(binding) = candidate.binding else {
            result.push(CommittedDynamicCandidate::KeptCurrent);
            continue;
        };
        let adapter_claim = claims
            .next()
            .unwrap_or_else(|| panic!("dynamic candidate commit lost a private adapter claim"));
        let connection = connections
            .next()
            .unwrap_or_else(|| panic!("dynamic candidate commit lost a connection"));
        let binding = binding
            .get()
            .unwrap_or_else(|| panic!("dynamic adapter commit did not publish its binding"));
        assert_eq!(binding.adapter, adapter_claim.entity());
        result.push(CommittedDynamicCandidate::New {
            selected: binding.selected,
            slot: binding.slot,
            adapter_claim,
            connection,
        });
    }
    assert!(
        claims.next().is_none(),
        "dynamic candidate commit published an extra private claim"
    );
    assert!(
        connections.next().is_none(),
        "dynamic candidate commit published an extra connection"
    );
    Ok(DynamicCandidateMaterialization::Live {
        candidates: result,
        cleanup: committed.cleanup,
    })
}

#[cfg(test)]
pub(crate) fn verify_existing_canonical_with_graph(
    graph: &mut dyn SignalRegistrationGraphHooks,
    entity: SignalEntity,
    dependencies: Box<[SignalEntity]>,
    replayable: bool,
    output_delivery: Option<OutputDeliveryKey>,
) -> Result<(), SignalRegistrationError> {
    let mut plan = PendingSignalMaterialization::new(graph);
    plan.existing_expectations.push(ExistingCanonicalExpectation {
        entity,
        dependencies: Some(dependencies),
        replayable: Some(replayable),
        output_delivery,
    });
    match plan.preflight(PlannedSignalEntityRef::Existing(entity), true) {
        Ok(_) => {
            plan.outcome = PlanOutcome::Committed;
            Ok(())
        }
        Err(error) => {
            plan.reject(error);
            Err(error)
        }
    }
}

struct MaterializerPanicGuard {
    state: Arc<SignalMaterializerState>,
    plan_id: MaterializationPlanId,
    armed: bool,
}

impl Drop for MaterializerPanicGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = lock_materializer_state(&self.state);
        if matches!(
            &*state,
            SharedMaterializerState::Initializing { plan_id, .. }
                if *plan_id == self.plan_id
        ) {
            *state = SharedMaterializerState::Panicked;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use bevy_ecs::entity::Entity;

    struct TestGraph {
        world: World,
        existing: BTreeMap<u64, ExistingSignalEntityRegistration>,
        schedules: Vec<InternedScheduleLabel>,
        commits: usize,
    }

    impl Default for TestGraph {
        fn default() -> Self {
            Self {
                world: World::new(),
                existing: BTreeMap::new(),
                schedules: Vec::new(),
                commits: 0,
            }
        }
    }

    impl SignalRegistrationGraphHooks for TestGraph {
        fn registration_world(&mut self) -> &mut World {
            &mut self.world
        }

        fn registration_existing_signal(&self, entity: SignalEntity) -> Option<ExistingSignalEntityRegistration> {
            self.existing.get(&entity.entity().to_bits()).cloned()
        }

        fn registration_preflight_existing_runtimes(
            &self,
            _entities: &[SignalEntity],
        ) -> Result<(), SignalRegistrationError> {
            Ok(())
        }

        fn registration_schedule_is_registered(&self, schedule: InternedScheduleLabel) -> bool {
            self.schedules.contains(&schedule)
        }

        fn registration_existing_schedule(
            &self,
            entity: SignalEntity,
        ) -> Result<(Option<InternedScheduleLabel>, Option<InternedScheduleLabel>), SignalRegistrationError> {
            self.existing
                .contains_key(&entity.entity().to_bits())
                .then_some((None, None))
                .ok_or(SignalRegistrationError::EntityUnavailable)
        }

        fn registration_claim_count(&self, entity: SignalEntity) -> Option<usize> {
            self.existing.contains_key(&entity.entity().to_bits()).then_some(0)
        }

        fn registration_replay_maintenance_armed(&self, entity: SignalEntity) -> Option<bool> {
            self.existing.contains_key(&entity.entity().to_bits()).then_some(false)
        }

        fn registration_preflight_canonical_state(
            &self,
            _entity: &RegistrationEntityPreflight,
        ) -> Result<(), SignalRegistrationError> {
            Ok(())
        }

        fn registration_preflight_reserved_output_policy(
            &self,
            entity: &RegistrationEntityPreflight,
        ) -> Result<(), SignalRegistrationError> {
            if let Some(pending_delivery) = entity.pending_output_delivery
                && !pending_delivery.is_coherent(entity.output_type)
            {
                return Err(SignalRegistrationError::ConflictingCanonicalState);
            }
            if let PlannedSignalEntityRef::Existing(existing_entity) = entity.entity
                && let Some(expected_key) = entity.expected_output_delivery
                && let Some(existing) = self.existing.get(&existing_entity.entity().to_bits())
                && let Some(actual_key) = existing.output_delivery
                && actual_key != expected_key
            {
                return Err(SignalRegistrationError::ConflictingCanonicalState);
            }
            Ok(())
        }

        fn registration_preflight_dynamic_connections(
            &self,
            _plan: &RegistrationPreflight,
        ) -> Result<(), SignalRegistrationError> {
            Ok(())
        }

        fn registration_preflight_activation(
            &self,
            _plan: &RegistrationPreflight,
        ) -> Result<RegistrationActivationPreflight, SignalRegistrationError> {
            Ok(RegistrationActivationPreflight {
                count_increments: Box::new([]),
                active_routes: Box::new([]),
                effective_membership: Box::new([]),
                activated: Box::new([]),
            })
        }

        fn registration_commit_signal_entity(&mut self, entity: RegistrationEntityCommit) {
            let input_type = entity.runtime.input_type();
            let output_type = entity.runtime.output_type();
            let output_delivery = Some(entity.runtime.output_delivery().key());
            self.existing.insert(
                entity.entity.entity().to_bits(),
                ExistingSignalEntityRegistration {
                    ordered_input_consumption: entity.runtime.ordered_input_consumption(),
                    dependencies: entity.dependencies,
                    replayable: false,
                    input_type,
                    output_type,
                    output_delivery,
                },
            );
            self.commits += 1;
        }

        fn registration_commit_replay(&mut self, replay: RegistrationReplayCommit) {
            if replay.replay.is_replayable()
                && let Some(existing) = self.existing.get_mut(&replay.entity.entity().to_bits())
            {
                existing.replayable = true;
            }
        }

        fn registration_commit_materialized_dependency(&mut self, _link: RegistrationMaterializedDependencyCommit) {}

        fn registration_commit_schedule(&mut self, _schedule: RegistrationScheduleCommit) {}

        fn registration_commit_activation(&mut self, _activation: RegistrationActivationCommit) {}

        fn registration_commit_dynamic_connection(
            &mut self,
            connection: RegistrationDynamicConnectionCommit,
        ) -> DynamicSignalConnection {
            DynamicSignalConnection::new(connection.source, connection.target)
        }

        fn registration_commit_replay_maintenance(&mut self, _terminal: SignalEntity) {}

        fn registration_commit_terminal_retainer(&mut self, _retainer: RegistrationTerminalRetainerCommit) {}

        fn registration_publish_initial_claim(&mut self, _terminal: SignalEntity) -> SignalClaim {
            panic!("claim publication is not exercised by registration foundation tests")
        }
    }

    #[test]
    fn ordered_input_consumption_accepts_only_the_two_private_shapes() {
        let manager = OrderedInputConsumption::ManagerBridge;
        let chain = OrderedInputConsumption::chain::<u8>();
        let input_type = TypeId::of::<ChainInput<u8>>();
        let output_type = TypeId::of::<Vec<VecDiff<u8>>>();
        for dependency_count in 0..4 {
            assert_eq!(
                manager.accepts_shape(dependency_count, TypeId::of::<u8>(), TypeId::of::<()>(), false),
                dependency_count == 1
            );
            assert_eq!(
                chain.accepts_shape(dependency_count, input_type, output_type, false),
                dependency_count == 2
            );
        }
        assert!(!manager.accepts_shape(1, input_type, output_type, false));
        assert!(!manager.accepts_shape(1, input_type, TypeId::of::<()>(), true));
        assert!(!chain.accepts_shape(2, input_type, output_type, true));
        assert!(!chain.accepts_shape(2, TypeId::of::<ChainInput<u16>>(), output_type, false));
        assert!(!chain.accepts_shape(2, input_type, TypeId::of::<Vec<VecDiff<u16>>>(), false));
        assert!(chain.accepts_dependency(input_type, false, None));
        assert!(!chain.accepts_dependency(input_type, true, None));
        assert!(!chain.accepts_dependency(input_type, false, Some(manager)));
        assert!(!chain.accepts_dependency(TypeId::of::<u8>(), false, None));
        assert!(manager.accepts_dependency(input_type, true, None));
    }

    #[test]
    fn ordered_chain_constructor_marks_only_the_new_typed_consumer() {
        let mut graph = TestGraph::default();
        let mut plan = PendingSignalMaterialization::new(&mut graph);
        let consumer = {
            let mut registrar = SignalRegistrar::new(&mut plan);
            let left = registrar
                .signal_entity(
                    (),
                    |In(())| Some(ChainInput::<u8>::Left(Vec::new())),
                    OutputDelivery::move_only(),
                )
                .expect("left tagger");
            let right = registrar
                .signal_entity(
                    (),
                    |In(())| Some(ChainInput::<u8>::Right(Vec::new())),
                    OutputDelivery::move_only(),
                )
                .expect("right tagger");
            registrar
                .ordered_chain_consumer(
                    [left, right],
                    |input, _epoch| match input {
                        ChainInput::Left(diffs) | ChainInput::Right(diffs) => Some(diffs),
                    },
                    OutputDelivery::move_only(),
                )
                .expect("chain consumer")
                .planned
        };
        assert_eq!(plan.entities.len(), 3);
        for pending in &plan.entities {
            let is_consumer = pending.reference == consumer;
            assert_eq!(
                pending.runtime.ordered_input_consumption(),
                is_consumer.then_some(OrderedInputConsumption::chain::<u8>())
            );
            assert_eq!(pending.runtime.collection_poll_append().is_some(), is_consumer);
            assert!(!pending.replay.is_replayable());
        }
    }

    #[test]
    fn signal_kind_events_are_exact() {
        assert_eq!(
            TypeId::of::<<ScalarSignalKind<u8> as SignalKind>::Event>(),
            TypeId::of::<u8>()
        );
        assert_eq!(
            TypeId::of::<<VecSignalKind<u8> as SignalKind>::Event>(),
            TypeId::of::<Vec<VecDiff<u8>>>()
        );
        assert_eq!(
            TypeId::of::<<MapSignalKind<u8, u16> as SignalKind>::Event>(),
            TypeId::of::<Vec<MapDiff<u8, u16>>>()
        );
    }

    #[test]
    fn homogeneous_inputs_are_deterministically_canonicalized() {
        let plan_id = MaterializationPlanId::next();
        let first = SignalEntityRef::<ScalarSignalKind<u8>>::new(PlannedSignalEntityRef::Pending { plan_id, index: 2 });
        let second =
            SignalEntityRef::<ScalarSignalKind<u8>>::new(PlannedSignalEntityRef::Pending { plan_id, index: 1 });
        let mut inputs = sealed::RegistrarInputs::into_planned_inputs(vec![first, second, first]);
        inputs.sort_unstable();
        inputs.dedup();
        assert_eq!(
            inputs,
            vec![
                PlannedSignalEntityRef::Pending { plan_id, index: 1 },
                PlannedSignalEntityRef::Pending { plan_id, index: 2 },
            ]
        );
    }

    #[test]
    fn simultaneously_live_plans_have_distinct_ids_registration_ownership_observability() {
        let mut first_graph = TestGraph::default();
        let mut second_graph = TestGraph::default();
        let first_plan = PendingSignalMaterialization::new(&mut first_graph);
        let mut second_plan = PendingSignalMaterialization::new(&mut second_graph);

        assert_ne!(first_plan.id(), second_plan.id());
        let foreign_reference = PlannedSignalEntityRef::Pending {
            plan_id: first_plan.id(),
            index: 0,
        };
        assert!(!second_plan.pending_reference_exists(foreign_reference));
        assert!(matches!(
            second_plan.preflight(foreign_reference, false),
            Err(SignalRegistrationError::EntityUnavailable)
        ));
        drop(second_plan);
        assert_eq!(second_graph.commits, 0);
    }

    #[test]
    fn rejected_materializer_is_permanently_rejected() {
        let materializer =
            ScalarSignalMaterializer::<u8>::new(|_registrar| Err(SignalRegistrationError::ConflictingCanonicalState));
        let shared = materializer.shared_clone();
        let mut graph = TestGraph::default();

        assert_eq!(
            materializer.materialize_with_graph(&mut graph),
            Err(SignalRegistrationError::ConflictingCanonicalState)
        );
        assert_eq!(
            shared.materialize_with_graph(&mut graph),
            Err(SignalRegistrationError::SharedMaterializerRejected)
        );
        assert_eq!(graph.commits, 0);
    }

    #[test]
    fn panic_is_remembered_by_shared_materializer() {
        let materializer = ScalarSignalMaterializer::<u8>::new(|_registrar| panic!("initializer panic"));
        let shared = materializer.shared_clone();
        let mut graph = TestGraph::default();

        let panic = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            let _result = materializer.materialize_with_graph(&mut graph);
        }));
        assert!(panic.is_err());
        assert_eq!(
            shared.materialize_with_graph(&mut graph),
            Err(SignalRegistrationError::SharedMaterializerPanicked)
        );
        assert_eq!(graph.commits, 0);
    }

    #[test]
    fn unavailable_existing_dependency_rejects_before_commit() {
        let missing: SignalEntity = Entity::from_raw_u32(17).unwrap().into();
        let materializer = ScalarSignalMaterializer::<u8>::new(move |_registrar| {
            Ok(SignalEntityRef::new(PlannedSignalEntityRef::Existing(missing)))
        });
        let mut graph = TestGraph::default();

        assert_eq!(
            materializer.materialize_with_graph(&mut graph),
            Err(SignalRegistrationError::EntityUnavailable)
        );
        assert_eq!(graph.commits, 0);
    }

    #[test]
    fn unit_and_homogeneous_inputs_are_distinct_registration_ownership_observability() {
        let materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
            let first = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?;
            let second = registrar.signal_entity((), |In(())| Some(2), OutputDelivery::broadcast())?;
            let array = registrar.signal_entity(
                [first, second],
                |In(value): In<i32>| Some(value),
                OutputDelivery::broadcast(),
            )?;
            let vector = registrar.signal_entity(
                vec![first, second],
                |In(value): In<i32>| Some(value),
                OutputDelivery::broadcast(),
            )?;
            let boxed = registrar.signal_entity(
                vec![array, vector].into_boxed_slice(),
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )?;
            Ok(boxed)
        });
        let mut graph = TestGraph::default();

        materializer
            .materialize_with_graph(&mut graph)
            .expect("all supported homogeneous input containers should materialize");
        assert_eq!(graph.commits, 5);
    }

    #[test]
    fn empty_homogeneous_inputs_reject_before_allocation_registration_ownership_observability() {
        let materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity(
                Vec::<ScalarSignalEntityRef<'_, i32>>::new(),
                |In(value): In<i32>| Some(value),
                OutputDelivery::move_only(),
            )
        });
        let mut graph = TestGraph::default();
        let entities_before = graph.world.entities().len();

        assert_eq!(
            materializer.materialize_with_graph(&mut graph),
            Err(SignalRegistrationError::EmptyRegistrarInputs)
        );
        assert_eq!(graph.world.entities().len(), entities_before);
        assert_eq!(graph.commits, 0);
    }

    #[test]
    fn same_attempt_recursion_rejects_and_discards_candidates_registration_ownership_observability() {
        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, AtomicOrdering::SeqCst);
            }
        }

        let recursive = Arc::new(OnceLock::<ScalarSignalMaterializer<i32>>::new());
        let recursive_for_initializer = recursive.clone();
        let drops = Arc::new(AtomicUsize::new(0));
        let probe = DropProbe(drops.clone());
        let materializer = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let _candidate: ScalarSignalEntityRef<'_, i32> = registrar.signal_entity(
                (),
                move |In(())| {
                    core::hint::black_box(&probe);
                    Some(1)
                },
                OutputDelivery::move_only(),
            )?;
            registrar.materialize_signal(
                recursive_for_initializer
                    .get()
                    .expect("recursive materializer should be installed")
                    .shared_clone(),
            )
        });
        recursive
            .set(materializer.shared_clone())
            .unwrap_or_else(|_| panic!("recursive materializer should be installed once"));
        let mut graph = TestGraph::default();
        let entities_before = graph.world.entities().len();

        assert_eq!(
            materializer.materialize_with_graph(&mut graph),
            Err(SignalRegistrationError::CycleDetected)
        );
        assert_eq!(graph.world.entities().len(), entities_before);
        assert_eq!(graph.commits, 0);
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn ready_materializer_reuses_identity_without_recommit_registration_ownership_observability() {
        let materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let shared = materializer.shared_clone();
        let mut graph = TestGraph::default();

        let first = materializer
            .materialize_with_graph(&mut graph)
            .expect("first materialization should succeed");
        let second = shared
            .materialize_with_graph(&mut graph)
            .expect("ready materializer should be reusable");

        assert_eq!(first, second);
        assert_eq!(graph.commits, 1);
    }

    #[test]
    fn ready_side_materialization_participates_in_preflight_registration_ownership_observability() {
        let side = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
        });
        let side_for_plan = side.shared_clone();
        let mut graph = TestGraph::default();
        side.materialize_with_graph(&mut graph)
            .expect("side identity should materialize");

        let terminal = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let _side = registrar.materialize_signal(side_for_plan)?;
            registrar.signal_entity((), |In(())| Some(2), OutputDelivery::move_only())
        });
        terminal
            .materialize_with_graph(&mut graph)
            .expect("an existing side materialization should be included in preflight inventory");

        assert_eq!(graph.commits, 2);
    }

    fn assert_keep_current_alias_runtime(keep_original: bool) {
        use crate::{
            JonmoPlugin,
            dynamic::*,
            graph::graph_snapshot,
            signal::{Signal, SignalExt},
        };
        use bevy_app::App;

        let mut app = App::new();
        app.add_plugins(JonmoPlugin::default());
        app.update();
        let selected = crate::signal::once(41).cache().broadcast();
        let initial = selected.clone().boxed_replayable();
        let owner = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let mut initial = Some(initial);
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    initial.take().map(|signal| {
                        DynamicManagerUpdate::new([ScalarSlotCommand::Select { signal }], DynamicEmissions::<i32>::None)
                    })
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
        .register(app.world_mut());
        for _ in 0..4 {
            app.update();
        }
        let before = graph_snapshot(app.world_mut());
        let subscriptions = app
            .world()
            .get::<DynamicSignalOwner>(owner.entity().entity())
            .expect("dynamic owner")
            .subscriptions();
        let [subscription] = subscriptions.as_slice() else {
            panic!("expected one current subscription");
        };
        let selected_node = before.node(subscription.selected.entity()).expect("selected node");
        assert!(selected_node.replay_maintenance_armed);
        assert_eq!(selected_node.replay_relationships.len(), 1);
        assert!(matches!(
            selected_node
                .replay_relationships
                .first()
                .expect("selected cursor")
                .state,
            crate::graph::ReplayRelationshipSnapshotState::Published { .. }
        ));
        let selected_for_alias = selected.clone().boxed_replayable();
        let selected = keep_original.then(|| selected.boxed_replayable());
        let alias = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            registrar.materialize_signal(Box::new(selected_for_alias).into_signal_materializer())
        });
        let retained_alias = alias.shared_clone();
        let inbox_before = cleanup::queued_message_count(app.world());
        let mut adapter_constructions = 0;
        let result = materialize_dynamic_candidates(
            app.world_mut(),
            vec![DynamicCandidateRequest {
                selected: alias,
                slot: DynamicStableSlot {
                    identity: subscription.slot.identity,
                    generation: subscription.slot.generation + 1,
                },
                current_selected: Some(subscription.selected),
            }],
            owner.entity(),
            SameSignalIdentity::KeepCurrent,
            |_| {
                adapter_constructions += 1;
                |In(_): In<i32>| Some(())
            },
        )
        .expect("all-KeepCurrent publication");
        let DynamicCandidateMaterialization::Live { candidates, cleanup } = result else {
            panic!("KeepCurrent owner was not retired");
        };
        assert!(matches!(
            candidates.as_slice(),
            [CommittedDynamicCandidate::KeptCurrent]
        ));
        cleanup.finish(app.world_mut());
        assert_eq!(adapter_constructions, 0);
        assert!(matches!(
            &*lock_materializer_state(&retained_alias.state),
            SharedMaterializerState::Ready { cache, .. } if cache.terminal == subscription.selected
        ));
        let after = graph_snapshot(app.world_mut());
        let mut expected = before;
        let selected_node = expected
            .nodes
            .iter_mut()
            .find(|node| node.entity == subscription.selected.entity())
            .expect("selected node");
        if keep_original {
            selected_node.terminal_retainer_count += 1;
            selected_node.live_terminal_retainer_count += 1;
        }
        assert_eq!(after, expected);
        assert_eq!(cleanup::queued_message_count(app.world()), inbox_before);

        for candidates in [
            vec![DynamicCandidateRequest {
                selected: retained_alias.shared_clone(),
                slot: subscription.slot,
                current_selected: Some(subscription.selected),
            }],
            Vec::new(),
        ] {
            let result = materialize_dynamic_candidates(
                app.world_mut(),
                candidates,
                owner.entity(),
                SameSignalIdentity::KeepCurrent,
                |_| {
                    adapter_constructions += 1;
                    |In(_): In<i32>| Some(())
                },
            )
            .expect("ready/empty no-new-claims plan");
            let DynamicCandidateMaterialization::Live { candidates, cleanup } = result else {
                panic!("KeepCurrent owner was not retired");
            };
            cleanup.finish(app.world_mut());
            assert!(
                candidates
                    .iter()
                    .all(|candidate| matches!(candidate, CommittedDynamicCandidate::KeptCurrent))
            );
            assert_eq!(adapter_constructions, 0);
            assert_eq!(graph_snapshot(app.world_mut()), after);
            assert_eq!(cleanup::queued_message_count(app.world()), inbox_before);
        }
        owner.try_release(app.world_mut()).expect("owner release");
        drop(retained_alias);
        drop(selected);
        for _ in 0..4 {
            app.update();
        }
        assert!(graph_snapshot(app.world_mut()).nodes.is_empty());
    }

    #[test]
    fn keep_current_alias_preserves_exact_runtime_registration_lifecycle() {
        assert_keep_current_alias_runtime(true);
    }

    #[test]
    fn keep_current_alias_takes_over_final_retainer_registration_lifecycle() {
        assert_keep_current_alias_runtime(false);
    }

    #[test]
    fn keep_current_commits_side_effects_registration_lifecycle() {
        use crate::{JonmoPlugin, dynamic::SameSignalIdentity, graph::graph_snapshot};
        use bevy_app::{App, Last};

        let mut app = App::new();
        app.add_plugins(JonmoPlugin::default());
        app.update();
        let selected = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(41, |In(())| None::<i32>, OutputDelivery::broadcast())
        });
        let selected_claim = selected.shared_clone().register(app.world_mut());
        let owner = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(7), OutputDelivery::broadcast())
        })
        .register(app.world_mut());
        let unused_probe = Arc::new(());
        let side_probe = Arc::new(());
        let side_runs = Arc::new(AtomicUsize::new(0));
        let side = ScalarSignalMaterializer::<i32>::new({
            let side_probe = side_probe.clone();
            let side_runs = side_runs.clone();
            move |registrar| {
                registrar.replayable_signal_source(
                    0,
                    move |In(())| {
                        side_runs.fetch_add(1, AtomicOrdering::SeqCst);
                        Some(Arc::strong_count(&side_probe) as i32)
                    },
                    OutputDelivery::broadcast(),
                )
            }
        });
        let alias = ScalarSignalMaterializer::<i32>::new({
            let selected = selected.shared_clone();
            let side = side.shared_clone();
            let unused_probe = unused_probe.clone();
            move |registrar| {
                let _side = registrar.materialize_signal(side)?;
                let unused = registrar.signal_entity(
                    (),
                    move |In(())| Some(Arc::strong_count(&unused_probe) as i32),
                    OutputDelivery::move_only(),
                )?;
                let _unused: ScalarSignalEntityRef<'_, i32> =
                    registrar.signal_entity(unused, |In(value): In<i32>| Some(value), OutputDelivery::move_only())?;
                let selected = registrar.materialize_signal(selected)?;
                registrar.apply_schedule(selected, Last)
            }
        });
        let retained_alias = alias.shared_clone();
        let before = graph_snapshot(app.world_mut());
        let mut adapter_constructions = 0;
        let result = materialize_dynamic_candidates(
            app.world_mut(),
            vec![DynamicCandidateRequest {
                selected: alias,
                slot: DynamicStableSlot {
                    identity: 0,
                    generation: 1,
                },
                current_selected: Some(selected_claim.entity()),
            }],
            owner.entity(),
            SameSignalIdentity::KeepCurrent,
            |_| {
                adapter_constructions += 1;
                |In(_): In<i32>| Some(())
            },
        )
        .expect("side-effectful all-KeepCurrent plan");
        let DynamicCandidateMaterialization::Live { candidates, cleanup } = result else {
            panic!("KeepCurrent owner was not retired");
        };
        assert!(matches!(
            candidates.as_slice(),
            [CommittedDynamicCandidate::KeptCurrent]
        ));
        cleanup.finish(app.world_mut());
        assert_eq!(adapter_constructions, 0);
        assert_eq!(Arc::strong_count(&unused_probe), 1);
        assert_eq!(Arc::strong_count(&side_probe), 2);
        assert_eq!(
            app.world().registration_existing_schedule(selected_claim.entity()),
            Ok((Some(Last.intern()), Some(Last.intern())))
        );
        let side_terminal = match &*lock_materializer_state(&side.state) {
            SharedMaterializerState::Ready { cache, .. } => cache.terminal,
            _ => panic!("independent side materializer must be published Ready"),
        };
        let after = graph_snapshot(app.world_mut());
        assert_eq!(after.nodes.len(), before.nodes.len() + 1);
        let side_node = after.node(side_terminal.entity()).expect("retained side entity");
        assert_eq!(side_node.claim_count, Some(1));
        assert!(side_node.replay_maintenance_armed);
        assert_eq!(side_node.live_terminal_retainer_count, 1);
        assert_eq!(
            after
                .node(selected_claim.entity().entity())
                .expect("selected node")
                .claim_count,
            before
                .node(selected_claim.entity().entity())
                .expect("selected node")
                .claim_count
        );
        assert_eq!(side_runs.load(AtomicOrdering::SeqCst), 0);
        for _ in 0..4 {
            app.update();
        }
        assert!(side_runs.load(AtomicOrdering::SeqCst) > 0);
        owner.try_release(app.world_mut()).expect("owner release");
        selected_claim.try_release(app.world_mut()).expect("selected release");
        drop(retained_alias);
        drop(selected);
        drop(side);
        for _ in 0..4 {
            app.update();
        }
        assert_eq!(Arc::strong_count(&side_probe), 1);
        assert!(graph_snapshot(app.world_mut()).nodes.is_empty());
    }

    #[test]
    fn committed_move_only_replayable_rejects_post_commit_broadcast_normalization() {
        use crate::graph::graph_snapshot;
        use bevy_app::App;

        let mut app = App::new();
        app.add_plugins(crate::JonmoPlugin::default());

        let linear_replay = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(7, |In(())| None::<i32>, OutputDelivery::move_only())
        });
        let retained = linear_replay.shared_clone();
        let claim = linear_replay.register(app.world_mut());
        let before = graph_snapshot(app.world_mut());

        let alias = retained.shared_clone();
        let conflict = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let existing = registrar.materialize_signal(alias)?;
            let normalized = registrar
                .scalar_broadcast_replayable_terminal(existing)?
                .expect("replayable terminal");
            Ok(normalized)
        })
        .try_register(app.world_mut());

        assert_eq!(
            conflict.unwrap_err(),
            SignalRegistrationError::ConflictingCanonicalState
        );
        assert_eq!(graph_snapshot(app.world_mut()), before);

        drop(retained);
        claim.release(app.world_mut());
    }

    #[test]
    fn existing_move_only_entity_rejects_second_consumer_and_allows_dormant_reactivation() {
        use crate::graph::graph_snapshot;
        use bevy_app::App;

        let mut app = App::new();
        app.add_plugins(crate::JonmoPlugin::default());

        let source = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(10), OutputDelivery::move_only())
        });
        let retained_source = source.shared_clone();

        let dormant_source = retained_source.shared_clone();
        let dormant = ScalarSignalMaterializer::<()>::new(move |registrar| {
            let source = registrar.materialize_signal(dormant_source)?;
            registrar.signal_entity(source, |In(_): In<i32>| None::<()>, OutputDelivery::move_only())
        });
        let retained_dormant = dormant.shared_clone();
        let temp_claim = dormant.register(app.world_mut());
        let dormant_entity = temp_claim.entity();
        temp_claim.release(app.world_mut());

        let active_source = retained_source.shared_clone();
        let active = ScalarSignalMaterializer::<()>::new(move |registrar| {
            let source = registrar.materialize_signal(active_source)?;
            registrar.signal_entity(
                [source, source],
                |In(_): In<i32>| None::<()>,
                OutputDelivery::move_only(),
            )
        });
        let active_claim = active.register(app.world_mut());
        let before = graph_snapshot(app.world_mut());

        let rejected = retained_dormant.shared_clone().try_register(app.world_mut());
        assert_eq!(rejected.unwrap_err(), SignalRegistrationError::MoveOnlyFanout);
        assert_eq!(graph_snapshot(app.world_mut()), before);

        active_claim.release(app.world_mut());
        let reactivated = retained_dormant
            .try_register(app.world_mut())
            .expect("dormant dependent must reactivate after active dependent releases");
        assert_eq!(reactivated.entity(), dormant_entity);
        reactivated.release(app.world_mut());
        drop(retained_source);
        drop(source);
    }

    #[test]
    fn existing_warming_move_only_replayable_and_dynamic_slots_enforce_move_only_fanout_and_try_broadcast_fallback() {
        use crate::{DynamicSignalOperatorSpec, SameSignalIdentity, graph::graph_snapshot};
        use bevy_app::App;

        let mut app = App::new();
        app.add_plugins(crate::JonmoPlugin::default());

        // 1. Existing Warming move_only() replayable entity with 1 Pending replay consumer rejects
        //    a second Pending replay consumer immediately with MoveOnlyFanout, while fallback
        //    `scalar_try_broadcast_terminal` declines in-place mutation (`Ok(None)`).
        let warming = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(())| None::<i32>,
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        let retained_warming = warming.shared_clone();
        let first_consumer_src = retained_warming.shared_clone();
        let first_consumer = ScalarSignalMaterializer::<()>::new(move |registrar| {
            let src = registrar.materialize_signal(first_consumer_src)?;
            registrar.signal_entity(src, |In(_): In<i32>| None::<()>, OutputDelivery::move_only())
        })
        .register(app.world_mut());
        let before_second = graph_snapshot(app.world_mut());

        let second_consumer_src = retained_warming.shared_clone();
        let second_rejected = ScalarSignalMaterializer::<()>::new(move |registrar| {
            let src = registrar.materialize_signal(second_consumer_src)?;
            registrar.signal_entity(src, |In(_): In<i32>| None::<()>, OutputDelivery::move_only())
        })
        .try_register(app.world_mut());
        assert_eq!(second_rejected.unwrap_err(), SignalRegistrationError::MoveOnlyFanout);
        assert_eq!(graph_snapshot(app.world_mut()), before_second);

        first_consumer.release(app.world_mut());

        // Fallback `scalar_try_broadcast_terminal` on an Existing Move replayable returns `None`,
        // allowing `SignalExt::broadcast` to append a downstream identity broadcast entity cleanly.
        let fallback_src = retained_warming.shared_clone();
        let fallback_broadcast = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let existing = registrar.materialize_signal(fallback_src)?;
            assert!(registrar.scalar_try_broadcast_terminal(existing)?.is_none());
            registrar.signal_entity(existing, |In(v): In<i32>| Some(v), OutputDelivery::broadcast())
        });
        let fb_1 = fallback_broadcast.shared_clone().register(app.world_mut());
        let fb_2 = fallback_broadcast.register(app.world_mut());
        assert_eq!(fb_1.entity(), fb_2.entity());
        fb_1.release(app.world_mut());
        fb_2.release(app.world_mut());
        drop(retained_warming);
        drop(warming);

        // 2. Dynamic candidate materialization rejects both cross-slot move_only() collisions (slot
        //    0 and slot 1 selecting the same move_only() replayable entity) and
        //    external-consumer-vs-dynamic-slot move_only() collisions.
        type IndexedUpdate = crate::DynamicManagerUpdate<
            crate::IndexedSlotCommand<crate::signal::BoxedReplayableSignal<i32>>,
            (usize, i32),
        >;
        let owner = ScalarSignalMaterializer::<Vec<(usize, i32)>>::new(|registrar| {
            let manager = registrar.signal_entity((), |In(())| None::<IndexedUpdate>, OutputDelivery::move_only())?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| crate::DynamicEmissions::One((index, value)))
                    .scalar_output(Some)
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
        .register(app.world_mut());

        let candidate = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(11, |In(())| None::<i32>, OutputDelivery::move_only())
        });
        let before_dynamic = graph_snapshot(app.world_mut());

        let cross_slot = materialize_dynamic_candidates(
            app.world_mut(),
            vec![
                DynamicCandidateRequest {
                    selected: candidate.shared_clone(),
                    slot: DynamicStableSlot {
                        identity: 0,
                        generation: 1,
                    },
                    current_selected: None,
                },
                DynamicCandidateRequest {
                    selected: candidate,
                    slot: DynamicStableSlot {
                        identity: 1,
                        generation: 1,
                    },
                    current_selected: None,
                },
            ],
            owner.entity(),
            SameSignalIdentity::KeepCurrent,
            |_| |In(_): In<i32>| Some(()),
        );
        assert!(matches!(cross_slot, Err(SignalRegistrationError::MoveOnlyFanout)));
        assert_eq!(graph_snapshot(app.world_mut()), before_dynamic);

        let committed_candidate = ScalarSignalMaterializer::<i32>::new(|registrar| {
            registrar.replayable_signal_source(12, |In(())| None::<i32>, OutputDelivery::move_only())
        });
        let retained_committed = committed_candidate.shared_clone();
        let ext_src = retained_committed.shared_clone();
        let ext_claim = ScalarSignalMaterializer::<()>::new(move |registrar| {
            let src = registrar.materialize_signal(ext_src)?;
            registrar.signal_entity(src, |In(_): In<i32>| None::<()>, OutputDelivery::move_only())
        })
        .register(app.world_mut());
        let before_ext_collision = graph_snapshot(app.world_mut());

        let ext_vs_dynamic = materialize_dynamic_candidates(
            app.world_mut(),
            vec![DynamicCandidateRequest {
                selected: retained_committed.shared_clone(),
                slot: DynamicStableSlot {
                    identity: 0,
                    generation: 2,
                },
                current_selected: None,
            }],
            owner.entity(),
            SameSignalIdentity::KeepCurrent,
            |_| |In(_): In<i32>| Some(()),
        );
        assert!(matches!(ext_vs_dynamic, Err(SignalRegistrationError::MoveOnlyFanout)));
        assert_eq!(graph_snapshot(app.world_mut()), before_ext_collision);

        ext_claim.release(app.world_mut());
        let before_existing_cross_slot = graph_snapshot(app.world_mut());
        let existing_cross_slot = materialize_dynamic_candidates(
            app.world_mut(),
            vec![
                DynamicCandidateRequest {
                    selected: retained_committed.shared_clone(),
                    slot: DynamicStableSlot {
                        identity: 0,
                        generation: 3,
                    },
                    current_selected: None,
                },
                DynamicCandidateRequest {
                    selected: retained_committed,
                    slot: DynamicStableSlot {
                        identity: 1,
                        generation: 3,
                    },
                    current_selected: None,
                },
            ],
            owner.entity(),
            SameSignalIdentity::KeepCurrent,
            |_| |In(_): In<i32>| Some(()),
        );
        assert!(matches!(
            existing_cross_slot,
            Err(SignalRegistrationError::MoveOnlyFanout)
        ));
        assert_eq!(graph_snapshot(app.world_mut()), before_existing_cross_slot);

        owner.release(app.world_mut());
        drop(committed_candidate);
    }
}
