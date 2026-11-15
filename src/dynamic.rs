//! Typed dynamic signal operator declarations.
//!
//! Public code supplies only manager commands, selected-input adaptation, stable reduction, replay
//! policy, output kind, same-identity behavior, and optional candidate-error handling. Jonmo owns
//! subscriptions, fresh adapters, claims, connections, stable slots, generations, replay cursors,
//! replacement cohorts, aggregation queues, and cleanup.
//!
//! Each active manager-to-bridge relationship consumes every future manager output in order,
//! preserving each [`DynamicManagerUpdate`]'s validation and rejection boundary. Other scalar
//! consumers, scalar replay state, and scalar polling still select the last successful output.
//! Updates are queued for normal bridge execution, not interpreted inline by the producer.
//!
//! Attachment does not reconstruct earlier commands or commands lost upstream of the supplied
//! manager. A ready replayable manager provides one bootstrap update, validated normally before
//! future updates; warming provides no bootstrap. The latest scalar replay value is not implicitly
//! an authoritative slot-layout checkpoint, and intermediate commands do not advance that cache.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet, VecDeque},
    vec::Vec,
};
use bevy_ecs::{
    component::Component,
    lifecycle::HookContext,
    system::In,
    world::{DeferredWorld, World},
};
#[cfg(test)]
use bevy_platform::sync::Mutex;
use bevy_platform::sync::{Arc, OnceLock};
use core::marker::PhantomData;

use crate::{
    MapSignalKind, OutputDelivery, ReplayInitialization, ScalarSignalEntityRef, ScalarSignalKind, SignalEntityRef,
    SignalKind, SignalRegistrar, SignalRegistrationError, VecSignalKind,
    graph::{ErasedSignalValue, SignalClaim, SignalEntity},
    registration::StoredOutputDelivery,
    signal::ReplayableSignal,
    signal_map::ReplayableSignalMap,
    signal_vec::ReplayableSignalVec,
};

/// A scalar dynamic slot update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarSlotCommand<S> {
    /// Selects `signal`, superseding any older pending selection.
    Select {
        /// Replayable signal selected for the slot.
        signal: S,
    },
    /// Clears both the current selection and any pending selection.
    Clear,
}

/// An indexed dynamic slot update.
///
/// Commands in one manager update are validated and interpreted sequentially against the latest
/// accepted layout. Any invalid index rejects the complete update before candidate materialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexedSlotCommand<S> {
    /// Inserts a new slot before the item currently at `index`; `index == len` appends.
    Insert {
        /// Index in the desired layout at this point in the update.
        index: usize,
        /// Replayable signal selected for the new slot.
        signal: S,
    },
    /// Replaces the selection at an existing index while preserving its stable slot identity.
    Replace {
        /// Existing index in the desired layout at this point in the update.
        index: usize,
        /// Replayable replacement signal.
        signal: S,
    },
    /// Removes an existing slot.
    Remove {
        /// Existing index in the desired layout at this point in the update.
        index: usize,
    },
    /// Moves an existing slot without replacing its subscription or generation.
    Move {
        /// Existing source index at this point in the update.
        old_index: usize,
        /// Final destination index (`0..len`) where the moved slot will reside.
        new_index: usize,
    },
    /// Clears all slots.
    Clear,
    /// Atomically replaces all slots; an empty vector is a ready authoritative empty cohort.
    ReplaceAll {
        /// Replayable selections in replacement order.
        signals: Vec<S>,
    },
}

/// A keyed dynamic slot update.
///
/// Commands in one manager update are validated and interpreted sequentially against the latest
/// accepted layout. Any missing or duplicate key rejects the complete update before candidate
/// materialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyedSlotCommand<K, S> {
    /// Inserts a key that does not currently exist.
    Insert {
        /// New public key.
        key: K,
        /// Replayable signal selected for the key.
        signal: S,
    },
    /// Replaces the selection at an existing key while preserving its stable slot identity.
    Replace {
        /// Existing public key.
        key: K,
        /// Replayable replacement signal.
        signal: S,
    },
    /// Removes an existing key.
    Remove {
        /// Existing public key.
        key: K,
    },
    /// Renames an existing key without replacing its subscription or generation.
    Rekey {
        /// Existing public key.
        old_key: K,
        /// New public key, which must not already exist unless it equals `old_key`.
        new_key: K,
    },
    /// Clears all keys.
    Clear,
    /// Atomically replaces all keys; duplicate keys reject the complete update.
    ReplaceAll {
        /// Replacement entries in semantic publication order.
        entries: Vec<(K, S)>,
    },
}

/// Ordered semantic output produced by a dynamic manager or selected-input adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DynamicEmissions<Q> {
    /// Produces no semantic queue item.
    None,
    /// Produces one semantic queue item.
    One(Q),
    /// Produces multiple semantic queue items in order.
    Many(Vec<Q>),
}

impl<Q> DynamicEmissions<Q> {
    pub(crate) fn append_to(self, queue: &mut Vec<Q>) {
        match self {
            Self::None => {}
            Self::One(item) => queue.push(item),
            Self::Many(items) => queue.extend(items),
        }
    }
}

/// One ordered step in an atomic dynamic manager update.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicManagerStep<C, Q> {
    command: Option<C>,
    emissions: DynamicEmissions<Q>,
}

impl<C, Q> DynamicManagerStep<C, Q> {
    /// Creates a command step whose semantic emissions are reduced immediately before any selected
    /// candidate initialization produced by that command.
    pub fn command(command: C, emissions: DynamicEmissions<Q>) -> Self {
        Self {
            command: Some(command),
            emissions,
        }
    }

    /// Creates an emission-only step.
    pub fn emissions(emissions: DynamicEmissions<Q>) -> Self {
        Self {
            command: None,
            emissions,
        }
    }
}

/// One atomic manager update containing validated slot commands and ordered semantic emissions.
///
/// All commands are validated sequentially before any candidate is materialized. If candidates
/// warm, the framework delays this update's ordered semantic program until every required
/// candidate has an authoritative replay value. [`ReplaceAll`](IndexedSlotCommand::ReplaceAll)
/// may be followed by incremental command steps in the same update. Empty replacement cohorts
/// publish exactly one reducer invocation containing their manager emissions.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicManagerUpdate<C, Q> {
    steps: Vec<DynamicManagerStep<C, Q>>,
}

impl<C, Q> DynamicManagerUpdate<C, Q> {
    /// Creates one atomic manager update.
    ///
    /// `emissions` form the first step, followed by the commands in iterator order. Use
    /// [`sequenced`](Self::sequenced) when candidate initialization must be interleaved with
    /// command-specific semantic emissions.
    pub fn new(commands: impl IntoIterator<Item = C>, emissions: DynamicEmissions<Q>) -> Self {
        let mut steps = Vec::new();
        steps.push(DynamicManagerStep::emissions(emissions));
        steps.extend(
            commands
                .into_iter()
                .map(|command| DynamicManagerStep::command(command, DynamicEmissions::None)),
        );
        Self { steps }
    }

    /// Creates one atomic update from explicitly ordered command and emission steps.
    pub fn sequenced(steps: impl IntoIterator<Item = DynamicManagerStep<C, Q>>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
        }
    }

    pub(crate) fn into_steps(self) -> Vec<DynamicManagerStep<C, Q>> {
        self.steps
            .into_iter()
            .filter(|step| {
                step.command.is_some()
                    || match &step.emissions {
                        DynamicEmissions::None => false,
                        DynamicEmissions::One(_) => true,
                        DynamicEmissions::Many(emissions) => !emissions.is_empty(),
                    }
            })
            .collect()
    }
}

/// Behavior when a stable slot selects its current full-generational signal identity again.
///
/// This declaration is mandatory so repeated identity never gains hidden operator-specific
/// ownership behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SameSignalIdentity {
    /// Keeps the current adapter, claim, connection, replay cursor, and generation unchanged.
    ///
    /// A newer keep-current command still supersedes and discards older pending intent.
    KeepCurrent,
    /// Stages a fresh adapter, connection, and generation as a real replacement.
    ///
    /// The old generation remains current until the new generation is authoritatively ready.
    Restart,
}

/// Framework-controlled policy for one selected dynamic slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarSlot(());
/// Framework-controlled policy for an ordered collection of stable dynamic slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedSlots(());
/// Framework-controlled policy for dynamic slots addressed by public keys.
pub struct KeyedSlots<K>(PhantomData<fn() -> K>);

impl<K> Clone for KeyedSlots<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for KeyedSlots<K> {}

impl<K> core::fmt::Debug for KeyedSlots<K> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("KeyedSlots")
    }
}

impl<K> PartialEq for KeyedSlots<K> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<K> Eq for KeyedSlots<K> {}

/// Sealed policy requiring selected scalar signals to implement [`ReplayableSignal`].
///
/// Normalize non-replayable selections with `.cache()` before constructing the spec.
pub struct ReplayScalar<T>(PhantomData<fn() -> T>);
/// Sealed policy requiring selected vector signals to implement [`ReplayableSignalVec`].
///
/// Normalize non-replayable selections with `.cache()`. Authoritative empty replacements are ready
/// values; warming remains distinct.
pub struct ReplaySignalVec<T>(PhantomData<fn() -> T>);
/// Sealed policy requiring selected map signals to implement [`ReplayableSignalMap`].
///
/// Normalize non-replayable selections with `.cache()`. Authoritative empty replacements are ready
/// values; warming remains distinct.
pub struct ReplaySignalMap<K, V>(PhantomData<fn() -> (K, V)>);

pub(crate) mod sealed {
    use bevy_platform::sync::{Arc, OnceLock};

    use crate::{
        ScalarSignalEntityRef, SignalEntityRef, SignalKind, SignalRegistrar, SignalRegistrationError,
        graph::SignalEntity,
    };

    pub trait DynamicSlotPolicy<S> {}

    pub trait DynamicSelectedReplay<S> {
        fn retain_pending(events: &mut alloc::vec::Vec<Self::Event>, event: Self::Event)
        where
            Self: super::DynamicSelectedReplay<S>;

        fn into_materializer(signal: S) -> crate::SignalMaterializer<Self::Kind>
        where
            Self: super::DynamicSelectedReplay<S>;
    }

    pub trait DynamicOutputKind {
        fn plan_output<'world, 'materialization>(
            registrar: &mut SignalRegistrar<'world, 'materialization>,
            input: ScalarSignalEntityRef<'materialization, ()>,
            owner: Arc<OnceLock<SignalEntity>>,
            delivery: crate::OutputDelivery<<Self as SignalKind>::Event>,
        ) -> Result<SignalEntityRef<'materialization, Self>, SignalRegistrationError>
        where
            Self: super::DynamicOutputKind,
            <Self as SignalKind>::Event: Send + Sync;
    }

    pub trait DynamicOutputPlanning<Kind: super::DynamicOutputKind> {
        fn plan_output<'world, 'materialization>(
            registrar: &mut SignalRegistrar<'world, 'materialization>,
            input: ScalarSignalEntityRef<'materialization, ()>,
            owner: Arc<OnceLock<SignalEntity>>,
            delivery: crate::OutputDelivery<<Kind as SignalKind>::Event>,
        ) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError>;
    }

    pub trait CompleteDynamicSignalOperatorSpec<'materialization> {
        fn plan<'world>(
            spec: Self,
            registrar: &mut SignalRegistrar<'world, 'materialization>,
        ) -> Result<
            SignalEntityRef<
                'materialization,
                <Self as super::CompleteDynamicSignalOperatorSpec<'materialization>>::OutputKind,
            >,
            SignalRegistrationError,
        >
        where
            Self: Sized + super::CompleteDynamicSignalOperatorSpec<'materialization>;
    }
}

/// Framework-controlled mapping from a slot policy to its command and public adapter context.
pub trait DynamicSlotPolicy<S>: sealed::DynamicSlotPolicy<S> + Send + Sync + 'static {
    /// Commands accepted by this slot policy.
    type Command: Send + Sync + 'static;
    /// Logical context visible to selected-input adapter semantics.
    type Context: Clone + Send + Sync + 'static;
}

impl<S> sealed::DynamicSlotPolicy<S> for ScalarSlot {}
impl<S: Send + Sync + 'static> DynamicSlotPolicy<S> for ScalarSlot {
    type Command = ScalarSlotCommand<S>;
    type Context = ();
}

impl<S> sealed::DynamicSlotPolicy<S> for IndexedSlots {}
impl<S: Send + Sync + 'static> DynamicSlotPolicy<S> for IndexedSlots {
    type Command = IndexedSlotCommand<S>;
    type Context = usize;
}

impl<K, S> sealed::DynamicSlotPolicy<S> for KeyedSlots<K> {}
impl<K, S> DynamicSlotPolicy<S> for KeyedSlots<K>
where
    K: Clone + Ord + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    type Command = KeyedSlotCommand<K, S>;
    type Context = K;
}

/// Framework-controlled selected-input replay policy.
///
/// Framework planning hooks are intentionally unavailable downstream:
///
/// ```compile_fail,E0576
/// use jonmo::dynamic::{DynamicSelectedReplay, ReplayScalar};
///
/// fn cannot_extract_materializer<S>()
/// where
///     ReplayScalar<i32>: DynamicSelectedReplay<S>,
/// {
///     let _ = <ReplayScalar<i32> as DynamicSelectedReplay<S>>::into_materializer;
/// }
/// ```
pub trait DynamicSelectedReplay<S>: sealed::DynamicSelectedReplay<S> + Send + Sync + 'static {
    /// Event received from the selected replayable signal.
    type Event: Clone + Send + Sync + 'static;
    /// Selected signal kind.
    type Kind: SignalKind<Event = Self::Event>;
}

impl<T, S> sealed::DynamicSelectedReplay<S> for ReplayScalar<T>
where
    T: Clone + Send + Sync + 'static,
    S: ReplayableSignal<Item = T>,
{
    fn retain_pending(
        events: &mut Vec<<Self as DynamicSelectedReplay<S>>::Event>,
        event: <Self as DynamicSelectedReplay<S>>::Event,
    ) {
        events.clear();
        events.push(event);
    }

    fn into_materializer(signal: S) -> crate::SignalMaterializer<<Self as DynamicSelectedReplay<S>>::Kind> {
        Box::new(signal).into_signal_materializer()
    }
}

impl<T, S> DynamicSelectedReplay<S> for ReplayScalar<T>
where
    T: Clone + Send + Sync + 'static,
    S: ReplayableSignal<Item = T>,
{
    type Event = T;
    type Kind = ScalarSignalKind<T>;
}

impl<T, S> sealed::DynamicSelectedReplay<S> for ReplaySignalVec<T>
where
    T: Clone + Send + Sync + 'static,
    S: ReplayableSignalVec<Item = T>,
{
    fn retain_pending(
        events: &mut Vec<<Self as DynamicSelectedReplay<S>>::Event>,
        event: <Self as DynamicSelectedReplay<S>>::Event,
    ) {
        // Deltas require the replay baseline, and stateful adapters can observe batch boundaries.
        events.push(event);
    }

    fn into_materializer(signal: S) -> crate::SignalMaterializer<<Self as DynamicSelectedReplay<S>>::Kind> {
        Box::new(signal).into_signal_vec_materializer()
    }
}

impl<T, S> DynamicSelectedReplay<S> for ReplaySignalVec<T>
where
    T: Clone + Send + Sync + 'static,
    S: ReplayableSignalVec<Item = T>,
{
    type Event = Vec<crate::signal_vec::VecDiff<T>>;
    type Kind = VecSignalKind<T>;
}

impl<K, V, S> sealed::DynamicSelectedReplay<S> for ReplaySignalMap<K, V>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    S: ReplayableSignalMap<Key = K, Value = V>,
{
    fn retain_pending(
        events: &mut Vec<<Self as DynamicSelectedReplay<S>>::Event>,
        event: <Self as DynamicSelectedReplay<S>>::Event,
    ) {
        events.push(event);
    }

    fn into_materializer(signal: S) -> crate::SignalMaterializer<<Self as DynamicSelectedReplay<S>>::Kind> {
        Box::new(signal).into_signal_map_materializer()
    }
}

impl<K, V, S> DynamicSelectedReplay<S> for ReplaySignalMap<K, V>
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    S: ReplayableSignalMap<Key = K, Value = V>,
{
    type Event = Vec<crate::signal_map::MapDiff<K, V>>;
    type Kind = MapSignalKind<K, V>;
}

/// Framework-controlled marker for signal kinds supported as dynamic stable outputs.
///
/// Output planning and owner binding remain private framework operations:
///
/// ```compile_fail,E0576
/// use jonmo::{ScalarSignalKind, dynamic::DynamicOutputKind};
///
/// let _ = <ScalarSignalKind<i32> as DynamicOutputKind>::plan_output;
/// ```
pub trait DynamicOutputKind: SignalKind + sealed::DynamicOutputKind + Sized {}

impl<T> sealed::DynamicOutputKind for ScalarSignalKind<T>
where
    T: Send + Sync + 'static,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<<Self as SignalKind>::Event>,
    ) -> Result<SignalEntityRef<'materialization, Self>, SignalRegistrationError> {
        registrar.signal_entity(
            input,
            move |In(()): In<()>, world: &mut World| take_dynamic_output::<T>(world, &owner),
            delivery,
        )
    }
}
impl<T> DynamicOutputKind for ScalarSignalKind<T> where T: Send + Sync + 'static {}

impl<T> sealed::DynamicOutputKind for VecSignalKind<T>
where
    T: Send + Sync + 'static,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<<Self as SignalKind>::Event>,
    ) -> Result<SignalEntityRef<'materialization, Self>, SignalRegistrationError> {
        registrar.signal_vec_entity(
            input,
            move |In(()): In<()>, world: &mut World| {
                take_dynamic_output::<Vec<crate::signal_vec::VecDiff<T>>>(world, &owner)
            },
            delivery,
        )
    }
}
impl<T> DynamicOutputKind for VecSignalKind<T> where T: Send + Sync + 'static {}

impl<K, V> sealed::DynamicOutputKind for MapSignalKind<K, V>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<<Self as SignalKind>::Event>,
    ) -> Result<SignalEntityRef<'materialization, Self>, SignalRegistrationError> {
        registrar.signal_map_entity(
            input,
            move |In(()): In<()>, world: &mut World| {
                take_dynamic_output::<Vec<crate::signal_map::MapDiff<K, V>>>(world, &owner)
            },
            delivery,
        )
    }
}
impl<K, V> DynamicOutputKind for MapSignalKind<K, V>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
}

#[doc(hidden)]
pub struct DynamicOrdinaryOutput(());
#[doc(hidden)]
pub struct DynamicReplayableOutput(());

impl<Kind> sealed::DynamicOutputPlanning<Kind> for DynamicOrdinaryOutput
where
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<<Kind as SignalKind>::Event>,
    ) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError> {
        <Kind as sealed::DynamicOutputKind>::plan_output(registrar, input, owner, delivery)
    }
}

impl<T> sealed::DynamicOutputPlanning<ScalarSignalKind<T>> for DynamicReplayableOutput
where
    T: Clone + Send + Sync + 'static,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<T>,
    ) -> Result<SignalEntityRef<'materialization, ScalarSignalKind<T>>, SignalRegistrationError> {
        registrar.replayable_signal_entity(
            input,
            move |In(()): In<()>, world: &mut World| take_dynamic_output::<T>(world, &owner),
            ReplayInitialization::Warming,
            delivery,
        )
    }
}

impl<T> sealed::DynamicOutputPlanning<VecSignalKind<T>> for DynamicReplayableOutput
where
    T: Clone + Send + Sync + 'static,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<Vec<crate::signal_vec::VecDiff<T>>>,
    ) -> Result<SignalEntityRef<'materialization, VecSignalKind<T>>, SignalRegistrationError> {
        registrar.replayable_signal_vec_entity(
            input,
            move |In(()): In<()>, world: &mut World| {
                take_dynamic_output::<Vec<crate::signal_vec::VecDiff<T>>>(world, &owner)
            },
            ReplayInitialization::Warming,
            delivery,
        )
    }
}

impl<K, V> sealed::DynamicOutputPlanning<MapSignalKind<K, V>> for DynamicReplayableOutput
where
    K: Clone + Ord + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn plan_output<'world, 'materialization>(
        registrar: &mut SignalRegistrar<'world, 'materialization>,
        input: ScalarSignalEntityRef<'materialization, ()>,
        owner: Arc<OnceLock<SignalEntity>>,
        delivery: OutputDelivery<Vec<crate::signal_map::MapDiff<K, V>>>,
    ) -> Result<SignalEntityRef<'materialization, MapSignalKind<K, V>>, SignalRegistrationError> {
        registrar.replayable_signal_map_entity(
            input,
            move |In(()): In<()>, world: &mut World| {
                take_dynamic_output::<Vec<crate::signal_map::MapDiff<K, V>>>(world, &owner)
            },
            ReplayInitialization::Warming,
            delivery,
        )
    }
}

#[doc(hidden)]
pub struct DynamicNeedsReplay(());
#[doc(hidden)]
pub struct DynamicNeedsAdapter(());
#[doc(hidden)]
pub struct DynamicNeedsOutput(());
#[doc(hidden)]
pub struct DynamicNeedsIdentityPolicy<Mode = DynamicOrdinaryOutput>(PhantomData<fn() -> Mode>);
#[doc(hidden)]
pub struct DynamicComplete<Mode = DynamicOrdinaryOutput>(PhantomData<fn() -> Mode>);
#[doc(hidden)]
pub struct DynamicMissing(());

/// A typed, declarative dynamic operator specification.
///
/// Policy-specific constructors begin a type-state sequence that must declare selected replay,
/// adapter semantics, stable output reduction, and same-identity behavior before
/// [`SignalRegistrar::dynamic_operator`] accepts the value. Selected inputs must satisfy a sealed
/// replay policy; call `.cache()` before selection when necessary.
///
/// Commands in one manager update are validated sequentially and committed atomically. If any
/// required candidate is warming, the old published generation remains authoritative and the
/// ordered semantic journal waits for the complete cohort. An empty cohort publishes one
/// authoritative empty replacement. Valid replacement stages new adapters and connections before
/// releasing old ownership.
///
/// Command, generation, or candidate rejection discards candidate values and ownership while
/// preserving the complete old generation. The default behavior then panics descriptively;
/// `.on_candidate_error(...)` installs typed continuation instead.
///
/// Output planning mirrors [`SignalRegistrar`]:
/// - [`scalar_output`](Self::scalar_output), [`signal_vec_output`](Self::signal_vec_output), and
///   [`signal_map_output`](Self::signal_map_output) plan ordinary linear output entities without a
///   replay cache or maintenance claim, supporting move-only queue items and move-only outputs.
/// - [`replayable_scalar_output`](Self::replayable_scalar_output),
///   [`replayable_signal_vec_output`](Self::replayable_signal_vec_output), and
///   [`replayable_signal_map_output`](Self::replayable_signal_map_output) plan managed `Warming`
///   replay terminals that normalize through `.cache()` without adding another entity or
///   maintenance lifecycle.
///
/// Ordinary dynamic outputs accept move-only queue items and move-only output payloads:
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     SameSignalIdentity, ScalarSlotCommand, SignalRegistrar, SignalRegistrationError,
///     prelude::*,
/// };
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct NonClone(usize);
///
/// fn valid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select {
///                     signal: signal::always(1usize).dedupe(),
///                 }],
///                 DynamicEmissions::<NonClone>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     let _ = registrar.dynamic_operator(
///         DynamicSignalOperatorSpec::scalar(manager)
///             .replay_scalar::<usize>()
///             .adapter(|(), value| DynamicEmissions::One(NonClone(value)))
///             .scalar_output(|values| values.into_iter().last())
///             .same_signal_identity(SameSignalIdentity::KeepCurrent),
///     )?;
///     Ok(())
/// }
/// ```
///
/// Replayable dynamic outputs reject non-`Clone` payloads at compile time:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// struct NonClone(usize);
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select { signal: signal::always(1usize).dedupe() }],
///                 DynamicEmissions::<NonClone>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<usize>()
///         .adapter(|(), value| DynamicEmissions::One(NonClone(value)))
///         .replayable_scalar_output(|values| values.into_iter().last());
///     Ok(())
/// }
/// ```
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
///     signal_vec::VecDiff,
/// };
///
/// struct NonClone(usize);
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select { signal: signal::always(1usize).dedupe() }],
///                 DynamicEmissions::<NonClone>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<usize>()
///         .adapter(|(), value| DynamicEmissions::One(NonClone(value)))
///         .replayable_signal_vec_output(|values| Some(vec![VecDiff::Replace { values }]));
///     Ok(())
/// }
/// ```
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
///     signal_map::MapDiff,
/// };
///
/// struct NonClone(usize);
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select { signal: signal::always(1usize).dedupe() }],
///                 DynamicEmissions::<NonClone>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<usize>()
///         .adapter(|(), value| DynamicEmissions::One(NonClone(value)))
///         .replayable_signal_map_output(|values| {
///             Some(vec![MapDiff::Replace {
///                 entries: values.into_iter().enumerate().collect(),
///             }])
///         });
///     Ok(())
/// }
/// ```
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
///     signal_map::MapDiff,
/// };
///
/// #[derive(Clone)]
/// struct KeyWithoutOrd(usize);
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select { signal: signal::always(1usize).dedupe() }],
///                 DynamicEmissions::<KeyWithoutOrd>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<usize>()
///         .adapter(|(), value| DynamicEmissions::One(KeyWithoutOrd(value)))
///         .replayable_signal_map_output(|keys| {
///             Some(vec![MapDiff::Replace {
///                 entries: keys.into_iter().map(|key| (key, 1usize)).collect(),
///             }])
///         });
///     Ok(())
/// }
/// ```
///
/// A non-replayable selected scalar is rejected at compile time:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select { signal: signal::always(1) }],
///                 DynamicEmissions::<i32>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::scalar(manager).replay_scalar::<i32>();
///     Ok(())
/// }
/// ```
///
/// An incomplete specification cannot be passed to the registrar:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| {
///             Some(DynamicManagerUpdate::new(
///                 [ScalarSlotCommand::Select { signal: signal::always(1).dedupe() }],
///                 DynamicEmissions::<i32>::None,
///             ))
///         },
///         OutputDelivery::move_only(),
///     )?;
///     registrar.dynamic_operator(DynamicSignalOperatorSpec::scalar(manager))?;
///     Ok(())
/// }
/// ```
///
/// Replay selection alone is still incomplete:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| Some(DynamicManagerUpdate::new(
///             [ScalarSlotCommand::Select { signal: signal::always(1).dedupe() }],
///             DynamicEmissions::<i32>::None,
///         )),
///         OutputDelivery::move_only(),
///     )?;
///     let spec = DynamicSignalOperatorSpec::scalar(manager).replay_scalar::<i32>();
///     registrar.dynamic_operator(spec)?;
///     Ok(())
/// }
/// ```
///
/// Adapter declaration alone is still incomplete:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| Some(DynamicManagerUpdate::new(
///             [ScalarSlotCommand::Select { signal: signal::always(1).dedupe() }],
///             DynamicEmissions::<i32>::None,
///         )),
///         OutputDelivery::move_only(),
///     )?;
///     let spec = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<i32>()
///         .adapter(|(), value| DynamicEmissions::One(value));
///     registrar.dynamic_operator(spec)?;
///     Ok(())
/// }
/// ```
///
/// Output declaration still requires an explicit same-identity policy:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| Some(DynamicManagerUpdate::new(
///             [ScalarSlotCommand::Select { signal: signal::always(1).dedupe() }],
///             DynamicEmissions::<i32>::None,
///         )),
///         OutputDelivery::move_only(),
///     )?;
///     let spec = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<i32>()
///         .adapter(|(), value| DynamicEmissions::One(value))
///         .scalar_output(|values| values.into_iter().last());
///     registrar.dynamic_operator(spec)?;
///     Ok(())
/// }
/// ```
///
/// The candidate-error handler is unavailable before output selection:
///
/// ```compile_fail,E0599
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery,
///     ScalarSlotCommand, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| Some(DynamicManagerUpdate::new(
///             [ScalarSlotCommand::Select { signal: signal::always(1).dedupe() }],
///             DynamicEmissions::<i32>::None,
///         )),
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::scalar(manager)
///         .replay_scalar::<i32>()
///         .adapter(|(), value| DynamicEmissions::One(value))
///         .on_candidate_error(|_| {});
///     Ok(())
/// }
/// ```
///
/// A keyed operator adapter receives a key, not an indexed-slot position:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::{
///     DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, KeyedSlotCommand,
///     OutputDelivery, SignalRegistrar, SignalRegistrationError, prelude::*,
/// };
///
/// fn invalid<'world, 'materialization>(
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// ) -> Result<(), SignalRegistrationError> {
///     let manager = registrar.signal_entity(
///         (),
///         move |In(())| Some(DynamicManagerUpdate::new(
///             [KeyedSlotCommand::Insert { key: "key", signal: signal::always(1).dedupe() }],
///             DynamicEmissions::<i32>::None,
///         )),
///         OutputDelivery::move_only(),
///     )?;
///     let _ = DynamicSignalOperatorSpec::keyed(manager)
///         .replay_scalar::<i32>()
///         .adapter(|_index: usize, value| DynamicEmissions::One(value));
///     Ok(())
/// }
/// ```
///
/// Indexed-only commands are not part of the keyed command surface:
///
/// ```compile_fail,E0599
/// use jonmo::{KeyedSlotCommand, prelude::BoxedReplayableSignal};
///
/// let _: KeyedSlotCommand<i32, BoxedReplayableSignal<i32>> =
///     KeyedSlotCommand::Move { old_index: 0, new_index: 1 };
/// ```
///
/// Replay policy markers cannot be constructed or implemented externally:
///
/// ```compile_fail,E0423
/// use core::marker::PhantomData;
/// use jonmo::ReplayScalar;
/// let _ = ReplayScalar::<i32>(PhantomData);
/// ```
///
/// ```compile_fail,E0277
/// use jonmo::{ScalarSignalKind, dynamic::DynamicSelectedReplay};
///
/// struct ForgedReplay;
///
/// impl<S> DynamicSelectedReplay<S> for ForgedReplay {
///     type Event = i32;
///     type Kind = ScalarSignalKind<i32>;
/// }
/// ```
#[must_use]
pub struct DynamicSignalOperatorSpec<
    'materialization,
    P,
    S,
    Q,
    RP = DynamicMissing,
    A = DynamicMissing,
    Kind = DynamicMissing,
    R = DynamicMissing,
    State = DynamicNeedsReplay,
> where
    P: DynamicSlotPolicy<S>,
    Q: 'static,
{
    manager: ScalarSignalEntityRef<'materialization, DynamicManagerUpdate<P::Command, Q>>,
    replay: RP,
    adapter: A,
    reducer: R,
    output_delivery: Option<StoredOutputDelivery>,
    same_identity: Option<SameSignalIdentity>,
    error_handler: Option<DynamicCandidateErrorHandler>,
    on_deactivate: Option<DynamicDeactivateCallback>,
    #[allow(clippy::type_complexity)]
    marker: PhantomData<fn() -> (P, S, Q, Kind, State)>,
}

impl<'materialization, S, Q>
    DynamicSignalOperatorSpec<
        'materialization,
        ScalarSlot,
        S,
        Q,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
    >
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
{
    /// Begins a scalar-slot dynamic operator specification from its graph-visible manager.
    pub fn scalar(
        manager: ScalarSignalEntityRef<'materialization, DynamicManagerUpdate<ScalarSlotCommand<S>, Q>>,
    ) -> Self {
        Self {
            manager,
            replay: DynamicMissing(()),
            adapter: DynamicMissing(()),
            reducer: DynamicMissing(()),
            output_delivery: None,
            same_identity: None,
            error_handler: None,
            on_deactivate: None,
            marker: PhantomData,
        }
    }
}

impl<'materialization, S, Q>
    DynamicSignalOperatorSpec<
        'materialization,
        IndexedSlots,
        S,
        Q,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
    >
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
{
    /// Begins an indexed-slot dynamic operator specification.
    pub fn indexed(
        manager: ScalarSignalEntityRef<'materialization, DynamicManagerUpdate<IndexedSlotCommand<S>, Q>>,
    ) -> Self {
        Self {
            manager,
            replay: DynamicMissing(()),
            adapter: DynamicMissing(()),
            reducer: DynamicMissing(()),
            output_delivery: None,
            same_identity: None,
            error_handler: None,
            on_deactivate: None,
            marker: PhantomData,
        }
    }
}

impl<'materialization, K, S, Q>
    DynamicSignalOperatorSpec<
        'materialization,
        KeyedSlots<K>,
        S,
        Q,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
    >
where
    K: Clone + Ord + Send + Sync + 'static,
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
{
    /// Begins a keyed-slot dynamic operator specification.
    pub fn keyed(
        manager: ScalarSignalEntityRef<'materialization, DynamicManagerUpdate<KeyedSlotCommand<K, S>, Q>>,
    ) -> Self {
        Self {
            manager,
            replay: DynamicMissing(()),
            adapter: DynamicMissing(()),
            reducer: DynamicMissing(()),
            output_delivery: None,
            same_identity: None,
            error_handler: None,
            on_deactivate: None,
            marker: PhantomData,
        }
    }
}

impl<'materialization, P, S, Q>
    DynamicSignalOperatorSpec<
        'materialization,
        P,
        S,
        Q,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicNeedsReplay,
    >
where
    P: DynamicSlotPolicy<S>,
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
{
    /// Selects the framework-verified scalar replay policy for selected inputs.
    pub fn replay_scalar<T>(
        self,
    ) -> DynamicSignalOperatorSpec<
        'materialization,
        P,
        S,
        Q,
        ReplayScalar<T>,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicNeedsAdapter,
    >
    where
        T: Clone + Send + Sync + 'static,
        S: ReplayableSignal<Item = T>,
    {
        DynamicSignalOperatorSpec {
            manager: self.manager,
            replay: ReplayScalar(PhantomData),
            adapter: DynamicMissing(()),
            reducer: DynamicMissing(()),
            output_delivery: self.output_delivery,
            same_identity: self.same_identity,
            error_handler: self.error_handler,
            on_deactivate: self.on_deactivate,
            marker: PhantomData,
        }
    }

    /// Selects the framework-verified vector replay policy for selected inputs.
    pub fn replay_signal_vec<T>(
        self,
    ) -> DynamicSignalOperatorSpec<
        'materialization,
        P,
        S,
        Q,
        ReplaySignalVec<T>,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicNeedsAdapter,
    >
    where
        T: Clone + Send + Sync + 'static,
        S: ReplayableSignalVec<Item = T>,
    {
        DynamicSignalOperatorSpec {
            manager: self.manager,
            replay: ReplaySignalVec(PhantomData),
            adapter: DynamicMissing(()),
            reducer: DynamicMissing(()),
            output_delivery: self.output_delivery,
            same_identity: self.same_identity,
            error_handler: self.error_handler,
            on_deactivate: self.on_deactivate,
            marker: PhantomData,
        }
    }

    /// Selects the framework-verified map replay policy for selected inputs.
    pub fn replay_signal_map<K, V>(
        self,
    ) -> DynamicSignalOperatorSpec<
        'materialization,
        P,
        S,
        Q,
        ReplaySignalMap<K, V>,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicNeedsAdapter,
    >
    where
        K: Clone + Ord + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
        S: ReplayableSignalMap<Key = K, Value = V>,
    {
        DynamicSignalOperatorSpec {
            manager: self.manager,
            replay: ReplaySignalMap(PhantomData),
            adapter: DynamicMissing(()),
            reducer: DynamicMissing(()),
            output_delivery: self.output_delivery,
            same_identity: self.same_identity,
            error_handler: self.error_handler,
            on_deactivate: self.on_deactivate,
            marker: PhantomData,
        }
    }
}

impl<'materialization, P, S, Q, RP>
    DynamicSignalOperatorSpec<
        'materialization,
        P,
        S,
        Q,
        RP,
        DynamicMissing,
        DynamicMissing,
        DynamicMissing,
        DynamicNeedsAdapter,
    >
where
    P: DynamicSlotPolicy<S>,
    Q: 'static,
    RP: DynamicSelectedReplay<S>,
{
    /// Declares selected-input adapter semantics.
    pub fn adapter<A>(
        self,
        adapter: A,
    ) -> DynamicSignalOperatorSpec<'materialization, P, S, Q, RP, A, DynamicMissing, DynamicMissing, DynamicNeedsOutput>
    where
        A: FnMut(P::Context, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    {
        DynamicSignalOperatorSpec {
            manager: self.manager,
            replay: self.replay,
            adapter,
            reducer: DynamicMissing(()),
            output_delivery: self.output_delivery,
            same_identity: self.same_identity,
            error_handler: self.error_handler,
            on_deactivate: self.on_deactivate,
            marker: PhantomData,
        }
    }
}

macro_rules! output_transition {
    ($(#[$meta:meta])* $method:ident, $mode:ty, $kind:ty, $event:ty, [$($generic:ident),*], [$($bound:tt)*]) => {
        impl<'materialization, P, S, Q, RP, A>
            DynamicSignalOperatorSpec<
                'materialization,
                P,
                S,
                Q,
                RP,
                A,
                DynamicMissing,
                DynamicMissing,
                DynamicNeedsOutput,
            >
        where
            P: DynamicSlotPolicy<S>,
            Q: 'static,
            RP: DynamicSelectedReplay<S>,
            A: FnMut(P::Context, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
        {
            $(#[$meta])*
            pub fn $method<$($generic,)* R>(
                self,
                reducer: R,
            ) -> DynamicSignalOperatorSpec<
                'materialization,
                P,
                S,
                Q,
                RP,
                A,
                $kind,
                R,
                DynamicNeedsIdentityPolicy<$mode>,
            >
            where
                $($bound)*
                R: FnMut(Vec<Q>) -> Option<$event> + Send + Sync + 'static,
            {
                DynamicSignalOperatorSpec {
                    manager: self.manager,
                    replay: self.replay,
                    adapter: self.adapter,
                    reducer,
                    output_delivery: Some(OutputDelivery::<$event>::move_only().into_stored()),
                    same_identity: self.same_identity,
                    error_handler: self.error_handler,
                    on_deactivate: self.on_deactivate,
                    marker: PhantomData,
                }
            }
        }
    };
}

output_transition!(
    /// Declares an ordinary (non-replayable) stable scalar output reducer, defaulting to
    /// [`OutputDelivery::move_only`].
    scalar_output,
    DynamicOrdinaryOutput,
    ScalarSignalKind<O>,
    O,
    [O],
    [O: Send + Sync + 'static,]
);
output_transition!(
    /// Declares an ordinary (non-replayable) stable vector output reducer, defaulting to
    /// [`OutputDelivery::move_only`].
    signal_vec_output,
    DynamicOrdinaryOutput,
    VecSignalKind<T>,
    Vec<crate::signal_vec::VecDiff<T>>,
    [T],
    [T: Send + Sync + 'static,]
);
output_transition!(
    /// Declares an ordinary (non-replayable) stable map output reducer, defaulting to
    /// [`OutputDelivery::move_only`].
    signal_map_output,
    DynamicOrdinaryOutput,
    MapSignalKind<K, V>,
    Vec<crate::signal_map::MapDiff<K, V>>,
    [K, V],
    [K: Send + Sync + 'static, V: Send + Sync + 'static,]
);
output_transition!(
    /// Declares a managed `Warming` replayable stable scalar output reducer, defaulting to
    /// [`OutputDelivery::move_only`].
    replayable_scalar_output,
    DynamicReplayableOutput,
    ScalarSignalKind<O>,
    O,
    [O],
    [O: Clone + Send + Sync + 'static,]
);
output_transition!(
    /// Declares a managed `Warming` replayable stable vector output reducer, defaulting to
    /// [`OutputDelivery::move_only`].
    replayable_signal_vec_output,
    DynamicReplayableOutput,
    VecSignalKind<T>,
    Vec<crate::signal_vec::VecDiff<T>>,
    [T],
    [T: Clone + Send + Sync + 'static,]
);
output_transition!(
    /// Declares a managed `Warming` replayable stable map output reducer, defaulting to
    /// [`OutputDelivery::move_only`].
    replayable_signal_map_output,
    DynamicReplayableOutput,
    MapSignalKind<K, V>,
    Vec<crate::signal_map::MapDiff<K, V>>,
    [K, V],
    [K: Clone + Ord + Send + Sync + 'static, V: Clone + Send + Sync + 'static,]
);

impl<'materialization, P, S, Q, RP, A, Kind, R, Mode>
    DynamicSignalOperatorSpec<'materialization, P, S, Q, RP, A, Kind, R, DynamicNeedsIdentityPolicy<Mode>>
where
    P: DynamicSlotPolicy<S>,
    Q: 'static,
    Kind: DynamicOutputKind,
{
    /// Sets the stable output entity's delivery policy.
    pub fn output_delivery(mut self, delivery: OutputDelivery<Kind::Event>) -> Self {
        self.output_delivery = Some(delivery.into_stored());
        self
    }

    /// Installs a typed runtime rejection handler for invalid dynamic commands, slot/generation
    /// exhaustion, and candidate registration failures.
    ///
    /// A rejected dynamic update or candidate is discarded completely and the current generation is
    /// preserved before this handler runs. Without a handler, the same rejection panics
    /// descriptively after discard.
    pub fn on_candidate_error(mut self, handler: impl FnMut(SignalRegistrationError) + Send + Sync + 'static) -> Self {
        self.error_handler = Some(Box::new(handler));
        self
    }

    /// Completes the specification with explicit same-identity behavior.
    ///
    /// [`KeepCurrent`](SameSignalIdentity::KeepCurrent) preserves current topology and cursor;
    /// [`Restart`](SameSignalIdentity::Restart) stages a fresh generation.
    pub fn same_signal_identity(
        self,
        same_identity: SameSignalIdentity,
    ) -> DynamicSignalOperatorSpec<'materialization, P, S, Q, RP, A, Kind, R, DynamicComplete<Mode>> {
        DynamicSignalOperatorSpec {
            manager: self.manager,
            replay: self.replay,
            adapter: self.adapter,
            reducer: self.reducer,
            output_delivery: self.output_delivery,
            same_identity: Some(same_identity),
            error_handler: self.error_handler,
            on_deactivate: self.on_deactivate,
            marker: PhantomData,
        }
    }
}

impl<'materialization, P, S, Q, RP, A, Kind, R, Mode>
    DynamicSignalOperatorSpec<'materialization, P, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    P: DynamicSlotPolicy<S>,
    Q: 'static,
    Kind: DynamicOutputKind,
{
    /// Sets the stable output entity's delivery policy.
    pub fn output_delivery(mut self, delivery: OutputDelivery<Kind::Event>) -> Self {
        self.output_delivery = Some(delivery.into_stored());
        self
    }

    /// Installs a typed runtime rejection handler for invalid dynamic commands, slot/generation
    /// exhaustion, and candidate registration failures.
    ///
    /// A rejected dynamic update or candidate is discarded completely and the current generation is
    /// preserved before this handler runs. Without a handler, the same rejection panics
    /// descriptively after discard.
    pub fn on_candidate_error(mut self, handler: impl FnMut(SignalRegistrationError) + Send + Sync + 'static) -> Self {
        self.error_handler = Some(Box::new(handler));
        self
    }

    pub(crate) fn on_deactivate(mut self, callback: impl FnMut() + Send + Sync + 'static) -> Self {
        self.on_deactivate = Some(Box::new(callback));
        self
    }
}

/// A fully declared dynamic operator specification accepted by
/// [`SignalRegistrar::dynamic_operator`].
///
/// Registration dispatch remains private even for completed specifications:
///
/// ```compile_fail,E0599
/// use jonmo::{SignalRegistrar, dynamic::CompleteDynamicSignalOperatorSpec};
///
/// fn cannot_plan<'world, 'materialization, Spec>(
///     spec: Spec,
///     registrar: &mut SignalRegistrar<'world, 'materialization>,
/// )
/// where
///     Spec: CompleteDynamicSignalOperatorSpec<'materialization>,
/// {
///     let _ = spec.plan(registrar);
/// }
/// ```
pub trait CompleteDynamicSignalOperatorSpec<'materialization>:
    sealed::CompleteDynamicSignalOperatorSpec<'materialization>
{
    /// Stable output kind.
    type OutputKind: DynamicOutputKind;
}

fn bound_owner(owner: &OnceLock<SignalEntity>) -> SignalEntity {
    owner
        .get()
        .copied()
        .unwrap_or_else(|| panic!("dynamic stable output identity was not bound before execution"))
}

fn finish_dynamic_process(world: &mut World, owner_entity: SignalEntity, process: DynamicRuntimeProcess) -> bool {
    let DynamicRuntimeProcess {
        wake_output,
        subscriptions_changed,
        cleanup,
        registration_cleanup,
    } = process;
    if subscriptions_changed {
        let has_deferred_cleanup = {
            let mut owner = world
                .get_mut::<DynamicSignalOwner>(*owner_entity)
                .expect("processed dynamic owner must remain installed");
            owner.refresh_subscriptions();
            owner.runtime.has_after_output_cleanup()
        };
        if has_deferred_cleanup {
            crate::graph::record_pending_after_output_cleanup(world, owner_entity);
        }
    }
    if !cleanup.is_empty() {
        let subscription_count = {
            let mut owner = world
                .get_mut::<DynamicSignalOwner>(*owner_entity)
                .expect("processed dynamic owner must remain installed");
            let count = owner.subscriptions.len();
            owner
                .subscriptions
                .extend(cleanup.iter().map(DynamicSignalSubscription::inspect));
            count
        };
        if subscriptions_changed {
            crate::graph::sync_dynamic_owner_levels(world, owner_entity);
        }
        let releases = crate::graph::preflight_dynamic_connection_releases(world, cleanup)
            .unwrap_or_else(|error| panic!("dynamic subscription cleanup preflight failed: {error:?}"));
        world
            .get_mut::<DynamicSignalOwner>(*owner_entity)
            .expect("preflight must not remove the dynamic owner")
            .subscriptions
            .truncate(subscription_count);
        crate::cleanup::process_dynamic_connection_releases(world, releases);
    } else if subscriptions_changed {
        crate::graph::sync_dynamic_owner_levels(world, owner_entity);
    }
    registration_cleanup.finish(world);
    wake_output
}

pub(crate) fn flush_all_after_output_cleanup(world: &mut World) {
    let owners = if let Some(pending) = crate::graph::take_pending_after_output_cleanup(world) {
        if pending.is_empty() {
            return;
        }
        pending
    } else {
        let mut query = world.query::<(bevy_ecs::entity::Entity, &DynamicSignalOwner)>();
        query
            .iter(world)
            .filter_map(|(entity, owner)| {
                owner
                    .runtime
                    .has_after_output_cleanup()
                    .then_some(SignalEntity::from(entity))
            })
            .collect::<Vec<_>>()
    };
    for owner_entity in owners {
        let cleanup = {
            let Some(mut owner) = world.get_mut::<DynamicSignalOwner>(*owner_entity) else {
                continue;
            };
            owner.runtime.take_after_output_cleanup()
        };
        if cleanup.is_empty() {
            continue;
        }
        finish_dynamic_process(
            world,
            owner_entity,
            DynamicRuntimeProcess {
                subscriptions_changed: true,
                registration_cleanup: crate::registration::RegistrationCleanup::default(),
                wake_output: false,
                cleanup,
            },
        );
    }
}

fn process_dynamic_manager<Update: Send + Sync + 'static>(
    world: &mut World,
    owner_binding: &Arc<OnceLock<SignalEntity>>,
    update: Update,
) -> Option<()> {
    let owner = world.get::<DynamicSignalOwner>(*bound_owner(owner_binding))?;
    if !Arc::ptr_eq(&owner.incarnation, owner_binding) {
        return None;
    }
    process_dynamic_input(
        world,
        bound_owner(owner_binding),
        DynamicRuntimeInput::Manager(Box::new(update)),
        None,
    )
}

fn process_dynamic_adapter<Event: Send + Sync + 'static>(
    world: &mut World,
    binding: &OnceLock<DynamicAdapterBinding>,
    event: Event,
) -> Option<()> {
    let binding = binding
        .get()
        .expect("dynamic adapter executed before binding publication");
    let mut owner = world.get_mut::<DynamicSignalOwner>(*binding.owner)?;
    if !owner.materializing && owner.inputs.is_empty() {
        // Ordinary adapter events cannot reenter World and do not change ownership. Let the
        // runtime validate the full binding once, without queueing or rebuilding inspections.
        let process = owner.runtime.process_adapter(binding, Box::new(event), true);
        if !process.subscriptions_changed && process.cleanup.is_empty() && process.registration_cleanup.is_empty() {
            return process.wake_output.then_some(());
        }
        let epoch = DynamicOwnerEpoch::capture(world, binding.owner)?;
        let wake_output = finish_dynamic_process(world, binding.owner, process);
        return (wake_output && epoch.is_current(world)).then_some(());
    }
    if !owner.subscriptions.iter().any(|subscription| {
        subscription.adapter == binding.adapter
            && subscription.selected == binding.selected
            && subscription.slot == binding.slot
    }) {
        return None;
    }
    process_dynamic_input(
        world,
        binding.owner,
        DynamicRuntimeInput::Adapter(*binding, Box::new(event)),
        Some(binding.adapter),
    )
}

fn process_dynamic_input(
    world: &mut World,
    owner_entity: SignalEntity,
    input: DynamicRuntimeInput,
    executing_adapter: Option<SignalEntity>,
) -> Option<()> {
    let mut owner = world.get_mut::<DynamicSignalOwner>(*owner_entity)?;
    owner.inputs.push_back(input);
    if owner.materializing {
        return None;
    }
    let epoch = DynamicOwnerEpoch::capture(world, owner_entity)?;
    let mut wake_output = false;
    while epoch.is_current(world) {
        let Some(input) = world.get_mut::<DynamicSignalOwner>(*owner_entity)?.inputs.pop_front() else {
            break;
        };
        let process = match input {
            DynamicRuntimeInput::Adapter(binding, event) => {
                let defer_self_adapter = executing_adapter == Some(binding.adapter);
                world
                    .get_mut::<DynamicSignalOwner>(*owner_entity)?
                    .runtime
                    .process_adapter(&binding, event, defer_self_adapter)
            }
            DynamicRuntimeInput::Manager(update) => {
                let Some(plan) = world
                    .get_mut::<DynamicSignalOwner>(*owner_entity)?
                    .runtime
                    .prepare_manager(update)
                else {
                    continue;
                };
                apply_dynamic_plan(world, epoch.clone(), plan)?
            }
        };
        wake_output |= finish_dynamic_process(world, owner_entity, process);
    }
    (wake_output && epoch.is_current(world)).then_some(())
}

fn apply_dynamic_plan(
    world: &mut World,
    epoch: DynamicOwnerEpoch,
    plan: DynamicManagerPlan,
) -> Option<DynamicRuntimeProcess> {
    let owner_entity = epoch.owner;
    world.get_mut::<DynamicSignalOwner>(*owner_entity)?.materializing = true;
    let scope = DynamicMaterializationScope { world, epoch };
    let (committed, registration_cleanup) = match (plan.materialize)(scope.world, owner_entity) {
        Ok(crate::registration::DynamicCandidateMaterialization::Live { candidates, cleanup }) => {
            (Ok(candidates), cleanup)
        }
        Ok(crate::registration::DynamicCandidateMaterialization::Retired { claims, cleanup }) => {
            for claim in claims {
                claim.release(scope.world);
            }
            cleanup.finish(scope.world);
            return None;
        }
        Err(error) => {
            if !scope.epoch.is_current(scope.world) {
                return None;
            }
            (Err(error), crate::registration::RegistrationCleanup::default())
        }
    };
    // Commit defers hook-capable cleanup until the resident runtime has adopted the new tokens.
    // Otherwise a cleanup hook could retire the owner while those tokens are still only local.
    let mut process = (plan.apply)(
        scope
            .world
            .get_mut::<DynamicSignalOwner>(*owner_entity)
            .expect("materialized dynamic owner")
            .runtime
            .as_any_mut(),
        committed,
    );
    process.registration_cleanup = registration_cleanup;
    Some(process)
}

fn take_dynamic_output<Event>(world: &mut World, owner_binding: &Arc<OnceLock<SignalEntity>>) -> Option<Event>
where
    Event: Send + Sync + 'static,
{
    let owner_entity = bound_owner(owner_binding);
    let (output, cleanup) = {
        let mut owner = world.get_mut::<DynamicSignalOwner>(*owner_entity)?;
        if !Arc::ptr_eq(&owner.incarnation, owner_binding) {
            return None;
        }
        let output = owner.runtime.take_output();
        let cleanup = owner.runtime.take_after_output_cleanup();
        (output, cleanup)
    };
    if !cleanup.is_empty() {
        let epoch = DynamicOwnerEpoch::capture(world, owner_entity)?;
        finish_dynamic_process(
            world,
            owner_entity,
            DynamicRuntimeProcess {
                subscriptions_changed: true,
                registration_cleanup: crate::registration::RegistrationCleanup::default(),
                wake_output: false,
                cleanup,
            },
        );
        if !epoch.is_current(world) {
            return None;
        }
    }
    let output = output?;
    Some(
        output
            .downcast::<Event>()
            .map(|output| *output)
            .unwrap_or_else(|_| panic!("dynamic stable reducer produced an event of the wrong output kind")),
    )
}

fn plan_runtime<'world, 'materialization, Mode, ManagerEvent, Kind>(
    registrar: &mut SignalRegistrar<'world, 'materialization>,
    manager: ScalarSignalEntityRef<'materialization, ManagerEvent>,
    runtime: Box<dyn ErasedDynamicSignalRuntime>,
    output_delivery: StoredOutputDelivery,
) -> Result<SignalEntityRef<'materialization, Kind>, SignalRegistrationError>
where
    Mode: sealed::DynamicOutputPlanning<Kind>,
    ManagerEvent: Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
{
    let runtime_owner = DynamicSignalOwner::new(runtime);
    let owner_binding = runtime_owner.incarnation.clone();
    let manager_bridge = registrar.ordered_manager_bridge(manager, {
        let owner_binding = owner_binding.clone();
        move |In(update): In<ManagerEvent>, world: &mut World| process_dynamic_manager(world, &owner_binding, update)
    })?;
    let output = <Mode as sealed::DynamicOutputPlanning<Kind>>::plan_output(
        registrar,
        manager_bridge,
        owner_binding.clone(),
        OutputDelivery::from_stored(output_delivery),
    )?;
    registrar.install_entity_component(output, move |world, owner, _dependencies| {
        owner_binding
            .set(owner)
            .unwrap_or_else(|_| panic!("dynamic stable output identity was published more than once"));
        world.entity_mut(*owner).insert(runtime_owner);
    })?;
    Ok(output)
}

struct PendingDynamicCandidate<Event> {
    subscription: DynamicSignalSubscription,
    events: Vec<Event>,
}

impl<Event> PendingDynamicCandidate<Event> {
    fn new(
        selected: SignalEntity,
        adapter_claim: SignalClaim,
        connection: DynamicSignalConnection,
        slot: DynamicStableSlot,
    ) -> Self {
        Self {
            subscription: DynamicSignalSubscription {
                selected,
                adapter_claim,
                connection,
                slot,
            },
            events: Vec::new(),
        }
    }
}

type DynamicCandidateErrorHandler = Box<dyn FnMut(SignalRegistrationError) + Send + Sync + 'static>;
type DynamicDeactivateCallback = Box<dyn FnMut() + Send + Sync + 'static>;

struct DynamicRuntimeCore<Q> {
    next_generation: u64,
    queue: Vec<Q>,
    after_output_cleanup: Vec<DynamicSignalSubscription>,
    same_identity: SameSignalIdentity,
    error_handler: Option<DynamicCandidateErrorHandler>,
    on_deactivate: Option<DynamicDeactivateCallback>,
}

impl<Q> DynamicRuntimeCore<Q> {
    fn new(
        same_identity: SameSignalIdentity,
        error_handler: Option<DynamicCandidateErrorHandler>,
        on_deactivate: Option<DynamicDeactivateCallback>,
    ) -> Self {
        Self {
            next_generation: 0,
            queue: Vec::new(),
            after_output_cleanup: Vec::new(),
            same_identity,
            error_handler,
            on_deactivate,
        }
    }

    fn reject_candidate(&mut self, error: SignalRegistrationError) {
        if let Some(handler) = self.error_handler.as_mut() {
            handler(error);
        } else {
            panic!("dynamic update or candidate rejected before commit ({error:?}): {error}");
        }
    }

    fn allocate_generation(&mut self, next_generation: &mut u64) -> Option<u64> {
        let Some(generation) = next_generation.checked_add(1) else {
            self.reject_candidate(SignalRegistrationError::DynamicGenerationOverflow);
            return None;
        };
        *next_generation = generation;
        Some(generation)
    }

    fn allocate_slot(&mut self, next_slot_identity: &mut u64, next_generation: &mut u64) -> Option<DynamicStableSlot> {
        let Some(identity) = next_slot_identity.checked_add(1) else {
            self.reject_candidate(SignalRegistrationError::DynamicGenerationOverflow);
            return None;
        };
        let generation = self.allocate_generation(next_generation)?;
        *next_slot_identity = identity;
        Some(DynamicStableSlot { identity, generation })
    }

    fn candidates_ready<Event>(candidates: &BTreeMap<DynamicStableSlot, PendingDynamicCandidate<Event>>) -> bool {
        candidates.values().all(|candidate| !candidate.events.is_empty())
    }

    fn collect_keep_current_lineage(
        mut slot: DynamicStableSlot,
        predecessors: &mut BTreeMap<DynamicStableSlot, DynamicStableSlot>,
        normalized_candidates: &mut Vec<DynamicStableSlot>,
    ) {
        while let Some(previous) = predecessors.remove(&slot) {
            normalized_candidates.push(previous);
            slot = previous;
        }
    }

    fn retain_pending_adapter_event<S, RP>(
        candidates: Option<&mut BTreeMap<DynamicStableSlot, PendingDynamicCandidate<RP::Event>>>,
        binding: &DynamicAdapterBinding,
        event: RP::Event,
    ) -> bool
    where
        RP: DynamicSelectedReplay<S>,
    {
        let Some(candidates) = candidates else {
            return false;
        };
        let Some(candidate) = candidates.get_mut(&binding.slot) else {
            return false;
        };
        if candidate.subscription.adapter_claim.entity() != binding.adapter
            || candidate.subscription.selected != binding.selected
        {
            return false;
        }
        <RP as sealed::DynamicSelectedReplay<S>>::retain_pending(&mut candidate.events, event);
        true
    }

    fn ignored_process() -> DynamicRuntimeProcess {
        DynamicRuntimeProcess {
            subscriptions_changed: false,
            registration_cleanup: crate::registration::RegistrationCleanup::default(),
            wake_output: false,
            cleanup: Vec::new(),
        }
    }

    fn current_adapter_process(&self) -> DynamicRuntimeProcess {
        DynamicRuntimeProcess {
            subscriptions_changed: false,
            registration_cleanup: crate::registration::RegistrationCleanup::default(),
            wake_output: !self.queue.is_empty(),
            cleanup: Vec::new(),
        }
    }

    fn finish_pending_publication(
        &mut self,
        adapter: SignalEntity,
        defer_self_adapter: bool,
        published: bool,
        mut cleanup: Vec<DynamicSignalSubscription>,
    ) -> DynamicRuntimeProcess {
        if defer_self_adapter
            && let Some(index) = cleanup
                .iter()
                .position(|subscription| subscription.adapter_claim.entity() == adapter)
        {
            self.after_output_cleanup.push(cleanup.remove(index));
        }
        DynamicRuntimeProcess {
            subscriptions_changed: published,
            registration_cleanup: crate::registration::RegistrationCleanup::default(),
            wake_output: published || !self.queue.is_empty(),
            cleanup,
        }
    }

    fn take_output<Out>(&mut self, reducer: &mut impl FnMut(Vec<Q>) -> Option<Out>) -> Option<ErasedSignalValue>
    where
        Out: Send + Sync + 'static,
    {
        let queue = core::mem::take(&mut self.queue);
        reducer(queue).map(|output| Box::new(output) as ErasedSignalValue)
    }

    fn has_after_output_cleanup(&self) -> bool {
        !self.after_output_cleanup.is_empty()
    }

    fn take_after_output_cleanup(&mut self) -> Vec<DynamicSignalSubscription> {
        core::mem::take(&mut self.after_output_cleanup)
    }

    fn clear_queues(&mut self) {
        self.queue.clear();
    }

    fn on_deactivate(&mut self) {
        if let Some(on_deactivate) = self.on_deactivate.as_mut() {
            on_deactivate();
        }
    }
}

#[derive(Clone, Copy)]
enum ScalarProjectedSlot {
    Empty,
    Existing(u64),
    Candidate(DynamicStableSlot),
}

enum PendingScalarSemantic<Q> {
    Manager(Vec<Q>),
    Candidate(DynamicStableSlot),
}

struct PendingScalarCohort<Q, Event> {
    projection: ScalarProjectedSlot,
    candidates: BTreeMap<DynamicStableSlot, PendingDynamicCandidate<Event>>,
    semantics: Vec<PendingScalarSemantic<Q>>,
}

#[allow(clippy::type_complexity)]
struct ScalarDynamicRuntime<S, Q, RP, A, Kind, R>
where
    RP: DynamicSelectedReplay<S>,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
{
    current: Option<DynamicSignalSubscription>,
    pending: Option<PendingScalarCohort<Q, RP::Event>>,
    next_slot_identity: u64,
    core: DynamicRuntimeCore<Q>,
    adapter: A,
    reducer: R,
    marker: PhantomData<fn() -> (S, RP, Kind)>,
}

impl<S, Q, RP, A, Kind, R> ScalarDynamicRuntime<S, Q, RP, A, Kind, R>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut((), RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
{
    fn desired_projection(&self) -> ScalarProjectedSlot {
        self.pending.as_ref().map_or_else(
            || {
                self.current
                    .as_ref()
                    .map_or(ScalarProjectedSlot::Empty, |subscription| {
                        ScalarProjectedSlot::Existing(subscription.slot.identity)
                    })
            },
            |pending| pending.projection,
        )
    }

    fn current_selected_for_identity(&self, identity: u64) -> Option<SignalEntity> {
        self.current
            .as_ref()
            .filter(|subscription| subscription.slot.identity == identity)
            .map(|subscription| subscription.selected)
    }

    fn pending_ready(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| DynamicRuntimeCore::<Q>::candidates_ready(&pending.candidates))
    }

    fn publish_pending(&mut self) -> Vec<DynamicSignalSubscription> {
        let mut pending = self
            .pending
            .take()
            .unwrap_or_else(|| panic!("ready scalar cohort disappeared before publication"));

        for semantic in pending.semantics {
            match semantic {
                PendingScalarSemantic::Manager(mut emissions) => self.core.queue.append(&mut emissions),
                PendingScalarSemantic::Candidate(slot) => {
                    let events = pending
                        .candidates
                        .get_mut(&slot)
                        .map(|candidate| core::mem::take(&mut candidate.events))
                        .unwrap_or_else(|| panic!("scalar cohort published before candidate readiness"));
                    for event in events {
                        (self.adapter)((), event).append_to(&mut self.core.queue);
                    }
                }
            }
        }

        let mut cleanup = Vec::new();
        match pending.projection {
            ScalarProjectedSlot::Empty => {
                cleanup.extend(self.current.take());
            }
            ScalarProjectedSlot::Existing(identity) => {
                assert!(
                    self.current
                        .as_ref()
                        .is_some_and(|subscription| subscription.slot.identity == identity),
                    "scalar cohort referenced a missing current slot"
                );
            }
            ScalarProjectedSlot::Candidate(slot) => {
                let candidate = pending
                    .candidates
                    .remove(&slot)
                    .unwrap_or_else(|| panic!("scalar cohort referenced a missing candidate"));
                cleanup.extend(self.current.replace(candidate.subscription));
            }
        }
        cleanup.extend(pending.candidates.into_values().map(|candidate| candidate.subscription));
        cleanup
    }

    fn prepare_update(
        &mut self,
        steps: Vec<DynamicManagerStep<ScalarSlotCommand<S>, Q>>,
    ) -> Option<DynamicManagerPlan> {
        struct PlannedScalarCandidate {
            slot: DynamicStableSlot,
            kept: Option<ScalarProjectedSlot>,
            superseded_on_keep: Option<DynamicStableSlot>,
        }

        let mut projection = self.desired_projection();
        // Any scalar command supersedes the old cohort, including its journal-only
        // transients. Candidates created within this atomic update remain independent.
        let old_candidate_slots = self
            .pending
            .as_ref()
            .filter(|_| steps.iter().any(|step| step.command.is_some()))
            .map(|pending| pending.candidates.keys().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut next_generation = self.core.next_generation;
        let mut next_slot_identity = self.next_slot_identity;
        let mut requests = Vec::new();
        let mut planned_candidates = Vec::new();
        let mut semantics = Vec::new();

        for DynamicManagerStep { command, emissions } in steps {
            let mut manager_emissions = Vec::new();
            emissions.append_to(&mut manager_emissions);
            if !manager_emissions.is_empty() {
                semantics.push(PendingScalarSemantic::Manager(manager_emissions));
            }
            let Some(command) = command else {
                continue;
            };

            match command {
                ScalarSlotCommand::Select { signal } => {
                    let previous = projection;
                    let slot = match previous {
                        ScalarProjectedSlot::Empty => {
                            self.core.allocate_slot(&mut next_slot_identity, &mut next_generation)?
                        }
                        ScalarProjectedSlot::Existing(identity) => DynamicStableSlot {
                            identity,
                            generation: self.core.allocate_generation(&mut next_generation)?,
                        },
                        ScalarProjectedSlot::Candidate(previous_slot) => DynamicStableSlot {
                            identity: previous_slot.identity,
                            generation: self.core.allocate_generation(&mut next_generation)?,
                        },
                    };
                    let current_selected = self.current_selected_for_identity(slot.identity);
                    requests.push(crate::registration::DynamicCandidateRequest {
                        selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                        slot,
                        current_selected,
                    });
                    planned_candidates.push(PlannedScalarCandidate {
                        slot,
                        kept: current_selected.map(|_| ScalarProjectedSlot::Existing(slot.identity)),
                        superseded_on_keep: match previous {
                            ScalarProjectedSlot::Candidate(slot) => Some(slot),
                            ScalarProjectedSlot::Empty | ScalarProjectedSlot::Existing(_) => None,
                        },
                    });
                    projection = ScalarProjectedSlot::Candidate(slot);
                    semantics.push(PendingScalarSemantic::Candidate(slot));
                }
                ScalarSlotCommand::Clear => projection = ScalarProjectedSlot::Empty,
            }
        }

        Some(DynamicManagerPlan::new::<Self, _>(
            requests,
            self.core.same_identity,
            move |runtime, committed| {
                let mut candidates = BTreeMap::new();
                if let Some(old_pending) = runtime.pending.take() {
                    candidates = old_pending.candidates;
                    let mut old_semantics = old_pending.semantics;
                    old_semantics.append(&mut semantics);
                    semantics = old_semantics;
                }

                let mut predecessors = planned_candidates
                    .iter()
                    .filter_map(|planned| planned.superseded_on_keep.map(|previous| (planned.slot, previous)))
                    .collect::<BTreeMap<_, _>>();
                let mut normalized_candidates = Vec::new();
                let mut removed_semantic_slots = BTreeSet::new();
                for (result, planned) in committed.into_iter().zip(planned_candidates) {
                    match result {
                        crate::registration::CommittedDynamicCandidate::KeptCurrent => {
                            let kept = planned.kept.unwrap_or_else(|| {
                                panic!("scalar selection from empty unexpectedly kept a current identity")
                            });
                            if matches!(projection, ScalarProjectedSlot::Candidate(slot) if slot == planned.slot) {
                                projection = kept;
                            }
                            removed_semantic_slots.insert(planned.slot);
                            DynamicRuntimeCore::<Q>::collect_keep_current_lineage(
                                planned.slot,
                                &mut predecessors,
                                &mut normalized_candidates,
                            );
                        }
                        crate::registration::CommittedDynamicCandidate::New {
                            selected,
                            slot,
                            adapter_claim,
                            connection,
                        } => {
                            assert_eq!(
                                slot, planned.slot,
                                "scalar dynamic candidate changed stable slot during commit"
                            );
                            let replaced = candidates.insert(
                                slot,
                                PendingDynamicCandidate::new(selected, adapter_claim, connection, slot),
                            );
                            assert!(replaced.is_none(), "scalar candidate slot generation was reused");
                        }
                    }
                }

                let mut cleanup = Vec::new();
                for slot in normalized_candidates.into_iter().chain(old_candidate_slots) {
                    if let Some(candidate) = candidates.remove(&slot) {
                        cleanup.push(candidate.subscription);
                    }
                    removed_semantic_slots.insert(slot);
                }
                if !removed_semantic_slots.is_empty() {
                    semantics.retain(|semantic| {
                        !matches!(semantic, PendingScalarSemantic::Candidate(slot) if removed_semantic_slots.contains(slot))
                    });
                }

                runtime.core.next_generation = next_generation;
                runtime.next_slot_identity = next_slot_identity;
                runtime.pending = Some(PendingScalarCohort {
                    projection,
                    candidates,
                    semantics,
                });
                let published = runtime.pending_ready();
                if published {
                    cleanup.extend(runtime.publish_pending());
                }
                DynamicRuntimeProcess {
                    subscriptions_changed: true,
                    registration_cleanup: crate::registration::RegistrationCleanup::default(),
                    wake_output: published || !runtime.core.queue.is_empty(),
                    cleanup,
                }
            },
        ))
    }
}

impl<S, Q, RP, A, Kind, R> ErasedDynamicSignalRuntime for ScalarDynamicRuntime<S, Q, RP, A, Kind, R>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut((), RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
{
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }

    fn reject_candidate(&mut self, error: SignalRegistrationError) {
        self.core.reject_candidate(error);
    }

    fn append_subscriptions(&self, subscriptions: &mut Vec<DynamicSubscriptionInspection>) {
        subscriptions.extend(
            self.current
                .iter()
                .map(DynamicSignalSubscription::inspect)
                .chain(
                    self.pending
                        .iter()
                        .flat_map(|pending| pending.candidates.values())
                        .map(|candidate| candidate.subscription.inspect()),
                )
                .chain(
                    self.core
                        .after_output_cleanup
                        .iter()
                        .map(DynamicSignalSubscription::inspect),
                ),
        );
    }

    fn take_subscriptions(&mut self) -> Vec<DynamicSignalSubscription> {
        let mut subscriptions = self.current.take().into_iter().collect::<Vec<_>>();
        if let Some(pending) = self.pending.take() {
            subscriptions.extend(pending.candidates.into_values().map(|candidate| candidate.subscription));
        }
        subscriptions.append(&mut self.core.after_output_cleanup);
        subscriptions
    }

    fn clear_queues(&mut self) {
        self.next_slot_identity = 0;
        self.core.clear_queues();
    }

    fn on_deactivate(&mut self) {
        self.core.on_deactivate();
    }

    fn prepare_manager(&mut self, update: Box<dyn core::any::Any>) -> Option<DynamicManagerPlan> {
        let update = update
            .downcast::<DynamicManagerUpdate<ScalarSlotCommand<S>, Q>>()
            .unwrap_or_else(|_| panic!("scalar dynamic manager produced an event of the wrong type"));
        let steps = update.into_steps();
        if steps.is_empty() {
            return None;
        }
        self.prepare_update(steps)
    }

    fn process_adapter(
        &mut self,
        binding: &DynamicAdapterBinding,
        event: Box<dyn core::any::Any>,
        defer_self_adapter: bool,
    ) -> DynamicRuntimeProcess {
        let event = event
            .downcast::<RP::Event>()
            .map(|event| *event)
            .unwrap_or_else(|_| panic!("dynamic selected-input adapter received the wrong event type"));
        if self.current.as_ref().is_some_and(|current| {
            current.slot == binding.slot
                && current.adapter_claim.entity() == binding.adapter
                && current.selected == binding.selected
        }) {
            (self.adapter)((), event).append_to(&mut self.core.queue);
            return self.core.current_adapter_process();
        }
        if !DynamicRuntimeCore::<Q>::retain_pending_adapter_event::<S, RP>(
            self.pending.as_mut().map(|pending| &mut pending.candidates),
            binding,
            event,
        ) {
            return DynamicRuntimeCore::<Q>::ignored_process();
        }
        let published = self.pending_ready();
        let cleanup = if published { self.publish_pending() } else { Vec::new() };
        self.core
            .finish_pending_publication(binding.adapter, defer_self_adapter, published, cleanup)
    }

    fn take_output(&mut self) -> Option<ErasedSignalValue> {
        self.core.take_output(&mut self.reducer)
    }

    fn has_after_output_cleanup(&self) -> bool {
        self.core.has_after_output_cleanup()
    }

    fn take_after_output_cleanup(&mut self) -> Vec<DynamicSignalSubscription> {
        self.core.take_after_output_cleanup()
    }
}

impl<'materialization, S, Q, RP, A, Kind, R, Mode> sealed::CompleteDynamicSignalOperatorSpec<'materialization>
    for DynamicSignalOperatorSpec<'materialization, ScalarSlot, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut((), RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
    Mode: sealed::DynamicOutputPlanning<Kind>,
{
    fn plan<'world>(
        spec: Self,
        registrar: &mut SignalRegistrar<'world, 'materialization>,
    ) -> Result<
        SignalEntityRef<'materialization, <Self as CompleteDynamicSignalOperatorSpec<'materialization>>::OutputKind>,
        SignalRegistrationError,
    > {
        let output_delivery = spec
            .output_delivery
            .expect("complete scalar dynamic spec must carry output delivery");
        let same_identity = spec
            .same_identity
            .expect("complete scalar dynamic spec must carry same-identity policy");
        let runtime = ScalarDynamicRuntime::<S, Q, RP, A, Kind, R> {
            current: None,
            pending: None,
            next_slot_identity: 0,
            core: DynamicRuntimeCore::new(same_identity, spec.error_handler, spec.on_deactivate),
            adapter: spec.adapter,
            reducer: spec.reducer,
            marker: PhantomData,
        };
        plan_runtime::<Mode, _, _>(registrar, spec.manager, Box::new(runtime), output_delivery)
    }
}

impl<'materialization, S, Q, RP, A, Kind, R, Mode> CompleteDynamicSignalOperatorSpec<'materialization>
    for DynamicSignalOperatorSpec<'materialization, ScalarSlot, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut((), RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
    Mode: sealed::DynamicOutputPlanning<Kind>,
{
    type OutputKind = Kind;
}

#[derive(Clone, Copy)]
enum IndexedProjectedSlot {
    Existing(u64),
    Candidate(DynamicStableSlot),
}

enum PendingIndexedSemantic<Q> {
    Manager(Vec<Q>),
    Candidate { slot: DynamicStableSlot, index: usize },
}

struct PendingIndexedCohort<Q, Event> {
    projection: Vec<IndexedProjectedSlot>,
    candidates: BTreeMap<DynamicStableSlot, PendingDynamicCandidate<Event>>,
    semantics: Vec<PendingIndexedSemantic<Q>>,
}

#[allow(clippy::type_complexity)]
struct IndexedDynamicRuntime<S, Q, RP, A, Kind, R>
where
    RP: DynamicSelectedReplay<S>,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
{
    current: Vec<DynamicSignalSubscription>,
    pending: Option<PendingIndexedCohort<Q, RP::Event>>,
    next_slot_identity: u64,
    core: DynamicRuntimeCore<Q>,
    adapter: A,
    reducer: R,
    marker: PhantomData<fn() -> (S, RP, Kind)>,
}

impl<S, Q, RP, A, Kind, R> IndexedDynamicRuntime<S, Q, RP, A, Kind, R>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(usize, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
{
    fn desired_projection(&self) -> Vec<IndexedProjectedSlot> {
        self.pending.as_ref().map_or_else(
            || {
                self.current
                    .iter()
                    .map(|subscription| IndexedProjectedSlot::Existing(subscription.slot.identity))
                    .collect()
            },
            |pending| pending.projection.clone(),
        )
    }

    fn current_selected_for_identity(&self, identity: u64) -> Option<SignalEntity> {
        self.current
            .iter()
            .find(|subscription| subscription.slot.identity == identity)
            .map(|subscription| subscription.selected)
    }

    fn pending_ready(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| DynamicRuntimeCore::<Q>::candidates_ready(&pending.candidates))
    }

    fn publish_pending(&mut self) -> Vec<DynamicSignalSubscription> {
        let mut pending = self
            .pending
            .take()
            .unwrap_or_else(|| panic!("ready indexed cohort disappeared before publication"));

        for semantic in pending.semantics {
            match semantic {
                PendingIndexedSemantic::Manager(mut emissions) => self.core.queue.append(&mut emissions),
                PendingIndexedSemantic::Candidate { slot, index } => {
                    let events = pending
                        .candidates
                        .get_mut(&slot)
                        .map(|candidate| core::mem::take(&mut candidate.events))
                        .unwrap_or_else(|| panic!("indexed cohort published before candidate readiness"));
                    for event in events {
                        (self.adapter)(index, event).append_to(&mut self.core.queue);
                    }
                }
            }
        }

        let mut current = core::mem::take(&mut self.current)
            .into_iter()
            .map(|subscription| (subscription.slot.identity, subscription))
            .collect::<BTreeMap<_, _>>();
        let mut published = Vec::with_capacity(pending.projection.len());
        for slot in pending.projection {
            match slot {
                IndexedProjectedSlot::Existing(identity) => published.push(
                    current
                        .remove(&identity)
                        .unwrap_or_else(|| panic!("indexed cohort referenced a missing current slot")),
                ),
                IndexedProjectedSlot::Candidate(slot) => published.push(
                    pending
                        .candidates
                        .remove(&slot)
                        .map(|candidate| candidate.subscription)
                        .unwrap_or_else(|| panic!("indexed cohort referenced a missing candidate")),
                ),
            }
        }
        self.current = published;

        current
            .into_values()
            .chain(pending.candidates.into_values().map(|candidate| candidate.subscription))
            .collect()
    }

    fn prepare_update(
        &mut self,
        steps: Vec<DynamicManagerStep<IndexedSlotCommand<S>, Q>>,
    ) -> Option<DynamicManagerPlan> {
        struct PlannedIndexedCandidate {
            slot: DynamicStableSlot,
            kept: Option<IndexedProjectedSlot>,
            superseded_on_keep: Option<DynamicStableSlot>,
        }

        let mut projection = self.desired_projection();
        // Slot-specific edits cancel old generations of the targeted stable identity,
        // not unrelated journal-only transients. Clear/ReplaceAll supersede the old cohort.
        let old_candidate_slots = self
            .pending
            .as_ref()
            .map(|pending| pending.candidates.keys().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut superseded_identities = BTreeSet::new();
        let mut supersedes_cohort = false;
        let mut next_generation = self.core.next_generation;
        let mut next_slot_identity = self.next_slot_identity;
        let mut requests = Vec::new();
        let mut planned_candidates = Vec::new();
        let mut semantics = Vec::new();

        for DynamicManagerStep { command, emissions } in steps {
            let mut manager_emissions = Vec::new();
            emissions.append_to(&mut manager_emissions);
            if !manager_emissions.is_empty() {
                semantics.push(PendingIndexedSemantic::Manager(manager_emissions));
            }
            let Some(command) = command else {
                continue;
            };

            match command {
                IndexedSlotCommand::Insert { index, signal } => {
                    if index > projection.len() {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    }
                    let slot = self.core.allocate_slot(&mut next_slot_identity, &mut next_generation)?;
                    requests.push(crate::registration::DynamicCandidateRequest {
                        selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                        slot,
                        current_selected: None,
                    });
                    planned_candidates.push(PlannedIndexedCandidate {
                        slot,
                        kept: None,
                        superseded_on_keep: None,
                    });
                    projection.insert(index, IndexedProjectedSlot::Candidate(slot));
                    semantics.push(PendingIndexedSemantic::Candidate { slot, index });
                }
                IndexedSlotCommand::Replace { index, signal } => {
                    let Some(previous) = projection.get(index).copied() else {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    };
                    let generation = self.core.allocate_generation(&mut next_generation)?;
                    let identity = match previous {
                        IndexedProjectedSlot::Existing(identity) => identity,
                        IndexedProjectedSlot::Candidate(slot) => slot.identity,
                    };
                    superseded_identities.insert(identity);
                    let slot = DynamicStableSlot { identity, generation };
                    let current_selected = self.current_selected_for_identity(identity);
                    requests.push(crate::registration::DynamicCandidateRequest {
                        selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                        slot,
                        current_selected,
                    });
                    planned_candidates.push(PlannedIndexedCandidate {
                        slot,
                        kept: current_selected.map(|_| IndexedProjectedSlot::Existing(identity)),
                        superseded_on_keep: match previous {
                            IndexedProjectedSlot::Candidate(slot) => Some(slot),
                            IndexedProjectedSlot::Existing(_) => None,
                        },
                    });
                    projection[index] = IndexedProjectedSlot::Candidate(slot);
                    semantics.push(PendingIndexedSemantic::Candidate { slot, index });
                }
                IndexedSlotCommand::Remove { index } => {
                    if index >= projection.len() {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    }
                    let identity = match projection.remove(index) {
                        IndexedProjectedSlot::Existing(identity) => identity,
                        IndexedProjectedSlot::Candidate(slot) => slot.identity,
                    };
                    superseded_identities.insert(identity);
                }
                IndexedSlotCommand::Move { old_index, new_index } => {
                    if old_index >= projection.len() || new_index >= projection.len() {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    }
                    let slot = projection.remove(old_index);
                    projection.insert(new_index, slot);
                }
                IndexedSlotCommand::Clear => {
                    supersedes_cohort = true;
                    projection.clear();
                }
                IndexedSlotCommand::ReplaceAll { signals } => {
                    supersedes_cohort = true;
                    let previous_projection = core::mem::take(&mut projection);
                    projection.reserve(signals.len());
                    for (index, signal) in signals.into_iter().enumerate() {
                        let previous = previous_projection.get(index).copied();
                        let slot = match previous {
                            Some(IndexedProjectedSlot::Existing(identity)) => DynamicStableSlot {
                                identity,
                                generation: self.core.allocate_generation(&mut next_generation)?,
                            },
                            Some(IndexedProjectedSlot::Candidate(previous_slot)) => DynamicStableSlot {
                                identity: previous_slot.identity,
                                generation: self.core.allocate_generation(&mut next_generation)?,
                            },
                            None => self.core.allocate_slot(&mut next_slot_identity, &mut next_generation)?,
                        };
                        let current_selected = self.current_selected_for_identity(slot.identity);
                        requests.push(crate::registration::DynamicCandidateRequest {
                            selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                            slot,
                            current_selected,
                        });
                        planned_candidates.push(PlannedIndexedCandidate {
                            slot,
                            kept: current_selected.map(|_| IndexedProjectedSlot::Existing(slot.identity)),
                            superseded_on_keep: match previous {
                                Some(IndexedProjectedSlot::Candidate(slot)) => Some(slot),
                                Some(IndexedProjectedSlot::Existing(_)) | None => None,
                            },
                        });
                        projection.push(IndexedProjectedSlot::Candidate(slot));
                        semantics.push(PendingIndexedSemantic::Candidate { slot, index });
                    }
                }
            }
        }

        Some(DynamicManagerPlan::new::<Self, _>(
            requests,
            self.core.same_identity,
            move |runtime, committed| {
                let mut candidates = BTreeMap::new();
                if let Some(old_pending) = runtime.pending.take() {
                    candidates = old_pending.candidates;
                    let mut old_semantics = old_pending.semantics;
                    old_semantics.append(&mut semantics);
                    semantics = old_semantics;
                }

                let mut predecessors = planned_candidates
                    .iter()
                    .filter_map(|planned| planned.superseded_on_keep.map(|previous| (planned.slot, previous)))
                    .collect::<BTreeMap<_, _>>();
                let mut normalized_candidates = Vec::new();
                let mut removed_semantic_slots = BTreeSet::new();
                for (result, planned) in committed.into_iter().zip(planned_candidates) {
                    match result {
                        crate::registration::CommittedDynamicCandidate::KeptCurrent => {
                            let kept = planned
                                .kept
                                .unwrap_or_else(|| panic!("indexed insertion unexpectedly kept a current identity"));
                            for projected in &mut projection {
                                if matches!(projected, IndexedProjectedSlot::Candidate(slot) if *slot == planned.slot) {
                                    *projected = kept;
                                }
                            }
                            removed_semantic_slots.insert(planned.slot);
                            DynamicRuntimeCore::<Q>::collect_keep_current_lineage(
                                planned.slot,
                                &mut predecessors,
                                &mut normalized_candidates,
                            );
                        }
                        crate::registration::CommittedDynamicCandidate::New {
                            selected,
                            slot,
                            adapter_claim,
                            connection,
                        } => {
                            assert_eq!(
                                slot, planned.slot,
                                "indexed candidate changed stable slot during commit"
                            );
                            let replaced = candidates.insert(
                                slot,
                                PendingDynamicCandidate::new(selected, adapter_claim, connection, slot),
                            );
                            assert!(replaced.is_none(), "indexed candidate slot generation was reused");
                        }
                    }
                }

                let mut cleanup = Vec::new();
                for slot in normalized_candidates.into_iter().chain(
                    old_candidate_slots
                        .into_iter()
                        .filter(|slot| supersedes_cohort || superseded_identities.contains(&slot.identity)),
                ) {
                    if let Some(candidate) = candidates.remove(&slot) {
                        cleanup.push(candidate.subscription);
                    }
                    removed_semantic_slots.insert(slot);
                }
                if !removed_semantic_slots.is_empty() {
                    semantics.retain(|semantic| {
                        !matches!(semantic, PendingIndexedSemantic::Candidate { slot, .. } if removed_semantic_slots.contains(slot))
                    });
                }

                runtime.core.next_generation = next_generation;
                runtime.next_slot_identity = next_slot_identity;
                runtime.pending = Some(PendingIndexedCohort {
                    projection,
                    candidates,
                    semantics,
                });
                let published = runtime.pending_ready();
                if published {
                    cleanup.extend(runtime.publish_pending());
                }
                DynamicRuntimeProcess {
                    subscriptions_changed: true,
                    registration_cleanup: crate::registration::RegistrationCleanup::default(),
                    wake_output: published || !runtime.core.queue.is_empty(),
                    cleanup,
                }
            },
        ))
    }
}

impl<S, Q, RP, A, Kind, R> ErasedDynamicSignalRuntime for IndexedDynamicRuntime<S, Q, RP, A, Kind, R>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(usize, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
{
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }

    fn reject_candidate(&mut self, error: SignalRegistrationError) {
        self.core.reject_candidate(error);
    }

    fn append_subscriptions(&self, subscriptions: &mut Vec<DynamicSubscriptionInspection>) {
        subscriptions.extend(
            self.current
                .iter()
                .map(DynamicSignalSubscription::inspect)
                .chain(
                    self.pending
                        .iter()
                        .flat_map(|pending| pending.candidates.values())
                        .map(|candidate| candidate.subscription.inspect()),
                )
                .chain(
                    self.core
                        .after_output_cleanup
                        .iter()
                        .map(DynamicSignalSubscription::inspect),
                ),
        );
    }

    fn take_subscriptions(&mut self) -> Vec<DynamicSignalSubscription> {
        let mut subscriptions = core::mem::take(&mut self.current);
        if let Some(pending) = self.pending.take() {
            subscriptions.extend(pending.candidates.into_values().map(|candidate| candidate.subscription));
        }
        subscriptions.append(&mut self.core.after_output_cleanup);
        subscriptions
    }

    fn clear_queues(&mut self) {
        self.next_slot_identity = 0;
        self.core.clear_queues();
    }

    fn on_deactivate(&mut self) {
        self.core.on_deactivate();
    }

    fn prepare_manager(&mut self, update: Box<dyn core::any::Any>) -> Option<DynamicManagerPlan> {
        let update = update
            .downcast::<DynamicManagerUpdate<IndexedSlotCommand<S>, Q>>()
            .unwrap_or_else(|_| panic!("indexed dynamic manager produced an event of the wrong type"));
        let steps = update.into_steps();
        if steps.is_empty() {
            return None;
        }
        self.prepare_update(steps)
    }

    fn process_adapter(
        &mut self,
        binding: &DynamicAdapterBinding,
        event: Box<dyn core::any::Any>,
        defer_self_adapter: bool,
    ) -> DynamicRuntimeProcess {
        let event = event
            .downcast::<RP::Event>()
            .map(|event| *event)
            .unwrap_or_else(|_| panic!("indexed dynamic adapter received the wrong event type"));
        if let Some(index) = self.current.iter().position(|subscription| {
            subscription.slot == binding.slot
                && subscription.adapter_claim.entity() == binding.adapter
                && subscription.selected == binding.selected
        }) {
            (self.adapter)(index, event).append_to(&mut self.core.queue);
            return self.core.current_adapter_process();
        }
        if !DynamicRuntimeCore::<Q>::retain_pending_adapter_event::<S, RP>(
            self.pending.as_mut().map(|pending| &mut pending.candidates),
            binding,
            event,
        ) {
            return DynamicRuntimeCore::<Q>::ignored_process();
        }
        let published = self.pending_ready();
        let cleanup = if published { self.publish_pending() } else { Vec::new() };
        self.core
            .finish_pending_publication(binding.adapter, defer_self_adapter, published, cleanup)
    }

    fn take_output(&mut self) -> Option<ErasedSignalValue> {
        self.core.take_output(&mut self.reducer)
    }

    fn has_after_output_cleanup(&self) -> bool {
        self.core.has_after_output_cleanup()
    }

    fn take_after_output_cleanup(&mut self) -> Vec<DynamicSignalSubscription> {
        self.core.take_after_output_cleanup()
    }
}

impl<'materialization, S, Q, RP, A, Kind, R, Mode> sealed::CompleteDynamicSignalOperatorSpec<'materialization>
    for DynamicSignalOperatorSpec<'materialization, IndexedSlots, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(usize, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
    Mode: sealed::DynamicOutputPlanning<Kind>,
{
    fn plan<'world>(
        spec: Self,
        registrar: &mut SignalRegistrar<'world, 'materialization>,
    ) -> Result<
        SignalEntityRef<'materialization, <Self as CompleteDynamicSignalOperatorSpec<'materialization>>::OutputKind>,
        SignalRegistrationError,
    > {
        let output_delivery = spec
            .output_delivery
            .expect("complete indexed dynamic spec must carry output delivery");
        let same_identity = spec
            .same_identity
            .expect("complete indexed dynamic spec must carry same-identity policy");
        let runtime = IndexedDynamicRuntime::<S, Q, RP, A, Kind, R> {
            current: Vec::new(),
            pending: None,
            next_slot_identity: 0,
            core: DynamicRuntimeCore::new(same_identity, spec.error_handler, spec.on_deactivate),
            adapter: spec.adapter,
            reducer: spec.reducer,
            marker: PhantomData,
        };
        plan_runtime::<Mode, _, _>(registrar, spec.manager, Box::new(runtime), output_delivery)
    }
}

impl<'materialization, S, Q, RP, A, Kind, R, Mode> CompleteDynamicSignalOperatorSpec<'materialization>
    for DynamicSignalOperatorSpec<'materialization, IndexedSlots, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(usize, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
    Mode: sealed::DynamicOutputPlanning<Kind>,
{
    type OutputKind = Kind;
}

#[derive(Clone, Copy)]
enum KeyedProjectedSlot {
    Existing(u64),
    Candidate(DynamicStableSlot),
}

enum PendingKeyedSemantic<K, Q> {
    Manager(Vec<Q>),
    Candidate { slot: DynamicStableSlot, key: K },
}

struct PendingKeyedCohort<K, Q, Event> {
    projection: BTreeMap<K, KeyedProjectedSlot>,
    candidates: BTreeMap<DynamicStableSlot, PendingDynamicCandidate<Event>>,
    semantics: Vec<PendingKeyedSemantic<K, Q>>,
}

#[allow(clippy::type_complexity)]
struct KeyedDynamicRuntime<K, S, Q, RP, A, Kind, R>
where
    RP: DynamicSelectedReplay<S>,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
{
    current: BTreeMap<K, DynamicSignalSubscription>,
    pending: Option<PendingKeyedCohort<K, Q, RP::Event>>,
    next_slot_identity: u64,
    core: DynamicRuntimeCore<Q>,
    adapter: A,
    reducer: R,
    marker: PhantomData<fn() -> (S, RP, Kind)>,
}

impl<K, S, Q, RP, A, Kind, R> KeyedDynamicRuntime<K, S, Q, RP, A, Kind, R>
where
    K: Clone + Ord + Send + Sync + 'static,
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(K, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
{
    fn desired_projection(&self) -> BTreeMap<K, KeyedProjectedSlot> {
        self.pending.as_ref().map_or_else(
            || {
                self.current
                    .iter()
                    .map(|(key, subscription)| (key.clone(), KeyedProjectedSlot::Existing(subscription.slot.identity)))
                    .collect()
            },
            |pending| pending.projection.clone(),
        )
    }

    fn current_selected_for_identity(&self, identity: u64) -> Option<SignalEntity> {
        self.current
            .values()
            .find(|subscription| subscription.slot.identity == identity)
            .map(|subscription| subscription.selected)
    }

    fn pending_ready(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| DynamicRuntimeCore::<Q>::candidates_ready(&pending.candidates))
    }

    fn publish_pending(&mut self) -> Vec<DynamicSignalSubscription> {
        let mut pending = self
            .pending
            .take()
            .unwrap_or_else(|| panic!("ready keyed cohort disappeared before publication"));

        for semantic in pending.semantics {
            match semantic {
                PendingKeyedSemantic::Manager(mut emissions) => self.core.queue.append(&mut emissions),
                PendingKeyedSemantic::Candidate { slot, key } => {
                    let events = pending
                        .candidates
                        .get_mut(&slot)
                        .map(|candidate| core::mem::take(&mut candidate.events))
                        .unwrap_or_else(|| panic!("keyed cohort published before candidate readiness"));
                    for event in events {
                        (self.adapter)(key.clone(), event).append_to(&mut self.core.queue);
                    }
                }
            }
        }

        let mut current = core::mem::take(&mut self.current)
            .into_values()
            .map(|subscription| (subscription.slot.identity, subscription))
            .collect::<BTreeMap<_, _>>();
        let mut published = BTreeMap::new();
        for (key, slot) in pending.projection {
            let subscription = match slot {
                KeyedProjectedSlot::Existing(identity) => current
                    .remove(&identity)
                    .unwrap_or_else(|| panic!("keyed cohort referenced a missing current slot")),
                KeyedProjectedSlot::Candidate(slot) => pending
                    .candidates
                    .remove(&slot)
                    .map(|candidate| candidate.subscription)
                    .unwrap_or_else(|| panic!("keyed cohort referenced a missing candidate")),
            };
            published.insert(key, subscription);
        }
        self.current = published;

        current
            .into_values()
            .chain(pending.candidates.into_values().map(|candidate| candidate.subscription))
            .collect()
    }

    fn prepare_update(
        &mut self,
        steps: Vec<DynamicManagerStep<KeyedSlotCommand<K, S>, Q>>,
    ) -> Option<DynamicManagerPlan> {
        struct PlannedKeyedCandidate {
            slot: DynamicStableSlot,
            kept: Option<KeyedProjectedSlot>,
            superseded_on_keep: Option<DynamicStableSlot>,
        }

        let mut projection = self.desired_projection();
        // Slot-specific edits cancel old generations of the targeted stable identity,
        // not unrelated journal-only transients. Clear/ReplaceAll supersede the old cohort.
        let old_candidate_slots = self
            .pending
            .as_ref()
            .map(|pending| pending.candidates.keys().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut superseded_identities = BTreeSet::new();
        let mut supersedes_cohort = false;
        let mut next_generation = self.core.next_generation;
        let mut next_slot_identity = self.next_slot_identity;
        let mut requests = Vec::new();
        let mut planned_candidates = Vec::new();
        let mut semantics = Vec::new();

        for DynamicManagerStep { command, emissions } in steps {
            let mut manager_emissions = Vec::new();
            emissions.append_to(&mut manager_emissions);
            if !manager_emissions.is_empty() {
                semantics.push(PendingKeyedSemantic::Manager(manager_emissions));
            }
            let Some(command) = command else {
                continue;
            };

            match command {
                KeyedSlotCommand::Insert { key, signal } => {
                    if projection.contains_key(&key) {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    }
                    let slot = self.core.allocate_slot(&mut next_slot_identity, &mut next_generation)?;
                    requests.push(crate::registration::DynamicCandidateRequest {
                        selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                        slot,
                        current_selected: None,
                    });
                    planned_candidates.push(PlannedKeyedCandidate {
                        slot,
                        kept: None,
                        superseded_on_keep: None,
                    });
                    projection.insert(key.clone(), KeyedProjectedSlot::Candidate(slot));
                    semantics.push(PendingKeyedSemantic::Candidate { slot, key });
                }
                KeyedSlotCommand::Replace { key, signal } => {
                    let Some(previous) = projection.get(&key).copied() else {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    };
                    let generation = self.core.allocate_generation(&mut next_generation)?;
                    let identity = match previous {
                        KeyedProjectedSlot::Existing(identity) => identity,
                        KeyedProjectedSlot::Candidate(slot) => slot.identity,
                    };
                    superseded_identities.insert(identity);
                    let slot = DynamicStableSlot { identity, generation };
                    let current_selected = self.current_selected_for_identity(identity);
                    requests.push(crate::registration::DynamicCandidateRequest {
                        selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                        slot,
                        current_selected,
                    });
                    planned_candidates.push(PlannedKeyedCandidate {
                        slot,
                        kept: current_selected.map(|_| KeyedProjectedSlot::Existing(identity)),
                        superseded_on_keep: match previous {
                            KeyedProjectedSlot::Candidate(slot) => Some(slot),
                            KeyedProjectedSlot::Existing(_) => None,
                        },
                    });
                    projection.insert(key.clone(), KeyedProjectedSlot::Candidate(slot));
                    semantics.push(PendingKeyedSemantic::Candidate { slot, key });
                }
                KeyedSlotCommand::Remove { key } => {
                    let Some(previous) = projection.remove(&key) else {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    };
                    let identity = match previous {
                        KeyedProjectedSlot::Existing(identity) => identity,
                        KeyedProjectedSlot::Candidate(slot) => slot.identity,
                    };
                    superseded_identities.insert(identity);
                }
                KeyedSlotCommand::Rekey { old_key, new_key } => {
                    if old_key == new_key {
                        if !projection.contains_key(&old_key) {
                            self.core
                                .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                            return None;
                        }
                        continue;
                    }
                    if projection.contains_key(&new_key) {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    }
                    let Some(slot) = projection.remove(&old_key) else {
                        self.core
                            .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                        return None;
                    };
                    projection.insert(new_key, slot);
                }
                KeyedSlotCommand::Clear => {
                    supersedes_cohort = true;
                    projection.clear();
                }
                KeyedSlotCommand::ReplaceAll { entries } => {
                    supersedes_cohort = true;
                    let previous_projection = core::mem::take(&mut projection);
                    for (key, signal) in entries {
                        if projection.contains_key(&key) {
                            self.core
                                .reject_candidate(SignalRegistrationError::InvalidDynamicCommand);
                            return None;
                        }
                        let previous = previous_projection.get(&key).copied();
                        let slot = match previous {
                            Some(KeyedProjectedSlot::Existing(identity)) => DynamicStableSlot {
                                identity,
                                generation: self.core.allocate_generation(&mut next_generation)?,
                            },
                            Some(KeyedProjectedSlot::Candidate(previous_slot)) => DynamicStableSlot {
                                identity: previous_slot.identity,
                                generation: self.core.allocate_generation(&mut next_generation)?,
                            },
                            None => self.core.allocate_slot(&mut next_slot_identity, &mut next_generation)?,
                        };
                        let current_selected = self.current_selected_for_identity(slot.identity);
                        requests.push(crate::registration::DynamicCandidateRequest {
                            selected: <RP as sealed::DynamicSelectedReplay<S>>::into_materializer(signal),
                            slot,
                            current_selected,
                        });
                        planned_candidates.push(PlannedKeyedCandidate {
                            slot,
                            kept: current_selected.map(|_| KeyedProjectedSlot::Existing(slot.identity)),
                            superseded_on_keep: match previous {
                                Some(KeyedProjectedSlot::Candidate(slot)) => Some(slot),
                                Some(KeyedProjectedSlot::Existing(_)) | None => None,
                            },
                        });
                        projection.insert(key.clone(), KeyedProjectedSlot::Candidate(slot));
                        semantics.push(PendingKeyedSemantic::Candidate { slot, key });
                    }
                }
            }
        }

        Some(DynamicManagerPlan::new::<Self, _>(
            requests,
            self.core.same_identity,
            move |runtime, committed| {
                let mut candidates = BTreeMap::new();
                if let Some(old_pending) = runtime.pending.take() {
                    candidates = old_pending.candidates;
                    let mut old_semantics = old_pending.semantics;
                    old_semantics.append(&mut semantics);
                    semantics = old_semantics;
                }

                let mut predecessors = planned_candidates
                    .iter()
                    .filter_map(|planned| planned.superseded_on_keep.map(|previous| (planned.slot, previous)))
                    .collect::<BTreeMap<_, _>>();
                let mut normalized_candidates = Vec::new();
                let mut removed_semantic_slots = BTreeSet::new();
                for (result, planned) in committed.into_iter().zip(planned_candidates) {
                    match result {
                        crate::registration::CommittedDynamicCandidate::KeptCurrent => {
                            let kept = planned
                                .kept
                                .unwrap_or_else(|| panic!("keyed insertion unexpectedly kept a current identity"));
                            for projected in projection.values_mut() {
                                if matches!(projected, KeyedProjectedSlot::Candidate(slot) if *slot == planned.slot) {
                                    *projected = kept;
                                }
                            }
                            removed_semantic_slots.insert(planned.slot);
                            DynamicRuntimeCore::<Q>::collect_keep_current_lineage(
                                planned.slot,
                                &mut predecessors,
                                &mut normalized_candidates,
                            );
                        }
                        crate::registration::CommittedDynamicCandidate::New {
                            selected,
                            slot,
                            adapter_claim,
                            connection,
                        } => {
                            assert_eq!(slot, planned.slot, "keyed candidate changed stable slot during commit");
                            let replaced = candidates.insert(
                                slot,
                                PendingDynamicCandidate::new(selected, adapter_claim, connection, slot),
                            );
                            assert!(replaced.is_none(), "keyed candidate slot generation was reused");
                        }
                    }
                }

                let mut cleanup = Vec::new();
                for slot in normalized_candidates.into_iter().chain(
                    old_candidate_slots
                        .into_iter()
                        .filter(|slot| supersedes_cohort || superseded_identities.contains(&slot.identity)),
                ) {
                    if let Some(candidate) = candidates.remove(&slot) {
                        cleanup.push(candidate.subscription);
                    }
                    removed_semantic_slots.insert(slot);
                }
                if !removed_semantic_slots.is_empty() {
                    semantics.retain(|semantic| {
                        !matches!(semantic, PendingKeyedSemantic::Candidate { slot, .. } if removed_semantic_slots.contains(slot))
                    });
                }

                runtime.core.next_generation = next_generation;
                runtime.next_slot_identity = next_slot_identity;
                runtime.pending = Some(PendingKeyedCohort {
                    projection,
                    candidates,
                    semantics,
                });
                let published = runtime.pending_ready();
                if published {
                    cleanup.extend(runtime.publish_pending());
                }
                DynamicRuntimeProcess {
                    subscriptions_changed: true,
                    registration_cleanup: crate::registration::RegistrationCleanup::default(),
                    wake_output: published || !runtime.core.queue.is_empty(),
                    cleanup,
                }
            },
        ))
    }
}

impl<K, S, Q, RP, A, Kind, R> ErasedDynamicSignalRuntime for KeyedDynamicRuntime<K, S, Q, RP, A, Kind, R>
where
    K: Clone + Ord + Send + Sync + 'static,
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(K, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
{
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }

    fn reject_candidate(&mut self, error: SignalRegistrationError) {
        self.core.reject_candidate(error);
    }

    fn append_subscriptions(&self, subscriptions: &mut Vec<DynamicSubscriptionInspection>) {
        subscriptions.extend(
            self.current
                .values()
                .map(DynamicSignalSubscription::inspect)
                .chain(
                    self.pending
                        .iter()
                        .flat_map(|pending| pending.candidates.values())
                        .map(|candidate| candidate.subscription.inspect()),
                )
                .chain(
                    self.core
                        .after_output_cleanup
                        .iter()
                        .map(DynamicSignalSubscription::inspect),
                ),
        );
    }

    fn take_subscriptions(&mut self) -> Vec<DynamicSignalSubscription> {
        let mut subscriptions = core::mem::take(&mut self.current).into_values().collect::<Vec<_>>();
        if let Some(pending) = self.pending.take() {
            subscriptions.extend(pending.candidates.into_values().map(|candidate| candidate.subscription));
        }
        subscriptions.append(&mut self.core.after_output_cleanup);
        subscriptions
    }

    fn clear_queues(&mut self) {
        self.next_slot_identity = 0;
        self.core.clear_queues();
    }

    fn on_deactivate(&mut self) {
        self.core.on_deactivate();
    }

    fn prepare_manager(&mut self, update: Box<dyn core::any::Any>) -> Option<DynamicManagerPlan> {
        let update = update
            .downcast::<DynamicManagerUpdate<KeyedSlotCommand<K, S>, Q>>()
            .unwrap_or_else(|_| panic!("keyed dynamic manager produced an event of the wrong type"));
        let steps = update.into_steps();
        if steps.is_empty() {
            return None;
        }
        self.prepare_update(steps)
    }

    fn process_adapter(
        &mut self,
        binding: &DynamicAdapterBinding,
        event: Box<dyn core::any::Any>,
        defer_self_adapter: bool,
    ) -> DynamicRuntimeProcess {
        let event = event
            .downcast::<RP::Event>()
            .map(|event| *event)
            .unwrap_or_else(|_| panic!("keyed dynamic adapter received the wrong event type"));
        if let Some((key, _)) = self.current.iter().find(|(_, subscription)| {
            subscription.slot == binding.slot
                && subscription.adapter_claim.entity() == binding.adapter
                && subscription.selected == binding.selected
        }) {
            (self.adapter)(key.clone(), event).append_to(&mut self.core.queue);
            return self.core.current_adapter_process();
        }
        if !DynamicRuntimeCore::<Q>::retain_pending_adapter_event::<S, RP>(
            self.pending.as_mut().map(|pending| &mut pending.candidates),
            binding,
            event,
        ) {
            return DynamicRuntimeCore::<Q>::ignored_process();
        }
        let published = self.pending_ready();
        let cleanup = if published { self.publish_pending() } else { Vec::new() };
        self.core
            .finish_pending_publication(binding.adapter, defer_self_adapter, published, cleanup)
    }

    fn take_output(&mut self) -> Option<ErasedSignalValue> {
        self.core.take_output(&mut self.reducer)
    }

    fn has_after_output_cleanup(&self) -> bool {
        self.core.has_after_output_cleanup()
    }

    fn take_after_output_cleanup(&mut self) -> Vec<DynamicSignalSubscription> {
        self.core.take_after_output_cleanup()
    }
}

impl<'materialization, K, S, Q, RP, A, Kind, R, Mode> sealed::CompleteDynamicSignalOperatorSpec<'materialization>
    for DynamicSignalOperatorSpec<'materialization, KeyedSlots<K>, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    K: Clone + Ord + Send + Sync + 'static,
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(K, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
    Mode: sealed::DynamicOutputPlanning<Kind>,
{
    fn plan<'world>(
        spec: Self,
        registrar: &mut SignalRegistrar<'world, 'materialization>,
    ) -> Result<
        SignalEntityRef<'materialization, <Self as CompleteDynamicSignalOperatorSpec<'materialization>>::OutputKind>,
        SignalRegistrationError,
    > {
        let output_delivery = spec
            .output_delivery
            .expect("complete keyed dynamic spec must carry output delivery");
        let same_identity = spec
            .same_identity
            .expect("complete keyed dynamic spec must carry same-identity policy");
        let runtime = KeyedDynamicRuntime::<K, S, Q, RP, A, Kind, R> {
            current: BTreeMap::new(),
            pending: None,
            next_slot_identity: 0,
            core: DynamicRuntimeCore::new(same_identity, spec.error_handler, spec.on_deactivate),
            adapter: spec.adapter,
            reducer: spec.reducer,
            marker: PhantomData,
        };
        plan_runtime::<Mode, _, _>(registrar, spec.manager, Box::new(runtime), output_delivery)
    }
}

impl<'materialization, K, S, Q, RP, A, Kind, R, Mode> CompleteDynamicSignalOperatorSpec<'materialization>
    for DynamicSignalOperatorSpec<'materialization, KeyedSlots<K>, S, Q, RP, A, Kind, R, DynamicComplete<Mode>>
where
    K: Clone + Ord + Send + Sync + 'static,
    S: Send + Sync + 'static,
    Q: Send + Sync + 'static,
    RP: DynamicSelectedReplay<S>,
    A: FnMut(K, RP::Event) -> DynamicEmissions<Q> + Send + Sync + 'static,
    Kind: DynamicOutputKind,
    Kind::Event: Send + Sync,
    R: FnMut(Vec<Q>) -> Option<Kind::Event> + Send + Sync + 'static,
    Mode: sealed::DynamicOutputPlanning<Kind>,
{
    type OutputKind = Kind;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DynamicStableSlot {
    pub(crate) identity: u64,
    pub(crate) generation: u64,
}

#[must_use]
pub(crate) struct DynamicSignalConnection {
    source: SignalEntity,
    target: SignalEntity,
}

impl DynamicSignalConnection {
    pub(crate) fn new(source: SignalEntity, target: SignalEntity) -> Self {
        Self { source, target }
    }

    pub(crate) fn source(&self) -> SignalEntity {
        self.source
    }

    pub(crate) fn target(&self) -> SignalEntity {
        self.target
    }
}

pub(crate) struct DynamicSignalSubscription {
    pub(crate) selected: SignalEntity,
    pub(crate) adapter_claim: SignalClaim,
    pub(crate) connection: DynamicSignalConnection,
    pub(crate) slot: DynamicStableSlot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DynamicSubscriptionInspection {
    pub(crate) selected: SignalEntity,
    pub(crate) adapter: SignalEntity,
    pub(crate) source: SignalEntity,
    pub(crate) target: SignalEntity,
    pub(crate) slot: DynamicStableSlot,
}

impl DynamicSignalSubscription {
    pub(crate) fn inspect(&self) -> DynamicSubscriptionInspection {
        DynamicSubscriptionInspection {
            selected: self.selected,
            adapter: self.adapter_claim.entity(),
            source: self.connection.source(),
            target: self.connection.target(),
            slot: self.slot,
        }
    }
}

pub(crate) struct DynamicRuntimeProcess {
    pub(crate) wake_output: bool,
    subscriptions_changed: bool,
    pub(crate) cleanup: Vec<DynamicSignalSubscription>,
    registration_cleanup: crate::registration::RegistrationCleanup,
}

type DynamicCandidateResult = Result<Vec<crate::registration::CommittedDynamicCandidate>, SignalRegistrationError>;

type DynamicCandidateMaterializer =
    dyn FnOnce(
        &mut World,
        SignalEntity,
    ) -> Result<crate::registration::DynamicCandidateMaterialization, SignalRegistrationError>;
type DynamicPlanApplication = dyn FnOnce(&mut dyn core::any::Any, DynamicCandidateResult) -> DynamicRuntimeProcess;

// Only registration needs World access. Keeping that phase outside the runtime borrow lets
// lifecycle hooks drain the real subscriptions, including while candidate allocation is suspended.
struct DynamicManagerPlan {
    materialize: Box<DynamicCandidateMaterializer>,
    apply: Box<DynamicPlanApplication>,
}

impl DynamicManagerPlan {
    fn new<Runtime, Kind>(
        requests: Vec<crate::registration::DynamicCandidateRequest<Kind>>,
        same_identity: SameSignalIdentity,
        apply: impl FnOnce(&mut Runtime, Vec<crate::registration::CommittedDynamicCandidate>) -> DynamicRuntimeProcess
        + 'static,
    ) -> Self
    where
        Runtime: ErasedDynamicSignalRuntime,
        Kind: SignalKind,
        Kind::Event: Send + Sync,
    {
        Self {
            materialize: Box::new(move |world, owner| {
                crate::registration::materialize_dynamic_candidates(world, requests, owner, same_identity, |binding| {
                    move |In(event): In<Kind::Event>, world: &mut World| process_dynamic_adapter(world, &binding, event)
                })
            }),
            apply: Box::new(move |runtime, committed| {
                let runtime = runtime
                    .downcast_mut::<Runtime>()
                    .expect("dynamic plan runtime type changed");
                match committed {
                    Ok(committed) => apply(runtime, committed),
                    Err(error) => {
                        runtime.reject_candidate(error);
                        DynamicRuntimeProcess {
                            subscriptions_changed: false,
                            registration_cleanup: crate::registration::RegistrationCleanup::default(),
                            wake_output: false,
                            cleanup: Vec::new(),
                        }
                    }
                }
            }),
        }
    }
}

trait ErasedDynamicSignalRuntime: Send + Sync + 'static {
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any;
    fn reject_candidate(&mut self, error: SignalRegistrationError);
    fn append_subscriptions(&self, subscriptions: &mut Vec<DynamicSubscriptionInspection>);
    fn take_subscriptions(&mut self) -> Vec<DynamicSignalSubscription>;
    fn clear_queues(&mut self);
    fn on_deactivate(&mut self);
    fn prepare_manager(&mut self, update: Box<dyn core::any::Any>) -> Option<DynamicManagerPlan>;
    fn process_adapter(
        &mut self,
        binding: &DynamicAdapterBinding,
        event: Box<dyn core::any::Any>,
        defer_self_adapter: bool,
    ) -> DynamicRuntimeProcess;
    fn take_output(&mut self) -> Option<ErasedSignalValue>;
    fn has_after_output_cleanup(&self) -> bool;
    fn take_after_output_cleanup(&mut self) -> Vec<DynamicSignalSubscription>;
}

#[derive(Clone)]
pub(crate) struct DynamicOwnerEpoch {
    owner: SignalEntity,
    incarnation: Arc<OnceLock<SignalEntity>>,
    // The same component can deactivate and reactivate without being replaced.
    input_epoch: u64,
}

impl DynamicOwnerEpoch {
    pub(crate) fn capture(world: &World, owner: SignalEntity) -> Option<Self> {
        Some(Self {
            owner,
            incarnation: world.get::<DynamicSignalOwner>(*owner)?.incarnation.clone(),
            input_epoch: crate::graph::signal_input_epoch(world, owner),
        })
    }

    pub(crate) fn is_current(&self, world: &World) -> bool {
        world
            .get::<DynamicSignalOwner>(*self.owner)
            .is_some_and(|owner| Arc::ptr_eq(&owner.incarnation, &self.incarnation))
            && crate::graph::signal_input_epoch(world, self.owner) == self.input_epoch
    }
}

struct DynamicMaterializationScope<'world> {
    world: &'world mut World,
    epoch: DynamicOwnerEpoch,
}

impl Drop for DynamicMaterializationScope<'_> {
    fn drop(&mut self) {
        if self.epoch.is_current(self.world) {
            self.world
                .get_mut::<DynamicSignalOwner>(*self.epoch.owner)
                .expect("current dynamic owner")
                .materializing = false;
        }
    }
}

enum DynamicRuntimeInput {
    Manager(Box<dyn core::any::Any + Send + Sync>),
    Adapter(DynamicAdapterBinding, Box<dyn core::any::Any + Send + Sync>),
}

fn release_dynamic_signal_owner(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    let Some(mut owner) = world.get_mut::<DynamicSignalOwner>(entity) else {
        return;
    };
    let subscriptions = owner.drain();
    if !subscriptions.is_empty() {
        world
            .commands()
            .queue(move |world: &mut World| crate::graph::release_dynamic_subscriptions(world, subscriptions));
    }
}

#[derive(Component)]
#[component(on_discard = release_dynamic_signal_owner, on_remove = release_dynamic_signal_owner)]
pub(crate) struct DynamicSignalOwner {
    runtime: Box<dyn ErasedDynamicSignalRuntime>,
    subscriptions: Vec<DynamicSubscriptionInspection>,
    // Reuse the runner binding identity to invalidate plans and callbacks on component replacement.
    incarnation: Arc<OnceLock<SignalEntity>>,
    active: bool,
    materializing: bool,
    inputs: VecDeque<DynamicRuntimeInput>,
}

impl DynamicSignalOwner {
    fn new(runtime: Box<dyn ErasedDynamicSignalRuntime>) -> Self {
        let mut subscriptions = Vec::new();
        runtime.append_subscriptions(&mut subscriptions);
        Self {
            runtime,
            subscriptions,
            incarnation: Arc::new(OnceLock::new()),
            active: false,
            materializing: false,
            inputs: VecDeque::new(),
        }
    }

    pub(crate) fn mark_activated(&mut self) {
        self.active = true;
    }

    pub(crate) fn subscriptions(&self) -> Vec<DynamicSubscriptionInspection> {
        self.subscriptions.clone()
    }

    fn refresh_subscriptions(&mut self) {
        self.subscriptions.clear();
        self.runtime.append_subscriptions(&mut self.subscriptions);
    }

    pub(crate) fn drain(&mut self) -> Vec<DynamicSignalSubscription> {
        self.materializing = false;
        self.inputs.clear();
        self.runtime.clear_queues();
        if core::mem::replace(&mut self.active, false) {
            self.runtime.on_deactivate();
        }
        self.subscriptions.clear();
        self.runtime.take_subscriptions()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DynamicAdapterBinding {
    pub(crate) owner: SignalEntity,
    pub(crate) selected: SignalEntity,
    pub(crate) adapter: SignalEntity,
    pub(crate) slot: DynamicStableSlot,
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::{collections::VecDeque, format, string::String, vec};
    use bevy::prelude::{App, Local, MinimalPlugins, ResMut, Resource};
    use bevy_ecs::schedule::ScheduleLabel;

    use crate::{
        JonmoPlugin, ScalarSignalMaterializer,
        graph::{GraphSnapshot, SignalEntity, graph_snapshot, process_signal_graph_for_schedule},
        replay::{ReplayCacheState, ScalarReplayCache},
        signal::{self, BoxedReplayableSignal, Signal, SignalExt},
        signal_map::{BoxedReplayableSignalMap, MapDiff, SignalMap, SignalMapExt},
        signal_vec::{BoxedReplayableSignalVec, SignalVec, SignalVecExt, VecDiff},
    };

    type TestReplaySignal = BoxedReplayableSignal<i32>;

    struct MaterializedTestSignal<T: 'static> {
        materializer: ScalarSignalMaterializer<T>,
    }

    impl<T: 'static> Clone for MaterializedTestSignal<T> {
        fn clone(&self) -> Self {
            Self {
                materializer: self.materializer.shared_clone(),
            }
        }
    }

    impl<T: 'static> Signal for MaterializedTestSignal<T> {
        type Item = T;

        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
            self.materializer
        }
    }

    struct MaterializedTestVec(crate::VecSignalMaterializer<i32>);

    impl SignalVec for MaterializedTestVec {
        type Item = i32;

        fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<i32> {
            self.0
        }
    }

    struct MaterializedTestMap(crate::MapSignalMaterializer<i32, i32>);

    impl SignalMap for MaterializedTestMap {
        type Key = i32;
        type Value = i32;

        fn into_signal_map_materializer(self: Box<Self>) -> crate::MapSignalMaterializer<i32, i32> {
            self.0
        }
    }

    #[derive(Resource, Default)]
    struct PendingCollectionEvents {
        vectors: BTreeMap<usize, Vec<VecDiff<i32>>>,
        maps: BTreeMap<usize, Vec<MapDiff<i32, i32>>>,
    }

    fn pending_test_vector(index: usize) -> BoxedReplayableSignalVec<i32> {
        MaterializedTestVec(crate::VecSignalMaterializer::new(move |registrar| {
            let system = move |In(()), mut events: ResMut<PendingCollectionEvents>| events.vectors.remove(&index);
            if index == 1 {
                registrar.replayable_signal_vec_entity(
                    (),
                    system,
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            } else {
                registrar.replayable_signal_vec_source(
                    if index == 0 { vec![1] } else { vec![] },
                    system,
                    OutputDelivery::broadcast(),
                )
            }
        }))
        .cache()
        .boxed_replayable()
    }

    fn pending_test_map(index: usize) -> BoxedReplayableSignalMap<i32, i32> {
        MaterializedTestMap(crate::MapSignalMaterializer::new(move |registrar| {
            let system = move |In(()), mut events: ResMut<PendingCollectionEvents>| events.maps.remove(&index);
            if index == 1 {
                registrar.replayable_signal_map_entity(
                    (),
                    system,
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            } else {
                registrar.replayable_signal_map_source(
                    if index == 0 {
                        BTreeMap::from([(1, 10)])
                    } else {
                        BTreeMap::new()
                    },
                    system,
                    OutputDelivery::broadcast(),
                )
            }
        }))
        .cache()
        .boxed_replayable()
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash, ScheduleLabel)]
    struct UnregisteredDynamicSchedule;

    #[derive(Resource, Default)]
    struct ForwardSelectedValue(Option<i32>);

    #[derive(Resource, Default)]
    struct RetainedDynamicOutputs(Vec<i32>);

    struct EmptyDynamicRuntime;

    impl ErasedDynamicSignalRuntime for EmptyDynamicRuntime {
        fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
            self
        }

        fn reject_candidate(&mut self, error: SignalRegistrationError) {
            panic!("empty runtime rejected a candidate: {error:?}");
        }

        fn append_subscriptions(&self, _subscriptions: &mut Vec<DynamicSubscriptionInspection>) {}

        fn take_subscriptions(&mut self) -> Vec<DynamicSignalSubscription> {
            Vec::new()
        }

        fn clear_queues(&mut self) {}

        fn on_deactivate(&mut self) {}

        fn prepare_manager(&mut self, _update: Box<dyn core::any::Any>) -> Option<DynamicManagerPlan> {
            panic!("empty test runtime must not process manager updates")
        }

        fn process_adapter(
            &mut self,
            _binding: &DynamicAdapterBinding,
            _event: Box<dyn core::any::Any>,
            _defer_self_adapter: bool,
        ) -> DynamicRuntimeProcess {
            panic!("empty test runtime must not process adapter events")
        }

        fn take_output(&mut self) -> Option<ErasedSignalValue> {
            None
        }

        fn has_after_output_cleanup(&self) -> bool {
            false
        }

        fn take_after_output_cleanup(&mut self) -> Vec<DynamicSignalSubscription> {
            Vec::new()
        }
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        app
    }

    fn owner_subscriptions(world: &World, owner: SignalEntity) -> Vec<DynamicSubscriptionInspection> {
        world
            .get::<DynamicSignalOwner>(*owner)
            .unwrap_or_else(|| panic!("dynamic output {owner:?} should have an owner runtime"))
            .subscriptions()
    }

    fn process_test_update(
        world: &mut World,
        owner: SignalEntity,
        update: Box<dyn core::any::Any>,
    ) -> DynamicRuntimeProcess {
        let Some(plan) = world
            .get_mut::<DynamicSignalOwner>(*owner)
            .expect("test owner")
            .runtime
            .prepare_manager(update)
        else {
            return DynamicRuntimeProcess {
                subscriptions_changed: false,
                registration_cleanup: crate::registration::RegistrationCleanup::default(),
                wake_output: false,
                cleanup: Vec::new(),
            };
        };
        let epoch = DynamicOwnerEpoch::capture(world, owner).expect("test owner epoch");
        apply_dynamic_plan(world, epoch, plan).expect("test owner was not retired")
    }

    fn queued_scalar_output(world: &mut World, owner: SignalEntity) -> Vec<i32> {
        let output = world
            .get_mut::<DynamicSignalOwner>(*owner)
            .unwrap_or_else(|| panic!("dynamic output {owner:?} should have an owner runtime"))
            .runtime
            .take_output()
            .unwrap_or_else(|| panic!("test reducer should always return the drained queue"));
        *output
            .downcast::<Vec<i32>>()
            .unwrap_or_else(|_| panic!("dynamic output queue should contain i32 values"))
    }

    fn scalar_replay_state<T>(world: &World, signal: SignalEntity) -> ReplayCacheState<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        world
            .get::<ScalarReplayCache<T>>(*signal)
            .unwrap_or_else(|| panic!("signal {signal:?} should have a scalar replay cache"))
            .snapshot()
            .cloned()
    }

    fn assert_adapter_routes(
        snapshot: &GraphSnapshot,
        owner: SignalEntity,
        subscriptions: &[DynamicSubscriptionInspection],
    ) {
        for subscription in subscriptions {
            assert_eq!(subscription.source, subscription.adapter);
            assert_eq!(subscription.target, owner);
            assert_eq!(
                snapshot
                    .node(*subscription.adapter)
                    .unwrap_or_else(|| panic!("adapter {:?} should remain materialized", subscription.adapter))
                    .active_downstreams,
                vec![*owner],
            );
        }
    }

    #[test]
    fn rejected_runtime_candidates_preserve_committed_dynamic_state() {
        type Update = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;

        let errors = Arc::new(Mutex::new(Vec::new()));
        let owner_materializer = Arc::new(OnceLock::<ScalarSignalMaterializer<Vec<i32>>>::new());

        let selected = signal::once(7).cache().boxed_replayable();
        let unregistered = signal::once(8)
            .schedule::<UnregisteredDynamicSchedule>()
            .cache()
            .boxed_replayable();
        let cyclic = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::new({
                let owner_materializer = owner_materializer.clone();
                move |registrar| {
                    let owner = owner_materializer
                        .get()
                        .unwrap_or_else(|| panic!("dynamic owner materializer should be bound"))
                        .shared_clone();
                    let owner = registrar.materialize_signal(owner)?;
                    registrar.signal_entity(owner, |In(_): In<Vec<i32>>| Some(9), OutputDelivery::broadcast())
                }
            }),
        }
        .cache()
        .boxed_replayable();
        let updates = Arc::new(Mutex::new(VecDeque::from([
            Update::new(
                [ScalarSlotCommand::Select { signal: selected }],
                DynamicEmissions::One(10),
            ),
            Update::new(
                [ScalarSlotCommand::Select { signal: unregistered }],
                DynamicEmissions::One(20),
            ),
            Update::new(
                [ScalarSlotCommand::Select { signal: cyclic }],
                DynamicEmissions::One(30),
            ),
        ])));

        let materializer = ScalarSignalMaterializer::<Vec<i32>>::new({
            let errors = errors.clone();
            move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        updates
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .pop_front()
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .replayable_scalar_output(Some)
                        .same_signal_identity(SameSignalIdentity::Restart)
                        .on_candidate_error(move |error| {
                            errors
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .push(error);
                        }),
                )
            }
        });
        assert!(owner_materializer.set(materializer.shared_clone()).is_ok());

        let mut app = test_app();
        let claim = materializer
            .try_register(app.world_mut())
            .expect("dynamic operator should register");
        let owner = claim.entity();
        app.update();

        let committed_subscriptions = owner_subscriptions(app.world(), owner);
        assert_eq!(committed_subscriptions.len(), 1);
        assert_eq!(
            committed_subscriptions[0].slot,
            DynamicStableSlot {
                identity: 1,
                generation: 1
            }
        );
        assert_eq!(queued_scalar_output(app.world_mut(), owner), Vec::<i32>::new());
        let committed_replay = scalar_replay_state::<Vec<i32>>(app.world(), owner);
        assert!(matches!(
            &committed_replay,
            ReplayCacheState::Ready { current, .. } if current == &vec![10, 7]
        ));
        let committed_graph = graph_snapshot(app.world_mut());

        app.update();
        assert_eq!(
            errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            &[SignalRegistrationError::ScheduleNotRegistered],
        );
        assert_eq!(graph_snapshot(app.world_mut()), committed_graph);
        assert_eq!(owner_subscriptions(app.world(), owner), committed_subscriptions);
        assert_eq!(queued_scalar_output(app.world_mut(), owner), Vec::<i32>::new());
        assert_eq!(scalar_replay_state::<Vec<i32>>(app.world(), owner), committed_replay);

        app.update();
        assert_eq!(
            errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            &[
                SignalRegistrationError::ScheduleNotRegistered,
                SignalRegistrationError::CycleDetected,
            ],
        );
        assert_eq!(graph_snapshot(app.world_mut()), committed_graph);
        assert_eq!(owner_subscriptions(app.world(), owner), committed_subscriptions);
        assert_eq!(queued_scalar_output(app.world_mut(), owner), Vec::<i32>::new());
        assert_eq!(scalar_replay_state::<Vec<i32>>(app.world(), owner), committed_replay);

        claim.release(app.world_mut());
    }

    #[test]
    fn default_candidate_rejection_discards_before_panicking_and_preserves_current_generation() {
        type Update = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;

        let selected = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_source(
                    7,
                    |In(()), mut next: ResMut<ForwardSelectedValue>| next.0.take(),
                    OutputDelivery::broadcast(),
                )
            }),
        }
        .cache()
        .boxed_replayable();
        let invalid = signal::once(8)
            .schedule::<UnregisteredDynamicSchedule>()
            .cache()
            .boxed_replayable();
        let updates = Arc::new(Mutex::new(VecDeque::from([
            Update::new([ScalarSlotCommand::Select { signal: selected }], DynamicEmissions::None),
            Update::new(
                [ScalarSlotCommand::Select { signal: invalid }],
                DynamicEmissions::One(99),
            ),
        ])));
        let materializer = ScalarSignalMaterializer::<Vec<i32>>::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .replayable_scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        });

        let mut app = test_app();
        app.init_resource::<ForwardSelectedValue>();
        let claim = materializer.register(app.world_mut());
        let owner = claim.entity();
        app.update();

        let committed_graph = graph_snapshot(app.world_mut());
        let committed_subscriptions = owner_subscriptions(app.world(), owner);
        let committed_replay = scalar_replay_state::<Vec<i32>>(app.world(), owner);
        let committed_entity_count = app.world().entities().len();
        assert_eq!(queued_scalar_output(app.world_mut(), owner), Vec::<i32>::new());

        let mut process_signals = process_signal_graph_for_schedule(bevy_app::Last.intern());
        let panic = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            process_signals(app.world_mut());
        }))
        .expect_err("the default candidate policy must panic after discarding an invalid candidate");
        let panic_message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic payload");
        assert!(panic_message.contains("ScheduleNotRegistered"));
        assert_eq!(app.world().entities().len(), committed_entity_count);
        assert_eq!(graph_snapshot(app.world_mut()), committed_graph);
        assert_eq!(owner_subscriptions(app.world(), owner), committed_subscriptions);
        assert_eq!(scalar_replay_state::<Vec<i32>>(app.world(), owner), committed_replay);
        assert_eq!(queued_scalar_output(app.world_mut(), owner), Vec::<i32>::new());

        app.world_mut().resource_mut::<ForwardSelectedValue>().0 = Some(9);
        process_signals(app.world_mut());
        assert!(matches!(
            scalar_replay_state::<Vec<i32>>(app.world(), owner),
            ReplayCacheState::Ready { current, .. } if current == vec![9]
        ));
        assert_eq!(owner_subscriptions(app.world(), owner), committed_subscriptions);
        claim.release(app.world_mut());
    }

    #[test]
    fn repeated_selected_identity_uses_independent_indexed_and_keyed_routes() {
        type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, (usize, i32)>;

        let selected = signal::once(5).cache().broadcast();
        let indexed_updates = Arc::new(Mutex::new(VecDeque::from([
            IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![selected.clone().boxed_replayable(), selected.boxed_replayable()],
                }],
                DynamicEmissions::None,
            ),
            IndexedUpdate::new([IndexedSlotCommand::Remove { index: 0 }], DynamicEmissions::None),
        ])));
        let indexed = ScalarSignalMaterializer::<Vec<(usize, i32)>>::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    indexed_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        });
        let mut indexed_app = test_app();
        let indexed_claim = indexed
            .try_register(indexed_app.world_mut())
            .expect("indexed dynamic operator should register");
        let indexed_owner = indexed_claim.entity();
        indexed_app.update();

        let indexed_subscriptions = owner_subscriptions(indexed_app.world(), indexed_owner);
        assert_eq!(indexed_subscriptions.len(), 2);
        assert_eq!(indexed_subscriptions[0].selected, indexed_subscriptions[1].selected);
        assert_ne!(indexed_subscriptions[0].adapter, indexed_subscriptions[1].adapter);
        assert_ne!(indexed_subscriptions[0].slot, indexed_subscriptions[1].slot);
        assert_adapter_routes(
            &graph_snapshot(indexed_app.world_mut()),
            indexed_owner,
            &indexed_subscriptions,
        );

        indexed_app.update();
        let indexed_remaining = owner_subscriptions(indexed_app.world(), indexed_owner);
        assert_eq!(indexed_remaining, vec![indexed_subscriptions[1]]);
        let indexed_after_removal = graph_snapshot(indexed_app.world_mut());
        assert_adapter_routes(&indexed_after_removal, indexed_owner, &indexed_remaining);
        assert!(
            indexed_after_removal
                .node(*indexed_subscriptions[0].adapter)
                .is_none_or(|node| !node.active_downstreams.contains(&*indexed_owner))
        );
        indexed_claim.release(indexed_app.world_mut());

        type KeyedUpdate = DynamicManagerUpdate<KeyedSlotCommand<&'static str, TestReplaySignal>, (&'static str, i32)>;

        let selected = signal::once(6).cache().broadcast();
        let keyed_updates = Arc::new(Mutex::new(VecDeque::from([
            KeyedUpdate::new(
                [KeyedSlotCommand::ReplaceAll {
                    entries: vec![
                        ("first", selected.clone().boxed_replayable()),
                        ("second", selected.boxed_replayable()),
                    ],
                }],
                DynamicEmissions::None,
            ),
            KeyedUpdate::new([KeyedSlotCommand::Remove { key: "first" }], DynamicEmissions::None),
        ])));
        let keyed = ScalarSignalMaterializer::<Vec<(&'static str, i32)>>::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    keyed_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One((key, value)))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        });
        let mut keyed_app = test_app();
        let keyed_claim = keyed
            .try_register(keyed_app.world_mut())
            .expect("keyed dynamic operator should register");
        let keyed_owner = keyed_claim.entity();
        keyed_app.update();

        let keyed_subscriptions = owner_subscriptions(keyed_app.world(), keyed_owner);
        assert_eq!(keyed_subscriptions.len(), 2);
        assert_eq!(keyed_subscriptions[0].selected, keyed_subscriptions[1].selected);
        assert_ne!(keyed_subscriptions[0].adapter, keyed_subscriptions[1].adapter);
        assert_ne!(keyed_subscriptions[0].slot, keyed_subscriptions[1].slot);
        assert_adapter_routes(
            &graph_snapshot(keyed_app.world_mut()),
            keyed_owner,
            &keyed_subscriptions,
        );

        keyed_app.update();
        let keyed_remaining = owner_subscriptions(keyed_app.world(), keyed_owner);
        assert_eq!(keyed_remaining, vec![keyed_subscriptions[1]]);
        let keyed_after_removal = graph_snapshot(keyed_app.world_mut());
        assert_adapter_routes(&keyed_after_removal, keyed_owner, &keyed_remaining);
        assert!(
            keyed_after_removal
                .node(*keyed_subscriptions[0].adapter)
                .is_none_or(|node| !node.active_downstreams.contains(&*keyed_owner))
        );
        keyed_claim.release(keyed_app.world_mut());
    }

    #[test]
    fn same_batch_keep_current_normalization_preserves_current_topology_and_reaps_transient_candidates() {
        type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, i32>;

        let current = signal::once(1).cache().broadcast();
        let replacement = signal::once(2).cache().boxed_replayable();
        let indexed_updates = Arc::new(Mutex::new(VecDeque::from([
            IndexedUpdate::new(
                [IndexedSlotCommand::Insert {
                    index: 0,
                    signal: current.clone().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ),
            IndexedUpdate::sequenced([
                DynamicManagerStep::command(
                    IndexedSlotCommand::Replace {
                        index: 0,
                        signal: replacement,
                    },
                    DynamicEmissions::One(10),
                ),
                DynamicManagerStep::command(
                    IndexedSlotCommand::Replace {
                        index: 0,
                        signal: current.boxed_replayable(),
                    },
                    DynamicEmissions::One(20),
                ),
            ]),
        ])));
        let indexed = ScalarSignalMaterializer::<Vec<i32>>::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    indexed_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|_index, value| DynamicEmissions::One(value))
                    .replayable_scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        });
        let mut indexed_app = test_app();
        let indexed_claim = indexed
            .try_register(indexed_app.world_mut())
            .expect("indexed dynamic operator should register");
        let indexed_owner = indexed_claim.entity();
        indexed_app.update();
        let indexed_current = owner_subscriptions(indexed_app.world(), indexed_owner);
        assert_eq!(indexed_current.len(), 1);
        let indexed_current_adapter = indexed_current[0].adapter;
        let indexed_current_claim_count = graph_snapshot(indexed_app.world_mut())
            .node(*indexed_current_adapter)
            .and_then(|node| node.claim_count);
        indexed_app.update();
        indexed_app.update();
        let indexed_after = owner_subscriptions(indexed_app.world(), indexed_owner);
        assert_eq!(indexed_after, indexed_current);
        let indexed_graph = graph_snapshot(indexed_app.world_mut());
        assert_adapter_routes(&indexed_graph, indexed_owner, &indexed_after);
        assert_eq!(
            indexed_graph
                .node(*indexed_current_adapter)
                .and_then(|node| node.claim_count),
            indexed_current_claim_count,
        );
        assert!(matches!(
            scalar_replay_state::<Vec<i32>>(indexed_app.world(), indexed_owner),
            ReplayCacheState::Ready { current, .. } if current == vec![10, 20]
        ));
        indexed_claim.release(indexed_app.world_mut());

        type KeyedUpdate = DynamicManagerUpdate<KeyedSlotCommand<&'static str, TestReplaySignal>, i32>;

        let current = signal::once(3).cache().broadcast();
        let replacement = signal::once(4).cache().boxed_replayable();
        let keyed_updates = Arc::new(Mutex::new(VecDeque::from([
            KeyedUpdate::new(
                [KeyedSlotCommand::Insert {
                    key: "slot",
                    signal: current.clone().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ),
            KeyedUpdate::sequenced([
                DynamicManagerStep::command(
                    KeyedSlotCommand::Replace {
                        key: "slot",
                        signal: replacement,
                    },
                    DynamicEmissions::One(30),
                ),
                DynamicManagerStep::command(
                    KeyedSlotCommand::Replace {
                        key: "slot",
                        signal: current.boxed_replayable(),
                    },
                    DynamicEmissions::One(40),
                ),
            ]),
        ])));
        let keyed = ScalarSignalMaterializer::<Vec<i32>>::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    keyed_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|_key, value| DynamicEmissions::One(value))
                    .replayable_scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        });
        let mut keyed_app = test_app();
        let keyed_claim = keyed
            .try_register(keyed_app.world_mut())
            .expect("keyed dynamic operator should register");
        let keyed_owner = keyed_claim.entity();
        keyed_app.update();
        let keyed_current = owner_subscriptions(keyed_app.world(), keyed_owner);
        assert_eq!(keyed_current.len(), 1);
        let keyed_current_adapter = keyed_current[0].adapter;
        let keyed_current_claim_count = graph_snapshot(keyed_app.world_mut())
            .node(*keyed_current_adapter)
            .and_then(|node| node.claim_count);
        keyed_app.update();
        keyed_app.update();
        let keyed_after = owner_subscriptions(keyed_app.world(), keyed_owner);
        assert_eq!(keyed_after, keyed_current);
        let keyed_graph = graph_snapshot(keyed_app.world_mut());
        assert_adapter_routes(&keyed_graph, keyed_owner, &keyed_after);
        assert_eq!(
            keyed_graph
                .node(*keyed_current_adapter)
                .and_then(|node| node.claim_count),
            keyed_current_claim_count,
        );
        assert!(matches!(
            scalar_replay_state::<Vec<i32>>(keyed_app.world(), keyed_owner),
            ReplayCacheState::Ready { current, .. } if current == vec![30, 40]
        ));
        keyed_claim.release(keyed_app.world_mut());
    }

    #[test]
    fn pending_collection_batches_execute_stateful_adapters_only_in_command_order() {
        macro_rules! check_replay {
            ($replay:ident::<$($item:ty),+>, $source:ident, $field:ident, $baseline:expr, $deltas:expr, $warming:expr, $empty:expr) => {
                macro_rules! check_policy {
                    ($policy:ident, $command:expr, $adapter_context:expr, $expected_context:expr) => {{
                        let mut app = test_app();
                        app.init_resource::<PendingCollectionEvents>();
                        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
                        let materializer = ScalarSignalMaterializer::<Vec<String>>::new({
                            let calls = calls.clone();
                            move |registrar| {
                                let mut sources = Some((0..3).map($source).collect::<Vec<_>>());
                                let manager = registrar.signal_entity(
                                    (),
                                    move |In(())| {
                                        let sources = sources.take()?;
                                        Some(DynamicManagerUpdate::sequenced(sources.into_iter().enumerate().map(
                                            |(index, source)| {
                                                DynamicManagerStep::command(
                                                    $command(index, source),
                                                    DynamicEmissions::One(format!("manager:{index}")),
                                                )
                                            },
                                        )))
                                    },
                                    OutputDelivery::move_only(),
                                )?;
                                let mut invocation = 0;
                                registrar.dynamic_operator(
                                    DynamicSignalOperatorSpec::$policy(manager)
                                        .$replay::<$($item),+>()
                                        .adapter(move |context, event| {
                                            let entry = format!("{}:{}:{event:?}", invocation, $adapter_context(context));
                                            invocation += 1;
                                            calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(entry.clone());
                                            DynamicEmissions::One(entry)
                                        })
                                        .replayable_scalar_output(Some)
                                        .same_signal_identity(SameSignalIdentity::KeepCurrent),
                                )
                            }
                        });
                        let claim = materializer.register(app.world_mut());
                        let owner = claim.entity();
                        app.update();
                        assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                        let deltas = $deltas;
                        for batch in deltas.iter().cloned() {
                            app.world_mut().resource_mut::<PendingCollectionEvents>().$field.insert(0, batch);
                            app.update();
                            assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                        }
                        let waiting = owner_subscriptions(app.world(), owner);
                        assert_eq!(waiting.len(), 3);
                        app.world_mut().resource_mut::<PendingCollectionEvents>().$field.insert(1, $warming);
                        app.update();
                        let mut expected_calls = vec![format!("0:{}:{:?}", $expected_context(0), $baseline)];
                        for (index, batch) in deltas.iter().enumerate() {
                            expected_calls.push(format!("{}:{}:{batch:?}", index + 1, $expected_context(0)));
                        }
                        let mut expected = vec![String::from("manager:0")];
                        expected.extend(expected_calls.clone());
                        expected.push(String::from("manager:1"));
                        let warming = format!("{}:{}:{:?}", expected_calls.len(), $expected_context(1), $warming);
                        expected_calls.push(warming.clone());
                        expected.push(warming);
                        expected.push(String::from("manager:2"));
                        let empty = format!("{}:{}:{:?}", expected_calls.len(), $expected_context(2), $empty);
                        expected_calls.push(empty.clone());
                        expected.push(empty);
                        assert_eq!(*calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), expected_calls);
                        assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner),
                            ReplayCacheState::Ready { current, .. } if current == expected));
                        let published = owner_subscriptions(app.world(), owner);
                        assert_adapter_routes(&graph_snapshot(app.world_mut()), owner, &published);
                        for subscription in &waiting {
                            if !published.contains(subscription) {
                                assert!(app.world().get_entity(*subscription.adapter).is_err());
                            }
                        }
                        for _ in 0..3 { app.update(); }
                        assert_eq!(*calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), expected_calls);
                        claim.release(app.world_mut());
                        app.world_mut().flush();
                        crate::cleanup::drain_world_cleanup(app.world_mut());
                        for subscription in waiting {
                            assert!(app.world().get_entity(*subscription.adapter).is_err());
                        }
                    }};
                }
                check_policy!(scalar, |_, signal| ScalarSlotCommand::Select { signal }, |()| 0usize, |_| 0usize);
                check_policy!(indexed, |index, signal| IndexedSlotCommand::Insert { index, signal }, |index: usize| index, |index: usize| index);
                check_policy!(keyed, |index, signal| KeyedSlotCommand::Insert { key: 9usize - index, signal }, |key: usize| key, |index: usize| 9 - index);
            };
        }
        check_replay!(
            replay_signal_vec::<i32>,
            pending_test_vector,
            vectors,
            vec![VecDiff::Replace { values: vec![1] }],
            vec![
                vec![VecDiff::Push { value: 2 }],
                vec![VecDiff::UpdateAt { index: 0, value: 3 }, VecDiff::RemoveAt { index: 1 }]
            ],
            vec![VecDiff::Replace { values: vec![9] }],
            vec![VecDiff::<i32>::Replace { values: vec![] }]
        );
        check_replay!(
            replay_signal_map::<i32, i32>,
            pending_test_map,
            maps,
            vec![MapDiff::Replace { entries: vec![(1, 10)] }],
            vec![
                vec![MapDiff::Insert { key: 2, value: 20 }],
                vec![MapDiff::Update { key: 1, value: 30 }, MapDiff::Remove { key: 2 }]
            ],
            vec![MapDiff::Replace { entries: vec![(9, 90)] }],
            vec![MapDiff::<i32, i32>::Replace { entries: vec![] }]
        );
    }

    #[test]
    fn cancelling_accumulated_collection_batches_reaps_candidates_and_rejects_stale_generations() {
        macro_rules! check_replay {
            ($replay:ident::<$($item:ty),+>, $cache:ty, $source:ident, $field:ident, $baseline:expr, $deltas:expr, $warming:expr) => {
                macro_rules! check_policy {
                    ($policy:ident, $command:expr, $clear:expr, $context:expr, $expected_context:expr, $published_count:expr) => {{
                        let mut app = test_app();
                        app.init_resource::<PendingCollectionEvents>();
                        let updates = Arc::new(Mutex::new(VecDeque::from([
                            DynamicManagerUpdate::sequenced((0..2).map(|index| {
                                DynamicManagerStep::command(
                                    $command(index, $source(index)),
                                    DynamicEmissions::One(format!("manager:{index}")),
                                )
                            })),
                        ])));
                        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
                        let materializer = ScalarSignalMaterializer::<Vec<String>>::new({
                            let updates = updates.clone();
                            let calls = calls.clone();
                            move |registrar| {
                                let manager = registrar.signal_entity(
                                    (),
                                    move |In(())| {
                                        updates
                                            .lock()
                                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                                            .pop_front()
                                    },
                                    OutputDelivery::move_only(),
                                )?;
                                let mut invocation = 0;
                                registrar.dynamic_operator(
                                    DynamicSignalOperatorSpec::$policy(manager)
                                        .$replay::<$($item),+>()
                                        .adapter(move |context, event| {
                                            let entry = format!("{}:{}:{event:?}", invocation, $context(context));
                                            invocation += 1;
                                            calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(entry.clone());
                                            DynamicEmissions::One(entry)
                                        })
                                        .replayable_scalar_output(Some)
                                        .same_signal_identity(SameSignalIdentity::KeepCurrent),
                                )
                            }
                        });
                        let claim = materializer.register(app.world_mut());
                        let owner = claim.entity();
                        app.update();
                        assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                        let deltas = $deltas;
                        for batch in deltas.iter().cloned() {
                            app.world_mut().resource_mut::<PendingCollectionEvents>().$field.insert(0, batch);
                            app.update();
                            assert!(app.world().resource::<PendingCollectionEvents>().$field.is_empty());
                            assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                        }
                        let cancelled = owner_subscriptions(app.world(), owner);
                        assert_eq!(cancelled.len(), 2);
                        assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner), ReplayCacheState::Warming));
                        let buffered = cancelled.first().expect("ready first candidate");
                        let revision = match app.world().get::<$cache>(*buffered.selected).expect("selected replay cache").snapshot() {
                            crate::replay::ReplayCacheSnapshot::Ready { revision, .. } => revision,
                            crate::replay::ReplayCacheSnapshot::Warming => panic!("first candidate must be ready"),
                        };
                        assert_eq!(revision.get(), 1 + u64::try_from(deltas.len()).expect("bounded batch count"));
                        let waiting = graph_snapshot(app.world_mut());
                        let relationships = &waiting.node(*buffered.selected).expect("ready source").replay_relationships;
                        let [relationship] = relationships.as_slice() else { panic!("one replay relationship required") };
                        assert_eq!(relationship.dependent, *buffered.adapter);
                        assert!(matches!(relationship.state,
                            crate::graph::ReplayRelationshipSnapshotState::Published { cursor } if cursor == revision));

                        updates.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push_back(
                            DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                                $clear, DynamicEmissions::One(String::from("manager:clear")),
                            )]),
                        );
                        app.update();
                        assert!(owner_subscriptions(app.world(), owner).is_empty());
                        assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                        let expected_cancelled = vec![String::from("manager:0"), String::from("manager:1"), String::from("manager:clear")];
                        assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner),
                            ReplayCacheState::Ready { current, .. } if current == expected_cancelled));
                        for subscription in &cancelled {
                            assert!(app.world().get_entity(*subscription.adapter).is_err());
                            assert!(app.world().get_entity(*subscription.selected).is_err());
                        }

                        let after_cancel = graph_snapshot(app.world_mut());
                        for subscription in &cancelled {
                            let binding = DynamicAdapterBinding {
                                owner,
                                selected: subscription.selected,
                                adapter: subscription.adapter,
                                slot: subscription.slot,
                            };
                            let stale = app.world_mut().get_mut::<DynamicSignalOwner>(*binding.owner).expect("dynamic owner").runtime.process_adapter(
                                &binding, Box::new($warming), false,
                            );
                            assert!(!stale.wake_output);
                            assert!(stale.cleanup.is_empty());
                        }
                        let output = app.world_mut().get_mut::<DynamicSignalOwner>(*owner).expect("dynamic owner").runtime.take_output().expect("test reducer returns the queue");
                        let queue = (output as Box<dyn core::any::Any>).downcast::<Vec<String>>().expect("string queue");
                        assert!(queue.is_empty());
                        crate::cleanup::drain_world_cleanup(app.world_mut());
                        assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);
                        assert_eq!(graph_snapshot(app.world_mut()), after_cancel);

                        updates.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push_back(
                            DynamicManagerUpdate::sequenced((0..2).map(|index| {
                                DynamicManagerStep::command(
                                    $command(index, $source(index)),
                                    DynamicEmissions::One(format!("manager:new:{index}")),
                                )
                            })),
                        );
                        app.update();
                        let replacement = owner_subscriptions(app.world(), owner);
                        assert_eq!(replacement.len(), 2);
                        for subscription in &replacement {
                            assert!(cancelled.iter().all(|old| subscription.slot.generation > old.slot.generation));
                        }
                        for subscription in &cancelled {
                            let binding = DynamicAdapterBinding {
                                owner,
                                selected: subscription.selected,
                                adapter: subscription.adapter,
                                slot: subscription.slot,
                            };
                            let stale = app.world_mut().get_mut::<DynamicSignalOwner>(*binding.owner).expect("dynamic owner").runtime.process_adapter(
                                &binding, Box::new($warming), false,
                            );
                            assert!(!stale.wake_output);
                            assert!(stale.cleanup.is_empty());
                        }
                        assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                        assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner),
                            ReplayCacheState::Ready { current, .. } if current == expected_cancelled));
                        app.world_mut().resource_mut::<PendingCollectionEvents>().$field.insert(1, $warming);
                        app.update();
                        let first = format!("0:{}:{:?}", $expected_context(0), $baseline);
                        let second = format!("1:{}:{:?}", $expected_context(1), $warming);
                        let expected_calls = vec![first.clone(), second.clone()];
                        assert_eq!(*calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), expected_calls);
                        let expected = vec![String::from("manager:new:0"), first, String::from("manager:new:1"), second];
                        assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner),
                            ReplayCacheState::Ready { current, .. } if current == expected));
                        let published = owner_subscriptions(app.world(), owner);
                        assert_eq!(published.len(), $published_count);
                        assert_adapter_routes(&graph_snapshot(app.world_mut()), owner, &published);
                        for _ in 0..3 { app.update(); }
                        assert_eq!(*calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), expected_calls);
                        claim.release(app.world_mut());
                        app.world_mut().flush();
                        crate::cleanup::drain_world_cleanup(app.world_mut());
                        for subscription in replacement {
                            assert!(app.world().get_entity(*subscription.adapter).is_err());
                            assert!(app.world().get_entity(*subscription.selected).is_err());
                        }
                    }};
                }
                check_policy!(scalar, |_, signal| ScalarSlotCommand::Select { signal }, ScalarSlotCommand::Clear, |()| 0usize, |_| 0usize, 1);
                check_policy!(indexed, |index, signal| IndexedSlotCommand::Insert { index, signal }, IndexedSlotCommand::Clear, |index: usize| index, |index: usize| index, 2);
                check_policy!(keyed, |index, signal| KeyedSlotCommand::Insert { key: 9usize - index, signal }, KeyedSlotCommand::Clear, |key: usize| key, |index: usize| 9 - index, 2);
            };
        }
        check_replay!(
            replay_signal_vec::<i32>,
            crate::replay::VecReplayCache<i32>,
            pending_test_vector,
            vectors,
            vec![VecDiff::Replace { values: vec![1] }],
            vec![
                vec![VecDiff::Push { value: 2 }],
                vec![VecDiff::UpdateAt { index: 0, value: 3 }, VecDiff::RemoveAt { index: 1 }]
            ],
            vec![VecDiff::Replace { values: vec![9] }]
        );
        check_replay!(replay_signal_map::<i32, i32>, crate::replay::MapReplayCache<i32, i32>, pending_test_map, maps,
            vec![MapDiff::Replace { entries: vec![(1, 10)] }],
            vec![vec![MapDiff::Insert { key: 2, value: 20 }], vec![MapDiff::Update { key: 1, value: 30 }, MapDiff::Remove { key: 2 }]],
            vec![MapDiff::Replace { entries: vec![(9, 90)] }]);
    }

    #[test]
    fn command_scoped_supersession_preserves_unrelated_transient_ownership_and_cursors() {
        macro_rules! check_replay {
            ($signal:ty, $replay:ident::<$($item:ty),+>, $source:ident, $field:ident, $ready:expr, $warming:expr, $empty:expr) => {
                macro_rules! check_policy {
                    ($policy:ident, $command:ty, $select:expr, $remove:expr, $unrelated:expr, $clear:expr, $empty_replace:expr, $replace:expr, $targeted_replace:expr, $scalar:expr, $context:expr) => {
                        for supersession in 0..7 {
                            let same_lineage = supersession >= 5;
                            let cancels_transient = supersession != 0 && (supersession != 4 || $scalar);
                            let replaces = (3..=5).contains(&supersession);
                            let published_empty = matches!(supersession, 1 | 2 | 6);
                            let mut app = test_app();
                            app.init_resource::<PendingCollectionEvents>();
                            type Update = DynamicManagerUpdate<$command, String>;
                            let calls = Arc::new(Mutex::new(Vec::<String>::new()));
                            let materializer = ScalarSignalMaterializer::<Vec<String>>::new({
                                let calls = calls.clone();
                                move |registrar| {
                                    let manager =
                                        registrar.signal_entity((), |In(())| None::<Update>, OutputDelivery::move_only())?;
                                    registrar.dynamic_operator(
                                        DynamicSignalOperatorSpec::$policy(manager)
                                            .$replay::<$($item),+>()
                                            .adapter(move |context, event| {
                                                let entry = format!("{}:{event:?}", $context(context));
                                                calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(entry.clone());
                                                DynamicEmissions::One(entry)
                                            })
                                            .replayable_scalar_output(Some)
                                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                                    )
                                }
                            });
                            let claim = materializer.register(app.world_mut());
                            let owner_entity = claim.entity();

                            let process_update = |app: &mut App, update: Update| {
                                let process = process_test_update(
                                    app.world_mut(), owner_entity, Box::new(update),
                                );
                                assert!(process.cleanup.is_empty());
                                finish_dynamic_process(app.world_mut(), owner_entity, process)
                            };
                            let no_ops = || [
                                Update::new([], DynamicEmissions::None),
                                Update::new([], DynamicEmissions::Many(Vec::new())),
                                Update::sequenced([]),
                                Update::sequenced([
                                    DynamicManagerStep::emissions(DynamicEmissions::None),
                                    DynamicManagerStep::emissions(DynamicEmissions::Many(Vec::new())),
                                ]),
                            ];
                            let empty_graph = graph_snapshot(app.world_mut());
                            for update in no_ops() {
                                assert!(!process_update(&mut app, update));
                                assert_eq!(graph_snapshot(app.world_mut()), empty_graph);
                                assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner_entity), ReplayCacheState::Warming));
                            }
                            // Removal/reinsertion delimits a lineage; replacement does not.
                            let middle = if same_lineage {
                                DynamicManagerStep::emissions(DynamicEmissions::One(String::from("remove")))
                            } else {
                                DynamicManagerStep::command($remove, DynamicEmissions::One(String::from("remove")))
                            };
                            let desired_command = if same_lineage { $targeted_replace($source(0)) } else { $select($source(0)) };
                            assert!(!process_update(&mut app, Update::sequenced([
                                DynamicManagerStep::command($select($source(1)), DynamicEmissions::One(String::from("transient"))),
                                middle,
                                DynamicManagerStep::command(desired_command, DynamicEmissions::One(String::from("desired"))),
                            ])));
                            app.update();
                            let subscriptions = owner_subscriptions(app.world(), owner_entity);
                            let [transient, desired] = subscriptions.as_slice() else { panic!("two pending subscriptions required") };
                            assert_eq!(transient.slot.generation, 1);
                            assert_eq!(desired.slot.generation, 2);
                            let before = graph_snapshot(app.world_mut());
                            assert_adapter_routes(&before, owner_entity, &subscriptions);
                            assert!(matches!(before.node(*desired.selected).expect("ready selected").replay_relationships
                                .first().expect("ready cursor").state, crate::graph::ReplayRelationshipSnapshotState::Published { .. }));
                            assert!(matches!(before.node(*transient.selected).expect("warming selected").replay_relationships
                                .first().expect("warming cursor").state, crate::graph::ReplayRelationshipSnapshotState::Pending));
                            for subscription in &subscriptions {
                                assert_eq!(before.node(*subscription.adapter).expect("owned adapter").claim_count, Some(1));
                            }
                            for _ in 0..3 {
                                for update in no_ops().into_iter().chain([
                                    Update::new([], DynamicEmissions::One(String::from("emission"))),
                                    Update::new($unrelated, DynamicEmissions::None),
                                ]) {
                                    assert!(!process_update(&mut app, update));
                                    assert_eq!(owner_subscriptions(app.world(), owner_entity), subscriptions);
                                    assert_eq!(graph_snapshot(app.world_mut()), before);
                                    assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                                }
                            }

                            let mut cancelled = Vec::new();
                            if supersession != 0 {
                                let command = match supersession {
                                    1 => $clear,
                                    2 => $empty_replace,
                                    3 => $replace($source(2)),
                                    6 => $remove,
                                    _ => $targeted_replace($source(2)),
                                };
                                let process = process_test_update(
                                    app.world_mut(), owner_entity,
                                    Box::new(Update::new([command], DynamicEmissions::One(String::from("supersede")))),
                                );
                                assert_eq!(process.wake_output, published_empty);
                                cancelled = process.cleanup.iter().map(DynamicSignalSubscription::inspect).collect();
                                assert_eq!(cancelled, if cancels_transient { subscriptions.clone() } else { vec![desired.clone()] });
                                assert_eq!(finish_dynamic_process(app.world_mut(), owner_entity, process), published_empty);
                                for subscription in &cancelled {
                                    assert!(app.world().get_entity(*subscription.adapter).is_err());
                                    assert!(app.world().get_entity(*subscription.selected).is_err());
                                }
                                let after = graph_snapshot(app.world_mut());
                                if !cancels_transient {
                                    assert_eq!(after.node(*transient.selected), before.node(*transient.selected));
                                    assert_eq!(after.node(*transient.adapter), before.node(*transient.adapter));
                                }
                                if replaces {
                                    let next = owner_subscriptions(app.world(), owner_entity);
                                    assert_eq!(next.len(), if cancels_transient { 1 } else { 2 });
                                    assert_eq!(next.last().expect("replacement").slot.generation, 3);
                                }
                                assert!(calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty());
                                app.update();
                            }

                            app.world_mut().resource_mut::<PendingCollectionEvents>().$field.insert(1, $warming);
                            app.update();
                            let transient_event = format!("{}:{:?}", $context(Default::default()), $warming);
                            let mut expected_calls = Vec::new();
                            let mut expected = vec![String::from("transient")];
                            if !cancels_transient {
                                expected_calls.push(transient_event.clone());
                                expected.push(transient_event);
                            }
                            expected.extend([String::from("remove"), String::from("desired")]);
                            if supersession == 0 {
                                let desired_event = format!("{}:{:?}", $context(Default::default()), $ready);
                                expected_calls.push(desired_event.clone());
                                expected.push(desired_event);
                            }
                            expected.extend(core::iter::repeat_n(String::from("emission"), 3));
                            if supersession != 0 {
                                expected.push(String::from("supersede"));
                                if replaces {
                                    let replacement_event = format!("{}:{:?}", $context(Default::default()), $empty);
                                    expected_calls.push(replacement_event.clone());
                                    expected.push(replacement_event);
                                }
                            }
                            assert_eq!(*calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), expected_calls);
                            if published_empty {
                                let output = app.world_mut().get_mut::<DynamicSignalOwner>(*owner_entity).expect("dynamic owner").runtime.take_output()
                                    .expect("empty replacement invokes the reducer once");
                                let output = (output as Box<dyn core::any::Any>).downcast::<Vec<String>>().expect("string queue");
                                assert_eq!(*output, expected);
                            } else {
                                assert!(matches!(scalar_replay_state::<Vec<String>>(app.world(), owner_entity),
                                    ReplayCacheState::Ready { current, .. } if current == expected));
                            }
                            assert!(app.world().get_entity(*transient.adapter).is_err());
                            assert!(app.world().get_entity(*transient.selected).is_err());
                            let published = owner_subscriptions(app.world(), owner_entity);
                            if supersession == 0 {
                                assert_eq!(published, vec![desired.clone()]);
                            } else {
                                assert_eq!(published.len(), usize::from(replaces));
                            }
                            if !cancels_transient {
                                cancelled.push(transient.clone());
                            }
                            let after = graph_snapshot(app.world_mut());
                            for subscription in cancelled {
                                let binding = DynamicAdapterBinding {
                                    owner: owner_entity, selected: subscription.selected,
                                    adapter: subscription.adapter, slot: subscription.slot,
                                };
                                let stale = app.world_mut().get_mut::<DynamicSignalOwner>(*owner_entity).expect("dynamic owner").runtime.process_adapter(
                                    &binding, Box::new($warming), false,
                                );
                                assert!(!stale.wake_output);
                                assert!(stale.cleanup.is_empty());
                            }
                            for update in no_ops() {
                                assert!(!process_update(&mut app, update));
                            }
                            assert!(app.world_mut().get_mut::<DynamicSignalOwner>(*owner_entity).expect("dynamic owner").runtime.take_after_output_cleanup().is_empty());
                            crate::cleanup::drain_world_cleanup(app.world_mut());
                            assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);
                            assert_eq!(graph_snapshot(app.world_mut()), after);
                            for _ in 0..3 { app.update(); }
                            assert_eq!(*calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), expected_calls);
                            claim.release(app.world_mut());
                            app.world_mut().flush();
                            crate::cleanup::drain_world_cleanup(app.world_mut());
                            for subscription in published {
                                assert!(app.world().get_entity(*subscription.adapter).is_err());
                                assert!(app.world().get_entity(*subscription.selected).is_err());
                            }
                        }
                    };
                }
                check_policy!(scalar, ScalarSlotCommand<$signal>, |signal| ScalarSlotCommand::Select { signal },
                    ScalarSlotCommand::Clear, [], ScalarSlotCommand::Clear, ScalarSlotCommand::Clear,
                    |signal| ScalarSlotCommand::Select { signal }, |signal| ScalarSlotCommand::Select { signal }, true, |()| 0usize);
                check_policy!(indexed, IndexedSlotCommand<$signal>, |signal| IndexedSlotCommand::Insert { index: 0, signal },
                    IndexedSlotCommand::Remove { index: 0 }, [IndexedSlotCommand::Move { old_index: 0, new_index: 0 }],
                    IndexedSlotCommand::Clear, IndexedSlotCommand::ReplaceAll { signals: vec![] },
                    |signal| IndexedSlotCommand::ReplaceAll { signals: vec![signal] },
                    |signal| IndexedSlotCommand::Replace { index: 0, signal }, false, |index: usize| index);
                check_policy!(keyed, KeyedSlotCommand<usize, $signal>, |signal| KeyedSlotCommand::Insert { key: 0, signal },
                    KeyedSlotCommand::Remove { key: 0 }, [KeyedSlotCommand::Rekey { old_key: 0, new_key: 7 }, KeyedSlotCommand::Rekey { old_key: 7, new_key: 0 }],
                    KeyedSlotCommand::Clear, KeyedSlotCommand::ReplaceAll { entries: vec![] },
                    |signal| KeyedSlotCommand::ReplaceAll { entries: vec![(0, signal)] },
                    |signal| KeyedSlotCommand::Replace { key: 0, signal }, false, |key: usize| key);
            };
        }
        check_replay!(
            BoxedReplayableSignalVec<i32>,
            replay_signal_vec::<i32>,
            pending_test_vector,
            vectors,
            vec![VecDiff::Replace { values: vec![1] }],
            vec![VecDiff::Replace { values: vec![9] }],
            vec![VecDiff::<i32>::Replace { values: vec![] }]
        );
        check_replay!(BoxedReplayableSignalMap<i32, i32>, replay_signal_map::<i32, i32>, pending_test_map, maps,
            vec![MapDiff::Replace { entries: vec![(1, 10)] }], vec![MapDiff::Replace { entries: vec![(9, 90)] }], vec![MapDiff::<i32, i32>::Replace { entries: vec![] }]);
    }

    #[test]
    fn long_keep_current_preserves_exact_ownership_and_cursor_and_rejects_cancelled_bindings() {
        macro_rules! check {
            ($policy:ident, $initial:expr, $replace:expr, $adapter:expr) => {
                for length in [2, 7] {
                    for warming in [false, true] {
                        let mut app = test_app();
                        app.init_resource::<ForwardSelectedValue>();
                        let current = MaterializedTestSignal {
                            materializer: ScalarSignalMaterializer::new(|registrar| {
                                registrar.replayable_signal_source(
                                    10,
                                    |In(()), mut next: ResMut<ForwardSelectedValue>| next.0.take(),
                                    OutputDelivery::broadcast(),
                                )
                            }),
                        }
                        .cache()
                        .broadcast();
                        let mut initial = Some($initial(current.clone().boxed_replayable()));
                        let materializer = ScalarSignalMaterializer::<Vec<i32>>::new(move |registrar| {
                            let manager = registrar.signal_entity(
                                (),
                                move |In(())| {
                                    initial
                                        .take()
                                        .map(|initial| DynamicManagerUpdate::new([initial], DynamicEmissions::<i32>::None))
                                },
                                OutputDelivery::move_only(),
                            )?;
                            registrar.dynamic_operator(
                                DynamicSignalOperatorSpec::$policy(manager)
                                    .replay_scalar::<i32>()
                                    .adapter($adapter)
                                    .replayable_scalar_output(Some)
                                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
                            )
                        });
                        let claim = materializer.register(app.world_mut());
                        let owner_entity = claim.entity();
                        app.update();
                        let subscriptions = owner_subscriptions(app.world(), owner_entity);
                        let [subscription] = subscriptions.as_slice() else { panic!("one current subscription required") };
                        let before = graph_snapshot(app.world_mut());
                        let selected_node = before.node(*subscription.selected).expect("current selected source");
                        assert_eq!(selected_node.replay_relationships.len(), 1);
                        assert!(matches!(selected_node.replay_relationships.first().expect("current cursor").state,
                            crate::graph::ReplayRelationshipSnapshotState::Published { .. }));
                        assert_eq!(before.node(*subscription.adapter).expect("current adapter").claim_count, Some(1));

                        let mut steps = Vec::new();
                        for index in 0..length {
                            let candidate = MaterializedTestSignal {
                                materializer: ScalarSignalMaterializer::new(move |registrar| {
                                    if warming {
                                        registrar.replayable_signal_entity(
                                            (),
                                            |In(())| None::<i32>,
                                            ReplayInitialization::Warming,
                                            OutputDelivery::broadcast(),
                                        )
                                    } else {
                                        registrar.replayable_signal_source(
                                            20 + index,
                                            |In(())| None::<i32>,
                                            OutputDelivery::broadcast(),
                                        )
                                    }
                                }),
                            }.cache().boxed_replayable();
                            steps.push(DynamicManagerStep::command($replace(candidate), DynamicEmissions::One(index)));
                        }
                        steps.push(DynamicManagerStep::command(
                            $replace(current.clone().boxed_replayable()),
                            DynamicEmissions::One(length),
                        ));

                        let process = process_test_update(
                            app.world_mut(), owner_entity, Box::new(DynamicManagerUpdate::sequenced(steps)),
                        );
                        assert!(process.wake_output);
                        let cancelled = process.cleanup.iter().map(DynamicSignalSubscription::inspect).collect::<Vec<_>>();
                        assert_eq!(cancelled.len(), usize::try_from(length).expect("positive length"));
                        let staged = graph_snapshot(app.world_mut());
                        for candidate in &cancelled {
                            assert_ne!(candidate.adapter, subscription.adapter);
                            assert_eq!(candidate.slot.identity, subscription.slot.identity);
                            assert!(candidate.slot.generation > subscription.slot.generation);
                            assert_eq!(staged.node(*candidate.adapter).expect("cancelled adapter before release").claim_count, Some(1));
                        }
                        assert!(finish_dynamic_process(app.world_mut(), owner_entity, process));
                        assert_eq!(owner_subscriptions(app.world(), owner_entity), subscriptions);
                        let after_cleanup = graph_snapshot(app.world_mut());
                        assert_eq!(after_cleanup.nodes.len(), before.nodes.len());
                        assert_eq!(after_cleanup.node(*subscription.selected), before.node(*subscription.selected));
                        assert_eq!(after_cleanup.node(*subscription.adapter), before.node(*subscription.adapter));
                        let owner_before = before.node(*owner_entity).expect("owner before normalization");
                        let owner_after = after_cleanup.node(*owner_entity).expect("owner after normalization");
                        assert_eq!(owner_after.claim_count, owner_before.claim_count);
                        assert_eq!(owner_after.dynamic_subscriptions, owner_before.dynamic_subscriptions);
                        assert_eq!(owner_after.active_downstreams, owner_before.active_downstreams);
                        assert_eq!(owner_after.canonical_static_dependencies, owner_before.canonical_static_dependencies);
                        assert_eq!(queued_scalar_output(app.world_mut(), owner_entity), (0..=length).collect::<Vec<_>>());
                        for candidate in cancelled {
                            assert!(app.world().get_entity(*candidate.adapter).is_err());
                            assert!(app.world().get_entity(*candidate.selected).is_err());
                            let binding = DynamicAdapterBinding {
                                owner: owner_entity,
                                selected: candidate.selected,
                                adapter: candidate.adapter,
                                slot: candidate.slot,
                            };
                            let stale = app.world_mut().get_mut::<DynamicSignalOwner>(*binding.owner).expect("dynamic owner").runtime.process_adapter(
                                &binding, Box::new(999), false,
                            );
                            assert!(!stale.wake_output);
                            assert!(stale.cleanup.is_empty());
                        }
                        assert!(queued_scalar_output(app.world_mut(), owner_entity).is_empty());
                        crate::cleanup::drain_world_cleanup(app.world_mut());
                        assert_eq!(graph_snapshot(app.world_mut()), after_cleanup);
                        assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);
                        app.world_mut().resource_mut::<ForwardSelectedValue>().0 = Some(11);
                        app.update();
                        assert_eq!(owner_subscriptions(app.world(), owner_entity), subscriptions);
                        assert!(matches!(scalar_replay_state::<Vec<i32>>(app.world(), owner_entity),
                            ReplayCacheState::Ready { current, .. } if current == vec![11]));
                        let after = graph_snapshot(app.world_mut());
                        assert_ne!(after.node(*subscription.selected).expect("updated source").replay_relationships, selected_node.replay_relationships);
                        drop(current);
                        claim.release(app.world_mut());
                        app.world_mut().flush();
                        crate::cleanup::drain_world_cleanup(app.world_mut());
                        assert!(app.world().get_entity(*subscription.adapter).is_err());
                    }
                }
            };
        }
        check!(
            scalar,
            |signal| ScalarSlotCommand::Select { signal },
            |signal| ScalarSlotCommand::Select { signal },
            |(), value| DynamicEmissions::One(value)
        );
        check!(
            indexed,
            |signal| IndexedSlotCommand::Insert { index: 0, signal },
            |signal| IndexedSlotCommand::Replace { index: 0, signal },
            |_: usize, value| DynamicEmissions::One(value)
        );
        check!(
            keyed,
            |signal| KeyedSlotCommand::Insert { key: 7usize, signal },
            |signal| KeyedSlotCommand::Replace { key: 7usize, signal },
            |_: usize, value| DynamicEmissions::One(value)
        );
    }

    #[test]
    fn candidate_initializer_panic_keeps_runtime_and_committed_claims_in_world() {
        let mut app = test_app();
        app.init_resource::<OwnershipUpdates>();
        app.init_resource::<ForwardSelectedValue>();
        let materializer = ownership_test_materializer();
        let selected = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_source(
                    5,
                    |In(()), mut next: ResMut<ForwardSelectedValue>| next.0.take(),
                    OutputDelivery::broadcast(),
                )
            }),
        }
        .cache()
        .boxed_replayable();
        app.world_mut()
            .resource_mut::<OwnershipUpdates>()
            .0
            .push_back(DynamicManagerUpdate::new(
                [IndexedSlotCommand::Insert {
                    index: 0,
                    signal: selected,
                }],
                DynamicEmissions::None,
            ));
        let claim = materializer.register(app.world_mut());
        let owner = claim.entity();
        app.update();
        let before = graph_snapshot(app.world_mut());
        let selected = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::<i32>::new(|_| panic!("selected initializer panic")),
        }
        .cache()
        .boxed_replayable();
        app.world_mut()
            .resource_mut::<OwnershipUpdates>()
            .0
            .push_back(DynamicManagerUpdate::new(
                [IndexedSlotCommand::Insert {
                    index: 1,
                    signal: selected,
                }],
                DynamicEmissions::One(99),
            ));
        let mut process = process_signal_graph_for_schedule(bevy_app::Last.intern());
        assert!(std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| process(app.world_mut()))).is_err());
        assert_eq!(graph_snapshot(app.world_mut()), before);
        assert!(
            !app.world()
                .get::<DynamicSignalOwner>(*owner)
                .expect("runtime survived unwind")
                .materializing
        );
        app.world_mut().resource_mut::<ForwardSelectedValue>().0 = Some(6);
        process(app.world_mut());
        assert!(
            matches!(scalar_replay_state::<Vec<i32>>(app.world(), owner), ReplayCacheState::Ready { current, .. } if current == vec![6])
        );
        claim.release(app.world_mut());
    }

    #[test]
    fn adapter_fast_path_preserves_binding_checks_and_queued_manager_order() {
        type Update = DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, i32>;
        let mut app = test_app();
        app.init_resource::<OwnershipUpdates>();
        app.world_mut()
            .resource_mut::<OwnershipUpdates>()
            .0
            .push_back(Update::new(
                [IndexedSlotCommand::Insert {
                    index: 0,
                    signal: signal::once(1).cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ));
        let claim = ownership_test_materializer().register(app.world_mut());
        let owner_entity = claim.entity();
        app.update();
        let subscriptions = owner_subscriptions(app.world(), owner_entity);
        let [subscription] = subscriptions.as_slice() else {
            panic!("expected one selected subscription");
        };
        let binding = DynamicAdapterBinding {
            owner: owner_entity,
            selected: subscription.selected,
            adapter: subscription.adapter,
            slot: subscription.slot,
        };
        let binding = OnceLock::from(binding);
        let owner_binding = app
            .world()
            .get::<DynamicSignalOwner>(*owner_entity)
            .expect("owner")
            .incarnation
            .clone();
        assert!(process_dynamic_adapter(app.world_mut(), &binding, 2_i32).is_some());
        assert_eq!(owner_subscriptions(app.world(), owner_entity), subscriptions);
        assert_eq!(
            take_dynamic_output::<Vec<i32>>(app.world_mut(), &owner_binding),
            Some(vec![2])
        );

        app.world_mut()
            .get_mut::<DynamicSignalOwner>(*owner_entity)
            .expect("owner")
            .inputs
            .push_back(DynamicRuntimeInput::Manager(Box::new(Update::new(
                [],
                DynamicEmissions::One(10),
            ))));
        assert!(process_dynamic_adapter(app.world_mut(), &binding, 3_i32).is_some());
        assert_eq!(
            take_dynamic_output::<Vec<i32>>(app.world_mut(), &owner_binding),
            Some(vec![10, 3])
        );

        app.world_mut()
            .get_mut::<DynamicSignalOwner>(*owner_entity)
            .expect("owner")
            .inputs
            .push_back(DynamicRuntimeInput::Manager(Box::new(Update::new(
                [IndexedSlotCommand::Clear],
                DynamicEmissions::One(20),
            ))));
        assert!(process_dynamic_adapter(app.world_mut(), &binding, 4_i32).is_some());
        assert_eq!(
            take_dynamic_output::<Vec<i32>>(app.world_mut(), &owner_binding),
            Some(vec![20])
        );
        assert!(process_dynamic_adapter(app.world_mut(), &binding, 5_i32).is_none());
        assert!(owner_subscriptions(app.world(), owner_entity).is_empty());
        // These direct runner calls bypass the graph pass that reconciles execution levels.
        app.update();
        claim.release(app.world_mut());
        app.update();
        assert!(graph_snapshot(app.world_mut()).nodes.is_empty());
    }

    #[derive(Resource)]
    struct QueuedManagerReentry {
        owner: SignalEntity,
        ran: bool,
    }

    #[test]
    fn reentrant_manager_input_waits_for_the_current_materialization() {
        let mut app = test_app();
        app.init_resource::<OwnershipUpdates>();
        app.world_mut()
            .register_component_hooks::<bevy_ecs::system::SystemIdMarker>()
            .on_add(|mut world, _| {
                if world
                    .get_resource::<QueuedManagerReentry>()
                    .is_some_and(|state| !state.ran)
                {
                    world.resource_mut::<QueuedManagerReentry>().ran = true;
                    world.commands().queue(|world: &mut World| {
                        let owner = world.resource::<QueuedManagerReentry>().owner;
                        assert!(
                            world
                                .get::<DynamicSignalOwner>(*owner)
                                .expect("materializing owner")
                                .materializing
                        );
                        let update = DynamicManagerUpdate::<IndexedSlotCommand<TestReplaySignal>, i32>::new(
                            [],
                            DynamicEmissions::One(99),
                        );
                        assert!(
                            process_dynamic_input(world, owner, DynamicRuntimeInput::Manager(Box::new(update)), None)
                                .is_none()
                        );
                    });
                }
            });
        let claim = ownership_test_materializer().register(app.world_mut());
        let owner = claim.entity();
        app.world_mut()
            .resource_mut::<OwnershipUpdates>()
            .0
            .push_back(DynamicManagerUpdate::new(
                [IndexedSlotCommand::Insert {
                    index: 0,
                    signal: signal::once(1).cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ));
        app.update();
        app.insert_resource(QueuedManagerReentry { owner, ran: false });
        app.world_mut()
            .resource_mut::<OwnershipUpdates>()
            .0
            .push_back(DynamicManagerUpdate::new(
                [IndexedSlotCommand::Insert {
                    index: 1,
                    signal: signal::once(2).cache().boxed_replayable(),
                }],
                DynamicEmissions::One(10),
            ));
        app.update();
        assert!(app.world().resource::<QueuedManagerReentry>().ran);
        assert!(
            matches!(scalar_replay_state::<Vec<i32>>(app.world(), owner), ReplayCacheState::Ready { current, .. } if current == vec![10, 2, 99])
        );
        assert!(
            app.world()
                .get::<DynamicSignalOwner>(*owner)
                .expect("settled owner")
                .inputs
                .is_empty()
        );
        claim.release(app.world_mut());
    }

    #[derive(Resource)]
    struct InitializationOwner {
        owner: SignalEntity,
        claim: Option<SignalClaim>,
        replace: bool,
        initialized: bool,
    }

    struct RetireOwnerOnInitialize;

    impl bevy_ecs::world::FromWorld for RetireOwnerOnInitialize {
        fn from_world(world: &mut World) -> Self {
            let (owner, replace, claim) = {
                let mut state = world.resource_mut::<InitializationOwner>();
                state.initialized = true;
                (state.owner, state.replace, state.claim.take())
            };
            assert!(world.get::<DynamicSignalOwner>(*owner).is_some());
            if let Some(claim) = claim {
                claim.try_release(world).expect("initialization can release the owner");
            } else if replace {
                world
                    .entity_mut(*owner)
                    .insert(DynamicSignalOwner::new(Box::new(EmptyDynamicRuntime)));
            } else {
                world.entity_mut(*owner).remove::<DynamicSignalOwner>();
            }
            Self
        }
    }

    #[derive(Resource, Default)]
    struct OwnershipUpdates(VecDeque<DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, i32>>);

    fn ownership_test_materializer() -> ScalarSignalMaterializer<Vec<i32>> {
        ScalarSignalMaterializer::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                |In(()), mut updates: ResMut<OwnershipUpdates>| updates.0.pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|_, value| DynamicEmissions::One(value))
                    .replayable_scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        })
    }

    #[test]
    fn candidate_spawn_hook_can_retire_dynamic_owner() {
        candidate_lifecycle_hook_can_retire_dynamic_owner(false);
    }

    #[test]
    fn candidate_cleanup_hook_can_retire_dynamic_owner() {
        candidate_lifecycle_hook_can_retire_dynamic_owner(true);
    }

    fn candidate_lifecycle_hook_can_retire_dynamic_owner(during_cleanup: bool) {
        for mode in 0..3 {
            let mut app = test_app();
            app.init_resource::<OwnershipUpdates>();
            let baseline = graph_snapshot(app.world_mut());
            let hook: bevy_ecs::lifecycle::ComponentHook = |mut world, _| {
                if world
                    .get_resource::<InitializationOwner>()
                    .is_some_and(|state| !state.initialized)
                {
                    world.resource_mut::<InitializationOwner>().initialized = true;
                    world.commands().queue(|world: &mut World| {
                        let owner = world.resource::<InitializationOwner>().owner;
                        crate::registration::SignalRegistrationGraphHooks::registration_preflight_existing_runtimes(
                            world,
                            &[owner],
                        )
                        .expect("in-flight inspection is valid");
                        <RetireOwnerOnInitialize as bevy_ecs::world::FromWorld>::from_world(world);
                    });
                }
            };
            let hooks = app
                .world_mut()
                .register_component_hooks::<bevy_ecs::system::SystemIdMarker>();
            if during_cleanup {
                hooks.on_remove(hook);
            } else {
                hooks.on_add(hook);
            }
            let materializer = ownership_test_materializer();
            let claim = materializer.register(app.world_mut());
            let owner = claim.entity();
            for selected in [
                signal::once(1).cache().boxed_replayable(),
                signal::from_system::<i32, _, _, _>(|In(())| None::<i32>)
                    .cache()
                    .boxed_replayable(),
            ] {
                let index = owner_subscriptions(app.world(), owner).len();
                app.world_mut()
                    .resource_mut::<OwnershipUpdates>()
                    .0
                    .push_back(DynamicManagerUpdate::new(
                        [IndexedSlotCommand::Insert {
                            index,
                            signal: selected,
                        }],
                        DynamicEmissions::None,
                    ));
                app.update();
            }
            let subscriptions = owner_subscriptions(app.world(), owner);
            assert_eq!(
                subscriptions.len(),
                2,
                "current and never-ready candidate are owned before reentry"
            );
            let (external_claim, initialization_claim) = if mode == 2 {
                (None, Some(claim))
            } else {
                (Some(claim), None)
            };
            app.insert_resource(InitializationOwner {
                owner,
                claim: initialization_claim,
                replace: mode == 1,
                initialized: false,
            });
            let allocation_probe = Arc::new(());
            let selected = MaterializedTestSignal {
                materializer: ScalarSignalMaterializer::new({
                    let allocation_probe = allocation_probe.clone();
                    move |registrar| {
                        registrar.signal_entity((), |In(())| 99_i32, OutputDelivery::move_only())?;
                        registrar.replayable_signal_source(
                            7,
                            move |In(())| {
                                let _probe = &allocation_probe;
                                None::<i32>
                            },
                            OutputDelivery::broadcast(),
                        )
                    }
                }),
            }
            .cache()
            .boxed_replayable();
            app.world_mut()
                .resource_mut::<OwnershipUpdates>()
                .0
                .push_back(DynamicManagerUpdate::new(
                    [IndexedSlotCommand::Insert {
                        index: 2,
                        signal: selected,
                    }],
                    DynamicEmissions::None,
                ));
            app.update();
            assert!(app.world().resource::<InitializationOwner>().initialized);
            assert_eq!(
                Arc::strong_count(&allocation_probe),
                1,
                "retired candidate systems were reaped"
            );
            app.world_mut().flush();
            crate::cleanup::drain_world_cleanup(app.world_mut());
            for subscription in subscriptions {
                assert!(app.world().get_entity(*subscription.adapter).is_err());
                assert!(app.world().get_entity(*subscription.selected).is_err());
            }
            if let Some(claim) = external_claim {
                if mode == 0 {
                    assert!(
                        app.world().get::<DynamicSignalOwner>(*owner).is_none(),
                        "retired owner was not restored"
                    );
                    app.world_mut()
                        .entity_mut(*owner)
                        .insert(DynamicSignalOwner::new(Box::new(EmptyDynamicRuntime)));
                }
                assert!(owner_subscriptions(app.world(), owner).is_empty());
                claim.release(app.world_mut());
            }
            app.world_mut().flush();
            crate::cleanup::drain_world_cleanup(app.world_mut());
            assert_eq!(graph_snapshot(app.world_mut()), baseline);
        }
    }

    #[test]
    fn selected_local_initialization_can_retire_dynamic_owner() {
        local_initialization_retirement(false);
    }

    #[test]
    fn manager_local_initialization_can_retire_dynamic_owner() {
        local_initialization_retirement(true);
    }

    fn local_initialization_retirement(manager: bool) {
        for mode in 0..3 {
            let mut app = test_app();
            let claim = if manager {
                signal::from_system::<TestReplaySignal, _, _, _>(|In(()), _local: Local<RetireOwnerOnInitialize>| {
                    signal::once(7).cache().boxed_replayable()
                })
                .flatten()
                .register(app.world_mut())
            } else {
                let selected =
                    signal::from_system::<i32, _, _, _>(|In(()), _local: Local<RetireOwnerOnInitialize>| Some(7))
                        .cache();
                signal::once(selected).flatten().register(app.world_mut())
            };
            let owner = claim.entity();
            let (external_claim, initialization_claim) = if mode == 2 {
                (None, Some(claim))
            } else {
                (Some(claim), None)
            };
            app.insert_resource(InitializationOwner {
                owner,
                claim: initialization_claim,
                replace: mode == 1,
                initialized: false,
            });
            app.update();
            assert!(app.world().resource::<InitializationOwner>().initialized);
            app.world_mut().flush();
            crate::cleanup::drain_world_cleanup(app.world_mut());
            if let Some(claim) = external_claim {
                if mode == 0 {
                    app.world_mut()
                        .entity_mut(*owner)
                        .insert(DynamicSignalOwner::new(Box::new(EmptyDynamicRuntime)));
                }
                claim.release(app.world_mut());
            }
        }
    }

    #[test]
    fn active_dynamic_owner_removal_and_replacement_drain_current_and_pending_subscriptions_once() {
        type Update = DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, (usize, i32)>;

        for replace in [false, true] {
            let mut app = test_app();
            let baseline = graph_snapshot(app.world_mut());
            let current = signal::once(1).cache().boxed_replayable();
            let never_ready = signal::from_system(|In(())| None::<i32>).cache().boxed_replayable();
            let updates = Arc::new(Mutex::new(VecDeque::from([
                Update::new(
                    [IndexedSlotCommand::ReplaceAll { signals: vec![current] }],
                    DynamicEmissions::None,
                ),
                Update::new(
                    [IndexedSlotCommand::Insert {
                        index: 1,
                        signal: never_ready,
                    }],
                    DynamicEmissions::None,
                ),
            ])));
            let materializer = ScalarSignalMaterializer::<Vec<(usize, i32)>>::new(move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        updates
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .pop_front()
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::indexed(manager)
                        .replay_scalar::<i32>()
                        .adapter(|index, value| DynamicEmissions::One((index, value)))
                        .scalar_output(Some)
                        .same_signal_identity(SameSignalIdentity::Restart),
                )
            });
            let claim = materializer.register(app.world_mut());
            let owner = claim.entity();
            app.update();
            app.update();

            let subscriptions = owner_subscriptions(app.world(), owner);
            assert_eq!(subscriptions.len(), 2);
            assert_adapter_routes(&graph_snapshot(app.world_mut()), owner, &subscriptions);

            if replace {
                app.world_mut()
                    .entity_mut(*owner)
                    .insert(DynamicSignalOwner::new(Box::new(EmptyDynamicRuntime)));
            } else {
                assert!(app.world().get::<DynamicSignalOwner>(*owner).is_some());
                app.world_mut().entity_mut(*owner).remove::<DynamicSignalOwner>();
            }
            app.world_mut().flush();
            crate::cleanup::drain_world_cleanup(app.world_mut());
            app.world_mut().flush();

            for subscription in &subscriptions {
                assert!(app.world().get_entity(*subscription.adapter).is_err());
                assert!(app.world().get_entity(*subscription.selected).is_err());
            }
            if replace {
                assert!(owner_subscriptions(app.world(), owner).is_empty());
            } else {
                assert!(app.world().get::<DynamicSignalOwner>(*owner).is_none());
            }
            assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);

            let after_cleanup = graph_snapshot(app.world_mut());
            crate::cleanup::drain_world_cleanup(app.world_mut());
            app.world_mut().flush();
            assert_eq!(graph_snapshot(app.world_mut()), after_cleanup);
            assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);

            if !replace {
                app.world_mut()
                    .entity_mut(*owner)
                    .insert(DynamicSignalOwner::new(Box::new(EmptyDynamicRuntime)));
            }
            app.update();
            claim.release(app.world_mut());
            app.world_mut().flush();
            crate::cleanup::drain_world_cleanup(app.world_mut());
            app.world_mut().flush();
            assert_eq!(graph_snapshot(app.world_mut()), baseline);
        }
    }

    #[test]
    fn retained_dynamic_output_reregisters_without_dormancy_or_second_maintenance_lifecycle() {
        let mut app = test_app();
        app.init_resource::<RetainedDynamicOutputs>();
        let selected = signal::once(5).cache();
        let dynamic = signal::once(selected).flatten().broadcast();
        let retained = dynamic.clone();
        let first_claim = dynamic.register(app.world_mut());
        let owner = first_claim.entity();
        app.update();

        let subscriptions = owner_subscriptions(app.world(), owner);
        assert_eq!(subscriptions.len(), 1);
        let initial_replay = scalar_replay_state::<i32>(app.world(), owner);
        let initial_revision = match initial_replay {
            ReplayCacheState::Ready { current: 5, revision } => revision,
            state => panic!("retained dynamic output should be ready with 5, got {state:?}"),
        };
        let active_count = graph_snapshot(app.world_mut())
            .node(*owner)
            .and_then(|node| node.claim_count)
            .expect("the retained dynamic output should have a claim count");
        assert_eq!(
            active_count, 2,
            "one external and one maintenance claim should be active"
        );

        first_claim.release(app.world_mut());
        let maintenance_snapshot = graph_snapshot(app.world_mut());
        let maintenance_owner = maintenance_snapshot
            .node(*owner)
            .expect("maintenance must retain the owner");
        assert_eq!(maintenance_owner.claim_count, Some(1));
        assert!(!maintenance_owner.registrar_inactive);
        assert_eq!(owner_subscriptions(app.world(), owner), subscriptions);

        let second_claim = retained.clone().cache().register(app.world_mut());
        assert_eq!(second_claim.entity(), owner);
        let explicit_cache_claim = crate::signal::SignalExt::cache(retained.clone()).register(app.world_mut());
        assert_eq!(explicit_cache_claim.entity(), owner);
        explicit_cache_claim.release(app.world_mut());
        assert_eq!(
            graph_snapshot(app.world_mut())
                .node(*owner)
                .and_then(|node| node.claim_count),
            Some(2),
            "cache normalization must add only the requested direct claim"
        );
        assert_eq!(owner_subscriptions(app.world(), owner), subscriptions);
        assert_eq!(
            scalar_replay_state::<i32>(app.world(), owner),
            ReplayCacheState::Ready {
                current: 5,
                revision: initial_revision,
            }
        );

        let observer = retained
            .clone()
            .map(|In(value): In<i32>, mut outputs: ResMut<RetainedDynamicOutputs>| {
                outputs.0.push(value);
            })
            .register(app.world_mut());
        let observer_entity = observer.entity();
        app.update();
        assert_eq!(app.world().resource::<RetainedDynamicOutputs>().0, [5]);
        let replay_relationship = graph_snapshot(app.world_mut())
            .node(*owner)
            .expect("retained owner should remain materialized")
            .replay_relationships
            .iter()
            .find(|relationship| relationship.dependent == *observer_entity)
            .cloned()
            .expect("the late observer should have a targeted replay relationship");
        assert!(matches!(
            replay_relationship.state,
            crate::graph::ReplayRelationshipSnapshotState::Published { cursor }
                if cursor == initial_revision
        ));

        observer.release(app.world_mut());
        second_claim.release(app.world_mut());
        assert_eq!(
            graph_snapshot(app.world_mut())
                .node(*owner)
                .and_then(|node| node.claim_count),
            Some(1)
        );
        assert_eq!(owner_subscriptions(app.world(), owner), subscriptions);

        let selected_entities = subscriptions
            .iter()
            .flat_map(|subscription| [subscription.selected, subscription.adapter])
            .collect::<Vec<_>>();
        drop(retained);
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
        assert!(app.world().get_entity(*owner).is_err());
        for entity in &selected_entities {
            assert!(app.world().get_entity(**entity).is_err());
        }

        let settled = graph_snapshot(app.world_mut());
        let mut roots_after_boundary = Vec::new();
        crate::cleanup::drain_world_cleanup_with_stale_signal_roots(app.world_mut(), |roots| {
            roots_after_boundary.extend_from_slice(roots);
        });
        app.world_mut().flush();
        assert!(roots_after_boundary.is_empty());
        assert_eq!(graph_snapshot(app.world_mut()), settled);
        assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);
    }

    #[test]
    fn final_owner_release_drains_current_and_never_ready_cohort_once() {
        type Update = DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, (usize, i32)>;

        let mut app = test_app();
        let baseline = graph_snapshot(app.world_mut());
        let current = signal::once(1).cache().boxed_replayable();
        let never_ready = signal::from_system(|In(())| None::<i32>).cache().boxed_replayable();
        let updates = Arc::new(Mutex::new(VecDeque::from([
            Update::new(
                [IndexedSlotCommand::ReplaceAll { signals: vec![current] }],
                DynamicEmissions::None,
            ),
            Update::new(
                [IndexedSlotCommand::Insert {
                    index: 1,
                    signal: never_ready,
                }],
                DynamicEmissions::None,
            ),
        ])));
        let materializer = ScalarSignalMaterializer::<Vec<(usize, i32)>>::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        });
        let claim = materializer
            .try_register(app.world_mut())
            .expect("indexed dynamic operator should register");
        let owner = claim.entity();
        app.update();
        app.update();

        let subscriptions = owner_subscriptions(app.world(), owner);
        assert_eq!(subscriptions.len(), 2);
        assert_eq!(
            subscriptions[0].slot,
            DynamicStableSlot {
                identity: 1,
                generation: 1
            }
        );
        assert_eq!(
            subscriptions[1].slot,
            DynamicStableSlot {
                identity: 2,
                generation: 2
            }
        );
        assert_ne!(subscriptions[0].selected, subscriptions[1].selected);
        assert_ne!(subscriptions[0].adapter, subscriptions[1].adapter);
        let active = graph_snapshot(app.world_mut());
        assert_eq!(active.node(*owner).and_then(|node| node.claim_count), Some(1));
        for subscription in &subscriptions {
            assert_eq!(
                active.node(*subscription.adapter).and_then(|node| node.claim_count),
                Some(1),
            );
        }
        assert_adapter_routes(&active, owner, &subscriptions);

        claim.release(app.world_mut());
        app.world_mut().flush();
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
        let released = graph_snapshot(app.world_mut());
        assert_eq!(released, baseline);
        assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);

        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
        assert_eq!(graph_snapshot(app.world_mut()), released);
        assert_eq!(crate::cleanup::queued_message_count(app.world()), 0);
    }

    fn single_static_dependency_of(world: &World, snapshot: &GraphSnapshot, entity: SignalEntity) -> SignalEntity {
        let deps = snapshot
            .node(*entity)
            .and_then(|node| node.canonical_static_dependencies.as_ref())
            .expect("sink entity should record canonical static dependencies");
        assert_eq!(deps.len(), 1, "expected a single static dependency");
        *world
            .get::<DynamicSignalOwner>(deps[0])
            .expect("static dependency should be a dynamic owner")
            .incarnation
            .get()
            .expect("dynamic owner incarnation initialized")
    }

    #[test]
    fn ordinary_dynamic_outputs_install_no_replay_cache_or_maintenance_and_support_non_clone_payloads() {
        #[derive(Debug, PartialEq, Eq)]
        struct NonCloneToken(usize);

        type ScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, NonCloneToken>;
        type VecUpdate = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, VecDiff<NonCloneToken>>;
        type MapUpdate =
            DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, MapDiff<NonCloneToken, NonCloneToken>>;

        let mut app = test_app();

        // 1. Ordinary scalar, vec, and map outputs accept `!Clone` queue items and `!Clone` (`!Ord`
        //    key) outputs while installing zero replay cache or maintenance claims.
        let scalar_updates = Arc::new(Mutex::new(VecDeque::from([ScalarUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::once(7).cache().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
        let scalar_seen = Arc::new(Mutex::new(Vec::new()));
        let scalar_seen_sink = Arc::clone(&scalar_seen);
        let scalar_claim = ScalarSignalMaterializer::<()>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    scalar_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            let output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(NonCloneToken(value as usize)))
                    .scalar_output(|mut queued: Vec<NonCloneToken>| queued.pop())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )?;
            registrar.signal_entity(
                output,
                move |In(token): In<NonCloneToken>| {
                    scalar_seen_sink
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(token);
                    Some(())
                },
                OutputDelivery::move_only(),
            )
        })
        .register(app.world_mut());
        let snapshot = graph_snapshot(app.world_mut());
        let scalar_owner = single_static_dependency_of(app.world(), &snapshot, scalar_claim.entity());
        let scalar_owner_node = snapshot.node(*scalar_owner).expect("scalar owner materialized");
        assert_eq!(scalar_owner_node.replay_cache_readiness, None);
        assert!(!scalar_owner_node.replay_maintenance_armed);
        assert_eq!(
            scalar_owner_node.claim_count,
            Some(1),
            "ordinary dynamic scalar output must not hold a replay maintenance claim"
        );

        let vec_updates = Arc::new(Mutex::new(VecDeque::from([VecUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::once(11).cache().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
        let vec_seen = Arc::new(Mutex::new(Vec::new()));
        let vec_seen_sink = Arc::clone(&vec_seen);
        let vec_claim = ScalarSignalMaterializer::<()>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    vec_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            let output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| {
                        DynamicEmissions::One(VecDiff::Push {
                            value: NonCloneToken(value as usize),
                        })
                    })
                    .signal_vec_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::Restart),
            )?;
            registrar.signal_entity(
                output,
                move |In(diffs): In<Vec<VecDiff<NonCloneToken>>>| {
                    vec_seen_sink
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .extend(diffs.into_iter().map(|diff| match diff {
                            VecDiff::Push { value } => value,
                            other => panic!("unexpected vec diff: {other:?}"),
                        }));
                    Some(())
                },
                OutputDelivery::move_only(),
            )
        })
        .register(app.world_mut());
        let snapshot = graph_snapshot(app.world_mut());
        let vec_owner = single_static_dependency_of(app.world(), &snapshot, vec_claim.entity());
        let vec_owner_node = snapshot.node(*vec_owner).expect("vec owner materialized");
        assert_eq!(vec_owner_node.replay_cache_readiness, None);
        assert!(!vec_owner_node.replay_maintenance_armed);
        assert_eq!(
            vec_owner_node.claim_count,
            Some(1),
            "ordinary dynamic vec output must not hold a replay maintenance claim"
        );

        let map_updates = Arc::new(Mutex::new(VecDeque::from([MapUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::once(19).cache().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
        let map_seen = Arc::new(Mutex::new(Vec::new()));
        let map_seen_sink = Arc::clone(&map_seen);
        let map_claim = ScalarSignalMaterializer::<()>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    map_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            let output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| {
                        DynamicEmissions::One(MapDiff::Insert {
                            key: NonCloneToken(value as usize),
                            value: NonCloneToken((value as usize) * 10),
                        })
                    })
                    .signal_map_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::Restart),
            )?;
            registrar.signal_entity(
                output,
                move |In(diffs): In<Vec<MapDiff<NonCloneToken, NonCloneToken>>>| {
                    map_seen_sink
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .extend(diffs.into_iter().map(|diff| match diff {
                            MapDiff::Insert { key, value } => (key, value),
                            other => panic!("unexpected map diff: {other:?}"),
                        }));
                    Some(())
                },
                OutputDelivery::move_only(),
            )
        })
        .register(app.world_mut());
        let snapshot = graph_snapshot(app.world_mut());
        let map_owner = single_static_dependency_of(app.world(), &snapshot, map_claim.entity());
        let map_owner_node = snapshot.node(*map_owner).expect("map owner materialized");
        assert_eq!(map_owner_node.replay_cache_readiness, None);
        assert!(!map_owner_node.replay_maintenance_armed);
        assert_eq!(
            map_owner_node.claim_count,
            Some(1),
            "ordinary dynamic map output must not hold a replay maintenance claim"
        );

        app.update();
        assert_eq!(
            *scalar_seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![NonCloneToken(7)]
        );
        assert_eq!(
            *vec_seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![NonCloneToken(11)]
        );
        assert_eq!(
            *map_seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![(NonCloneToken(19), NonCloneToken(190))]
        );

        scalar_claim.release(app.world_mut());
        vec_claim.release(app.world_mut());
        map_claim.release(app.world_mut());
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();

        // 2. Even when retained behind `.broadcast()`, an ordinary dynamic output has no
        //    maintenance claim, so releasing external claims immediately deactivates the dynamic
        //    owner (`1 -> 0`) and drains its active slot subscriptions.
        type RetainedUpdate = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;
        let retained_updates = Arc::new(Mutex::new(VecDeque::from([RetainedUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::always(42).dedupe().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
        let retained_materializer = ScalarSignalMaterializer::<i32>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    retained_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|queued| queued.last().copied())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        });
        let retained_broadcast = MaterializedTestSignal {
            materializer: retained_materializer,
        }
        .broadcast();
        let external_claim = retained_broadcast.clone().register(app.world_mut());
        let snapshot = graph_snapshot(app.world_mut());
        let retained_owner = single_static_dependency_of(app.world(), &snapshot, external_claim.entity());
        assert!(crate::inspect_scalar_replay_cache::<i32>(app.world(), retained_owner).is_none());
        app.update();
        assert_eq!(owner_subscriptions(app.world(), retained_owner).len(), 1);
        assert_eq!(
            graph_snapshot(app.world_mut())
                .node(*retained_owner)
                .and_then(|node| node.claim_count),
            Some(1),
        );

        external_claim.release(app.world_mut());
        assert!(
            owner_subscriptions(app.world(), retained_owner).is_empty(),
            "ordinary dynamic owner must drain active subscriptions immediately on 1 -> 0 release"
        );
        assert!(
            graph_snapshot(app.world_mut())
                .node(*retained_owner)
                .is_some_and(|node| node.registrar_inactive && node.claim_count == Some(0)),
            "retained broadcast handle keeps structure alive without keeping ordinary dynamic owner active"
        );

        drop(retained_broadcast);
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
        assert!(app.world().get_entity(*retained_owner).is_err());
    }

    #[test]
    fn dynamic_operator_rejects_shared_move_only_replayable_terminal_and_allows_broadcast() {
        type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<TestReplaySignal>, (usize, i32)>;
        type KeyedUpdate = DynamicManagerUpdate<KeyedSlotCommand<&'static str, TestReplaySignal>, (&'static str, i32)>;
        type ScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;

        let make_shared_move_only = |seed: i32| MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::<i32>::new(move |registrar| {
                registrar.replayable_signal_source(seed, |In(())| None::<i32>, OutputDelivery::move_only())
            }),
        };
        let shared_move_only_indexed = make_shared_move_only(77);
        let shared_broadcast = signal::once(88).cache().broadcast();

        let mut app = test_app();

        // 1. Indexed cross-slot collision on an uninitialized shared move-only replayable terminal
        //    rejects the entire `ReplaceAll` cohort with `MoveOnlyFanout`, marks that shared
        //    materializer `Rejected` (`SharedMaterializerRejected` on retry), and preserves the
        //    previously active generation.
        let indexed_errors = Arc::new(Mutex::new(Vec::new()));
        let indexed_errors_sink = Arc::clone(&indexed_errors);
        let indexed_updates = Arc::new(Mutex::new(VecDeque::from([
            IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![signal::once(10).cache().boxed_replayable()],
                }],
                DynamicEmissions::None,
            ),
            IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![
                        shared_move_only_indexed.clone().cache().boxed_replayable(),
                        shared_move_only_indexed.clone().cache().boxed_replayable(),
                    ],
                }],
                DynamicEmissions::None,
            ),
            IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![shared_move_only_indexed.cache().boxed_replayable()],
                }],
                DynamicEmissions::None,
            ),
            IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![
                        shared_broadcast.clone().boxed_replayable(),
                        shared_broadcast.clone().boxed_replayable(),
                    ],
                }],
                DynamicEmissions::None,
            ),
        ])));
        let indexed_outputs = Arc::new(Mutex::new(Vec::new()));
        let indexed_outputs_sink = Arc::clone(&indexed_outputs);
        let indexed_claim = ScalarSignalMaterializer::<()>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    indexed_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            let output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        indexed_errors_sink
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(error);
                    }),
            )?;
            registrar.signal_entity(
                output,
                move |In(batch): In<Vec<(usize, i32)>>| {
                    indexed_outputs_sink
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(batch);
                    Some(())
                },
                OutputDelivery::move_only(),
            )
        })
        .register(app.world_mut());
        let snapshot = graph_snapshot(app.world_mut());
        let indexed_owner = single_static_dependency_of(app.world(), &snapshot, indexed_claim.entity());

        app.update();
        let gen1_subscriptions = owner_subscriptions(app.world(), indexed_owner);
        assert_eq!(gen1_subscriptions.len(), 1);
        assert_eq!(
            *indexed_outputs.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![(0, 10)]]
        );

        app.update();
        assert_eq!(
            *indexed_errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![SignalRegistrationError::MoveOnlyFanout]
        );
        assert_eq!(
            owner_subscriptions(app.world(), indexed_owner),
            gen1_subscriptions,
            "rejected cross-slot move-only cohort must preserve active generation 1"
        );

        app.update();
        assert_eq!(
            *indexed_errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![
                SignalRegistrationError::MoveOnlyFanout,
                SignalRegistrationError::SharedMaterializerRejected,
            ]
        );
        assert_eq!(owner_subscriptions(app.world(), indexed_owner), gen1_subscriptions);

        app.update();
        assert_eq!(owner_subscriptions(app.world(), indexed_owner).len(), 2);
        assert_eq!(
            *indexed_outputs.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![(0, 10)], vec![(0, 88), (1, 88)]]
        );
        indexed_claim.release(app.world_mut());

        // 2. Keyed cross-slot collision on a fresh shared move-only replayable terminal rejects
        //    `MoveOnlyFanout` and succeeds with broadcast.
        let shared_move_only_keyed = make_shared_move_only(78);
        let keyed_errors = Arc::new(Mutex::new(Vec::new()));
        let keyed_errors_sink = Arc::clone(&keyed_errors);
        let keyed_updates = Arc::new(Mutex::new(VecDeque::from([
            KeyedUpdate::new(
                [KeyedSlotCommand::ReplaceAll {
                    entries: vec![
                        ("a", shared_move_only_keyed.clone().cache().boxed_replayable()),
                        ("b", shared_move_only_keyed.cache().boxed_replayable()),
                    ],
                }],
                DynamicEmissions::None,
            ),
            KeyedUpdate::new(
                [KeyedSlotCommand::ReplaceAll {
                    entries: vec![
                        ("a", shared_broadcast.clone().boxed_replayable()),
                        ("b", shared_broadcast.boxed_replayable()),
                    ],
                }],
                DynamicEmissions::None,
            ),
        ])));
        let keyed_claim = ScalarSignalMaterializer::<Vec<(&'static str, i32)>>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    keyed_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One((key, value)))
                    .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        keyed_errors_sink
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(error);
                    }),
            )
        })
        .register(app.world_mut());
        let keyed_owner = keyed_claim.entity();

        app.update();
        assert_eq!(
            *keyed_errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![SignalRegistrationError::MoveOnlyFanout]
        );
        assert!(owner_subscriptions(app.world(), keyed_owner).is_empty());

        app.update();
        assert_eq!(owner_subscriptions(app.world(), keyed_owner).len(), 2);
        keyed_claim.release(app.world_mut());

        // 3. External active dependent + dynamic `ScalarSlotCommand::Select` collision AND
        //    `SameSignalIdentity::Restart` re-selection collision on an already-materialized
        //    move-only replayable source.
        let shared_move_only = make_shared_move_only(79);
        let external_claim = shared_move_only
            .clone()
            .map(|In(_): In<i32>| ())
            .register(app.world_mut());
        let scalar_errors = Arc::new(Mutex::new(Vec::new()));
        let scalar_errors_sink = Arc::clone(&scalar_errors);
        let scalar_updates = Arc::new(Mutex::new(VecDeque::from([
            // Rejected because `external_claim` already holds an active dependent on
            // `shared_move_only`.
            ScalarUpdate::new(
                [ScalarSlotCommand::Select {
                    signal: shared_move_only.clone().cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ),
            // Succeeds once `external_claim` is released before the next update!
            ScalarUpdate::new(
                [ScalarSlotCommand::Select {
                    signal: shared_move_only.clone().cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ),
            // Rejected under `SameSignalIdentity::Restart` because activating the new adapter while
            // the current adapter is still active requires broadcast fan-out!
            ScalarUpdate::new(
                [ScalarSlotCommand::Select {
                    signal: shared_move_only.cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ),
        ])));
        let scalar_claim = ScalarSignalMaterializer::<i32>::new(|registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    scalar_updates
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop_front()
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|queued| queued.last().copied())
                    .same_signal_identity(SameSignalIdentity::Restart)
                    .on_candidate_error(move |error| {
                        scalar_errors_sink
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(error);
                    }),
            )
        })
        .register(app.world_mut());
        let scalar_owner = scalar_claim.entity();

        app.update();
        assert_eq!(
            *scalar_errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![SignalRegistrationError::MoveOnlyFanout]
        );
        assert!(owner_subscriptions(app.world(), scalar_owner).is_empty());

        external_claim.release(app.world_mut());
        app.update();
        let active_move_only_sub = owner_subscriptions(app.world(), scalar_owner);
        assert_eq!(active_move_only_sub.len(), 1);

        app.update();
        assert_eq!(
            *scalar_errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![
                SignalRegistrationError::MoveOnlyFanout,
                SignalRegistrationError::MoveOnlyFanout,
            ]
        );
        assert_eq!(
            owner_subscriptions(app.world(), scalar_owner),
            active_move_only_sub,
            "Restart collision on move-only terminal must preserve the active slot subscription"
        );

        scalar_claim.release(app.world_mut());
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
    }

    #[test]
    fn scalar_keep_current_replays_fresh_snapshot_after_clear_and_warming_transient_clear() {
        type Step = DynamicManagerStep<ScalarSlotCommand<TestReplaySignal>, i32>;
        type Update = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;

        #[derive(Resource, Default)]
        struct TransientWarmingOutput(Option<i32>);

        let ready_a = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_source(10, |In(())| None::<i32>, OutputDelivery::broadcast())
            }),
        }
        .cache()
        .broadcast();

        let warming_transient = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_entity(
                    (),
                    |In(()), mut pending: ResMut<TransientWarmingOutput>| pending.0.take(),
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            }),
        }
        .cache()
        .boxed_replayable();

        let updates = Arc::new(Mutex::new(VecDeque::from([
            // Initial selection of A -> emits [1, 10] on slot { identity: 1, generation: 1 }.
            Update::sequenced([Step::command(
                ScalarSlotCommand::Select {
                    signal: ready_a.clone().boxed_replayable(),
                },
                DynamicEmissions::One(1),
            )]),
            // Same identity without Clear -> KeepCurrent suppresses duplicate snapshot, emits only
            // [99].
            Update::sequenced([Step::command(
                ScalarSlotCommand::Select {
                    signal: ready_a.clone().boxed_replayable(),
                },
                DynamicEmissions::One(99),
            )]),
            // Failure Mode A: [Clear, Select(A)] -> allocates new slot lineage and replays A's
            // snapshot 10.
            Update::sequenced([
                Step::command(ScalarSlotCommand::Clear, DynamicEmissions::One(-1)),
                Step::command(
                    ScalarSlotCommand::Select {
                        signal: ready_a.clone().boxed_replayable(),
                    },
                    DynamicEmissions::One(2),
                ),
            ]),
            // Failure Mode B: [Select(warming_transient), Clear, Select(A)] -> holds cohort until
            // warming_transient emits 77, then publishes [77, -2, 3, 10] with A's fresh snapshot!
            Update::sequenced([
                Step::command(
                    ScalarSlotCommand::Select {
                        signal: warming_transient,
                    },
                    DynamicEmissions::None,
                ),
                Step::command(ScalarSlotCommand::Clear, DynamicEmissions::One(-2)),
                Step::command(
                    ScalarSlotCommand::Select {
                        signal: ready_a.boxed_replayable(),
                    },
                    DynamicEmissions::One(3),
                ),
            ]),
        ])));

        let recorded = Arc::new(Mutex::new(Vec::<Vec<i32>>::new()));
        let mut app = test_app();
        app.init_resource::<TransientWarmingOutput>();

        let claim = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::<Vec<i32>>::new({
                let updates = updates.clone();
                move |registrar| {
                    let manager = registrar.signal_entity(
                        (),
                        move |In(())| {
                            updates
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .pop_front()
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::scalar(manager)
                            .replay_scalar::<i32>()
                            .adapter(|(), value| DynamicEmissions::One(value))
                            .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }
            }),
        }
        .map({
            let recorded = recorded.clone();
            move |In(batch): In<Vec<i32>>| {
                recorded
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(batch);
            }
        })
        .register(app.world_mut());

        // Frame 1: initial Select(A) -> [1, 10].
        app.update();
        assert_eq!(
            *recorded.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1, 10]]
        );

        // Frame 2: Select(A) without Clear -> KeepCurrent emits only [99].
        app.update();
        assert_eq!(
            *recorded.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1, 10], vec![99]]
        );

        // Frame 3 (Failure Mode A): [Clear, Select(A)] -> [-1, 2, 10].
        app.update();
        assert_eq!(
            *recorded.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1, 10], vec![99], vec![-1, 2, 10]]
        );

        // Frame 4 (Failure Mode B, still warming): cohort held, nothing new emitted yet.
        app.update();
        assert_eq!(
            *recorded.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1, 10], vec![99], vec![-1, 2, 10]]
        );

        // Frame 5: warming_transient warms with 77 -> cohort publishes [77, -2, 3, 10].
        app.world_mut().resource_mut::<TransientWarmingOutput>().0 = Some(77);
        app.update();
        assert_eq!(
            *recorded.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1, 10], vec![99], vec![-1, 2, 10], vec![77, -2, 3, 10]]
        );

        claim.release(app.world_mut());
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
    }

    #[test]
    fn dynamic_operator_on_deactivate_runs_and_queued_adapter_releases_immediately() {
        type Step = DynamicManagerStep<ScalarSlotCommand<TestReplaySignal>, i32>;
        type Update = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;

        #[derive(Resource, Default)]
        struct TransientEvent(Option<i32>);

        let deactivations = Arc::new(Mutex::new(0usize));
        let transient_selected = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_entity(
                    (),
                    |In(()), mut event: ResMut<TransientEvent>| event.0.take(),
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            }),
        }
        .cache()
        .boxed_replayable();
        let ready_selected = signal::once(5).cache().broadcast();

        let updates = Arc::new(Mutex::new(VecDeque::from([
            Update::sequenced([
                Step::command(
                    ScalarSlotCommand::Select {
                        signal: transient_selected,
                    },
                    DynamicEmissions::None,
                ),
                Step::command(
                    ScalarSlotCommand::Select {
                        signal: ready_selected.clone().boxed_replayable(),
                    },
                    DynamicEmissions::None,
                ),
            ]),
            Update::sequenced([Step::command(
                ScalarSlotCommand::Select {
                    signal: ready_selected.boxed_replayable(),
                },
                DynamicEmissions::None,
            )]),
        ])));

        let mut app = test_app();
        app.init_resource::<TransientEvent>();

        let materializer = ScalarSignalMaterializer::<Vec<i32>>::new({
            let updates = updates.clone();
            let deactivations = deactivations.clone();
            move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        updates
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .pop_front()
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                        .same_signal_identity(SameSignalIdentity::Restart)
                        .on_deactivate(move || {
                            *deactivations.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) += 1;
                        }),
                )
            }
        });
        let retained = materializer.shared_clone();
        let claim = materializer.register(app.world_mut());
        let owner = claim.entity();

        // Stage the pending cohort [Select(transient_selected), Select(ready_selected)].
        app.update();
        let subs = owner_subscriptions(app.world(), owner);
        assert_eq!(subs.len(), 2);
        let transient_sub = subs[0];

        // Simulate a queued adapter event drained via process_dynamic_input(..., None) (e.g.
        // reentrant queue drain after the adapter has already returned): the retired transient
        // adapter must be released immediately rather than deferred into after_output_cleanup!
        let binding = DynamicAdapterBinding {
            owner,
            selected: transient_sub.selected,
            adapter: transient_sub.adapter,
            slot: transient_sub.slot,
        };
        process_dynamic_input(
            app.world_mut(),
            owner,
            DynamicRuntimeInput::Adapter(binding, Box::new(42_i32)),
            None,
        );

        assert!(
            !app.world()
                .get::<DynamicSignalOwner>(*owner)
                .expect("dynamic owner")
                .runtime
                .has_after_output_cleanup(),
            "queued adapter drain must release retired transient adapter immediately instead of deferring to after_output_cleanup"
        );
        assert_eq!(owner_subscriptions(app.world(), owner).len(), 1);

        // Deactivate 1 -> 0: on_deactivate fires once and slot identity resets for reactivation.
        assert_eq!(
            *deactivations.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            0
        );
        claim.release(app.world_mut());
        assert_eq!(
            *deactivations.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            1
        );

        let reactivated = retained.register(app.world_mut());
        app.update();
        let reactivated_subs = owner_subscriptions(app.world(), reactivated.entity());
        assert_eq!(reactivated_subs.len(), 1);
        assert_eq!(
            reactivated_subs[0].slot.identity, 1,
            "deactivation must reset next_slot_identity so reactivation starts cleanly at slot identity 1"
        );

        reactivated.release(app.world_mut());
        assert_eq!(
            *deactivations.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            2
        );
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
    }

    #[test]
    fn dynamic_candidates_inherit_owner_explicit_schedule() {
        type Update = DynamicManagerUpdate<ScalarSlotCommand<TestReplaySignal>, i32>;

        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash, ScheduleLabel)]
        struct ExplicitOwnerSchedule;

        let mut app = App::new();
        app.init_schedule(ExplicitOwnerSchedule);
        app.add_plugins((
            MinimalPlugins,
            JonmoPlugin::default().with_schedule::<ExplicitOwnerSchedule>(),
        ));

        let untagged_selected = signal::once(123).cache().boxed_replayable();
        let updates = Arc::new(Mutex::new(VecDeque::from([Update::new(
            [ScalarSlotCommand::Select {
                signal: untagged_selected,
            }],
            DynamicEmissions::None,
        )])));

        let claim = MaterializedTestSignal {
            materializer: ScalarSignalMaterializer::<i32>::new(move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| {
                        updates
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .pop_front()
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(|queued| queued.last().copied())
                        .same_signal_identity(SameSignalIdentity::Restart),
                )
            }),
        }
        .schedule::<ExplicitOwnerSchedule>()
        .register(app.world_mut());
        let owner = claim.entity();

        process_signal_graph_for_schedule(ExplicitOwnerSchedule.intern())(app.world_mut());

        let subs = owner_subscriptions(app.world(), owner);
        assert_eq!(subs.len(), 1);
        let snapshot = graph_snapshot(app.world_mut());
        let expected_schedule = crate::graph::ScheduleSnapshot::new(ExplicitOwnerSchedule.intern());
        for entity in [owner, subs[0].adapter, subs[0].selected] {
            let node = snapshot.node(*entity).expect("dynamic entity should exist");
            assert_eq!(
                node.cached_schedule,
                Some(expected_schedule.clone()),
                "entity {entity:?} should inherit ExplicitOwnerSchedule from dynamic owner"
            );
        }

        claim.release(app.world_mut());
        crate::cleanup::drain_world_cleanup(app.world_mut());
        app.world_mut().flush();
    }
}
