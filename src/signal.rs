//! Signal builders and combinators for constructing reactive [`System`] dependency graphs.
//!
//! This module provides the core [`Signal`] trait and its extension trait [`SignalExt`], which
//! together form the foundation for declarative, push-based reactive dataflows in [jonmo](crate).
//! For a high-level overview of the signal graph runtime and key concepts, see the [`Signal`] trait
//! documentation.
use super::{
    graph::SignalClaim,
    signal_map::SignalMap,
    signal_vec::{SignalVec, VecDiff},
    utils::ancestor_map,
};
use crate::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery, ReplayInitialization,
    SameSignalIdentity, ScalarSignalMaterializer, ScalarSlotCommand, SignalRegistrationError,
};
use bevy_ecs::{
    prelude::*,
    schedule::ScheduleLabel,
    system::{Adapt, CombinatorSystem, Combine, IntoAdapterSystem, RunSystemError, SystemIn},
};
cfg_if::cfg_if! {
    if #[cfg(feature = "tracing")] {
        use core::fmt;
    }
}
use bevy_platform::prelude::*;
cfg_if::cfg_if! {
    if #[cfg(feature = "time")] {
        use bevy_time::{Time, Timer, TimerMode};
        use core::time::Duration;
    }
}
use core::{marker::PhantomData, ops};

/// A lazy, composable scalar chain in [jonmo](crate)'s reactive dependency graph.
///
/// # Overview
///
/// A concrete [`Signal`] value is a lazy Rust description of dataflow. Registration materializes
/// its framework-created backing into one or more world-resident signal entities and returns one
/// move-only direct claim on the terminal. Cloning a cloneable chain shares its lazy terminal
/// identity; cloning the Rust value does not itself acquire a claim.
///
/// The trait remains externally implementable: custom types provide a framework-created
/// [`ScalarSignalMaterializer`] and receive no raw graph or ownership authority.
///
/// # Runtime Model
///
/// Once per frame (in the schedule configured via [`JonmoPlugin`](crate::JonmoPlugin), default
/// [`Last`](bevy_app::Last)), the signal graph is processed:
///
/// 1. Source signals (those with no upstreams) run first, producing outputs.
/// 2. Outputs are forwarded to downstream signals in topological order, ensuring each signal
///    receives its inputs only after all its upstreams have run.
/// 3. Propagation continues recursively until all reachable signals have been processed.
///
/// This is a **push-based** model: values flow downstream automatically each frame a system
/// produces output.
///
/// # Flow Control via [`Option`]
///
/// Signal systems return [`Option<T>`] (or types that convert to it). Returning [`None`] terminates
/// propagation for that branch of the graph for the current frame, so downstream signals simply
/// won't run. This enables filtering, gating, and conditional logic directly within the signal
/// graph.
///
/// # Replay Inspection and Imperative Evaluation
///
/// Signals are push-based by default and execute within configured Bevy schedules. Replayable
/// terminals can be inspected without running systems via [`crate::inspect_scalar_replay_cache`],
/// [`crate::inspect_vec_replay_cache`], and [`crate::inspect_map_replay_cache`]. When
/// out-of-schedule imperative subgraph execution is explicitly required, enable the non-default
/// `imperative-signal-evaluation` feature for `jonmo::graph::run_signal_subgraph_once`.
///
/// # Composition
///
/// Signals compose via the combinators in [`SignalExt`]. Methods like [`map`](SignalExt::map),
/// [`dedupe`](SignalExt::dedupe), [`filter`](SignalExt::filter), and [`switch`](SignalExt::switch)
/// allow complex dataflows to be built declaratively from simple primitives. Type erasure via
/// boxing ([`.boxed`](SignalExt::boxed)) and [`SignalEither`] enable heterogeneous signal
/// composition when concrete types differ across branches.
///
/// # Related Traits
///
/// - [`SignalExt`]: extension trait providing all signal combinators
/// - [`SignalVec`]: collection-oriented signals with diff-based semantics for [`Vec`] mutations
/// - [`SignalMap`]: collection-oriented signals with diff-based semantics for
///   [`BTreeMap`](alloc::collections::BTreeMap) mutations
/// - [`Broadcast`]: explicit fan-out wrapper for sharing a scalar chain across multiple consumers
///
/// Ordinary scalar signal chains are linear (`!Clone`); call [`SignalExt::broadcast`] to opt into
/// shared fan-out:
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// fn assert_clone<T: Clone>(_: &T) {}
///
/// let chain = signal::from_system(|In(_)| 1).map_in(|x: i32| x + 1);
/// assert_clone(&chain);
/// ```
///
/// Registration and combinators consume the chain, so a chain cannot be reused after registration
/// and [`SignalExt::cache`] cannot be called on a [`SignalClaim`]:
///
/// ```compile_fail,E0382
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let chain = signal::from_system(|In(_)| 1);
/// let _first = chain.register(&mut world);
/// let _second = chain.register(&mut world);
/// ```
///
/// ```compile_fail,E0599
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// let mut world = World::new();
/// let claim = signal::from_system(|In(_)| 1).register(&mut world);
/// let _ = claim.cache();
/// ```
///
/// Public implementations provide only materializer transfer; they cannot define or override a
/// world-taking registration method:
///
/// ```compile_fail,E0407
/// use bevy_ecs::world::World;
/// use jonmo::{ScalarSignalMaterializer, graph::SignalClaim, signal::Signal};
///
/// struct InvalidSignal;
///
/// impl Signal for InvalidSignal {
///     type Item = i32;
///
///     fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
///         unimplemented!()
///     }
///
///     fn register_signal(self, _world: &mut World) -> SignalClaim {
///         unimplemented!()
///     }
/// }
/// ```
pub trait Signal: Send + Sync + 'static {
    /// Output type.
    type Item;

    /// Transfers this concrete chain's framework-created, one-shot materializer backing.
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item>;
}

impl<U: 'static> Signal for Box<dyn Signal<Item = U> + Send + Sync> {
    type Item = U;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        (*self).into_signal_materializer()
    }
}

/// A type-erased, move-only [`Signal`].
///
/// Useful when storing linear signals of different concrete types in the same collection or
/// returning them from branching logic.
///
/// ```compile_fail,E0277
/// use bevy_ecs::prelude::*;
/// use jonmo::prelude::*;
///
/// fn assert_clone<T: Clone>(_: &T) {}
///
/// let boxed = signal::from_system(|In(_)| 1).boxed();
/// assert_clone(&boxed);
/// ```
pub type BoxedSignal<T> = Box<dyn Signal<Item = T> + Send + Sync>;

/// A type-erased, move-only [`ReplayableSignal`].
pub struct BoxedReplayableSignal<T: 'static>(Box<dyn ReplayableSignal<Item = T> + Send + Sync>);

impl<T: 'static> BoxedReplayableSignal<T> {
    /// Returns this already-verified replayable signal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable terminal in-place to broadcast fan-out without adding another
    /// cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_scalar_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<Sched, T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_scalar_schedule(self)
    }

    /// Returns this already-boxed replayable signal without adding another allocation.
    pub fn boxed_replayable(self) -> Self {
        self
    }
}

impl<T: 'static> Signal for BoxedReplayableSignal<T> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.0.into_signal_materializer()
    }
}

impl<T: 'static> replayable_signal_sealed::Sealed for BoxedReplayableSignal<T> {}

impl<T> ReplayableSignal for BoxedReplayableSignal<T> where T: Clone + Send + Sync + 'static {}

/// Type-level mode marker for an ordinary non-replayable [`Broadcast`] handle.
pub struct OrdinaryBroadcast(());

/// Type-level mode marker for a verified replayable [`Broadcast`] handle.
pub struct ReplayableBroadcastMarker(());

/// Explicit cloneable fan-out wrapper created by [`SignalExt::broadcast`].
///
/// Cloning `Broadcast` shares one canonical lazy terminal identity without acquiring a graph claim
/// until a clone is registered or consumed by a downstream combinator. Downstream combinators on a
/// `Broadcast` clone return ordinary linear (`!Clone`) signal chains unless `.broadcast()` is
/// called again at the new fan-out point.
pub struct Broadcast<T: 'static, Mode = OrdinaryBroadcast> {
    materializer: ScalarSignalMaterializer<T>,
    _mode: PhantomData<fn() -> Mode>,
}

/// Alias for a [`Broadcast`] handle backed by a verified [`ReplayableSignal`] terminal.
pub type ReplayableBroadcast<T> = Broadcast<T, ReplayableBroadcastMarker>;

impl<T: Clone + 'static, Mode> Clone for Broadcast<T, Mode> {
    fn clone(&self) -> Self {
        Self {
            materializer: self.materializer.shared_clone(),
            _mode: PhantomData,
        }
    }
}

impl<T: 'static> Broadcast<T, OrdinaryBroadcast> {
    /// Returns this already-broadcast handle without adding another graph entity.
    pub fn broadcast(self) -> Self {
        self
    }
}

impl<T: 'static> Broadcast<T, ReplayableBroadcastMarker> {
    /// Returns this already-verified replayable broadcast handle without adding another cache
    /// entity, claim, or targeted snapshot transfer.
    pub fn cache(self) -> Self {
        self
    }

    /// Returns this already-broadcast replayable handle without adding another graph entity.
    pub fn broadcast(self) -> Self {
        self
    }

    /// Assigns a schedule to this verified replayable broadcast handle while preserving its replay
    /// capability.
    ///
    /// Note that `.schedule::<Sched>()` returns a linear [`ReplayableScheduled`] handle and tags
    /// the shared canonical terminal when materialized. When multiple clones should share the
    /// scheduled terminal, prefer calling `.schedule::<Sched>().broadcast()` (rather than cloning
    /// before `.schedule::<Sched>()`), as assigning conflicting schedules to an already-registered
    /// canonical terminal fails registration.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<Sched, T>
    where
        T: Clone + Send + Sync + 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<T: 'static, Mode: Send + Sync + 'static> Signal for Broadcast<T, Mode> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<T: 'static> replayable_signal_sealed::Sealed for Broadcast<T, ReplayableBroadcastMarker> {}

impl<T: Clone + Send + Sync + 'static> ReplayableSignal for Broadcast<T, ReplayableBroadcastMarker> {}

pub(crate) fn broadcast_from_materializer<T: 'static>(materializer: ScalarSignalMaterializer<T>) -> Broadcast<T> {
    Broadcast {
        materializer,
        _mode: PhantomData,
    }
}

fn replayable_scalar_broadcast<S>(signal: S) -> ReplayableBroadcast<S::Item>
where
    S: ReplayableSignal,
    S::Item: Clone + Send + Sync + 'static,
{
    let upstream = Box::new(signal).into_signal_materializer();
    let materializer = ScalarSignalMaterializer::new(move |registrar| {
        let upstream = registrar.materialize_signal(upstream)?;
        registrar
            .scalar_broadcast_replayable_terminal(upstream)?
            .ok_or(SignalRegistrationError::ConflictingCanonicalState)
    });
    Broadcast {
        materializer,
        _mode: PhantomData,
    }
}

fn replayable_scalar_schedule<Sched, S>(signal: S) -> ReplayableScheduled<Sched, S::Item>
where
    Sched: ScheduleLabel + Default + 'static,
    S: ReplayableSignal + 'static,
{
    let upstream = Box::new(signal).into_signal_materializer();
    let materializer = ScalarSignalMaterializer::new(move |registrar| {
        let upstream = registrar.materialize_signal(upstream)?;
        registrar.apply_schedule(upstream, Sched::default())
    });
    Scheduled {
        materializer,
        _marker: PhantomData,
    }
}

pub(crate) mod replayable_signal_sealed {
    pub trait Sealed {}
}

/// A scalar signal whose terminal replay contract is managed and verified by Jonmo.
///
/// A newly attached dependent receives a targeted authoritative snapshot and its own relationship
/// cursor; only accepted revisions newer than that cursor are subsequently delivered. `Warming`
/// means no authoritative item exists yet. For `Signal<Item = Option<T>>`, `Ready(None)` is an
/// authoritative item rather than warming. A retained replay terminal owns exactly one private
/// maintenance claim so state and revision remain current. Replay guarantees eventual correct
/// handoff, not same-frame or zero-latency switching.
pub trait ReplayableSignal: Signal + replayable_signal_sealed::Sealed {}

/// A scalar cache wrapper created by [`SignalExt::cache`].
pub struct CachedSignal<Upstream: Signal> {
    materializer: ScalarSignalMaterializer<Upstream::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream: Signal> CachedSignal<Upstream> {
    /// Returns this already-verified cache without adding another graph entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable cache terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<Upstream::Item>
    where
        Upstream::Item: Clone + Send + Sync + 'static,
    {
        replayable_scalar_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable cache terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<Sched, Upstream::Item>
    where
        Upstream: 'static,
        Upstream::Item: Clone + Send + Sync + 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<Upstream> Signal for CachedSignal<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + Send + Sync + 'static,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_sealed::Sealed for CachedSignal<Upstream> where Upstream: Signal {}

impl<Upstream> ReplayableSignal for CachedSignal<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + Send + Sync + 'static,
{
}

/// Signal graph node which takes an input of [`In<()>`] and has no upstreams. See
/// [`from_system`] for examples.
pub(crate) struct Source<O: 'static> {
    materializer: ScalarSignalMaterializer<O>,
}

impl<O> Signal for Source<O>
where
    O: 'static,
{
    type Item = O;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which applies a [`System`] to its upstream, see [`.map`](SignalExt::map).
pub struct Map<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for Map<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = O;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

struct FilterSystemAdapter<T>(PhantomData<fn() -> T>);

impl<T, Predicate> Adapt<Predicate> for FilterSystemAdapter<T>
where
    T: Clone + Send + Sync + 'static,
    Predicate: System<In = In<T>, Out = bool>,
{
    type In = In<T>;
    type Out = Option<T>;

    fn adapt(
        &mut self,
        input: T,
        run_system: impl FnOnce(SystemIn<'_, Predicate>) -> Result<Predicate::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(run_system(input.clone())?.then_some(input))
    }
}

struct MapBoolSystemCombinator<O>(PhantomData<fn() -> O>);

impl<TrueSystem, FalseSystem, O, SystemOutput> Combine<TrueSystem, FalseSystem> for MapBoolSystemCombinator<O>
where
    TrueSystem: System<In = In<()>, Out = SystemOutput>,
    FalseSystem: System<In = In<()>, Out = SystemOutput>,
    O: 'static,
    SystemOutput: Into<Option<O>>,
{
    type In = In<bool>;
    type Out = Option<O>;

    fn combine<T>(
        input: bool,
        data: &mut T,
        true_system: impl FnOnce(SystemIn<'_, TrueSystem>, &mut T) -> Result<TrueSystem::Out, RunSystemError>,
        false_system: impl FnOnce(SystemIn<'_, FalseSystem>, &mut T) -> Result<FalseSystem::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(if input {
            true_system((), data)?.into()
        } else {
            false_system((), data)?.into()
        })
    }
}

struct MapTrueSystemAdapter<O>(PhantomData<fn() -> O>);

impl<System, O> Adapt<System> for MapTrueSystemAdapter<O>
where
    System: bevy_ecs::system::System<In = In<()>, Out = O>,
    O: 'static,
{
    type In = In<bool>;
    type Out = Option<Option<O>>;

    fn adapt(
        &mut self,
        input: bool,
        run_system: impl FnOnce(SystemIn<'_, System>) -> Result<System::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(Some(if input { Some(run_system(())?) } else { None }))
    }
}

struct MapFalseSystemAdapter<O>(PhantomData<fn() -> O>);

impl<System, O> Adapt<System> for MapFalseSystemAdapter<O>
where
    System: bevy_ecs::system::System<In = In<()>, Out = O>,
    O: 'static,
{
    type In = In<bool>;
    type Out = Option<Option<O>>;

    fn adapt(
        &mut self,
        input: bool,
        run_system: impl FnOnce(SystemIn<'_, System>) -> Result<System::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(Some(if !input { Some(run_system(())?) } else { None }))
    }
}

struct MapOptionSystemCombinator<I, O>(PhantomData<fn() -> (I, O)>);

impl<SomeSystem, NoneSystem, I, O, SystemOutput> Combine<SomeSystem, NoneSystem> for MapOptionSystemCombinator<I, O>
where
    SomeSystem: System<In = In<I>, Out = SystemOutput>,
    NoneSystem: System<In = In<()>, Out = SystemOutput>,
    I: 'static,
    O: 'static,
    SystemOutput: Into<Option<O>>,
{
    type In = In<Option<I>>;
    type Out = Option<O>;

    fn combine<T>(
        input: Option<I>,
        data: &mut T,
        some_system: impl FnOnce(SystemIn<'_, SomeSystem>, &mut T) -> Result<SomeSystem::Out, RunSystemError>,
        none_system: impl FnOnce(SystemIn<'_, NoneSystem>, &mut T) -> Result<NoneSystem::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(match input {
            Some(value) => some_system(value, data)?.into(),
            None => none_system((), data)?.into(),
        })
    }
}

struct MapSomeSystemAdapter<I, O>(PhantomData<fn() -> (I, O)>);

impl<System, I, O> Adapt<System> for MapSomeSystemAdapter<I, O>
where
    System: bevy_ecs::system::System<In = In<I>, Out = O>,
    I: 'static,
    O: 'static,
{
    type In = In<Option<I>>;
    type Out = Option<Option<O>>;

    fn adapt(
        &mut self,
        input: Option<I>,
        run_system: impl FnOnce(SystemIn<'_, System>) -> Result<System::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(Some(match input {
            Some(value) => Some(run_system(value)?),
            None => None,
        }))
    }
}

struct MapNoneSystemAdapter<I, O>(PhantomData<fn() -> (I, O)>);

impl<System, I, O> Adapt<System> for MapNoneSystemAdapter<I, O>
where
    System: bevy_ecs::system::System<In = In<()>, Out = O>,
    I: 'static,
    O: 'static,
{
    type In = In<Option<I>>;
    type Out = Option<Option<O>>;

    fn adapt(
        &mut self,
        input: Option<I>,
        run_system: impl FnOnce(SystemIn<'_, System>) -> Result<System::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        Ok(Some(match input {
            Some(_) => None,
            None => Some(run_system(())?),
        }))
    }
}

/// Signal graph node wrapper that assigns a schedule to a signal chain, see
/// [`.schedule`](SignalExt::schedule).
pub struct Scheduled<Sched, I: 'static, Mode = OrdinaryBroadcast> {
    materializer: ScalarSignalMaterializer<I>,
    _marker: PhantomData<fn() -> (Sched, Mode)>,
}

/// Alias for a [`Scheduled`] handle backed by a verified [`ReplayableSignal`] terminal.
pub type ReplayableScheduled<Sched, I> = Scheduled<Sched, I, ReplayableBroadcastMarker>;

impl<Sched: 'static, I: 'static> Scheduled<Sched, I, ReplayableBroadcastMarker> {
    /// Returns this already-verified replayable scheduled handle without adding another cache
    /// entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable scheduled terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<I>
    where
        I: Clone + Send + Sync + 'static,
    {
        replayable_scalar_broadcast(self)
    }

    /// Reassigns this verified replayable terminal's schedule while preserving its replay
    /// capability.
    pub fn schedule<NextSched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<NextSched, I>
    where
        I: Clone + Send + Sync + 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<Sched: 'static, I: 'static, Mode: Send + Sync + 'static> Signal for Scheduled<Sched, I, Mode> {
    type Item = I;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Sched: 'static, I: 'static> replayable_signal_sealed::Sealed for Scheduled<Sched, I, ReplayableBroadcastMarker> {}

impl<Sched: 'static, I: Clone + Send + Sync + 'static> ReplayableSignal
    for Scheduled<Sched, I, ReplayableBroadcastMarker>
{
}

/// Signal graph node which maps its upstream [`Entity`] to a corresponding [`Component`], see
/// [.component](SignalExt::component).
pub struct MapComponent<Upstream, C: 'static> {
    signal: Map<Upstream, C>,
}

impl<Upstream, C> Signal for MapComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = C;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which maps its upstream [`Entity`] to a corresponding [`Option<Component>`],
/// see [.component_option](SignalExt::component_option).
pub struct ComponentOption<Upstream, C: 'static> {
    signal: Map<Upstream, Option<C>>,
}

impl<Upstream, C> Signal for ComponentOption<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = Option<C>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which maps its upstream [`Entity`] to its [`Changed`] `C` [`Component`],
/// see [.component_changed](SignalExt::component_changed).
pub struct ComponentChanged<Upstream, C: 'static> {
    signal: Map<Upstream, C>,
}

impl<Upstream, C> Signal for ComponentChanged<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = C;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node with maps its upstream [`Entity`] to whether it has some [`Component`], see
/// [`.has_component`](SignalExt::has_component).
pub struct HasComponent<Upstream, C> {
    signal: Map<Upstream, bool>,
    _marker: PhantomData<fn() -> C>,
}

impl<Upstream, C> Signal for HasComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = bool;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which does not forward upstream duplicates, see
/// [`.dedupe`](SignalExt::dedupe).
pub struct Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: 'static,
{
    materializer: ScalarSignalMaterializer<Upstream::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: 'static,
{
    /// Returns this intrinsically replayable signal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable dedupe terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<Upstream::Item>
    where
        Upstream::Item: Clone + Send + Sync + 'static,
    {
        replayable_scalar_broadcast(self)
    }

    /// Assigns a schedule to this intrinsically replayable dedupe terminal while preserving its
    /// replay capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<Sched, Upstream::Item>
    where
        Upstream: 'static,
        Upstream::Item: Clone + Send + Sync + 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<Upstream> Signal for Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + Send + Sync + 'static,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_sealed::Sealed for Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream> ReplayableSignal for Dedupe<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + Send + Sync + 'static,
{
}

/// Signal graph node that terminates forever after forwarding a single upstream value, see
/// [`.first`](SignalExt::first).
pub struct First<Upstream>
where
    Upstream: Signal,
{
    signal: Take<Upstream>,
}

impl<Upstream> Signal for First<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node that terminates forever after forwarding `count` upstream values, see
/// [`.take`](SignalExt::take).
pub struct Take<Upstream>
where
    Upstream: Signal,
{
    signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for Take<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node that skips the first `count` upstream values before forwarding, see
/// [`.skip`](SignalExt::skip).
pub struct Skip<Upstream>
where
    Upstream: Signal,
{
    signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Signal for Skip<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which combines two upstreams, see [`.zip`](SignalExt::zip).
pub struct Zip<Left, Right>
where
    Left: Signal,
    Right: Signal,
    Left::Item: 'static,
    Right::Item: 'static,
{
    materializer: ScalarSignalMaterializer<(Left::Item, Right::Item)>,
    _marker: PhantomData<fn() -> (Left, Right)>,
}

impl<Left, Right> Signal for Zip<Left, Right>
where
    Left: Signal,
    Right: Signal,
    Left::Item: Clone + Send + Sync + 'static,
    Right::Item: Clone + Send + Sync + 'static,
{
    type Item = (Left::Item, Right::Item);

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which outputs equality between its upstream and a fixed value, see
/// [`.eq`](SignalExt::eq).
pub struct Eq<Upstream> {
    signal: Map<Upstream, bool>,
}

impl<Upstream> Signal for Eq<Upstream>
where
    Upstream: Signal,
{
    type Item = bool;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which outputs inequality between its upstream and a fixed value, see
/// [`.neq`](SignalExt::neq).
pub struct Neq<Upstream> {
    signal: Map<Upstream, bool>,
}

impl<Upstream> Signal for Neq<Upstream>
where
    Upstream: Signal,
{
    type Item = bool;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which applies [`ops::Not`] to its upstream, see [`.not`](SignalExt::not).
pub struct Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: ops::Not,
    <<Upstream as Signal>::Item as ops::Not>::Output: 'static,
{
    signal: Map<Upstream, <Upstream::Item as ops::Not>::Output>,
}

impl<Upstream> Signal for Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: ops::Not,
    <<Upstream as Signal>::Item as ops::Not>::Output: 'static,
{
    type Item = <Upstream::Item as ops::Not>::Output;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

/// Signal graph node which selectively terminates propagation, see [`.filter`](SignalExt::filter).
pub struct Filter<Upstream>
where
    Upstream: Signal,
    Upstream::Item: 'static,
{
    materializer: ScalarSignalMaterializer<Upstream::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> Signal for Filter<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Clone + Send + Sync + 'static,
{
    type Item = Upstream::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which forwards the upstream output [`Signal`]'s output, see
/// [`.flatten`](SignalExt::flatten).
pub struct Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Signal,
{
    materializer: ScalarSignalMaterializer<<Upstream::Item as Signal>::Item>,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<fn() -> (Upstream, Upstream::Item)>,
}

impl<Upstream> Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: ReplayableSignal,
    <Upstream::Item as Signal>::Item: Clone + Send + Sync + 'static,
{
    /// Returns this intrinsically replayable signal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable flatten terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<<Upstream::Item as Signal>::Item> {
        replayable_scalar_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable flatten terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> ReplayableScheduled<Sched, <Upstream::Item as Signal>::Item>
    where
        Upstream: 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<Upstream> Signal for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: ReplayableSignal,
    <Upstream::Item as Signal>::Item: Clone + Send + Sync + 'static,
{
    type Item = <Upstream::Item as Signal>::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

impl<Upstream> replayable_signal_sealed::Sealed for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: ReplayableSignal,
    <Upstream::Item as Signal>::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream> ReplayableSignal for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: ReplayableSignal,
    <Upstream::Item as Signal>::Item: Clone + Send + Sync + 'static,
{
}

/// Signal graph node which maps its upstream to a [`Signal`], forwarding its output, see
/// [`.switch`](SignalExt::switch).
pub struct Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: ReplayableSignal,
    Other::Item: Clone,
{
    signal: Flatten<Map<Upstream, Other>>,
}

impl<Upstream, Other> Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: ReplayableSignal,
    Other::Item: Clone + Send + Sync + 'static,
{
    /// Returns this intrinsically replayable signal without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable switch terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<Other::Item> {
        replayable_scalar_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable switch terminal while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<Sched, Other::Item>
    where
        Upstream: 'static,
        Other: 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<Upstream, Other> Signal for Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: ReplayableSignal,
    Other::Item: Clone + Send + Sync + 'static,
{
    type Item = Other::Item;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        Box::new(self.signal).into_signal_materializer()
    }
}

impl<Upstream, Other> replayable_signal_sealed::Sealed for Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: ReplayableSignal,
    Other::Item: Clone + Send + Sync + 'static,
{
}

impl<Upstream, Other> ReplayableSignal for Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: ReplayableSignal,
    Other::Item: Clone + Send + Sync + 'static,
{
}

/// Signal graph node which maps its upstream to a [`SignalVec`], see
/// [`.switch_signal_vec`](SignalExt::switch_signal_vec).
pub struct SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalVec,
{
    materializer: crate::VecSignalMaterializer<Switched::Item>,
    _marker: PhantomData<fn() -> (Upstream, Switched)>,
}

impl<Upstream, Switched> SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: super::signal_vec::ReplayableSignalVec,
    Switched::Item: Clone + Send + Sync + 'static,
{
    /// Returns this already-verified replayable switch vector without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable switch vector terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> super::signal_vec::ReplayableBroadcastVec<Switched::Item> {
        super::signal_vec::replayable_vec_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable switch vector while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> super::signal_vec::ReplayableScheduledVec<Sched, Switched::Item>
    where
        Upstream: 'static,
        Switched: 'static,
    {
        super::signal_vec::replayable_vec_schedule(self)
    }
}

impl<Upstream, Switched> SignalVec for SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: super::signal_vec::ReplayableSignalVec,
    Switched::Item: Clone + Send + Sync + 'static,
{
    type Item = Switched::Item;

    fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which maps its upstream to a [`SignalMap`], see
/// [`.switch_signal_map`](SignalExt::switch_signal_map).
pub struct SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalMap,
{
    materializer: crate::MapSignalMaterializer<Switched::Key, Switched::Value>,
    _marker: PhantomData<fn() -> (Upstream, Switched)>,
}

impl<Upstream, Switched> SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: super::signal_map::ReplayableSignalMap,
    Switched::Key: Clone + Ord + Send + Sync + 'static,
    Switched::Value: Clone + Send + Sync + 'static,
{
    /// Returns this already-verified replayable switch map without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable switch map terminal in-place to broadcast fan-out without
    /// adding another cache entity or maintenance claim.
    pub fn broadcast(self) -> super::signal_map::ReplayableBroadcastMap<Switched::Key, Switched::Value> {
        super::signal_map::replayable_map_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable switch map while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(
        self,
    ) -> super::signal_map::ReplayableScheduledMap<Sched, Switched::Key, Switched::Value>
    where
        Upstream: 'static,
        Switched: 'static,
    {
        super::signal_map::replayable_map_schedule(self)
    }
}

impl<Upstream, Switched> SignalMap for SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: super::signal_map::ReplayableSignalMap,
    Switched::Key: Clone + Ord + Send + Sync + 'static,
    Switched::Value: Clone + Send + Sync + 'static,
{
    type Key = Switched::Key;
    type Value = Switched::Value;

    fn into_signal_map_materializer(self: Box<Self>) -> crate::MapSignalMaterializer<Self::Key, Self::Value> {
        self.materializer
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "time")] {
        /// Signal graph node which delays the propagation of subsequent upstream outputs, see
        /// [`.throttle`](SignalExt::throttle).
        pub struct Throttle<Upstream>
        where
            Upstream: Signal,
        {
            signal: Map<Upstream, Upstream::Item>,
        }

        impl<Upstream> Signal for Throttle<Upstream>
        where
            Upstream: Signal,
        {
            type Item = Upstream::Item;

            fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
                Box::new(self.signal).into_signal_materializer()
            }
        }
    }
}

/// Signal graph node whose [`System`] depends on its upstream [`bool`] value, see
/// [`.map_bool`](SignalExt::map_bool).
pub struct MapBool<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for MapBool<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    O: 'static,
{
    type Item = O;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`true`], see
/// [`.map_true`](SignalExt::map_true).
pub struct MapTrue<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<Option<O>>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for MapTrue<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    O: 'static,
{
    type Item = Option<O>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`false`], see
/// [`.map_false`](SignalExt::map_false).
pub struct MapFalse<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<Option<O>>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for MapFalse<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    O: 'static,
{
    type Item = Option<O>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node whose [`System`] depends on its upstream [`Option`] value, see
/// [`.map_option`](SignalExt::map_option).
pub struct MapOption<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<O>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for MapOption<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = O;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`Some`], see
/// [`.map_some`](SignalExt::map_some).
pub struct MapSome<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<Option<O>>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for MapSome<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = Option<O>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`None`], see
/// [`.map_none`](SignalExt::map_none).
pub struct MapNone<Upstream, O: 'static> {
    materializer: ScalarSignalMaterializer<Option<O>>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, O> Signal for MapNone<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = Option<O>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Signal graph node which maps its upstream [`Vec`] to a [`SignalVec`], see
/// [`.to_signal_vec`](SignalExt::to_signal_vec).
pub struct ToSignalVec<Upstream>
where
    Upstream: Signal,
    Upstream::Item: IntoIterator,
    <Upstream::Item as IntoIterator>::Item: 'static,
{
    materializer: crate::VecSignalMaterializer<<Upstream::Item as IntoIterator>::Item>,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream, T> SignalVec for ToSignalVec<Upstream>
where
    Upstream: Signal<Item = Vec<T>>,
    T: 'static,
{
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> crate::VecSignalMaterializer<Self::Item> {
        self.materializer
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "tracing")] {
        /// Signal graph node that debug logs its upstream's output, see [`.debug`](SignalExt::debug).
        pub struct Debug<Upstream>
        where
            Upstream: Signal,
        {
            signal: Map<Upstream, Upstream::Item>,
        }

        impl<Upstream> Signal for Debug<Upstream>
        where
            Upstream: Signal,
        {
            type Item = Upstream::Item;

            fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
                Box::new(self.signal).into_signal_materializer()
            }
        }
    }
}

/// Creates a [`Signal`] from a [`System`] that takes [`In<()>`].
pub fn from_system<O, IOO, F, M>(system: F) -> impl Signal<Item = O>
where
    O: Send + Sync + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<()>, IOO, M> + Send + Sync + 'static,
{
    Source {
        materializer: ScalarSignalMaterializer::new(move |registrar| {
            registrar.signal_entity((), system, OutputDelivery::move_only())
        }),
    }
}

/// Creates a [`Signal`] from an [`FnMut`].
pub fn from_function<O, F>(mut f: F) -> impl Signal<Item = O>
where
    O: Send + Sync + 'static,
    F: FnMut() -> O + Send + Sync + 'static,
{
    from_system(move |In(_)| f())
}

/// Creates a [`Signal`] that always outputs the same value.
pub fn always<O>(value: O) -> impl Signal<Item = O>
where
    O: Clone + Send + Sync + 'static,
{
    from_system(move |In(_)| value.clone())
}

/// Creates a [`Signal`] that outputs a value once and then terminates forever.
pub fn once<O>(value: O) -> impl Signal<Item = O>
where
    O: Send + Sync + 'static,
{
    let mut value = Some(value);
    from_system(move |In(_)| value.take())
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`](crate::utils::LazyEntity).
pub fn from_entity(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = Entity> {
    let mut entity = Some(entity);
    from_system(move |In(_), mut cached: Local<Option<Entity>>| {
        *cached.get_or_insert_with(|| entity.take().unwrap().into())
    })
}

/// Creates a [`Signal`] from an [`Entity`]'s or [`LazyEntity`](crate::utils::LazyEntity)'s
/// `generations`-th generation ancestor's [`Entity`], terminating for frames where it does not
/// exist. Passing `0` to `generation` will output the [`Entity`] itself.
pub fn from_ancestor(
    entity: impl Into<Entity> + Send + Sync + 'static,
    generations: usize,
) -> impl Signal<Item = Entity> {
    from_entity(entity).map(ancestor_map(generations))
}

/// Creates a [`Signal`] from an [`Entity`]'s or [`LazyEntity`](crate::utils::LazyEntity)'s parent's
/// [`Entity`], terminating for frames where it does not exist.
pub fn from_parent(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = Entity> {
    from_ancestor(entity, 1)
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`](crate::utils::LazyEntity) and a
/// [`Component`], terminating for the frame if the [`Entity`] does not exist or the [`Component`]
/// does not exist on the [`Entity`].
pub fn from_component<C>(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = C>
where
    C: Component + Clone,
{
    let mut entity = Some(entity);
    from_system(move |In(_), mut cached: Local<Option<Entity>>, components: Query<&C>| {
        let entity = *cached.get_or_insert_with(|| entity.take().unwrap().into());
        components.get(entity).ok().cloned()
    })
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`](crate::utils::LazyEntity) and a
/// [`Component`], always outputting an [`Option`].
pub fn from_component_option<C>(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = Option<C>>
where
    C: Component + Clone,
{
    let mut entity = Some(entity);
    from_system(move |In(_), mut cached: Local<Option<Entity>>, components: Query<&C>| {
        let entity = *cached.get_or_insert_with(|| entity.take().unwrap().into());
        Some(components.get(entity).ok().cloned())
    })
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`](crate::utils::LazyEntity) and a
/// [`Component`], only outputting on frames where the [`Component`] has [`Changed`]. Terminates for
/// the frame if the [`Entity`] does not exist, the [`Component`] does not exist, or the
/// [`Component`] has not changed.
pub fn from_component_changed<C>(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = C>
where
    C: Component + Clone,
{
    let mut entity = Some(entity);
    from_system(
        move |In(_), mut cached: Local<Option<Entity>>, components: Query<&C, Changed<C>>| {
            let entity = *cached.get_or_insert_with(|| entity.take().unwrap().into());
            components.get(entity).ok().cloned()
        },
    )
}

/// Creates a signal from a [`Resource`], terminating for the frame if the [`Resource`] does not
/// exist.
pub fn from_resource<R>() -> impl Signal<Item = R>
where
    R: Resource + Clone,
{
    from_system(move |In(_), resource: Option<Res<R>>| resource.as_deref().cloned())
}

/// Creates a signal from a [`Resource`], always outputting an [`Option`].
pub fn from_resource_option<R>() -> impl Signal<Item = Option<R>>
where
    R: Resource + Clone,
{
    from_system(move |In(_), resource: Option<Res<R>>| Some(resource.as_deref().cloned()))
}

/// Creates a signal from a [`Resource`], only outputting on frames where the [`Resource`] has
/// [`Changed`]. Terminates for the frame if the [`Resource`] does not exist or has not changed.
pub fn from_resource_changed<R>() -> impl Signal<Item = R>
where
    R: Resource + Clone,
{
    from_system(move |In(_), resource: Option<Res<R>>| resource.filter(|r| r.is_changed()).map(|r| r.clone()))
}

/// Converts an `Option<Signal<A>>` into a `Signal<Option<A>>`.
///
/// If the value is `None` then it emits `None` once and then terminates (implemented as
/// `signal::once(None)`).
///
/// If the value is `Some(signal)` then it will return the result of the signal,
/// except wrapped in `Some`.
///
/// When feeding the result into [`.switch`](SignalExt::switch) or [`.flatten`](SignalExt::flatten),
/// call [`.cache`](SignalExt::cache) so the selected branch implements [`ReplayableSignal`].
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// #[derive(Resource, Clone, PartialEq)]
/// struct Enabled(bool);
///
/// let mut world = World::new();
/// world.insert_resource(Enabled(false));
///
/// // Turn `Signal<Item = Option<Signal<Item = T>>>` into `Signal<Item = Option<T>>`.
/// let maybe_value = signal::from_resource::<Enabled>()
///     .dedupe()
///     .map_in(|enabled: Enabled| enabled.0)
///     .map_true_in(|| signal::from_system(|In(_)| 42))
///     .map_in(|opt| signal::option(opt).cache())
///     .flatten();
/// ```
pub struct OptionSignal<S: Signal>
where
    S::Item: 'static,
{
    materializer: ScalarSignalMaterializer<Option<S::Item>>,
    _marker: PhantomData<fn() -> S>,
}

impl<S> Signal for OptionSignal<S>
where
    S: Signal,
    S::Item: 'static,
{
    type Item = Option<S::Item>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.materializer
    }
}

/// Converts an [`Option<Signal>`] into a linear, non-replayable [`Signal`] producing
/// `Option<S::Item>`, emitting `None` once when `value` is `None`.
///
/// Because `option` does not implicitly attach a replay cache, `S::Item` does not need to
/// implement [`Clone`]. Chain [`.cache()`](SignalExt::cache) when passing the resulting signal to
/// replay-requiring consumers such as [`SignalExt::switch`].
pub fn option<S>(value: Option<S>) -> OptionSignal<S>
where
    S: Signal,
    S::Item: Send + Sync + 'static,
{
    let branch = match value {
        Some(signal) => signal.map_in(Some).left_either(),
        None => once(None).right_either(),
    };
    OptionSignal {
        materializer: Box::new(branch).into_signal_materializer(),
        _marker: PhantomData,
    }
}

/// Enables returning different concrete [`Signal`] types from branching logic without boxing,
/// although note that all [`Signal`]s are boxed internally regardless.
///
/// Inspired by <https://github.com/rayon-rs/either>.
#[allow(missing_docs)]
pub enum SignalEither<L, R>
where
    L: Signal,
    R: Signal,
{
    Left(L),
    Right(R),
}

impl<T, L, R> SignalEither<L, R>
where
    T: Clone + Send + Sync + 'static,
    L: ReplayableSignal<Item = T>,
    R: ReplayableSignal<Item = T>,
{
    /// Returns this already-verified replayable branch without adding another cache entity.
    pub fn cache(self) -> Self {
        self
    }

    /// Normalizes this replayable branch terminal in-place to broadcast fan-out without adding
    /// another cache entity or maintenance claim.
    pub fn broadcast(self) -> ReplayableBroadcast<T> {
        replayable_scalar_broadcast(self)
    }

    /// Assigns a schedule to this verified replayable branch while preserving its replay
    /// capability.
    pub fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> ReplayableScheduled<Sched, T>
    where
        L: 'static,
        R: 'static,
    {
        replayable_scalar_schedule(self)
    }
}

impl<T: 'static, L: Signal<Item = T>, R: Signal<Item = T>> Signal for SignalEither<L, R>
where
    L: Signal<Item = T>,
    R: Signal<Item = T>,
{
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        match *self {
            SignalEither::Left(left) => Box::new(left).into_signal_materializer(),
            SignalEither::Right(right) => Box::new(right).into_signal_materializer(),
        }
    }
}

impl<T, L, R> replayable_signal_sealed::Sealed for SignalEither<L, R>
where
    T: Clone + Send + Sync + 'static,
    L: ReplayableSignal<Item = T>,
    R: ReplayableSignal<Item = T>,
{
}

impl<T, L, R> ReplayableSignal for SignalEither<L, R>
where
    T: Clone + Send + Sync + 'static,
    L: ReplayableSignal<Item = T>,
    R: ReplayableSignal<Item = T>,
{
}

/// Blanket trait for transforming [`Signal`]s into [`SignalEither::Left`] or
/// [`SignalEither::Right`].
pub trait IntoSignalEither: Sized
where
    Self: Signal,
{
    /// Wrap this [`Signal`] in the [`SignalEither::Left`] variant.
    ///
    /// Useful for conditional branching where different [`Signal`] types need to be returned
    /// from the same function or closure, particularly with combinators like
    /// [`.switch`](SignalExt::switch) or [`.flatten`](SignalExt::flatten).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource)]
    /// struct UseSquare(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(UseSquare(true));
    ///
    /// let signal = signal::from_system(|In(_), res: Res<UseSquare>| res.0)
    ///     .map(move |In(use_square): In<bool>| {
    ///         if use_square {
    ///             signal::from_system(|In(_)| 10)
    ///                 .map(|In(x): In<i32>| x * x)
    ///                 .cache()
    ///                 .left_either()
    ///         } else {
    ///             signal::from_system(|In(_)| 42).cache().right_either()
    ///         }
    ///     })
    ///     .flatten();
    /// // Both branches produce compatible SignalEither types despite having different
    /// // concrete Signal types (CachedSignal<Map<...>> vs CachedSignal<...>)
    /// ```
    fn left_either<R>(self) -> SignalEither<Self, R>
    where
        R: Signal,
    {
        SignalEither::Left(self)
    }

    /// Wrap this [`Signal`] in the [`SignalEither::Right`] variant.
    ///
    /// Useful for conditional branching where different [`Signal`] types need to be returned
    /// from the same function or closure, particularly with combinators like
    /// [`.switch`](SignalExt::switch) or [`.flatten`](SignalExt::flatten).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource)]
    /// struct Mode(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(Mode(false));
    ///
    /// let signal = signal::from_system(|In(_), res: Res<Mode>| res.0)
    ///     .map(move |In(mode): In<bool>| {
    ///         if mode {
    ///             signal::from_system(|In(_)| "A").cache().left_either()
    ///         } else {
    ///             signal::from_system(|In(_)| "B")
    ///                 .filter(|In(x): In<&str>| x.len() > 0)
    ///                 .cache()
    ///                 .right_either()
    ///         }
    ///     })
    ///     .flatten();
    /// // Both branches are compatible via SignalEither despite different types
    /// ```
    fn right_either<L>(self) -> SignalEither<L, Self>
    where
        L: Signal,
    {
        SignalEither::Right(self)
    }
}

impl<T: Signal> IntoSignalEither for T {}

/// Extension trait providing combinator methods for [`Signal`]s.
pub trait SignalExt: Signal {
    /// Explicitly shares this signal across multiple downstream consumers.
    ///
    /// Cloning the returned [`Broadcast`] handle shares one canonical lazy terminal identity
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
    /// let _ = signal::once(NonClone(1)).broadcast();
    /// ```
    fn broadcast(self) -> Broadcast<Self::Item>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            if let Some(upstream) = registrar.scalar_try_broadcast_terminal(upstream)? {
                return Ok(upstream);
            }
            registrar.signal_entity(
                upstream,
                |In(value): In<Self::Item>| Some(value),
                OutputDelivery::broadcast(),
            )
        });
        Broadcast {
            materializer,
            _mode: PhantomData,
        }
    }

    /// Normalizes this terminal to framework-managed replayability.
    ///
    /// A non-replayable terminal gains exactly one managed cache entity and maintenance lifecycle.
    /// An already verified replayable terminal preserves its identity and adds no entity, replay
    /// state, or second maintenance claim.
    ///
    /// Move-only (`!Clone`) items cannot be cached for replay:
    ///
    /// ```compile_fail,E0277
    /// use jonmo::prelude::*;
    ///
    /// struct NonClone(usize);
    ///
    /// let _ = signal::once(NonClone(1)).cache();
    /// ```
    fn cache(self) -> CachedSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(Box::new(self).into_signal_materializer())?;
            if let Some(upstream) = registrar.scalar_cache_terminal(upstream)? {
                return Ok(upstream);
            }
            registrar.replayable_signal_entity(
                upstream,
                |In(value): In<Self::Item>| Some(value),
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        CachedSignal {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass the output of this [`Signal`] to a [`System`], continuing propagation if the [`System`]
    /// returns [`Some`] or terminating for the frame if it returns [`None`]. If the [`System`]
    /// logic is infallible, wrapping the result in an [`Option`] is unnecessary.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system(|In(_)| 1).map(|In(x): In<i32>| x * 2); // outputs `2`
    /// ```
    fn map<O, IOO, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(upstream, system, OutputDelivery::move_only())
        });
        Map {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Pass the output of this [`Signal`] to an [`FnMut`], continuing propagation if the [`FnMut`]
    /// returns [`Some`] or terminating for the frame if it returns [`None`]. If the [`FnMut`] logic
    /// is infallible, wrapping the result in an [`Option`] is unnecessary.
    ///
    /// Convenient when additional [`SystemParam`](bevy_ecs::system::SystemParam)s aren't necessary.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system(|In(_)| 1).map_in(|x: i32| x * 2); // outputs `2`
    /// ```
    fn map_in<O, IOO, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        F: FnMut(Self::Item) -> IOO + Send + Sync + 'static,
    {
        self.map(move |In(item)| function(item))
    }

    /// Pass a reference to the output of this [`Signal`] to an [`FnMut`], continuing propagation if
    /// the [`FnMut`] returns [`Some`] or terminating for the frame if it returns [`None`]. If the
    /// [`FnMut`] logic is infallible, wrapping the result in an [`Option`] is unnecessary.
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
    /// signal::from_system(|In(_)| 1).map_in_ref(ToString::to_string); // outputs `"1"`
    /// ```
    fn map_in_ref<O, IOO, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        F: FnMut(&Self::Item) -> IOO + Send + Sync + 'static,
    {
        self.map(move |In(item)| function(&item))
    }

    /// Map this [`Signal`]'s output [`Entity`] to its `C` [`Component`], terminating for the frame
    /// if it does not exist.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Component, Clone)]
    /// struct Value(u32);
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Value(0)).id();
    /// signal::from_entity(entity).component::<Value>(); // outputs `Value(0)`
    ///
    /// let entity = world.spawn_empty().id();
    /// signal::from_entity(entity).component::<Value>(); // terminates
    /// ```
    fn component<C>(self) -> MapComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone,
    {
        MapComponent {
            signal: self.map(|In(entity): In<Entity>, components: Query<&C>| components.get(entity).ok().cloned()),
        }
    }

    /// Map this [`Signal`]'s output [`Entity`] to its `C` [`Component`], always outputting an
    /// [`Option`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Component, Clone)]
    /// struct Value(u32);
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Value(0)).id();
    /// signal::from_entity(entity).component_option::<Value>(); // outputs `Some(Value(0))`
    ///
    /// let entity = world.spawn_empty().id();
    /// signal::from_entity(entity).component::<Value>(); // outputs `None`
    /// ```
    fn component_option<C>(self) -> ComponentOption<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone,
    {
        ComponentOption {
            signal: self
                .map(|In(entity): In<Entity>, components: Query<&C>| Some(components.get(entity).ok().cloned())),
        }
    }

    /// Map this [`Signal`]'s output [`Entity`] to its `C` [`Component`] on frames it has
    /// [`Changed`], terminating for the frame if it does not exist or has not [`Changed`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Component, Clone)]
    /// struct Value(u32);
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Value(0)).id();
    /// signal::from_entity(entity).component_changed::<Value>(); // outputs `Value(0)` on changed frames
    ///
    /// let entity = world.spawn_empty().id();
    /// signal::from_entity(entity).component_changed::<Value>(); // terminates
    /// ```
    fn component_changed<C>(self) -> ComponentChanged<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone,
    {
        ComponentChanged {
            signal: self
                .map(|In(entity): In<Entity>, components: Query<&C, Changed<C>>| components.get(entity).ok().cloned()),
        }
    }

    /// Map this [`Signal`]'s output [`Entity`] to a [`bool`] representing whether it has some
    /// [`Component`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Component, Clone)]
    /// struct Value(u32);
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Value(0)).id();
    /// signal::from_entity(entity).has_component::<Value>(); // outputs `true`
    ///
    /// let entity = world.spawn_empty().id();
    /// signal::from_entity(entity).has_component::<Value>(); // outputs `false`
    /// ```
    fn has_component<C>(self) -> HasComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component,
    {
        HasComponent {
            signal: self.map(|In(entity): In<Entity>, components: Query<&C>| components.contains(entity)),
            _marker: PhantomData,
        }
    }

    /// Terminates this [`Signal`] on frames where the output was the same as the last.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state / 2
    ///     }
    /// })
    /// .dedupe(); // outputs `0`, `1`, `2`, `3`, ...
    /// ```
    fn dedupe(self) -> Dedupe<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.replayable_signal_entity(
                upstream,
                |In(current): In<Self::Item>, mut cache: Local<Option<Self::Item>>| {
                    let changed = cache.as_ref() != Some(&current);
                    if changed {
                        *cache = Some(current.clone());
                        Some(current)
                    } else {
                        None
                    }
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        });
        Dedupe {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Outputs up to the first `count` values from this [`Signal`], and then terminates for all
    /// subsequent frames.
    ///
    /// If `count` is `0`, this will never propagate any values.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state
    ///     }
    /// })
    /// .take(2); // outputs `1`, `2`, then terminates forever
    /// ```
    fn take(self, count: usize) -> Take<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        Take {
            signal: self.map(move |In(item): In<Self::Item>, mut emitted: Local<usize>| {
                if *emitted >= count {
                    None
                } else {
                    *emitted += 1;
                    Some(item)
                }
            }),
        }
    }

    /// Skips the first `count` values from this [`Signal`], then outputs all subsequent values.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state
    ///     }
    /// })
    /// .skip(2); // skips `1`, `2`, then outputs `3`, `4`, `5`, ...
    /// ```
    fn skip(self, count: usize) -> Skip<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        Skip {
            signal: self.map(move |In(item): In<Self::Item>, mut skipped: Local<usize>| {
                if *skipped < count {
                    *skipped += 1;
                    None
                } else {
                    Some(item)
                }
            }),
        }
    }

    /// Outputs this [`Signal`]'s first value and then terminates for all subsequent frames.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state / 2
    ///     }
    /// })
    /// .first(); // outputs `0` then terminates forever
    /// ```
    fn first(self) -> First<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        First { signal: self.take(1) }
    }

    /// Output this [`Signal`]'s equality with a fixed value.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal = signal::from_system(|In(_)| 0).broadcast();
    /// signal.clone().eq(0); // outputs `true`
    /// signal.eq(1); // outputs `false`
    /// ```
    fn eq(self, value: Self::Item) -> Eq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + Send + Sync,
    {
        Eq {
            signal: self.map(move |In(item): In<Self::Item>| item == value),
        }
    }

    /// Output this [`Signal`]'s inequality with a fixed value.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal = signal::from_system(|In(_)| 0).broadcast();
    /// signal.clone().neq(0); // outputs `false`
    /// signal.neq(1); // outputs `true`
    /// ```
    fn neq(self, value: Self::Item) -> Neq<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + Send + Sync,
    {
        Neq {
            signal: self.map(move |In(item): In<Self::Item>| item != value),
        }
    }

    /// Applies [`ops::Not`] to this [`Signal`]'s output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system(|In(_)| true).not(); // outputs `false`
    /// ```
    fn not(self) -> Not<Self>
    where
        Self: Sized,
        <Self as Signal>::Item: ops::Not + 'static,
        <<Self as Signal>::Item as ops::Not>::Output: Send + Sync + 'static,
    {
        Not {
            signal: self.map(|In(item): In<Self::Item>| ops::Not::not(item)),
        }
    }

    /// Terminate this [`Signal`] on frames where the `predicate` [`System`] returns `false`.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state % 2
    ///     }
    /// })
    /// .filter(|In(i): In<usize>| i % 2 == 0); // outputs `2`, `4`, `6`, `8`, ...
    /// ```
    fn filter<M>(self, predicate: impl IntoSystem<In<Self::Item>, bool, M> + Send + Sync + 'static) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(
                upstream,
                IntoAdapterSystem::new(FilterSystemAdapter(PhantomData), predicate),
                OutputDelivery::move_only(),
            )
        });
        Filter {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Combines this [`Signal`] with another [`Signal`], outputting a tuple with both of their
    /// latest outputs. The resulting [`Signal`] will only output a value when both input
    /// [`Signal`]s have outputted a value.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal_1 = signal::from_system(|In(_)| 1);
    /// let signal_2 = signal::from_system(|In(_)| 2);
    /// signal_1.zip(signal_2); // outputs `(1, 2)`
    /// ```
    fn zip<Other>(self, other: Other) -> Zip<Self, Other>
    where
        Self: Sized,
        Other: Signal,
        Self::Item: Clone + Send + Sync + 'static,
        Other::Item: Clone + Send + Sync + 'static,
    {
        let left_wrapper =
            Box::new(self.map(|In(left): In<Self::Item>| (Some(left), None::<Other::Item>))).into_signal_materializer();
        let right_wrapper = Box::new(other.map(|In(right): In<Other::Item>| (None::<Self::Item>, Some(right))))
            .into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let left_wrapper = registrar.materialize_signal(left_wrapper)?;
            let right_wrapper = registrar.materialize_signal(right_wrapper)?;
            registrar.signal_entity(
                [left_wrapper, right_wrapper],
                #[allow(clippy::type_complexity)]
                move |In((left_option, right_option)): In<(Option<Self::Item>, Option<Other::Item>)>,
                      mut left_cache: Local<Option<Self::Item>>,
                      mut right_cache: Local<Option<Other::Item>>| {
                    if left_option.is_some() {
                        *left_cache = left_option;
                    }
                    if right_option.is_some() {
                        *right_cache = right_option;
                    }
                    left_cache
                        .as_ref()
                        .zip(right_cache.as_ref())
                        .map(|(left, right)| (left.clone(), right.clone()))
                },
                OutputDelivery::move_only(),
            )
        });

        Zip {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Outputs this [`Signal`]'s output [`Signal`]'s output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone, PartialEq)]
    /// struct Toggle(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(Toggle(false));
    /// let signal = signal::from_resource::<Toggle>()
    ///     .dedupe()
    ///     .map(move |In(toggle): In<Toggle>| {
    ///         if toggle.0 {
    ///             signal::from_system(|In(_)| 1).dedupe().left_either()
    ///         } else {
    ///             signal::from_system(|In(_)| 2).dedupe().right_either()
    ///         }
    ///     })
    ///     .flatten();
    ///
    /// // `signal` outputs `2`
    /// world.resource_mut::<Toggle>().0 = true;
    /// // `signal` outputs `1`
    /// ```
    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: ReplayableSignal + 'static,
        <Self::Item as Signal>::Item: Clone + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            let manager = registrar.signal_entity(
                upstream,
                |In(signal): In<Self::Item>| {
                    Some(DynamicManagerUpdate::new(
                        [ScalarSlotCommand::Select { signal }],
                        DynamicEmissions::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<<Self::Item as Signal>::Item>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .replayable_scalar_output(|values| values.into_iter().last())
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        });
        Flatten {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Maps this [`Signal`]'s output to another [`Signal`], switching to its output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone, PartialEq)]
    /// struct Toggle(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(Toggle(false));
    /// let signal =
    ///     signal::from_resource::<Toggle>()
    ///         .dedupe()
    ///         .switch(move |In(toggle): In<Toggle>| {
    ///             if toggle.0 {
    ///                 signal::from_system(|In(_)| 1).dedupe().left_either()
    ///             } else {
    ///                 signal::from_system(|In(_)| 2).dedupe().right_either()
    ///             }
    ///         });
    /// // `signal` outputs `2`
    /// world.resource_mut::<Toggle>().0 = true;
    /// // `signal` outputs `1`
    /// ```
    fn switch<S, F, M>(self, switcher: F) -> Switch<Self, S>
    where
        Self: Sized,
        Self::Item: 'static,
        S: ReplayableSignal + 'static,
        S::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        Switch {
            signal: self.map(switcher).flatten(),
        }
    }

    /// Maps this [`Signal`]'s output to a [`SignalVec`], switching to its output. Useful when the
    /// reactive list target changes depending on some other state.
    ///
    /// The selected [`SignalVec`] must implement
    /// [`ReplayableSignalVec`](super::signal_vec::ReplayableSignalVec). Call
    /// [`SignalVecExt::cache`](super::signal_vec::SignalVecExt::cache) before selection when its
    /// terminal is not intrinsically replayable.
    ///
    /// ```compile_fail,E0277
    /// use bevy_ecs::prelude::*;
    /// use jonmo::{OutputDelivery, VecSignalMaterializer, prelude::*};
    ///
    /// struct PlainVector;
    ///
    /// impl SignalVec for PlainVector {
    ///     type Item = i32;
    ///
    ///     fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
    ///         VecSignalMaterializer::new(|registrar| {
    ///             registrar.signal_vec_entity(
    ///                 (),
    ///                 |In(())| Some(vec![VecDiff::Replace { values: vec![1] }]),
    ///                 OutputDelivery::move_only(),
    ///             )
    ///         })
    ///     }
    /// }
    ///
    /// // Normalize `PlainVector` with `.cache()` before selecting it.
    /// let _ = signal::always(()).switch_signal_vec(|In(())| PlainVector);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone, PartialEq)]
    /// struct Toggle(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(Toggle(false));
    ///
    /// let signal = signal::from_resource::<Toggle>()
    ///     .dedupe()
    ///     .switch_signal_vec(move |In(toggle): In<Toggle>, world: &mut World| {
    ///         if toggle.0 {
    ///             MutableVec::builder()
    ///                 .values([1, 2, 3])
    ///                 .spawn(world)
    ///                 .signal_vec()
    ///         } else {
    ///             MutableVec::builder()
    ///                 .values([10, 20])
    ///                 .spawn(world)
    ///                 .signal_vec()
    ///         }
    ///     });
    /// // `signal` outputs `SignalVec -> [10, 20]`
    /// world.resource_mut::<Toggle>().0 = true;
    /// // `signal` outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn switch_signal_vec<S, F, M>(self, switcher: F) -> SwitchSignalVec<Self, S>
    where
        Self: Sized,
        S: super::signal_vec::ReplayableSignalVec,
        S::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        let manager = Box::new(self.map(switcher)).into_signal_materializer();
        let materializer = crate::VecSignalMaterializer::new(move |registrar| {
            let manager = registrar.materialize_signal(manager)?;
            let manager = registrar.signal_entity(
                manager,
                |In(signal): In<S>| {
                    Some(DynamicManagerUpdate::new(
                        [ScalarSlotCommand::Select { signal }],
                        DynamicEmissions::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_signal_vec::<S::Item>()
                    .adapter(|(), diffs| DynamicEmissions::Many(diffs))
                    .replayable_signal_vec_output(|diffs| (!diffs.is_empty()).then_some(diffs))
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        });
        SwitchSignalVec {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Maps this [`Signal`]'s output to a [`SignalMap`], switching to its output. Useful when the
    /// reactive map target changes depending on some other state.
    ///
    /// The selected [`SignalMap`] must implement
    /// [`ReplayableSignalMap`](super::signal_map::ReplayableSignalMap). Call
    /// [`SignalMapExt::cache`](super::signal_map::SignalMapExt::cache) before selection when its
    /// terminal is not intrinsically replayable.
    ///
    /// ```compile_fail,E0277
    /// use bevy_ecs::prelude::*;
    /// use jonmo::{MapSignalMaterializer, OutputDelivery, prelude::*};
    ///
    /// struct PlainMap;
    ///
    /// impl SignalMap for PlainMap {
    ///     type Key = i32;
    ///     type Value = i32;
    ///
    ///     fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
    ///         MapSignalMaterializer::new(|registrar| {
    ///             registrar.signal_map_entity(
    ///                 (),
    ///                 |In(())| Some(vec![MapDiff::Replace { entries: vec![(1, 2)] }]),
    ///                 OutputDelivery::move_only(),
    ///             )
    ///         })
    ///     }
    /// }
    ///
    /// // Normalize `PlainMap` with `.cache()` before selecting it.
    /// let _ = signal::always(()).switch_signal_map(|In(())| PlainMap);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone, PartialEq)]
    /// struct Toggle(bool);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(Toggle(false));
    ///
    /// let signal = signal::from_resource::<Toggle>()
    ///     .dedupe()
    ///     .switch_signal_map(move |In(toggle): In<Toggle>, world: &mut World| {
    ///         if toggle.0 {
    ///             MutableBTreeMap::builder()
    ///                 .values([(1, 2), (3, 4)])
    ///                 .spawn(world)
    ///                 .signal_map()
    ///         } else {
    ///             MutableBTreeMap::builder()
    ///                 .values([(10, 20)])
    ///                 .spawn(world)
    ///                 .signal_map()
    ///         }
    ///     });
    /// // `signal` outputs `SignalMap -> {10: 20}`
    /// world.resource_mut::<Toggle>().0 = true;
    /// // `signal` outputs `SignalMap -> {1: 2, 3: 4}`
    /// ```
    fn switch_signal_map<S, F, M>(self, switcher: F) -> SwitchSignalMap<Self, S>
    where
        Self: Sized,
        S: super::signal_map::ReplayableSignalMap,
        S::Key: Clone + Ord + Send + Sync + 'static,
        S::Value: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        let manager = Box::new(self.map(switcher)).into_signal_materializer();
        let materializer = crate::MapSignalMaterializer::new(move |registrar| {
            let manager = registrar.materialize_signal(manager)?;
            let manager = registrar.signal_entity(
                manager,
                |In(signal): In<S>| {
                    Some(DynamicManagerUpdate::new(
                        [ScalarSlotCommand::Select { signal }],
                        DynamicEmissions::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_signal_map::<S::Key, S::Value>()
                    .adapter(|(), diffs| DynamicEmissions::Many(diffs))
                    .replayable_signal_map_output(|diffs| (!diffs.is_empty()).then_some(diffs))
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        });
        SwitchSignalMap {
            materializer,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "time")]
    /// Delays subsequent outputs from this [`Signal`] for some [`Duration`].
    ///
    /// # Example
    ///
    /// ```
    /// use core::time::Duration;
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<usize>| {
    ///        *state += 1;
    ///        *state
    ///     }
    /// })
    /// .throttle(Duration::from_secs(1)); // outputs `1`, terminates for the next 1 second of frames, outputs `2`, terminates for the next 1 second of frames, outputs `3`, ...
    /// ```
    fn throttle(self, duration: Duration) -> Throttle<Self>
    where
        Self: Sized,
        Self::Item: Send + Sync + 'static,
    {
        Throttle {
            signal: self.map(
                move |In(item): In<Self::Item>, time: Res<Time>, mut timer_option: Local<Option<Timer>>| {
                    match timer_option.as_mut() {
                        None => {
                            *timer_option = Some(Timer::new(duration, TimerMode::Once));
                            Some(item)
                        }
                        Some(timer) => {
                            if timer.tick(time.delta()).is_finished() {
                                timer.reset();
                                Some(item)
                            } else {
                                None
                            }
                        }
                    }
                },
            ),
        }
    }

    /// Maps this [`Signal`] to some [`System`] depending on its [`bool`] output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_bool(|In(_)| 1, |In(_)| 0); // outputs `1`, `0`, `1`, `0`, ...
    /// ```
    fn map_bool<O, IOO, TF, FF, TM, FM>(self, true_system: TF, false_system: FF) -> MapBool<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        TF: IntoSystem<In<()>, IOO, TM> + Send + Sync + 'static,
        FF: IntoSystem<In<()>, IOO, FM> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            let system = CombinatorSystem::<MapBoolSystemCombinator<O>, _, _>::new(
                IntoSystem::into_system(true_system),
                IntoSystem::into_system(false_system),
                "signal::map_bool".into(),
            );
            registrar.signal_entity(upstream, system, OutputDelivery::move_only())
        });
        MapBool {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Maps this [`Signal`] to some [`FnMut`] depending on its [`bool`] output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_bool_in(|| 1, || 0); // outputs `1`, `0`, `1`, `0`, ...
    /// ```
    fn map_bool_in<O, IOO, TF, FF>(self, mut true_fn: TF, mut false_fn: FF) -> MapBool<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        TF: FnMut() -> IOO + Send + Sync + 'static,
        FF: FnMut() -> IOO + Send + Sync + 'static,
    {
        let materializer = Box::new(self.map::<O, _, _, _>(
            move |In(item): In<Self::Item>| {
                if item { true_fn().into() } else { false_fn().into() }
            },
        ))
        .into_signal_materializer();
        MapBool {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Switches to a [`Signal`] produced by one of two factories, depending on this [`Signal`]'s
    /// [`bool`] output.
    ///
    /// The selected factory is called for each upstream output. The two factories may produce
    /// different concrete [`Signal`] types as long as their [`Signal::Item`] types match.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal = signal::from_system(|In(_)| false)
    ///     .map_bool_signal(|| signal::always("enabled"), || signal::once("disabled"));
    /// // `signal` switches to the false branch and outputs `"disabled"`.
    /// ```
    fn map_bool_signal<I, TS, FS, TF, FF>(
        self,
        mut true_fn: TF,
        mut false_fn: FF,
    ) -> Switch<Self, SignalEither<CachedSignal<TS>, CachedSignal<FS>>>
    where
        Self: Sized + Signal<Item = bool>,
        I: Clone + Send + Sync + 'static,
        TS: Signal<Item = I>,
        FS: Signal<Item = I>,
        TF: FnMut() -> TS + Send + Sync + 'static,
        FF: FnMut() -> FS + Send + Sync + 'static,
    {
        self.switch(move |In(value): In<bool>| {
            if value {
                SignalEither::Left(SignalExt::cache(true_fn()))
            } else {
                SignalEither::Right(SignalExt::cache(false_fn()))
            }
        })
    }

    /// If this [`Signal`] outputs [`true`], output the [`System`] result wrapped in [`Some`],
    /// otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_true(|In(_)| 1); // outputs `Some(1)`, `None`, `Some(1)`, `None`, ...
    /// ```
    fn map_true<O, F, M>(self, system: F) -> MapTrue<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Send + Sync + 'static,
        F: IntoSystem<In<()>, O, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(
                upstream,
                IntoAdapterSystem::new(MapTrueSystemAdapter(PhantomData), system),
                OutputDelivery::move_only(),
            )
        });
        MapTrue {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`true`], output the [`FnMut`] result wrapped in [`Some`],
    /// otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_true_in(|| 1); // outputs `Some(1)`, `None`, `Some(1)`, `None`, ...
    /// ```
    fn map_true_in<O, F>(self, mut function: F) -> MapTrue<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Send + Sync + 'static,
        F: FnMut() -> O + Send + Sync + 'static,
    {
        let materializer =
            Box::new(self.map::<Option<O>, _, _, _>(
                move |In(item): In<Self::Item>| {
                    if item { Some(Some(function())) } else { Some(None) }
                },
            ))
            .into_signal_materializer();
        MapTrue {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`false`], output the [`System`] result wrapped in [`Some`],
    /// otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_false(|In(_)| 1); // outputs `None`, `Some(1)`, `None`, `Some(1)`, ...
    /// ```
    fn map_false<O, F, M>(self, system: F) -> MapFalse<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Send + Sync + 'static,
        F: IntoSystem<In<()>, O, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(
                upstream,
                IntoAdapterSystem::new(MapFalseSystemAdapter(PhantomData), system),
                OutputDelivery::move_only(),
            )
        });
        MapFalse {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`false`], output the [`FnMut`] result wrapped in [`Some`],
    /// otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_false_in(|| 1); // outputs `None`, `Some(1)`, `None`, `Some(1)`, ...
    /// ```
    fn map_false_in<O, F>(self, mut function: F) -> MapFalse<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Send + Sync + 'static,
        F: FnMut() -> O + Send + Sync + 'static,
    {
        let materializer =
            Box::new(self.map::<Option<O>, _, _, _>(
                move |In(item): In<Self::Item>| {
                    if !item { Some(Some(function())) } else { Some(None) }
                },
            ))
            .into_signal_materializer();
        MapFalse {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Maps this [`Signal`] to some [`System`] depending on its [`Option`] output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Option<bool>>| {
    ///        *state = if state.is_some() { None } else { Some(true) };
    ///        *state
    ///     }
    /// })
    /// .map_option(
    ///     |In(state): In<bool>| state,
    ///     |In(_)| false,
    /// ); // outputs `true`, `false`, `true`, `false`, ...
    /// ```
    fn map_option<I, O, IOO, SF, NF, SM, NM>(self, some_system: SF, none_system: NF) -> MapOption<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        SF: IntoSystem<In<I>, IOO, SM> + Send + Sync + 'static,
        NF: IntoSystem<In<()>, IOO, NM> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            let system = CombinatorSystem::<MapOptionSystemCombinator<I, O>, _, _>::new(
                IntoSystem::into_system(some_system),
                IntoSystem::into_system(none_system),
                "signal::map_option".into(),
            );
            registrar.signal_entity(upstream, system, OutputDelivery::move_only())
        });
        MapOption {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Maps this [`Signal`] to some [`FnMut`] depending on its [`Option`] output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Option<bool>>| {
    ///         *state = if state.is_some() { None } else { Some(true) };
    ///         *state
    ///     }
    /// })
    /// .map_option_in(|state: bool| state, || false); // outputs `true`, `false`, `true`, `false`, ...
    /// ```
    fn map_option_in<I, O, IOO, SF, NF>(self, mut some_fn: SF, mut none_fn: NF) -> MapOption<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        SF: FnMut(I) -> IOO + Send + Sync + 'static,
        NF: FnMut() -> IOO + Send + Sync + 'static,
    {
        let materializer = Box::new(self.map::<O, _, _, _>(move |In(item): In<Self::Item>| match item {
            Some(value) => some_fn(value).into(),
            None => none_fn().into(),
        }))
        .into_signal_materializer();
        MapOption {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`Some`], output the [`System`] (which takes [`In`] the [`Some`]
    /// value) result wrapped in [`Some`], otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Option<bool>>| {
    ///        *state = if state.is_some() { None } else { Some(true) };
    ///        *state
    ///     }
    /// })
    /// .map_some(
    ///     |In(state): In<bool>| state
    /// ); // outputs `Some(true)`, `None`, `Some(true)`, `None`, ...
    /// ```
    fn map_some<I, O, F, M>(self, system: F) -> MapSome<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Send + Sync + 'static,
        F: IntoSystem<In<I>, O, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(
                upstream,
                IntoAdapterSystem::new(MapSomeSystemAdapter(PhantomData), system),
                OutputDelivery::move_only(),
            )
        });
        MapSome {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`Some`], output the [`FnMut`] (which takes [`In`] the [`Some`]
    /// value) result wrapped in [`Some`], otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Option<bool>>| {
    ///         *state = if state.is_some() { None } else { Some(true) };
    ///         *state
    ///     }
    /// })
    /// .map_some_in(|state: bool| state); // outputs `Some(true)`, `None`, `Some(true)`, `None`, ...
    /// ```
    fn map_some_in<I, O, F>(self, mut function: F) -> MapSome<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Send + Sync + 'static,
        F: FnMut(I) -> O + Send + Sync + 'static,
    {
        let materializer = Box::new(
            self.map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>| match item {
                Some(value) => Some(Some(function(value))),
                None => Some(None),
            }),
        )
        .into_signal_materializer();
        MapSome {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`None`], output the [`System`] result wrapped in [`Some`],
    /// otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Option<bool>>| {
    ///         *state = if state.is_some() { None } else { Some(true) };
    ///         *state
    ///     }
    /// })
    /// .map_none(|In(_)| false); // outputs `None`, `Some(false)`, `None`, `Some(false)`, ...
    /// ```
    fn map_none<I, O, F, M>(self, none_system: F) -> MapNone<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Send + Sync + 'static,
        F: IntoSystem<In<()>, O, M> + Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(
                upstream,
                IntoAdapterSystem::new(MapNoneSystemAdapter(PhantomData), none_system),
                OutputDelivery::move_only(),
            )
        });
        MapNone {
            materializer,
            _marker: PhantomData,
        }
    }

    /// If this [`Signal`] outputs [`None`], output the [`FnMut`] result wrapped in [`Some`],
    /// otherwise output [`None`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Option<bool>>| {
    ///         *state = if state.is_some() { None } else { Some(true) };
    ///         *state
    ///     }
    /// })
    /// .map_none_in(|| false); // outputs `None`, `Some(false)`, `None`, `Some(false)`, ...
    /// ```
    fn map_none_in<I, O, F>(self, mut function: F) -> MapNone<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Send + Sync + 'static,
        F: FnMut() -> O + Send + Sync + 'static,
    {
        let materializer = Box::new(
            self.map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>| match item {
                Some(_) => Some(None),
                None => Some(Some(function())),
            }),
        )
        .into_signal_materializer();
        MapNone {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Transforms this [`Signal`]'s [`Vec`] output into a [`SignalVec`] by emitting one owned
    /// [`VecDiff::Replace`] per upstream vector emission.
    ///
    /// This conversion is purely structural: it neither deduplicates consecutive equal vectors nor
    /// installs a replay cache, so move-only items (`!Clone`, `!PartialEq`, such as
    /// `Builder`) flow directly through linear chains. Compose
    /// [`.dedupe()`](SignalExt::dedupe) upstream when unchanged vectors should be suppressed, or
    /// [`.cache()`](crate::signal_vec::SignalVecExt::cache) downstream when replay is required.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system({
    ///     move |In(_), mut state: Local<Vec<usize>>| {
    ///         let new = state
    ///             .get(state.len().saturating_sub(1))
    ///             .map(|last| last + 1)
    ///             .unwrap_or_default();
    ///         state.push(new);
    ///         state.clone()
    ///     }
    /// })
    /// .to_signal_vec(); // outputs a `SignalVec` of `[0]`, `[0, 1]`, `[0, 1, 2]`, `[0, 1, 2, 3]`, ...
    /// ```
    fn to_signal_vec<T>(self) -> ToSignalVec<Self>
    where
        Self: Sized,
        Self: Signal<Item = Vec<T>>,
        T: Send + Sync + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = crate::VecSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_vec_entity(
                upstream,
                |In(items): In<Vec<T>>| Some(vec![VecDiff::Replace { values: items }]),
                OutputDelivery::move_only(),
            )
        });
        ToSignalVec {
            materializer,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "tracing")]
    #[track_caller]
    /// Adds debug logging to this [`Signal`]'s output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// signal::from_system(|In(_)| 1)
    ///     .debug() // logs `1`
    ///     .map_in(|x: i32| x * 2)
    ///     .debug(); // logs `2`
    /// ```
    fn debug(self) -> Debug<Self>
    where
        Self: Sized,
        Self::Item: fmt::Debug + Send + Sync + 'static,
    {
        let location = core::panic::Location::caller();
        Debug {
            signal: self.map(move |In(item)| {
                bevy_log::debug!("[{}] {:#?}", location, item);
                item
            }),
        }
    }

    /// Erases the type of this [`Signal`], allowing it to be used in conjunction with [`Signal`]s
    /// of other concrete types.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let condition = true;
    /// let signal = if condition {
    ///     signal::from_system(|In(_)| 1)
    ///         .map_in(|x: i32| x * 2)
    ///         .boxed() // erases `Map<..., i32>`
    /// } else {
    ///     signal::from_system(|In(_)| 1).dedupe().boxed() // erases `Dedupe<...>`
    /// }; // without the `.boxed()`, the compiler would not allow this
    /// ```
    fn boxed(self) -> BoxedSignal<Self::Item>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Erases this signal's concrete type while preserving its verified replay capability.
    fn boxed_replayable(self) -> BoxedReplayableSignal<Self::Item>
    where
        Self: Sized + ReplayableSignal,
        Self::Item: Clone + Send + Sync + 'static,
    {
        BoxedReplayableSignal(Box::new(self))
    }

    /// Assign a schedule to this signal chain.
    ///
    /// When registered, signals in the chain will be tagged to run in the specified schedule,
    /// enabling control over when signal processing occurs within each frame. Signals without
    /// an explicit `.schedule()` call run in the [`JonmoPlugin`](crate::JonmoPlugin)'s default
    /// schedule ([`Last`](bevy_app::Last) unless configured otherwise).
    ///
    /// # Semantics
    ///
    /// `.schedule::<S>()` applies the schedule `S` to:
    /// 1. The signal it's called on (the "caller")
    /// 2. All upstream signals that don't already have a schedule
    /// 3. All downstream signals, until another `.schedule()` is encountered
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_app::prelude::*;
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// // All signals in this chain run in the Update schedule
    /// let signal = signal::from_system(|In(_)| 42i32)
    ///     .map_in(|x: i32| x + 1)
    ///     .schedule::<Update>();
    /// ```
    ///
    /// # Plugin Configuration
    ///
    /// Any schedule used with `.schedule()` must be registered with
    /// [`JonmoPlugin::with_schedule`](crate::JonmoPlugin::with_schedule).
    ///
    /// ```
    /// use bevy_app::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// # let mut app = App::new();
    /// app.add_plugins(
    ///     JonmoPlugin::new::<PostUpdate>() // signals run in PostUpdate by default
    ///         .with_schedule::<Update>(), // enables `.schedule::<Update>()`
    /// );
    /// ```
    ///
    /// # Multi-Schedule Chains
    ///
    /// Multiple `.schedule()` calls create schedule boundaries. Each `.schedule()` tags
    /// the signal it's called on (the "caller"):
    ///
    /// ```
    /// use bevy_app::prelude::*;
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal = signal::from_system(|In(_)| Some(1i32)) // Update (inherits from downstream)
    ///     .map_in(|x: i32| x + 1) // Update (caller of .schedule::<Update>())
    ///     .schedule::<Update>()
    ///     .map_in(|x: i32| x * 2) // PostUpdate (caller of .schedule::<PostUpdate>())
    ///     .schedule::<PostUpdate>()
    ///     .map_in(|x: i32| x - 1); // PostUpdate (inherits from upstream)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the schedule has not been registered with
    /// [`JonmoPlugin::with_schedule`](crate::JonmoPlugin::with_schedule).
    fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> Scheduled<Sched, Self::Item>
    where
        Self: Sized + 'static,
    {
        let upstream = Box::new(self).into_signal_materializer();
        let materializer = ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.apply_schedule(upstream, Sched::default())
        });
        Scheduled {
            materializer,
            _marker: PhantomData,
        }
    }

    /// Attempts to materialize this signal through its framework-owned registrar and acquire one
    /// claim.
    ///
    /// Complete allocation-free preflight returns a structured [`SignalRegistrationError`] on
    /// detectable rejection without mutating graph or Bevy allocation/materialization state.
    /// Successful commit has no ordinary structured failure path. This guarantee composes through
    /// static and dynamic built-ins and external custom implementations.
    fn try_register(self, world: &mut World) -> Result<SignalClaim, SignalRegistrationError>
    where
        Self: Sized,
    {
        Box::new(self).into_signal_materializer().try_register(world)
    }

    /// Activate this [`Signal`] and all its upstreams, causing them to be evaluated
    /// every frame until they are [`SignalClaim::release`]-ed, see [`SignalClaim`].
    fn register(self, world: &mut World) -> SignalClaim
    where
        Self: Sized,
    {
        Box::new(self).into_signal_materializer().register(world)
    }
}

impl<T: ?Sized> SignalExt for T where T: Signal {}

/// Creates a [`Signal`] that outputs `true` if all input [`Signal`]s are equal, and `false`
/// otherwise.
///
/// Accepts 2 or more [`Signal`]s.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 1);
/// let s2 = signal::from_system(|In(_)| 1);
/// let s3 = signal::from_system(|In(_)| 1);
///
/// let eq_signal = signal::eq!(s1, s2, s3); // outputs `true`
/// ```
#[macro_export]
macro_rules! eq {
    // Entry point
    ($s1:expr, $s2:expr $(, $rest:expr)* $(,)?) => {
        $crate::__signal_zip_and_map!($s1, $s2 $(, $rest)*; |val| {
            $crate::eq!(@check val, $s1, $s2 $(, $rest)*)
        })
    };

    // Check logic
    // Base case: 2 signals
    (@check $val:expr, $a:expr, $b:expr) => {
        $val.0 == $val.1
    };
    // Recursive case: 3+ signals
    (@check $val:expr, $head:expr, $($tail:expr),+) => {
        $val.1 == $val.0.1 && $crate::eq!(@check $val.0, $($tail),+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __signal_zip_and_map {
    // Single argument case
    ($s1:expr $(,)?; |$val:ident| $body:block) => {
        $s1.map(|In($val)| $body)
    };
    // Entry point
    ($s1:expr, $s2:expr $(, $rest:expr)* $(,)?; |$val:ident| $body:block) => {
        $crate::__signal_zip_and_map!(@combine $s1, $s2 $(, $rest)*)
            .map(|In($val @ $crate::__signal_zip_and_map!(@pattern $s1, $s2 $(, $rest)*))| $body)
    };

    // Combine chain
    (@combine $first:expr, $second:expr) => {
        $first.zip($second)
    };
    (@combine $first:expr, $second:expr, $($rest:expr),+) => {
        $crate::__signal_zip_and_map!(@combine $first.zip($second), $($rest),+)
    };

    // Pattern generator (nested tuple of `_` matching the combine nesting)
    (@pattern $s1:expr, $s2:expr $(, $rest:expr)*) => {
        $crate::__signal_zip_and_map!(@pattern_helper (_, _) $(, $rest)*)
    };
    (@pattern_helper $acc:tt $(,)?) => { $acc };
    (@pattern_helper $acc:tt, $head:expr $(, $tail:expr)*) => {
        $crate::__signal_zip_and_map!(@pattern_helper ($acc, _) $(, $tail)*)
    };
}

/// Creates a [`Signal`] that outputs `true` if all input [`Signal`]s are `true`.
///
/// Accepts 2 or more [`Signal`]s. All signals must output `bool`.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| true);
/// let s2 = signal::from_system(|In(_)| true);
///
/// let all_signal = signal::all!(s1, s2); // outputs `true`
/// ```
#[macro_export]
macro_rules! all {
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_reduce_binop!(&&; $($args),+)
    };
}

/// Creates a [`Signal`] that outputs `true` if any input [`Signal`] is `true`.
///
/// Accepts 2 or more [`Signal`]s. All signals must output `bool`.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| true);
/// let s2 = signal::from_system(|In(_)| false);
///
/// let any_signal = signal::any!(s1, s2); // outputs `true`
/// ```
#[macro_export]
macro_rules! any {
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_reduce_binop!(||; $($args),+)
    };
}

/// Creates a [`Signal`] that outputs `true` if all input [`Signal`]s are distinct (pairwise
/// unequal).
///
/// Accepts 2 or more [`Signal`]s.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 1);
/// let s2 = signal::from_system(|In(_)| 2);
/// let s3 = signal::from_system(|In(_)| 3);
///
/// let distinct_signal = signal::distinct!(s1, s2, s3); // outputs `true`
/// ```
#[macro_export]
macro_rules! distinct {
    ($s1:expr, $s2:expr $(, $rest:expr)* $(,)?) => {
        $crate::__signal_zip_and_map!($s1, $s2 $(, $rest)*; |val| {
            $crate::distinct!(@check val; $s1, $s2 $(, $rest)*)
        })
    };

    // Base case: 2 signals
    (@check $val:expr; $a:expr, $b:expr) => {
        $val.0 != $val.1
    };
    // Recursive case: 3+ signals
    (@check $val:expr; $head:expr, $($tail:expr),+) => {
        $crate::distinct!(@neq_all $val.1, $val.0; $($tail),+)
            && $crate::distinct!(@check $val.0; $($tail),+)
    };

    // Compare `$target` against every element in the left-nested tuple `$nested`
    (@neq_all $target:expr, $pair:expr; $a:expr, $b:expr) => {
        $target != $pair.1 && $target != $pair.0
    };
    (@neq_all $target:expr, $nested:expr; $head:expr, $($tail:expr),+) => {
        $target != $nested.1 && $crate::distinct!(@neq_all $target, $nested.0; $($tail),+)
    };
}

/// Creates a [`Signal`] that outputs the sum of all input [`Signal`]s.
///
/// Accepts 2 or more [`Signal`]s. All signals must output numeric types that implement `Add`.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 1);
/// let s2 = signal::from_system(|In(_)| 2);
/// let s3 = signal::from_system(|In(_)| 3);
///
/// let sum_signal = signal::sum!(s1, s2, s3); // outputs `6`
/// ```
#[macro_export]
macro_rules! sum {
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_reduce_binop!(+; $($args),+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __signal_reduce_binop {
    // Single argument case
    ($op:tt; $s1:expr $(,)?) => {
        $s1
    };
    // Entry point
    ($op:tt; $s1:expr, $s2:expr $(, $rest:expr)* $(,)?) => {
        $crate::__signal_zip_and_map!($s1, $s2 $(, $rest)*; |val| {
            $crate::__signal_reduce_binop!(@apply $op, val, $s1, $s2 $(, $rest)*)
        })
    };

    // Apply logic
    // Base case: 2 signals
    (@apply $op:tt, $val:expr, $a:expr, $b:expr) => {
        $val.0 $op $val.1
    };
    // Recursive case: 3+ signals
    (@apply $op:tt, $val:expr, $head:expr, $($tail:expr),+) => {
        $val.1 $op $crate::__signal_reduce_binop!(@apply $op, $val.0, $($tail),+)
    };
}

/// Creates a [`Signal`] that outputs the product of all input [`Signal`]s.
///
/// Accepts 2 or more [`Signal`]s. All signals must output numeric types that implement `Mul`.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 2);
/// let s2 = signal::from_system(|In(_)| 3);
/// let s3 = signal::from_system(|In(_)| 4);
///
/// let product_signal = signal::product!(s1, s2, s3); // outputs `24`
/// ```
#[macro_export]
macro_rules! product {
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_reduce_binop!(*; $($args),+)
    };
}

/// Creates a [`Signal`] that outputs the minimum of all input [`Signal`]s.
///
/// Accepts 2 or more [`Signal`]s. All signals must output types that implement
/// [`PartialOrd`].
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 5);
/// let s2 = signal::from_system(|In(_)| 2);
/// let s3 = signal::from_system(|In(_)| 7);
///
/// let min_signal = signal::min!(s1, s2, s3); // outputs `2`
/// ```
#[macro_export]
macro_rules! min {
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_reduce_cmp!(<; $($args),+)
    };
}

/// Creates a [`Signal`] that outputs the maximum of all input [`Signal`]s.
///
/// Accepts 2 or more [`Signal`]s. All signals must output types that implement
/// [`PartialOrd`].
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 5);
/// let s2 = signal::from_system(|In(_)| 2);
/// let s3 = signal::from_system(|In(_)| 7);
///
/// let max_signal = signal::max!(s1, s2, s3); // outputs `7`
/// ```
#[macro_export]
macro_rules! max {
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_reduce_cmp!(>; $($args),+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __signal_reduce_cmp {
    // Single argument case
    ($cmp:tt; $s1:expr $(,)?) => {
        $s1
    };
    // Entry point
    ($cmp:tt; $s1:expr, $s2:expr $(, $rest:expr)* $(,)?) => {
        $crate::__signal_zip_and_map!($s1, $s2 $(, $rest)*; |val| {
            $crate::__signal_reduce_cmp!(@select $cmp, val, $s1, $s2 $(, $rest)*)
        })
    };

    // Selection logic
    // Base case: 2 signals
    (@select $cmp:tt, $val:expr, $a:expr, $b:expr) => {
        if $val.0 $cmp $val.1 { $val.0 } else { $val.1 }
    };
    // Recursive case: 3+ signals
    (@select $cmp:tt, $val:expr, $head:expr, $($tail:expr),+) => {
        {
            let rhs = $crate::__signal_reduce_cmp!(@select $cmp, $val.0, $($tail),+);
            if $val.1 $cmp rhs { $val.1 } else { rhs }
        }
    };
}

/// Zips multiple [`Signal`]s into a single [`Signal`] that outputs a flat tuple of all their
/// outputs. Unlike chaining [`.zip()`](SignalExt::zip) calls which produces nested tuples
/// like `(((A, B), C), D)`, this macro produces flat tuples like `(A, B, C, D)`.
///
/// The resulting [`Signal`] will only output a value when all input [`Signal`]s have outputted a
/// value.
///
/// Accepts 1 or more [`Signal`]s.
///
/// # Example
///
/// ```
/// use bevy_ecs::prelude::*;
/// use jonmo::{prelude::*, signal};
///
/// let s1 = signal::from_system(|In(_)| 1);
/// let s2 = signal::from_system(|In(_)| "hello");
/// let s3 = signal::from_system(|In(_)| 3.14);
/// let s4 = signal::from_system(|In(_)| true);
///
/// // Outputs a flat tuple (i32, &str, f64, bool)
/// let zipped = signal::zip!(s1, s2, s3, s4);
///
/// // Compare to chained .zip() which would give (((i32, &str), f64), bool)
/// ```
#[macro_export]
macro_rules! zip {
    // Single signal case - just return it as-is wrapped in a 1-tuple for consistency
    ($s1:expr $(,)?) => {
        $s1.map(|In(__v)| (__v,))
    };
    // Two signals - use zip directly (already flat)
    ($s1:expr, $s2:expr $(,)?) => {
        $s1.zip($s2)
    };
    // Three or more signals - build combine chain then flatten
    ($s1:expr, $s2:expr $(, $rest:expr)+ $(,)?) => {
        $crate::__signal_zip_and_map!($s1, $s2 $(, $rest)+; |__v| {
            $crate::__signal_zip_flatten!(__v; $s1, $s2 $(, $rest)+;)
        })
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __signal_zip_flatten {
    // Entry point - start building from innermost (.0.0...) outward
    ($val:expr; $($signals:expr),+;) => {
        $crate::__signal_zip_flatten!(@collect $val; $($signals),+; ())
    };

    // Base case: 2 signals left - prepend val.0, val.1 to accumulator
    (@collect $val:expr; $s1:expr, $s2:expr; ($($acc:expr),*)) => {
        ($val.0, $val.1 $(, $acc)*)
    };

    // Recursive case: 3+ signals - add val.1 to accumulator, recurse with val.0
    (@collect $val:expr; $head:expr, $($tail:expr),+; ($($acc:expr),*)) => {
        $crate::__signal_zip_flatten!(@collect $val.0; $($tail),+; ($val.1 $(, $acc)*))
    };
}

pub use all;
pub use any;
pub use distinct;
pub use eq;
pub use max;
pub use min;
pub use product;
pub use sum;
pub use zip;

#[cfg(test)]
mod tests {
    use crate::{
        JonmoPlugin,
        graph::{
            DirectOwnershipModel, GraphNodeSnapshot, SignalClaimCount, graph_snapshot,
            test_has_signal_lifetime_retention,
        },
        prelude::{IntoSignalVecEither, SignalMapExt, SignalVecExt, clone},
        replay::{ReplayCacheSnapshot, ScalarReplayCache},
        signal::{self, BoxedSignal, IntoSignalEither, ReplayableSignal, SignalExt},
        signal_map::MutableBTreeMap,
        signal_vec::{MutableVec, VecDiff},
        utils::LazyEntity,
    };
    use core::{convert::identity, fmt};

    // Import Bevy prelude for MinimalPlugins and other common items
    use bevy::prelude::*;
    use bevy_platform::sync::*;
    use bevy_time::TimePlugin;

    // Add Duration
    use core::time::Duration;
    use test_log::test;

    // Helper component and resource for testing Add Default
    #[derive(Component, Clone, Debug, PartialEq, Reflect, Default, Resource)]
    struct TestData(i32);

    #[derive(Resource, Default, Debug)]
    struct SignalOutput<T: Send + Sync + 'static + Clone + fmt::Debug>(Option<T>);

    #[derive(Resource, Default, Debug)]
    struct SignalOutputVec<T: Send + Sync + 'static + Clone + fmt::Debug>(Vec<T>);

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::default()));
        app.register_type::<TestData>();
        app
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct RegistrationObservation {
        exists: bool,
        count: Option<usize>,
        static_dependencies: Vec<Entity>,
        active_downstreams: Vec<Entity>,
        pending_input_count: Option<usize>,
    }

    impl RegistrationObservation {
        fn from_node(node: &GraphNodeSnapshot) -> Self {
            Self {
                exists: node.exists,
                count: node.claim_count,
                static_dependencies: node.canonical_static_dependencies.clone().unwrap_or_default(),
                active_downstreams: node.active_downstreams.clone(),
                pending_input_count: node.pending_input_count,
            }
        }

        fn absent() -> Self {
            Self {
                exists: false,
                count: None,
                static_dependencies: Vec::new(),
                active_downstreams: Vec::new(),
                pending_input_count: None,
            }
        }
    }

    fn registration_observation(world: &mut World, entity: Entity) -> RegistrationObservation {
        graph_snapshot(world)
            .node(entity)
            .map(RegistrationObservation::from_node)
            .unwrap_or_else(RegistrationObservation::absent)
    }

    fn single_static_dependency(world: &mut World, entity: Entity) -> Entity {
        let snapshot = graph_snapshot(world);
        let node = snapshot
            .node(entity)
            .expect("registered signal should be present in the graph snapshot");
        let dependencies = node
            .canonical_static_dependencies
            .as_ref()
            .expect("registered signal should have canonical static dependencies");
        assert_eq!(
            dependencies.len(),
            1,
            "expected one canonical static dependency for {entity:?}; snapshot={snapshot:#?}"
        );
        dependencies[0]
    }

    // Helper system to capture signal output
    fn capture_output<T: Send + Sync + 'static + Clone + fmt::Debug>(
        In(value): In<T>,
        mut output: ResMut<SignalOutput<T>>,
    ) {
        #[cfg(feature = "tracing")]
        bevy_log::debug!(
            "Capture Output System: Received {:?}, updating resource from {:?} to Some({:?})",
            value,
            output.0,
            value
        );
        output.0 = Some(value);
    }

    fn get_output<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &World) -> Option<T> {
        world.resource::<SignalOutput<T>>().0.clone()
    }

    fn capture_output_vec<T: Send + Sync + 'static + Clone + fmt::Debug>(
        In(value): In<T>,
        mut output: ResMut<SignalOutputVec<T>>,
    ) {
        output.0.push(value);
    }

    fn get_and_clear_output_vec<T: Send + Sync + 'static + Clone + fmt::Debug>(world: &mut World) -> Vec<T> {
        world
            .get_resource_mut::<SignalOutputVec<T>>()
            .map(|mut res| core::mem::take(&mut res.0))
            .unwrap_or_default()
    }

    #[test]
    fn test_map() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal = signal::from_system(|In(_)| 1)
            .map(|In(x): In<i32>| x + 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_map_in() {
        // === Part 1: Basic Infallible Mapping ===
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();
        let signal = signal::from_system(|In(_)| 10i32)
            .map_in(|x: i32| format!("Value: {x}"))
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("Value: 10".to_string()),
            "Infallible mapping from i32 to String failed."
        );
        signal.release(app.world_mut());

        // Reset for next test part
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // === Part 2: Fallible Mapping (Some/None) ===
        app.init_resource::<SignalOutput<i32>>();
        // Note: .pop() removes from the end, so the sequence is 4, 3, 2, 1
        let inputs = Arc::new(Mutex::new(vec![1, 2, 3, 4]));
        let source_signal = signal::from_system(clone!((inputs) move |In(_)| {
            inputs.lock().unwrap().pop()
        }));

        let signal = source_signal
            .map_in(|x: i32| if x % 2 == 0 { Some(x * 10) } else { None })
            .map(capture_output::<i32>)
            .register(app.world_mut());

        // Input: 4 (even) -> Some(40)
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(40), "Failed on first valid item.");

        // Input: 3 (odd) -> None. Output should remain 40.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(40),
            "Output should not change when closure returns None."
        );

        // Input: 2 (even) -> Some(20)
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(20), "Failed on second valid item.");

        // Input: 1 (odd) -> None. Output should remain 20.
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            Some(20),
            "Output should not change on subsequent None."
        );
        signal.release(app.world_mut());

        // === Part 3: FnMut Stateful Closure ===
        app.init_resource::<SignalOutput<()>>(); // We don't care about the value, just that it runs
        let call_count = Arc::new(Mutex::new(0));
        let signal = signal::from_system(|In(_)| ())
            .map_in(clone!((call_count) move |_: ()| {
                *call_count.lock().unwrap() += 1;
            }))
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        app.update();
        app.update();

        assert_eq!(
            *call_count.lock().unwrap(),
            3,
            "FnMut closure should be called on each update"
        );
        signal.release(app.world_mut());

        // === Part 4: Cleanup ===
        app.init_resource::<SignalOutput<i32>>();
        let signal = signal::from_system(|In(_)| 123)
            .map_in(|x| x)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(123));

        // Cleanup and reset
        signal.release(app.world_mut());
        app.world_mut().resource_mut::<SignalOutput<i32>>().0 = None;

        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            None,
            "Signal should not fire after cleanup"
        );
    }

    #[test]
    fn test_map_in_ref() {
        // === Part 1: Basic Infallible Mapping ===
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();
        let signal = signal::from_system(|In(_)| 42i32)
            // Use a function that takes a reference, the primary use case for map_in_ref.
            .map_in_ref(ToString::to_string)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("42".to_string()),
            "Infallible mapping by reference failed."
        );
        signal.release(app.world_mut());

        // Reset for next test part
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // === Part 2: Fallible Mapping (Some/None) ===
        // Note: .pop() removes from the end, so the sequence is "FOUR", "three", "TWO", "one"
        let inputs = Arc::new(Mutex::new(vec!["one", "TWO", "three", "FOUR"]));
        let source_signal = signal::from_system(clone!((inputs) move |In(_)| {
            inputs.lock().unwrap().pop()
        }));

        let signal = source_signal
            // The closure takes a reference to the item. Since the item is &'static str,
            // the closure gets &&'static str.
            .map_in_ref(|s: &&'static str| {
                if s.chars().all(char::is_uppercase) {
                    Some(s.to_lowercase())
                } else {
                    None
                }
            })
            .map(capture_output::<String>)
            .register(app.world_mut());

        // Input: "FOUR" (uppercase) -> Some("four")
        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("four".to_string()),
            "Failed on first valid item."
        );

        // Input: "three" (lowercase) -> None. Output should remain "four".
        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("four".to_string()),
            "Output should not change when closure returns None."
        );

        // Input: "TWO" (uppercase) -> Some("two")
        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("two".to_string()),
            "Failed on second valid item."
        );

        // Input: "one" (lowercase) -> None. Output should remain "two".
        app.update();
        assert_eq!(
            get_output::<String>(app.world()),
            Some("two".to_string()),
            "Output should not change on subsequent None."
        );
        signal.release(app.world_mut());

        // === Part 3: FnMut Stateful Closure ===
        app.init_resource::<SignalOutput<()>>();
        let call_count = Arc::new(Mutex::new(0));
        let signal = signal::from_system(|In(_)| 1i32)
            .map_in_ref(clone!((call_count) move |_: &i32| {
                *call_count.lock().unwrap() += 1;
            }))
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        app.update();
        app.update();

        assert_eq!(
            *call_count.lock().unwrap(),
            3,
            "FnMut closure should be called on each update"
        );
        signal.release(app.world_mut());

        // === Part 4: Cleanup ===
        app.init_resource::<SignalOutput<i32>>();
        let signal = signal::from_system(|In(_)| 123)
            // Simple dereference to pass the value through
            .map_in_ref(|x: &i32| *x)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(123));

        // Cleanup and reset
        signal.release(app.world_mut());
        app.world_mut().resource_mut::<SignalOutput<i32>>().0 = None;

        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            None,
            "Signal should not fire after cleanup"
        );
    }

    #[test]
    fn test_component() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();
        let entity = app.world_mut().spawn(TestData(1)).id();
        let signal = signal::from_entity(entity)
            .component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(1)));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_component_option() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<Option<TestData>>::default());
        let entity_with = app.world_mut().spawn(TestData(1)).id();
        let entity_without = app.world_mut().spawn_empty().id();
        let signal = signal::from_entity(entity_with)
            .component_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(1))));
        signal.release(app.world_mut());
        let signal = signal::from_entity(entity_without)
            .component_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(None));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_has_component() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());
        let entity_with = app.world_mut().spawn(TestData(1)).id();
        let entity_without = app.world_mut().spawn_empty().id();
        let signal = signal::from_entity(entity_with)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.release(app.world_mut());
        let signal = signal::from_entity(entity_without)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_component_changed() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();
        let entity = app.world_mut().spawn(TestData(1)).id();
        let signal = signal::from_entity(entity)
            .component_changed::<TestData>()
            .map(capture_output)
            .register(app.world_mut());

        // First update: component is newly added, counts as changed
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(1)));

        // Reset output to verify signal doesn't fire on unchanged frame
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;

        // Second update: no change, signal should not fire
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Signal should not fire when component hasn't changed"
        );

        // Third update: mutate component, signal should fire
        app.world_mut().get_mut::<TestData>(entity).unwrap().0 = 2;
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(2)));

        signal.release(app.world_mut());

        // Test entity without component - signal should not fire
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        let entity_without = app.world_mut().spawn_empty().id();
        let signal = signal::from_entity(entity_without)
            .component_changed::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Signal should not fire for entity without component"
        );
        signal.release(app.world_mut());
    }

    #[test]
    fn test_dedupe() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let values = Arc::new(Mutex::new(vec![1, 1, 2, 3, 3, 3, 4]));
        let signal = signal::from_system(clone!((values) move |In(_)| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        }))
        .dedupe()
        .map(clone!((counter) move |In(val): In<i32>| {
            *counter.lock().unwrap() += 1;
            val
        }))
        .map(capture_output)
        .register(app.world_mut());
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(get_output::<i32>(app.world()), Some(4));
        assert_eq!(*counter.lock().unwrap(), 4);
        assert_eq!(values.lock().unwrap().len(), 0);
        signal.release(app.world_mut());
    }

    #[test]
    fn dedupe_is_intrinsically_replayable_and_normalizes_cache_identity() {
        fn assert_replayable<Signal: ReplayableSignal>(_signal: &Signal) {}

        let mut app = create_test_app();
        app.insert_resource(TestData(1));
        let signal = signal::from_resource::<TestData>()
            .map_in(|value| value.0)
            .dedupe()
            .broadcast();
        assert_replayable(&signal);

        let direct = signal.clone().register(app.world_mut());
        let terminal = direct.entity();
        assert!(matches!(
            app.world()
                .get::<ScalarReplayCache<i32>>(*terminal)
                .expect("dedupe terminal should own its intrinsic replay cache")
                .snapshot(),
            ReplayCacheSnapshot::Warming
        ));

        let first_values = Arc::new(Mutex::new(Vec::new()));
        let first = signal
            .clone()
            .map({
                let first_values = first_values.clone();
                move |In(value)| first_values.lock().unwrap().push(value)
            })
            .register(app.world_mut());
        app.update();
        assert_eq!(*first_values.lock().unwrap(), vec![1]);
        let first_revision = match app
            .world()
            .get::<ScalarReplayCache<i32>>(*terminal)
            .expect("dedupe terminal should retain its replay cache")
            .snapshot()
        {
            ReplayCacheSnapshot::Ready { revision, .. } => revision,
            ReplayCacheSnapshot::Warming => panic!("dedupe should become ready after its first accepted output"),
        };

        app.update();
        assert_eq!(*first_values.lock().unwrap(), vec![1]);
        assert!(matches!(
            app.world()
                .get::<ScalarReplayCache<i32>>(*terminal)
                .expect("dedupe terminal should retain its replay cache")
                .snapshot(),
            ReplayCacheSnapshot::Ready { revision, .. } if revision == first_revision
        ));

        app.world_mut().resource_mut::<TestData>().0 = 2;
        app.update();
        assert_eq!(*first_values.lock().unwrap(), vec![1, 2]);

        let entities_before_cache = app.world().entities().len();
        let normalized = signal.clone().cache().register(app.world_mut());
        assert_eq!(normalized.entity(), terminal);
        assert_eq!(app.world().entities().len(), entities_before_cache);

        let second_values = Arc::new(Mutex::new(Vec::new()));
        let second = signal
            .clone()
            .map({
                let second_values = second_values.clone();
                move |In(value)| second_values.lock().unwrap().push(value)
            })
            .register(app.world_mut());
        app.update();
        assert_eq!(*first_values.lock().unwrap(), vec![1, 2]);
        assert_eq!(*second_values.lock().unwrap(), vec![2]);

        app.world_mut().resource_mut::<TestData>().0 = 3;
        app.update();
        assert_eq!(*first_values.lock().unwrap(), vec![1, 2, 3]);
        assert_eq!(*second_values.lock().unwrap(), vec![2, 3]);

        normalized.release(app.world_mut());
        second.release(app.world_mut());
        first.release(app.world_mut());
        direct.release(app.world_mut());
        assert_eq!(
            app.world().get::<SignalClaimCount>(*terminal).map(|count| **count),
            Some(1),
            "a retained intrinsic replay terminal should keep exactly one maintenance claim"
        );

        drop(signal);
        app.update();
        assert!(app.world().get_entity(*terminal).is_err());
    }

    #[test]
    fn test_first() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let values = Arc::new(Mutex::new(vec![10, 20, 30]));
        let signal = signal::from_system(clone!((values) move |In(_)| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        }))
        .first()
        .map(clone!((counter) move |In(val): In<i32>| {
            *counter.lock().unwrap() += 1;
            val
        }))
        .map(capture_output)
        .register(app.world_mut());
        app.update();
        app.update();
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(10));
        assert_eq!(*counter.lock().unwrap(), 1);
        assert_eq!(values.lock().unwrap().len(), 0);
        signal.release(app.world_mut());
    }

    #[test]
    fn test_once() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let signal = signal::once(42)
            .map(clone!((counter) move |In(val): In<i32>| {
                *counter.lock().unwrap() += 1;
                val
            }))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(42));
        assert_eq!(*counter.lock().unwrap(), 1);
        app.update();
        app.update();
        // Should still be 42 and counter should still be 1 (no further emissions)
        assert_eq!(get_output::<i32>(app.world()), Some(42));
        assert_eq!(*counter.lock().unwrap(), 1);
        signal.release(app.world_mut());
    }

    #[test]
    fn test_take() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let values = Arc::new(Mutex::new(vec![10, 20, 30]));
        let signal = signal::from_system(clone!((values) move |In(_)| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        }))
        .take(2)
        .map(clone!((counter) move |In(val): In<i32>| {
            *counter.lock().unwrap() += 1;
            val
        }))
        .map(capture_output)
        .register(app.world_mut());
        app.update();
        app.update();
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(20));
        assert_eq!(*counter.lock().unwrap(), 2);
        assert_eq!(values.lock().unwrap().len(), 0);
        signal.release(app.world_mut());
    }

    #[test]
    fn test_skip() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let values = Arc::new(Mutex::new(vec![10, 20, 30, 40, 50]));
        let signal = signal::from_system(clone!((values) move |In(_)| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        }))
        .skip(2) // Skip first 2 values (10, 20), then output rest
        .map(clone!((counter) move |In(val): In<i32>| {
            *counter.lock().unwrap() += 1;
            val
        }))
        .map(capture_output)
        .register(app.world_mut());

        // First two updates: values 10, 20 are skipped
        app.update();
        assert_eq!(get_output::<i32>(app.world()), None);
        app.update();
        assert_eq!(get_output::<i32>(app.world()), None);

        // Third update: value 30 is output
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(30));

        // Fourth update: value 40 is output
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(40));

        // Fifth update: value 50 is output
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(50));

        // Counter should be 3 (only values after skip were processed by the map)
        assert_eq!(*counter.lock().unwrap(), 3);
        assert_eq!(values.lock().unwrap().len(), 0);
        signal.release(app.world_mut());
    }

    #[test]
    fn test_zip() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<(i32, &'static str)>>();
        let signal = signal::from_system(move |In(_)| 10)
            .zip(signal::from_system(move |In(_)| "hello"))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<(i32, &'static str)>(app.world()), Some((10, "hello")));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_zip_emits_after_both_emit() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutputVec<(i32, i32)>>();

        let left = signal::from_system(|In(_), mut state: Local<i32>| {
            *state += 1;
            *state
        });
        let right = signal::from_system(|In(_)| 10).first();

        let signal = left.zip(right).map(capture_output_vec).register(app.world_mut());

        app.update();
        // With proper FRP semantics, each signal runs at most once per cycle.
        // Zip uses a cache to accumulate inputs, then runs once to read the combined state.
        assert_eq!(get_and_clear_output_vec::<(i32, i32)>(app.world_mut()), vec![(1, 10)]);

        app.update();
        assert_eq!(get_and_clear_output_vec::<(i32, i32)>(app.world_mut()), vec![(2, 10)]);

        app.update();
        assert_eq!(get_and_clear_output_vec::<(i32, i32)>(app.world_mut()), vec![(3, 10)]);

        signal.release(app.world_mut());
    }

    #[test]
    fn test_chain_multi_level_same_frame() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let signal = signal::from_system(|In(_), mut state: Local<i32>| {
            *state += 1;
            *state
        })
        .map_in(|x: i32| x + 1)
        .map_in(|x: i32| x * 2)
        .map(capture_output)
        .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(4));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_fan_out_fan_in_zip() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<(i32, i32)>>();

        let source = signal::from_system(|In(_), mut state: Local<i32>| {
            *state += 1;
            *state
        })
        .broadcast();

        let left = source.clone().map_in(|x: i32| x + 1);
        let right = source.map_in(|x: i32| x + 10);

        let signal = left.zip(right).map(capture_output).register(app.world_mut());

        app.update();
        assert_eq!(get_output::<(i32, i32)>(app.world()), Some((2, 11)));

        app.update();
        assert_eq!(get_output::<(i32, i32)>(app.world()), Some((3, 12)));

        signal.release(app.world_mut());
    }

    #[test]
    fn test_zip_macro() {
        let mut app = create_test_app();

        // Test 1 signal - returns 1-tuple
        app.init_resource::<SignalOutput<(i32,)>>();
        let s1 = signal::from_system(|In(_)| 1);
        let signal = zip!(s1).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<(i32,)>(app.world()), Some((1,)));
        signal.release(app.world_mut());

        // Test 2 signals - same as .zip()
        app.init_resource::<SignalOutput<(i32, &'static str)>>();
        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| "two");
        let signal = zip!(s1, s2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<(i32, &'static str)>(app.world()), Some((1, "two")));
        signal.release(app.world_mut());

        // Test 3 signals - flat tuple (not nested)
        app.init_resource::<SignalOutput<(i32, &'static str, f64)>>();
        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| "two");
        let s3 = signal::from_system(|In(_)| 3.0);
        let signal = zip!(s1, s2, s3).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<(i32, &'static str, f64)>(app.world()),
            Some((1, "two", 3.0))
        );
        signal.release(app.world_mut());

        // Test 4 signals - verifies flattening works for more signals
        app.init_resource::<SignalOutput<(i32, &'static str, f64, bool)>>();
        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| "two");
        let s3 = signal::from_system(|In(_)| 3.0);
        let s4 = signal::from_system(|In(_)| true);
        let signal = zip!(s1, s2, s3, s4).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<(i32, &'static str, f64, bool)>(app.world()),
            Some((1, "two", 3.0, true))
        );
        signal.release(app.world_mut());

        // Test 5 signals - ensures unlimited recursion works
        app.init_resource::<SignalOutput<(i32, i32, i32, i32, i32)>>();
        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 3);
        let s4 = signal::from_system(|In(_)| 4);
        let s5 = signal::from_system(|In(_)| 5);
        let signal = zip!(s1, s2, s3, s4, s5).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<(i32, i32, i32, i32, i32)>(app.world()),
            Some((1, 2, 3, 4, 5))
        );
        signal.release(app.world_mut());
    }

    #[test]
    fn test_flatten() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal_1 = signal::from_system(|In(_)| 1).dedupe().broadcast();
        let signal_2 = signal::from_system(|In(_)| 2).dedupe().broadcast();

        #[derive(Resource, Default)]
        struct SignalSelector(bool);

        app.init_resource::<SignalSelector>();
        let signal = signal::from_system(
            move |In(_), selector: Res<SignalSelector>| {
                if selector.0 { signal_1.clone() } else { signal_2.clone() }
            },
        )
        .flatten()
        .map(capture_output)
        .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));
        app.world_mut().resource_mut::<SignalSelector>().0 = true;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_eq() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();
        let source = signal::from_system(|In(_)| 1).broadcast();
        let signal = source.clone().eq(1).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.release(app.world_mut());
        let signal = source.eq(2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_neq() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();
        let source = signal::from_system(|In(_)| 1).broadcast();
        let signal = source.clone().neq(2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.release(app.world_mut());
        let signal = source.neq(1).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_not() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();
        let signal = signal::from_system(|In(_)| true)
            .not()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.release(app.world_mut());
        let signal = signal::from_system(|In(_)| false)
            .not()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_filter() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let values = Arc::new(Mutex::new(vec![1, 2, 3, 4, 5, 6]));
        let signal = signal::from_system(move |In(_)| {
            let mut values_lock = values.lock().unwrap();
            if values_lock.is_empty() {
                None
            } else {
                Some(values_lock.remove(0))
            }
        })
        .filter(|In(x): In<i32>| x % 2 == 0)
        .map(capture_output)
        .register(app.world_mut());
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(get_output::<i32>(app.world()), Some(6));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_switch() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal_1 = signal::from_system(|In(_)| 1).dedupe().broadcast();
        let signal_2 = signal::from_system(|In(_)| 2).dedupe().broadcast();

        #[derive(Resource, Default)]
        struct SwitcherToggle(bool);

        app.init_resource::<SwitcherToggle>();
        let signal = signal::from_system(move |In(_), mut toggle: ResMut<SwitcherToggle>| {
            let current = toggle.0;
            toggle.0 = !toggle.0;
            current
        })
        .switch(
            move |In(use_1): In<bool>| {
                if use_1 { signal_1.clone() } else { signal_2.clone() }
            },
        )
        .map(capture_output)
        .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1));
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));
        signal.release(app.world_mut());
    }

    // Ensure these helpers are also present in the tests module.
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

    #[test]
    fn test_switch_signal_vec() {
        {
            // --- Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // Two different data sources to switch between.
            let list_a = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
            let list_b = MutableVec::builder().values([100, 200, 300]).spawn(app.world_mut());
            let signal_a = list_a.signal_vec().map_in(identity).cache().broadcast();
            let signal_b = list_b.signal_vec().broadcast();

            // A resource to control which list is active.
            #[derive(Resource, Clone, Copy, PartialEq, Debug)]
            enum ListSelector {
                A,
                B,
            }

            app.insert_resource(ListSelector::A);

            // The control signal reads the resource.
            let control_signal = signal::from_system(|In(_), selector: Res<ListSelector>| *selector).dedupe();

            // The signal chain under test.
            let switched_signal = control_signal.dedupe().switch_signal_vec(
                clone!((/* list_a, list_b, */ signal_a, signal_b) move |In(selector): In<ListSelector>| match selector {
                    // can use .left/right_either or .boxed_replayable
                    ListSelector:: A => signal_a.clone().left_either(),
                    ListSelector:: B => signal_b.clone().right_either(),
                    // TODO: this is the more conservative case, so it should be covered by the above case of signal reuse, but ideally they would both be explicitly tested
                    // ListSelector:: A => list_a.signal_vec().map_in(identity).cache().boxed_replayable(),
                    // ListSelector:: B => list_b.signal_vec().boxed_replayable(),
                }),
            );

            // Register the final signal to a capture system.
            let handle = switched_signal.for_each(capture_vec_output).register(app.world_mut());

            // --- Test 1: Initial State --- The first update should select List A and emit
            // its initial state.
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace { values: vec![10, 20] },
                "Initial state should be a Replace with List A's contents."
            );

            // --- Test 2: Forwarding Diffs from Active List (A) --- A mutation to List A
            // should be forwarded.
            list_a.write(app.world_mut()).push(30);
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Push to active list should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Push { value: 30 },
                "Should forward Push diff from List A."
            );

            // --- Test 3: The Switch to List B --- Change the selector and update. This
            // should trigger a switch.
            *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching lists should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec![100, 200, 300]
                },
                "Switch should emit a Replace with List B's contents."
            );

            // --- Test 4: Ignoring Diffs from Old List (A) --- A mutation to the now-inactive
            // List A should be ignored. This should not be seen
            list_a.write(app.world_mut()).push(99);
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert!(diffs.is_empty(), "Should ignore diffs from the old, inactive list.");

            // --- Test 5: Forwarding Diffs from New Active List (B) --- A mutation to List B
            // should now be forwarded. remove 100
            list_b.write(app.world_mut()).remove(0);
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "RemoveAt from new active list should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::RemoveAt { index: 0 },
                "Should forward RemoveAt diff from List B."
            );

            // --- Test 6: Memoization (No-Op Switch) --- "Switching" to the already active
            // list should produce no diffs.
            *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert!(
                diffs.is_empty(),
                "Re-selecting the same list should produce no diffs due to memoization."
            );

            // --- Test 7: Switch Back to List A --- Switching back should give us the current
            // state of List A.
            *app.world_mut().resource_mut::<ListSelector>() = ListSelector::A;
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching back to A should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec![10, 20, 30, 99],
                    // Note: includes the `push(30)` from earlier
                },
                "Switching back should Replace with the current state of List A."
            );
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_switch_signal_vec_initially_empty() {
        {
            // --- Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalVecOutput<i32>>();

            // Start with an EMPTY list - this tests the `was_initially_empty` code path
            let list_a: MutableVec<i32> = app.world_mut().into();
            let list_b = MutableVec::builder().values([100, 200, 300]).spawn(app.world_mut());
            let signal_a = list_a.signal_vec().map_in(identity).cache().broadcast();
            let signal_b = list_b.signal_vec().broadcast();

            // A resource to control which list is active.
            #[derive(Resource, Clone, Copy, PartialEq, Debug)]
            enum ListSelector {
                A,
                B,
            }

            app.insert_resource(ListSelector::A);

            // The control signal reads the resource.
            let control_signal = signal::from_system(|In(_), selector: Res<ListSelector>| *selector).dedupe();

            // The signal chain under test.
            let switched_signal = control_signal.dedupe().switch_signal_vec(
                clone!((signal_a, signal_b) move |In(selector): In<ListSelector>| match selector {
                    ListSelector::A => signal_a.clone().left_either(),
                    ListSelector::B => signal_b.clone().right_either(),
                }),
            );

            // Register the final signal to a capture system.
            let handle = switched_signal.for_each(capture_vec_output).register(app.world_mut());

            // --- Test 1: Initial State with Empty List ---
            // The first update should select List A and authoritatively replay its empty state.
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![VecDiff::Replace { values: vec![] }],
                "Initial update with empty list should produce an authoritative empty Replace"
            );

            // --- Test 2: Push to Empty Active List ---
            // A push to the initially empty List A should be forwarded as a Push.
            list_a.write(app.world_mut()).push(10);
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Push to active list should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Push { value: 10 },
                "Should forward Push diff from List A."
            );

            // --- Test 3: Another Push ---
            list_a.write(app.world_mut()).push(20);
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Second push should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Push { value: 20 },
                "Should forward second Push diff from List A."
            );

            // --- Test 4: Switch to List B (non-empty) ---
            *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching to non-empty list should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace {
                    values: vec![100, 200, 300]
                },
                "Switch should emit a Replace with List B's contents."
            );

            // --- Test 5: Switch Back to List A (now non-empty) ---
            // When switching back, List A now has items, so it should emit Replace.
            *app.world_mut().resource_mut::<ListSelector>() = ListSelector::A;
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching back to A should produce one diff.");
            assert_eq!(
                diffs[0],
                VecDiff::Replace { values: vec![10, 20] },
                "Switching back should Replace with the current state of List A."
            );

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_switch_signal_map() {
        use crate::signal_map::{IntoSignalMapEither, MapDiff, MutableBTreeMap, SignalMapExt};

        #[derive(Resource, Default, core::fmt::Debug)]
        struct SignalMapOutput<K, V>(Vec<MapDiff<K, V>>)
        where
            K: Send + Sync + 'static + Clone + core::fmt::Debug,
            V: Send + Sync + 'static + Clone + core::fmt::Debug;

        fn capture_map_output<K, V>(In(diffs): In<Vec<MapDiff<K, V>>>, mut output: ResMut<SignalMapOutput<K, V>>)
        where
            K: Send + Sync + 'static + Clone + core::fmt::Debug,
            V: Send + Sync + 'static + Clone + core::fmt::Debug,
        {
            output.0.extend(diffs);
        }

        fn get_and_clear_map_output<K, V>(world: &mut World) -> Vec<MapDiff<K, V>>
        where
            K: Send + Sync + 'static + Clone + core::fmt::Debug,
            V: Send + Sync + 'static + Clone + core::fmt::Debug,
        {
            world
                .get_resource_mut::<SignalMapOutput<K, V>>()
                .map(|mut res| core::mem::take(&mut res.0))
                .unwrap_or_default()
        }

        {
            // --- Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalMapOutput<i32, i32>>();

            // Two different data sources to switch between.
            let map_a = MutableBTreeMap::builder()
                .values([(1, 10), (2, 20)])
                .spawn(app.world_mut());
            let map_b = MutableBTreeMap::builder()
                .values([(1, 100), (2, 200), (3, 300)])
                .spawn(app.world_mut());
            let signal_a = map_a.signal_map().map_value(|In(x): In<i32>| x).cache().broadcast();
            let signal_b = map_b.signal_map().broadcast();

            // A resource to control which map is active.
            #[derive(Resource, Clone, Copy, PartialEq, Debug)]
            enum MapSelector {
                A,
                B,
            }

            app.insert_resource(MapSelector::A);

            // The control signal reads the resource.
            let control_signal = signal::from_system(|In(_), selector: Res<MapSelector>| *selector).dedupe();

            // The signal chain under test.
            let switched_signal = control_signal.dedupe().switch_signal_map(
                clone!((signal_a, signal_b) move |In(selector): In<MapSelector>| match selector {
                    // can use .left/right_either or .boxed_replayable
                    MapSelector::A => signal_a.clone().left_either(),
                    MapSelector::B => signal_b.clone().right_either(),
                }),
            );

            // Register the final signal to a capture system.
            let handle = switched_signal.for_each(capture_map_output).register(app.world_mut());

            // --- Test 1: Initial State --- The first update should select Map A and emit
            // its initial state.
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Initial update should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Replace { entries } if entries == &vec![(1, 10), (2, 20)]),
                "Initial state should be a Replace with Map A's contents."
            );

            // --- Test 2: Forwarding Diffs from Active Map (A) --- A mutation to Map A
            // should be forwarded.
            map_a.write(app.world_mut()).insert(3, 30);
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Insert to active map should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Insert { key: 3, value: 30 }),
                "Should forward Insert diff from Map A."
            );

            // --- Test 3: The Switch to Map B --- Change the selector and update. This
            // should trigger a switch.
            *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching maps should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Replace { entries } if entries == &vec![(1, 100), (2, 200), (3, 300)]),
                "Switch should emit a Replace with Map B's contents."
            );

            // --- Test 4: Ignoring Diffs from Old Map (A) --- A mutation to the now-inactive
            // Map A should be ignored.
            map_a.write(app.world_mut()).insert(4, 99);
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert!(diffs.is_empty(), "Should ignore diffs from the old, inactive map.");

            // --- Test 5: Forwarding Diffs from New Active Map (B) --- A mutation to Map B
            // should now be forwarded. remove key 1
            map_b.write(app.world_mut()).remove(&1);
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Remove from new active map should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Remove { key: 1 }),
                "Should forward Remove diff from Map B."
            );

            // --- Test 6: Memoization (No-Op Switch) --- "Switching" to the already active
            // map should produce no diffs.
            *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert!(
                diffs.is_empty(),
                "Re-selecting the same map should produce no diffs due to memoization."
            );

            // --- Test 7: Switch Back to Map A --- Switching back should give us the current
            // state of Map A.
            *app.world_mut().resource_mut::<MapSelector>() = MapSelector::A;
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching back to A should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Replace { entries } if entries == &vec![(1, 10), (2, 20), (3, 30), (4, 99)]),
                "Switching back should Replace with the current state of Map A."
            );
            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_switch_signal_map_initially_empty() {
        use crate::signal_map::{IntoSignalMapEither, MapDiff, MutableBTreeMap, SignalMapExt};

        #[derive(Resource, Default, core::fmt::Debug)]
        struct SignalMapOutput<K, V>(Vec<MapDiff<K, V>>)
        where
            K: Send + Sync + 'static + Clone + core::fmt::Debug,
            V: Send + Sync + 'static + Clone + core::fmt::Debug;

        fn capture_map_output<K, V>(In(diffs): In<Vec<MapDiff<K, V>>>, mut output: ResMut<SignalMapOutput<K, V>>)
        where
            K: Send + Sync + 'static + Clone + core::fmt::Debug,
            V: Send + Sync + 'static + Clone + core::fmt::Debug,
        {
            output.0.extend(diffs);
        }

        fn get_and_clear_map_output<K, V>(world: &mut World) -> Vec<MapDiff<K, V>>
        where
            K: Send + Sync + 'static + Clone + core::fmt::Debug,
            V: Send + Sync + 'static + Clone + core::fmt::Debug,
        {
            world
                .get_resource_mut::<SignalMapOutput<K, V>>()
                .map(|mut res| core::mem::take(&mut res.0))
                .unwrap_or_default()
        }

        {
            // --- Setup ---
            let mut app = create_test_app();
            app.init_resource::<SignalMapOutput<i32, i32>>();

            // Start with an EMPTY map - this tests the `was_initially_empty` code path
            let map_a: MutableBTreeMap<i32, i32> = app.world_mut().into();
            let map_b = MutableBTreeMap::builder()
                .values([(1, 100), (2, 200), (3, 300)])
                .spawn(app.world_mut());
            let signal_a = map_a.signal_map().map_value(|In(x): In<i32>| x).cache().broadcast();
            let signal_b = map_b.signal_map().broadcast();

            // A resource to control which map is active.
            #[derive(Resource, Clone, Copy, PartialEq, Debug)]
            enum MapSelector {
                A,
                B,
            }

            app.insert_resource(MapSelector::A);

            // The control signal reads the resource.
            let control_signal = signal::from_system(|In(_), selector: Res<MapSelector>| *selector).dedupe();

            // The signal chain under test.
            let switched_signal = control_signal.dedupe().switch_signal_map(
                clone!((signal_a, signal_b) move |In(selector): In<MapSelector>| match selector {
                    MapSelector::A => signal_a.clone().left_either(),
                    MapSelector::B => signal_b.clone().right_either(),
                }),
            );

            // Register the final signal to a capture system.
            let handle = switched_signal.for_each(capture_map_output).register(app.world_mut());

            // --- Test 1: Initial State with Empty Map ---
            // The first update should select Map A and authoritatively replay its empty state.
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(
                diffs,
                vec![MapDiff::Replace { entries: vec![] }],
                "Initial update with empty map should produce an authoritative empty Replace"
            );

            // --- Test 2: Insert to Empty Active Map ---
            // An insert to the initially empty Map A should be forwarded as an Insert.
            map_a.write(app.world_mut()).insert(1, 10);
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Insert to active map should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Insert { key: 1, value: 10 }),
                "Should forward Insert diff from Map A."
            );

            // --- Test 3: Another Insert ---
            map_a.write(app.world_mut()).insert(2, 20);
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Second insert should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Insert { key: 2, value: 20 }),
                "Should forward second Insert diff from Map A."
            );

            // --- Test 4: Switch to Map B (non-empty) ---
            *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching to non-empty map should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Replace { entries } if entries == &vec![(1, 100), (2, 200), (3, 300)]),
                "Switch should emit a Replace with Map B's contents."
            );

            // --- Test 5: Switch Back to Map A (now non-empty) ---
            // When switching back, Map A now has items, so it should emit Replace.
            *app.world_mut().resource_mut::<MapSelector>() = MapSelector::A;
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert_eq!(diffs.len(), 1, "Switching back to A should produce one diff.");
            assert!(
                matches!(&diffs[0], MapDiff::Replace { entries } if entries == &vec![(1, 10), (2, 20)]),
                "Switching back should Replace with the current state of Map A."
            );

            handle.release(app.world_mut());
        }
    }

    #[test]
    fn test_switch_signal_vec_populated_to_empty_collection_correctness_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();

        let list_a = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
        let list_b: MutableVec<i32> = app.world_mut().into();
        let signal_a = list_a.signal_vec().map_in(identity).cache().broadcast();
        let signal_b = list_b.signal_vec().broadcast();

        #[derive(Resource, Clone, Copy, PartialEq, Debug)]
        enum ListSelector {
            A,
            B,
        }

        app.insert_resource(ListSelector::A);
        let selector = signal::from_system(|In(_), selector: Res<ListSelector>| *selector).dedupe();
        let switched = selector.switch_signal_vec(
            clone!((signal_a, signal_b) move |In(selector): In<ListSelector>| match selector {
                ListSelector::A => signal_a.clone().left_either(),
                ListSelector::B => signal_b.clone().right_either(),
            }),
        );
        let handle = switched.for_each(capture_vec_output).register(app.world_mut());
        let mut downstream = Vec::new();

        app.update();
        let initial_diffs = get_and_clear_vec_output::<i32>(app.world_mut());
        for diff in initial_diffs.iter().cloned() {
            diff.apply_to_vec(&mut downstream);
        }
        assert_eq!(
            downstream,
            vec![10, 20],
            "initial List A did not materialize from emitted diffs; actual diffs: {initial_diffs:?}"
        );

        app.world_mut().resource_mut::<SignalVecOutput<i32>>().0.clear();
        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
        app.update();
        let switch_diffs = get_and_clear_vec_output::<i32>(app.world_mut());
        for diff in switch_diffs.iter().cloned() {
            diff.apply_to_vec(&mut downstream);
        }
        let expected_diffs = vec![VecDiff::Replace { values: vec![] }];
        let correct = switch_diffs == expected_diffs && downstream.is_empty();

        handle.release(app.world_mut());
        assert!(
            correct,
            "populated-to-empty switch must emit {expected_diffs:?} and reconstruct an empty downstream; actual diffs: {switch_diffs:?}, actual downstream state: {downstream:?}"
        );
    }

    #[test]
    fn test_switch_signal_map_populated_to_empty_collection_correctness_characterization() {
        use alloc::collections::BTreeMap;

        use crate::signal_map::{IntoSignalMapEither, MapDiff, MutableBTreeMap, SignalMapExt};

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

        #[derive(Resource, Default)]
        struct CapturedMapDiffs(Vec<MapDiff<i32, i32>>);

        fn capture_map_diffs(In(diffs): In<Vec<MapDiff<i32, i32>>>, mut captured: ResMut<CapturedMapDiffs>) {
            captured.0.extend(diffs);
        }

        let mut app = create_test_app();
        app.init_resource::<CapturedMapDiffs>();

        let map_a = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20)])
            .spawn(app.world_mut());
        let map_b: MutableBTreeMap<i32, i32> = app.world_mut().into();
        let signal_a = map_a
            .signal_map()
            .map_value(|In(value): In<i32>| value)
            .cache()
            .broadcast();
        let signal_b = map_b.signal_map().broadcast();

        #[derive(Resource, Clone, Copy, PartialEq, Debug)]
        enum MapSelector {
            A,
            B,
        }

        app.insert_resource(MapSelector::A);
        let selector = signal::from_system(|In(_), selector: Res<MapSelector>| *selector).dedupe();
        let switched = selector.switch_signal_map(
            clone!((signal_a, signal_b) move |In(selector): In<MapSelector>| match selector {
                MapSelector::A => signal_a.clone().left_either(),
                MapSelector::B => signal_b.clone().right_either(),
            }),
        );
        let handle = switched.for_each(capture_map_diffs).register(app.world_mut());
        let mut downstream = BTreeMap::new();

        app.update();
        let initial_diffs = core::mem::take(&mut app.world_mut().resource_mut::<CapturedMapDiffs>().0);
        apply_map_diffs(&mut downstream, initial_diffs.iter().cloned());
        assert_eq!(
            downstream,
            [(1, 10), (2, 20)].into_iter().collect(),
            "initial Map A did not materialize from emitted diffs; actual diffs: {initial_diffs:?}"
        );

        app.world_mut().resource_mut::<CapturedMapDiffs>().0.clear();
        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
        app.update();
        let switch_diffs = core::mem::take(&mut app.world_mut().resource_mut::<CapturedMapDiffs>().0);
        apply_map_diffs(&mut downstream, switch_diffs.iter().cloned());
        let expected_diffs = vec![MapDiff::Replace { entries: vec![] }];
        let correct = switch_diffs == expected_diffs && downstream.is_empty();

        handle.release(app.world_mut());
        assert!(
            correct,
            "populated-to-empty switch must emit {expected_diffs:?} and reconstruct an empty downstream; actual diffs: {switch_diffs:?}, actual downstream state: {downstream:?}"
        );
    }

    #[test]
    fn test_switch_signal_vec_repeated_empty_boundary_collection_correctness_oracle() {
        fn assert_step(
            app: &App,
            cursor: &mut usize,
            downstream: &mut Vec<i32>,
            selected_model: &[i32],
            expected_new_diffs: &[VecDiff<i32>],
            step: &str,
        ) {
            let captured = &app.world().resource::<SignalVecOutput<i32>>().0;
            let new_diffs = captured[*cursor..].to_vec();
            *cursor = captured.len();
            for diff in new_diffs.iter().cloned() {
                diff.apply_to_vec(downstream);
            }
            assert_eq!(
                new_diffs, expected_new_diffs,
                "{step}: unexpected emitted diffs; all captured diffs: {captured:?}"
            );
            assert_eq!(
                downstream.as_slice(),
                selected_model,
                "{step}: diff-reconstructed downstream diverged from selected-source plain model; new diffs: {new_diffs:?}, all captured diffs: {captured:?}"
            );
        }

        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();

        let list_a = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
        let list_b: MutableVec<i32> = app.world_mut().into();
        let list_c = MutableVec::builder().values([70, 80, 90]).spawn(app.world_mut());
        let signal_a = list_a.signal_vec().broadcast();
        let signal_b = list_b.signal_vec().broadcast();
        let signal_c = list_c.signal_vec().broadcast();

        #[derive(Resource, Clone, Copy, PartialEq, Debug)]
        enum ListSelector {
            A,
            B,
            C,
        }

        app.insert_resource(ListSelector::A);
        let selector = signal::from_system(|In(_), selector: Res<ListSelector>| *selector);
        let switched = selector.switch_signal_vec(
            clone!((signal_a, signal_b, signal_c) move |In(selector): In<ListSelector>| match selector {
                ListSelector::A => signal_a.clone(),
                ListSelector::B => signal_b.clone(),
                ListSelector::C => signal_c.clone(),
            }),
        );
        let handle = switched.for_each(capture_vec_output).register(app.world_mut());

        let mut model_a = vec![10, 20];
        let mut model_b = Vec::new();
        let mut model_c = vec![70, 80, 90];
        let mut downstream = Vec::new();
        let mut cursor = 0;

        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[VecDiff::Replace {
                values: model_a.clone(),
            }],
            "initial A",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[VecDiff::Replace { values: vec![] }],
            "A -> empty B",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::C;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_c,
            &[VecDiff::Replace {
                values: model_c.clone(),
            }],
            "empty B -> C",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[VecDiff::Replace { values: vec![] }],
            "C -> empty B",
        );

        list_a.write(app.world_mut()).push(30);
        model_a.push(30);
        app.update();
        assert_step(&app, &mut cursor, &mut downstream, &model_b, &[], "mutate inactive A");

        list_c.write(app.world_mut()).remove(1);
        model_c.remove(1);
        app.update();
        assert_step(&app, &mut cursor, &mut downstream, &model_b, &[], "mutate inactive C");

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::C;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_c,
            &[VecDiff::Replace {
                values: model_c.clone(),
            }],
            "reselect mutated C",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[VecDiff::Replace { values: vec![] }],
            "mutated C -> empty B",
        );

        list_b.write(app.world_mut()).push(5);
        model_b.push(5);
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[VecDiff::Push { value: 5 }],
            "populate active B",
        );

        list_b.write(app.world_mut()).clear();
        model_b.clear();
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[VecDiff::Clear],
            "clear active B",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::A;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[VecDiff::Replace {
                values: model_a.clone(),
            }],
            "reselect mutated A",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[VecDiff::Replace { values: vec![] }],
            "mutated A -> empty B again",
        );

        list_a.write(app.world_mut()).clear();
        model_a.clear();
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[],
            "clear inactive A while repeated B selection remains silent",
        );

        *app.world_mut().resource_mut::<ListSelector>() = ListSelector::A;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[VecDiff::Replace { values: vec![] }],
            "empty B -> inactive-cleared empty A",
        );

        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[],
            "repeated outer emission selecting the same vector identity",
        );

        handle.release(app.world_mut());
    }

    #[test]
    fn test_switch_signal_map_repeated_empty_boundary_collection_correctness_oracle() {
        use alloc::collections::BTreeMap;

        use crate::signal_map::{MapDiff, MutableBTreeMap, SignalMapExt};

        #[derive(Resource, Default, Debug)]
        struct CapturedMapDiffs(Vec<MapDiff<i32, i32>>);

        fn capture_map_diffs(In(diffs): In<Vec<MapDiff<i32, i32>>>, mut captured: ResMut<CapturedMapDiffs>) {
            captured.0.extend(diffs);
        }

        fn apply_map_diff(state: &mut BTreeMap<i32, i32>, diff: MapDiff<i32, i32>) {
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

        fn replacement(model: &BTreeMap<i32, i32>) -> MapDiff<i32, i32> {
            MapDiff::Replace {
                entries: model.iter().map(|(&key, &value)| (key, value)).collect(),
            }
        }

        fn assert_step(
            app: &App,
            cursor: &mut usize,
            downstream: &mut BTreeMap<i32, i32>,
            selected_model: &BTreeMap<i32, i32>,
            expected_new_diffs: &[MapDiff<i32, i32>],
            step: &str,
        ) {
            let captured = &app.world().resource::<CapturedMapDiffs>().0;
            let new_diffs = captured[*cursor..].to_vec();
            *cursor = captured.len();
            for diff in new_diffs.iter().cloned() {
                apply_map_diff(downstream, diff);
            }
            assert_eq!(
                new_diffs, expected_new_diffs,
                "{step}: unexpected emitted diffs; all captured diffs: {captured:?}"
            );
            assert_eq!(
                downstream, selected_model,
                "{step}: diff-reconstructed downstream diverged from selected-source plain model; new diffs: {new_diffs:?}, all captured diffs: {captured:?}"
            );
        }

        let mut app = create_test_app();
        app.init_resource::<CapturedMapDiffs>();

        let map_a = MutableBTreeMap::builder()
            .values([(1, 10), (2, 20)])
            .spawn(app.world_mut());
        let map_b: MutableBTreeMap<i32, i32> = app.world_mut().into();
        let map_c = MutableBTreeMap::builder()
            .values([(7, 70), (8, 80), (9, 90)])
            .spawn(app.world_mut());
        let signal_a = map_a.signal_map().broadcast();
        let signal_b = map_b.signal_map().broadcast();
        let signal_c = map_c.signal_map().broadcast();

        #[derive(Resource, Clone, Copy, PartialEq, Debug)]
        enum MapSelector {
            A,
            B,
            C,
        }

        app.insert_resource(MapSelector::A);
        let selector = signal::from_system(|In(_), selector: Res<MapSelector>| *selector);
        let switched = selector.switch_signal_map(
            clone!((signal_a, signal_b, signal_c) move |In(selector): In<MapSelector>| match selector {
                MapSelector::A => signal_a.clone(),
                MapSelector::B => signal_b.clone(),
                MapSelector::C => signal_c.clone(),
            }),
        );
        let handle = switched.for_each(capture_map_diffs).register(app.world_mut());

        let mut model_a: BTreeMap<_, _> = [(1, 10), (2, 20)].into_iter().collect();
        let mut model_b = BTreeMap::new();
        let mut model_c: BTreeMap<_, _> = [(7, 70), (8, 80), (9, 90)].into_iter().collect();
        let mut downstream = BTreeMap::new();
        let mut cursor = 0;

        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[replacement(&model_a)],
            "initial A",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[replacement(&model_b)],
            "A -> empty B",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::C;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_c,
            &[replacement(&model_c)],
            "empty B -> C",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[replacement(&model_b)],
            "C -> empty B",
        );

        map_a.write(app.world_mut()).insert(3, 30);
        model_a.insert(3, 30);
        app.update();
        assert_step(&app, &mut cursor, &mut downstream, &model_b, &[], "mutate inactive A");

        map_c.write(app.world_mut()).remove(&8);
        model_c.remove(&8);
        app.update();
        assert_step(&app, &mut cursor, &mut downstream, &model_b, &[], "mutate inactive C");

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::C;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_c,
            &[replacement(&model_c)],
            "reselect mutated C",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[replacement(&model_b)],
            "mutated C -> empty B",
        );

        map_b.write(app.world_mut()).insert(4, 40);
        model_b.insert(4, 40);
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[MapDiff::Insert { key: 4, value: 40 }],
            "populate active B",
        );

        map_b.write(app.world_mut()).clear();
        model_b.clear();
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[MapDiff::Clear],
            "clear active B",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::A;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[replacement(&model_a)],
            "reselect mutated A",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::B;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[replacement(&model_b)],
            "mutated A -> empty B again",
        );

        map_a.write(app.world_mut()).clear();
        model_a.clear();
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_b,
            &[],
            "clear inactive A while repeated B selection remains silent",
        );

        *app.world_mut().resource_mut::<MapSelector>() = MapSelector::A;
        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[replacement(&model_a)],
            "empty B -> inactive-cleared empty A",
        );

        app.update();
        assert_step(
            &app,
            &mut cursor,
            &mut downstream,
            &model_a,
            &[],
            "repeated outer emission selecting the same map identity",
        );

        handle.release(app.world_mut());
    }

    #[test]
    fn test_throttle() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins.build().disable::<TimePlugin>(), JonmoPlugin::default()));
        app.init_resource::<Time>(); // manually managing time for the test
        app.init_resource::<SignalOutput<i32>>();

        // A simple counter to generate a new value (1, 2, 3...) for each update.
        let source_signal = signal::from_system(|In(_), mut counter: Local<i32>| {
            *counter += 1;
            *counter
        });

        let handle = source_signal
            .throttle(Duration::from_millis(100))
            .map(capture_output)
            .register(app.world_mut());

        // 1. Initial emission: The first update runs with near-zero delta. The first value should
        //    always pass.
        app.update();
        assert_eq!(
            get_output(app.world()),
            Some(1),
            "First value (1) should pass immediately."
        );
        // Clear the output to prepare for the next check.
        app.world_mut().resource_mut::<SignalOutput<i32>>().0 = None;

        // 2. Blocked emission: Manually advance time by 50ms.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(50));
        app.update(); // `time.delta()` is now 50ms. The timer ticks but isn't finished.
        assert_eq!(
            get_output::<i32>(app.world()),
            None,
            "Value (2) emitted after 50ms should be blocked."
        );

        // 3. Another blocked emission: Advance time by another 40ms (total 90ms).
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(40));
        app.update(); // `time.delta()` is 40ms. Timer ticks to 90ms. Still not finished.
        assert_eq!(
            get_output::<i32>(app.world()),
            None,
            "Value (3) emitted after 90ms total should be blocked."
        );
        app.world_mut().entity(handle.entity().entity());
        // 4. Allowed emission: Advance time by another 20ms (total 110ms).
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(20));
        app.update(); // `time.delta()` is 20ms. Timer ticks to 110ms, which is >= 100ms.
        // The source signal has been called 4 times now. The value should be 4.
        assert_eq!(
            get_output(app.world()),
            Some(4),
            "Value (4) should pass after total duration > 100ms."
        );
        app.world_mut().resource_mut::<SignalOutput<i32>>().0 = None;

        // 5. Blocked again: The timer was reset on the last pass. A small advance is not enough.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(10));
        app.update();
        assert_eq!(
            get_output::<i32>(app.world()),
            None,
            "Value (5) immediately after a pass should be blocked again."
        );

        // 6. Pass again: Advance time to pass the threshold again.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_millis(100));
        app.update();
        // The source signal has now been called 6 times. The value should be 6.
        assert_eq!(
            get_output(app.world()),
            Some(6),
            "Value (6) should pass again after another full duration."
        );

        handle.release(app.world_mut());
    }

    #[test]
    fn test_map_bool() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<&'static str>>();

        // Test true case
        let signal_true = signal::from_system(|In(_)| true)
            .map_bool(|In(_)| "True Branch", |In(_)| "False Branch")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("True Branch"));
        signal_true.release(app.world_mut());

        // Test false case
        let signal_false = signal::from_system(|In(_)| false)
            .map_bool(|In(_)| "True Branch", |In(_)| "False Branch")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("False Branch"));
        signal_false.release(app.world_mut());
    }

    #[test]
    fn test_map_true() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<&'static str>>>();

        // --- Test true case ---
        let source_true = signal::from_system(|In(_)| true);
        let signal_true = source_true
            .map_true(|In(_)| "Was True")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<&'static str>>(app.world()),
            Some(Some("Was True")),
            "map_true should execute system and output Some(value) when input is true"
        );
        signal_true.release(app.world_mut());

        // --- Test false case --- Reset output
        app.world_mut().resource_mut::<SignalOutput<Option<&'static str>>>().0 = None;
        let source_false = signal::from_system(|In(_)| false);
        let signal_false = source_false
            .map_true(|In(_)| "Was True")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<&'static str>>(app.world()),
            Some(None),
            "map_true should not execute system and output None when input is false"
        );
        signal_false.release(app.world_mut());
    }

    #[test]
    fn test_map_false() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<&'static str>>>();

        // --- Test false case ---
        let source_false = signal::from_system(|In(_)| false);
        let signal_false = source_false
            .map_false(|In(_)| "Was False")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<&'static str>>(app.world()),
            Some(Some("Was False")),
            "map_false should execute system and output Some(value) when input is false"
        );
        signal_false.release(app.world_mut());

        // --- Test true case --- Reset output
        app.world_mut().resource_mut::<SignalOutput<Option<&'static str>>>().0 = None;
        let source_true = signal::from_system(|In(_)| true);
        let signal_true = source_true
            .map_false(|In(_)| "Was False")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<&'static str>>(app.world()),
            Some(None),
            "map_false should not execute system and output None when input is true"
        );
        signal_true.release(app.world_mut());
    }

    #[test]
    fn test_map_option() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<String>>();

        // Test Some case
        let signal_some = signal::from_system(|In(_)| Some(42))
            .map_option(
                |In(value): In<i32>| format!("Some({value})"),
                |In(_)| "None".to_string(),
            )
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), Some("Some(42)".to_string()));
        signal_some.release(app.world_mut());

        // Reset output
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // Test None case
        let signal_none = signal::from_system(|In(_)| None::<i32>)
            .map_option(
                |In(value): In<i32>| format!("Some({value})"),
                |In(_)| "None".to_string(),
            )
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), Some("None".to_string()));
        signal_none.release(app.world_mut());
    }

    #[test]
    fn test_map_some() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<String>>>();

        // --- Test Some case ---
        let source_some = signal::from_system(|In(_)| Some(42));
        let signal_some = source_some
            .map_some(|In(val): In<i32>| format!("Got {val}"))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world()),
            Some(Some("Got 42".to_string())),
            "map_some should execute system with inner value and output Some(new_value) for a Some input"
        );
        signal_some.release(app.world_mut());

        // --- Test None case --- Reset output
        app.world_mut().resource_mut::<SignalOutput<Option<String>>>().0 = None;
        let source_none = signal::from_system(|In(_)| None::<i32>);
        let signal_none = source_none
            .map_some(|In(val): In<i32>| format!("Got {val}"))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<String>>(app.world()),
            Some(None),
            "map_some should not execute system and output None for a None input"
        );
        signal_none.release(app.world_mut());
    }

    #[test]
    fn test_map_none() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<&'static str>>>();

        // --- Test None case ---
        let source_none = signal::from_system(|In(_)| None::<i32>);
        let signal_none = source_none
            .map_none(|In(_)| "Was None")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<&'static str>>(app.world()),
            Some(Some("Was None")),
            "map_none should execute system and output Some(value) for a None input"
        );
        signal_none.release(app.world_mut());

        // --- Test Some case --- Reset output
        app.world_mut().resource_mut::<SignalOutput<Option<&'static str>>>().0 = None;
        let source_some = signal::from_system(|In(_)| Some(42));
        let signal_some = source_some
            .map_none(|In(_)| "Was None")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<&'static str>>(app.world()),
            Some(None),
            "map_none should not execute system and output None for a Some input"
        );
        signal_some.release(app.world_mut());
    }

    #[test]
    fn test_to_signal_vec() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source_vec = Arc::new(Mutex::new(None::<Vec<i32>>));
        let source_signal = signal::from_system(clone!((source_vec) move |In(_)| {
            source_vec.lock().unwrap().take()
        }));
        let signal_vec = source_signal.to_signal_vec();
        let handle = signal_vec.for_each(capture_vec_output::<i32>).register(app.world_mut());

        // Frame 1: Initial state, no emission.
        app.update();
        assert!(
            get_and_clear_vec_output::<i32>(app.world_mut()).is_empty(),
            "Should not emit anything when source is None"
        );

        // Frame 2: Emit the first vector.
        *source_vec.lock().unwrap() = Some(vec![1, 2, 3]);
        app.update();
        let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Expected one diff on first emission");
        if let VecDiff::Replace { values } = &diffs[0] {
            assert_eq!(values, &vec![1, 2, 3]);
        } else {
            panic!("Expected a Replace diff, got {:?}", diffs[0]);
        }

        // Frame 3: Emit the same vector. `to_signal_vec` is purely structural and forwards it.
        *source_vec.lock().unwrap() = Some(vec![1, 2, 3]);
        app.update();
        let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::Replace { values: vec![1, 2, 3] }],
            "to_signal_vec should forward repeated equal vectors without implicit deduplication"
        );

        // Frame 4: Emit a different vector.
        *source_vec.lock().unwrap() = Some(vec![4, 5]);
        app.update();
        let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Expected one diff on new vector emission");
        if let VecDiff::Replace { values } = &diffs[0] {
            assert_eq!(values, &vec![4, 5]);
        } else {
            panic!("Expected a Replace diff, got {:?}", diffs[0]);
        }
        handle.release(app.world_mut());
    }

    #[test]
    fn to_signal_vec_cache_is_authoritative_without_a_synthetic_empty_baseline() {
        fn assert_replayable<Signal: crate::signal_vec::ReplayableSignalVec>(_signal: &Signal) {}

        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        let source_value = Arc::new(Mutex::new(None::<Vec<i32>>));
        let cached = signal::from_system(clone!((source_value) move |In(())| {
            source_value.lock().unwrap().take()
        }))
        .to_signal_vec()
        .cache()
        .broadcast();
        assert_replayable(&cached);

        let direct = cached.clone().register(app.world_mut());
        let entities_before_repeated_cache = app.world().entities().len();
        let repeated_cache = cached.clone().cache().register(app.world_mut());
        assert_eq!(repeated_cache.entity(), direct.entity());
        assert_eq!(app.world().entities().len(), entities_before_repeated_cache);

        let observer = cached
            .clone()
            .for_each(capture_vec_output::<i32>)
            .register(app.world_mut());
        app.update();
        assert!(get_and_clear_vec_output::<i32>(app.world_mut()).is_empty());

        *source_value.lock().unwrap() = Some(Vec::new());
        app.update();
        assert_eq!(
            get_and_clear_vec_output::<i32>(app.world_mut()),
            vec![VecDiff::Replace { values: Vec::new() }]
        );

        *source_value.lock().unwrap() = Some(Vec::new());
        app.update();
        assert_eq!(
            get_and_clear_vec_output::<i32>(app.world_mut()),
            vec![VecDiff::Replace { values: Vec::new() }],
            "caching to_signal_vec must not reintroduce implicit deduplication"
        );

        *source_value.lock().unwrap() = Some(vec![1, 2]);
        app.update();
        assert_eq!(
            get_and_clear_vec_output::<i32>(app.world_mut()),
            vec![VecDiff::Replace { values: vec![1, 2] }]
        );

        observer.release(app.world_mut());
        repeated_cache.release(app.world_mut());
        direct.release(app.world_mut());
        drop(cached);
    }

    #[test]
    fn test_builder_from_system() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let signal = signal::from_system(|In(_)| 42)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(42));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_function() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        // Test basic FnMut closure
        let counter = Arc::new(Mutex::new(0));
        let signal = signal::from_function(clone!((counter) move || {
            let mut c = counter.lock().unwrap();
            *c += 1;
            *c
        }))
        .map(capture_output)
        .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1));

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(3));

        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_always() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();

        let signal = signal::always(42).map(capture_output).register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(42));

        // Verify it still outputs the same value on subsequent updates
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(42));

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(42));

        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_entity() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let test_entity = app.world_mut().spawn_empty().id();
        let signal = signal::from_entity(test_entity)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(test_entity));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_lazy_entity() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let lazy = LazyEntity::new();
        let test_entity = app.world_mut().spawn_empty().id();
        lazy.set(test_entity);

        let signal = signal::from_entity(lazy).map(capture_output).register(app.world_mut());

        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(test_entity));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_ancestor() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let grandparent = app.world_mut().spawn_empty().id();
        let parent = app.world_mut().spawn_empty().id();
        let child = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(grandparent).add_child(parent);
        app.world_mut().entity_mut(parent).add_child(child);

        // Test self (generations = 0)
        let signal_self = signal::from_ancestor(child, 0)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(child));
        signal_self.release(app.world_mut());

        // Test parent (generations = 1)
        let signal_parent = signal::from_ancestor(child, 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal_parent.release(app.world_mut());

        // Test grandparent (generations = 2)
        let signal_gp = signal::from_ancestor(child, 2)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(grandparent));
        signal_gp.release(app.world_mut());

        // Test invalid generation
        app.world_mut().resource_mut::<SignalOutput<Entity>>().0 = None;
        let signal_invalid = signal::from_ancestor(child, 3)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Entity>(app.world()),
            None,
            "Should terminate for invalid ancestor"
        );
        signal_invalid.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_ancestor_lazy() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let lazy_child = LazyEntity::new();
        let grandparent = app.world_mut().spawn_empty().id();
        let parent = app.world_mut().spawn_empty().id();
        let child = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(grandparent).add_child(parent);
        app.world_mut().entity_mut(parent).add_child(child);
        lazy_child.set(child);

        // Test parent (generations = 1)
        let signal_parent = signal::from_ancestor(lazy_child.clone(), 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal_parent.release(app.world_mut());

        // Test grandparent (generations = 2)
        let signal_gp = signal::from_ancestor(lazy_child, 2)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(grandparent));
        signal_gp.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_parent() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let parent = app.world_mut().spawn_empty().id();
        let child = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child);

        let signal = signal::from_parent(child).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal.release(app.world_mut());

        // Test no parent
        app.world_mut().resource_mut::<SignalOutput<Entity>>().0 = None;
        let no_parent_entity = app.world_mut().spawn_empty().id();
        let signal_no_parent = signal::from_parent(no_parent_entity)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Entity>(app.world()),
            None,
            "Should terminate for entity with no parent"
        );
        signal_no_parent.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_parent_lazy() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let lazy_child = LazyEntity::new();
        let parent = app.world_mut().spawn_empty().id();
        let child = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(parent).add_child(child);
        lazy_child.set(child);

        let signal = signal::from_parent(lazy_child)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_component() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        // Test component exists
        let entity_with = app.world_mut().spawn(TestData(42)).id();
        let signal_with = signal::from_component::<TestData>(entity_with)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));
        signal_with.release(app.world_mut());

        // Test component missing
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        let entity_without = app.world_mut().spawn_empty().id();
        let signal_without = signal::from_component::<TestData>(entity_without)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when component is missing"
        );
        signal_without.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_component_lazy() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        let lazy = LazyEntity::new();
        let entity_with = app.world_mut().spawn(TestData(42)).id();
        lazy.set(entity_with);

        let signal = signal::from_component::<TestData>(lazy)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_component_option() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<TestData>>>();

        // Test component exists
        let entity_with = app.world_mut().spawn(TestData(42)).id();
        let signal_with = signal::from_component_option::<TestData>(entity_with)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(42))));
        signal_with.release(app.world_mut());

        // Test component missing
        let entity_without = app.world_mut().spawn_empty().id();
        let signal_without = signal::from_component_option::<TestData>(entity_without)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<TestData>>(app.world()),
            Some(None),
            "Should output Some(None) when component is missing"
        );
        signal_without.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_component_option_lazy() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<TestData>>>();

        let lazy = LazyEntity::new();
        let entity_with = app.world_mut().spawn(TestData(42)).id();
        lazy.set(entity_with);

        let signal = signal::from_component_option::<TestData>(lazy)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(42))));
        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_resource() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        // Test resource exists
        app.insert_resource(TestData(42));
        let signal_with = signal::from_resource::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));
        signal_with.release(app.world_mut());

        // Test resource missing
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        app.world_mut().remove_resource::<TestData>();
        let signal_without = signal::from_resource::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when resource is missing"
        );
        signal_without.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_resource_option() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<TestData>>>();

        // Test resource exists
        app.insert_resource(TestData(42));
        let signal_with = signal::from_resource_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(42))));
        signal_with.release(app.world_mut());

        // Test resource missing
        app.world_mut().remove_resource::<TestData>();
        let signal_without = signal::from_resource_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<TestData>>(app.world()),
            Some(None),
            "Should output Some(None) when resource is missing"
        );
        signal_without.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_component_changed() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        // Spawn entity with component
        let entity = app.world_mut().spawn(TestData(42)).id();
        let signal = signal::from_component_changed::<TestData>(entity)
            .map(capture_output)
            .register(app.world_mut());

        // First update should emit because component was just added (Changed)
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));

        // Clear output and update again - should NOT emit because nothing changed
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when component has not changed"
        );

        // Mutate the component and update - should emit
        app.world_mut().entity_mut(entity).get_mut::<TestData>().unwrap().0 = 100;
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(100)));

        signal.release(app.world_mut());

        // Test component missing
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        let entity_without = app.world_mut().spawn_empty().id();
        let signal_without = signal::from_component_changed::<TestData>(entity_without)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when component is missing"
        );
        signal_without.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_component_changed_lazy() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        let lazy = LazyEntity::new();
        let entity = app.world_mut().spawn(TestData(42)).id();
        lazy.set(entity);

        let signal = signal::from_component_changed::<TestData>(lazy)
            .map(capture_output)
            .register(app.world_mut());

        // First update should emit because component was just added (Changed)
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));

        // Clear output and update again - should NOT emit because nothing changed
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when component has not changed"
        );

        // Mutate the component and update - should emit
        app.world_mut().entity_mut(entity).get_mut::<TestData>().unwrap().0 = 100;
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(100)));

        signal.release(app.world_mut());
    }

    #[test]
    fn test_builder_from_resource_changed() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<TestData>>();

        // Insert resource
        app.insert_resource(TestData(42));
        let signal = signal::from_resource_changed::<TestData>()
            .map(capture_output)
            .register(app.world_mut());

        // First update should emit because resource was just added (Changed)
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));

        // Clear output and update again - should NOT emit because nothing changed
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when resource has not changed"
        );

        // Mutate the resource and update - should emit
        app.world_mut().resource_mut::<TestData>().0 = 100;
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(100)));

        signal.release(app.world_mut());

        // Test resource missing
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        app.world_mut().remove_resource::<TestData>();
        let signal_without = signal::from_resource_changed::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<TestData>(app.world()),
            None,
            "Should terminate when resource is missing"
        );
        signal_without.release(app.world_mut());
    }

    #[test]
    fn test_signal_option() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<i32>>>();

        // Test Some variant - signal wrapped in Some
        let some_signal = crate::signal::option(Some(signal::from_system(|In(_)| 42)))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<i32>>(app.world()),
            Some(Some(42)),
            "signal::option(Some(signal)) should output Some(value)"
        );
        some_signal.release(app.world_mut());

        // Test None variant - constant None
        app.world_mut().resource_mut::<SignalOutput<Option<i32>>>().0 = None;
        let none_signal = crate::signal::option(None::<BoxedSignal<i32>>)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(
            get_output::<Option<i32>>(app.world()),
            Some(None),
            "signal::option(None) should output None"
        );
        none_signal.release(app.world_mut());

        // Test with dynamic signal
        app.world_mut().resource_mut::<SignalOutput<Option<i32>>>().0 = None;
        let counter = Arc::new(Mutex::new(0));
        let dynamic_signal = crate::signal::option(Some(signal::from_system(clone!((counter) move |In(_)| {
            let mut c = counter.lock().unwrap();
            *c += 1;
            *c
        }))))
        .map(capture_output)
        .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<Option<i32>>(app.world()), Some(Some(1)));
        app.update();
        assert_eq!(get_output::<Option<i32>>(app.world()), Some(Some(2)));
        app.update();
        assert_eq!(get_output::<Option<i32>>(app.world()), Some(Some(3)));
        dynamic_signal.release(app.world_mut());
    }

    #[test]
    fn test_signal_option_with_switch() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<Option<i32>>>();

        // Test switching between Some and None based on a condition
        let condition = Arc::new(Mutex::new(true));
        let signal = signal::from_system(clone!((condition) move |In(_)| {
            *condition.lock().unwrap()
        }))
        .map(move |In(use_signal): In<bool>| {
            if use_signal {
                crate::signal::option(Some(signal::from_system(|In(_)| 42))).cache()
            } else {
                crate::signal::option(None).cache()
            }
        })
        .flatten()
        .map(capture_output)
        .register(app.world_mut());

        // First update: condition is true, should output Some(42)
        app.update();
        assert_eq!(
            get_output::<Option<i32>>(app.world()),
            Some(Some(42)),
            "Should output Some(42) when condition is true"
        );

        // Change condition to false
        *condition.lock().unwrap() = false;
        app.update();
        assert_eq!(
            get_output::<Option<i32>>(app.world()),
            Some(None),
            "Should output None when condition is false"
        );

        // Change back to true
        *condition.lock().unwrap() = true;
        app.update();
        assert_eq!(
            get_output::<Option<i32>>(app.world()),
            Some(Some(42)),
            "Should output Some(42) when condition is true again"
        );

        signal.release(app.world_mut());
    }

    #[test]
    fn retained_signal_chain_outlives_released_claim() {
        let mut app = create_test_app();
        let source_signal_struct = signal::from_system(|In(_)| 1).broadcast();
        let claim = source_signal_struct.clone().register(app.world_mut());
        let system_entity = claim.entity().entity();
        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "System entity should exist after registration"
        );
        assert!(
            test_has_signal_lifetime_retention(app.world(), system_entity),
            "the retained chain should keep its terminal materialized"
        );
        assert_eq!(
            **app.world().get::<SignalClaimCount>(system_entity).unwrap(),
            1,
            "SignalClaimCount should be 1"
        );
        claim.release(app.world_mut());
        assert_eq!(
            **app.world().get::<SignalClaimCount>(system_entity).unwrap(),
            0,
            "SignalClaimCount should be 0 after cleanup"
        );

        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "the retained chain should keep its terminal materialized after claim release"
        );
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));
        drop(source_signal_struct);

        app.update();
        app.update();
        assert!(
            app.world().get_entity(system_entity).is_err(),
            "the terminal should be reaped after its final retainer and direct claim are gone"
        );
    }

    #[test]
    fn multiple_chain_retainers_delay_terminal_reaping() {
        let mut app = create_test_app();
        let first_retainer = signal::from_system(|In(_)| 1).broadcast();
        let second_retainer = first_retainer.clone();
        let claim = first_retainer.clone().register(app.world_mut());
        let system_entity = claim.entity().entity();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));

        claim.release(app.world_mut());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 0);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));

        drop(first_retainer);
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));

        drop(second_retainer);
        app.update();
        assert!(app.world().get_entity(system_entity).is_err());
    }

    #[test]
    fn multiple_claims_share_one_retained_terminal() {
        let mut app = create_test_app();
        let source_signal_struct = signal::from_system(|In(_)| 1).broadcast();
        let first_claim = source_signal_struct.clone().register(app.world_mut());
        let system_entity = first_claim.entity().entity();
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);

        let second_claim = source_signal_struct.clone().register(app.world_mut());
        assert_eq!(system_entity, second_claim.entity().entity());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 2);

        first_claim.release(app.world_mut());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));

        drop(source_signal_struct);
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);

        second_claim.release(app.world_mut());
        app.update();
        assert!(app.world().get_entity(system_entity).is_err());
    }

    #[test]
    fn retained_chain_keeps_each_materialized_entity_until_final_retainer_drop() {
        let mut app = create_test_app();
        let source = signal::from_system(|In(_)| 1);
        let mapped = source.map(|In(value)| value + 1).broadcast();
        let claim = mapped.clone().register(app.world_mut());
        let broadcast_entity = claim.entity().entity();
        let mapped_entity = single_static_dependency(app.world_mut(), broadcast_entity);
        let source_entity = single_static_dependency(app.world_mut(), mapped_entity);
        assert!(app.world().get_entity(broadcast_entity).is_ok());
        assert!(app.world().get_entity(mapped_entity).is_ok());
        assert!(app.world().get_entity(source_entity).is_ok());
        assert!(test_has_signal_lifetime_retention(app.world(), broadcast_entity));
        assert!(test_has_signal_lifetime_retention(app.world(), mapped_entity));
        assert!(test_has_signal_lifetime_retention(app.world(), source_entity));
        assert_eq!(**app.world().get::<SignalClaimCount>(broadcast_entity).unwrap(), 1);
        assert_eq!(**app.world().get::<SignalClaimCount>(mapped_entity).unwrap(), 1);
        assert_eq!(**app.world().get::<SignalClaimCount>(source_entity).unwrap(), 1);
        claim.release(app.world_mut());

        assert_eq!(**app.world().get::<SignalClaimCount>(broadcast_entity).unwrap(), 0);
        assert_eq!(**app.world().get::<SignalClaimCount>(mapped_entity).unwrap(), 0);
        assert_eq!(**app.world().get::<SignalClaimCount>(source_entity).unwrap(), 0);
        assert!(app.world().get_entity(broadcast_entity).is_ok());
        assert!(app.world().get_entity(mapped_entity).is_ok());
        assert!(app.world().get_entity(source_entity).is_ok());
        assert!(test_has_signal_lifetime_retention(app.world(), broadcast_entity));
        assert!(test_has_signal_lifetime_retention(app.world(), mapped_entity));
        assert!(test_has_signal_lifetime_retention(app.world(), source_entity));

        drop(mapped);
        app.update();

        assert!(app.world().get_entity(broadcast_entity).is_err());
        assert!(app.world().get_entity(mapped_entity).is_err());
        assert!(app.world().get_entity(source_entity).is_err());
    }

    #[test]
    fn retained_terminal_reactivates_after_claim_count_reaches_zero() {
        let mut app = create_test_app();
        let source_signal_struct = signal::from_system(|In(_)| 1).broadcast();
        let first_claim = source_signal_struct.clone().register(app.world_mut());
        let system_entity = first_claim.entity().entity();
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);

        first_claim.release(app.world_mut());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 0);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));

        let second_claim = source_signal_struct.clone().register(app.world_mut());
        assert_eq!(system_entity, second_claim.entity().entity());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);
        assert!(test_has_signal_lifetime_retention(app.world(), system_entity));

        drop(source_signal_struct);
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert_eq!(**app.world().get::<SignalClaimCount>(system_entity).unwrap(), 1);

        second_claim.release(app.world_mut());
        assert!(app.world().get_entity(system_entity).is_err());
    }

    #[test]
    fn duplicate_shared_chain_claim_counts_registration_ownership_characterization() {
        let mut app = create_test_app();
        let shared = signal::from_system(|In(_)| 1).map_in(|value| value + 1).broadcast();
        let first = shared.clone().register(app.world_mut());
        let second = shared.clone().register(app.world_mut());
        let terminal = first.entity().entity();
        assert_eq!(terminal, second.entity().entity());
        let mapped = single_static_dependency(app.world_mut(), terminal);
        let source = single_static_dependency(app.world_mut(), mapped);

        let mut expected_model = DirectOwnershipModel::default();
        expected_model.add_node(source, []);
        expected_model.add_node(mapped, [source]);
        expected_model.add_node(terminal, [mapped]);
        let first_token = expected_model.acquire_external(terminal);
        let second_token = expected_model.acquire_external(terminal);
        let observed_registered = (
            registration_observation(app.world_mut(), source),
            registration_observation(app.world_mut(), mapped),
            registration_observation(app.world_mut(), terminal),
        );
        let expected_registered_counts = (
            expected_model.count(source),
            expected_model.count(mapped),
            expected_model.count(terminal),
        );

        first.release(app.world_mut());
        expected_model.release(first_token);
        let observed_after_first_release = (
            registration_observation(app.world_mut(), source),
            registration_observation(app.world_mut(), mapped),
            registration_observation(app.world_mut(), terminal),
        );
        let expected_after_first_counts = (
            expected_model.count(source),
            expected_model.count(mapped),
            expected_model.count(terminal),
        );

        second.release(app.world_mut());
        expected_model.release(second_token);
        let observed_after_final_release = (
            registration_observation(app.world_mut(), source),
            registration_observation(app.world_mut(), mapped),
            registration_observation(app.world_mut(), terminal),
        );
        let expected_after_final_counts = (
            expected_model.count(source),
            expected_model.count(mapped),
            expected_model.count(terminal),
        );

        drop(shared);
        app.update();
        let observed_after_retainer_drop = (
            registration_observation(app.world_mut(), source),
            registration_observation(app.world_mut(), mapped),
            registration_observation(app.world_mut(), terminal),
        );

        let observed = [
            (
                observed_registered.0.count,
                observed_registered.1.count,
                observed_registered.2.count,
            ),
            (
                observed_after_first_release.0.count,
                observed_after_first_release.1.count,
                observed_after_first_release.2.count,
            ),
            (
                observed_after_final_release.0.count,
                observed_after_final_release.1.count,
                observed_after_final_release.2.count,
            ),
        ];
        let expected = [
            (
                Some(expected_registered_counts.0),
                Some(expected_registered_counts.1),
                Some(expected_registered_counts.2),
            ),
            (
                Some(expected_after_first_counts.0),
                Some(expected_after_first_counts.1),
                Some(expected_after_first_counts.2),
            ),
            (
                Some(expected_after_final_counts.0),
                Some(expected_after_final_counts.1),
                Some(expected_after_final_counts.2),
            ),
        ];

        assert_eq!(
            (observed, observed_after_retainer_drop),
            (
                expected,
                (
                    RegistrationObservation::absent(),
                    RegistrationObservation::absent(),
                    RegistrationObservation::absent(),
                ),
            ),
            "two external registrations should count twice only at the broadcast terminal; its already-active immediate dependency chain should be claimed once and all entities should reap after balanced release"
        );
    }

    #[test]
    fn diamond_direct_counts_and_balanced_paths_registration_ownership_characterization() {
        let mut app = create_test_app();
        let source = signal::from_system(|In(_)| 1).broadcast();
        let branch_a = source.clone().map_in(|value| value + 10).broadcast();
        let branch_b = source.map_in(|value| value + 20);
        let diamond = branch_a.clone().zip(branch_b).broadcast();
        let terminal_handle = diamond.clone().register(app.world_mut());
        let terminal = terminal_handle.entity().entity();

        let zip_entity = single_static_dependency(app.world_mut(), terminal);
        let terminal_snapshot = graph_snapshot(app.world_mut());
        let wrappers = terminal_snapshot
            .node(zip_entity)
            .expect("diamond zip entity should be observable")
            .canonical_static_dependencies
            .clone()
            .expect("diamond zip entity should have canonical static dependencies");
        assert_eq!(wrappers.len(), 2, "zip must provide two distinct immediate wrappers");
        let branches = wrappers
            .iter()
            .map(|wrapper| single_static_dependency(app.world_mut(), *wrapper))
            .collect::<Vec<_>>();
        assert_ne!(branches[0], branches[1]);

        let branch_path_handle = branch_a.register(app.world_mut());
        let branch_a_entity = branch_path_handle.entity().entity();
        assert!(branches.contains(&branch_a_entity));
        let branch_b_entity = branches
            .iter()
            .copied()
            .find(|branch| *branch != branch_a_entity)
            .expect("diamond must contain the second branch");
        let branch_a_map_entity = single_static_dependency(app.world_mut(), branch_a_entity);
        let source_entity = single_static_dependency(app.world_mut(), branch_a_map_entity);
        assert_eq!(
            single_static_dependency(app.world_mut(), branch_b_entity),
            source_entity,
            "both branches must converge on one broadcast source identity"
        );
        let raw_source_entity = single_static_dependency(app.world_mut(), source_entity);

        let wrapper_for = |branch: Entity, world: &mut World| {
            wrappers
                .iter()
                .copied()
                .find(|wrapper| single_static_dependency(world, *wrapper) == branch)
                .expect("every branch should feed one zip wrapper")
        };
        let wrapper_a = wrapper_for(branch_a_entity, app.world_mut());
        let wrapper_b = wrapper_for(branch_b_entity, app.world_mut());

        let mut expected_model = DirectOwnershipModel::default();
        expected_model.add_node(raw_source_entity, []);
        expected_model.add_node(source_entity, [raw_source_entity]);
        expected_model.add_node(branch_a_map_entity, [source_entity]);
        expected_model.add_node(branch_a_entity, [branch_a_map_entity]);
        expected_model.add_node(branch_b_entity, [source_entity]);
        expected_model.add_node(wrapper_a, [branch_a_entity]);
        expected_model.add_node(wrapper_b, [branch_b_entity]);
        expected_model.add_node(zip_entity, [wrapper_a, wrapper_b]);
        expected_model.add_node(terminal, [zip_entity]);
        let terminal_token = expected_model.acquire_external(terminal);
        let branch_token = expected_model.acquire_external(branch_a_entity);

        let entities = [
            raw_source_entity,
            source_entity,
            branch_a_map_entity,
            branch_a_entity,
            branch_b_entity,
            wrapper_a,
            wrapper_b,
            zip_entity,
            terminal,
        ];
        let observe_counts =
            |app: &mut App| entities.map(|entity| registration_observation(app.world_mut(), entity).count);
        let expected_counts = |model: &DirectOwnershipModel| entities.map(|entity| Some(model.count(entity)));

        let observed_with_branch_owner = observe_counts(&mut app);
        let expected_with_branch_owner = expected_counts(&expected_model);

        branch_path_handle.release(app.world_mut());
        expected_model.release(branch_token);
        let observed_after_branch_release = observe_counts(&mut app);
        let expected_after_branch_release = expected_counts(&expected_model);

        terminal_handle.release(app.world_mut());
        expected_model.release(terminal_token);
        let observed_after_final_release = observe_counts(&mut app);
        let expected_after_final_release = expected_counts(&expected_model);

        drop(diamond);
        app.update();
        let observed_after_retainer_drop = observe_counts(&mut app);

        assert_eq!(
            (
                [
                    observed_with_branch_owner,
                    observed_after_branch_release,
                    observed_after_final_release,
                ],
                observed_after_retainer_drop,
            ),
            (
                [
                    expected_with_branch_owner,
                    expected_after_branch_release,
                    expected_after_final_release,
                ],
                [None; 9],
            ),
            "one terminal claim should activate each immediate branch once and give the shared source broadcast entity two claims (and the raw source one claim); an extra owner on an already-active branch must not recursively reacquire the source, and both release paths must balance exactly"
        );
    }

    #[test]
    fn dormant_retained_chain_reactivation_registration_ownership_characterization() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutputVec<usize>>();
        let source_executions = Arc::new(Mutex::new(0usize));
        let retained = signal::from_system(clone!((source_executions) move |In(_)| {
            let mut executions = source_executions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *executions += 1;
            *executions
        }))
        .map(capture_output_vec::<usize>)
        .broadcast();

        let first = retained.clone().register(app.world_mut());
        let terminal = first.entity().entity();
        let mapped = single_static_dependency(app.world_mut(), terminal);
        let source = single_static_dependency(app.world_mut(), mapped);
        app.update();
        let first_output = get_and_clear_output_vec::<usize>(app.world_mut());
        first.release(app.world_mut());

        let first_dormant_source = registration_observation(app.world_mut(), source);
        let first_dormant_mapped = registration_observation(app.world_mut(), mapped);
        let first_dormant_terminal = registration_observation(app.world_mut(), terminal);
        app.update();
        app.update();
        let executions_after_first_dormancy = *source_executions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dormant_output = get_and_clear_output_vec::<usize>(app.world_mut());

        let second = retained.clone().register(app.world_mut());
        assert_eq!(second.entity().entity(), terminal);
        assert_eq!(single_static_dependency(app.world_mut(), terminal), mapped);
        assert_eq!(single_static_dependency(app.world_mut(), mapped), source);
        let second_active_source = registration_observation(app.world_mut(), source);
        let second_active_mapped = registration_observation(app.world_mut(), mapped);
        let second_active_terminal = registration_observation(app.world_mut(), terminal);
        app.update();
        let second_output = get_and_clear_output_vec::<usize>(app.world_mut());
        second.release(app.world_mut());

        let second_dormant_source = registration_observation(app.world_mut(), source);
        let second_dormant_mapped = registration_observation(app.world_mut(), mapped);
        let second_dormant_terminal = registration_observation(app.world_mut(), terminal);
        app.update();
        let executions_after_second_dormancy = *source_executions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let third = retained.clone().register(app.world_mut());
        let third_active_source = registration_observation(app.world_mut(), source);
        let third_active_mapped = registration_observation(app.world_mut(), mapped);
        let third_active_terminal = registration_observation(app.world_mut(), terminal);
        app.update();
        let third_output = get_and_clear_output_vec::<usize>(app.world_mut());
        third.release(app.world_mut());

        for (source_state, mapped_state, terminal_state) in [
            (&first_dormant_source, &first_dormant_mapped, &first_dormant_terminal),
            (&second_dormant_source, &second_dormant_mapped, &second_dormant_terminal),
        ] {
            assert_eq!(source_state.count, Some(0));
            assert!(source_state.active_downstreams.is_empty());
            assert_eq!(source_state.pending_input_count, Some(0));
            assert_eq!(mapped_state.count, Some(0));
            assert!(mapped_state.active_downstreams.is_empty());
            assert_eq!(mapped_state.pending_input_count, Some(0));
            assert_eq!(terminal_state.count, Some(0));
            assert_eq!(terminal_state.pending_input_count, Some(0));
        }
        for (source_state, mapped_state, terminal_state) in [
            (&second_active_source, &second_active_mapped, &second_active_terminal),
            (&third_active_source, &third_active_mapped, &third_active_terminal),
        ] {
            assert_eq!(source_state.count, Some(1));
            assert_eq!(source_state.active_downstreams, vec![mapped]);
            assert_eq!(mapped_state.count, Some(1));
            assert_eq!(mapped_state.static_dependencies, vec![source]);
            assert_eq!(mapped_state.active_downstreams, vec![terminal]);
            assert_eq!(mapped_state.pending_input_count, Some(0));
            assert_eq!(terminal_state.count, Some(1));
            assert_eq!(terminal_state.static_dependencies, vec![mapped]);
            assert_eq!(terminal_state.pending_input_count, Some(0));
        }

        assert_eq!(
            (
                first_output,
                executions_after_first_dormancy,
                dormant_output,
                second_output,
                executions_after_second_dormancy,
                third_output,
            ),
            (vec![1], 1, Vec::<usize>::new(), vec![2], 2, vec![3]),
            "a retained count-zero chain must be dormant: no source execution or stale input is allowed between releases, and every reactivation must rebuild exactly one route without count or route drift"
        );
    }

    #[test]
    fn dynamic_stable_outputs_normalize_cache_identity_registration_ownership_observability() {
        fn assert_replayable_scalar<T: ReplayableSignal>(_signal: &T) {}
        fn assert_replayable_vec<T: crate::signal_vec::ReplayableSignalVec>(_signal: &T) {}
        fn assert_replayable_map<T: crate::signal_map::ReplayableSignalMap>(_signal: &T) {}

        let mut app = create_test_app();
        let scalar_inner = signal::always(7).dedupe().broadcast();
        let scalar = signal::always(true)
            .dedupe()
            .switch(move |In(_)| scalar_inner.clone())
            .broadcast();
        assert_replayable_scalar(&scalar);
        let scalar_claim = scalar.clone().register(app.world_mut());
        let scalar_entity_count = app.world_mut().query::<Entity>().iter(app.world()).count();
        let scalar_cached_claim = scalar.cache().register(app.world_mut());
        assert_eq!(scalar_cached_claim.entity(), scalar_claim.entity());
        assert_eq!(
            app.world_mut().query::<Entity>().iter(app.world()).count(),
            scalar_entity_count
        );

        let vector_source = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
        let vector = signal::always(true)
            .dedupe()
            .switch_signal_vec(move |In(_)| vector_source.signal_vec())
            .broadcast();
        assert_replayable_vec(&vector);
        let vector_claim = vector.clone().register(app.world_mut());
        let vector_entity_count = app.world_mut().query::<Entity>().iter(app.world()).count();
        let vector_cached_claim = vector.cache().register(app.world_mut());
        assert_eq!(vector_cached_claim.entity(), vector_claim.entity());
        assert_eq!(
            app.world_mut().query::<Entity>().iter(app.world()).count(),
            vector_entity_count
        );

        let map_source = MutableBTreeMap::builder().values([(1, 2)]).spawn(app.world_mut());
        let map = signal::always(true)
            .dedupe()
            .switch_signal_map(move |In(_)| map_source.signal_map())
            .broadcast();
        assert_replayable_map(&map);
        let map_claim = map.clone().register(app.world_mut());
        let map_entity_count = app.world_mut().query::<Entity>().iter(app.world()).count();
        let map_cached_claim = map.cache().register(app.world_mut());
        assert_eq!(map_cached_claim.entity(), map_claim.entity());
        assert_eq!(
            app.world_mut().query::<Entity>().iter(app.world()).count(),
            map_entity_count
        );

        scalar_cached_claim.release(app.world_mut());
        scalar_claim.release(app.world_mut());
        vector_cached_claim.release(app.world_mut());
        vector_claim.release(app.world_mut());
        map_cached_claim.release(app.world_mut());
        map_claim.release(app.world_mut());
    }

    #[test]
    fn switch_signal_vec_final_dynamic_release_registration_ownership_characterization() {
        #[derive(Resource, Clone, Copy, PartialEq, Eq)]
        enum Selector {
            A,
            B,
        }

        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();
        app.insert_resource(Selector::A);

        let list_a = MutableVec::builder().values([10]).spawn(app.world_mut());
        let list_b = MutableVec::builder().values([20]).spawn(app.world_mut());
        let signal_a = list_a.signal_vec().map_in(identity).cache().broadcast();
        let signal_b = list_b.signal_vec().map_in(identity).cache().broadcast();

        let signal_a_probe = signal_a.clone().register(app.world_mut());
        let signal_a_entity = signal_a_probe.entity().entity();
        let signal_b_probe = signal_b.clone().register(app.world_mut());
        let signal_b_entity = signal_b_probe.entity().entity();

        let selector = signal::from_system(|In(_), selector: Res<Selector>| *selector).dedupe();
        let switched = selector.switch_signal_vec(clone!((signal_a, signal_b) move |In(selector)| {
            match selector {
                Selector::A => signal_a.clone().left_either(),
                Selector::B => signal_b.clone().right_either(),
            }
        }));
        let outer_handle = switched.for_each(capture_vec_output::<i32>).register(app.world_mut());

        app.update();
        get_and_clear_vec_output::<i32>(app.world_mut());
        let initial = (
            registration_observation(app.world_mut(), signal_a_entity),
            registration_observation(app.world_mut(), signal_b_entity),
        );
        assert_eq!(initial.0.active_downstreams.len(), 1);
        let signal_a_route = initial.0.active_downstreams.clone();

        *app.world_mut().resource_mut::<Selector>() = Selector::B;
        app.update();
        get_and_clear_vec_output::<i32>(app.world_mut());
        let replaced = (
            registration_observation(app.world_mut(), signal_a_entity),
            registration_observation(app.world_mut(), signal_b_entity),
        );
        assert_eq!(replaced.1.active_downstreams.len(), 1);
        let signal_b_route = replaced.1.active_downstreams.clone();

        outer_handle.release(app.world_mut());
        let cleaned = (
            registration_observation(app.world_mut(), signal_a_entity),
            registration_observation(app.world_mut(), signal_b_entity),
        );

        list_a.write(app.world_mut()).push(11);
        app.update();
        let superseded_output = get_and_clear_vec_output::<i32>(app.world_mut());

        signal_a_probe.release(app.world_mut());
        signal_b_probe.release(app.world_mut());

        let inactive_a = RegistrationObservation {
            exists: true,
            count: Some(2),
            static_dependencies: initial.0.static_dependencies.clone(),
            active_downstreams: Vec::new(),
            pending_input_count: Some(0),
        };
        let inactive_b = RegistrationObservation {
            exists: true,
            count: Some(2),
            static_dependencies: initial.1.static_dependencies.clone(),
            active_downstreams: Vec::new(),
            pending_input_count: Some(0),
        };
        let active_a = RegistrationObservation {
            exists: true,
            count: Some(3),
            static_dependencies: initial.0.static_dependencies.clone(),
            active_downstreams: signal_a_route,
            pending_input_count: Some(0),
        };
        let active_b = RegistrationObservation {
            exists: true,
            count: Some(3),
            static_dependencies: replaced.1.static_dependencies.clone(),
            active_downstreams: signal_b_route,
            pending_input_count: Some(0),
        };

        assert_eq!(
            (initial, replaced, cleaned, superseded_output),
            (
                (active_a, inactive_b.clone()),
                (inactive_a.clone(), active_b),
                (inactive_a, inactive_b),
                Vec::new(),
            ),
            "switch_signal_vec must release its identity-probe registration, replace the selected forwarder, and explicitly clean the final Local-owned forwarder; a superseded retained source must remain silent and final cleanup must match no subscription"
        );
    }

    #[test]
    fn scalar_flatten_dynamic_release_registration_ownership_observability() {
        #[derive(Resource, Clone, Copy, PartialEq, Eq)]
        enum Selector {
            A,
            B,
        }

        let mut app = create_test_app();
        app.insert_resource(Selector::A);
        let inner_a = signal::from_system(|In(_)| 10).dedupe().broadcast();
        let inner_b = signal::from_system(|In(_)| 20).dedupe().broadcast();
        let inner_a_handle = inner_a.clone().register(app.world_mut());
        let inner_a_entity = inner_a_handle.entity().entity();
        inner_a_handle.release(app.world_mut());
        let inner_b_handle = inner_b.clone().register(app.world_mut());
        let inner_b_entity = inner_b_handle.entity().entity();
        inner_b_handle.release(app.world_mut());

        let flattened = signal::from_system(|In(_), selector: Res<Selector>| *selector)
            .dedupe()
            .switch(clone!((inner_a, inner_b) move |In(selector)| match selector {
                Selector::A => inner_a.clone().left_either(),
                Selector::B => inner_b.clone().right_either(),
            }));
        let outer_handle = flattened.register(app.world_mut());
        app.update();
        let initial = (
            registration_observation(app.world_mut(), inner_a_entity),
            registration_observation(app.world_mut(), inner_b_entity),
        );
        assert_eq!(initial.0.count, Some(2));
        assert_eq!(initial.0.active_downstreams.len(), 1);
        assert_eq!(initial.1.count, Some(1));

        *app.world_mut().resource_mut::<Selector>() = Selector::B;
        app.update();
        let replaced = (
            registration_observation(app.world_mut(), inner_a_entity),
            registration_observation(app.world_mut(), inner_b_entity),
        );
        assert_eq!(replaced.0.count, Some(1));
        assert!(replaced.0.active_downstreams.is_empty());
        assert_eq!(replaced.1.count, Some(2));
        assert_eq!(replaced.1.active_downstreams.len(), 1);

        outer_handle.release(app.world_mut());
        let cleaned = (
            registration_observation(app.world_mut(), inner_a_entity),
            registration_observation(app.world_mut(), inner_b_entity),
        );
        assert_eq!(cleaned.0.count, Some(1));
        assert!(cleaned.0.active_downstreams.is_empty());
        assert_eq!(cleaned.1.count, Some(1));
        assert!(cleaned.1.active_downstreams.is_empty());
    }

    #[test]
    fn test_signal_eq() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let s1 = signal::from_system(|In(_)| 1).broadcast();
        let s2 = signal::from_system(|In(_)| 1).broadcast();
        let s3 = signal::from_system(|In(_)| 1);
        let s4 = signal::from_system(|In(_)| 2);

        // Test equality
        let eq_signal = crate::signal::eq!(s1.clone(), s2.clone(), s3);
        let equality_claim = eq_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));
        equality_claim.release(app.world_mut());

        // Test inequality
        let neq_signal = crate::signal::eq!(s1, s2, s4);
        let inequality_claim = neq_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));
        inequality_claim.release(app.world_mut());
    }

    #[test]
    fn test_signal_and() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let t = signal::from_system(|In(_)| true).broadcast();
        let f = signal::from_system(|In(_)| false);

        // Test AND (all true)
        let all_signal = crate::signal::all!(t.clone(), t.clone(), t.clone());
        let all_true_claim = all_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));
        all_true_claim.release(app.world_mut());

        // Test AND (one false)
        let all_signal = crate::signal::all!(t.clone(), f, t);
        let one_false_claim = all_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));
        one_false_claim.release(app.world_mut());
    }

    #[test]
    fn test_signal_or() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let t = signal::from_system(|In(_)| true);
        let f = signal::from_system(|In(_)| false).broadcast();

        // Test OR (all false)
        let any_signal = crate::signal::any!(f.clone(), f.clone(), f.clone());
        let all_false_claim = any_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));
        all_false_claim.release(app.world_mut());

        // Test OR (one true)
        let any_signal = crate::signal::any!(f.clone(), t, f);
        let one_true_claim = any_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));
        one_true_claim.release(app.world_mut());
    }

    #[test]
    fn test_signal_distinct() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 3);

        // Test distinct (all different)
        let distinct_signal = crate::signal::distinct!(s1, s2, s3);
        let _all_distinct_claim = distinct_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));

        // Test distinct (some equal)
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let s1 = signal::from_system(|In(_)| 1).broadcast();
        let s2 = signal::from_system(|In(_)| 2);

        let distinct_signal = crate::signal::distinct!(s1.clone(), s2, s1);
        let _repeated_claim = distinct_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));
    }

    #[test]
    fn test_signal_sum() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);

        // Test sum with 2 signals
        let sum_signal = crate::signal::sum!(s1, s2);
        let _two_signal_claim = sum_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(3));

        // Test sum with 4 signals
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 3);
        let s4 = signal::from_system(|In(_)| 4);

        let sum_signal = crate::signal::sum!(s1, s2, s3, s4);
        let _four_signal_claim = sum_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(10));
    }

    #[test]
    fn test_signal_product() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 2);
        let s2 = signal::from_system(|In(_)| 3);

        // Test product with 2 signals
        let product_signal = crate::signal::product!(s1, s2);
        let _two_signal_claim = product_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(6));

        // Test product with 4 signals
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 3);
        let s4 = signal::from_system(|In(_)| 4);

        let product_signal = crate::signal::product!(s1, s2, s3, s4);
        let _four_signal_claim = product_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(24));
    }

    #[test]
    fn test_signal_min() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 5);
        let s2 = signal::from_system(|In(_)| 2);

        // Test min with 2 signals
        let min_signal = crate::signal::min!(s1, s2);
        let _two_signal_claim = min_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(2));

        // Test min with 4 signals
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 10);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 7);
        let s4 = signal::from_system(|In(_)| 3);

        let min_signal = crate::signal::min!(s1, s2, s3, s4);
        let _four_signal_claim = min_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(2));
    }

    #[test]
    fn test_signal_max() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 5);
        let s2 = signal::from_system(|In(_)| 2);

        // Test max with 2 signals
        let max_signal = crate::signal::max!(s1, s2);
        let _two_signal_claim = max_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(5));

        // Test max with 4 signals
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let s1 = signal::from_system(|In(_)| 10);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 7);
        let s4 = signal::from_system(|In(_)| 3);

        let max_signal = crate::signal::max!(s1, s2, s3, s4);
        let _four_signal_claim = max_signal.map(capture_output::<i32>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(10));
    }

    #[test]
    fn test_map_bool_in() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<&'static str>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<bool>| {
                *state = !*state;
                *state
            }
        })
        .map_bool_in(|| "true", || "false")
        .map(capture_output::<&'static str>);

        let _claim = signal.register(app.world_mut());

        // First update: state toggles to true
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<&'static str>>().0, Some("true"));

        // Second update: state toggles to false
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<&'static str>>().0, Some("false"));

        // Third update: state toggles to true
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<&'static str>>().0, Some("true"));
    }

    #[test]
    fn test_map_bool_signal_initial_selection() {
        fn assert_initial_selection(selector: bool, expected: &'static str) {
            let mut app = create_test_app();
            app.init_resource::<SignalOutput<&'static str>>();

            let signal = signal::once(selector)
                .map_bool_signal(|| signal::always("true"), || signal::once("false"))
                .map(capture_output::<&'static str>)
                .register(app.world_mut());

            app.update();
            assert_eq!(get_output::<&'static str>(app.world()), Some(expected));
            signal.release(app.world_mut());
        }

        assert_initial_selection(false, "false");
        assert_initial_selection(true, "true");
    }

    #[test]
    fn test_map_bool_signal_branch_updates_and_current_value_replay() {
        #[derive(Resource, Clone, PartialEq)]
        struct Selector(bool);

        #[derive(Resource, Clone, PartialEq)]
        struct TrueBranchValue(i32);

        #[derive(Resource, Clone, PartialEq)]
        struct FalseBranchValue(i32);

        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        app.insert_resource(Selector(false));
        app.insert_resource(TrueBranchValue(10));
        app.insert_resource(FalseBranchValue(20));

        let signal = signal::from_resource::<Selector>()
            .dedupe()
            .map_in(|selector| selector.0)
            .map_bool_signal(
                || signal::from_resource::<TrueBranchValue>().map_in(|value| value.0),
                || {
                    signal::from_resource::<FalseBranchValue>()
                        .dedupe()
                        .map_in(|value| value.0)
                },
            )
            .map(capture_output::<i32>)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(20));

        app.world_mut().resource_mut::<FalseBranchValue>().0 = 21;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(21));

        app.world_mut().resource_mut::<TrueBranchValue>().0 = 11;
        app.world_mut().resource_mut::<Selector>().0 = true;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(11));

        app.world_mut().resource_mut::<TrueBranchValue>().0 = 12;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(12));

        app.world_mut().resource_mut::<FalseBranchValue>().0 = 22;
        app.world_mut().resource_mut::<Selector>().0 = false;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(22));

        signal.release(app.world_mut());
    }

    #[test]
    fn test_map_bool_signal_reused_branches_cleanup_with_flattened_output() {
        #[derive(Resource, Clone, PartialEq)]
        struct Selector(bool);

        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        app.insert_resource(Selector(false));

        let true_signal = signal::from_system(|In(_)| 1).broadcast();
        let false_signal = signal::from_system(|In(_)| 2).broadcast();
        let switched = signal::from_resource::<Selector>()
            .dedupe()
            .map_in(|selector| selector.0)
            .map_bool_signal(move || true_signal.clone(), move || false_signal.clone())
            .map(capture_output::<i32>)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));

        app.world_mut().resource_mut::<Selector>().0 = true;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(1));

        app.world_mut().resource_mut::<Selector>().0 = false;
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(2));

        switched.release(app.world_mut());
        app.update();
    }

    #[test]
    fn test_map_bool_in_with_optional_termination() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<bool>| {
                *state = !*state;
                *state
            }
        })
        .map_bool_in(
            || Some(1), // true returns Some(1)
            || None,    // false terminates
        )
        .map(capture_output::<i32>);

        let _claim = signal.register(app.world_mut());

        // First update: state toggles to true, outputs 1
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(1));

        // Second update: state toggles to false, terminates (None)
        app.world_mut().resource_mut::<SignalOutput<i32>>().0 = None;
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, None);

        // Third update: state toggles to true, outputs 1 again
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(1));
    }

    #[test]
    fn test_map_true_in() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<Option<i32>>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<bool>| {
                *state = !*state;
                *state
            }
        })
        .map_true_in(|| 42);

        let _claim = signal.map(capture_output::<Option<i32>>).register(app.world_mut());

        // First update: state toggles to true, outputs Some(42)
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(Some(42)));

        // Second update: state toggles to false, outputs None
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(None));

        // Third update: state toggles to true, outputs Some(42) again
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(Some(42)));
    }

    #[test]
    fn test_map_false_in() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<Option<i32>>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<bool>| {
                *state = !*state;
                *state
            }
        })
        .map_false_in(|| 99);

        let _claim = signal.map(capture_output::<Option<i32>>).register(app.world_mut());

        // First update: state toggles to true, outputs None
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(None));

        // Second update: state toggles to false, outputs Some(99)
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(Some(99)));

        // Third update: state toggles to true, outputs None
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(None));
    }

    #[test]
    fn test_map_option_in() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<i32>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<Option<i32>>| {
                *state = match *state {
                    None => Some(10),
                    Some(10) => Some(20),
                    Some(20) => None,
                    Some(_) => Some(10),
                };
                *state
            }
        })
        .map_option_in(|value: i32| value * 2, || -1);

        let _claim = signal.map(capture_output::<i32>).register(app.world_mut());

        // First update: state is Some(10), outputs 20
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(20));

        // Second update: state is Some(20), outputs 40
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(40));

        // Third update: state is None, outputs -1
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(-1));

        // Fourth update: state is Some(10), outputs 20
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<i32>>().0, Some(20));
    }

    #[test]
    fn test_map_some_in() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<Option<i32>>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<Option<i32>>| {
                *state = if state.is_some() { None } else { Some(42) };
                *state
            }
        })
        .map_some_in(|value: i32| value * 3);

        let _claim = signal.map(capture_output::<Option<i32>>).register(app.world_mut());

        // First update: state is Some(42), outputs Some(126)
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(Some(126)));

        // Second update: state is None, outputs None
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(None));

        // Third update: state is Some(42), outputs Some(126) again
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(Some(126)));
    }

    #[test]
    fn test_map_none_in() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<Option<i32>>::default());

        let signal = signal::from_system({
            move |In(_), mut state: Local<Option<i32>>| {
                *state = if state.is_some() { None } else { Some(100) };
                *state
            }
        })
        .map_none_in(|| 999);

        let _claim = signal.map(capture_output::<Option<i32>>).register(app.world_mut());

        // First update: state is Some(100), outputs None
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(None));

        // Second update: state is None, outputs Some(999)
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(Some(999)));

        // Third update: state is Some(100), outputs None
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<Option<i32>>>().0, Some(None));
    }

    #[test]
    fn multi_schedule_chain_tags_correctly() {
        use crate::graph::ScheduleTag;
        use bevy_app::{PostUpdate, Update};
        use bevy_ecs::schedule::ScheduleLabel;

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            crate::JonmoPlugin::new::<PostUpdate>().with_schedule::<Update>(),
        ));

        // Build a chain: source -> map1 -> schedule::<Update> -> map2 -> schedule::<PostUpdate> ->
        // map3 Note: Scheduled doesn't create its own entity, it just tags its inner signal.
        // So the actual entities are: source, map1 (tagged Update), map2 (tagged PostUpdate), map3
        //
        // Expected tags:
        //   source: Update (tagged by schedule::<Update>'s upstream propagation)
        //   map1: Update (caller of schedule::<Update>)
        //   map2: PostUpdate (caller of schedule::<PostUpdate>, overwrites inherited Update)
        //   map3: PostUpdate (inherits from map2's hint)

        let handle = signal::from_system(|In(_)| Some(1i32))
            .map_in(|x: i32| x + 1)
            .schedule::<Update>()
            .map_in(|x: i32| x * 2)
            .schedule::<PostUpdate>()
            .map_in(|x: i32| x - 1)
            .register(app.world_mut());

        // The handle points to map3 (the outermost signal)
        let map3_entity = handle.entity().entity();

        let map2_entity = single_static_dependency(app.world_mut(), map3_entity);
        let map1_entity = single_static_dependency(app.world_mut(), map2_entity);
        let source_entity = single_static_dependency(app.world_mut(), map1_entity);
        let world = app.world();

        // Verify schedule tags
        let source_tag = world.get::<ScheduleTag>(source_entity);
        assert!(source_tag.is_some(), "source should have a schedule tag");
        assert_eq!(source_tag.unwrap().0, Update.intern(), "source should be tagged Update");

        let map1_tag = world.get::<ScheduleTag>(map1_entity);
        assert!(map1_tag.is_some(), "map1 should have a schedule tag");
        assert_eq!(
            map1_tag.unwrap().0,
            Update.intern(),
            "map1 should be tagged Update (caller of schedule::<Update>)"
        );

        let map2_tag = world.get::<ScheduleTag>(map2_entity);
        assert!(map2_tag.is_some(), "map2 should have a schedule tag");
        assert_eq!(
            map2_tag.unwrap().0,
            PostUpdate.intern(),
            "map2 should be tagged PostUpdate (caller of schedule::<PostUpdate>)"
        );

        let map3_tag = world.get::<ScheduleTag>(map3_entity);
        assert!(map3_tag.is_some(), "map3 should have a schedule tag");
        assert_eq!(
            map3_tag.unwrap().0,
            PostUpdate.intern(),
            "map3 should inherit PostUpdate"
        );
    }

    #[test]
    fn test_boxed_replayable_inherent_methods_preserve_replayable_with_prelude() {
        use crate::prelude::*;

        fn assert_replayable<S: ReplayableSignal<Item = i32>>(signal: S) -> S {
            signal
        }

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin::new::<Last>().with_schedule::<Update>()));
        app.init_resource::<SignalOutput<i32>>();

        let cached = assert_replayable(signal::always(10i32).cache().boxed_replayable().cache());
        let broadcast = assert_replayable(signal::always(20i32).cache().boxed_replayable().broadcast());
        let scheduled = assert_replayable(signal::always(30i32).cache().boxed_replayable().schedule::<Update>());
        let reboxed = assert_replayable(cached.boxed_replayable());

        let claim_a = broadcast.clone().map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(20));
        claim_a.release(app.world_mut());

        let claim_b = scheduled.map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(30));
        claim_b.release(app.world_mut());

        let claim_c = reboxed.map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(10));
        claim_c.release(app.world_mut());
    }

    #[test]
    fn test_system_adapters_propagate_skipped_validation() {
        #[derive(Resource)]
        struct Factor(i32);

        let mut app = create_test_app();
        let hits = Arc::new(Mutex::new(Vec::<Option<i32>>::new()));

        let make_recorder = |hits: &Arc<Mutex<Vec<Option<i32>>>>| {
            let hits = Arc::clone(hits);
            move |In(val): In<Option<i32>>| {
                hits.lock().unwrap().push(val);
            }
        };

        use bevy_ecs::system::If;

        // Without Factor resource present, all 7 adapters/combinators must skip propagation
        // rather than emitting `None` / `Some(None)` downstream.
        let c_filter = signal::always(5i32)
            .filter(|In(x): In<i32>, If(f): If<Res<Factor>>| x > f.0)
            .map({
                let hits = Arc::clone(&hits);
                move |In(v): In<i32>| hits.lock().unwrap().push(Some(v))
            })
            .register(app.world_mut());
        let c_map_bool = signal::always(true)
            .map_bool(|In(()), If(f): If<Res<Factor>>| f.0, |In(())| 0)
            .map({
                let hits = Arc::clone(&hits);
                move |In(v): In<i32>| hits.lock().unwrap().push(Some(v))
            })
            .register(app.world_mut());
        let c_map_true = signal::always(true)
            .map_true(|In(()), If(f): If<Res<Factor>>| f.0)
            .map(make_recorder(&hits))
            .register(app.world_mut());
        let c_map_false = signal::always(false)
            .map_false(|In(()), If(f): If<Res<Factor>>| f.0)
            .map(make_recorder(&hits))
            .register(app.world_mut());
        let c_map_option = signal::always(Some(2i32))
            .map_option(|In(x): In<i32>, If(f): If<Res<Factor>>| x * f.0, |In(())| 0)
            .map({
                let hits = Arc::clone(&hits);
                move |In(v): In<i32>| hits.lock().unwrap().push(Some(v))
            })
            .register(app.world_mut());
        let c_map_some = signal::always(Some(3i32))
            .map_some(|In(x): In<i32>, If(f): If<Res<Factor>>| x * f.0)
            .map(make_recorder(&hits))
            .register(app.world_mut());
        let c_map_none = signal::always(None::<i32>)
            .map_none(|In(()), If(f): If<Res<Factor>>| f.0)
            .map(make_recorder(&hits))
            .register(app.world_mut());

        app.update();
        assert!(
            hits.lock().unwrap().is_empty(),
            "Skipped system validation must terminate propagation instead of emitting None: {:?}",
            *hits.lock().unwrap()
        );

        app.insert_resource(Factor(3));
        app.update();
        assert_eq!(hits.lock().unwrap().len(), 7);

        c_filter.release(app.world_mut());
        c_map_bool.release(app.world_mut());
        c_map_true.release(app.world_mut());
        c_map_false.release(app.world_mut());
        c_map_option.release(app.world_mut());
        c_map_some.release(app.world_mut());
        c_map_none.release(app.world_mut());
    }
}
