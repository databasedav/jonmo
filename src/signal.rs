//! Signal builders and combinators for constructing reactive [`System`] dependency graphs.
//!
//! This module provides the core [`Signal`] trait and its extension trait [`SignalExt`], which
//! together form the foundation for declarative, push-based reactive dataflows in [jonmo](crate).
//! For a high-level overview of the signal graph runtime and key concepts, see the [`Signal`] trait
//! documentation.
use super::{
    graph::{
        LazySignal, SignalHandle, SignalHandles, SignalSystem, Upstream, UpstreamIter, apply_schedule_to_signal,
        downcast_any_clone, lazy_signal_from_system, pipe_signal, poll_signal, trigger_signal_subgraph,
    },
    signal_map::{SignalMap, SignalMapExt},
    signal_vec::{ReplayOnce, SignalVec, SignalVecExt, VecDiff, trigger_replay},
    utils::{LazyEntity, ancestor_map},
};
use crate::prelude::clone;
use bevy_ecs::{entity_disabling::Internal, prelude::*, schedule::ScheduleLabel, system::SystemState};
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
use dyn_clone::{DynClone, clone_trait_object};

/// A composable node in [jonmo](crate)'s reactive dependency graph, backed by a Bevy [`System`].
///
/// # Overview
///
/// A [`Signal`] represents a node in [jonmo](crate)'s reactive dependency graph. Under the hood,
/// each signal node is backed by a Bevy [`System`] whose output is forwarded to its downstream
/// dependents. Signals are declarative *descriptions* of dataflow; they don't execute until
/// registered into a [`World`] via [`SignalExt::register`], at which point the underlying systems
/// are inserted into the ECS and wired into the graph.
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
/// # Polling
///
/// While signals are push-based by default, [jonmo](crate) also provides a polling API for cases
/// where pull-based access is needed. Polling allows you to synchronously query a signal's most
/// recent output from within another system, providing an escape hatch from the standard push
/// semantics, see [`poll_signal`].
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
/// - [`SignalDynClone`]: for signals that need to be cloneable in type-erased contexts
pub trait Signal: Send + Sync + 'static {
    /// Output type.
    type Item;

    /// Registers the [`System`]s associated with this [`Signal`] by consuming its boxed form.
    ///
    /// All concrete signal types must implement this method.
    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle;

    /// Registers the [`System`]s associated with this [`Signal`].
    fn register_signal(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.boxed().register_boxed_signal(world)
    }
}

impl<U: 'static> Signal for Box<dyn Signal<Item = U> + Send + Sync> {
    type Item = U;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        (*self).register_boxed_signal(world)
    }
}

/// An extension trait for [`Signal`] types that implement [`Clone`].
///
/// Relevant in contexts where some function may require a [`Clone`] [`Signal`], but the concrete
/// type can't be known at compile-time.
pub trait SignalDynClone: Signal + DynClone {}

clone_trait_object!(<T> SignalDynClone<Item = T>);

impl<T: Signal + Clone + 'static> SignalDynClone for T {}

impl<O: 'static> Signal for Box<dyn SignalDynClone<Item = O> + Send + Sync> {
    type Item = O;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        (*self).register_boxed_signal(world)
    }
}

/// A type-erased, cloneable [`Signal`].
///
/// This is a convenience alias for `Box<dyn SignalDynClone<Item = T> + Send + Sync>`.
/// Useful when you need to store signals of different concrete types in the same collection,
/// or return them from branching logic.
pub type BoxedSignal<T> = Box<dyn SignalDynClone<Item = T> + Send + Sync>;

/// Signal graph node which takes an input of [`In<()>`] and has no upstreams. See
/// [`from_system`] for examples.
pub struct Source<O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
}

impl<O> Clone for Source<O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<O> Signal for Source<O>
where
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        SignalHandle::new(self.signal.register(world))
    }
}

/// Signal graph node which applies a [`System`] to its upstream, see [`.map`](SignalExt::map).
pub struct Map<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
}

impl<Upstream, O> Clone for Map<Upstream, O>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            upstream: self.upstream.clone(),
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for Map<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        let SignalHandle(upstream) = self.upstream.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, upstream, signal);
        signal.into()
    }
}

/// Signal graph node wrapper that assigns a schedule to a signal chain, see
/// [`.schedule`](SignalExt::schedule).
pub struct Scheduled<Sched, I> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Sched, I)>,
}

impl<Sched, I> Clone for Scheduled<Sched, I> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Sched: 'static, I: 'static> Signal for Scheduled<Sched, I> {
    type Item = I;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which maps its upstream [`Entity`] to a corresponding [`Component`], see
/// [.component](SignalExt::component).
pub struct MapComponent<Upstream, C> {
    signal: Map<Upstream, C>,
}

impl<Upstream, C> Clone for MapComponent<Upstream, C>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream, C> Signal for MapComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = C;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which maps its upstream [`Entity`] to a corresponding [`Option<Component>`],
/// see [.component_option](SignalExt::component_option).
pub struct ComponentOption<Upstream, C> {
    signal: Map<Upstream, Option<C>>,
}

impl<Upstream, C> Clone for ComponentOption<Upstream, C>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream, C> Signal for ComponentOption<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = Option<C>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which maps its upstream [`Entity`] to its [`Changed`] `C` [`Component`],
/// see [.component_changed](SignalExt::component_changed).
pub struct ComponentChanged<Upstream, C> {
    signal: Map<Upstream, C>,
}

impl<Upstream, C> Clone for ComponentChanged<Upstream, C>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream, C> Signal for ComponentChanged<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = C;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node with maps its upstream [`Entity`] to whether it has some [`Component`], see
/// [`.has_component`](SignalExt::has_component).
pub struct HasComponent<Upstream, C> {
    signal: Map<Upstream, bool>,
    _marker: PhantomData<fn() -> C>,
}

impl<Upstream, C> Clone for HasComponent<Upstream, C>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, C> Signal for HasComponent<Upstream, C>
where
    Upstream: Signal<Item = Entity>,
    C: Component,
{
    type Item = bool;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which does not forward upstream duplicates, see
/// [`.dedupe`](SignalExt::dedupe).
pub struct Dedupe<Upstream>
where
    Upstream: Signal,
{
    signal: Map<Upstream, Upstream::Item>,
}

impl<Upstream> Clone for Dedupe<Upstream>
where
    Upstream: Signal + Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for Dedupe<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node that terminates forever after forwarding a single upstream value, see
/// [`.first`](SignalExt::first).
pub struct First<Upstream>
where
    Upstream: Signal,
{
    signal: Take<Upstream>,
}

impl<Upstream> Clone for First<Upstream>
where
    Upstream: Signal + Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for First<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
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

impl<Upstream> Clone for Take<Upstream>
where
    Upstream: Signal + Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for Take<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node that skips the first `count` upstream values before forwarding, see
/// [`.skip`](SignalExt::skip).
pub struct Skip<Upstream>
where
    Upstream: Signal,
{
    signal: Filter<Upstream>,
}

impl<Upstream> Clone for Skip<Upstream>
where
    Upstream: Signal + Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for Skip<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which combines two upstreams, see [`.zip`](SignalExt::zip).
#[allow(clippy::type_complexity)]
pub struct Zip<Left, Right>
where
    Left: Signal,
    Right: Signal,
{
    left_wrapper: Map<Left, (Option<Left::Item>, Option<Right::Item>)>,
    right_wrapper: Map<Right, (Option<Left::Item>, Option<Right::Item>)>,
    signal: LazySignal,
}

impl<Left, Right> Clone for Zip<Left, Right>
where
    Left: Signal + Clone,
    Right: Signal + Clone,
{
    fn clone(&self) -> Self {
        Self {
            left_wrapper: self.left_wrapper.clone(),
            right_wrapper: self.right_wrapper.clone(),
            signal: self.signal.clone(),
        }
    }
}

impl<Left, Right> Signal for Zip<Left, Right>
where
    Left: Signal,
    Right: Signal,
{
    type Item = (Left::Item, Right::Item);

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        let signal = self.signal.register(world);
        let SignalHandle(left_upstream) = self.left_wrapper.register(world);
        let SignalHandle(right_upstream) = self.right_wrapper.register(world);
        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);
        SignalHandle(signal)
    }
}

/// Signal graph node which outputs equality between its upstream and a fixed value, see
/// [`.eq`](SignalExt::eq).
pub struct Eq<Upstream> {
    signal: Map<Upstream, bool>,
}

impl<Upstream> Clone for Eq<Upstream>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for Eq<Upstream>
where
    Upstream: Signal,
{
    type Item = bool;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which outputs inequality between its upstream and a fixed value, see
/// [`.neq`](SignalExt::neq).
pub struct Neq<Upstream> {
    signal: Map<Upstream, bool>,
}

impl<Upstream> Clone for Neq<Upstream>
where
    Upstream: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for Neq<Upstream>
where
    Upstream: Signal,
    Upstream::Item: PartialEq,
{
    type Item = bool;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which applies [`ops::Not`] to its upstream, see [`.not`](SignalExt::not).
pub struct Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: ops::Not,
    <<Upstream as Signal>::Item as ops::Not>::Output: Clone,
{
    signal: Map<Upstream, <Upstream::Item as ops::Not>::Output>,
}

impl<Upstream> Clone for Not<Upstream>
where
    Upstream: Signal + Clone,
    <Upstream as Signal>::Item: ops::Not,
    <<Upstream as Signal>::Item as ops::Not>::Output: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream> Signal for Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: ops::Not,
    <<Upstream as Signal>::Item as ops::Not>::Output: Clone,
{
    type Item = <Upstream::Item as ops::Not>::Output;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which selectively terminates propagation, see [`.filter`](SignalExt::filter).
pub struct Filter<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> Clone for Filter<Upstream> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream> Signal for Filter<Upstream>
where
    Upstream: Signal,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which forwards the upstream output [`Signal`]'s output, see
/// [`.flatten`](SignalExt::flatten).
pub struct Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Signal,
{
    signal: LazySignal,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<fn() -> (Upstream, Upstream::Item)>,
}

impl<Upstream> Clone for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Signal,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream> Signal for Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Signal,
    <Upstream::Item as Signal>::Item: Clone,
{
    type Item = <Upstream::Item as Signal>::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which maps its upstream to a [`Signal`], forwarding its output, see
/// [`.switch`](SignalExt::switch).
pub struct Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: Signal,
    Other::Item: Clone,
{
    signal: Flatten<Map<Upstream, Other>>,
}

impl<Upstream, Other> Clone for Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: Signal,
    Other::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
        }
    }
}

impl<Upstream, Other> Signal for Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: Signal,
    Other::Item: Clone,
{
    type Item = Other::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which maps its upstream to a [`SignalVec`], see
/// [`.switch_signal_vec`](SignalExt::switch_signal_vec).
pub struct SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalVec,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, Switched)>,
}

impl<Upstream, Switched> Clone for SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalVec,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, Switched> SignalVec for SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalVec,
    Switched::Item: Clone + Send + Sync + 'static,
{
    type Item = Switched::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which maps its upstream to a [`SignalMap`], see
/// [`.switch_signal_map`](SignalExt::switch_signal_map).
pub struct SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalMap,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, Switched)>,
}

impl<Upstream, Switched> Clone for SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalMap,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, Switched> SignalMap for SwitchSignalMap<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalMap,
    Switched::Key: Clone + Send + Sync + 'static,
    Switched::Value: Clone + Send + Sync + 'static,
{
    type Key = Switched::Key;
    type Value = Switched::Value;

    fn register_boxed_signal_map(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
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

        impl<Upstream> Clone for Throttle<Upstream>
        where
            Upstream: Signal + Clone,
        {
            fn clone(&self) -> Self {
                Self {
                    signal: self.signal.clone(),
                }
            }
        }

        impl<Upstream> Signal for Throttle<Upstream>
        where
            Upstream: Signal,
        {
            type Item = Upstream::Item;

            fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
                self.signal.register(world)
            }
        }
    }
}

/// Signal graph node whose [`System`] depends on its upstream [`bool`] value, see
/// [`.map_bool`](SignalExt::map_bool).
pub struct MapBool<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> Clone for MapBool<Upstream, O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for MapBool<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`true`], see
/// [`.map_true`](SignalExt::map_true).
pub struct MapTrue<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> Clone for MapTrue<Upstream, O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for MapTrue<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    O: 'static,
{
    type Item = Option<O>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`false`], see
/// [`.map_false`](SignalExt::map_false).
pub struct MapFalse<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> Clone for MapFalse<Upstream, O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for MapFalse<Upstream, O>
where
    Upstream: Signal<Item = bool>,
    O: 'static,
{
    type Item = Option<O>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node whose [`System`] depends on its upstream [`Option`] value, see
/// [`.map_option`](SignalExt::map_option).
pub struct MapOption<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> Clone for MapOption<Upstream, O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for MapOption<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`Some`], see
/// [`.map_some`](SignalExt::map_some).
pub struct MapSome<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> Clone for MapSome<Upstream, O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for MapSome<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = Option<O>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node whose system only runs when its upstream outputs [`None`], see
/// [`.map_none`](SignalExt::map_none).
pub struct MapNone<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> Clone for MapNone<Upstream, O> {
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, O> Signal for MapNone<Upstream, O>
where
    Upstream: Signal,
    O: 'static,
{
    type Item = Option<O>;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which maps its upstream [`Vec`] to a [`SignalVec`], see
/// [`.to_signal_vec`](SignalExt::to_signal_vec).
pub struct ToSignalVec<Upstream>
where
    Upstream: Signal,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> Clone for ToSignalVec<Upstream>
where
    Upstream: Signal,
{
    fn clone(&self) -> Self {
        Self {
            signal: self.signal.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Upstream, T> SignalVec for ToSignalVec<Upstream>
where
    Upstream: Signal<Item = Vec<T>>,
    T: 'static,
{
    type Item = T;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
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

        impl<Upstream> Clone for Debug<Upstream>
        where
            Upstream: Signal + Clone,
        {
            fn clone(&self) -> Self {
                Self {
                    signal: self.signal.clone(),
                }
            }
        }

        impl<Upstream> Signal for Debug<Upstream>
        where
            Upstream: Signal,
        {
            type Item = Upstream::Item;

            fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
                self.signal.register(world)
            }
        }
    }
}

/// Creates a [`Signal`] from a [`System`] that takes [`In<()>`].
pub fn from_system<O, IOO, F, M>(system: F) -> impl Signal<Item = O> + Clone
where
    O: Clone + Send + Sync + 'static,
    IOO: Into<Option<O>> + 'static,
    F: IntoSystem<In<()>, IOO, M> + Send + Sync + 'static,
{
    Source {
        signal: lazy_signal_from_system(system),
        _marker: PhantomData,
    }
}

/// Creates a [`Signal`] from an [`FnMut`].
pub fn from_function<O, F>(mut f: F) -> impl Signal<Item = O> + Clone
where
    O: Clone + Send + Sync + 'static,
    F: FnMut() -> O + Send + Sync + 'static,
{
    from_system(move |In(_)| f())
}

/// Creates a [`Signal`] that always outputs the same value.
pub fn always<O>(value: O) -> impl Signal<Item = O> + Clone
where
    O: Clone + Send + Sync + 'static,
{
    from_system(move |In(_)| value.clone())
}

/// Creates a [`Signal`] that outputs a value once and then terminates forever.
pub fn once<O>(value: O) -> impl Signal<Item = O> + Clone
where
    O: Clone + Send + Sync + 'static,
{
    from_system(move |In(_), mut emitted: Local<bool>| {
        if *emitted {
            None
        } else {
            *emitted = true;
            Some(value.clone())
        }
    })
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`].
pub fn from_entity(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = Entity> + Clone {
    let mut entity = Some(entity);
    from_system(move |In(_), mut cached: Local<Option<Entity>>| {
        *cached.get_or_insert_with(|| entity.take().unwrap().into())
    })
}

/// Creates a [`Signal`] from an [`Entity`]'s or [`LazyEntity`]'s `generations`-th generation
/// ancestor's [`Entity`], terminating for frames where it does not exist. Passing `0` to
/// `generation` will output the [`Entity`] itself.
pub fn from_ancestor(
    entity: impl Into<Entity> + Send + Sync + 'static,
    generations: usize,
) -> impl Signal<Item = Entity> + Clone {
    from_entity(entity).map(ancestor_map(generations))
}

/// Creates a [`Signal`] from an [`Entity`]'s or [`LazyEntity`]'s parent's [`Entity`],
/// terminating for frames where it does not exist.
pub fn from_parent(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = Entity> + Clone {
    from_ancestor(entity, 1)
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`] and a [`Component`],
/// terminating for the frame if the [`Entity`] does not exist or the [`Component`] does not exist
/// on the [`Entity`].
pub fn from_component<C>(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = C> + Clone
where
    C: Component + Clone,
{
    let mut entity = Some(entity);
    from_system(move |In(_), mut cached: Local<Option<Entity>>, components: Query<&C>| {
        let entity = *cached.get_or_insert_with(|| entity.take().unwrap().into());
        components.get(entity).ok().cloned()
    })
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`] and a [`Component`], always
/// outputting an [`Option`].
pub fn from_component_option<C>(
    entity: impl Into<Entity> + Send + Sync + 'static,
) -> impl Signal<Item = Option<C>> + Clone
where
    C: Component + Clone,
{
    let mut entity = Some(entity);
    from_system(move |In(_), mut cached: Local<Option<Entity>>, components: Query<&C>| {
        let entity = *cached.get_or_insert_with(|| entity.take().unwrap().into());
        Some(components.get(entity).ok().cloned())
    })
}

/// Creates a [`Signal`] from an [`Entity`] or [`LazyEntity`] and a [`Component`], only
/// outputting on frames where the [`Component`] has [`Changed`]. Terminates for the frame if the
/// [`Entity`] does not exist, the [`Component`] does not exist, or the [`Component`] has not
/// changed.
pub fn from_component_changed<C>(entity: impl Into<Entity> + Send + Sync + 'static) -> impl Signal<Item = C> + Clone
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
pub fn from_resource<R>() -> impl Signal<Item = R> + Clone
where
    R: Resource + Clone,
{
    from_system(move |In(_), resource: Option<Res<R>>| resource.as_deref().cloned())
}

/// Creates a signal from a [`Resource`], always outputting an [`Option`].
pub fn from_resource_option<R>() -> impl Signal<Item = Option<R>> + Clone
where
    R: Resource + Clone,
{
    from_system(move |In(_), resource: Option<Res<R>>| Some(resource.as_deref().cloned()))
}

/// Creates a signal from a [`Resource`], only outputting on frames where the [`Resource`] has
/// [`Changed`]. Terminates for the frame if the [`Resource`] does not exist or has not changed.
pub fn from_resource_changed<R>() -> impl Signal<Item = R> + Clone
where
    R: Resource + Clone,
{
    from_system(move |In(_), resource: Option<Res<R>>| resource.filter(|r| r.is_changed()).map(|r| r.clone()))
}

/// Converts an `Option<Signal<A>>` into a `Signal<Option<A>>`.
///
/// This is mostly useful with [`.switch`](SignalExt::switch) or [`.flatten`](SignalExt::flatten).
///
/// If the value is `None` then it emits `None` once and then terminates (implemented as
/// `signal::once(None)`).
///
/// If the value is `Some(signal)` then it will return the result of the signal,
/// except wrapped in `Some`.
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
///     .map_in(signal::option)
///     .flatten();
/// ```
pub fn option<S>(value: Option<S>) -> impl Signal<Item = Option<S::Item>> + Clone
where
    S: Signal + Clone,
    S::Item: Clone + Send + Sync + 'static,
{
    match value {
        Some(signal) => signal.map_in(Some).left_either(),
        None => once(None).right_either(),
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

impl<L, R> Clone for SignalEither<L, R>
where
    L: Signal + Clone,
    R: Signal + Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Left(left) => Self::Left(left.clone()),
            Self::Right(right) => Self::Right(right.clone()),
        }
    }
}

impl<T, L: Signal<Item = T>, R: Signal<Item = T>> Signal for SignalEither<L, R>
where
    L: Signal<Item = T>,
    R: Signal<Item = T>,
{
    type Item = T;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        match *self {
            SignalEither::Left(left) => left.register_signal(world),
            SignalEither::Right(right) => right.register_signal(world),
        }
    }
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
    ///                 .left_either()
    ///         } else {
    ///             signal::from_system(|In(_)| 42).right_either()
    ///         }
    ///     })
    ///     .flatten();
    /// // Both branches produce compatible SignalEither types despite having different
    /// // concrete Signal types (Map vs Source)
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
    ///             signal::from_system(|In(_)| "A").left_either()
    ///         } else {
    ///             signal::from_system(|In(_)| "B")
    ///                 .filter(|In(x): In<&str>| x.len() > 0)
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
        O: Clone + Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Self::Item>, IOO, M> + Send + Sync + 'static,
    {
        Map {
            upstream: self,
            signal: lazy_signal_from_system(system),
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
        O: Clone + Send + Sync + 'static,
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
        O: Clone + Send + Sync + 'static,
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
        Dedupe {
            signal: self.map(|In(current): In<Self::Item>, mut cache: Local<Option<Self::Item>>| {
                let mut changed = false;
                if let Some(ref p) = *cache {
                    if *p != current {
                        changed = true;
                    }
                } else {
                    changed = true;
                }
                if changed {
                    *cache = Some(current.clone());
                    Some(current)
                } else {
                    None
                }
            }),
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
        Self::Item: Clone + Send + Sync + 'static,
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
        Self::Item: Clone + Send + Sync + 'static,
    {
        Skip {
            signal: self.filter(move |_: In<Self::Item>, mut skipped: Local<usize>| {
                if *skipped < count {
                    *skipped += 1;
                    false
                } else {
                    true
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
        Self::Item: Clone + Send + Sync + 'static,
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
    /// let signal = signal::from_system(|In(_)| 0);
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
    /// let signal = signal::from_system(|In(_)| 0);
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
        <<Self as Signal>::Item as ops::Not>::Output: Clone + Send + Sync,
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
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(predicate);
            let SignalHandle(signal) = self
                .map::<Self::Item, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| {
                    match world.run_system_with(system, item.clone()) {
                        Ok(true) => Some(item),
                        // terminate on false or error
                        Ok(false) | Err(_) => None,
                    }
                })
                .register(world);

            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal
        });
        Filter {
            signal,
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
        let left_wrapper = self.map(|In(left): In<Self::Item>| (Some(left), None::<Other::Item>));
        let right_wrapper = other.map(|In(right): In<Other::Item>| (None::<Self::Item>, Some(right)));
        let signal = lazy_signal_from_system::<_, (Self::Item, Other::Item), _, _, _>(
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
                if left_cache.is_some() && right_cache.is_some() {
                    left_cache.clone().zip(right_cache.clone())
                } else {
                    None
                }
            },
        );

        Zip {
            left_wrapper,
            right_wrapper,
            signal,
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
    ///             signal::from_system(|In(_)| 1).left_either()
    ///         } else {
    ///             signal::from_system(|In(_)| 2).right_either()
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
        Self::Item: Signal + Clone + 'static,
        <Self::Item as Signal>::Item: Clone + Send + Sync,
    {
        #[derive(Component)]
        struct FlattenState<T> {
            value: Option<T>,
        }

        let signal = LazySignal::new(move |world: &mut World| {
            // State entity that holds the latest value, serving as the communication channel between dynamic
            // inner signals and the static output signal.
            let reader_entity = LazyEntity::new();

            // The output signal (reader). Reads from state and propagates. Has no upstream dependencies;
            // triggered manually by the forwarder.
            let reader_system = *from_system::<<Self::Item as Signal>::Item, _, _, _>(
                    clone!((reader_entity) move |In(_), mut query: Query<&mut FlattenState<<Self::Item as Signal>::Item>, Allow<Internal>>| {
                        query.get_mut(*reader_entity).unwrap().value.take()
                    }),
                )
                .register(world);
            reader_entity.set(*reader_system);
            world
                .entity_mut(*reader_system)
                .insert(FlattenState::<<Self::Item as Signal>::Item> { value: None });

            // The subscription manager reacts to the outer signal emitting new inner signals.
            let manager_system = self
                .map(
                    move |In(inner_signal): In<Self::Item>,
                          world: &mut World,
                          mut active_forwarder: Local<Option<SignalHandle>>,
                          mut active_signal_id: Local<Option<SignalSystem>>| {
                        // `register` is idempotent for existing signals.
                        let new_signal_id = inner_signal.clone().register(world);

                        // If unchanged, balance the ref-count and return early.
                        if Some(*new_signal_id) == *active_signal_id {
                            new_signal_id.cleanup(world);
                            return;
                        }

                        if let Some(old_handle) = active_forwarder.take() {
                            old_handle.cleanup(world);
                        }

                        let initial_value = poll_signal(world, *new_signal_id)
                            .and_then(downcast_any_clone::<<Self::Item as Signal>::Item>);

                        if let Some(value) = initial_value {
                            world
                                .get_mut::<FlattenState<<Self::Item as Signal>::Item>>(*reader_system)
                                .unwrap()
                                .value = Some(value);
                        }

                        // Forward subsequent values to the reader.
                        let forwarder_handle = inner_signal
                            .map(move |In(value), world: &mut World| {
                                world
                                    .get_mut::<FlattenState<<Self::Item as Signal>::Item>>(*reader_system)
                                    .unwrap()
                                    .value = Some(value);
                                trigger_signal_subgraph(world, [reader_system], Box::new(()));
                            })
                            .register(world);

                        *active_forwarder = Some(forwarder_handle);
                        *active_signal_id = Some(*new_signal_id);

                        trigger_signal_subgraph(world, [reader_system], Box::new(()));
                    },
                )
                .register(world);

            // Entity hierarchy for automatic cleanup.
            world
                .entity_mut(*reader_system)
                .insert(SignalHandles::from([manager_system]));
            reader_system
        });
        Flatten {
            signal,
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
    ///                 signal::from_system(|In(_)| 1).left_either()
    ///             } else {
    ///                 signal::from_system(|In(_)| 2).right_either()
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
        S: Signal + Clone + 'static,
        S::Item: Clone + Send + Sync,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        Switch {
            signal: self.map(switcher).flatten(),
        }
    }

    /// Maps this [`Signal`]'s output to a [`SignalVec`], switching to its output. Useful when the
    /// reactive list target changes depending on some other state.
    ///
    /// **Note:** The switched [`SignalVec`] must be downstream of a
    /// [`MutableVec`](super::signal_vec::MutableVec) to ensure proper replay of initial state
    /// on switch. The following do not yet work with this combinator:
    /// [`SignalExt::to_signal_vec`],
    /// [`MutableBTreeMap::signal_vec_keys`](super::signal_map::MutableBTreeMap::signal_vec_keys),
    /// and
    /// [`MutableBTreeMap::signal_vec_entries`](super::signal_map::MutableBTreeMap::signal_vec_entries).
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
        S: SignalVec + Clone,
        S::Item: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        // Private component to queue diffs, serving as the communication channel between dynamic
        // inner signal vecs and the static output signal.
        #[derive(Component)]
        struct SwitcherQueue<T: Send + Sync + 'static>(Vec<VecDiff<T>>);

        let signal = LazySignal::new(move |world: &mut World| {
            // The output signal (reader). Reads from queue and propagates. Has no upstream
            // dependencies; triggered manually by the forwarder.
            let output_signal_entity = LazyEntity::new();
            let output_signal = *from_system::<Vec<VecDiff<S::Item>>, _, _, _>(
                clone!((output_signal_entity) move |In(_), mut q: Query<&mut SwitcherQueue<S::Item>, Allow<Internal>>| {
                    let mut queue = q.get_mut(*output_signal_entity).unwrap();
                    if queue.0.is_empty() {
                        None
                    } else {
                        Some(core::mem::take(&mut queue.0))
                    }
                }),
            )
            .register(world);
            output_signal_entity.set(*output_signal);
            world
                .entity_mut(*output_signal)
                .insert(SwitcherQueue(Vec::<VecDiff<S::Item>>::new()));

            // The subscription manager reacts to the outer signal emitting new inner signal vecs.
            let manager_system =
                move |In(inner_signal_vec): In<S>,
                      world: &mut World,
                      mut active_forwarder: Local<Option<SignalHandle>>,
                      mut active_signal: Local<Option<SignalSystem>>| {
                    // `register` is idempotent for existing signals.
                    let new_signal_handle = inner_signal_vec.clone().register_signal_vec(world);
                    let new_signal = *new_signal_handle;

                    // If unchanged, balance the ref-count and return early.
                    if Some(new_signal) == *active_signal {
                        new_signal_handle.cleanup(world);
                        return;
                    }

                    if let Some(old_handle) = active_forwarder.take() {
                        old_handle.cleanup(world);
                    }

                    *active_signal = Some(new_signal);

                    // Forward diffs to the output signal's queue.
                    let forwarder_logic = move |In(diffs): In<Vec<VecDiff<S::Item>>>, world: &mut World| {
                        if !diffs.is_empty() {
                            // The output signal may have been cleaned up during a cleanup race.
                            if let Some(mut queue) = world.get_mut::<SwitcherQueue<S::Item>>(*output_signal) {
                                queue.0.extend(diffs);
                                trigger_signal_subgraph(world, [output_signal], Box::new(()));
                            }
                        }
                    };
                    let new_forwarder_handle = inner_signal_vec.for_each(forwarder_logic).register(world);
                    *active_forwarder = Some(new_forwarder_handle);

                    // Synchronously send the initial `Replace` diff.
                    let mut upstreams = SystemState::<Query<&Upstream, Allow<Internal>>>::new(world);
                    let upstreams = upstreams.get(world);
                    let upstreams = UpstreamIter::new(&upstreams, new_signal).collect::<Vec<_>>();
                    for signal in [new_signal].into_iter().chain(upstreams.into_iter()) {
                        let entity = *signal;
                        if world.get::<super::signal_vec::VecReplayTrigger>(entity).is_some() {
                            world.entity_mut(entity).insert(ReplayOnce);
                            trigger_replay::<super::signal_vec::VecReplayTrigger>(world, entity);
                            break;
                        }
                    }
                };

            // Entity hierarchy for automatic cleanup.
            let manager_handle = self.map(switcher).map(manager_system).register(world);
            world
                .entity_mut(*output_signal)
                .insert(SignalHandles::from([manager_handle]));
            output_signal
        });
        SwitchSignalVec {
            signal,
            _marker: PhantomData,
        }
    }

    /// Maps this [`Signal`]'s output to a [`SignalMap`], switching to its output. Useful when the
    /// reactive map target changes depending on some other state.
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
        S: SignalMap + Clone,
        S::Key: Clone + Send + Sync + 'static,
        S::Value: Clone + Send + Sync + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + Send + Sync + 'static,
    {
        // Private component to queue diffs, serving as the communication channel between dynamic
        // inner signal maps and the static output signal.
        #[derive(Component)]
        struct SwitcherQueue<K: Send + Sync + 'static, V: Send + Sync + 'static>(Vec<super::signal_map::MapDiff<K, V>>);

        let signal = LazySignal::new(move |world: &mut World| {
            // The output signal (reader). Reads from queue and propagates. Has no upstream
            // dependencies; triggered manually by the forwarder.
            let output_signal_entity = LazyEntity::new();
            let output_signal = *from_system::<Vec<super::signal_map::MapDiff<S::Key, S::Value>>, _, _, _>(
                clone!((output_signal_entity) move |In(_), mut q: Query<&mut SwitcherQueue<S::Key, S::Value>, Allow<Internal>>| {
                    let mut queue = q.get_mut(*output_signal_entity).unwrap();
                    if queue.0.is_empty() {
                        None
                    } else {
                        Some(core::mem::take(&mut queue.0))
                    }
                }),
            )
            .register(world);
            output_signal_entity.set(*output_signal);
            world
                .entity_mut(*output_signal)
                .insert(SwitcherQueue(Vec::<super::signal_map::MapDiff<S::Key, S::Value>>::new()));

            // The subscription manager reacts to the outer signal emitting new inner signal maps.
            let manager_system =
                move |In(inner_signal_map): In<S>,
                      world: &mut World,
                      mut active_forwarder: Local<Option<SignalHandle>>,
                      mut active_signal: Local<Option<SignalSystem>>| {
                    // `register` is idempotent for existing signals.
                    let new_signal_handle = inner_signal_map.clone().register_signal_map(world);
                    let new_signal = *new_signal_handle;

                    // If unchanged, balance the ref-count and return early.
                    if Some(new_signal) == *active_signal {
                        new_signal_handle.cleanup(world);
                        return;
                    }

                    if let Some(old_handle) = active_forwarder.take() {
                        old_handle.cleanup(world);
                    }

                    *active_signal = Some(new_signal);

                    // Forward diffs to the output signal's queue.
                    let forwarder_logic = move |In(diffs): In<Vec<super::signal_map::MapDiff<S::Key, S::Value>>>,
                                                world: &mut World| {
                        if !diffs.is_empty() {
                            // The output signal may have been cleaned up during a cleanup race.
                            if let Some(mut queue) = world.get_mut::<SwitcherQueue<S::Key, S::Value>>(*output_signal) {
                                queue.0.extend(diffs);
                                trigger_signal_subgraph(world, [output_signal], Box::new(()));
                            }
                        }
                    };
                    let new_forwarder_handle = inner_signal_map.for_each(forwarder_logic).register(world);
                    *active_forwarder = Some(new_forwarder_handle);

                    // Synchronously send the initial `Replace` diff.
                    let mut upstreams = SystemState::<Query<&Upstream, Allow<Internal>>>::new(world);
                    let upstreams = upstreams.get(world);
                    let upstreams = UpstreamIter::new(&upstreams, new_signal).collect::<Vec<_>>();
                    for signal in [new_signal].into_iter().chain(upstreams.into_iter()) {
                        let entity = *signal;
                        if world.get::<super::signal_map::MapReplayTrigger>(entity).is_some() {
                            world.entity_mut(entity).insert(ReplayOnce);
                            trigger_replay::<super::signal_map::MapReplayTrigger>(world, entity);
                            break;
                        }
                    }
                };

            // Entity hierarchy for automatic cleanup.
            let manager_handle = self.map(switcher).map(manager_system).register(world);
            world
                .entity_mut(*output_signal)
                .insert(SignalHandles::from([manager_handle]));
            output_signal
        });
        SwitchSignalMap {
            signal,
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
        Self::Item: Clone + Send + Sync + 'static,
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
        O: Clone + Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        TF: IntoSystem<In<()>, IOO, TM> + Send + Sync + 'static,
        FF: IntoSystem<In<()>, IOO, FM> + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let true_system = world.register_system(true_system);
            let false_system = world.register_system(false_system);
            let SignalHandle(signal) = self
                .map::<O, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| {
                    world
                        .run_system_with(if item { true_system } else { false_system }, ())
                        .ok()
                        .and_then(Into::into)
                })
                .register(world);

            // just attach the systems to the lifetime of the signal
            world
                .entity_mut(*signal)
                .add_child(true_system.entity())
                .add_child(false_system.entity());
            signal
        });
        MapBool {
            signal,
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
        O: Clone + Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        TF: FnMut() -> IOO + Send + Sync + 'static,
        FF: FnMut() -> IOO + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let SignalHandle(signal) = self
                .map::<O, _, _, _>(
                    move |In(item): In<Self::Item>| {
                        if item { true_fn().into() } else { false_fn().into() }
                    },
                )
                .register(world);
            signal
        });
        MapBool {
            signal,
            _marker: PhantomData,
        }
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
        O: Clone + Send + Sync + 'static,
        F: IntoSystem<In<()>, O, M> + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let true_system = world.register_system(system);
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| {
                    if item {
                        Some(world.run_system_with(true_system, ()).ok())
                    } else {
                        Some(None)
                    }
                })
                .register(world);

            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(true_system.entity());
            signal
        });
        MapTrue {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: FnMut() -> O + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(
                    move |In(item): In<Self::Item>| {
                        if item { Some(Some(function())) } else { Some(None) }
                    },
                )
                .register(world);
            signal
        });
        MapTrue {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: IntoSystem<In<()>, O, M> + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let false_system = world.register_system(system);
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| {
                    if !item {
                        Some(world.run_system_with(false_system, ()).ok())
                    } else {
                        Some(None)
                    }
                })
                .register(world);

            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(false_system.entity());
            signal
        });
        MapFalse {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: FnMut() -> O + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(
                    move |In(item): In<Self::Item>| {
                        if !item { Some(Some(function())) } else { Some(None) }
                    },
                )
                .register(world);
            signal
        });
        MapFalse {
            signal,
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
        O: Clone + Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        SF: IntoSystem<In<I>, IOO, SM> + Send + Sync + 'static,
        NF: IntoSystem<In<()>, IOO, NM> + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let some_system = world.register_system(some_system);
            let none_system = world.register_system(none_system);
            let SignalHandle(signal) = self
                .map::<O, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| match item {
                    Some(value) => world.run_system_with(some_system, value).ok().and_then(Into::into),
                    None => world.run_system_with(none_system, ()).ok().and_then(Into::into),
                })
                .register(world);

            // just attach the system to the lifetime of the signal
            world
                .entity_mut(*signal)
                .add_child(some_system.entity())
                .add_child(none_system.entity());
            signal
        });
        MapOption {
            signal,
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
        O: Clone + Send + Sync + 'static,
        IOO: Into<Option<O>> + 'static,
        SF: FnMut(I) -> IOO + Send + Sync + 'static,
        NF: FnMut() -> IOO + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let SignalHandle(signal) = self
                .map::<O, _, _, _>(move |In(item): In<Self::Item>| match item {
                    Some(value) => some_fn(value).into(),
                    None => none_fn().into(),
                })
                .register(world);
            signal
        });
        MapOption {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: IntoSystem<In<I>, O, M> + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let some_system = world.register_system(system);
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| match item {
                    Some(value) => Some(world.run_system_with(some_system, value).ok()),
                    None => Some(None),
                })
                .register(world);

            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(some_system.entity());
            signal
        });
        MapSome {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: FnMut(I) -> O + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>| match item {
                    Some(value) => Some(Some(function(value))),
                    None => Some(None),
                })
                .register(world);
            signal
        });
        MapSome {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: IntoSystem<In<()>, O, M> + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let none_system = world.register_system(none_system);
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>, world: &mut World| match item {
                    Some(_) => Some(None),
                    None => Some(world.run_system_with(none_system, ()).ok()),
                })
                .register(world);

            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(none_system.entity());
            signal
        });
        MapNone {
            signal,
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
        O: Clone + Send + Sync + 'static,
        F: FnMut() -> O + Send + Sync + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let SignalHandle(signal) = self
                .map::<Option<O>, _, _, _>(move |In(item): In<Self::Item>| match item {
                    Some(_) => Some(None),
                    None => Some(Some(function())),
                })
                .register(world);
            signal
        });
        MapNone {
            signal,
            _marker: PhantomData,
        }
    }

    /// Transforms this [`Signal`]'s [`Vec`] output into the corresponding [`SignalVec`]. Requires
    /// that the [`Vec`] items be [`PartialEq`] so the [`Vec`] can be
    /// [`.dedupe`](SignalExt::dedupe)-ed to prevent sending a full [`VecDiff::Replace`] every
    /// frame, which would be akin to immediate mode.
    ///
    /// Useful in situations where some [`System`] produces a static list every so often but one
    /// would like to render it dynamically e.g. with
    /// [`Builder::children_signal_vec`](crate::Builder::children_signal_vec).
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
        T: PartialEq + Clone + Send + Sync + 'static,
    {
        let lazy_signal = LazySignal::new(move |world: &mut World| {
            let handle = self
                .dedupe()
                .map(|In(items): In<Vec<T>>| vec![VecDiff::Replace { values: items }])
                .register(world);
            *handle
        });
        ToSignalVec {
            signal: lazy_signal,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "tracing")]
    #[track_caller]
    /// Adds debug logging to this [`Signal`]'s ouptut.
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
        Self::Item: fmt::Debug + Clone + Send + Sync + 'static,
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
    ///         .boxed() // this is a `Map<Source<i32>>`
    /// } else {
    ///     signal::from_system(|In(_)| 1).dedupe().boxed() // this is a `Dedupe<Source<i32>>`
    /// }; // without the `.boxed()`, the compiler would not allow this
    /// ```
    fn boxed(self) -> Box<dyn Signal<Item = Self::Item>>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Erases the type of this [`Signal`], allowing it to be used in conjunction with [`Signal`]s
    /// of other concrete types, particularly in cases where the consumer requires [`Clone`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let condition = true;
    /// if condition {
    ///     signal::from_system(|In(_)| 1)
    ///         .map_in(|x: i32| x * 2)
    ///         .boxed_clone() // this is a `Map<Source<i32>>`
    /// } else {
    ///     signal::from_system(|In(_)| 1).dedupe().boxed_clone() // this is a `Dedupe<Source<i32>>`
    /// }; // without the `.boxed_clone()`, the compiler would not allow this
    /// ```
    fn boxed_clone(self) -> Box<dyn SignalDynClone<Item = Self::Item> + Send + Sync>
    where
        Self: Sized + Clone,
    {
        Box::new(self)
    }

    /// Assign a schedule to this signal chain.
    ///
    /// When registered, signals in the chain will be tagged to run in the specified schedule,
    /// enabling control over when signal processing occurs within each frame.
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
    /// let signal = signal::from_system(|In(_)| Some(42i32))
    ///     .map_in(|x: i32| x + 1)
    ///     .schedule::<Update>();
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
    fn schedule<Sched: ScheduleLabel + Default + 'static>(self) -> Scheduled<Sched, Self::Item>
    where
        Self: Sized + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let handle = self.register_signal(world);
            apply_schedule_to_signal(world, *handle, Sched::default().intern());
            *handle
        });
        Scheduled {
            signal,
            _marker: PhantomData,
        }
    }

    /// Activate this [`Signal`] and all its upstreams, causing them to be evaluated
    /// every frame until they are [`SignalHandle::cleanup`]-ed, see [`SignalHandle`].
    fn register(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.register_signal(world)
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
    ($($args:expr),+ $(,)?) => {
        $crate::__signal_pairwise_all!(__signal_distinct_pair; $($args),+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __signal_distinct_pair {
    ($a:expr, $b:expr) => {
        $crate::eq!($a.clone(), $b.clone()).not()
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __signal_pairwise_all {
    // Base case: 2 args
    ($pair:ident; $a:expr, $b:expr $(,)?) => {
        $crate::$pair!($a, $b)
    };

    // Recursive case: 3+ args
    ($pair:ident; $head:expr, $next:expr, $($tail:expr),+ $(,)?) => {
        $crate::all!(
            $crate::__signal_pairwise_all!(@with_head $pair; $head, $next, $($tail),+),
            $crate::__signal_pairwise_all!($pair; $next, $($tail),+)
        )
    };

    // Compare a fixed head against all remaining elements
    (@with_head $pair:ident; $head:expr, $last:expr $(,)?) => {
        $crate::$pair!($head, $last)
    };
    (@with_head $pair:ident; $head:expr, $next:expr, $($tail:expr),+ $(,)?) => {
        $crate::all!(
            $crate::$pair!($head, $next),
            $crate::__signal_pairwise_all!(@with_head $pair; $head, $($tail),+)
        )
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
        graph::{LazySignalHolder, SignalRegistrationCount},
        prelude::{IntoSignalVecEither, SignalVecExt, clone},
        signal::{self, BoxedSignal, SignalExt, Upstream},
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());

        // === Part 4: Cleanup ===
        app.init_resource::<SignalOutput<i32>>();
        let signal = signal::from_system(|In(_)| 123)
            .map_in(|x| x)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(123));

        // Cleanup and reset
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
        let signal = signal::from_entity(entity_without)
            .component_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(None));
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
        let signal = signal::from_entity(entity_without)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_fan_out_fan_in_zip() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<(i32, i32)>>();

        let source = signal::from_system(|In(_), mut state: Local<i32>| {
            *state += 1;
            *state
        });

        let left = source.clone().map_in(|x: i32| x + 1);
        let right = source.map_in(|x: i32| x + 10);

        let signal = left.zip(right).map(capture_output).register(app.world_mut());

        app.update();
        assert_eq!(get_output::<(i32, i32)>(app.world()), Some((2, 11)));

        app.update();
        assert_eq!(get_output::<(i32, i32)>(app.world()), Some((3, 12)));

        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());

        // Test 2 signals - same as .zip()
        app.init_resource::<SignalOutput<(i32, &'static str)>>();
        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| "two");
        let signal = zip!(s1, s2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<(i32, &'static str)>(app.world()), Some((1, "two")));
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());

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
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_flatten() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal_1 = signal::from_system(|In(_)| 1).boxed_clone();
        let signal_2 = signal::from_system(|In(_)| 2).boxed_clone();

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
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_eq() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();
        let source = signal::from_system(|In(_)| 1);
        let signal = source.clone().eq(1).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());
        let signal = source.eq(2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_neq() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<bool>>();
        let source = signal::from_system(|In(_)| 1);
        let signal = source.clone().neq(2).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());
        let signal = source.neq(1).map(capture_output).register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
        let signal = signal::from_system(|In(_)| false)
            .not()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_switch() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal_1 = signal::from_system(|In(_)| 1).boxed_clone();
        let signal_2 = signal::from_system(|In(_)| 2).boxed_clone();

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
        signal.cleanup(app.world_mut());
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
            let signal_a = list_a.signal_vec().map_in(identity);
            let signal_b = list_b.signal_vec();

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
                    // can use .left/right_either or .boxed_clone
                    ListSelector:: A => signal_a.clone().left_either(),
                    ListSelector:: B => signal_b.clone().right_either(),
                    // TODO: this is the more conservative case, so it should be covered by the above case of signal reuse, but ideally they would both be explicitly tested
                    // ListSelector:: A => list_a.signal_vec().map_in(identity).boxed_clone(),
                    // ListSelector:: B => list_b.signal_vec().boxed_clone(),
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
            handle.cleanup(app.world_mut());
        }

        crate::signal_vec::tests::cleanup(true);
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
            let signal_a = list_a.signal_vec().map_in(identity);
            let signal_b = list_b.signal_vec();

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
            // The first update should select List A which is empty.
            // With an empty initial list, no Replace diff should be emitted.
            app.update();
            let diffs = get_and_clear_vec_output::<i32>(app.world_mut());
            assert!(
                diffs.is_empty(),
                "Initial update with empty list should produce no diffs, got: {:?}",
                diffs
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

            handle.cleanup(app.world_mut());
        }

        crate::signal_vec::tests::cleanup(true);
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
            let signal_a = map_a.signal_map().map_value(|In(x): In<i32>| x);
            let signal_b = map_b.signal_map();

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
                    // can use .left/right_either or .boxed_clone
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
            handle.cleanup(app.world_mut());
        }

        crate::signal_map::tests::cleanup(true);
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
            let signal_a = map_a.signal_map().map_value(|In(x): In<i32>| x);
            let signal_b = map_b.signal_map();

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
            // The first update should select Map A which is empty.
            // With an empty initial map, no Replace diff should be emitted.
            app.update();
            let diffs = get_and_clear_map_output::<i32, i32>(app.world_mut());
            assert!(
                diffs.is_empty(),
                "Initial update with empty map should produce no diffs, got: {:?}",
                diffs
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

            handle.cleanup(app.world_mut());
        }

        crate::signal_map::tests::cleanup(true);
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

        // 1. Initial emission: The first update runs with near-zero delta. The first value should always
        //    pass.
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
        app.world_mut().entity(**handle);
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

        handle.cleanup(app.world_mut());
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
        signal_true.cleanup(app.world_mut());

        // Test false case
        let signal_false = signal::from_system(|In(_)| false)
            .map_bool(|In(_)| "True Branch", |In(_)| "False Branch")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("False Branch"));
        signal_false.cleanup(app.world_mut());
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
        signal_true.cleanup(app.world_mut());

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
        signal_false.cleanup(app.world_mut());
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
        signal_false.cleanup(app.world_mut());

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
        signal_true.cleanup(app.world_mut());
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
        signal_some.cleanup(app.world_mut());

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
        signal_none.cleanup(app.world_mut());
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
        signal_some.cleanup(app.world_mut());

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
        signal_none.cleanup(app.world_mut());
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
        signal_none.cleanup(app.world_mut());

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
        signal_some.cleanup(app.world_mut());
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

        // Frame 3: Emit the same vector. Should be deduplicated.
        *source_vec.lock().unwrap() = Some(vec![1, 2, 3]);
        app.update();
        assert!(
            get_and_clear_vec_output::<i32>(app.world_mut()).is_empty(),
            "Should be deduplicated when source emits the same vector"
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
        handle.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal_self.cleanup(app.world_mut());

        // Test parent (generations = 1)
        let signal_parent = signal::from_ancestor(child, 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal_parent.cleanup(app.world_mut());

        // Test grandparent (generations = 2)
        let signal_gp = signal::from_ancestor(child, 2)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(grandparent));
        signal_gp.cleanup(app.world_mut());

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
        signal_invalid.cleanup(app.world_mut());
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
        signal_parent.cleanup(app.world_mut());

        // Test grandparent (generations = 2)
        let signal_gp = signal::from_ancestor(lazy_child, 2)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(grandparent));
        signal_gp.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());

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
        signal_no_parent.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal_with.cleanup(app.world_mut());

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
        signal_without.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal_with.cleanup(app.world_mut());

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
        signal_without.cleanup(app.world_mut());
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
        signal.cleanup(app.world_mut());
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
        signal_with.cleanup(app.world_mut());

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
        signal_without.cleanup(app.world_mut());
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
        signal_with.cleanup(app.world_mut());

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
        signal_without.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());

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
        signal_without.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());
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

        signal.cleanup(app.world_mut());

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
        signal_without.cleanup(app.world_mut());
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
        some_signal.cleanup(app.world_mut());

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
        none_signal.cleanup(app.world_mut());

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
        dynamic_signal.cleanup(app.world_mut());
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
                crate::signal::option(Some(signal::from_system(|In(_)| 42)))
            } else {
                crate::signal::option(None)
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

        signal.cleanup(app.world_mut());
    }

    #[test]
    fn simple_signal_lazy_outlives_handle() {
        let mut app = create_test_app();
        let source_signal_struct = signal::from_system(|In(_)| 1);
        let handle = source_signal_struct.clone().register(app.world_mut());
        let system_entity = handle.0.entity();
        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "System entity should exist after registration"
        );
        assert!(
            app.world().get::<LazySignalHolder>(system_entity).is_some(),
            "LazySignalHolder should exist on system entity"
        );
        assert_eq!(
            **app.world().get::<SignalRegistrationCount>(system_entity).unwrap(),
            1,
            "SignalRegistrationCount should be 1"
        );
        handle.cleanup(app.world_mut());
        assert_eq!(
            **app.world().get::<SignalRegistrationCount>(system_entity).unwrap(),
            0,
            "SignalRegistrationCount should be 0 after cleanup"
        );

        // LazySignalHolder is not removed because source_signal_struct (holding another
        // LazySignal clone) still exists, so holder.lazy_signal.inner.references > 1 at
        // the time of cleanup.
        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "System entity should still exist after handle cleanup (LazySignal struct alive)"
        );
        assert!(
            app.world().get::<LazySignalHolder>(system_entity).is_some(),
            "LazySignalHolder should still exist"
        );
        drop(source_signal_struct);

        // Dropping source_signal_struct reduces LazySignalState.references. If it becomes
        // 1 (only holder's copy left), LazySignal::drop does not queue. The system is not
        // queued to CLEANUP_SIGNALS yet. LazySignalHolder is still present. Runs
        // flush_cleanup_signals. CLEANUP_SIGNALS is empty.
        app.update();

        // Runs flush_cleanup_signals. CLEANUP_SIGNALS is empty.
        app.update();
        assert!(
            app.world().get_entity(system_entity).is_err(),
            "System entity persists because LazySignalHolder was not removed and its LazySignal did not trigger cleanup on its own drop"
        );
    }

    #[test]
    fn multiple_lazy_signal_clones_cleanup_behavior() {
        let mut app = create_test_app();

        // LS.state_refs = 1
        let s1 = signal::from_system(|In(_)| 1);

        // LS.state_refs = 2
        let s2 = s1.clone();

        // LS.state_refs = 3 (s1, s2, holder)
        let handle = s1.clone().register(app.world_mut());
        let system_entity = handle.0.entity();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // RegCount = 0. LS.state_refs = 3 for holder check. Holder not removed.
        handle.cleanup(app.world_mut());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 0);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // LS.state_refs = 2. Not queued.
        drop(s1);
        app.update();
        assert!(
            app.world().get_entity(system_entity).is_ok(),
            "Entity persists after s1 drop"
        );
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // LS.state_refs = 1 (only holder's copy). Not queued.
        drop(s2);
        app.update();
        assert!(
            app.world().get_entity(system_entity).is_err(),
            "Entity despawned after s2 drop, only holder left"
        );
    }

    #[test]
    fn multiple_handles_same_system() {
        let mut app = create_test_app();

        // LS.state_refs = 1
        let source_signal_struct = signal::from_system(|In(_)| 1);

        // LS.state_refs = 2 (struct, holder). RegCount = 1.
        let handle1 = source_signal_struct.clone().register(app.world_mut());
        let system_entity = handle1.0.entity();
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);

        // LS.state_refs = 2 (no new holder). RegCount = 2.
        let handle2 = source_signal_struct.clone().register(app.world_mut());
        assert_eq!(system_entity, handle2.0.entity());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 2);

        // RegCount = 1. Holder not removed (LS.state_refs=2).
        handle1.cleanup(app.world_mut());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // LS.state_refs = 1 (holder only). Not queued.
        drop(source_signal_struct);
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // RegCount = 0. LS.state_refs = 1 for holder. Holder IS removed. Queued.
        handle2.cleanup(app.world_mut());
        app.update();
        assert!(app.world().get_entity(system_entity).is_err());
    }

    #[test]
    fn chained_signals_cleanup() {
        let mut app = create_test_app();
        let source_s = signal::from_system(|In(_)| 1);

        // map_s holds source_s
        let map_s = source_s.map(|In(val)| val + 1);
        let handle = map_s.clone().register(app.world_mut());
        let map_entity = handle.0.entity();
        let source_entity = app
            .world()
            .get::<Upstream>(map_entity)
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .entity();
        assert!(app.world().get_entity(map_entity).is_ok());
        assert!(app.world().get_entity(source_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(map_entity).is_some());
        assert!(app.world().get::<LazySignalHolder>(source_entity).is_some());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(map_entity).unwrap(), 1);
        assert_eq!(**app.world().get::<SignalRegistrationCount>(source_entity).unwrap(), 1);
        handle.cleanup(app.world_mut());

        // map_entity: RegCount=0. Holder's LS refs > 1 (due to map_s). Holder not
        // removed. source_entity: RegCount=0. Holder's LS refs > 1 (due to
        // map_s.source_s). Holder not removed.
        assert_eq!(**app.world().get::<SignalRegistrationCount>(map_entity).unwrap(), 0);
        assert_eq!(**app.world().get::<SignalRegistrationCount>(source_entity).unwrap(), 0);
        assert!(app.world().get_entity(map_entity).is_ok());
        assert!(app.world().get_entity(source_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(map_entity).is_some());
        assert!(app.world().get::<LazySignalHolder>(source_entity).is_some());

        // Drops map_s's LazySignal, then source_s's LazySignal.
        drop(map_s);

        // For both, their LS.state_refs becomes 1 (holder only). queued.
        app.update();

        // Both entities despawned.
        assert!(app.world().get_entity(map_entity).is_err(), "Map entity persists");
        assert!(app.world().get_entity(source_entity).is_err(), "Source entity persists");
    }

    #[test]
    fn re_register_after_cleanup_while_lazy_alive() {
        let mut app = create_test_app();

        // LS.state_refs = 1
        let source_signal_struct = signal::from_system(|In(_)| 1);

        // LS.state_refs = 2 (struct, holder). RegCount = 1.
        let handle1 = source_signal_struct.clone().register(app.world_mut());
        let system_entity = handle1.0.entity();
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);

        // RegCount = 0. Holder not removed (LS.state_refs=2).
        handle1.cleanup(app.world_mut());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 0);
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // RegCount becomes 1 again on same entity. LS.state_refs=2.
        let handle2 = source_signal_struct.clone().register(app.world_mut());
        assert_eq!(system_entity, handle2.0.entity());
        assert_eq!(**app.world().get::<SignalRegistrationCount>(system_entity).unwrap(), 1);
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // LS.state_refs = 1 (holder only). Not queued.
        drop(source_signal_struct);
        app.update();
        assert!(app.world().get_entity(system_entity).is_ok());
        assert!(app.world().get::<LazySignalHolder>(system_entity).is_some());

        // RegCount = 0. Holder's LS.state_refs = 1. Holder IS removed. despawned.
        handle2.cleanup(app.world_mut());
        assert!(app.world().get_entity(system_entity).is_err());
    }

    #[test]
    fn test_signal_eq() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 1);
        let s3 = signal::from_system(|In(_)| 1);
        let s4 = signal::from_system(|In(_)| 2);

        // Test equality
        let eq_signal = crate::signal::eq!(s1.clone(), s2.clone(), s3.clone());
        eq_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));

        // Test inequality
        let neq_signal = crate::signal::eq!(s1.clone(), s2.clone(), s4.clone());
        neq_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));
    }

    #[test]
    fn test_signal_and() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let t = signal::from_system(|In(_)| true);
        let f = signal::from_system(|In(_)| false);

        // Test AND (all true)
        let all_signal = crate::signal::all!(t.clone(), t.clone(), t.clone());
        all_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));

        // Test AND (one false)
        let all_signal = crate::signal::all!(t.clone(), f.clone(), t.clone());
        all_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));
    }

    #[test]
    fn test_signal_or() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let t = signal::from_system(|In(_)| true);
        let f = signal::from_system(|In(_)| false);

        // Test OR (all false)
        let any_signal = crate::signal::any!(f.clone(), f.clone(), f.clone());
        any_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(false));

        // Test OR (one true)
        let any_signal = crate::signal::any!(f.clone(), t.clone(), f.clone());
        any_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));
    }

    #[test]
    fn test_signal_distinct() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);
        let s3 = signal::from_system(|In(_)| 3);

        // Test distinct (all different)
        let distinct_signal = crate::signal::distinct!(s1.clone(), s2.clone(), s3.clone());
        distinct_signal.map(capture_output::<bool>).register(app.world_mut());
        app.update();
        assert_eq!(app.world().resource::<SignalOutput<bool>>().0, Some(true));

        // Test distinct (some equal)
        let mut app = create_test_app();
        app.insert_resource(SignalOutput::<bool>::default());

        let s1 = signal::from_system(|In(_)| 1);
        let s2 = signal::from_system(|In(_)| 2);

        let distinct_signal = crate::signal::distinct!(s1.clone(), s2.clone(), s1.clone());
        distinct_signal.map(capture_output::<bool>).register(app.world_mut());
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
        sum_signal.map(capture_output::<i32>).register(app.world_mut());
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
        sum_signal.map(capture_output::<i32>).register(app.world_mut());
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
        product_signal.map(capture_output::<i32>).register(app.world_mut());
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
        product_signal.map(capture_output::<i32>).register(app.world_mut());
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
        min_signal.map(capture_output::<i32>).register(app.world_mut());
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
        min_signal.map(capture_output::<i32>).register(app.world_mut());
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
        max_signal.map(capture_output::<i32>).register(app.world_mut());
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
        max_signal.map(capture_output::<i32>).register(app.world_mut());
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

        signal.register(app.world_mut());

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

        signal.register(app.world_mut());

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

        signal.map(capture_output::<Option<i32>>).register(app.world_mut());

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

        signal.map(capture_output::<Option<i32>>).register(app.world_mut());

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

        signal.map(capture_output::<i32>).register(app.world_mut());

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

        signal.map(capture_output::<Option<i32>>).register(app.world_mut());

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

        signal.map(capture_output::<Option<i32>>).register(app.world_mut());

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
        use crate::graph::{ScheduleTag, Upstream};
        use bevy_app::{PostUpdate, Update};
        use bevy_ecs::schedule::ScheduleLabel;

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            crate::JonmoPlugin::new::<PostUpdate>().with_schedule::<Update>(),
        ));

        // Build a chain: source -> map1 -> schedule::<Update> -> map2 -> schedule::<PostUpdate> -> map3
        // Note: Scheduled doesn't create its own entity, it just tags its inner signal.
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
        let map3_entity = **handle;

        // Walk upstream to find all entities
        let world = app.world();

        // map3 -> map2 -> map1 -> source
        let map3_upstreams = world.get::<Upstream>(map3_entity).unwrap();
        assert_eq!(map3_upstreams.len(), 1);
        let map2_entity = **map3_upstreams.iter().next().unwrap();

        let map2_upstreams = world.get::<Upstream>(map2_entity).unwrap();
        assert_eq!(map2_upstreams.len(), 1);
        let map1_entity = **map2_upstreams.iter().next().unwrap();

        let map1_upstreams = world.get::<Upstream>(map1_entity).unwrap();
        assert_eq!(map1_upstreams.len(), 1);
        let source_entity = **map1_upstreams.iter().next().unwrap();

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
}
