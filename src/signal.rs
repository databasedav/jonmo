//! Signal builders and combinators for constructing reactive [`System`] dependency graphs, see
//! [`SignalExt`].
use super::{
    graph::{
        LazySignal, SignalHandle, SignalSystem, Upstream, UpstreamIter, downcast_any_clone, lazy_signal_from_system,
        pipe_signal, poll_signal, process_signals,
    },
    signal_vec::{Replayable, SignalVec, SignalVecExt, VecDiff},
    utils::{LazyEntity, SSs, ancestor_map},
};
use crate::prelude::clone;
use bevy_ecs::{entity_disabling::Internal, prelude::*, system::SystemState};
cfg_if::cfg_if! {
    if #[cfg(feature = "tracing")] {
        use bevy_log::prelude::*;
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

// TODO: fix the check-all-features command in kaaj
// TODO: make replayable actually able to replay cached signal vecs and add a test for this case
// (the filters example does not actually do this) TODO: upgrade to bevy 0.17

/// Monadic registration facade for structs that encapsulate some [`System`] which is a valid member
/// of the signal graph.
pub trait Signal: SSs {
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

/// Signal graph node which takes an input of [`In<()>`] and has no upstreams. See [`SignalBuilder`]
/// methods for examples.
#[derive(Clone)]
pub struct Source<O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
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
#[derive(Clone)]
pub struct Map<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
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

/// Signal graph node which maps its upstream [`Entity`] to a corresponding [`Component`], see
/// [.component](SignalExt::component).
#[derive(Clone)]
pub struct MapComponent<Upstream, C> {
    signal: Map<Upstream, C>,
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
#[derive(Clone)]
pub struct ComponentOption<Upstream, C> {
    signal: Map<Upstream, Option<C>>,
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

/// Signal graph node with maps its upstream [`Entity`] to whether it has some [`Component`], see
/// [`.has_component`](SignalExt::has_component).
#[derive(Clone)]
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

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Signal graph node which does not forward upstream duplicates, see
/// [`.dedupe`](SignalExt::dedupe).
#[derive(Clone)]
pub struct Dedupe<Upstream>
where
    Upstream: Signal,
{
    signal: Map<Upstream, Upstream::Item>,
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
#[derive(Clone)]
pub struct First<Upstream>
where
    Upstream: Signal,
{
    signal: Map<Upstream, Upstream::Item>,
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

/// Signal graph node which combines two upstreams, see [`.combine`](SignalExt::combine).
#[derive(Clone)]
pub struct Combine<Left, Right>
where
    Left: Signal,
    Right: Signal,
{
    #[allow(clippy::type_complexity)]
    left_wrapper: Map<Left, (Option<Left::Item>, Option<Right::Item>)>,
    #[allow(clippy::type_complexity)]
    right_wrapper: Map<Right, (Option<Left::Item>, Option<Right::Item>)>,
    signal: LazySignal,
}

impl<Left, Right> Signal for Combine<Left, Right>
where
    Left: Signal,
    Right: Signal,
{
    type Item = (Left::Item, Right::Item);

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        let SignalHandle(left_upstream) = self.left_wrapper.register(world);
        let SignalHandle(right_upstream) = self.right_wrapper.register(world);
        let signal = self.signal.register(world);
        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);
        signal.into()
    }
}

/// Signal graph node which outputs equality between its upstream and a fixed value, see
/// [`.eq`](SignalExt::eq).
#[derive(Clone)]
pub struct Eq<Upstream> {
    signal: Map<Upstream, bool>,
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
#[derive(Clone)]
pub struct Neq<Upstream> {
    signal: Map<Upstream, bool>,
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
#[derive(Clone)]
pub struct Not<Upstream>
where
    Upstream: Signal,
    <Upstream as Signal>::Item: ops::Not,
    <<Upstream as Signal>::Item as ops::Not>::Output: Clone,
{
    signal: Map<Upstream, <Upstream::Item as ops::Not>::Output>,
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
#[derive(Clone)]
pub struct Filter<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
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

#[derive(Component)]
struct FlattenState<T> {
    value: Option<T>,
}

/// Signal graph node which forwards the upstream output [`Signal`]'s output, see
/// [`.flatten`](SignalExt::flatten).
#[derive(Clone)]
pub struct Flatten<Upstream>
where
    Upstream: Signal,
    Upstream::Item: Signal,
{
    signal: LazySignal,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<fn() -> (Upstream, Upstream::Item)>,
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
#[derive(Clone)]
pub struct Switch<Upstream, Other>
where
    Upstream: Signal,
    Other: Signal,
    Other::Item: Clone,
{
    signal: Flatten<Map<Upstream, Other>>,
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
#[derive(Clone)]
pub struct SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalVec,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, Switched)>,
}

impl<Upstream, Switched> SignalVec for SwitchSignalVec<Upstream, Switched>
where
    Upstream: Signal,
    Switched: SignalVec,
    Upstream::Item: 'static,
    Switched::Item: Clone + SSs,
{
    type Item = Switched::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "time")] {
        /// Signal graph node which delays the propagation of subsequent upstream outputs, see
        /// [`.throttle`](SignalExt::throttle).
        #[derive(Clone)]
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

            fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
                self.signal.register(world)
            }
        }
    }
}

/// Signal graph node whose [`System`] depends on its upstream [`bool`] value, see
/// [`.map_bool`](SignalExt::map_bool).
#[derive(Clone)]
pub struct MapBool<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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
#[derive(Clone)]
pub struct MapTrue<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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
#[derive(Clone)]
pub struct MapFalse<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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
#[derive(Clone)]
pub struct MapOption<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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
#[derive(Clone)]
pub struct MapSome<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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
#[derive(Clone)]
pub struct MapNone<Upstream, O> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
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
#[derive(Clone)]
pub struct ToSignalVec<Upstream>
where
    Upstream: Signal,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
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
        #[derive(Clone)]
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

            fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
                self.signal.register(world)
            }
        }
    }
}

/// Provides static methods for creating [`Source`] signals.
pub struct SignalBuilder;

impl From<Entity> for Source<Entity> {
    fn from(entity: Entity) -> Self {
        SignalBuilder::from_entity(entity)
    }
}

impl SignalBuilder {
    /// Creates a [`Source`] signal from a [`System`] that takes [`In<()>`].
    pub fn from_system<O, IOO, F, M>(system: F) -> Source<O>
    where
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<()>, IOO, M> + SSs,
    {
        Source {
            signal: lazy_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    /// Creates a [`Source`] signal from an [`Entity`].
    pub fn from_entity(entity: Entity) -> Source<Entity> {
        Self::from_system(move |_: In<()>| entity)
    }

    /// Creates a [`Source`] signal from a [`LazyEntity`].
    pub fn from_lazy_entity(entity: LazyEntity) -> Source<Entity> {
        Self::from_system(move |_: In<()>| *entity)
    }

    /// Creates a [`Source`] signal from an [`Entity`]'s `generations`-th generation ancestor's
    /// [`Entity`], terminating for frames where it does not exist. Passing `0` to `generation` will
    /// output the [`Entity`] itself.
    pub fn from_ancestor(entity: Entity, generations: usize) -> Map<Source<Entity>, Entity> {
        Self::from_entity(entity).map(ancestor_map(generations))
    }

    /// Creates a [`Source`] signal from a [`LazyEntity`]'s `generations`-th generation ancestor's
    /// [`Entity`], terminating for frames where it does not exist. Passing `0` to `generation` will
    /// output the [`LazyEntity`]'s  [`Entity`] itself.
    pub fn from_ancestor_lazy(entity: LazyEntity, generations: usize) -> Map<Source<Entity>, Entity> {
        Self::from_lazy_entity(entity).map(ancestor_map(generations))
    }

    /// Creates a [`Source`] signal from an [`Entity`]'s parent's [`Entity`], terminating for frames
    /// where it does not exist. Passing `0` to `generation` will output the [`Entity`] itself.
    pub fn from_parent(entity: Entity) -> Map<Source<Entity>, Entity> {
        Self::from_ancestor(entity, 1)
    }

    /// Creates a [`Source`] signal from a [`LazyEntity`]'s parent's [`Entity`], terminating for
    /// frames where it does not exist. Passing `0` to `generation` will output the
    /// [`LazyEntity`]'s [`Entity`] itself.
    pub fn from_parent_lazy(entity: LazyEntity) -> Map<Source<Entity>, Entity> {
        Self::from_ancestor_lazy(entity, 1)
    }

    /// Creates a [`Source`] signal from an [`Entity`] and a [`Component`], terminating for the
    /// frame if the [`Entity`] does not exist or the [`Component`] does not exist on the
    /// [`Entity`].
    pub fn from_component<C>(entity: Entity) -> Source<C>
    where
        C: Component + Clone,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| components.get(entity).ok().cloned())
    }

    /// Creates a [`Source`] signal from a [`LazyEntity`] and a [`Component`], terminating for the
    /// frame if the [`Entity`] does not exist or the [`Component`] does not exist on the
    /// corresponding [`Entity`].
    pub fn from_component_lazy<C>(entity: LazyEntity) -> Source<C>
    where
        C: Component + Clone,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| components.get(*entity).ok().cloned())
    }

    /// Creates a [`Source`] signal from an [`Entity`] and a [`Component`], always outputting an
    /// [`Option`].
    pub fn from_component_option<C>(entity: Entity) -> Source<Option<C>>
    where
        C: Component + Clone,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| Some(components.get(entity).ok().cloned()))
    }

    /// Creates a [`Source`] signal from a [`LazyEntity`] and a [`Component`], always outputting an
    /// [`Option`].
    pub fn from_component_option_lazy<C>(entity: LazyEntity) -> Source<Option<C>>
    where
        C: Component + Clone,
    {
        Self::from_system(move |_: In<()>, components: Query<&C>| Some(components.get(*entity).ok().cloned()))
    }

    /// Creates a signal from a [`Resource`], terminating for the frame if the [`Resource`] does not
    /// exist.
    pub fn from_resource<R>() -> Source<R>
    where
        R: Resource + Clone,
    {
        Self::from_system(move |_: In<()>, resource: Option<Res<R>>| resource.as_deref().cloned())
    }

    /// Creates a signal from a [`Resource`], always outputting an [`Option`].
    pub fn from_resource_option<R>() -> Source<Option<R>>
    where
        R: Resource + Clone,
    {
        Self::from_system(move |_: In<()>, resource: Option<Res<R>>| Some(resource.as_deref().cloned()))
    }
}

/// Enables returning different concrete [`Signal`] types from branching logic without boxing,
/// although note that all [`Signal`]s are boxed internally regardless.
///
/// Inspired by <https://github.com/rayon-rs/either>.
#[derive(Clone)]
#[allow(missing_docs)]
pub enum SignalEither<L, R>
where
    L: Signal,
    R: Signal,
{
    Left(L),
    Right(R),
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
#[allow(missing_docs)]
pub trait IntoSignalEither: Sized
where
    Self: Signal,
{
    fn left_either<R>(self) -> SignalEither<Self, R>
    where
        R: Signal,
    {
        SignalEither::Left(self)
    }

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
    /// SignalBuilder::from_system(|_: In<()>| 1).map(|In(x): In<i32>| x * 2); // outputs `2`
    /// ```
    fn map<O, IOO, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Self::Item>, IOO, M> + SSs,
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
    /// SignalBuilder::from_system(|_: In<()>| 1).map_in(|x: i32| x * 2); // outputs `2`
    /// ```
    fn map_in<O, IOO, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: FnMut(Self::Item) -> IOO + SSs,
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
    /// SignalBuilder::from_system(|_: In<()>| 1).map_in_ref(ToString::to_string); // outputs `"1"`
    /// ```
    fn map_in_ref<O, IOO, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: FnMut(&Self::Item) -> IOO + SSs,
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
    /// SignalBuilder::from_entity(entity).component::<Value>(); // outputs `Value(0)`
    ///
    /// let entity = world.spawn_empty().id();
    /// SignalBuilder::from_entity(entity).component::<Value>(); // terminates
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
    /// SignalBuilder::from_entity(entity).component_option::<Value>(); // outputs `Some(Value(0))`
    ///
    /// let entity = world.spawn_empty().id();
    /// SignalBuilder::from_entity(entity).component::<Value>(); // outputs `None`
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
    /// SignalBuilder::from_entity(entity).has_component::<Value>(); // outputs `true`
    ///
    /// let entity = world.spawn_empty().id();
    /// SignalBuilder::from_entity(entity).has_component::<Value>(); // outputs `false`
    /// ```
    fn has_component<C>(self) -> HasComponent<Self, C>
    where
        Self: Sized,
        Self: Signal<Item = Entity>,
        C: Component + Clone,
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
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state / 2
    ///     }
    /// })
    /// .dedupe(); // outputs `0`, `1`, `2`, `3`, ...
    /// ```
    fn dedupe(self) -> Dedupe<Self>
    where
        Self: Sized,
        Self::Item: PartialEq + Clone + Send + 'static,
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

    /// Outputs this [`Signal`]'s first value and then terminates for all subsequent frames.
    ///
    /// After the first value is emitted, this signal will stop propagating any further values.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state / 2
    ///     }
    /// })
    /// .first(); // outputs `0` then terminates forever
    /// ```
    fn first(self) -> First<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
    {
        First {
            signal: self.map(|In(item): In<Self::Item>, mut first: Local<bool>| {
                if *first {
                    None
                } else {
                    *first = true;
                    Some(item)
                }
            }),
        }
    }

    /// Output this [`Signal`]'s equality with a fixed value.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal = SignalBuilder::from_system(|_: In<()>| 0);
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
    /// let signal = SignalBuilder::from_system(|_: In<()>| 0);
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
    /// SignalBuilder::from_system(|_: In<()>| true).not(); // outputs `false`
    /// ```
    fn not(self) -> Not<Self>
    where
        Self: Sized,
        <Self as Signal>::Item: ops::Not + 'static,
        <<Self as Signal>::Item as ops::Not>::Output: Clone,
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
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<usize>| {
    ///         *state += 1;
    ///         *state % 2
    ///     }
    /// })
    /// .filter(|In(i): In<usize>| i % 2 == 0); // outputs `2`, `4`, `6`, `8`, ...
    /// ```
    fn filter<M>(self, predicate: impl IntoSystem<In<Self::Item>, bool, M> + SSs) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
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
    /// outputs. The resulting [`Signal`] will only output a value when both input [`Signal`]s have
    /// outputted a value since the last resulting output, e.g. if on frame 1, this [`Signal`]
    /// outputs `1` and the other [`Signal`] outputs `None`, the resulting [`Signal`] will have no
    /// output, but then if on frame 2, this [`Signal`] outputs `None` and the other [`Signal`]
    /// outputs 2, then the resulting [`Signal`] will output `(1, 2)`.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let signal_1 = SignalBuilder::from_system(|_: In<()>| 1);
    /// let signal_2 = SignalBuilder::from_system(|_: In<()>| 2);
    /// signal_1.combine(signal_2); // outputs `(1, 2)`
    /// ```
    fn combine<Other>(self, other: Other) -> Combine<Self, Other>
    where
        Self: Sized,
        Other: Signal,
        Self::Item: Clone + Send + 'static,
        Other::Item: Clone + Send + 'static,
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
                    left_cache.take().zip(right_cache.take())
                } else {
                    None
                }
            },
        );
        Combine {
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
    /// let signal = SignalBuilder::from_resource::<Toggle>()
    ///     .dedupe()
    ///     .map(move |In(toggle): In<Toggle>| {
    ///         if toggle.0 {
    ///             SignalBuilder::from_system(|_: In<()>| 1)
    ///         } else {
    ///             SignalBuilder::from_system(|_: In<()>| 2)
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
        let signal =
            LazySignal::new(move |world: &mut World| {
                // 1. The state entity that holds the latest value. This is the communication channel between the
                //    dynamic inner signals and the static output signal.
                let state_entity = world
                    .spawn(FlattenState::<<Self::Item as Signal>::Item> { value: None })
                    .id();

                // 2. The final output signal (reader). Its ONLY job is to read the state component and propagate
                //    the value. It has no upstream dependencies in the graph; it is triggered manually by the
                //    forwarder.
                let reader_system = *SignalBuilder::from_system::<<Self::Item as Signal>::Item, _, _, _>(
                    move |_: In<()>, mut query: Query<&mut FlattenState<<Self::Item as Signal>::Item>>| {
                        if let Ok(mut state) = query.get_mut(state_entity) {
                            state.value.take()
                        } else {
                            None
                        }
                    },
                )
                .register(world);

                // 3. This is the "subscription manager" system. It reacts to the outer signal emitting new inner
                //    signals.
                let manager_system = self
                    .map(
                        // This closure contains the core logic for switching subscriptions.
                        move |In(inner_signal): In<Self::Item>,
                              world: &mut World,
                              mut active_forwarder: Local<Option<SignalHandle>>,
                              mut active_signal_id: Local<Option<SignalSystem>>| {
                            // A. Get the canonical ID of the newly emitted inner signal. `register` is
                            // idempotent; it just increments the ref-count if the system exists.
                            let new_signal_id = inner_signal.clone().register(world);

                            // B. MEMOIZATION: Check if the signal has actually changed from the last frame.
                            if Some(*new_signal_id) == *active_signal_id {
                                // The signal is the same. Do nothing. IMPORTANT: We must cleanup the handle from
                                // our `.register()` call above to balance the reference count, otherwise it will
                                // leak.
                                new_signal_id.cleanup(world);
                                return;
                            }

                            // C. TEARDOWN: The signal is new, so clean up the old forwarder and its ID.
                            if let Some(old_handle) = active_forwarder.take() {
                                old_handle.cleanup(world);
                            }

                            // The old signal ID handle is implicitly dropped when overwritten.
                            // ================== The Core Setup Logic for the NEW Signal ==================
                            // D. Get the initial value of the new inner signal, synchronously. This is done
                            // by creating a temporary one-shot signal.
                            let temp_handle = inner_signal.clone().first().register(world);
                            let initial_value = poll_signal(world, *temp_handle)
                                .and_then(downcast_any_clone::<<Self::Item as Signal>::Item>);

                            // The temporary handle must be cleaned up immediately.
                            temp_handle.cleanup(world);

                            // E. Write this initial value directly into the state component.
                            if let Some(value) = initial_value
                                && let Some(mut state) =
                                    world.get_mut::<FlattenState<<Self::Item as Signal>::Item>>(state_entity)
                            {
                                state.value = Some(value);
                            }

                            // F. Set up a persistent forwarder for all _subsequent_ updates from the new
                            // signal.
                            let forwarder_handle = inner_signal.map(
                        // This first map writes the value to the state component.
                        move |In(value), mut query: Query<&mut FlattenState<<Self::Item as Signal>::Item>>| {
                            if let Ok(mut state) = query.get_mut(state_entity) {
                                state.value = Some(value);
                            }
                        },
                    ).map(
                        // This second map triggers the reader to run immediately after the state is
                        // written.
                        move |_, world: &mut World| {
                            process_signals(world, [reader_system], Box::new(()));
                        },
                    ).register(world);

                            // G. Store the new forwarder's handle and the new signal's ID for the next frame.
                            *active_forwarder = Some(forwarder_handle);
                            *active_signal_id = Some(*new_signal_id);

                            // H. Manually trigger the reader to run RIGHT NOW to consume the initial value.
                            process_signals(world, [reader_system], Box::new(()));
                        },
                    )
                    .register(world);

                // 4. Set up entity hierarchy for automatic cleanup. When the `reader_system` (the final output) is
                //    cleaned up, it will also despawn its children.
                world
                    .entity_mut(*reader_system)
                    .add_child(state_entity)
                    .add_child(**manager_system);
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
    ///     SignalBuilder::from_resource::<Toggle>()
    ///         .dedupe()
    ///         .switch(move |In(toggle): In<Toggle>| {
    ///             if toggle.0 {
    ///                 SignalBuilder::from_system(|_: In<()>| 1)
    ///             } else {
    ///                 SignalBuilder::from_system(|_: In<()>| 2)
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
        F: IntoSystem<In<Self::Item>, S, M> + SSs,
    {
        Switch {
            signal: self.map(switcher).flatten(),
        }
    }

    /// Maps this [`Signal`]'s output to a [`SignalVec`], switching to its output. Useful when the
    /// reactive list target changes depending on some other state.
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
    /// let signal = SignalBuilder::from_resource::<Toggle>()
    ///     .dedupe()
    ///     .switch_signal_vec(move |In(toggle): In<Toggle>, world: &mut World| {
    ///         if toggle.0 {
    ///             MutableVecBuilder::from([1, 2, 3]).spawn(world).signal_vec()
    ///         } else {
    ///             MutableVecBuilder::from([10, 20]).spawn(world).signal_vec()
    ///         }
    ///     });
    /// // `signal` outputs `SignalVec -> [10, 20]`
    /// world.resource_mut::<Toggle>().0 = true;
    /// // `signal` outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn switch_signal_vec<S, F, M>(self, switcher: F) -> SwitchSignalVec<Self, S>
    where
        Self: Sized,
        Self::Item: 'static,
        S: SignalVec + Clone + 'static,
        S::Item: Clone + SSs,
        F: IntoSystem<In<Self::Item>, S, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            // A private component to queue diffs for the output system.
            #[derive(Component)]
            struct SwitcherQueue<T: SSs>(Vec<VecDiff<T>>);

            // The output system is a "poller". It runs when poked and drains its own queue.
            let output_system_entity = LazyEntity::new();
            let output_system = lazy_signal_from_system::<_, Vec<VecDiff<S::Item>>, _, _, _>(
                clone!((output_system_entity) move |_: In<()>, mut q: Query<&mut SwitcherQueue<S::Item>, Allow<Internal>>| {
                    if let Ok(mut queue) = q.get_mut(*output_system_entity) {
                        if queue.0.is_empty() {
                            None
                        } else {
                            Some(core::mem::take(&mut queue.0))
                        }
                    } else {
                        None
                    }
                }),
            )
            .register(world);
            output_system_entity.set(*output_system);

            // Add the queue component to the output system's entity so it can be found.
            world
                .entity_mut(*output_system)
                .insert(SwitcherQueue(Vec::<VecDiff<S::Item>>::new()));

            // This is the core logic. It runs whenever the outer signal changes.
            let manager_system_logic =
                move |In(inner_signal_vec): In<S>,
                      world: &mut World,
                      mut active_forwarder: Local<Option<SignalHandle>>,
                      mut active_memo_handle: Local<Option<SignalHandle>>,
                      mut active_signal_id: Local<Option<SignalSystem>>| {
                    // A. Get the canonical ID by registering. We need this for memoization.
                    let new_signal_id_handle = inner_signal_vec.clone().register_signal_vec(world);
                    let new_signal_id = *new_signal_id_handle;

                    // B. MEMOIZATION: If the signal hasn't changed, do nothing.
                    if Some(new_signal_id) == *active_signal_id {
                        // Balance the ref count for this temporary handle.
                        new_signal_id_handle.cleanup(world);
                        return;
                    }

                    // C. TEARDOWN: The signal is new. Clean up the old forwarder and the old
                    // memoization handle.
                    if let Some(old_handle) = active_forwarder.take() {
                        old_handle.cleanup(world);
                    }
                    if let Some(old_memo_handle) = active_memo_handle.take() {
                        old_memo_handle.cleanup(world);
                    }
                    *active_signal_id = Some(new_signal_id);

                    // D. FORWARDING: The forwarder's only job is to queue ALL diffs from the inner
                    // signal and poke the output system.
                    let forwarder_logic = move |In(diffs): In<Vec<VecDiff<S::Item>>>, world: &mut World| {
                        if !diffs.is_empty()
                            && let Some(mut queue) = world.get_mut::<SwitcherQueue<S::Item>>(*output_system)
                        {
                            queue.0.extend(diffs);
                            process_signals(world, [output_system], Box::new(()));
                        }
                    };

                    // E. PERSIST: Register the new forwarder to listen for diffs, and store both the
                    // forwarder's handle and the new memoization handle.
                    let new_forwarder_handle = inner_signal_vec.for_each(forwarder_logic).register(world);
                    *active_forwarder = Some(new_forwarder_handle);
                    *active_memo_handle = Some(new_signal_id_handle);

                    // F. MANUALLY TRIGGER REPLAY: Take the trigger from the new signal and run it
                    // _now_. This synchronously sends the initial `Replace` diff through the pipe we
                    // just established. Taking it also prevents the `trigger_replays` system from
                    // running it again. if let Some(trigger) =
                    // world.get_entity_mut(*new_signal_id).ok().and_then(|mut entity|
                    // entity.take::<super::signal_vec::VecReplayTrigger>()) {
                    let mut upstreams = SystemState::<Query<&Upstream, Allow<Internal>>>::new(world);
                    let upstreams = upstreams.get(world);
                    let upstreams = UpstreamIter::new(&upstreams, new_signal_id).collect::<Vec<_>>();
                    for signal in [new_signal_id].into_iter().chain(upstreams.into_iter()) {
                        if let Some(trigger) = world.entity_mut(*signal).take::<super::signal_vec::VecReplayTrigger>() {
                            trigger.trigger()(world);
                            break;
                        }
                    }
                };

            // Register the manager and tie its lifecycle to the output system.
            let manager_handle = self.map(switcher).map(manager_system_logic).register(world);
            world.entity_mut(*output_system).add_child(**manager_handle);
            output_system
        });
        SwitchSignalVec {
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
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<usize>| {
    ///        *state += 1;
    ///        *state
    ///     }
    /// })
    /// .throttle(Duration::from_secs(1)); // outputs `1`, terminates for the next 1 second of frames, outputs `2`, terminates for the next 1 second of frames, outputs `3`, ...
    /// ```
    fn throttle(self, duration: Duration) -> Throttle<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
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
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_bool(|_: In<()>| 1, |_: In<()>| 0); // outputs `1`, `0`, `1`, `0`, ...
    /// ```
    fn map_bool<O, IOO, TF, FF, TM, FM>(self, true_system: TF, false_system: FF) -> MapBool<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        TF: IntoSystem<In<()>, IOO, TM> + SSs,
        FF: IntoSystem<In<()>, IOO, FM> + SSs,
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

    /// If this [`Signal`] outputs [`true`], output the result of some [`System`], otherwise
    /// terminate for the frame.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_true(|_: In<()>| 1); // outputs `1`, terminates, outputs `1`, terminates, ...
    /// ```
    fn map_true<O, F, M>(self, system: F) -> MapTrue<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Clone + 'static,
        F: IntoSystem<In<()>, O, M> + SSs,
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

    /// If this [`Signal`] outputs [`false`], output the result of some [`System`], otherwise
    /// terminate for the frame.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<bool>| {
    ///         *state = !*state;
    ///         *state
    ///     }
    /// })
    /// .map_false(|_: In<()>| 1); // terminates, outputs `1`, terminates, outputs `1`, ...
    /// ```
    fn map_false<O, F, M>(self, system: F) -> MapFalse<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = bool>,
        O: Clone + 'static,
        F: IntoSystem<In<()>, O, M> + SSs,
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

    /// Maps this [`Signal`] to some [`System`] depending on its [`Option`] output.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<Option<bool>>| {
    ///        *state = if state.is_some() { None } else { Some(true) };
    ///        *state
    ///     }
    /// })
    /// .map_option(
    ///     |In(state): In<bool>| state,
    ///     |_: In<()>| false,
    /// ); // outputs `true`, `false`, `true`, `false`, ...
    /// ```
    fn map_option<I, O, IOO, SF, NF, SM, NM>(self, some_system: SF, none_system: NF) -> MapOption<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        SF: IntoSystem<In<I>, IOO, SM> + SSs,
        NF: IntoSystem<In<()>, IOO, NM> + SSs,
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

    /// If this [`Signal`] outputs [`Some`], output the result of some [`System`] which takes [`In`]
    /// the [`Some`] value, otherwise terminate for the frame.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<Option<bool>>| {
    ///        *state = if state.is_some() { None } else { Some(true) };
    ///        *state
    ///     }
    /// })
    /// .map_some(
    ///     |In(state): In<bool>| state
    /// ); // outputs `true`, terminates, outputs `true`, terminates, ...
    /// ```
    fn map_some<I, O, F, M>(self, system: F) -> MapSome<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Clone + 'static,
        F: IntoSystem<In<I>, O, M> + SSs,
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

    /// If this [`Signal`] outputs [`None`], output the result of some [`System`], otherwise
    /// terminate for the frame.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<Option<bool>>| {
    ///         *state = if state.is_some() { None } else { Some(true) };
    ///         *state
    ///     }
    /// })
    /// .map_none(|_: In<()>| false); // terminates, outputs `false`, terminates, outputs `false`, ...
    /// ```
    fn map_none<I, O, F, M>(self, none_system: F) -> MapNone<Self, O>
    where
        Self: Sized,
        Self: Signal<Item = Option<I>>,
        I: 'static,
        O: Clone + 'static,
        F: IntoSystem<In<()>, O, M> + SSs,
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

    /// Transforms this [`Signal`]'s [`Vec`] output into the corresponding [`SignalVec`]. Requires
    /// that the [`Vec`] items be [`PartialEq`] so the [`Vec`] can be
    /// [`.dedupe`](SignalExt::dedupe)-ed to prevent sending a full [`VecDiff::Replace`] every
    /// frame, which would be akin to immediate mode.
    ///
    /// Useful in situations where some [`System`] produces a static list every so often but one
    /// would like to render it dynamically e.g. with
    /// [`JonmoBuilder::children_signal_vec`](super::builder::JonmoBuilder::children_signal_vec).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system({
    ///     move |_: In<()>, mut state: Local<Vec<usize>>| {
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
        T: PartialEq + Clone + Send + 'static,
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

    // TODO: why won't doctest compile ?
    #[cfg(feature = "tracing")]
    /// Adds debug logging to this [`Signal`]'s ouptut.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system(|_: In<()>| 1)
    ///     .debug() // logs `1`
    ///     .map_in(|x: i32| x * 2);
    ///     .debug(); // logs `2`
    /// ```
    fn debug(self) -> Debug<Self>
    where
        Self: Sized,
        Self::Item: fmt::Debug + Clone + 'static,
    {
        let location = core::panic::Location::caller();
        Debug {
            signal: self.map(move |In(item)| {
                debug!("[{}] {:#?}", location, item);
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
    ///     SignalBuilder::from_system(|_: In<()>| 1).map_in(|x: i32| x * 2).boxed() // this is a `Map<Source<i32>>`
    /// } else {
    ///     SignalBuilder::from_system(|_: In<()>| 1).dedupe().boxed() // this is a `Dedupe<Source<i32>>`
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
    ///     SignalBuilder::from_system(|_: In<()>| 1)
    ///         .map_in(|x: i32| x * 2)
    ///         .boxed_clone() // this is a `Map<Source<i32>>`
    /// } else {
    ///     SignalBuilder::from_system(|_: In<()>| 1)
    ///         .dedupe()
    ///         .boxed_clone() // this is a `Dedupe<Source<i32>>`
    /// }; // without the `.boxed_clone()`, the compiler would not allow this
    /// ```
    fn boxed_clone(self) -> Box<dyn SignalDynClone<Item = Self::Item>>
    where
        Self: Sized + Clone,
    {
        Box::new(self)
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

#[cfg(test)]
mod tests {
    use crate::{
        JonmoPlugin,
        graph::{LazySignalHolder, SignalRegistrationCount},
        prelude::{SignalVecExt, clone},
        signal::{SignalBuilder, SignalExt, Upstream},
        signal_vec::{MutableVecBuilder, VecDiff},
        utils::{LazyEntity, SSs},
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
    struct SignalOutput<T: SSs + Clone + fmt::Debug>(Option<T>);

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app.register_type::<TestData>();
        app
    }

    // Helper system to capture signal output
    fn capture_output<T: SSs + Clone + fmt::Debug>(In(value): In<T>, mut output: ResMut<SignalOutput<T>>) {
        debug!(
            "Capture Output System: Received {:?}, updating resource from {:?} to Some({:?})",
            value, output.0, value
        );
        output.0 = Some(value);
    }

    fn get_output<T: SSs + Clone + fmt::Debug>(world: &World) -> Option<T> {
        world.resource::<SignalOutput<T>>().0.clone()
    }

    #[test]
    fn test_map() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal = SignalBuilder::from_system(|_: In<()>| 1)
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
        let signal = SignalBuilder::from_system(|_: In<()>| 10i32)
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
        let source_signal = SignalBuilder::from_system(clone!((inputs) move |_: In<()>| {
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
        let signal = SignalBuilder::from_system(|_: In<()>| ())
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
        let signal = SignalBuilder::from_system(|_: In<()>| 123)
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
        let signal = SignalBuilder::from_system(|_: In<()>| 42i32)
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
        let source_signal = SignalBuilder::from_system(clone!((inputs) move |_: In<()>| {
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
        let signal = SignalBuilder::from_system(|_: In<()>| 1i32)
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
        let signal = SignalBuilder::from_system(|_: In<()>| 123)
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
        let signal = SignalBuilder::from_entity(entity)
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
        let signal = SignalBuilder::from_entity(entity_with)
            .component_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(1))));
        signal.cleanup(app.world_mut());
        let signal = SignalBuilder::from_entity(entity_without)
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
        let signal = SignalBuilder::from_entity(entity_with)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(true));
        signal.cleanup(app.world_mut());
        let signal = SignalBuilder::from_entity(entity_without)
            .has_component::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_dedupe() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let counter = Arc::new(Mutex::new(0));
        let values = Arc::new(Mutex::new(vec![1, 1, 2, 3, 3, 3, 4]));
        let signal = SignalBuilder::from_system(clone!((values) move |_: In<()>| {
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
        let signal = SignalBuilder::from_system(clone!((values) move |_: In<()>| {
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
    fn test_combine() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<(i32, &'static str)>>();
        let signal = SignalBuilder::from_system(move |_: In<()>| 10)
            .combine(SignalBuilder::from_system(move |_: In<()>| "hello"))
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<(i32, &'static str)>(app.world()), Some((10, "hello")));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_flatten() {
        let mut app = create_test_app();
        app.init_resource::<SignalOutput<i32>>();
        let signal_1 = SignalBuilder::from_system(|_: In<()>| 1);
        let signal_2 = SignalBuilder::from_system(|_: In<()>| 2);

        #[derive(Resource, Default)]
        struct SignalSelector(bool);

        app.init_resource::<SignalSelector>();
        let signal = SignalBuilder::from_system(
            move |_: In<()>, selector: Res<SignalSelector>| {
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
        let source = SignalBuilder::from_system(|_: In<()>| 1);
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
        let source = SignalBuilder::from_system(|_: In<()>| 1);
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
        let signal = SignalBuilder::from_system(|_: In<()>| true)
            .not()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<bool>(app.world()), Some(false));
        signal.cleanup(app.world_mut());
        let signal = SignalBuilder::from_system(|_: In<()>| false)
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
        let signal = SignalBuilder::from_system(move |_: In<()>| {
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
        let signal_1 = SignalBuilder::from_system(|_: In<()>| 1);
        let signal_2 = SignalBuilder::from_system(|_: In<()>| 2);

        #[derive(Resource, Default)]
        struct SwitcherToggle(bool);

        app.init_resource::<SwitcherToggle>();
        let signal = SignalBuilder::from_system(move |_: In<()>, mut toggle: ResMut<SwitcherToggle>| {
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
    struct SignalVecOutput<T: SSs + Clone + fmt::Debug>(Vec<VecDiff<T>>);

    fn capture_vec_output<T>(In(diffs): In<Vec<VecDiff<T>>>, mut output: ResMut<SignalVecOutput<T>>)
    where
        T: SSs + Clone + fmt::Debug,
    {
        output.0.extend(diffs);
    }

    fn get_and_clear_vec_output<T: SSs + Clone + fmt::Debug>(world: &mut World) -> Vec<VecDiff<T>> {
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
            let list_a = MutableVecBuilder::from([10, 20]).spawn(app.world_mut());
            let list_b = MutableVecBuilder::from([100, 200, 300]).spawn(app.world_mut());

            // A resource to control which list is active.
            #[derive(Resource, Clone, Copy, PartialEq, Debug)]
            enum ListSelector {
                A,
                B,
            }

            app.insert_resource(ListSelector::A);

            // The control signal reads the resource.
            let control_signal =
                SignalBuilder::from_system(|_: In<()>, selector: Res<ListSelector>| *selector).dedupe();

            // The signal chain under test.
            let switched_signal = control_signal.dedupe().switch_signal_vec(
                clone!((list_a, list_b) move |In(selector): In<ListSelector>| match selector {
                    ListSelector:: A => list_a.signal_vec().map_in(identity).boxed_clone(),
                    ListSelector:: B => list_b.signal_vec().boxed_clone(),
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
    fn test_throttle() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins.build().disable::<TimePlugin>(), JonmoPlugin));
        app.init_resource::<Time>(); // manually managing time for the test
        app.init_resource::<SignalOutput<i32>>();

        // A simple counter to generate a new value (1, 2, 3...) for each update.
        let source_signal = SignalBuilder::from_system(|_: In<()>, mut counter: Local<i32>| {
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
        let signal_true = SignalBuilder::from_system(|_: In<()>| true)
            .map_bool(|_: In<()>| "True Branch", |_: In<()>| "False Branch")
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<&'static str>(app.world()), Some("True Branch"));
        signal_true.cleanup(app.world_mut());

        // Test false case
        let signal_false = SignalBuilder::from_system(|_: In<()>| false)
            .map_bool(|_: In<()>| "True Branch", |_: In<()>| "False Branch")
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
        let source_true = SignalBuilder::from_system(|_: In<()>| true);
        let signal_true = source_true
            .map_true(|_: In<()>| "Was True")
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
        let source_false = SignalBuilder::from_system(|_: In<()>| false);
        let signal_false = source_false
            .map_true(|_: In<()>| "Was True")
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
        let source_false = SignalBuilder::from_system(|_: In<()>| false);
        let signal_false = source_false
            .map_false(|_: In<()>| "Was False")
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
        let source_true = SignalBuilder::from_system(|_: In<()>| true);
        let signal_true = source_true
            .map_false(|_: In<()>| "Was False")
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
        let signal_some = SignalBuilder::from_system(|_: In<()>| Some(42))
            .map_option(
                |In(value): In<i32>| format!("Some({value})"),
                |_: In<()>| "None".to_string(),
            )
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<String>(app.world()), Some("Some(42)".to_string()));
        signal_some.cleanup(app.world_mut());

        // Reset output
        app.world_mut().resource_mut::<SignalOutput<String>>().0 = None;

        // Test None case
        let signal_none = SignalBuilder::from_system(|_: In<()>| None::<i32>)
            .map_option(
                |In(value): In<i32>| format!("Some({value})"),
                |_: In<()>| "None".to_string(),
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
        let source_some = SignalBuilder::from_system(|_: In<()>| Some(42));
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
        let source_none = SignalBuilder::from_system(|_: In<()>| None::<i32>);
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
        let source_none = SignalBuilder::from_system(|_: In<()>| None::<i32>);
        let signal_none = source_none
            .map_none(|_: In<()>| "Was None")
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
        let source_some = SignalBuilder::from_system(|_: In<()>| Some(42));
        let signal_some = source_some
            .map_none(|_: In<()>| "Was None")
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
        let source_signal = SignalBuilder::from_system(clone!((source_vec) move |_: In<()>| {
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

        let signal = SignalBuilder::from_system(|_: In<()>| 42)
            .map(capture_output)
            .register(app.world_mut());

        app.update();
        assert_eq!(get_output::<i32>(app.world()), Some(42));
        signal.cleanup(app.world_mut());
    }

    #[test]
    fn test_builder_from_entity() {
        let mut app = create_test_app();
        app.insert_resource(SignalOutput(Some(Entity::PLACEHOLDER)));

        let test_entity = app.world_mut().spawn_empty().id();
        let signal = SignalBuilder::from_entity(test_entity)
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

        let signal = SignalBuilder::from_lazy_entity(lazy)
            .map(capture_output)
            .register(app.world_mut());

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
        let signal_self = SignalBuilder::from_ancestor(child, 0)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(child));
        signal_self.cleanup(app.world_mut());

        // Test parent (generations = 1)
        let signal_parent = SignalBuilder::from_ancestor(child, 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal_parent.cleanup(app.world_mut());

        // Test grandparent (generations = 2)
        let signal_gp = SignalBuilder::from_ancestor(child, 2)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(grandparent));
        signal_gp.cleanup(app.world_mut());

        // Test invalid generation
        app.world_mut().resource_mut::<SignalOutput<Entity>>().0 = None;
        let signal_invalid = SignalBuilder::from_ancestor(child, 3)
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
        let signal_parent = SignalBuilder::from_ancestor_lazy(lazy_child.clone(), 1)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal_parent.cleanup(app.world_mut());

        // Test grandparent (generations = 2)
        let signal_gp = SignalBuilder::from_ancestor_lazy(lazy_child, 2)
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

        let signal = SignalBuilder::from_parent(child)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Entity>(app.world()), Some(parent));
        signal.cleanup(app.world_mut());

        // Test no parent
        app.world_mut().resource_mut::<SignalOutput<Entity>>().0 = None;
        let no_parent_entity = app.world_mut().spawn_empty().id();
        let signal_no_parent = SignalBuilder::from_parent(no_parent_entity)
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

        let signal = SignalBuilder::from_parent_lazy(lazy_child)
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
        let signal_with = SignalBuilder::from_component::<TestData>(entity_with)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));
        signal_with.cleanup(app.world_mut());

        // Test component missing
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        let entity_without = app.world_mut().spawn_empty().id();
        let signal_without = SignalBuilder::from_component::<TestData>(entity_without)
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

        let signal = SignalBuilder::from_component_lazy::<TestData>(lazy)
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
        let signal_with = SignalBuilder::from_component_option::<TestData>(entity_with)
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(42))));
        signal_with.cleanup(app.world_mut());

        // Test component missing
        let entity_without = app.world_mut().spawn_empty().id();
        let signal_without = SignalBuilder::from_component_option::<TestData>(entity_without)
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

        let signal = SignalBuilder::from_component_option_lazy::<TestData>(lazy)
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
        let signal_with = SignalBuilder::from_resource::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<TestData>(app.world()), Some(TestData(42)));
        signal_with.cleanup(app.world_mut());

        // Test resource missing
        app.world_mut().resource_mut::<SignalOutput<TestData>>().0 = None;
        app.world_mut().remove_resource::<TestData>();
        let signal_without = SignalBuilder::from_resource::<TestData>()
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
        let signal_with = SignalBuilder::from_resource_option::<TestData>()
            .map(capture_output)
            .register(app.world_mut());
        app.update();
        assert_eq!(get_output::<Option<TestData>>(app.world()), Some(Some(TestData(42))));
        signal_with.cleanup(app.world_mut());

        // Test resource missing
        app.world_mut().remove_resource::<TestData>();
        let signal_without = SignalBuilder::from_resource_option::<TestData>()
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
    fn simple_signal_lazy_outlives_handle() {
        let mut app = create_test_app();
        let source_signal_struct = SignalBuilder::from_system(|_: In<()>| 1);
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
        let s1 = SignalBuilder::from_system(|_: In<()>| 1);

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
        let source_signal_struct = SignalBuilder::from_system(|_: In<()>| 1);

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
        let source_s = SignalBuilder::from_system(|_: In<()>| 1);

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
        let source_signal_struct = SignalBuilder::from_system(|_: In<()>| 1);

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
}
