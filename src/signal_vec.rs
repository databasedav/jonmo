use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{change_detection::Mut, prelude::*, system::SystemId};
use bevy_log::debug;
use bevy_platform::{
    collections::HashMap,
    prelude::*,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use core::{any::Any, cmp::Ordering, fmt, marker::PhantomData, ops::Deref};

use super::{graph::*, signal::*, utils::*};

/// Describes the mutations made to the underlying [`MutableVec`] that are piped through
/// [`Downstream`] [`SignalVec`]s.
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

impl<T: Clone> VecDiff<T> {
    #[allow(missing_docs)]
    pub fn apply_to_vec(&self, vec: &mut Vec<T>) {
        match self {
            VecDiff::Replace { values } => *vec = values.clone(),
            VecDiff::InsertAt { index, value } => vec.insert(*index, value.clone()),
            VecDiff::UpdateAt { index, value } => {
                if let Some(elem) = vec.get_mut(*index) {
                    *elem = value.clone();
                }
            }
            VecDiff::RemoveAt { index } => {
                if *index < vec.len() {
                    vec.remove(*index);
                }
            }
            VecDiff::Move { old_index, new_index } => {
                if *old_index < vec.len() {
                    let val = vec.remove(*old_index);
                    vec.insert(*new_index, val);
                }
            }
            VecDiff::Push { value } => vec.push(value.clone()),
            VecDiff::Pop => {
                vec.pop();
            }
            VecDiff::Clear => vec.clear(),
        }
    }
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

/// Monadic registration facade for structs that encapsulate some [`System`] which is a valid member
/// of the signal graph [`Downstream`] of some source [`MutableVec`]; this is similar to [`Signal`]
/// but critically requires that the [`System`] outputs [`Option<VecDiff<Self::Item>>`] and will
/// often take [`In<VecDiff<T>>`].
pub trait SignalVec: SSs {
    /// Output type.
    type Item;

    /// Registers the [`System`]s associated with this [`SignalVec`] by consuming its boxed form.
    ///
    /// All concrete signal types must implement this method.
    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle;

    /// Registers the [`System`]s associated with this [`SignalVec`].
    fn register_signal_vec(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.boxed().register_boxed_signal_vec(world)
    }
}

impl<O: 'static> SignalVec for Box<dyn SignalVec<Item = O> + Send + Sync> {
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        let inner_box: Box<dyn SignalVec<Item = O> + Send + Sync> = *self;
        inner_box.register_boxed_signal_vec(world)
    }
}

/// Signal graph node with no [`Upstream`]s which forwards [`Vec<VecDiff<T>>`]s flushed from some
/// source [`MutableVec<T>`], see [`MutableVec::signal_vec`].
#[derive(Clone)]
pub struct Source<T> {
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> T>,
}

impl<T> SignalVec for Source<T>
where
    T: 'static,
{
    type Item = T;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which applies a [`System`] directly to the [`Vec<VecDiff>`]s of its upstream,
/// see [`.for_each`](SignalVecExt::for_each).
#[derive(Clone)]
pub struct ForEach<Upstream, O> {
    pub(crate) upstream: Upstream,
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> O>,
}

impl<Upstream, O> Signal for ForEach<Upstream, O>
where
    Upstream: SignalVec,
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

/// A map node in a `SignalVec` chain.
#[derive(Clone)]
pub struct Map<Upstream, O> {
    pub(crate) signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> SignalVec for Map<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

// #[derive(Clone)]
// pub(crate) struct MappedSignalItem<S: Signal> {
//     pub(crate) processor_handle: SignalHandle,
//     pub(crate) inner_signal_id: SignalSystem,
//     _marker: PhantomData<fn() -> S>,
// }

// // Fix for E0310: Add the 'static lifetime bound to T
// #[derive(Component)]
// pub(crate) struct MapSignalManager<T: 'static, S: Signal> {
//     pub(crate) items: Vec<MappedSignalItem<S>>,
//     pub(crate) system_id: SystemId<In<T>, S>,
// }

// #[derive(Component, Deref, DerefMut)]
// struct MappedSignalIndex(usize);

// fn with_map_signal_data<T, S, O>(
//     world: &mut World,
//     entity: Entity,
//     f: impl FnOnce(Mut<MapSignalManager<T, S>>) -> O,
// ) -> O
// where
//     T: 'static,
//     S: Signal,
//     S::Item: Send + Sync,
// {
//     f(world.get_mut::<MapSignalManager<T, S>>(entity).unwrap())
// }

/// A node that maps items to signals and flattens the result.
#[derive(Clone)]
pub struct MapSignal<Upstream, S: Signal> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S: Signal> SignalVec for MapSignal<Upstream, S>
where
    Upstream: SignalVec,
{
    type Item = S::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

fn find_index<'a>(indices: impl Iterator<Item = &'a bool>, index: usize) -> usize {
    indices.take(index).filter(|x| **x).count()
}

fn filter_helper<T, O, O2>(
    world: &mut World,
    diffs: Vec<VecDiff<T::Inner<'static>>>,
    system: SystemId<T, O>,
    f: impl Fn(T::Inner<'static>, O) -> Option<O2>,
    indices: &mut Vec<bool>,
) -> Option<Vec<VecDiff<O2>>>
where
    T: SystemInput + 'static,
    T::Inner<'static>: Clone,
    O: 'static,
{
    let mut output = vec![];
    for diff in diffs {
        let diff_option = match diff {
            VecDiff::Replace { values } => {
                *indices = Vec::with_capacity(values.len());
                let mut output = Vec::with_capacity(values.len());
                for input in values {
                    let value = world
                        .run_system_with(system, input.clone())
                        .ok()
                        .and_then(|output| f(input, output));
                    indices.push(value.is_some());
                    if let Some(value) = value {
                        output.push(value);
                    }
                }
                Some(VecDiff::Replace { values: output })
            }

            VecDiff::InsertAt { index, value } => {
                if let Some(value) = world
                    .run_system_with(system, value.clone())
                    .ok()
                    .and_then(|output| f(value, output))
                {
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
                if let Some(value) = world
                    .run_system_with(system, value.clone())
                    .ok()
                    .and_then(|output| f(value, output))
                {
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
                if indices.remove(old_index) {
                    indices.insert(new_index, true);

                    Some(VecDiff::Move {
                        old_index: find_index(indices.iter(), old_index),
                        new_index: find_index(indices.iter(), new_index),
                    })
                } else {
                    indices.insert(new_index, false);
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
                if let Some(value) = world
                    .run_system_with(system, value.clone())
                    .ok()
                    .and_then(|output| f(value, output))
                {
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
                indices.clear();
                Some(VecDiff::Clear)
            }
        };

        if let Some(diff) = diff_option {
            output.push(diff);
        }
    }

    if output.is_empty() { None } else { Some(output) }
}

#[derive(Clone)]
pub struct Filter<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for Filter<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

#[derive(Clone)]
pub struct FilterMap<Upstream, O>
where
    Upstream: SignalVec,
{
    signal: LazySignal,

    _marker: PhantomData<fn() -> (Upstream, O)>,
}

impl<Upstream, O> SignalVec for FilterMap<Upstream, O>
where
    Upstream: SignalVec,
    O: 'static,
{
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

struct FilterSignalItem<T> {
    signal: SignalHandle,
    value: T,
    filtered: bool,
}

#[derive(Component)]
struct FilterSignalData<T> {
    items: Vec<FilterSignalItem<T>>,
    diffs: Vec<VecDiff<T>>,
}

fn with_filter_signal_data<T: SSs, O>(
    world: &mut World,
    entity: Entity,
    f: impl FnOnce(Mut<FilterSignalData<T>>) -> O,
) -> O {
    let data = world.get_mut::<FilterSignalData<T>>(entity).unwrap();
    f(data)
}

fn find_filter_signal_index<T>(filter_signal_items: &[FilterSignalItem<T>], i: usize) -> usize {
    find_index(filter_signal_items.iter().map(|signal| &signal.filtered), i)
}

#[derive(Component, Deref, DerefMut)]
struct FilterSignalIndex(usize);

fn create_filter_signal_processor<T: Clone + SSs>(
    parent: Entity,
    entity: LazyEntity,
) -> impl Fn(In<bool>, Query<&FilterSignalIndex>, Query<&mut FilterSignalData<T>>) {
    move |In(filter),
          filter_signal_indices: Query<&FilterSignalIndex>,
          mut filter_signal_datas: Query<&mut FilterSignalData<T>>| {
        let mut filter_signal_data = filter_signal_datas.get_mut(parent).unwrap();
        let index = filter_signal_indices.get(entity.get()).unwrap().0;
        let mut new = None;
        if let Some(signal) = filter_signal_data.items.get(index)
            && signal.filtered != filter
        {
            let filtered_index = find_filter_signal_index(&filter_signal_data.items, index);
            if filter {
                new = Some((filtered_index, Some(signal.value.clone())))
            } else {
                new = Some((filtered_index, None))
            }
        }
        if let Some((filtered_index, value_option)) = new {
            if let Some(signal) = filter_signal_data.items.get_mut(index) {
                signal.filtered = filter;
            }
            filter_signal_data.diffs.push(if let Some(value) = value_option {
                VecDiff::InsertAt {
                    index: filtered_index,
                    value,
                }
            } else {
                VecDiff::RemoveAt { index: filtered_index }
            });
        }
    }
}

fn spawn_filter_signal<T: Clone + SSs>(
    world: &mut World,
    index: usize,
    signal: impl Signal<Item = bool> + 'static,
    parent: Entity,
) -> (SignalHandle, bool) {
    let entity = LazyEntity::new();
    let processor = create_filter_signal_processor::<T>(parent, entity.clone());

    // Use .map() to attach the processor. The dedupe is still important.
    let mapped_signal = signal.dedupe().map(processor);

    let handle = mapped_signal.register(world);
    entity.set(**handle);
    world.entity_mut(**handle).insert(FilterSignalIndex(index));

    // To get the initial value, we must poll the original signal before the map/dedupe.
    // The upstream of the `map` node is the `dedupe` node.
    let dedupe_handle = world.get::<Upstream>(**handle).unwrap().iter().next().cloned().unwrap();
    // The upstream of the `dedupe` node is the original signal.
    let original_signal_handle = world
        .get::<Upstream>(*dedupe_handle)
        .unwrap()
        .iter()
        .next()
        .cloned()
        .unwrap();

    let initial_value = poll_signal(world, original_signal_handle)
        .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<bool>().ok())
        .map(|boxed| *boxed)
        .expect("filter_signal's inner signal must emit an initial value upon registration");

    (handle, initial_value)
}

#[derive(Clone)]
pub struct FilterSignal<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for FilterSignal<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

#[derive(Component, Deref, DerefMut, Clone)]
pub struct EnumeratedIndex(usize);

fn index_signal_from_entity(
    entity: Entity,
) -> Dedupe<super::signal::Map<super::signal::Source<Option<EnumeratedIndex>>, Option<usize>>> {
    SignalBuilder::from_component_option::<EnumeratedIndex>(entity)
        .map(|In(opt): In<Option<EnumeratedIndex>>| opt.map(|c| c.0))
        .dedupe()
}

#[derive(Component, Default)]
struct EnumerateState {
    key_to_index: HashMap<usize, usize>,
    ordered_keys: Vec<usize>,
    next_key: usize,
}

#[derive(Clone)]
pub struct Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    signal: LazySignal,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for Enumerate<Upstream>
where
    Upstream: SignalVec,
{
    type Item = (Dedupe<super::signal::Source<Option<usize>>>, Upstream::Item);

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

#[derive(Clone)]
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

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct IsEmpty<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    signal: super::signal::Map<ToSignal<Upstream>, bool>,
}

impl<Upstream> Signal for IsEmpty<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    type Item = bool;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct Len<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    signal: super::signal::Map<ToSignal<Upstream>, usize>,
}

impl<Upstream> Signal for Len<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: Clone,
{
    type Item = usize;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
pub struct Sum<Upstream>
where
    Upstream: SignalVec,
{
    signal: super::signal::Map<ToSignal<Upstream>, Upstream::Item>,
}

impl<Upstream> Signal for Sum<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Clone)]
enum LrDiff<T> {
    Left(Vec<VecDiff<T>>),
    Right(Vec<VecDiff<T>>),
}

/// A node that chains two `SignalVec`s together.
#[derive(Clone)]
pub struct Chain<Left, Right>
where
    Left: SignalVec,
    Right: SignalVec<Item = Left::Item>,
{
    left_wrapper: ForEach<Left, LrDiff<Left::Item>>,
    right_wrapper: ForEach<Right, LrDiff<Left::Item>>,
    signal: LazySignal,
}

impl<Left, Right> SignalVec for Chain<Left, Right>
where
    Left: SignalVec,
    Right: SignalVec<Item = Left::Item>,
{
    type Item = Left::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        let SignalHandle(left_upstream) = self.left_wrapper.register(world);
        let SignalHandle(right_upstream) = self.right_wrapper.register(world);

        let signal = self.signal.register(world);

        pipe_signal(world, left_upstream, signal);
        pipe_signal(world, right_upstream, signal);

        signal.into()
    }
}

/// A node that intersperses a separator between adjacent items of a `SignalVec`.
#[derive(Clone)]
pub struct Intersperse<Upstream>
where
    Upstream: SignalVec,
{
    signal: ForEach<Upstream, Vec<VecDiff<Upstream::Item>>>,
}

impl<Upstream> SignalVec for Intersperse<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

#[derive(Component)]
struct IntersperseState<T> {
    /// The original, non-interspersed values. We need this to know the length.
    local_values: Vec<T>,
    /// Stable keys for the separators. The key at index `i` corresponds to the
    /// separator that comes *after* `local_values[i]`.
    separator_keys: Vec<usize>,
    /// Maps a separator's stable key to its current logical index (0, 1, 2...).
    key_to_index: HashMap<usize, usize>,
    /// A counter to generate new unique, stable keys.
    next_key: usize,
}

impl<T> Default for IntersperseState<T> {
    fn default() -> Self {
        Self {
            local_values: Vec::new(),
            separator_keys: Vec::new(),
            key_to_index: HashMap::new(),
            next_key: 0,
        }
    }
}

/// A node that intersperses a reactive separator generated by a system between
/// adjacent items of a `SignalVec`.
#[derive(Clone)]
pub struct IntersperseWith<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for IntersperseWith<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// A node that sorts the items of a `SignalVec` using a comparison system.
#[derive(Clone)]
pub struct SortBy<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for SortBy<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// A node that sorts the items of a `SignalVec` using an extracted key.
#[derive(Clone)]
pub struct SortByKey<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for SortByKey<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

#[derive(Clone)]
struct FlattenItem<O: SSs + Clone> {
    processor_handle: SignalHandle,
    values: Vec<O>,
    _marker: PhantomData<fn() -> O>,
}

#[derive(Component)]
struct FlattenData<O: SSs + Clone> {
    items: Vec<FlattenItem<O>>,
    diffs: Vec<VecDiff<O>>,
}

fn with_flatten_data<O: SSs + Clone, R>(
    world: &mut World,
    parent: Entity,
    f: impl FnOnce(Mut<FlattenData<O>>) -> R,
) -> R {
    f(world.get_mut::<FlattenData<O>>(parent).unwrap())
}

#[derive(Component, Deref, DerefMut)]
struct FlattenInnerIndex(usize);

#[allow(clippy::type_complexity)]
fn create_flatten_processor<O: Clone + SSs>(
    entity: LazyEntity,
    parent: Entity,
) -> impl Fn(In<Vec<VecDiff<O>>>, Query<&FlattenInnerIndex>, Query<&mut FlattenData<O>>) {
    move |In(diffs): In<Vec<VecDiff<O>>>,
          query_self_index: Query<&FlattenInnerIndex>,
          mut query_parent_data: Query<&mut FlattenData<O>>| {
        if let Ok(mut parent_data) = query_parent_data.get_mut(parent)
            && let Ok(&FlattenInnerIndex(self_index)) = query_self_index.get(entity.get())
        {
            if self_index >= parent_data.items.len() {
                return;
            }
            let offset: usize = parent_data.items[..self_index]
                .iter()
                .map(|item| item.values.len())
                .sum();

            for diff in diffs {
                let apply_and_queue = |pd: &mut Mut<FlattenData<O>>, d: VecDiff<O>| {
                    let translated_diff = match d {
                        VecDiff::Replace { values } => {
                            let old_len = pd.items[self_index].values.len();
                            for _ in 0..old_len {
                                pd.diffs.push(VecDiff::RemoveAt { index: offset });
                            }
                            for (i, v) in values.iter().enumerate() {
                                pd.diffs.push(VecDiff::InsertAt {
                                    index: offset + i,
                                    value: v.clone(),
                                });
                            }
                            pd.items[self_index].values = values;
                            return;
                        }
                        VecDiff::InsertAt { index, value } => {
                            pd.items[self_index].values.insert(index, value.clone());
                            VecDiff::InsertAt {
                                index: index + offset,
                                value,
                            }
                        }
                        VecDiff::UpdateAt { index, value } => {
                            pd.items[self_index].values[index] = value.clone();
                            VecDiff::UpdateAt {
                                index: index + offset,
                                value,
                            }
                        }
                        VecDiff::RemoveAt { index } => {
                            pd.items[self_index].values.remove(index);
                            VecDiff::RemoveAt { index: index + offset }
                        }
                        VecDiff::Move { old_index, new_index } => {
                            let val = pd.items[self_index].values.remove(old_index);
                            pd.items[self_index].values.insert(new_index, val);
                            VecDiff::Move {
                                old_index: old_index + offset,
                                new_index: new_index + offset,
                            }
                        }
                        VecDiff::Push { value } => {
                            let old_len = pd.items[self_index].values.len();
                            pd.items[self_index].values.push(value.clone());
                            VecDiff::InsertAt {
                                index: offset + old_len,
                                value,
                            }
                        }
                        VecDiff::Pop => {
                            pd.items[self_index].values.pop();
                            VecDiff::RemoveAt {
                                index: offset + pd.items[self_index].values.len(),
                            }
                        }
                        VecDiff::Clear => {
                            let old_len = pd.items[self_index].values.len();
                            pd.items[self_index].values.clear();
                            for _ in 0..old_len {
                                pd.diffs.push(VecDiff::RemoveAt { index: offset });
                            }
                            return;
                        }
                    };
                    pd.diffs.push(translated_diff);
                };
                apply_and_queue(&mut parent_data, diff);
            }
        }
    }
}

fn spawn_flatten_item<O: Clone + SSs>(
    world: &mut World,
    parent: Entity,
    index: usize,
    // Fix: Add `Clone` bound to allow using the signal twice.
    inner_signal: impl SignalVec<Item = O> + 'static + Clone,
) -> (FlattenItem<O>, Vec<O>) {
    // Use the cloned signal for the temporary state getter.
    let temp_get_state_signal = inner_signal.clone().to_signal().first();
    let handle = temp_get_state_signal.register(world);
    let initial_values = poll_signal(world, *handle)
        .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<Vec<O>>().ok())
        .map(|b| *b)
        .unwrap_or_default();
    handle.cleanup(world);

    // Use the original signal for the persistent processor.
    let entity = LazyEntity::new();
    let processor_system = create_flatten_processor(entity.clone(), parent);
    let processor_handle = inner_signal.for_each(processor_system).register(world);
    entity.set(
        world
            .entity_mut(**processor_handle)
            .insert(FlattenInnerIndex(index))
            .id(),
    );

    let item = FlattenItem {
        processor_handle,
        values: initial_values.clone(),
        _marker: PhantomData,
    };
    (item, initial_values)
}

/// A node that flattens a `SignalVec` of `SignalVec`s.
#[derive(Clone)]
pub struct Flatten<Upstream> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for Flatten<Upstream>
where
    Upstream: SignalVec,
    Upstream::Item: SignalVec + 'static,
    <Upstream::Item as SignalVec>::Item: Clone + SSs,
{
    type Item = <Upstream::Item as SignalVec>::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

// The Memoize struct remains the same.
#[derive(Clone)]
pub struct Replayable<Upstream>
where
    Upstream: SignalVec,
{
    signal: LazySignal,
    _marker: PhantomData<fn() -> Upstream>,
}

impl<Upstream> SignalVec for Replayable<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

#[derive(Clone)]
pub struct Debug<Upstream>
where
    Upstream: SignalVec,
{
    pub(crate) signal: ForEach<Upstream, Vec<VecDiff<Upstream::Item>>>,
}

impl<Upstream> SignalVec for Debug<Upstream>
where
    Upstream: SignalVec,
{
    type Item = Upstream::Item;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world)
    }
}

/// Extension trait providing combinator methods for [`SignalVec`]s.
pub trait SignalVecExt: SignalVec {
    /// Pass the "raw" [`Vec<VecDiff<Self::Item>>`] output of this [`SignalVec`] to a [`System`],
    /// continuing propagation if the [`System`] returns [`Some`] or terminating for the frame if it
    /// returns [`None`]. Unlike most other [`SignalVec`] methods,
    /// [`.for_each`](SignalVecExt::for_each), returns a [`Signal`], not a [`SignalVec`], since the
    /// output type need not be an [`Option<Vec<VecDiff>>`]. If the [`System`] logic is infallible,
    /// wrapping the result in an option is unnecessary.
    fn for_each<O, IOO, F, M>(self, system: F) -> ForEach<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        IOO: Into<Option<O>> + 'static,
        F: IntoSystem<In<Vec<VecDiff<Self::Item>>>, IOO, M> + SSs,
    {
        ForEach {
            upstream: self,
            signal: lazy_signal_from_system(system),
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`], transforming it.
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from([1, 2, 3]).signal_vec().map(|In(x): In<i32>| x * 2); // outputs `SignalVec -> [2, 4, 6]`
    /// ```
    fn map<O, F, M>(self, system: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        F: IntoSystem<In<Self::Item>, O, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(system);
            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<O>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                        let mut output: Vec<VecDiff<O>> = Vec::new();

                        for diff in diffs {
                            let diff_option: Option<VecDiff<O>> =
                                match diff {
                                    VecDiff::Replace { values } => {
                                        let mapped_values: Vec<O> = values
                                            .into_iter()
                                            .filter_map(|v| world.run_system_with(system, v).ok())
                                            .collect();
                                        Some(VecDiff::Replace { values: mapped_values })
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
                                        .map(|mapped_value| VecDiff::Push { value: mapped_value }),
                                    VecDiff::RemoveAt { index } => Some(VecDiff::RemoveAt { index }),
                                    VecDiff::Move { old_index, new_index } => {
                                        Some(VecDiff::Move { old_index, new_index })
                                    }
                                    VecDiff::Pop => Some(VecDiff::Pop),
                                    VecDiff::Clear => Some(VecDiff::Clear),
                                };

                            if let Some(diff) = diff_option {
                                output.push(diff);
                            }
                        }

                        if output.is_empty() { None } else { Some(output) }
                    },
                )
                .register(world);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal
        });

        Map {
            signal,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to an [`FnMut`], transforming it.
    ///
    /// Convenient when additional [`SystemParam`](bevy_ecs::system::SystemParams)s aren't
    /// necessary.
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from([1, 2, 3]).signal_vec().map_in(|x: i32| x * 2); // outputs `SignalVec -> [2, 4, 6]`
    /// ```
    fn map_in<O, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        F: FnMut(Self::Item) -> O + SSs,
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
    /// ```no_run
    /// MutableVec::from([1, 2, 3]).signal_vec().map_in_ref(ToString::to_string); // outputs `SignalVec -> ["1", "2", "3"]`
    /// ```
    fn map_in_ref<O, F>(self, mut function: F) -> Map<Self, O>
    where
        Self: Sized,
        Self::Item: 'static,
        O: Clone + 'static,
        F: FnMut(&Self::Item) -> O + SSs,
    {
        self.map(move |In(item)| function(&item))
    }

    // /// Creates a new `SignalVec` by mapping each item of the source `SignalVec` to a `Signal`.
    // ///
    // /// For each item in the source vector, the provided `system` is run to produce an
    // /// inner `Signal`. The output vector's value at a given index becomes the latest
    // /// value emitted by the corresponding inner `Signal`.
    // ///
    // /// This is useful for creating dynamic lists where each element has its own
    // /// independent, reactive state.
    // fn map_signal<S, F, M>(self, system: F) -> MapSignal<Self, S>
    // where
    //     Self: Sized,
    //     Self::Item: Clone + SSs,
    //     S: Signal + 'static + Clone,
    //     S::Item: Clone + Send + Sync,
    //     F: IntoSystem<In<Self::Item>, S, M> + SSs,
    // {
    //     /// A helper to spawn a forwarding processor for a new inner signal.
    //     /// Returns a handle for cleanup, the inner signal's canonical ID, and its initial value.
    //     fn spawn_processor<S: Signal + Clone>(
    //         world: &mut World,
    //         output_system: SignalSystem,
    //         index: usize,
    //         inner_signal: S,
    //     ) -> (SignalHandle, SignalSystem, S::Item)
    //     where
    //         S::Item: Clone + Send + Sync,
    //     {
    //         let inner_signal_id = inner_signal.clone().register(world);

    //         let temp_handle = inner_signal.clone().first().register(world);
    //         let initial_value = poll_signal(world, *temp_handle)
    //             .and_then(|any_val| (any_val as Box<dyn Any>).downcast::<S::Item>().ok())
    //             .map(|b| *b)
    //             .expect("map_signal's inner signal must emit an initial value");
    //         temp_handle.cleanup(world);

    //         let self_entity = LazyEntity::new();

    //         let processor = clone!((self_entity) move |In(value): In<S::Item>, world: &mut World| {
    //             if let Some(&MappedSignalIndex(index)) =
    // world.get::<MappedSignalIndex>(self_entity.get()) {                 let diff =
    // VecDiff::UpdateAt { index, value };                 process_signals_helper(world,
    // [output_system].into_iter(), Box::new(vec![diff]));             }
    //             None::<()>
    //         });

    //         let processor_handle = inner_signal.map::<(), _, _, _>(processor).register(world);
    //         self_entity.set(**processor_handle);
    //         world
    //             .entity_mut(**processor_handle)
    //             .insert(MappedSignalIndex(index));

    //         // Fix E0308: Return the `SignalSystem` ID directly.
    //         (processor_handle, *inner_signal_id, initial_value)
    //     }

    //     let signal = LazySignal::new(move |world: &mut World| {
    //         let system_id = world.register_system(system);
    //         let output_system =
    //             lazy_signal_from_system(|In(diffs): In<Vec<VecDiff<S::Item>>>| diffs)
    //                 .register(world);
    //         world
    //             .entity_mut(*output_system)
    //             .insert(Upstream(HashSet::new())); // we don't want this to be run as a root system

    //         /// Local state for the manager system.
    //         #[derive(Clone)]
    //         struct ProcessorState {
    //             handle: SignalHandle,
    //             inner_signal_id: SignalSystem,
    //         }

    //         let manager_system = self
    //             .for_each(
    //                 move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
    //                       world: &mut World,
    //                       mut processors: Local<Vec<ProcessorState>>,
    //                       mut is_not_initial: Local<bool>| {
    //                     let mut new_diffs: Vec<VecDiff<S::Item>> = vec![];

    //                     for diff in diffs {
    //                         match diff {
    //                             VecDiff::Replace { values } => {
    //                                 for processor in processors.drain(..) {
    //                                     processor.handle.cleanup(world);
    //                                 }

    //                                 let mut new_values = Vec::new();
    //                                 for (i, value) in values.into_iter().enumerate() {
    //                                     if let Ok(signal) = world.run_system_with(system_id, value)
    //                                     {
    //                                         let (handle, inner_signal_id, initial_value) =
    //                                             spawn_processor(world, output_system, i, signal);
    //                                         processors.push(ProcessorState {
    //                                             handle,
    //                                             inner_signal_id,
    //                                         });
    //                                         new_values.push(initial_value);
    //                                     }
    //                                 }
    //                                 new_diffs.push(VecDiff::Replace { values: new_values });
    //                             }
    //                             VecDiff::InsertAt { index, value } => {
    //                                 if let Ok(signal) = world.run_system_with(system_id, value) {
    //                                     let (handle, inner_signal_id, initial_value) =
    //                                         spawn_processor(world, output_system, index, signal);
    //                                     processors.insert(
    //                                         index,
    //                                         ProcessorState {
    //                                             handle,
    //                                             inner_signal_id,
    //                                         },
    //                                     );

    //                                     // Fix E0308: Dereference twice to get the Entity ID.
    //                                     for (i, item_to_update) in
    //                                         processors.iter().enumerate().skip(index + 1)
    //                                     {
    //                                         if let Some(mut idx) = world
    //                                             .get_mut::<MappedSignalIndex>(
    //                                                 **item_to_update.handle,
    //                                             )
    //                                         {
    //                                             idx.0 = i;
    //                                         }
    //                                     }
    //                                     new_diffs.push(VecDiff::InsertAt {
    //                                         index,
    //                                         value: initial_value,
    //                                     });
    //                                 }
    //                             }
    //                             VecDiff::UpdateAt { index, value } => {
    //                                 if let Ok(new_inner_signal) =
    //                                     world.run_system_with(system_id, value)
    //                                 {
    //                                     let new_inner_id = new_inner_signal.clone().register(world);

    //                                     // Fix E0308: Compare SignalSystem to SignalSystem.
    //                                     if processors[index].inner_signal_id == *new_inner_id {
    //                                         // Fix E0308: Construct handle from the system ID.
    //                                         new_inner_id.cleanup(world);
    //                                     } else {
    //                                         let old_processor = processors.remove(index);
    //                                         old_processor.handle.cleanup(world);

    //                                         let (new_handle, new_id, initial_value) =
    //                                             spawn_processor(
    //                                                 world,
    //                                                 output_system,
    //                                                 index,
    //                                                 new_inner_signal,
    //                                             );
    //                                         processors.insert(
    //                                             index,
    //                                             ProcessorState {
    //                                                 handle: new_handle,
    //                                                 inner_signal_id: new_id,
    //                                             },
    //                                         );
    //                                         new_diffs.push(VecDiff::UpdateAt {
    //                                             index,
    //                                             value: initial_value,
    //                                         });
    //                                     }
    //                                 }
    //                             }
    //                             VecDiff::RemoveAt { index } => {
    //                                 let processor = processors.remove(index);
    //                                 processor.handle.cleanup(world);

    //                                 // Fix E0308: Dereference twice to get the Entity ID.
    //                                 for (i, item_to_update) in
    //                                     processors.iter().enumerate().skip(index)
    //                                 {
    //                                     if let Some(mut idx) = world
    //                                         .get_mut::<MappedSignalIndex>(**item_to_update.handle)
    //                                     {
    //                                         idx.0 = i;
    //                                     }
    //                                 }
    //                                 new_diffs.push(VecDiff::RemoveAt { index });
    //                             }
    //                             VecDiff::Move {
    //                                 old_index,
    //                                 new_index,
    //                             } => {
    //                                 let processor = processors.remove(old_index);
    //                                 processors.insert(new_index, processor);

    //                                 let start = old_index.min(new_index);
    //                                 let end = old_index.max(new_index) + 1;
    //                                 // Fix E0308: Dereference twice to get the Entity ID.
    //                                 for (i, item_to_update) in
    //                                     processors.iter().enumerate().skip(start).take(end - start)
    //                                 {
    //                                     if let Some(mut idx) = world
    //                                         .get_mut::<MappedSignalIndex>(**item_to_update.handle)
    //                                     {
    //                                         idx.0 = i;
    //                                     }
    //                                 }
    //                                 new_diffs.push(VecDiff::Move {
    //                                     old_index,
    //                                     new_index,
    //                                 });
    //                             }
    //                             VecDiff::Push { value } => {
    //                                 let index = processors.len();
    //                                 if let Ok(signal) = world.run_system_with(system_id, value) {
    //                                     let (handle, inner_signal_id, initial_value) =
    //                                         spawn_processor(world, output_system, index, signal);
    //                                     processors.push(ProcessorState {
    //                                         handle,
    //                                         inner_signal_id,
    //                                     });
    //                                     new_diffs.push(VecDiff::Push {
    //                                         value: initial_value,
    //                                     });
    //                                 }
    //                             }
    //                             VecDiff::Pop => {
    //                                 if let Some(processor) = processors.pop() {
    //                                     processor.handle.cleanup(world);
    //                                     new_diffs.push(VecDiff::Pop);
    //                                 }
    //                             }
    //                             VecDiff::Clear => {
    //                                 for processor in processors.drain(..) {
    //                                     processor.handle.cleanup(world);
    //                                 }
    //                                 new_diffs.push(VecDiff::Clear);
    //                             }
    //                         }
    //                     }

    //                     if !new_diffs.is_empty() {
    //                         if *is_not_initial {
    //                             process_signals_helper(world, [output_system], Box::new(new_diffs));
    //                         } else {
    //                             *is_not_initial = true;
    //                             world.commands().queue(move |world: &mut World| {
    //                                 process_signals_helper(
    //                                     world,
    //                                     [output_system],
    //                                     Box::new(new_diffs),
    //                                 );
    //                             })
    //                         }
    //                     }
    //                 },
    //             )
    //             .register(world);

    //         world.entity_mut(*output_system).add_child(**manager_system);

    //         output_system
    //     });

    //     MapSignal {
    //         signal,
    //         _marker: PhantomData,
    //     }
    // }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`], only forwarding
    /// those which return `true`.
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from([1, 2, 3, 4]).signal_vec().filter(|In(x): In<i32>| x % 2 == 0); // outputs `SignalVec -> [2, 4]`
    /// ```
    fn filter<F, M>(self, predicate: F) -> Filter<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
        F: IntoSystem<In<Self::Item>, bool, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(predicate);
            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut indices: Local<Vec<bool>>| {
                        filter_helper(
                            world,
                            diffs,
                            system,
                            |item, include| {
                                if include { Some(item) } else { None }
                            },
                            &mut indices,
                        )
                    },
                )
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

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`], transforming it
    /// and only forwarding those which return `Some`.
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from(["1", "two", "NaN", "four", "5"]).signal_vec()
    ///     .filter_map(|In(s): In<&'static str>| s.parse().ok()); // outputs `SignalVec -> [1, 5]`
    /// ```
    fn filter_map<O, F, M>(self, system: F) -> FilterMap<Self, Self::Item>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
        O: Clone + 'static,
        F: IntoSystem<In<Self::Item>, Option<O>, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let system = world.register_system(system);
            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<O>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World, mut indices: Local<Vec<bool>>| {
                        filter_helper(world, diffs, system, |_, mapped| mapped, &mut indices)
                    },
                )
                .register(world);
            // just attach the system to the lifetime of the signal
            world.entity_mut(*signal).add_child(system.entity());
            signal
        });
        FilterMap {
            signal,
            _marker: PhantomData,
        }
    }

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`] that produces a
    /// [`Signal<Item = bool>`], only forwarding those whose [`Signal`] outputs `true`.
    ///
    /// # Example
    /// ```no_run
    /// #[derive(Resource, Clone)]
    /// struct Even(bool);
    ///
    /// MutableVec::from([1, 2, 3, 4]).signal_vec()
    ///      .filter_signal(|In(x): In<i32>, even: Res<Even>| {
    ///          x % 2 == if even.0 { 0 } else { 1 }
    ///      }); // outputs `SignalVec -> [2, 4]` when `Even(true)` and `SignalVec -> [1, 3]` when `Even(false)`
    /// ```
    fn filter_signal<F, S, M>(self, system: F) -> FilterSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        S: Signal<Item = bool> + 'static,
        F: IntoSystem<In<Self::Item>, S, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let parent_entity = LazyEntity::new();
            let system_id = world.register_system(system);

            let SignalHandle(output_signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(clone!((parent_entity) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                    let parent = parent_entity.get();
                    let mut new_diffs = vec![];

                    for diff in diffs.into_iter() {
                        match diff {
                            VecDiff::Replace { values } => {
                                let old_signals = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    data.items.drain(..).map(|item| item.signal).collect::<Vec<_>>()
                                });
                                for signal in old_signals {
                                    signal.cleanup(world);
                                }

                                let mut new_items = Vec::with_capacity(values.len());
                                let mut new_values = vec![];
                                for (i, value) in values.into_iter().enumerate() {
                                    if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                        let (handle, filtered) = spawn_filter_signal::<Self::Item>(world, i, signal, parent);
                                        if filtered {
                                            new_values.push(value.clone());
                                        }
                                        new_items.push(FilterSignalItem { signal: handle, value, filtered });
                                    }
                                }
                                with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    data.items = new_items;
                                    data.diffs.clear();
                                });
                                new_diffs.push(VecDiff::Replace { values: new_values });
                            },
                            VecDiff::InsertAt { index, value } => {
                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                    let (handle, filtered) = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                    let (new_filtered_index, signals_to_update) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                        data.items.insert(index, FilterSignalItem { signal: handle, value: value.clone(), filtered });
                                        (find_filter_signal_index(&data.items, index), data.items.get((index + 1)..).map(|items| items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>()))
                                    });
                                    if let Some(to_update) = signals_to_update {
                                        for signal in to_update {
                                            if let Some(mut signal_index) = world.get_mut::<FilterSignalIndex>(**signal) {
                                                **signal_index += 1;
                                            }
                                        }
                                    }
                                    if filtered {
                                        new_diffs.push(VecDiff::InsertAt { index: new_filtered_index, value });
                                    }
                                }
                            },
                            VecDiff::UpdateAt { index, value } => {
                                let (old_signal, old_filtered, filtered_index) = with_filter_signal_data(world, parent, |data: Mut<FilterSignalData<Self::Item>>| {
                                    let filtered_index = find_filter_signal_index(&data.items, index);
                                    let item = &data.items[index];
                                    (item.signal.clone(), item.filtered, filtered_index)
                                });

                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                    let (new_handle, new_filtered) = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                    old_signal.cleanup(world);
                                    with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                        let item = &mut data.items[index];
                                        item.signal = new_handle;
                                        item.value = value.clone();
                                        item.filtered = new_filtered;
                                    });
                                    if new_filtered {
                                        if old_filtered {
                                            new_diffs.push(VecDiff::UpdateAt { index: filtered_index, value });
                                        } else {
                                            new_diffs.push(VecDiff::InsertAt { index: filtered_index, value });
                                        }
                                    } else if old_filtered {
                                        new_diffs.push(VecDiff::RemoveAt { index: filtered_index });
                                    }
                                }
                            },
                            VecDiff::RemoveAt { index } => {
                                let (signal, filtered, filtered_index, signals_to_update) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    let filtered_index = find_filter_signal_index(&data.items, index);
                                    let item = data.items.remove(index);
                                    let signals_to_update = data.items.get(index..).map(|items| items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>());
                                    (item.signal, item.filtered, filtered_index, signals_to_update)
                                });
                                signal.cleanup(world);
                                if let Some(to_update) = signals_to_update {
                                    for signal in to_update {
                                        if let Some(mut signal_index) = world.get_mut::<FilterSignalIndex>(**signal) {
                                            **signal_index -= 1;
                                        }
                                    }
                                }
                                if filtered {
                                    new_diffs.push(VecDiff::RemoveAt { index: filtered_index });
                                }
                            },
                            VecDiff::Push { value } => {
                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                    let index = with_filter_signal_data(world, parent, |data: Mut<FilterSignalData<Self::Item>>| data.items.len());
                                    let (handle, filtered) = spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                    with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                        data.items.push(FilterSignalItem { signal: handle, value: value.clone(), filtered });
                                    });
                                    if filtered {
                                        new_diffs.push(VecDiff::Push { value });
                                    }
                                }
                            },
                            VecDiff::Pop => {
                                let (signal, filtered) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    let item = data.items.pop().expect("can't pop from empty vec");
                                    (item.signal, item.filtered)
                                });
                                signal.cleanup(world);
                                if filtered {
                                    new_diffs.push(VecDiff::Pop);
                                }
                            },
                            VecDiff::Move { old_index, new_index } => {
                                let (filtered, old_filtered_index, new_filtered_index, items_to_update) = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    let old_filtered_index = find_filter_signal_index(&data.items, old_index);
                                    let item = data.items.remove(old_index);
                                    let filtered = item.filtered;
                                    data.items.insert(new_index, item);
                                    let new_filtered_index = find_filter_signal_index(&data.items, new_index);
                                    // Collect the signal handles that need their indices updated.
                                    let items_to_update = data.items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>();
                                    (filtered, old_filtered_index, new_filtered_index, items_to_update)
                                });

                                // Now that the borrow from with_filter_signal_data is over, we can mutably access world.
                                for (i, signal) in items_to_update.into_iter().enumerate() {
                                    if let Some(mut signal_index) = world.get_mut::<FilterSignalIndex>(**signal) {
                                        **signal_index = i;
                                    }
                                }

                                if filtered {
                                    new_diffs.push(VecDiff::Move { old_index: old_filtered_index, new_index: new_filtered_index });
                                }
                            },
                            VecDiff::Clear => {
                                let signals = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| {
                                    data.diffs.clear();
                                    data.items.drain(..).map(|item| item.signal).collect::<Vec<_>>()
                                });
                                for signal in signals {
                                    signal.cleanup(world);
                                }
                                new_diffs.push(VecDiff::Clear);
                            },
                        }
                    }
                    let mut diffs = with_filter_signal_data(world, parent, |mut data: Mut<FilterSignalData<Self::Item>>| data.diffs.drain(..).collect::<Vec<_>>());
                    diffs.extend(new_diffs);
                    if diffs.is_empty() { None } else { Some(diffs) }
                }))
                .register(world);

            parent_entity.set(*output_signal);
            let SignalHandle(flusher) = SignalBuilder::from_entity(*output_signal)
                .map::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    |In(entity), filter_signal_datas: Query<&FilterSignalData<Self::Item>>| {
                        if let Ok(data) = filter_signal_datas.get(entity) {
                            if !data.diffs.is_empty() { Some(vec![]) } else { None }
                        } else {
                            None
                        }
                    },
                )
                .register(world);
            pipe_signal(world, flusher, output_signal);
            world
                .entity_mut(*output_signal)
                .insert(FilterSignalData::<Self::Item> {
                    items: vec![],
                    diffs: vec![],
                })
                .add_child(system_id.entity());
            output_signal
        });
        FilterSignal {
            signal,
            _marker: PhantomData,
        }
    }

    /// Transform each [`Item`](SignalVec::Item) into a tuple where the right item is the original
    /// item and the left item is a [`Signal<Item = Option<usize>>`] which outputs the item's index
    /// or [`None`] if it was removed.
    ///
    /// The [`Signal<Item = Option<usize>>`]s are deduped so any [`Downstream`] [`Signal`]s are only
    /// run on frames where the index has changed.
    ///
    /// # Example
    /// ```no_run
    /// let mut vec = MutableVec::from([1, 2, 5])
    /// let signal = vec.signal_vec().enumerate();
    /// signal; // outputs `SignalVec -> [(Signal -> Some(0), 1), (Signal -> Some(1), 2), (Signal -> Some(2), 5)]`
    /// vec.write().remove(1);
    /// commands.queue(vec.flush());
    /// signal; // outputs `SignalVec -> [(Signal -> Some(0), 1), (Signal -> Some(1), 5)]`
    /// ```
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let processor_entity_handle = LazyEntity::new();
            let processor_logic = clone!((processor_entity_handle) move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                                                     world: &mut World| {
                let processor_entity = processor_entity_handle.get();
                let mut state = world.get_mut::<EnumerateState>(processor_entity).unwrap();
                let mut out_diffs = vec![];

                fn new_key_helper(state: &mut EnumerateState) -> usize {
                    loop {
                        let key = state.next_key;
                        state.next_key = state.next_key.wrapping_add(1);
                        if !state.key_to_index.contains_key(&key) {
                            return key;
                        }
                    }
                }

                let create_index_signal = clone!((processor_entity_handle) move |key: usize| {
                    SignalBuilder::from_system(clone!((processor_entity_handle) move |_: In<()>, query: Query<&EnumerateState>| {
                        Some(query
                            .get(processor_entity_handle.get())
                            .ok()
                            .and_then(|s| s.key_to_index.get(&key).copied()))
                    }))
                    .dedupe()
                });

                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values } => {
                            state.key_to_index.clear();
                            state.ordered_keys.clear();

                            let new_values_with_signals = values.into_iter().enumerate().map(|(i, value)| {
                                let key = new_key_helper(&mut state);
                                state.ordered_keys.push(key);
                                state.key_to_index.insert(key, i);
                                (create_index_signal(key), value)
                            }).collect();
                            out_diffs.push(VecDiff::Replace { values: new_values_with_signals });
                        }
                        VecDiff::InsertAt { index, value } => {
                            let key = new_key_helper(&mut state);
                            state.ordered_keys.insert(index, key);
                            for (i, k) in state.ordered_keys.iter().copied().enumerate().skip(index).collect::<Vec<_>>() {
                                state.key_to_index.insert(k, i);
                            }

                            out_diffs.push(VecDiff::InsertAt {
                                index,
                                value: (create_index_signal(key), value),
                            });
                        }
                        VecDiff::UpdateAt { index, value } => {
                            let key = state.ordered_keys[index];
                            out_diffs.push(VecDiff::UpdateAt {
                                index,
                                value: (create_index_signal(key), value),
                            });
                        }
                        VecDiff::RemoveAt { index } => {
                            let removed_key = state.ordered_keys.remove(index);
                            state.key_to_index.remove(&removed_key);
                            for (i, k) in state.ordered_keys.iter().copied().enumerate().skip(index).collect::<Vec<_>>() {
                                state.key_to_index.insert(k, i);
                            }

                            out_diffs.push(VecDiff::RemoveAt { index });
                        }
                        VecDiff::Move { old_index, new_index } => {
                            let key = state.ordered_keys.remove(old_index);
                            state.ordered_keys.insert(new_index, key);

                            let start = old_index.min(new_index);
                            let end = old_index.max(new_index) + 1;
                            for (i, k) in state.ordered_keys.iter().copied().enumerate().skip(start).take(end - start).collect::<Vec<_>>() {
                                state.key_to_index.insert(k, i);
                            }

                            out_diffs.push(VecDiff::Move { old_index, new_index });
                        }
                        VecDiff::Push { value } => {
                            let key = new_key_helper(&mut state);
                            let index = state.ordered_keys.len();
                            state.ordered_keys.push(key);
                            state.key_to_index.insert(key, index);

                            out_diffs.push(VecDiff::Push {
                                value: (create_index_signal(key), value),
                            });
                        }
                        VecDiff::Pop => {
                            if let Some(key) = state.ordered_keys.pop() {
                                state.key_to_index.remove(&key);
                                out_diffs.push(VecDiff::Pop);
                            }
                        }
                        VecDiff::Clear => {
                            state.ordered_keys.clear();
                            state.key_to_index.clear();
                            out_diffs.push(VecDiff::Clear);
                        }
                    }
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            });

            let handle = self
                .for_each::<Vec<VecDiff<(Dedupe<super::signal::Source<Option<usize>>>, Self::Item)>>, _, _, _>(
                    processor_logic,
                )
                .register(world);
            processor_entity_handle.set(**handle);
            world.entity_mut(**handle).insert(EnumerateState::default());
            *handle
        });

        Enumerate {
            signal,
            _marker: PhantomData,
        }
    }

    /// Collect this [`SignalVec`]'s [`Item`](SignalVec::Item)s into a [`Vec`], transforming it into
    /// an [`Signal<Item = Vec<Item>>`].
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from([1, 2, 3]).signal_vec().to_signal(|In(vec): In<Vec<i32>>| vec.position(|&x| x == 2)); // outputs `1`
    /// ```
    fn to_signal(self) -> ToSignal<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + 'static,
    {
        let signal = self.for_each(|In(diffs), mut values: Local<Vec<Self::Item>>| {
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
            values.clone()
        });
        ToSignal { signal }
    }

    /// Transform this [`SignalVec`] into a [`Signal<Item = bool>`] which outputs whether it's
    /// populated.
    ///
    /// # Example
    /// ```
    /// let mut vec = MutableVec::from([1]);
    /// let signal = vec.signal_vec().is_empty();
    /// signal; // outputs `false`
    /// vec.write().pop();
    /// commands.queue(vec.flush());
    /// signal; // outputs `true`
    /// ```
    #[allow(clippy::wrong_self_convention)]
    fn is_empty(self) -> IsEmpty<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + 'static,
    {
        IsEmpty {
            signal: self.to_signal().map(|In(v): In<Vec<Self::Item>>| v.is_empty()),
        }
    }

    /// Transform this [`SignalVec`] into a [`Signal<Item = usize>`] which outputs its length.
    ///
    /// # Example
    /// ```
    /// let mut vec = MutableVec::from([1]);
    /// let signal = vec.signal_vec().is_empty();
    /// signal; // outputs `1`
    /// vec.write().pop();
    /// commands.queue(vec.flush());
    /// signal; // outputs `0`
    /// ```
    fn len(self) -> Len<Self>
    where
        Self: Sized,
        Self::Item: Clone + Send + 'static,
    {
        Len {
            signal: self.to_signal().map(|In(v): In<Vec<Self::Item>>| v.len()),
        }
    }

    /// Transform this [`SignalVec`] into a [`Signal<Item = Self::Item>`] which outputs its sum.
    ///
    /// # Example
    /// ```
    /// let mut vec = MutableVec::from([1, 2, 3]);
    /// let signal = vec.signal_vec().is_empty();
    /// signal; // outputs `6`
    /// vec.write().push(4);
    /// commands.queue(vec.flush());
    /// signal; // outputs `10`
    /// ```
    fn sum(self) -> Sum<Self>
    where
        Self: Sized,
        Self::Item: for<'a> core::iter::Sum<&'a Self::Item> + Clone + Send + 'static,
    {
        Sum {
            signal: self
                .to_signal()
                .map(|In(v): In<Vec<Self::Item>>| v.iter().sum::<Self::Item>()),
        }
    }

    /// Chains this [`SignalVec`] with another [`SignalVec`], producing a [`SignalVec`] with all the
    /// [`Item`](SignalVec::Item)s of `self`, followed by all the [`Item`](SignalVec::Item)s of
    /// `other`.
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from([1, 3]).signal_vec().chain(MutableVec::from([2, 4]).signal_vec()); // outputs `SignalVec -> [1, 3, 2, 4]`
    /// ```
    fn chain<S>(self, other: S) -> Chain<Self, S>
    where
        S: SignalVec<Item = Self::Item>,
        Self: Sized,
        Self::Item: SSs + Clone,
    {
        let left_wrapper = self.for_each(|In(diffs)| LrDiff::Left(diffs));
        let right_wrapper = other.for_each(|In(diffs)| LrDiff::Right(diffs));

        let signal = lazy_signal_from_system::<_, Vec<VecDiff<Self::Item>>, _, _, _>(
            move |In(lr_diff): In<LrDiff<Self::Item>>, mut left_len: Local<usize>, mut right_len: Local<usize>| {
                let mut out_diffs = Vec::new();

                let (is_left, diffs) = match lr_diff {
                    LrDiff::Left(diffs) => (true, diffs),
                    LrDiff::Right(diffs) => (false, diffs),
                };

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
                        // is_right
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
            },
        );

        Chain {
            left_wrapper,
            right_wrapper,
            signal,
        }
    }

    /// Places a [`Clone`] of `separator` between adjacent items of this [`SignalVec`].
    ///
    /// # Example
    /// ```no_run
    /// MutableVec::from([1, 2, 3]).signal_vec().intersperse(0); // outputs `SignalVec -> [1, 0, 2, 0, 3]`
    /// ```
    fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
    {
        let signal = self.for_each(
            move |In(diffs): In<Vec<VecDiff<Self::Item>>>, mut local_values: Local<Vec<Self::Item>>| {
                let mut out_diffs = Vec::new();

                for diff in diffs {
                    let old_len = local_values.len();
                    match diff {
                        VecDiff::Replace { mut values } => {
                            let new_len = values.len();

                            // Correctly set the new state *first*.
                            local_values.clear();
                            local_values.append(&mut values);

                            let mut interspersed = Vec::new();
                            if new_len > 0 {
                                interspersed.reserve(2 * new_len - 1);
                                // Iterate over our now-correct local state.
                                let mut iter = local_values.iter().cloned();
                                interspersed.push(iter.next().unwrap());
                                for item in iter {
                                    interspersed.push(separator.clone());
                                    interspersed.push(item);
                                }
                            }
                            out_diffs.push(VecDiff::Replace { values: interspersed });

                            // The buggy line `*local_values = values;` is removed.
                            // `local_values` now correctly holds the state for the next diff.
                        }
                        VecDiff::InsertAt { index, value } => {
                            local_values.insert(index, value.clone());
                            if old_len == 0 {
                                out_diffs.push(VecDiff::Push { value });
                            } else {
                                out_diffs.push(VecDiff::InsertAt {
                                    index: 2 * index,
                                    value,
                                });
                                // Insert separator *after* unless the item is at the very end.
                                if index < old_len {
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: 2 * index + 1,
                                        value: separator.clone(),
                                    });
                                }
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
                                out_diffs.push(VecDiff::RemoveAt { index: 0 });
                            } else if index == old_len - 1 {
                                // Removing the last item: remove the item and the separator *before* it.
                                out_diffs.push(VecDiff::Pop); // item
                                out_diffs.push(VecDiff::Pop); // separator
                            } else {
                                // Removing from start/middle: remove the item and the separator *after* it.
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // item
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // separator
                            }
                        }
                        VecDiff::Move { old_index, new_index } => {
                            let value = local_values.remove(old_index);
                            local_values.insert(new_index, value.clone());

                            // Decompose move into remove + insert for robustness.
                            let mut temp_out = Vec::new();
                            // 1. Generate remove diffs based on old position
                            if old_len == 1 { // Nothing to remove
                            } else if old_index == old_len - 1 {
                                temp_out.push(VecDiff::Pop);
                                temp_out.push(VecDiff::Pop);
                            } else {
                                temp_out.push(VecDiff::RemoveAt { index: 2 * old_index });
                                temp_out.push(VecDiff::RemoveAt { index: 2 * old_index });
                            }

                            // 2. Generate insert diffs based on new position
                            let current_len = local_values.len();
                            if current_len == 1 {
                                temp_out.push(VecDiff::Push { value });
                            } else {
                                temp_out.push(VecDiff::InsertAt {
                                    index: 2 * new_index,
                                    value,
                                });
                                if new_index < current_len - 1 {
                                    temp_out.push(VecDiff::InsertAt {
                                        index: 2 * new_index + 1,
                                        value: separator.clone(),
                                    });
                                }
                            }
                            out_diffs.extend(temp_out);
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
                            // This is guaranteed to be removing the last item.
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
        );
        Intersperse { signal }
    }

    /// Creates a new `SignalVec` by placing a reactive item, generated by a `separator_factory`
    /// system, between adjacent items of the original `SignalVec`.
    ///
    /// The factory system is provided with a reactive `IndexSignal` (`Signal<Item = Option<usize>>`)
    /// which emits the logical index of the separator whenever it changes. This allows the
    /// separator itself to be a fully reactive component.
    ///
    /// This is the most powerful way to create separators and is analogous to `.enumerate()`.
    ///
    /// # Example
    /// ```no_run
    /// my_list.signal_vec()
    ///     .intersperse_with(|In(index_signal): In<IndexSignal>| {
    ///         // This JonmoBuilder is now reactive to its own position.
    ///         JonmoBuilder::from(Node::default())
    ///             .component_signal(index_signal.map_in(|idx| Text::new(format!("Separator {}", idx.unwrap_or(0)))))
    ///     })
    /// ```
    fn intersperse_with<R, F, M>(self, separator_factory: F) -> IntersperseWith<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        R: Into<Self::Item> + SSs,
        F: IntoSystem<In<Dedupe<super::signal::Source<Option<usize>>>>, R, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            // 1. Register the user's factory system once.
            let factory_system_id = world.register_system(separator_factory);

            // 2. Create a handle for the state-holding entity.
            let state_entity_handle = LazyEntity::new();

            // 3. Define the helper to create an index signal for a given separator key.
            let create_index_signal =
                clone!((state_entity_handle) move |key: usize| {
                    SignalBuilder::from_system(
                        clone!((state_entity_handle) move |_: In<()>, query: Query<&IntersperseState<Self::Item>>| {
                            query
                                .get(state_entity_handle.get())
                                .ok()
                                .and_then(|s| s.key_to_index.get(&key).copied())
                        }),
                    )
                    .dedupe()
                });

            // 4. Define the main processor logic that handles incoming diffs.
            let processor_logic = clone!((state_entity_handle, create_index_signal) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                let state_entity = state_entity_handle.get();
                let mut out_diffs = Vec::new();

                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values } => {
                            // Phase 1 & 2: Get keys, create signals, and run factory systems
                            let (new_keys, separators) = {
                                let num_separators = values.len().saturating_sub(1);
                                let (keys, next_key_after) = {
                                    let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                    let start_key = state.next_key;
                                    state.next_key += num_separators;
                                    ((start_key..state.next_key).collect::<Vec<_>>(), state.next_key)
                                };

                                let seps = keys.iter().map(|&key| {
                                    world.run_system_with(factory_system_id, create_index_signal(key)).unwrap().into()
                                }).collect::<Vec<Self::Item>>();
                                (keys, seps)
                            };

                            // Phase 3: Update state and assemble output diff
                            let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                            state.local_values = values.clone();
                            state.separator_keys = new_keys;
                            state.key_to_index.clear();
                            for (i, key) in state.separator_keys.iter().copied().enumerate().collect::<Vec<_>>() {
                                state.key_to_index.insert(key, i);
                            }

                            let mut interspersed = Vec::new();
                            if !values.is_empty() {
                                let mut items_iter = values.into_iter();
                                interspersed.push(items_iter.next().unwrap());
                                for (sep, item) in separators.into_iter().zip(items_iter) {
                                    interspersed.push(sep);
                                    interspersed.push(item);
                                }
                            }
                            out_diffs.push(VecDiff::Replace { values: interspersed });
                        }
                        VecDiff::InsertAt { index, value } => {
                            let old_len = world.get::<IntersperseState<Self::Item>>(state_entity).unwrap().local_values.len();
                            out_diffs.push(VecDiff::InsertAt { index: 2 * index, value: value.clone() });

                            if index < old_len {
                                let key = {
                                    let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                    let key = state.next_key;
                                    state.next_key += 1;
                                    state.separator_keys.insert(index, key);
                                    key
                                };
                                let separator: Self::Item = world.run_system_with(factory_system_id, create_index_signal(key)).unwrap().into();
                                out_diffs.push(VecDiff::InsertAt { index: 2 * index + 1, value: separator });
                            }

                            let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                            state.local_values.insert(index, value);
                            for (i, k) in state.separator_keys.iter().copied().enumerate().skip(index).collect::<Vec<_>>() {
                                state.key_to_index.insert(k, i);
                            }
                        }
                        VecDiff::UpdateAt { index, value } => {
                            world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap().local_values[index] = value.clone();
                            out_diffs.push(VecDiff::UpdateAt { index: 2 * index, value });
                        }
                        VecDiff::RemoveAt { index } => {
                            let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                            state.local_values.remove(index);
                            out_diffs.push(VecDiff::RemoveAt { index: 2 * index });

                            if index < state.separator_keys.len() {
                                let removed_key = state.separator_keys.remove(index);
                                state.key_to_index.remove(&removed_key);
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index });
                                for (i, k) in state.separator_keys.iter().copied().enumerate().skip(index).collect::<Vec<_>>() {
                                    state.key_to_index.insert(k, i);
                                }
                            }
                        }
                        VecDiff::Push { value } => {
                            let old_len = world.get::<IntersperseState<Self::Item>>(state_entity).unwrap().local_values.len();

                            if old_len > 0 {
                                let key = {
                                    let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                    let key = state.next_key;
                                    state.next_key += 1;
                                    state.separator_keys.push(key);
                                    state.key_to_index.insert(key, old_len -1);
                                    key
                                };
                                let separator: Self::Item = world.run_system_with(factory_system_id, create_index_signal(key)).unwrap().into();
                                out_diffs.push(VecDiff::Push { value: separator });
                            }
                            world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap().local_values.push(value.clone());
                            out_diffs.push(VecDiff::Push { value });
                        }
                        VecDiff::Pop => {
                             let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                             if state.local_values.pop().is_some() {
                                out_diffs.push(VecDiff::Pop);
                                if let Some(removed_key) = state.separator_keys.pop() {
                                    state.key_to_index.remove(&removed_key);
                                    out_diffs.push(VecDiff::Pop);
                                }
                             }
                        }
                        VecDiff::Clear => {
                            let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                            state.local_values.clear();
                            state.separator_keys.clear();
                            state.key_to_index.clear();
                            out_diffs.push(VecDiff::Clear);
                        }
                        VecDiff::Move { old_index, new_index } => {
                            // `Move` is complex. Re-calculating the entire interspersed list via
                            // `Replace` is the safest way to handle it without subtle bugs.
                            let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                            let value = state.local_values.remove(old_index);
                            state.local_values.insert(new_index, value);
                            let values = state.local_values.clone();
                            drop(state); // release borrow

                            // Now we can re-use the `Replace` logic on the new `values`
                             let (new_keys, separators) = {
                                let num_separators = values.len().saturating_sub(1);
                                let (keys, _) = {
                                    let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                    let start_key = state.next_key;
                                    state.next_key += num_separators;
                                    ((start_key..state.next_key).collect::<Vec<_>>(), state.next_key)
                                };
                                let seps = keys.iter().map(|&key| {
                                    world.run_system_with(factory_system_id, create_index_signal(key)).unwrap().into()
                                }).collect::<Vec<Self::Item>>();
                                (keys, seps)
                            };

                            let mut state = world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                            state.separator_keys = new_keys;
                            state.key_to_index.clear();
                            for (i, key) in state.separator_keys.iter().copied().enumerate().collect::<Vec<_>>() {
                                state.key_to_index.insert(key, i);
                            }

                            let mut interspersed = Vec::new();
                            if !values.is_empty() {
                                let mut items_iter = values.into_iter();
                                interspersed.push(items_iter.next().unwrap());
                                for (sep, item) in separators.into_iter().zip(items_iter) {
                                    interspersed.push(sep);
                                    interspersed.push(item);
                                }
                            }
                            out_diffs.push(VecDiff::Replace { values: interspersed });
                        }
                    }
                }
                if out_diffs.is_empty() { None } else { Some(out_diffs) }
            });

            // 5. Register the `processor_logic` to get the final signal system.
            let upstream_handle = self.register(world);
            let processor_handle =
                lazy_signal_from_system::<_, Vec<VecDiff<Self::Item>>, _, _, _>(processor_logic)
                    .register(world);

            // 6. Finalize setup.
            state_entity_handle.set(*processor_handle);
            world
                .entity_mut(*processor_handle)
                // Insert the state component onto the processor's entity.
                .insert(IntersperseState::<Self::Item>::default())
                // Tie the factory's lifecycle to the processor's.
                .add_child(factory_system_id.entity());

            // 7. Connect the upstream signal to our new processor.
            pipe_signal(world, *upstream_handle, processor_handle);

            processor_handle.into()
        });

        IntersperseWith {
            signal,
            _marker: PhantomData,
        }
    }

    /// Creates a new `SignalVec` which keeps its items sorted according to a comparison system.
    ///
    /// The provided `compare_system` takes a tuple of two items `In<(Self::Item, Self::Item)>`
    /// and must return a `std::cmp::Ordering`. The output `SignalVec` will always contain the
    /// same items as the source, but reordered based on the system's logic.
    ///
    /// The sort is stable: if the comparison system returns `Ordering::Equal`, the original
    /// order of the equal items is preserved.
    fn sort_by<F, M>(self, compare_system: F) -> SortBy<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        F: IntoSystem<In<(Self::Item, Self::Item)>, Ordering, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let compare_system_id = world.register_system(compare_system);

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

            fn search<T: Clone + SSs>(
                world: &mut World,
                system_id: SystemId<In<(T, T)>, Ordering>,
                state: &SortState<T>,
                item_idx: usize,
            ) -> Result<usize, usize> {
                state.sorted_indices.binary_search_by(|&probe_idx| {
                    let a = state.values[probe_idx].clone();
                    let b = state.values[item_idx].clone();
                    world
                        .run_system_with(system_id, (a, b))
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| probe_idx.cmp(&item_idx))
                })
            }

            let SignalHandle(signal) = self
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut state: Local<SortState<Self::Item>>|
                          -> Option<Vec<VecDiff<Self::Item>>> {
                        let mut out_diffs = Vec::new();

                        for diff in diffs {
                            match diff {
                                VecDiff::Replace { values: new_values } => {
                                    state.values = new_values;
                                    state.sorted_indices = (0..state.values.len()).collect();

                                    // Fix: Clone the values *before* the sort to avoid borrow conflict.
                                    let values_for_sort = state.values.clone();
                                    state.sorted_indices.sort_unstable_by(|&a, &b| {
                                        let val_a = values_for_sort[a].clone();
                                        let val_b = values_for_sort[b].clone();
                                        world.run_system_with(compare_system_id, (val_a, val_b)).unwrap()
                                    });

                                    let sorted_values =
                                        state.sorted_indices.iter().map(|&i| state.values[i].clone()).collect();
                                    out_diffs.push(VecDiff::Replace { values: sorted_values });
                                }
                                VecDiff::Push { value } => {
                                    let new_idx = state.values.len();
                                    state.values.push(value.clone());
                                    let insert_pos = search(world, compare_system_id, &*state, new_idx).unwrap_err();
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
                                    let insert_pos = search(world, compare_system_id, &*state, index).unwrap_err();
                                    state.sorted_indices.insert(insert_pos, index);
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: insert_pos,
                                        value,
                                    });
                                }
                                VecDiff::RemoveAt { index } => {
                                    // Must search *before* removing from values.
                                    let remove_pos = search(world, compare_system_id, &*state, index).unwrap();
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
                                    let old_pos = search(world, compare_system_id, &*state, index).unwrap();
                                    state.sorted_indices.remove(old_pos);
                                    state.values[index] = value.clone();
                                    let new_pos = search(world, compare_system_id, &*state, index).unwrap_err();
                                    state.sorted_indices.insert(new_pos, index);

                                    if old_pos == new_pos {
                                        out_diffs.push(VecDiff::UpdateAt { index: old_pos, value });
                                    } else {
                                        out_diffs.push(VecDiff::Move {
                                            old_index: old_pos,
                                            new_index: new_pos,
                                        });
                                    }
                                }
                                VecDiff::Move { old_index, new_index } => {
                                    let val = state.values.remove(old_index);
                                    state.values.insert(new_index, val);
                                    for i in state.sorted_indices.iter_mut() {
                                        if *i == old_index {
                                            *i = new_index;
                                        } else if *i > old_index && *i <= new_index {
                                            *i -= 1;
                                        } else if *i < old_index && *i >= new_index {
                                            *i += 1;
                                        }
                                    }
                                }
                                VecDiff::Pop => {
                                    let index = state.values.len() - 1;
                                    let remove_pos = search(world, compare_system_id, &*state, index).unwrap();
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
                )
                .register(world);

            world.entity_mut(*signal).add_child(compare_system_id.entity());
            signal
        });

        SortBy {
            signal,
            _marker: PhantomData,
        }
    }

    // /// Sorts the `SignalVec` with a key extraction system, preserving the initial order of equal
    // elements. ///
    // /// The provided `key_extraction_system` takes an item `In<Self::Item>` and must return a key
    // /// `K` that implements `Ord`. The output `SignalVec` will be sorted based on this key.
    // ///
    // /// This is generally more efficient than `sort_by` if the key extraction is a non-trivial
    // operation, /// as the system is run only once per item change, rather than on every
    // comparison.
    fn sort_by_key<K, F, M>(self, key_extraction_system: F) -> SortByKey<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        K: Ord + Clone + SSs,
        F: IntoSystem<In<Self::Item>, K, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let key_system_id = world.register_system(key_extraction_system);

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

            let SignalHandle(signal) = self
                // Fix 1: Add turbofish to specify the generic type for `for_each`.
                .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                    move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                          world: &mut World,
                          mut state: Local<SortState<Self::Item, K>>|
                          -> Option<Vec<VecDiff<Self::Item>>> {
                        let mut out_diffs = Vec::new();

                        let get_key =
                            |world: &mut World, value: Self::Item| world.run_system_with(key_system_id, value).unwrap();

                        for diff in diffs {
                            match diff {
                                VecDiff::Replace { values: new_values } => {
                                    state.values = new_values;
                                    state.keys = state.values.iter().map(|v| get_key(world, v.clone())).collect();
                                    state.sorted_indices = (0..state.values.len()).collect();

                                    // Fix 2: Clone keys before sorting to avoid mutable/immutable borrow conflict.
                                    let keys_for_sort = state.keys.clone();
                                    state
                                        .sorted_indices
                                        .sort_unstable_by(|&a, &b| keys_for_sort[a].cmp(&keys_for_sort[b]));

                                    let sorted_values =
                                        state.sorted_indices.iter().map(|&i| state.values[i].clone()).collect();
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
                                    let old_pos = search(&*state, index).unwrap();
                                    state.sorted_indices.remove(old_pos);
                                    let new_key = get_key(world, value.clone());
                                    state.values[index] = value.clone();
                                    state.keys[index] = new_key;
                                    let new_pos = search(&*state, index).unwrap_err();
                                    state.sorted_indices.insert(new_pos, index);

                                    if old_pos == new_pos {
                                        out_diffs.push(VecDiff::UpdateAt { index: old_pos, value });
                                    } else {
                                        out_diffs.push(VecDiff::Move {
                                            old_index: old_pos,
                                            new_index: new_pos,
                                        });
                                    }
                                }
                                VecDiff::Move { old_index, new_index } => {
                                    let val = state.values.remove(old_index);
                                    state.values.insert(new_index, val);
                                    let key = state.keys.remove(old_index);
                                    state.keys.insert(new_index, key);

                                    for i in state.sorted_indices.iter_mut() {
                                        if *i == old_index {
                                            *i = new_index;
                                        } else if *i > old_index && *i <= new_index {
                                            *i -= 1;
                                        } else if *i < old_index && *i >= new_index {
                                            *i += 1;
                                        }
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
                )
                .register(world);

            world.entity_mut(*signal).add_child(key_system_id.entity());
            signal
        });

        SortByKey {
            signal,
            _marker: PhantomData,
        }
    }

    // fn flatten(self) -> Flatten<Self>
    // where
    //     Self: Sized,
    //     Self::Item: SignalVec + 'static + Clone,
    //     <Self::Item as SignalVec>::Item: Clone + SSs,
    // {
    //     let signal = LazySignal::new(move |world: &mut World| {
    //         let parent_entity = LazyEntity::new();

    //         let SignalHandle(output_signal) = self
    //             .for_each::<Vec<VecDiff<<Self::Item as SignalVec>::Item>>, _, _,
    // _>(clone!((parent_entity) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
    //                 let parent = parent_entity.get();
    //                 let mut new_diffs = vec![];

    //                 for diff in diffs {
    //                     match diff {
    //                         VecDiff::Push { value: inner_signal } => {
    //                             let index = with_flatten_data::<<Self::Item as SignalVec>::Item,
    // _>(world, parent, |data| data.items.len());                             let (item,
    // initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
    // let offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data|
    // data.items.iter().map(|i| i.values.len()).sum());
    // with_flatten_data(world, parent, |mut data| data.items.push(item));

    //                             for (i, value) in initial_values.into_iter().enumerate() {
    //                                 new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
    //                             }
    //                         }
    //                         VecDiff::InsertAt { index, value: inner_signal } => {
    //                             let (item, initial_values) = spawn_flatten_item(world, parent, index,
    // inner_signal);                             let offset: usize =
    // with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data|
    // data.items[..index].iter().map(|i| i.values.len()).sum());
    // with_flatten_data(world, parent, |mut data| data.items.insert(index, item));
    // let signals_to_update = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent,
    // |data| data.items.iter().map(|i| i.processor_handle.clone()).collect::<Vec<_>>());
    //                             for (i, handle) in signals_to_update.into_iter().enumerate() {
    //                                 if let Some(mut idx) =
    // world.get_mut::<FlattenInnerIndex>(**handle) {                                     idx.0 = i;
    //                                 }
    //                             }
    //                             for (i, value) in initial_values.into_iter().enumerate() {
    //                                 new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
    //                             }
    //                         }
    //                         VecDiff::RemoveAt { index } => {
    //                             let (item, offset) = with_flatten_data(world, parent, |mut data:
    // Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {                                 let
    // offset = data.items[..index].iter().map(|i| i.values.len()).sum();
    // (data.items.remove(index), offset)                             });
    //                             item.processor_handle.cleanup(world);
    //                             for _ in 0..item.values.len() {
    //                                 new_diffs.push(VecDiff::RemoveAt { index: offset });
    //                             }
    //                         }
    //                         VecDiff::Pop => {
    //                             let (item, offset) = with_flatten_data(world, parent, |mut data:
    // Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {                                 let
    // offset = data.items.iter().map(|i|
    // i.values.len()).sum::<usize>().saturating_sub(data.items.last().map_or(0, |i| i.values.len()));
    //                                 (data.items.pop().unwrap(), offset)
    //                             });
    //                             item.processor_handle.cleanup(world);
    //                             for _ in 0..item.values.len() {
    //                                 new_diffs.push(VecDiff::RemoveAt { index: offset });
    //                             }
    //                         }
    //                         VecDiff::Clear => {
    //                             let old_items = with_flatten_data(world, parent, |mut data:
    // Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.items.drain(..).collect::<Vec<_>>());
    //                             for item in old_items {
    //                                 item.processor_handle.cleanup(world);
    //                             }
    //                             new_diffs.push(VecDiff::Clear);
    //                         }
    //                         VecDiff::Replace { values } => {
    //                             let old_items = with_flatten_data(world, parent, |mut data:
    // Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.items.drain(..).collect::<Vec<_>>());
    //                             for item in old_items {
    //                                 item.processor_handle.cleanup(world);
    //                             }
    //                             new_diffs.push(VecDiff::Clear);

    //                             for (i, inner_signal) in values.into_iter().enumerate() {
    //                                 let (item, initial_values) = spawn_flatten_item(world, parent, i,
    // inner_signal);                                 with_flatten_data(world, parent, |mut data|
    // data.items.push(item));                                 for value in initial_values {
    //                                     new_diffs.push(VecDiff::Push { value });
    //                                 }
    //                             }
    //                         }
    //                         VecDiff::UpdateAt { index, value: inner_signal } => {
    //                             let (old_item, offset) = with_flatten_data(world, parent, |mut data:
    // Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {                                 let
    // offset = data.items[..index].iter().map(|i| i.values.len()).sum();
    // (data.items.remove(index), offset)                             });
    //                             old_item.processor_handle.cleanup(world);
    //                             for _ in 0..old_item.values.len() {
    //                                 new_diffs.push(VecDiff::RemoveAt { index: offset });
    //                             }

    //                             let (new_item, initial_values) = spawn_flatten_item(world, parent,
    // index, inner_signal);                             with_flatten_data(world, parent, |mut data|
    // data.items.insert(index, new_item));                             for (i, value) in
    // initial_values.into_iter().enumerate() {
    // new_diffs.push(VecDiff::InsertAt { index: offset + i, value });                             }
    //                         }
    //                         VecDiff::Move { old_index, new_index } => {
    //                             let (old_offset, moved_item, signals_to_update) =
    // with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
    //                                 let old_offset = data.items[..old_index].iter().map(|i|
    // i.values.len()).sum();                                 let moved_item =
    // data.items.remove(old_index);                                 data.items.insert(new_index,
    // moved_item);                                 let signals_to_update =
    // data.items.iter().map(|i| i.processor_handle.clone()).collect::<Vec<_>>();
    // (old_offset, data.items[new_index].clone(), signals_to_update)
    // });

    //                             for (i, handle) in signals_to_update.into_iter().enumerate() {
    //                                 if let Some(mut fi_index) =
    // world.get_mut::<FlattenInnerIndex>(**handle) {                                     fi_index.0
    // = i;                                 }
    //                             }

    //                             for _ in 0..moved_item.values.len() {
    //                                 new_diffs.push(VecDiff::RemoveAt { index: old_offset });
    //                             }

    //                             let new_offset: usize = with_flatten_data::<<Self::Item as
    // SignalVec>::Item, _>(world, parent, |data| data.items[..new_index].iter().map(|i|
    // i.values.len()).sum());

    //                             for (i, value) in moved_item.values.into_iter().enumerate() {
    //                                 new_diffs.push(VecDiff::InsertAt { index: new_offset + i, value
    // });                             }
    //                         }
    //                     }
    //                 }

    //                 let queued_diffs = with_flatten_data(world, parent, |mut data:
    // Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.diffs.drain(..).collect::<Vec<_>>());
    //                 new_diffs.extend(queued_diffs);

    //                 if new_diffs.is_empty() { None } else { Some(new_diffs) }
    //             }))
    //             .register(world);

    //         parent_entity.set(*output_signal);

    //         let SignalHandle(flusher) = SignalBuilder::from_entity(*output_signal)
    //             .map::<Vec<VecDiff<<Self::Item as SignalVec>::Item>>, _, _, _>(
    //                 move |In(_), data: Query<&FlattenData<<Self::Item as SignalVec>::Item>>| {
    //                     if let Ok(data) = data.get(parent_entity.get())
    //                         && !data.diffs.is_empty()
    //                     {
    //                         return Some(vec![]);
    //                     }
    //                     None
    //                 },
    //             )
    //             .register(world);

    //         pipe_signal(world, flusher, output_signal);

    //         world
    //             .entity_mut(*output_signal)
    //             .insert(FlattenData::<<Self::Item as SignalVec>::Item> {
    //                 items: vec![],
    //                 diffs: vec![],
    //             });
    //         output_signal
    //     });

    //     Flatten {
    //         signal,
    //         _marker: PhantomData,
    //     }
    // }

    // /// Creates a "replayable" `SignalVec` that caches its current state.
    // ///
    // /// When a new signal chain subscribes to a memoized `SignalVec`, it will
    // /// immediately receive a `VecDiff::Replace` with the complete, current state of the vector.
    // /// After that, it will receive normal incremental diffs as they are produced by the upstream
    // /// signal.
    // ///
    // /// This is essential for use with dynamic combinators like `switch_signal_vec`,
    // /// as it ensures that switching back to a previously used signal will correctly
    // /// restore the UI to that signal's current state.
    // fn replayable(self) -> Replayable<Self>
    // where
    //     Self: Sized,
    //     Self::Item: Clone + SSs,
    // {
    //     let signal = LazySignal::new(move |world: &mut World| {
    //         let self_entity = LazyEntity::new();

    //         #[derive(Component)]
    //         struct ReplayState<T: SSs> {
    //             values: Vec<T>,
    //             initialized_downstreams: HashSet<SignalSystem>,
    //         }

    //         let replay_system_logic = clone!((self_entity) move |In(incoming_diffs):
    // In<Vec<VecDiff<Self::Item>>>,                 world: &mut World| ->
    // Option<Vec<VecDiff<Self::Item>>> {             let entity = self_entity.get();
    //             // Check for new, uninitialized listeners.
    //             let current_downstreams: HashSet<SignalSystem> = world.get::<Downstream>(entity)
    //                 .map(|d| d.iter().copied().collect())
    //                 .unwrap_or_default();
    //             let mut state = world.get_mut::<ReplayState<Self::Item>>(entity).unwrap();

    //             // Always apply incoming diffs to the internal state first.
    //             for diff in &incoming_diffs {
    //                 diff.apply_to_vec(&mut state.values);
    //             }

    //             state.initialized_downstreams.retain(|s| current_downstreams.contains(s));

    //             let downstreams_to_init: Vec<SignalSystem> = current_downstreams.iter()
    //                 .filter(|&&d| !state.initialized_downstreams.contains(&d))
    //                 .copied()
    //                 .collect();

    //             // If there are new listeners, send a replay and STOP.
    //             if !downstreams_to_init.is_empty() {
    //                 // Mark them as initialized for the next tick.
    //                 for downstream in downstreams_to_init.iter().copied() {
    //                     state.initialized_downstreams.insert(downstream);
    //                 }

    //                 // The replay diff IS the authoritative state for this tick.
    //                 let replay_diff = VecDiff::Replace { values: state.values.clone() };

    //                 // Return ONLY the replay diff. Do not forward the original `incoming_diffs`.
    //                 for &downstream in downstreams_to_init.iter() {
    //                     world.entity_mut(*downstream).insert(SkipOnce);
    //                 }
    //                 process_signals(world, downstreams_to_init, Box::new(vec![replay_diff]));
    //             }

    //             // If no replay was sent, forward the incremental diffs as normal.
    //             if incoming_diffs.is_empty() {
    //                 None
    //             } else {
    //                 Some(incoming_diffs)
    //             }
    //         });

    //         let memoize_system =
    //             lazy_signal_from_system::<_, Vec<VecDiff<Self::Item>>, _, _,
    // _>(replay_system_logic).register(world);         self_entity.set(*memoize_system);
    //         world.entity_mut(*memoize_system).insert(ReplayState::<Self::Item> {
    //             values: Vec::new(),
    //             initialized_downstreams: HashSet::new(),
    //         });
    //         let upstream_handle = self.register(world);
    //         pipe_signal(world, *upstream_handle, memoize_system);
    //         memoize_system
    //     });

    //     Replayable {
    //         signal,
    //         _marker: PhantomData,
    //     }
    // }

    fn debug(self) -> Debug<Self>
    where
        Self: Sized,
        Self::Item: fmt::Debug + Clone + 'static,
    {
        let location = core::panic::Location::caller();
        Debug {
            signal: self.for_each(move |In(item)| {
                debug!("[{}] {:#?}", location, item);
                item
            }),
        }
    }

    fn boxed(self) -> Box<dyn SignalVec<Item = Self::Item>>
    where
        Self: Sized,
    {
        Box::new(self)
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

impl<T: ?Sized> SignalVecExt for T where T: SignalVec {}

//-------------------------------------------------------------------------------------------------
// Lock Guards
//-------------------------------------------------------------------------------------------------

/// A read guard for `MutableVec`, providing immutable access to the underlying `Vec`.
/// The guard holds the read lock, ensuring safe access.
pub struct MutableVecReadGuard<'a, T>
where
    T: Clone,
{
    guard: RwLockReadGuard<'a, MutableVecState<T>>,
}

impl<'a, T> Deref for MutableVecReadGuard<'a, T>
where
    T: Clone,
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
    T: Clone,
{
    guard: RwLockWriteGuard<'a, MutableVecState<T>>,
}

impl<'a, T> MutableVecWriteGuard<'a, T>
where
    T: Clone,
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

    /// Inserts an element at `index` within the vector, shifting all elements after it to the
    /// right. Queues a `VecDiff::InsertAt`.
    /// # Panics
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, value: T) {
        self.guard.vec.insert(index, value.clone());
        self.guard.pending_diffs.push(VecDiff::InsertAt { index, value });
    }

    /// Removes and returns the element at `index` within the vector, shifting all elements after it
    /// to the left. Queues a `VecDiff::RemoveAt`.
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
            self.guard.pending_diffs.push(VecDiff::UpdateAt { index, value });
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
            self.guard.pending_diffs.push(VecDiff::Move { old_index, new_index });
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
    T: Clone,
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
struct MutableVecState<T: Clone> {
    vec: Vec<T>,
    pending_diffs: Vec<VecDiff<T>>,
    signal: Option<LazySignal>,
}

/// A mutable vector that tracks changes as `VecDiff`s and sends them as a batch on `flush`.
/// This struct is `Clone`able, sharing the underlying state.
#[derive(Clone)]
pub struct MutableVec<T>
where
    T: Clone,
{
    state: Arc<RwLock<MutableVecState<T>>>,
}

#[derive(Component)]
struct QueuedVecDiffs<T>(Vec<VecDiff<T>>);

impl<T, A: Clone> From<T> for MutableVec<A>
where
    Vec<A>: From<T>,
{
    #[inline]
    fn from(values: T) -> Self {
        MutableVec {
            state: Arc::new(RwLock::new(MutableVecState {
                vec: values.into(),
                pending_diffs: Vec::new(),
                signal: None,
            })),
        }
    }
}

impl<T: Clone> MutableVec<T> {
    // ... new(), with_values(), read(), write() are the same ...
    // Make sure the state initializes with `lazy_signal: None`.
    // e.g., in `new()`:
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableVecState {
                vec: Vec::new(),
                pending_diffs: Vec::new(),
                signal: None,
            })),
        }
    }

    /// Creates a new `MutableVec` initialized with the given values.
    pub fn with_values(values: Vec<T>) -> Self {
        Self {
            state: Arc::new(RwLock::new(MutableVecState {
                pending_diffs: Vec::new(), // Start with empty diffs
                vec: values,
                signal: None,
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

    /// Returns the canonical `SignalVec` source for this `MutableVec`.
    ///
    /// The first time this is called, it creates the signal source. On all subsequent calls,
    /// it returns a clone of the same source. This ensures that a given `MutableVec` instance
    /// always corresponds to exactly one `Source` system in the `World`, which is the correct
    /// pattern for `jonmo`.
    pub fn signal_vec(&self) -> Source<T>
    where
        T: SSs,
    {
        // The idempotent caching logic is still correct.
        let mut state = self.state.write().unwrap();
        if let Some(lazy_signal) = &state.signal {
            return Source {
                signal: lazy_signal.clone(),
                _marker: PhantomData,
            };
        }

        let signal = LazySignal::new(clone!((self.state => state) move |world: &mut World| {
            let self_entity = LazyEntity::new();

            // This is a simple, "dumb" source. It just drains its queue.
            let source_system_logic = clone!((self_entity) move |_: In<()>, world: &mut World| {
                let mut diffs = world.get_mut::<QueuedVecDiffs<T>>(self_entity.get()).unwrap();
                if diffs.0.is_empty() { None } else { Some(diffs.0.drain(..).collect()) }
            });

            let signal_system = register_signal::<(), Vec<VecDiff<T>>, _, _, _>(world, source_system_logic);
            self_entity.set(*signal_system);

            // Queue the initial state if it exists. This will be emitted once.
            let initial_vec = state.read().unwrap().vec.clone();
            let initial_diffs = if !initial_vec.is_empty() {
                vec![VecDiff::Replace { values: initial_vec }]
            } else {
                vec![]
            };
            world.entity_mut(*signal_system).insert(QueuedVecDiffs(initial_diffs));

            signal_system
        }));

        state.signal = Some(signal.clone());
        Source {
            signal,
            _marker: PhantomData,
        }
    }

    /// Sends any pending `VecDiff`s to the signal system.
    pub fn flush_into_world(&self, world: &mut World)
    where
        T: SSs,
    {
        let mut state = self.state.write().unwrap();
        if state.pending_diffs.is_empty() {
            return;
        }

        // If the signal has been created and registered, send the diffs.
        if let Some(lazy_signal) = &state.signal
            && let LazySystem::Registered(signal_system) = *lazy_signal.inner.system.read().unwrap()
            && let Ok(mut entity) = world.get_entity_mut(*signal_system)
            && let Some(mut queued_diffs) = entity.get_mut::<QueuedVecDiffs<T>>()
        {
            queued_diffs.0.extend(state.pending_diffs.clone());
        }
        state.pending_diffs.clear();
    }

    pub fn flush(&self) -> impl Command
    where
        T: SSs,
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
    use bevy_log::tracing_subscriber::EnvFilter;
    use bevy_platform::sync::OnceLock; // Add `once_cell` to your Cargo.toml if you don't have it

    // Use a static OnceCell to ensure the subscriber is only initialized once
    static TRACING_INIT: OnceLock<()> = OnceLock::new();

    // A helper function to set up tracing
    fn setup_tracing() {
        TRACING_INIT.get_or_init(|| {
            bevy_log::tracing_subscriber::FmtSubscriber::builder()
                // Respect RUST_LOG environment variable for filtering
                // e.g., RUST_LOG=info,wgpu=error cargo test ...
                .with_env_filter(EnvFilter::from_default_env())
                .init();
            // You can also add other configurations here, like color, time, etc.
        });
    }

    // // Helper component and resource for testing (similar to signal.rs tests)
    // #[derive(Component, Clone, Debug, PartialEq, Default)]
    // struct TestItem(i32);

    #[derive(Resource, Default)]
    struct SignalVecOutput<T: Clone + fmt::Debug>(Vec<VecDiff<T>>);

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JonmoPlugin));
        app
    }

    // Helper system to capture signal_vec output
    fn capture_signal_vec_output<T>(In(diffs): In<Vec<VecDiff<T>>>, mut output: ResMut<SignalVecOutput<T>>)
    where
        T: SSs + Clone + fmt::Debug,
    {
        debug!(
            "Capture SignalVec Output: Received {:?}, extending resource from {:?} with new diffs",
            diffs, output.0
        );
        output.0.extend(diffs);
    }

    fn get_signal_vec_output<T: SSs + Clone + fmt::Debug>(world: &World) -> Vec<VecDiff<T>> {
        world.resource::<SignalVecOutput<T>>().0.clone()
    }

    // fn clear_signal_vec_output<T: SSs + Clone + fmt::Debug>(world: &mut World) {
    //     if let Some(mut output) = world.get_resource_mut::<SignalVecOutput<T>>() {
    //         output.0.clear();
    //     }
    // }

    // Helper to make assertions cleaner
    fn get_and_clear_output<T: SSs + Clone + fmt::Debug>(world: &mut World) -> Vec<VecDiff<T>> {
        let output = world.resource::<SignalVecOutput<T>>().0.clone();
        world.resource_mut::<SignalVecOutput<T>>().0.clear();
        output
    }

    // Add this PartialEq implementation for easier testing
    impl<T: SSs + PartialEq + fmt::Debug> PartialEq for VecDiff<T> {
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

    // #[test]
    // fn test_switch_signal_vec() {
    //     setup_tracing();

    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<i32>>();

    //     // Resource to control which SignalVec is active
    //     #[derive(Resource, Clone, Copy, PartialEq)]
    //     struct Mode(bool); // false = list A, true = list B
    //     app.insert_resource(Mode(false));

    //     // Two distinct MutableVecs to switch between
    //     let list_a = MutableVec::from([1, 2]);
    //     let list_b = MutableVec::from([100, 200, 300]);

    //     // Create the idempotent signal sources once.
    //     let signal_vec_a = list_a.signal_vec().replayable();
    //     let signal_vec_b = list_b.signal_vec().replayable();

    //     // The outer signal that reads the Mode resource
    //     let mode_signal = SignalBuilder::from_system(|_: In<()>, mode: Res<Mode>| *mode).dedupe();

    //     let switched_signal =
    //         mode_signal.switch_signal_vec(clone!((signal_vec_a, signal_vec_b) move |In(mode):
    // In<Mode>| {             if !mode.0 {
    //                 signal_vec_a.clone()
    //             } else {
    //                 signal_vec_b.clone()
    //             }
    //         }));

    //     let handle = switched_signal
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     // --- Step 1: Initial state (should be list_a) ---
    //     app.update();
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     // assert_eq!(diffs, vec![], "Initial state should produce no diffs");
    //     info!("Initial state: {:?}", diffs);
    //     assert_eq!(diffs.len(), 1, "Initial diff count was not 1");
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Replace { values: vec![1, 2] },
    //         "Initial state was not list_a"
    //     );

    //     // --- Step 2: Update list_a, verify it propagates ---
    //     list_a.write().push(3);
    //     list_a.flush_into_world(app.world_mut());
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     assert_eq!(diffs.len(), 1, "Diff count from list_a was not 1");
    //     assert_eq!(diffs[0], VecDiff::Push { value: 3 }, "Did not forward Push from list_a");
    //     // The state of list_a is now [1, 2, 3]

    //     // --- Step 3: Switch to list_b ---
    //     app.world_mut().resource_mut::<Mode>().0 = true;
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     assert_eq!(diffs.len(), 1, "Switching diff count was not 1");
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Replace {
    //             values: vec![100, 200, 300]
    //         },
    //         "Did not Replace with list_b's state on switch"
    //     );

    //     // --- Step 4: Update list_b to verify it's active ---
    //     list_b.write().pop();
    //     list_b.flush_into_world(app.world_mut());
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     assert_eq!(diffs.len(), 1, "Diff count from list_b was not 1");
    //     assert_eq!(diffs[0], VecDiff::Pop, "Did not forward Pop from list_b");
    //     // The state of list_b is now [100, 200]

    //     // --- Step 5: IGNORE diffs from the old list_a ---
    //     list_a.write().push(99); // This should be ignored
    //     list_a.flush_into_world(app.world_mut());
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     assert!(
    //         diffs.is_empty(),
    //         "Received a diff from the inactive list_a after switching"
    //     );
    //     // The state of list_a is now [1, 2, 3, 99]

    //     // --- Step 6: CRITICAL TEST - Switch back to list_a ---
    //     app.world_mut().resource_mut::<Mode>().0 = false;
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     // The "smart source" for list_a should see a new listener and send a
    //     // replay of its *current* state, which is [1, 2, 3, 99].
    //     assert_eq!(diffs.len(), 1, "Switching back to A should produce one diff");
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Replace {
    //             values: vec![1, 2, 3, 99]
    //         },
    //         "Did not correctly replay list A's updated state on re-subscribe"
    //     );

    //     handle.cleanup(app.world_mut());
    // }

    // #[test]
    // fn test_memoize_with_switch_signal_vec() {
    //     let mut app = create_test_app();
    //     app.init_resource::<SignalVecOutput<i32>>();

    //     // Resource to control the switch
    //     #[derive(Resource, Default)]
    //     struct Switcher(bool);
    //     app.init_resource::<Switcher>();

    //     // 1. Create two persistent MutableVecs. We don't use the `Resource` pattern here to prove
    // that     //    `.memoize()` works correctly with local, closure-captured state.
    //     let list_a = MutableVec::from([10, 20]);
    //     let list_b = MutableVec::from([99]);

    //     // 2. Create the memoized signal sources OUTSIDE the switcher. This is the pattern we want to
    // test.     let memoized_a = list_a.signal_vec();
    //     let memoized_b = list_b.signal_vec();

    //     let switcher_signal = SignalBuilder::from_system(|_: In<()>, s: Res<Switcher>| s.0).dedupe();

    //     let final_signal =
    //         switcher_signal.switch_signal_vec(
    //             move |In(use_b): In<bool>| {
    //                 if use_b { memoized_b.clone() } else { memoized_a.clone() }
    //             },
    //         );

    //     let handle = final_signal
    //         .for_each(capture_signal_vec_output)
    //         .register(app.world_mut());

    //     // --- Step 1: Initial state ---
    //     // The switcher starts as `false`, so it should subscribe to `memoized_a`.
    //     // The memoize combinator should see this new subscription and replay its state.
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     // The first diff from memoize is the full replay.
    //     // The second diff is the initial `Replace` from the MutableVec source itself.
    //     // The switch_signal_vec logic ensures only one `Replace` is ultimately sent.
    //     assert_eq!(diffs.len(), 1, "Initial setup should produce one diff");
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Replace { values: vec![10, 20] },
    //         "Initial state should be list A"
    //     );

    //     // --- Step 2: Update list_a, verify it propagates ---
    //     list_a.write().push(30);
    //     list_a.flush_into_world(app.world_mut());
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     assert_eq!(diffs.len(), 1);
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Push { value: 30 },
    //         "Push on active list A did not propagate"
    //     );
    //     // list_a's memoized state is now [10, 20, 30]

    //     // --- Step 3: Switch to list_b ---
    //     app.world_mut().resource_mut::<Switcher>().0 = true;
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     // switch_signal_vec sends a clear, then memoize sends its replay.
    //     // The UI should see a single atomic Replace.
    //     assert_eq!(diffs.len(), 1, "Switching to B should produce one diff");
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Replace { values: vec![99] },
    //         "Switch did not replay list B's state"
    //     );

    //     // --- Step 4: Update list_b to verify it's active ---
    //     list_b.write().push(100);
    //     list_b.flush_into_world(app.world_mut());
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     assert_eq!(diffs.len(), 1);
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Push { value: 100 },
    //         "Push on active list B did not propagate"
    //     );

    //     // --- Step 5: CRITICAL TEST - Switch back to list_a ---
    //     app.world_mut().resource_mut::<Switcher>().0 = false;
    //     app.update();
    //     let diffs = get_and_clear_output::<i32>(app.world_mut());

    //     // `memoize` should see a new subscriber and REPLAY its cached state.
    //     // The cached state for list_a is [10, 20, 30].
    //     assert_eq!(diffs.len(), 1, "Switching back to A should produce one diff");
    //     assert_eq!(
    //         diffs[0],
    //         VecDiff::Replace {
    //             values: vec![10, 20, 30]
    //         },
    //         "Did not correctly replay list A's updated state"
    //     );

    //     handle.cleanup(app.world_mut());
    // }

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

        mutable_vec.write().push(1u32);
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
    //     assert_eq!(output, vec![VecDiff::Replace { values: vec![TestItem(10), TestItem(20),
    // TestItem(30)] }]);     assert_eq!(mutable_vec.read().as_ref(), &[TestItem(10),
    // TestItem(20), TestItem(30)]);     handle.cleanup(app.world_mut());
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
    //     // The current `map` implementation for `SignalVecExt` doesn't filter if the mapping
    // system returns `None`.     // It maps values, and if a mapping returns `None` (which the
    // provided system signature doesn't allow directly,     // as it expects `O` not
    // `Option<O>`), it would skip that item in `Replace`, `InsertAt`, `UpdateAt`, `Push`.     //
    // For `RemoveAt`, `Move`, `Pop`, `Clear`, it passes them through.

    //     // To test filtering, the mapping system itself would need to return Option<O>.
    //     // The current `SignalVecExt::map` signature is:
    //     // F: IntoSystem<In<Self::Item>, O, M>
    //     // This means the system `F` must produce `O`, not `Option<O>`.
    //     // The filtering logic is: `world.run_system_with(system, v).ok()`
    //     // If `run_system_with` returns `Err` (e.g. system panics or has unmet dependencies),
    // it's filtered.     // It does NOT filter based on `Option<O>` from the user's system.

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

    //     // Given the current `MutableVec` structure, `state.signals` are just `SignalSystem`
    // (Entity).     // They are not automatically cleaned up when `MutableVec` is dropped.
    //     // The `SignalHandle::cleanup` is the primary mechanism.
    //     // If `source_signal_vec_struct` is dropped, its `LazySignal`'s strong_count decreases.
    //     // The `LazySignal` inside `LazySignalHolder` on the system entity is another.
    //     // If these are the only two, dropping `source_signal_vec_struct` makes the holder's copy
    // the last one.     // When `LazySignalHolder` is eventually dropped (e.g., because
    // `SignalRegistrationCount` is 0),     // its `LazySignal::drop` will queue the system for
    // cleanup.     app.update(); // Process potential cleanup queue
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
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Push { value:
    // "Item: 10".to_string() }]);     clear_signal_vec_output::<String>(app.world_mut());

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
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::InsertAt { index:
    // 0, value: "Item: 20".to_string() }]);     clear_signal_vec_output::<String>(app.
    // world_mut());

    //     // UpdateAt
    //     mutable_vec_int.set(0, 30);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::UpdateAt { index:
    // 0, value: "Item: 30".to_string() }]);     clear_signal_vec_output::<String>(app.
    // world_mut());

    //     // RemoveAt
    //     mutable_vec_int.remove(0);
    //     mutable_vec_int.flush().apply(app.world_mut());
    //     app.update();
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::RemoveAt { index:
    // 0 }]);     clear_signal_vec_output::<String>(app.world_mut());

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
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Move { old_index:
    // 0, new_index: 2 }]);     clear_signal_vec_output::<String>(app.world_mut());

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
    //     assert_eq!(get_signal_vec_output::<String>(app.world()), vec![VecDiff::Replace { values:
    // vec!["Item: 100".to_string(), "Item: 200".to_string()] }]);

    //     handle.cleanup(app.world_mut());
    // }
}
