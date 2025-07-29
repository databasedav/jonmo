//! Data structures and combinators for constructing reactive [`System`] dependency graphs on top of
//! [`Vec`] mutations, see [`MutableVec`] and [`SignalVecExt`].
use super::{
    graph::{
        LazySignal, LazySystem, SignalHandle, SignalSystem, Upstream, downcast_any_clone, lazy_signal_from_system,
        pipe_signal, poll_signal, process_signals, register_signal,
    },
    signal::{Signal, SignalBuilder, SignalExt},
    utils::{LazyEntity, SSs},
};
use crate::prelude::clone;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{change_detection::Mut, prelude::*, system::SystemId};
#[cfg(feature = "tracing")]
use bevy_log::debug;
use bevy_platform::{
    collections::HashMap,
    prelude::*,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use core::{cmp::Ordering, fmt, marker::PhantomData, ops::Deref};
use dyn_clone::{DynClone, clone_trait_object};

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

/// Monadic registration facade for structs that encapsulate some [`System`] which is a valid member
/// of the signal graph downstream of some source [`MutableVec`]; this is similar to [`Signal`] but
/// critically requires that the [`System`] outputs [`Option<VecDiff<Self::Item>>`].
pub trait SignalVec: SSs {
    /// Output type.
    type Item;

    /// Registers the [`System`]s associated with this [`SignalVec`] by consuming its boxed form.
    ///
    /// All concrete signal vec types must implement this method.
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
        (*self).register_boxed_signal_vec(world)
    }
}

/// An extension trait for [`SignalVec`] types that implement [`Clone`].
///
/// Relevant in contexts where some function may require a [`Clone`] [`SignalVec`], but the concrete
/// type can't be known at compile-time, e.g. in a
/// [`.switch_signal_vec`](SignalExt::switch_signal_vec).
pub trait SignalVecClone: SignalVec + DynClone {}

clone_trait_object!(< T > SignalVecClone < Item = T >);

impl<T: SignalVec + Clone + 'static> SignalVecClone for T {}

impl<O: 'static> SignalVec for Box<dyn SignalVecClone<Item = O> + Send + Sync> {
    type Item = O;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        (*self).register_boxed_signal_vec(world)
    }
}

/// Signal graph node with no upstreams which forwards [`Vec<VecDiff<T>>`]s flushed from some source
/// [`MutableVec<T>`], see [`MutableVec::signal_vec`].
#[derive(Clone)]
pub struct Source<T> {
    signal: LazySignal,
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

/// Signal graph node which applies a [`System`] directly to the "raw" [`Vec<VecDiff>`]s of its
/// upstream, see [`.for_each`](SignalVecExt::for_each).
#[derive(Clone)]
pub struct ForEach<Upstream, O> {
    upstream: Upstream,
    signal: LazySignal,
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

/// Signal graph node which applies a [`System`] to each [`Item`](SignalVec::Item) of its upstream,
/// see [`.map`](SignalVecExt::map).
#[derive(Clone)]
pub struct Map<Upstream, O> {
    signal: LazySignal,
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

#[derive(Component, Deref, DerefMut, Clone, Copy)]
struct ItemIndex(usize);

/// Signal graph node which applies a [`System`] to each [`Item`](SignalVec::Item) of its upstream,
/// forwarding the output of each resulting [`Signal`], see
/// [`.map_signal`](SignalVecExt::map_signal).
#[derive(Clone)]
pub struct MapSignal<Upstream, S: Signal> {
    signal: LazySignal,
    _marker: PhantomData<fn() -> (Upstream, S)>,
}

impl<Upstream, S: Signal> SignalVec for MapSignal<Upstream, S>
where
    Upstream: SignalVec,
    S: Signal + 'static,
    S::Item: Clone + SSs,
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
                // Determine if the item being moved is part of the filtered set.
                let was_included = indices[old_index];

                // Calculate the filtered old and new indices BEFORE and AFTER mutating `indices`.
                let filtered_old_index = find_index(indices.iter(), old_index);
                let temp_val = indices.remove(old_index);
                indices.insert(new_index, temp_val);
                let filtered_new_index = find_index(indices.iter(), new_index);

                if was_included {
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

/// Signal graph node which selectively forwards upstream [`Item`](SignalVec::Item)s, see
/// [`.filter`](SignalVecExt::filter).
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

/// Signal graph node which transforms and selectively forwards upstream [`Item`](SignalVec::Item)s,
/// see [`.filter_map`](SignalVecExt::filter_map).
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

fn spawn_filter_signal<T: Clone + SSs>(
    world: &mut World,
    index: usize,
    signal: impl Signal<Item = bool> + 'static,
    parent: Entity,
) -> (SignalHandle, bool) {
    let entity = LazyEntity::new();
    let processor_system = clone!((entity) move |In(filter): In<bool>, world: &mut World| {
        let self_entity = entity.get();

        // The processor might run after its parent has been cleaned up.
        let Ok(signal_index_comp) = world.query::<&FilterSignalIndex>().get(world, self_entity) else {
            return;
        };
        let item_index = signal_index_comp.0;
        let Ok(mut filter_signal_data) = world.query::<&mut FilterSignalData<T>>().get_mut(world, parent) else {
            return;
        };

        // This item_index might be stale if other items were removed in the same frame.
        // Ensure we don't panic.
        if item_index >= filter_signal_data.items.len() {
            return;
        }
        let old_filtered_state = filter_signal_data.items[item_index].filtered;
        if old_filtered_state == filter {
            // No change, do nothing.
            return;
        }
        let diff_to_queue = if filter {
            // Item is now INCLUDED. Find its insertion point based on the state _before_ it's
            // marked as included.
            let new_filtered_index = find_filter_signal_index(&filter_signal_data.items, item_index);
            filter_signal_data.items[item_index].filtered = true;
            VecDiff::InsertAt {
                index: new_filtered_index,
                value: filter_signal_data.items[item_index].value.clone(),
            }
        } else {
            // Item is now EXCLUDED. Find its removal point based on the state _before_ it's
            // marked as excluded.
            let old_filtered_index = find_filter_signal_index(&filter_signal_data.items, item_index);
            filter_signal_data.items[item_index].filtered = false;
            VecDiff::RemoveAt { index: old_filtered_index }
        };
        filter_signal_data.diffs.push(diff_to_queue);

        // Poke the main output signal to process its queue immediately.
        process_signals(world, [parent.into()], Box::new(Vec::<VecDiff<T>>::new()));
    });

    // Use .map() to attach the processor. The dedupe is still important.
    let mapped_signal = signal.dedupe().map(processor_system);
    let handle = mapped_signal.register(world);
    entity.set(**handle);
    world.entity_mut(**handle).insert(FilterSignalIndex(index));

    // To get the initial value, we must poll the original signal before the
    // map/dedupe. The upstream of the `map` node is the `dedupe` node.
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
        .and_then(downcast_any_clone::<bool>)
        .expect("filter_signal's inner signal must emit an initial value upon registration");
    (handle, initial_value)
}

/// Signal graph node which selectively forwards upstream [`Item`](SignalVec::Item)s depending on a
/// [`Signal<Item = bool>`], see [`.filter_signal`](SignalVecExt::filter_signal).
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

#[derive(Component, Default)]
struct EnumerateState {
    key_to_index: HashMap<usize, usize>,
    ordered_keys: Vec<usize>,
    next_key: usize,
}

/// Signal graph node which prepends an index signal to each upstream [`Item`](SignalVec::Item), see
/// [`.enumerate`](SignalVecExt::enumerate).
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
    type Item = (super::signal::Source<Option<usize>>, Upstream::Item);

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        self.signal.register(world).into()
    }
}

/// Signal graph node which collects upstream [`Item`](SignalVec::Item)s into a single [`Vec`], see
/// [`.to_signal`](SignalVecExt::to_signal).
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

/// Signal graph node which outputs whether its upstream is populated, see
/// [`.is_empty`](SignalVecExt::is_empty).
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

/// Signal graph node which outputs its upstream's length, see [`.len`](SignalVecExt::len).
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

/// Signal graph node which outputs its upstream's sum, see [`.sum`](SignalVecExt::sum).
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

/// Signal graph node which concatenates its upstreams, see [`.chain`](SignalVecExt::chain).
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

/// Signal graph node that places a separator between each upstream [`Item`](SignalVec::Item), see
/// [`.intersperse`](SignalVecExt::intersperse).
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
    /// Stable keys for the separators. The key at index `i` corresponds to the separator that comes
    /// _after_ `local_values[i]`.
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

/// Signal graph node that places a [`System`]-dependent separator between each upstream
/// [`Item`](SignalVec::Item), see [`.intersperse_with`](SignalVecExt::intersperse_with).
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

/// Signal graph node that sorts each upstream [`Item`](SignalVec::Item) based on a comparison
/// [`System`], see [`.sort_by`](SignalVecExt::sort_by).
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

/// Signal graph node that sorts each upstream [`Item`](SignalVec::Item) based on a key extraction
/// [`System`], see [`.sort_by_key`](SignalVecExt::sort_by_key).
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

/* #[derive(Clone)]
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
        if let Ok(mut parent_data) = query_parent_data.get_mut(parent) {
            if let Ok(&FlattenInnerIndex(self_index)) = query_self_index.get(entity.get()) {
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
                                VecDiff::RemoveAt {
                                    index: index + offset,
                                }
                            }
                            VecDiff::Move {
                                old_index,
                                new_index,
                            } => {
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
}

fn spawn_flatten_item<O: Clone + SSs>(
    world: &mut World,
    parent: Entity,
    index: usize,
    inner_signal: impl SignalVec<Item = O> + 'static + Clone,
) -> (FlattenItem<O>, Vec<O>) {
    // These are placeholder function calls based on the names
    let temp_get_state_signal = inner_signal.clone(); // .to_signal().first();
    let handle = temp_get_state_signal.register(world);
    let initial_values = poll_signal(world, &handle)
        .and_then(downcast_any_clone::<Vec<O>>)
        .unwrap_or_default();
    handle.cleanup(world);

    let entity = LazyEntity::new();
    let processor_system = create_flatten_processor(entity.clone(), parent);
    let processor_handle = inner_signal.for_each(processor_system).register(world);

    entity.set(
        world
            .entity_mut(*processor_handle)
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
} */

cfg_if::cfg_if! {
    if #[cfg(feature = "tracing")] {
        /// Signal graph node that debug logs its upstream's "raw" [`Vec<VecDiff>`]s, see
        /// [`.debug`](SignalVecExt::debug).
        #[derive(Clone)]
        pub struct Debug<Upstream>
        where
            Upstream: SignalVec,
        {
            signal: ForEach<Upstream, Vec<VecDiff<Upstream::Item>>>,
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
    }
}

/// Enables returning different concrete [`Signal`] types from branching logic without boxing,
/// although note that all [`Signal`]s are boxed internally regardless.
///
/// Inspired by <https://github.com/rayon-rs/either>.
#[derive(Clone)]
#[allow(missing_docs)]
pub enum SignalVecEither<L, R>
where
    L: SignalVec,
    R: SignalVec,
{
    Left(L),
    Right(R),
}

impl<T, L: SignalVec<Item = T>, R: SignalVec<Item = T>> SignalVec for SignalVecEither<L, R>
where
    L: SignalVec<Item = T>,
    R: SignalVec<Item = T>,
{
    type Item = T;

    fn register_boxed_signal_vec(self: Box<Self>, world: &mut World) -> SignalHandle {
        match *self {
            SignalVecEither::Left(left) => left.register_signal_vec(world),
            SignalVecEither::Right(right) => right.register_signal_vec(world),
        }
    }
}

/// Blanket trait for transforming [`Signal`]s into [`SignalVecEither::Left`] or
/// [`SignalVecEither::Right`].
#[allow(missing_docs)]
pub trait IntoSignalVecEither: Sized
where
    Self: SignalVec,
{
    fn left_either<R>(self) -> SignalVecEither<Self, R>
    where
        R: SignalVec,
    {
        SignalVecEither::Left(self)
    }

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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
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
    /// Convenient when additional [`SystemParam`](bevy_ecs::system::SystemParam)s aren't necessary.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
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

    /// Pass each [`Item`](SignalVec::Item) of this [`SignalVec`] to a [`System`] that produces a
    /// [`Signal`], forwarding the output of each resulting [`Signal`].
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// MutableVec::from([1, 2, 3]).signal_vec()
    ///     .map_signal(|In(x): In<i32>|
    ///         SignalBuilder::from_system(move |_: In<()>| x * 2).dedupe()
    ///     ); // outputs `SignalVec -> [2, 4, 6]`
    /// ```
    fn map_signal<S, F, M>(self, system: F) -> MapSignal<Self, S>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        S: Signal + 'static + Clone,
        S::Item: Clone + Send + Sync,
        F: IntoSystem<In<Self::Item>, S, M> + SSs,
    {
        fn spawn_processor<Item: Clone + SSs, S: Signal<Item = Item> + Clone + 'static>(
            world: &mut World,
            queue_entity: Entity,
            output_system: SignalSystem,
            index: usize,
            inner_signal: S,
        ) -> (SignalHandle, SignalSystem, Item) {
            let inner_signal_id = inner_signal.clone().register(world);
            let temp_handle = inner_signal.clone().first().register(world);
            let initial_value = poll_signal(world, *temp_handle)
                .and_then(downcast_any_clone::<Item>)
                .expect("map_signal's inner signal must emit an initial value");
            temp_handle.cleanup(world);
            let processor_entity = LazyEntity::new();
            let processor_logic = clone!((processor_entity) move |In(value): In<Item>, world: &mut World| {
                if let Some(item_index_comp) = world.get::<ItemIndex>(processor_entity.get()) {
                    let current_index = item_index_comp.0;
                    if let Some(mut queue) = world.get_mut::<QueuedVecDiffs<Item>>(queue_entity) {
                        queue.0.push(VecDiff::UpdateAt {
                            index: current_index,
                            value,
                        });
                        process_signals(world, [output_system], Box::new(()));
                    }
                }
            });
            let processor_handle = inner_signal.map(processor_logic).register(world);
            processor_entity.set(**processor_handle);
            world.entity_mut(**processor_handle).insert(ItemIndex(index));
            (processor_handle, *inner_signal_id, initial_value)
        }

        let signal = LazySignal::new(move |world: &mut World| {
            let factory_system_id = world.register_system(system);
            let state_and_queue_entity = world.spawn_empty().id();
            let output_system_handle =
                SignalBuilder::from_system::<Vec<VecDiff<S::Item>>, _, _, _>(move |_: In<()>, world: &mut World| {
                    if let Some(mut diffs) = world.get_mut::<QueuedVecDiffs<S::Item>>(state_and_queue_entity) {
                        if diffs.0.is_empty() {
                            None
                        } else {
                            Some(diffs.0.drain(..).collect())
                        }
                    } else {
                        None
                    }
                })
                .register(world);

            #[derive(Component)]
            struct ManagerState<S: Signal> {
                processors: Vec<(SignalHandle, SignalSystem)>,
                _phantom: PhantomData<S>,
            }

            let output_system_handle_clone = output_system_handle.clone();
            let manager_system_logic = move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                let mut new_diffs = Vec::new();
                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values } => {
                            let old_processors = {
                                let mut manager_state =
                                    world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                manager_state.processors.drain(..).collect::<Vec<_>>()
                            };
                            for (handle, _) in old_processors {
                                handle.cleanup(world);
                            }
                            let mut new_processors = Vec::with_capacity(values.len());
                            let mut new_values = Vec::with_capacity(values.len());
                            for (i, value) in values.into_iter().enumerate() {
                                if let Ok(signal) = world.run_system_with(factory_system_id, value) {
                                    let (handle, inner_id, initial_value) = spawn_processor(
                                        world,
                                        state_and_queue_entity,
                                        *output_system_handle_clone,
                                        i,
                                        signal,
                                    );
                                    new_processors.push((handle, inner_id));
                                    new_values.push(initial_value);
                                }
                            }
                            {
                                let mut manager_state =
                                    world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                manager_state.processors = new_processors;
                            }
                            new_diffs.push(VecDiff::Replace { values: new_values });
                        }
                        VecDiff::InsertAt { index, value } => {
                            if let Ok(signal) = world.run_system_with(factory_system_id, value) {
                                let (handle, inner_id, initial_value) = spawn_processor(
                                    world,
                                    state_and_queue_entity,
                                    *output_system_handle_clone,
                                    index,
                                    signal,
                                );
                                let handles_to_update = {
                                    let mut manager_state =
                                        world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                    manager_state.processors.insert(index, (handle, inner_id));
                                    manager_state
                                        .processors
                                        .iter()
                                        .enumerate()
                                        .skip(index + 1)
                                        .map(|(i, (proc_handle, _))| (**proc_handle, i))
                                        .collect::<Vec<_>>()
                                };
                                for (proc_handle_entity, new_index) in handles_to_update {
                                    world.get_mut::<ItemIndex>(*proc_handle_entity).unwrap().0 = new_index;
                                }
                                new_diffs.push(VecDiff::InsertAt {
                                    index,
                                    value: initial_value,
                                });
                            }
                        }
                        VecDiff::UpdateAt { index, value } => {
                            if let Ok(new_inner_signal) = world.run_system_with(factory_system_id, value) {
                                let new_inner_id = new_inner_signal.clone().register(world);
                                let is_same_signal = {
                                    let manager_state = world.get::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                    manager_state
                                        .processors
                                        .get(index)
                                        .is_some_and(|(_, id)| *id == *new_inner_id)
                                };
                                if is_same_signal {
                                    new_inner_id.cleanup(world);
                                } else {
                                    let old_handle = {
                                        let mut manager_state =
                                            world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                        if index < manager_state.processors.len() {
                                            Some(manager_state.processors.remove(index).0)
                                        } else {
                                            None
                                        }
                                    };
                                    if let Some(handle) = old_handle {
                                        handle.cleanup(world);
                                    }
                                    let (new_handle, new_id, initial_value) = spawn_processor(
                                        world,
                                        state_and_queue_entity,
                                        *output_system_handle_clone,
                                        index,
                                        new_inner_signal,
                                    );
                                    {
                                        let mut manager_state =
                                            world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                        manager_state.processors.insert(index, (new_handle, new_id));
                                    }
                                    new_diffs.push(VecDiff::UpdateAt {
                                        index,
                                        value: initial_value,
                                    });
                                }
                            }
                        }
                        VecDiff::RemoveAt { index } => {
                            let (handle, handles_to_update) = {
                                let mut manager_state =
                                    world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                let (handle, _) = manager_state.processors.remove(index);
                                let handles = manager_state
                                    .processors
                                    .iter()
                                    .enumerate()
                                    .skip(index)
                                    .map(|(i, (proc_handle, _))| (**proc_handle, i))
                                    .collect::<Vec<_>>();
                                (handle, handles)
                            };
                            handle.cleanup(world);
                            for (proc_handle_entity, new_index) in handles_to_update {
                                world.get_mut::<ItemIndex>(*proc_handle_entity).unwrap().0 = new_index;
                            }
                            new_diffs.push(VecDiff::RemoveAt { index });
                        }
                        VecDiff::Move { old_index, new_index } => {
                            let handles_to_update = {
                                let mut manager_state =
                                    world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                let processor = manager_state.processors.remove(old_index);
                                manager_state.processors.insert(new_index, processor);
                                let start = old_index.min(new_index);
                                let end = old_index.max(new_index) + 1;
                                manager_state
                                    .processors
                                    .iter()
                                    .enumerate()
                                    .skip(start)
                                    .take(end - start)
                                    .map(|(i, (proc_handle, _))| (**proc_handle, i))
                                    .collect::<Vec<_>>()
                            };
                            for (proc_handle_entity, new_index) in handles_to_update {
                                world.get_mut::<ItemIndex>(*proc_handle_entity).unwrap().0 = new_index;
                            }
                            new_diffs.push(VecDiff::Move { old_index, new_index });
                        }
                        VecDiff::Push { value } => {
                            let index = {
                                let manager_state = world.get::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                manager_state.processors.len()
                            };
                            if let Ok(signal) = world.run_system_with(factory_system_id, value) {
                                let (handle, inner_id, initial_value) = spawn_processor(
                                    world,
                                    state_and_queue_entity,
                                    *output_system_handle_clone,
                                    index,
                                    signal,
                                );
                                {
                                    let mut manager_state =
                                        world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                    manager_state.processors.push((handle, inner_id));
                                }
                                new_diffs.push(VecDiff::Push { value: initial_value });
                            }
                        }
                        VecDiff::Pop => {
                            let maybe_handle = {
                                let mut manager_state =
                                    world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                manager_state.processors.pop().map(|(handle, _)| handle)
                            };
                            if let Some(handle) = maybe_handle {
                                handle.cleanup(world);
                                new_diffs.push(VecDiff::Pop);
                            }
                        }
                        VecDiff::Clear => {
                            let old_processors = {
                                let mut manager_state =
                                    world.get_mut::<ManagerState<S>>(state_and_queue_entity).unwrap();
                                manager_state.processors.drain(..).collect::<Vec<_>>()
                            };
                            for (handle, _) in old_processors {
                                handle.cleanup(world);
                            }
                            new_diffs.push(VecDiff::Clear);
                        }
                    }
                }
                if !new_diffs.is_empty() {
                    if let Some(mut queue) = world.get_mut::<QueuedVecDiffs<S::Item>>(state_and_queue_entity) {
                        queue.0.extend(new_diffs);
                    }
                    process_signals(world, [*output_system_handle_clone], Box::new(()));
                }
            };
            let manager_handle = self.for_each(manager_system_logic).register(world);
            world
                .entity_mut(state_and_queue_entity)
                .insert((
                    ManagerState::<S> {
                        processors: Vec::new(),
                        _phantom: PhantomData,
                    },
                    QueuedVecDiffs::<S::Item>(vec![]),
                ))
                .add_child(**manager_handle)
                .add_child(factory_system_id.entity())
                .add_child(**output_system_handle);
            *output_system_handle
        });
        MapSignal {
            signal,
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// MutableVec::from(["1", "two", "NaN", "four", "5"]).signal_vec()
    ///     .filter_map(|In(s): In<&'static str>| s.parse::<u32>().ok()); // outputs `SignalVec -> [1, 5]`
    /// ```
    fn filter_map<O, F, M>(self, system: F) -> FilterMap<Self, O>
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// #[derive(Resource, Clone)]
    /// struct Even(bool);
    ///
    /// MutableVec::from([1, 2, 3, 4]).signal_vec()
    ///      .filter_signal(|In(x): In<i32>| {
    ///          SignalBuilder::from_resource().map_in(move |even: Even| x % 2 == if even.0 { 0 } else { 1 })
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
            let SignalHandle(output_signal) =
                self
                    .for_each::<Vec<VecDiff<Self::Item>>, _, _, _>(
                        clone!(
                            (parent_entity) move |In(source_diffs): In<Vec<VecDiff<Self::Item>>>,
                            world: & mut World | {
                                let parent = parent_entity.get();
                                let mut generated_diffs = vec![];
                                for diff in source_diffs.into_iter() {
                                    match diff {
                                        VecDiff::Replace { values } => {
                                            let old_signals =
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        data
                                                            .items
                                                            .drain(..)
                                                            .map(|item| item.signal)
                                                            .collect::<Vec<_>>()
                                                    },
                                                );
                                            for signal in old_signals {
                                                signal.cleanup(world);
                                            }
                                            let mut new_items = Vec::with_capacity(values.len());
                                            let mut new_values = vec![];
                                            for (i, value) in values.into_iter().enumerate() {
                                                if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                                    let (handle, filtered) =
                                                        spawn_filter_signal::<Self::Item>(world, i, signal, parent);
                                                    if filtered {
                                                        new_values.push(value.clone());
                                                    }
                                                    new_items.push(FilterSignalItem {
                                                        signal: handle,
                                                        value,
                                                        filtered,
                                                    });
                                                }
                                            }
                                            with_filter_signal_data(
                                                world,
                                                parent,
                                                |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                    data.items = new_items;
                                                    data.diffs.clear();
                                                },
                                            );
                                            generated_diffs.push(VecDiff::Replace { values: new_values });
                                        },
                                        VecDiff::InsertAt { index, value } => {
                                            if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                                let (handle, filtered) =
                                                    spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                                let (new_filtered_index, signals_to_update) =
                                                    with_filter_signal_data(
                                                        world,
                                                        parent,
                                                        |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                            data.items.insert(index, FilterSignalItem {
                                                                signal: handle,
                                                                value: value.clone(),
                                                                filtered,
                                                            });
                                                            (
                                                                find_filter_signal_index(&data.items, index),
                                                                data
                                                                    .items
                                                                    .get((index + 1)..)
                                                                    .map(
                                                                        |items| items
                                                                            .iter()
                                                                            .map(|item| item.signal.clone())
                                                                            .collect::<Vec<_>>(),
                                                                    ),
                                                            )
                                                        },
                                                    );
                                                if let Some(to_update) = signals_to_update {
                                                    for signal in to_update {
                                                        if let Some(mut signal_index) =
                                                            world.get_mut::<FilterSignalIndex>(**signal) {
                                                            **signal_index += 1;
                                                        }
                                                    }
                                                }
                                                if filtered {
                                                    generated_diffs.push(VecDiff::InsertAt {
                                                        index: new_filtered_index,
                                                        value,
                                                    });
                                                }
                                            }
                                        },
                                        VecDiff::UpdateAt { index, value } => {
                                            let (old_signal, old_filtered, filtered_index) =
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |data: Mut<FilterSignalData<Self::Item>>| {
                                                        let filtered_index =
                                                            find_filter_signal_index(&data.items, index);
                                                        let item = &data.items[index];
                                                        (item.signal.clone(), item.filtered, filtered_index)
                                                    },
                                                );
                                            if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                                let (new_handle, new_filtered) =
                                                    spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                                old_signal.cleanup(world);
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        let item = &mut data.items[index];
                                                        item.signal = new_handle;
                                                        item.value = value.clone();
                                                        item.filtered = new_filtered;
                                                    },
                                                );
                                                if new_filtered {
                                                    if old_filtered {
                                                        generated_diffs.push(VecDiff::UpdateAt {
                                                            index: filtered_index,
                                                            value,
                                                        });
                                                    } else {
                                                        generated_diffs.push(VecDiff::InsertAt {
                                                            index: filtered_index,
                                                            value,
                                                        });
                                                    }
                                                } else if old_filtered {
                                                    generated_diffs.push(
                                                        VecDiff::RemoveAt { index: filtered_index },
                                                    );
                                                }
                                            }
                                        },
                                        VecDiff::RemoveAt { index } => {
                                            let (signal, filtered, filtered_index, signals_to_update) =
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        let filtered_index =
                                                            find_filter_signal_index(&data.items, index);
                                                        let item = data.items.remove(index);
                                                        let signals_to_update =
                                                            data
                                                                .items
                                                                .get(index..)
                                                                .map(
                                                                    |items| items
                                                                        .iter()
                                                                        .map(|item| item.signal.clone())
                                                                        .collect::<Vec<_>>(),
                                                                );
                                                        (
                                                            item.signal,
                                                            item.filtered,
                                                            filtered_index,
                                                            signals_to_update,
                                                        )
                                                    },
                                                );
                                            signal.cleanup(world);
                                            if let Some(to_update) = signals_to_update {
                                                for signal in to_update {
                                                    if let Some(mut signal_index) =
                                                        world.get_mut::<FilterSignalIndex>(**signal) {
                                                        **signal_index -= 1;
                                                    }
                                                }
                                            }
                                            if filtered {
                                                generated_diffs.push(VecDiff::RemoveAt { index: filtered_index });
                                            }
                                        },
                                        VecDiff::Push { value } => {
                                            if let Ok(signal) = world.run_system_with(system_id, value.clone()) {
                                                let index =
                                                    with_filter_signal_data(
                                                        world,
                                                        parent,
                                                        |data: Mut<FilterSignalData<Self::Item>>| data.items.len(),
                                                    );
                                                let (handle, filtered) =
                                                    spawn_filter_signal::<Self::Item>(world, index, signal, parent);
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        data.items.push(FilterSignalItem {
                                                            signal: handle,
                                                            value: value.clone(),
                                                            filtered,
                                                        });
                                                    },
                                                );
                                                if filtered {
                                                    generated_diffs.push(VecDiff::Push { value });
                                                }
                                            }
                                        },
                                        VecDiff::Pop => {
                                            let (signal, filtered) =
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        let item =
                                                            data.items.pop().expect("can't pop from empty vec");
                                                        (item.signal, item.filtered)
                                                    },
                                                );
                                            signal.cleanup(world);
                                            if filtered {
                                                generated_diffs.push(VecDiff::Pop);
                                            }
                                        },
                                        VecDiff::Move { old_index, new_index } => {
                                            let (filtered, old_filtered_index, new_filtered_index, items_to_update) =
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        let old_filtered_index =
                                                            find_filter_signal_index(&data.items, old_index);
                                                        let item = data.items.remove(old_index);
                                                        let filtered = item.filtered;
                                                        data.items.insert(new_index, item);
                                                        let new_filtered_index =
                                                            find_filter_signal_index(&data.items, new_index);

                                                        // Collect the signal handles that need their indices updated.
                                                        let items_to_update = data.items.iter().map(|item| item.signal.clone()).collect::<Vec<_>>();
                                                        (
                                                            filtered,
                                                            old_filtered_index,
                                                            new_filtered_index,
                                                            items_to_update,
                                                        )
                                                    },
                                                );

                                            // Now that the borrow from with_filter_signal_data is over, we can mutably access
                                            // world.
                                            for (i, signal) in items_to_update.into_iter().enumerate() {
                                                if let Some(mut signal_index) =
                                                    world.get_mut::<FilterSignalIndex>(**signal) {
                                                    **signal_index = i;
                                                }
                                            }
                                            if filtered {
                                                generated_diffs.push(VecDiff::Move {
                                                    old_index: old_filtered_index,
                                                    new_index: new_filtered_index,
                                                });
                                            }
                                        },
                                        VecDiff::Clear => {
                                            let signals =
                                                with_filter_signal_data(
                                                    world,
                                                    parent,
                                                    |mut data: Mut<FilterSignalData<Self::Item>>| {
                                                        data.diffs.clear();
                                                        data
                                                            .items
                                                            .drain(..)
                                                            .map(|item| item.signal)
                                                            .collect::<Vec<_>>()
                                                    },
                                                );
                                            for signal in signals {
                                                signal.cleanup(world);
                                            }
                                            generated_diffs.push(VecDiff::Clear);
                                        },
                                    }
                                }
                                let mut queued_diffs =
                                    with_filter_signal_data(
                                        world,
                                        parent,
                                        |mut data: Mut<FilterSignalData<Self::Item>>| data
                                            .diffs
                                            .drain(..)
                                            .collect::<Vec<_>>(),
                                    );
                                queued_diffs.extend(generated_diffs);
                                if queued_diffs.is_empty() {
                                    None
                                } else {
                                    Some(queued_diffs)
                                }
                            }
                        ),
                    )
                    .register(world);
            parent_entity.set(*output_signal);
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
    /// The [`Signal<Item = Option<usize>>`]s are deduped so any downstream [`Signal`]s are only run
    /// on frames where the index has changed.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::from([1, 2, 5]);
    /// let signal = vec.signal_vec().enumerate();
    /// // signal outputs `SignalVec -> [(Signal -> Some(0), 1), (Signal -> Some(1), 2), (Signal -> Some(2), 5)]`
    /// vec.write().remove(1);
    /// world.commands().queue(vec.flush());
    /// // signal outputs `SignalVec -> [(Signal -> Some(0), 1), (Signal -> Some(1), 5)]`
    /// ```
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
        Self::Item: Clone + 'static,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let processor_entity_handle = LazyEntity::new();
            let processor_logic = clone!(
                (processor_entity_handle) move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                world: & mut World | {
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

                    let create_index_signal = clone!((processor_entity_handle) move | key: usize | {
                        SignalBuilder::from_system(
                            clone!((processor_entity_handle) move |_: In<()>, query: Query<&EnumerateState>| {
                                Some(
                                    query
                                        .get(processor_entity_handle.get())
                                        .ok()
                                        .and_then(|s| s.key_to_index.get(&key).copied()),
                                )
                            }),
                        )
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
                            },
                            VecDiff::InsertAt { index, value } => {
                                let key = new_key_helper(&mut state);
                                state.ordered_keys.insert(index, key);
                                for (
                                    i,
                                    k,
                                ) in state
                                    .ordered_keys
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .skip(index)
                                    .collect::<Vec<_>>() {
                                    state.key_to_index.insert(k, i);
                                }
                                out_diffs.push(VecDiff::InsertAt {
                                    index,
                                    value: (create_index_signal(key), value),
                                });
                            },
                            VecDiff::UpdateAt { index, value } => {
                                let key = state.ordered_keys[index];
                                out_diffs.push(VecDiff::UpdateAt {
                                    index,
                                    value: (create_index_signal(key), value),
                                });
                            },
                            VecDiff::RemoveAt { index } => {
                                let removed_key = state.ordered_keys.remove(index);
                                state.key_to_index.remove(&removed_key);
                                for (
                                    i,
                                    k,
                                ) in state
                                    .ordered_keys
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .skip(index)
                                    .collect::<Vec<_>>() {
                                    state.key_to_index.insert(k, i);
                                }
                                out_diffs.push(VecDiff::RemoveAt { index });
                            },
                            VecDiff::Move { old_index, new_index } => {
                                let key = state.ordered_keys.remove(old_index);
                                state.ordered_keys.insert(new_index, key);
                                let start = old_index.min(new_index);
                                let end = old_index.max(new_index) + 1;
                                for (
                                    i,
                                    k,
                                ) in state
                                    .ordered_keys
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .skip(start)
                                    .take(end - start)
                                    .collect::<Vec<_>>() {
                                    state.key_to_index.insert(k, i);
                                }
                                out_diffs.push(VecDiff::Move {
                                    old_index,
                                    new_index,
                                });
                            },
                            VecDiff::Push { value } => {
                                let key = new_key_helper(&mut state);
                                let index = state.ordered_keys.len();
                                state.ordered_keys.push(key);
                                state.key_to_index.insert(key, index);
                                out_diffs.push(VecDiff::Push { value: (create_index_signal(key), value) });
                            },
                            VecDiff::Pop => {
                                if let Some(key) = state.ordered_keys.pop() {
                                    state.key_to_index.remove(&key);
                                    out_diffs.push(VecDiff::Pop);
                                }
                            },
                            VecDiff::Clear => {
                                state.ordered_keys.clear();
                                state.key_to_index.clear();
                                out_diffs.push(VecDiff::Clear);
                            },
                        }
                    }
                    if out_diffs.is_empty() {
                        None
                    } else {
                        Some(out_diffs)
                    }
                }
            );
            let handle = self
                .for_each::<Vec<VecDiff<(super::signal::Source<Option<usize>>, Self::Item)>>, _, _, _>(processor_logic)
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// MutableVec::from([1, 2, 3]).signal_vec().to_signal().map_in(|vec: Vec<i32>| vec[1]); // outputs `2`
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::from([1]);
    /// let signal = vec.signal_vec().is_empty();
    /// // `signal` outputs `false`
    /// vec.write().pop();
    /// world.commands().queue(vec.flush());
    /// // `signal` outputs `true`
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::from([1]);
    /// let signal = vec.signal_vec().is_empty();
    /// // `signal` outputs `1`
    /// vec.write().pop();
    /// world.commands().queue(vec.flush());
    /// // `signal` outputs `0`
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::from([1, 2, 3]);
    /// let signal = vec.signal_vec().is_empty();
    /// // `signal` outputs `6`
    /// vec.write().push(4);
    /// world.commands().queue(vec.flush());
    /// // `signal` outputs `10`
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
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
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
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
                                // Removing the last item: remove the item and the separator before it.
                                out_diffs.push(VecDiff::Pop); // item
                                out_diffs.push(VecDiff::Pop); // separator
                            } else {
                                // Removing from start/middle: remove the item and the separator after it.
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // item
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index }); // separator
                            }
                        }
                        VecDiff::Move { old_index, new_index } => {
                            let value = local_values.remove(old_index);
                            local_values.insert(new_index, value);

                            // The most robust way to handle move is to recalculate via Replace,
                            // as managing indices of items and separators during a move is complex.
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
        );
        Intersperse { signal }
    }

    // TODO: the example is clearly a copout ...
    // TODO: why won't doctest compile ?
    /// Place the [`Item`](SignalVec::Item) output by the [`System`] between adjacent items of this
    /// [`SignalVec`]; the [`System`] takes [`In`] an `impl Signal<Item = Option<usize>>` which
    /// outputs the index of the corresponding [`Item`](SignalVec::Item) or [`None`] if it has been
    /// removed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    /// use jonmo::signal::{Dedupe, Source};
    ///
    /// MutableVec::from([1, 2, 3]).signal_vec().intersperse_with(|_: In<Dedupe<Source<Option<usize>>>| 0); // outputs `SignalVec -> [1, 0, 2, 0, 3]`
    /// ```
    fn intersperse_with<F, M>(self, separator_system: F) -> IntersperseWith<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        F: IntoSystem<In<super::signal::Source<Option<usize>>>, Self::Item, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            // 1. Register the user's factory system once.
            let factory_system_id = world.register_system(separator_system);

            // 2. Create a handle for the state-holding entity.
            let state_entity_handle = LazyEntity::new();

            // 3. Define the helper to create an index signal for a given separator key. This signal reads from
            //    the central state component.
            let create_index_signal = clone!((state_entity_handle) move | key: usize | {
                SignalBuilder::from_system(
                    clone!(
                        (state_entity_handle) move |_: In<()>,
                        query: Query<&IntersperseState<Self::Item>>| {
                            Some(
                                query
                                    .get(state_entity_handle.get())
                                    .ok()
                                    .and_then(|s| s.key_to_index.get(&key).copied()),
                            )
                        }
                    ),
                )
            });

            // 4. Define the main processor logic that handles incoming diffs.
            let processor_logic = clone!(
                (state_entity_handle, create_index_signal) move |In(diffs): In<Vec<VecDiff<Self::Item>>>,
                world: & mut World | {
                    let state_entity = state_entity_handle.get();
                    let mut out_diffs = Vec::new();
                    for diff in diffs {
                        match diff {
                            VecDiff::Replace { values } => {
                                // --- Phase 1: Update internal state FIRST ---
                                let (new_keys, new_values) = {
                                    let mut state =
                                        world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                    state.local_values = values.clone();
                                    let num_separators = values.len().saturating_sub(1);

                                    // Generate new stable keys for the separators
                                    let start_key = state.next_key;
                                    state.next_key += num_separators;
                                    let keys = (start_key .. state.next_key).collect::<Vec<_>>();

                                    // Update the state completely
                                    state.separator_keys = keys.clone();
                                    state.key_to_index.clear();
                                    for (
                                        i,
                                        key,
                                    ) in state.separator_keys.iter().copied().enumerate().collect::<Vec<_>>() {
                                        state.key_to_index.insert(key, i);
                                    }
                                    (keys, values)
                                };

                                // --- Phase 2: Generate separators with correct state ---
                                let separators = new_keys.iter().map(|&key| {
                                    world.run_system_with(factory_system_id, create_index_signal(key)).unwrap()
                                }).collect::<Vec<Self::Item>>();

                                // --- Phase 3: Assemble the final output diff ---
                                let mut interspersed = Vec::new();
                                if !new_values.is_empty() {
                                    let mut items_iter = new_values.into_iter();
                                    interspersed.push(items_iter.next().unwrap());
                                    for (sep, item) in separators.into_iter().zip(items_iter) {
                                        interspersed.push(sep);
                                        interspersed.push(item);
                                    }
                                }
                                out_diffs.push(VecDiff::Replace { values: interspersed });
                            },
                            VecDiff::InsertAt { index, value } => {
                                let old_len =
                                    world
                                        .get::<IntersperseState<Self::Item>>(state_entity)
                                        .unwrap()
                                        .local_values
                                        .len();
                                out_diffs.push(VecDiff::InsertAt {
                                    index: 2 * index,
                                    value: value.clone(),
                                });

                                // Only add a separator if necessary
                                if index < old_len || old_len == 0 && index == 0 {
                                    // --- Phase 1: Update State ---
                                    let new_separator = {
                                        let mut state =
                                            world
                                                .get_mut::<IntersperseState<Self::Item>>(state_entity)
                                                .unwrap();
                                        let key = state.next_key;
                                        state.next_key += 1;
                                        state.separator_keys.insert(index, key);

                                        // Recalculate all indices after the insertion point
                                        for (i, k) in state.separator_keys.iter().copied().enumerate().skip(index).collect::<Vec<_>>() {
                                            state.key_to_index.insert(k, i);
                                        }

                                        // --- Phase 2: Generate Separator ---
                                        let separator: Self::Item = world.run_system_with(factory_system_id, create_index_signal(key)).unwrap();
                                        separator
                                    };

                                    // --- Phase 3: Add to output ---
                                    out_diffs.push(VecDiff::InsertAt {
                                        index: 2 * index + 1,
                                        value: new_separator,
                                    });
                                }

                                // Final state update for the item itself
                                world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap().local_values.insert(index, value);
                            },
                            VecDiff::UpdateAt { index, value } => {
                                world
                                    .get_mut::<IntersperseState<Self::Item>>(state_entity)
                                    .unwrap()
                                    .local_values[index] =
                                    value.clone();
                                out_diffs.push(VecDiff::UpdateAt {
                                    index: 2 * index,
                                    value,
                                });
                            },
                            VecDiff::RemoveAt { index } => {
                                let mut state =
                                    world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                state.local_values.remove(index);

                                // Remove the item
                                out_diffs.push(VecDiff::RemoveAt { index: 2 * index });

                                // If we removed an item that had a separator after it
                                if index < state.separator_keys.len() {
                                    let removed_key = state.separator_keys.remove(index);
                                    state.key_to_index.remove(&removed_key);

                                    // Remove the separator
                                    out_diffs.push(VecDiff::RemoveAt { index: 2 * index });

                                    // Recalculate indices for remaining separators
                                    for (i, k) in state.separator_keys.iter().copied().enumerate().skip(index).collect::<Vec<_>>() {
                                        state.key_to_index.insert(k, i);
                                    }
                                }
                            },
                            VecDiff::Push { value } => {
                                let mut state =
                                    world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                let old_len = state.local_values.len();
                                state.local_values.push(value.clone());
                                if old_len > 0 {
                                    // --- Phase 1: Update State ---
                                    let separator = {
                                        let mut state =
                                            world
                                                .get_mut::<IntersperseState<Self::Item>>(state_entity)
                                                .unwrap();
                                        let key = state.next_key;
                                        state.next_key += 1;
                                        state.separator_keys.push(key);
                                        state.key_to_index.insert(key, old_len - 1);

                                        // --- Phase 2: Generate Separator ---
                                        world.run_system_with(factory_system_id, create_index_signal(key)).unwrap()
                                    };

                                    // --- Phase 3: Add to output ---
                                    out_diffs.push(VecDiff::Push { value: separator });
                                }
                                out_diffs.push(VecDiff::Push { value });
                            },
                            VecDiff::Pop => {
                                let mut state =
                                    world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                if state.local_values.pop().is_some() {
                                    // Pop the item
                                    out_diffs.push(VecDiff::Pop);
                                    if let Some(removed_key) = state.separator_keys.pop() {
                                        state.key_to_index.remove(&removed_key);

                                        // Pop the separator
                                        out_diffs.push(VecDiff::Pop);
                                    }
                                }
                            },
                            VecDiff::Clear => {
                                let mut state =
                                    world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                state.local_values.clear();
                                state.separator_keys.clear();
                                state.key_to_index.clear();
                                out_diffs.push(VecDiff::Clear);
                            },
                            VecDiff::Move { old_index, new_index } => {
                                // `Move` is complex. The most robust way to handle it is to re-calculate the
                                // entire interspersed list via `Replace`, as this avoids subtle state and index
                                // bugs. First, apply the move to our local state.
                                let values = {
                                    let mut state =
                                        world.get_mut::<IntersperseState<Self::Item>>(state_entity).unwrap();
                                    let value = state.local_values.remove(old_index);
                                    state.local_values.insert(new_index, value);
                                    state.local_values.clone()
                                };

                                // Now, re-run the `Replace` logic from above.
                                let separators = {
                                    let num_separators = values.len().saturating_sub(1);
                                    let (keys, mut state) = {
                                        let mut state =
                                            world
                                                .get_mut::<IntersperseState<Self::Item>>(state_entity)
                                                .unwrap();
                                        let start_key = state.next_key;
                                        state.next_key += num_separators;
                                        ((start_key .. state.next_key).collect::<Vec<_>>(), state)
                                    };
                                    state.separator_keys = keys.clone();
                                    state.key_to_index.clear();
                                    for (
                                        i,
                                        key,
                                    ) in state.separator_keys.iter().copied().enumerate().collect::<Vec<_>>() {
                                        state.key_to_index.insert(key, i);
                                    }
                                    keys.iter().map(|&key| {
                                        world
                                            .run_system_with(factory_system_id, create_index_signal(key))
                                            .unwrap()
                                    }).collect::<Vec<Self::Item>>()
                                };
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
                            },
                        }
                    }
                    if out_diffs.is_empty() {
                        None
                    } else {
                        Some(out_diffs)
                    }
                }
            );

            // 5. Register the `processor_logic` to get the final signal system.
            let upstream_handle = self.register(world);
            let processor_handle =
                lazy_signal_from_system::<_, Vec<VecDiff<Self::Item>>, _, _, _>(processor_logic).register(world);

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
            processor_handle
        });
        IntersperseWith {
            signal,
            _marker: PhantomData,
        }
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
    /// MutableVec::from([3, 2, 1]).signal_vec().sort_by(|In((left, right)): In<(i32, i32)>| left.cmp(&right)); // outputs `SignalVec -> [1, 2, 3]`
    /// ```
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
                                    // Must search _before_ removing from values.
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
                                    // 1. Find the old sorted position before any changes.
                                    let old_pos = search(world, compare_system_id, &*state, index).unwrap();

                                    // 2. Remove the old entry from sorted tracking.
                                    state.sorted_indices.remove(old_pos);

                                    // 3. Update the underlying value in our source-of-truth vec.
                                    state.values[index] = value.clone();

                                    // 4. Find where the *new* value should be inserted in the now N-1 element list.
                                    let new_pos = search(world, compare_system_id, &*state, index).unwrap_err();

                                    // 5. Re-insert the item's index into our tracking at the new position.
                                    state.sorted_indices.insert(new_pos, index);

                                    // 6. If the item's final sorted position is the same, emit a single UpdateAt.
                                    //    Otherwise, emit a move as RemoveAt + InsertAt.
                                    if new_pos == old_pos {
                                        out_diffs.push(VecDiff::UpdateAt { index: old_pos, value });
                                    } else {
                                        out_diffs.push(VecDiff::RemoveAt { index: old_pos });
                                        // The new insertion index must account for the removal.
                                        // If the new position is after the old one, the removal shifted it left.
                                        let final_insert_pos = if new_pos > old_pos { new_pos - 1 } else { new_pos };
                                        out_diffs.push(VecDiff::InsertAt {
                                            index: final_insert_pos,
                                            value,
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
    /// MutableVec::from([3, 2, 1]).signal_vec().sort_by_cmp(); // outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn sort_by_cmp(self) -> SortBy<Self>
    where
        Self: Sized,
        Self::Item: Ord + Clone + SSs,
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
    /// MutableVec::from([3, 2, 1]).signal_vec().sort_by_key(|In(x): In<i32>| -x); // outputs `SignalVec -> [1, 2, 3]`
    /// ```
    fn sort_by_key<K, F, M>(self, system: F) -> SortByKey<Self>
    where
        Self: Sized,
        Self::Item: Clone + SSs,
        K: Ord + Clone + SSs,
        F: IntoSystem<In<Self::Item>, K, M> + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let key_system_id = world.register_system(system);

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
                                    // 1. Find the old sorted position before any changes.
                                    let old_pos = search(&*state, index).unwrap();

                                    // 2. Remove the old entry from sorted tracking.
                                    state.sorted_indices.remove(old_pos);

                                    // 3. Update the underlying value and its key in our source-of-truth vecs.
                                    let new_key = get_key(world, value.clone());
                                    state.values[index] = value.clone();
                                    state.keys[index] = new_key;

                                    // 4. Find where the *new* value should be inserted in the now N-1 element list.
                                    let new_pos = search(&*state, index).unwrap_err();

                                    // 5. Re-insert the item's index into our tracking at the new position.
                                    state.sorted_indices.insert(new_pos, index);

                                    // 6. If the item's final sorted position is the same, emit a single UpdateAt.
                                    //    Otherwise, emit a move as RemoveAt + InsertAt.
                                    if old_pos == new_pos {
                                        out_diffs.push(VecDiff::UpdateAt { index: old_pos, value });
                                    } else {
                                        out_diffs.push(VecDiff::RemoveAt { index: old_pos });
                                        // The new insertion index must account for the removal.
                                        let final_insert_pos = if new_pos > old_pos { new_pos - 1 } else { new_pos };
                                        out_diffs.push(VecDiff::InsertAt {
                                            index: final_insert_pos,
                                            value,
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

    /* fn flatten(self) -> Flatten<Self>
    where
        Self: Sized, // This should be a trait like SignalVec
        Self::Item: SignalVec + 'static + Clone,
        <Self::Item as SignalVec>::Item: Clone + SSs,
    {
        let signal = LazySignal::new(move |world: &mut World| {
            let parent_entity = LazyEntity::new();

            let SignalHandle(output_signal) = self
            .for_each::<Vec<VecDiff<<Self::Item as SignalVec>::Item>>, _, _, _>(
                clone!((parent_entity) move |In(diffs): In<Vec<VecDiff<Self::Item>>>, world: &mut World| {
                    let parent = parent_entity.get();
                    let mut new_diffs = vec![];

                    for diff in diffs {
                        match diff {
                            VecDiff::Push { value: inner_signal } => {
                                let index = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items.len());
                                let (item, initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
                                let offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items.iter().map(|i| i.values.len()).sum());
                                with_flatten_data(world, parent, |mut data| data.items.push(item));
                                for (i, value) in initial_values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
                                }
                            }
                            VecDiff::InsertAt { index, value: inner_signal } => {
                                let (item, initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
                                let offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items[..index].iter().map(|i| i.values.len()).sum());
                                with_flatten_data(world, parent, |mut data| data.items.insert(index, item));
                                let signals_to_update = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items.iter().map(|i| i.processor_handle.clone()).collect::<Vec<_>>());
                                for (i, handle) in signals_to_update.into_iter().enumerate() {
                                    if let Some(mut idx) = world.get_mut::<FlattenInnerIndex>(*handle) {
                                        idx.0 = i;
                                    }
                                }
                                for (i, value) in initial_values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
                                }
                            }
                            VecDiff::RemoveAt { index } => {
                                let (item, offset) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let offset = data.items[..index].iter().map(|i| i.values.len()).sum();
                                    (data.items.remove(index), offset)
                                });
                                item.processor_handle.cleanup(world);
                                for _ in 0..item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                            }
                            VecDiff::Pop => {
                                let (item, offset) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let offset = data.items.iter().map(|i| i.values.len()).sum::<usize>().saturating_sub(data.items.last().map_or(0, |i| i.values.len()));
                                    (data.items.pop().unwrap(), offset)
                                });
                                item.processor_handle.cleanup(world);
                                for _ in 0..item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                            }
                            VecDiff::Clear => {
                                let old_items = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.items.drain(..).collect::<Vec<_>>());
                                for item in old_items {
                                    item.processor_handle.cleanup(world);
                                }
                                new_diffs.push(VecDiff::Clear);
                            }
                            VecDiff::Replace { values } => {
                                let old_items = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.items.drain(..).collect::<Vec<_>>());
                                for item in old_items {
                                    item.processor_handle.cleanup(world);
                                }
                                new_diffs.push(VecDiff::Clear);
                                for (i, inner_signal) in values.into_iter().enumerate() {
                                    let (item, initial_values) = spawn_flatten_item(world, parent, i, inner_signal);
                                    with_flatten_data(world, parent, |mut data| data.items.push(item));
                                    for value in initial_values {
                                        new_diffs.push(VecDiff::Push { value });
                                    }
                                }
                            }
                            VecDiff::UpdateAt { index, value: inner_signal } => {
                                let (old_item, offset) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let offset = data.items[..index].iter().map(|i| i.values.len()).sum();
                                    (data.items.remove(index), offset)
                                });
                                old_item.processor_handle.cleanup(world);
                                for _ in 0..old_item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: offset });
                                }
                                let (new_item, initial_values) = spawn_flatten_item(world, parent, index, inner_signal);
                                with_flatten_data(world, parent, |mut data| data.items.insert(index, new_item));
                                for (i, value) in initial_values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: offset + i, value });
                                }
                            }
                            VecDiff::Move { old_index, new_index } => {
                                let (old_offset, moved_item, signals_to_update) = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| {
                                    let old_offset = data.items[..old_index].iter().map(|i| i.values.len()).sum();
                                    let moved_item = data.items.remove(old_index);
                                    data.items.insert(new_index, moved_item);
                                    let signals_to_update = data.items.iter().map(|i| i.processor_handle.clone()).collect::<Vec<_>>();
                                    (old_offset, data.items[new_index].clone(), signals_to_update)
                                });

                                for (i, handle) in signals_to_update.into_iter().enumerate() {
                                    if let Some(mut fi_index) = world.get_mut::<FlattenInnerIndex>(*handle) {
                                        fi_index.0 = i;
                                    }
                                }
                                for _ in 0..moved_item.values.len() {
                                    new_diffs.push(VecDiff::RemoveAt { index: old_offset });
                                }
                                let new_offset: usize = with_flatten_data::<<Self::Item as SignalVec>::Item, _>(world, parent, |data| data.items[..new_index].iter().map(|i| i.values.len()).sum());
                                for (i, value) in moved_item.values.into_iter().enumerate() {
                                    new_diffs.push(VecDiff::InsertAt { index: new_offset + i, value });
                                }
                            }
                        }
                    }

                    let queued_diffs = with_flatten_data(world, parent, |mut data: Mut<FlattenData<<Self::Item as SignalVec>::Item>>| data.diffs.drain(..).collect::<Vec<_>>());
                    new_diffs.extend(queued_diffs);

                    if new_diffs.is_empty() {
                        None
                    } else {
                        Some(new_diffs)
                    }
                }),
            )
            .register(world);

            parent_entity.set(output_signal);

            let SignalHandle(flusher) = SignalBuilder::from_entity(output_signal)
                .map::<Vec<VecDiff<<Self::Item as SignalVec>::Item>>, _, _, _>(
                    move |In(_), data: Query<&FlattenData<<Self::Item as SignalVec>::Item>>| {
                        if let Ok(data) = data.get(parent_entity.get()) {
                            if !data.diffs.is_empty() {
                                return Some(vec![]);
                            }
                        }
                        None
                    },
                )
                .register(world);

            pipe_signal(world, flusher, output_signal);

            world
                .entity_mut(output_signal)
                .insert(FlattenData::<<Self::Item as SignalVec>::Item> {
                    items: vec![],
                    diffs: vec![],
                });

            output_signal
        });

        Flatten {
            signal,
            _marker: PhantomData,
        }
    } */

    #[cfg(feature = "tracing")]
    /// Adds debug logging to this [`SignalVec`]'s raw [`VecDiff`] outputs.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let mut world = World::new();
    /// let mut vec = MutableVec::from([1, 2, 3]);
    /// let signal = vec.signal_vec().debug();
    /// // signal logs `[ Replace { values: [ 1, 2, 3 ] } ]`
    /// vec.write().push(4);
    /// world.commands().queue(vec.flush());
    /// // signal logs `[ Push { value: 4 } ]`
    /// ```
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

    /// Erases the type of this [`SignalVec`], allowing it to be used in conjunction with
    /// [`SignalVec`]s of other concrete types.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// let condition = true;
    /// let signal = if condition {
    ///     MutableVec::from([1, 2, 3]).signal_vec().map_in(|x: i32| x * 2).boxed() // this is a `Map<Source<i32>>`
    /// } else {
    ///     MutableVec::from([1, 2, 3]).signal_vec().filter(|In(x): In<i32>| x % 2 == 0).boxed() // this is a `Filter<Source<i32>>`
    /// }; // without the `.boxed()`, the compiler would not allow this
    /// ```
    fn boxed(self) -> Box<dyn SignalVec<Item = Self::Item>>
    where
        Self: Sized,
    {
        Box::new(self)
    }

    /// Erases the type of this [`SignalVec`], allowing it to be used in conjunction with
    /// [`SignalVec`]s of other concrete types, particularly in cases where the consumer requires
    /// [`Clone`], e.g. [`.switch_signal_vec`](SignalExt::switch_signal_vec).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// SignalBuilder::from_system(|_: In<()>| true)
    /// .switch_signal_vec(|In(condition): In<bool>| {
    ///     if condition {
    ///         MutableVec::from([1, 2, 3]).signal_vec().map_in(|x: i32| x * 2).boxed_clone() // this is a `Map<Source<i32>>`
    ///     } else {
    ///         MutableVec::from([1, 2, 3]).signal_vec().filter(|In(x): In<i32>| x % 2 == 0).boxed_clone() // this is a `Filter<Source<i32>>`
    ///     } // without the `.boxed_clone()`, the compiler would not allow this
    /// });
    /// ```
    fn boxed_clone(self) -> Box<dyn SignalVecClone<Item = Self::Item> + Send + Sync>
    where
        Self: Sized + Clone,
    {
        Box::new(self)
    }

    /// Activate this [`SignalVec`] and all its upstreams, causing them to be evaluated every frame
    /// until they are [`SignalHandle::cleanup`]-ed, see [`SignalHandle`].
    fn register(self, world: &mut World) -> SignalHandle
    where
        Self: Sized,
    {
        self.register_signal_vec(world)
    }
}

impl<T: ?Sized> SignalVecExt for T where T: SignalVec {}

/// Provides immutable access to the underlying [`Vec`].
pub struct MutableVecReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, MutableVecState<T>>,
}

impl<'a, T> Deref for MutableVecReadGuard<'a, T>
where
    T: Clone,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.guard.vec
    }
}

/// Provides limited mutable access to the underlying [`Vec`].
pub struct MutableVecWriteGuard<'a, T> {
    guard: RwLockWriteGuard<'a, MutableVecState<T>>,
}

impl<'a, T> MutableVecWriteGuard<'a, T>
where
    T: Clone,
{
    /// Appends an element to the back of this [`MutableVec`], queueing a [`VecDiff::Push`].
    pub fn push(&mut self, value: T) {
        self.guard.vec.push(value.clone());
        self.guard.pending_diffs.push(VecDiff::Push { value });
    }

    /// If this [`MutableVec`] is not empty, removes the last element and returns it, queueing a
    /// [`VecDiff::Pop`], otherwise returns [`None`].
    pub fn pop(&mut self) -> Option<T> {
        let result = self.guard.vec.pop();
        if result.is_some() {
            self.guard.pending_diffs.push(VecDiff::Pop);
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
        self.guard.pending_diffs.push(VecDiff::InsertAt { index, value });
    }

    /// Removes and returns the element at position `index` within the vector, shifting all elements
    /// after it to the left, queueing a [`VecDiff::RemoveAt`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        let value = self.guard.vec.remove(index);
        self.guard.pending_diffs.push(VecDiff::RemoveAt { index });
        value
    }

    /// Clears this [`MutableVec`], removing all values, queueing a [`VecDiff::Clear`] if it was not
    /// empty.
    pub fn clear(&mut self) {
        if !self.guard.vec.is_empty() {
            self.guard.vec.clear();
            self.guard.pending_diffs.push(VecDiff::Clear);
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
            self.guard.pending_diffs.push(VecDiff::UpdateAt { index, value });
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
            self.guard.pending_diffs.push(VecDiff::Move { old_index, new_index });
        }
    }

    /// Replaces the entire contents of this [`MutableVec`] with the provided `values`, queueing a
    /// [`VecDiff::Replace`].
    pub fn replace<A>(&mut self, values: A)
    where
        Vec<T>: From<A>,
        A: Clone,
    {
        self.guard.vec = values.clone().into();
        self.guard
            .pending_diffs
            .push(VecDiff::Replace { values: values.into() });
    }
}

impl<'a, T> Deref for MutableVecWriteGuard<'a, T>
where
    T: Clone,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.guard.vec
    }
}

struct MutableVecState<T> {
    vec: Vec<T>,
    pending_diffs: Vec<VecDiff<T>>,
    signal: Option<LazySignal>,
}

/// Wrapper around a [`Vec`] that tracks mutations as [`VecDiff`]s and emits them on
/// [`.flush`](MutableVec::flush), enabling diff-less constant time reactive updates for downstream
/// [`SignalVec`]s.
#[derive(Clone)]
pub struct MutableVec<T> {
    state: Arc<RwLock<MutableVecState<T>>>,
}

#[derive(Component)]
pub(crate) struct QueuedVecDiffs<T>(pub(crate) Vec<VecDiff<T>>);

impl<T: Clone> Clone for QueuedVecDiffs<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, A> From<T> for MutableVec<A>
where
    Vec<A>: From<T>,
{
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

pub(crate) trait Replayable {
    fn trigger(self) -> Box<dyn FnOnce(&mut World) + Send + Sync>;
}

#[derive(Component)]
pub(crate) struct VecReplayTrigger(Box<dyn FnOnce(&mut World) + Send + Sync>);

impl Replayable for VecReplayTrigger {
    fn trigger(self) -> Box<dyn FnOnce(&mut World) + Send + Sync> {
        self.0
    }
}

impl<T> MutableVec<T> {
    /// Constructs a new, empty [`MutableVec<T>`].
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

    /// Locks this [`MutableVec`] with shared read access, blocking the current thread until it can
    /// be acquired, see [`RwLock::read`].
    pub fn read(&self) -> MutableVecReadGuard<'_, T> {
        MutableVecReadGuard {
            guard: self.state.read().unwrap(),
        }
    }

    /// Locks this [`MutableVec`] with exclusive write access, blocking the current thread until it
    /// can be acquired, see [`RwLock::write`].
    pub fn write(&self) -> MutableVecWriteGuard<'_, T> {
        MutableVecWriteGuard {
            guard: self.state.write().unwrap(),
        }
    }

    fn get_or_create_broadcaster_signal(&self) -> LazySignal
    where
        T: Clone + SSs,
    {
        let mut state = self.state.write().unwrap();

        // If the signal already exists, just clone and return it.
        if let Some(lazy_signal) = &state.signal {
            return lazy_signal.clone();
        }

        // Otherwise, create the broadcaster signal for the first time.
        let broadcaster_lazy_signal = LazySignal::new(move |world: &mut World| {
            let self_entity = LazyEntity::new();

            // This is the system for the one-and-only broadcaster. It just drains diffs that `flush` has put
            // into its component.
            let source_system_logic = clone!((self_entity) move |_: In<()>, world: &mut World| {
                if let Some(mut diffs) = world.get_mut::<QueuedVecDiffs<T>>(self_entity.get()) {
                    if diffs.0.is_empty() {
                        None
                    } else {
                        Some(diffs.0.drain(..).collect())
                    }
                } else {
                    None
                }
            });
            let signal_system = register_signal::<(), Vec<VecDiff<T>>, _, _, _>(world, source_system_logic);
            self_entity.set(*signal_system);

            // The broadcaster itself does not have an initial state to replay. It just needs the component to
            // receive flushed diffs.
            world.entity_mut(*signal_system).insert(QueuedVecDiffs::<T>(vec![]));
            signal_system
        });

        // Store it for future calls.
        state.signal = Some(broadcaster_lazy_signal.clone());
        broadcaster_lazy_signal
    }

    /// Returns a [`Source`] signal from this [`MutableVec`].
    pub fn signal_vec(&self) -> Source<T>
    where
        T: Clone + SSs,
    {
        let broadcaster_signal = self.get_or_create_broadcaster_signal();
        let replay_lazy_signal = LazySignal::new(clone!((self.state => state) move | world: & mut World | {
            let self_entity = LazyEntity::new();
            let broadcaster_system = broadcaster_signal.register(world);

            // This system has two modes tracked by its `has_run` Local state:
            //
            // 1. First run: It ignores upstream diffs and processes its own QueuedVecDiffs (the initial state).
            //
            // 2. Subsequent runs: It processes diffs from the upstream broadcaster.
            let replay_system_logic =
                clone!(
                    (self_entity) move | In(upstream_diffs): In<Vec<VecDiff<T>>>,
                    world: & mut World,
                    mut has_run: Local < bool >| {
                        if !*has_run {
                            // First run is triggered manually. Process the initial state queued on this
                            // system's entity.
                            *has_run = true;
                            let initial_diffs =
                                world
                                    .get_entity_mut(self_entity.get())
                                    .ok()
                                    .and_then(|mut entity| entity.take::<QueuedVecDiffs<T>>())
                                    .map(|queued| queued.0)
                                    .unwrap_or_default();
                            if initial_diffs.is_empty() {
                                None
                            } else {
                                Some(initial_diffs)
                            }
                        } else {
                            // Subsequent runs are triggered by the broadcaster. Process upstream diffs.
                            if upstream_diffs.is_empty() {
                                None
                            } else {
                                Some(upstream_diffs)
                            }
                        }
                    }
                );

            // 1. Register the replay system.
            let replay_signal = register_signal::<_, Vec<VecDiff<T>>, _, _, _>(world, replay_system_logic);
            self_entity.set(*replay_signal);

            // The trigger now unconditionally pokes the replay system with an empty input.
            // The replay system's own logic will handle whether it's the first run or not.
            let trigger = Box::new(move |world: &mut World| {
                process_signals(world, [replay_signal], Box::new(Vec::<VecDiff<T>>::new()));
            });

            // 2. Queue the initial state for this new subscriber.
            let initial_vec = state.read().unwrap().vec.clone();
            let initial_diffs = if !initial_vec.is_empty() {
                vec![VecDiff::Replace { values: initial_vec }]
            } else {
                vec![]
            };
            world.entity_mut(*replay_signal).insert((QueuedVecDiffs(initial_diffs), VecReplayTrigger(trigger)));

            // 3. Pipe the broadcaster to the replay node.
            pipe_signal(world, broadcaster_system, replay_signal);
            replay_signal
        }));
        Source {
            signal: replay_lazy_signal,
            _marker: PhantomData,
        }
    }

    /// Emits any pending [`VecDiff`]s to downstream [`SignalVec`]s.
    pub fn flush_into_world(&self, world: &mut World)
    where
        T: SSs,
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
            && let Some(mut queued_diffs) = entity.get_mut::<QueuedVecDiffs<T>>()
        {
            queued_diffs.0.append(&mut state.pending_diffs);
        }
    }

    /// Returns an `impl Command` that can be passed to [`Commands::queue`] to flush this
    /// [`MutableVec`]'s pending [`VecDiff`]s, see [`.flush_into_world`](Self::flush_into_world).
    pub fn flush(&self) -> impl Command
    where
        T: Clone + SSs,
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

    #[derive(Resource, Default, Debug)]
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

    // Helper function to apply a series of diffs to a vector to check the final state.
    fn apply_diffs<T: Clone>(initial: &mut Vec<T>, diffs: &[VecDiff<T>]) {
        for diff in diffs {
            diff.apply_to_vec(initial);
        }
    }

    #[derive(Resource, Default, Debug)]
    struct FinalSignalOutput<T: SSs + Clone + fmt::Debug>(Option<T>);

    fn capture_final_output<T>(In(value): In<T>, mut output: ResMut<FinalSignalOutput<T>>)
    where
        T: SSs + Clone + fmt::Debug,
    {
        output.0 = Some(value);
    }

    fn get_and_clear_final_output<T: SSs + Clone + fmt::Debug>(world: &mut World) -> Option<T> {
        world
            .get_resource_mut::<FinalSignalOutput<T>>()
            .and_then(|mut res| res.0.take())
    }

    #[test]
    fn test_for_each() {
        // A comprehensive test for `SignalVecExt::for_each`. This test verifies that
        // the combinator correctly receives a batch of `VecDiff`s and transforms the
        // `SignalVec` into a standard `Signal`.

        // --- 1. Setup ---
        let mut app = create_test_app();

        // The output of our `for_each` system will be the full, reconstructed vector.
        app.init_resource::<FinalSignalOutput<Vec<String>>>();
        let source_vec = MutableVec::from(vec!["a".to_string(), "b".to_string()]);

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
            let mut writer = source_vec.write();
            writer.push("c".to_string()); // source: ["a", "b", "c"]
            writer.set(0, "A".to_string()); // source: ["A", "b", "c"]
            writer.remove(1); // source: ["A", "c"]
        }
        source_vec.flush_into_world(app.world_mut());
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
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let expected_cleared_state: Vec<String> = Vec::new();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(expected_cleared_state),
            "State after Clear was not reconstructed correctly"
        );

        // --- 6. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_map() {
        // A comprehensive test for `SignalVecExt::map`, covering all `VecDiff`
        // types in both success and system-failure scenarios.

        // --- 1. Setup ---
        let mut app = create_test_app();

        // The output `SignalVec` will contain `String`s.
        app.init_resource::<SignalVecOutput<String>>();

        // The source vector that we will mutate.
        let source_vec = MutableVec::from(vec![1, 10]);

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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["1x".to_string(), "10x".to_string()]);

        // Test: `Push`
        source_vec.write().push(4);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(
            diffs[0],
            VecDiff::Push {
                value: "4x".to_string()
            }
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["1x".to_string(), "10x".to_string(), "4x".to_string()]
        );

        // Test: `InsertAt`
        source_vec.write().insert(1, 5);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["1x".to_string(), "5x".to_string(), "10x".to_string(), "4x".to_string()]
        );

        // Test: `UpdateAt`
        source_vec.write().set(0, 99);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["99x".to_string(), "5x".to_string(), "10x".to_string(), "4x".to_string()]
        );

        // Test: `RemoveAt` (passes through without mapping)
        source_vec.write().remove(2); // removes 10
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["99x".to_string(), "5x".to_string(), "4x".to_string()]
        );

        // Test: `Pop` (passes through without mapping)
        source_vec.write().pop(); // removes 4
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Pop);
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["99x".to_string(), "5x".to_string()]);

        // Test: `Move` (passes through without mapping)
        source_vec.write().move_item(1, 0); // moves 5 to front
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["5x".to_string(), "99x".to_string()]);

        // Test: `Clear` (passes through without mapping)
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Clear);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 3. Test System Failure Scenarios ---
        // Now, remove the resource to cause the mapping system to fail.
        app.world_mut().remove_resource::<MapFactor>();

        // Test: `Push` with failing system.
        // The `run_system_with(...).ok()` will return `None`, so the item is dropped.
        source_vec.write().push(88);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert!(
            diffs.is_empty(),
            "Pushing with a failing system should produce no diff."
        );
        assert!(current_state.is_empty(), "State should remain empty.");

        // Test: `Replace` with failing system.
        // `map` will filter_map over the values, and all will fail. The output should be
        // a `Replace` diff with an empty vector.
        source_vec.write().replace(vec![1, 2, 3]);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Replace with a failing system should produce one diff.");
        assert_eq!(
            diffs[0],
            VecDiff::Replace { values: vec![] },
            "Replace with failing system should result in an empty vector."
        );
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 4. Test System Recovery ---
        // Re-introduce the resource. The system should now succeed again.
        app.insert_resource(MapFactor("y".to_string()));

        // Test: `Push` after recovery. The source vec is [1, 2, 3] from the last
        // replace. Now we push 7. The `map` only processes the `Push` diff.
        source_vec.write().push(7);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Push after recovery should produce one diff.");
        assert_eq!(
            diffs[0],
            VecDiff::Push {
                value: "7y".to_string()
            }
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["7y".to_string()],
            "State should only contain the newly pushed item after recovery."
        );

        // --- 5. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_in() {
        // A comprehensive test for `SignalVecExt::map_in`, covering all `VecDiff`
        // types and verifying it correctly uses state captured by the closure.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<String>>();

        let source_vec = MutableVec::from(vec![1, 10, 100]);

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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["1a".to_string(), "10a".to_string(), "100a".to_string()]
        );

        // --- Change the captured state ---
        // Modify the captured factor. Subsequent mapping operations should use this
        // new value.
        *mapping_factor.write().unwrap() = "b".to_string();

        // Test: `Push` using the new factor "b"
        source_vec.write().push(4);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(
            diffs[0],
            VecDiff::Push {
                value: "4b".to_string()
            }
        );
        apply_diffs(&mut current_state, &diffs);
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
        source_vec.write().insert(1, 5);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
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
        source_vec.write().set(0, 99);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
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
        source_vec.write().remove(2); // removes 10
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
        apply_diffs(&mut current_state, &diffs);
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
        source_vec.write().pop(); // removes 4
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Pop);
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["99b".to_string(), "5b".to_string(), "100a".to_string()]
        );

        // Test: `Move`
        source_vec.write().move_item(1, 0); // moves 5 to front
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["5b".to_string(), "99b".to_string(), "100a".to_string()]
        );

        // Test: `Clear`
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Clear);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 3. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_map_in_ref() {
        // A comprehensive test for `SignalVecExt::map_in_ref`, covering all `VecDiff`
        // types and verifying it correctly uses a closure that takes a reference.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<String>>();

        // The source vector contains integers.
        let source_vec = MutableVec::from(vec![10, 20]);

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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["10".to_string(), "20".to_string()]);

        // Test: `Push`
        source_vec.write().push(30);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(
            diffs[0],
            VecDiff::Push {
                value: "30".to_string()
            }
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["10".to_string(), "20".to_string(), "30".to_string()]
        );

        // Test: `InsertAt`
        source_vec.write().insert(1, 5);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["10".to_string(), "5".to_string(), "20".to_string(), "30".to_string()]
        );

        // Test: `UpdateAt`
        source_vec.write().set(0, 99);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec!["99".to_string(), "5".to_string(), "20".to_string(), "30".to_string()]
        );

        // --- Test pass-through diffs ---
        // These diffs don't involve mapping values, so they are simply forwarded.

        // Test: `RemoveAt`
        source_vec.write().remove(2); // removes 20
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["99".to_string(), "5".to_string(), "30".to_string()]);

        // Test: `Pop`
        source_vec.write().pop(); // removes 30
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Pop);
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["99".to_string(), "5".to_string()]);

        // Test: `Move`
        source_vec.write().move_item(1, 0); // moves 5 to front
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["5".to_string(), "99".to_string()]);

        // Test: `Clear`
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Clear);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 3. Cleanup ---
        handle.cleanup(app.world_mut());
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
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<Name>>();

        // Setup: Create entities with `Name` components that our signals will track.
        let entity_a = app.world_mut().spawn(Name::new("Alice")).id();
        let entity_b = app.world_mut().spawn(Name::new("Bob")).id();

        // The source vec contains entities. The goal is to create a derived vec that
        // contains the _names_ of these entities.
        let entity_vec = MutableVec::from([entity_a, entity_b]);

        // This "factory" system takes an entity and creates a signal that tracks its
        // `Name`.
        let factory_system = |In(entity): In<Entity>| SignalBuilder::from_component::<Name>(entity).dedupe();

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
        entity_vec.write().push(entity_c);
        entity_vec.flush_into_world(app.world_mut());
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
        entity_vec.write().insert(1, entity_d);
        entity_vec.flush_into_world(app.world_mut());
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
        entity_vec.write().remove(2);
        entity_vec.flush_into_world(app.world_mut());
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
        entity_vec.write().set(0, entity_e);
        entity_vec.flush_into_world(app.world_mut());
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
        entity_vec.write().move_item(2, 0);
        entity_vec.flush_into_world(app.world_mut());
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
        entity_vec.write().pop();
        entity_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Pop should produce one diff");
        assert_eq!(diffs[0], VecDiff::Pop, "Pop diff is incorrect");

        // Test 10: Source SignalVec Clear.
        entity_vec.write().clear();
        entity_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<Name>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Clear should produce one diff");
        assert_eq!(diffs[0], VecDiff::Clear, "Clear diff is incorrect");
        assert!(entity_vec.read().is_empty());
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_filter() {
        // A comprehensive test for `SignalVecExt::filter`, covering all `VecDiff`
        // types and edge cases like items changing their filter status.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();

        // The source vector contains a mix of values to be filtered.
        let source_vec = MutableVec::from(vec![1, 2, 3, 4]);

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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![2, 4]);

        // --- 3. Test `Push` ---
        // Push a value that passes the filter (6).
        source_vec.write().push(6);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Push { value: 6 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![2, 4, 6]);

        // Push a value that fails the filter (7).
        source_vec.write().push(7);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert!(diffs.is_empty(), "Pushing a filtered-out value should produce no diff.");
        assert_eq!(current_state, vec![2, 4, 6]); // State should be unchanged.

        // --- 4. Test `UpdateAt` (Status Change) ---
        // Update an item from filtered-out to included (3 -> 8).
        // Source is [1, 2, 3, 4, 6, 7]. Update index 2.
        // It should be inserted into the output at index 1 (after 2).
        source_vec.write().set(2, 8);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(
            diffs[0],
            VecDiff::InsertAt { index: 1, value: 8 },
            "Update from filtered to included failed"
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![2, 8, 4, 6]);

        // Update an item from included to filtered-out (4 -> 9).
        // Source is now [1, 2, 8, 4, 6, 7]. Update index 3.
        // It should be removed from the output at index 2 (where 4 was).
        source_vec.write().set(3, 9);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(
            diffs[0],
            VecDiff::RemoveAt { index: 2 },
            "Update from included to filtered failed"
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![2, 8, 6]);

        // --- 5. Test `RemoveAt` ---
        // Remove an included item (2) from the source at index 1.
        source_vec.write().remove(1);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 }); // It was at the start of the output
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![8, 6]);

        // --- 6. Test `Move` ---
        // Source is now [1, 8, 9, 6, 7]. Filtered is [8, 6].
        // Move 8 (source index 1) to after 6 (source index 3).
        // Source becomes [1, 9, 6, 8, 7].
        // This should move index 0 (value 8) to index 1 in the output.
        source_vec.write().move_item(1, 3);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![6, 8]);

        // --- 7. Test `Clear` ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Clear);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 8. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_filter_map() {
        // A comprehensive test for `SignalVecExt::filter_map`, covering all `VecDiff`
        // types and the three status-change scenarios for `UpdateAt`.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<u32>>();

        // The source vector contains strings, some of which are parsable as numbers.
        let source_vec = MutableVec::from(vec!["10", "foo", "20", "bar"]);

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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![10, 20]);

        // --- 3. Test `Push` ---
        // Push a value that passes the filter_map ("30").
        source_vec.write().push("30");
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Push { value: 30 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![10, 20, 30]);

        // Push a value that fails the filter_map ("baz").
        source_vec.write().push("baz");
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert!(diffs.is_empty(), "Pushing a filtered-out value should produce no diff.");

        // --- 4. Test `UpdateAt` Status Changes ---
        // Case 1: None -> Some (Update "foo" at source index 1 to "40").
        source_vec.write().set(1, "40");
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::InsertAt { index: 1, value: 40 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![10, 40, 20, 30]);

        // Case 2: Some -> None (Update "20" at source index 2 to "qux").
        source_vec.write().set(2, "qux");
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 2 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![10, 40, 30]);

        // Case 3: Some -> Some (Update "10" at source index 0 to "50").
        source_vec.write().set(0, "50");
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::UpdateAt { index: 0, value: 50 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![50, 40, 30]);

        // --- 5. Test `RemoveAt` ---
        // Source is now: ["50", "40", "qux", "bar", "30", "baz"]
        // Filtered is: [50, 40, 30]
        // Remove "40" from source at index 1. This is at output index 1.
        source_vec.write().remove(1);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 1 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![50, 30]);

        // --- 6. Test `Move` ---
        // Source is now: ["50", "qux", "bar", "30", "baz"]
        // Filtered is: [50, 30]
        // Move "30" (source index 3) to the front (source index 0).
        // This moves output index 1 to output index 0.
        source_vec.write().move_item(3, 0);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![30, 50]);

        // --- 7. Test `Clear` ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], VecDiff::Clear);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 8. Cleanup ---
        handle.cleanup(app.world_mut());
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
        // A resource to control the filter's behavior.
        #[derive(Resource, Clone, PartialEq, Debug)]
        struct FilterMode(bool);

        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();

        // --- Setup --- Initially, filter for even numbers.
        app.insert_resource(FilterMode(true));
        let source_vec = MutableVec::from(vec![1, 2, 3, 4]);
        let filtered_signal = source_vec.signal_vec().filter_signal(|In(val): In<i32>| {
            SignalBuilder::from_resource::<FilterMode>().map(move |In(mode): In<FilterMode>| {
                // even or odd
                if mode.0 { val % 2 == 0 } else { val % 2 != 0 }
            })
        });
        let handle = filtered_signal
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        // --- 1. Initial State (Even) ---
        app.update();
        let mut current_state = vec![];
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(current_state, vec![2, 4], "Initial state (even) is incorrect.");

        // --- 2. Change Filter (to Odd) ---
        *app.world_mut().resource_mut::<FilterMode>() = FilterMode(false);
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(current_state, vec![1, 3], "State after switching to odd is incorrect.");

        // --- 3. Source Vec Push (respecting Odd filter) --- Push an odd number, should
        // appear.
        source_vec.write().push(5);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(
            current_state,
            vec![1, 3, 5],
            "State after pushing odd number is incorrect."
        );

        // Push an even number, should be filtered out.
        source_vec.write().push(6);
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));

        // Source vec is now [1, 2, 3, 4, 5, 6]. Evens are [2, 4, 6].
        assert_eq!(
            current_state,
            vec![2, 4, 6],
            "State after switching back to even is incorrect."
        );

        // --- 5. Source Vec RemoveAt --- Remove `4` (even) from source vec (at index 3).
        source_vec.write().remove(3);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(current_state, vec![2, 6], "State after removing '4' is incorrect.");

        // --- 6. Source Vec UpdateAt --- Update `3` (odd, at index 2) to `8` (even). This
        // should insert `8` into the filtered list.
        source_vec.write().set(2, 8);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));

        // Source is now [1, 2, 8, 5, 6]. Evens are [2, 8, 6].
        assert_eq!(
            current_state,
            vec![2, 8, 6],
            "State after updating 3 to 8 is incorrect."
        );

        // Update `6` (even, at index 4) to `7` (odd). This should remove `6` from the
        // filtered list.
        source_vec.write().set(4, 7);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));

        // Source is now [1, 2, 8, 5, 7]. Evens are [2, 8].
        assert_eq!(current_state, vec![2, 8], "State after updating 6 to 7 is incorrect.");

        // --- 7. Source Vec Clear ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Clear should produce one diff.");
        assert_eq!(diffs[0], VecDiff::Clear, "Expected a Clear diff.");
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty(), "State after clear should be empty.");

        // --- Cleanup ---
        handle.cleanup(app.world_mut());
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
        let source_vec = MutableVec::from(vec![-2, -1, 0, 1, 2]);

        // --- The Signal Chain ---
        let final_signal = source_vec
            .signal_vec()
            // Filter 1: controlled by IsPositive resource.
            .filter_signal(|In(val): In<i32>| {
                SignalBuilder::from_resource::<IsPositive>()
                    .map(move |In(res): In<IsPositive>| if res.0 { val > 0 } else { val <= 0 })
            })
            // Filter 2: controlled by IsEven resource.
            .filter_signal(|In(val): In<i32>| {
                SignalBuilder::from_resource::<IsEven>()
                    .map(move |In(res): In<IsEven>| if res.0 { val % 2 == 0 } else { val % 2 != 0 })
            });
        let handle = final_signal
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        // --- 1. Initial State --- IsPositive(false) -> pass [-2, -1, 0] IsEven(true)
        //  -> from the above, pass [-2, 0]
        app.update();
        let mut current_state = vec![];
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(current_state, vec![-2, 0], "Initial state is incorrect.");

        // --- 2. Change the first filter's condition --- Now IsPositive(true) -> pass [1,
        // 2] The second filter IsEven(true) is unchanged -> from the new set, pass [2]
        // The combined effect should be to remove [-2, 0] and insert [2].
        *app.world_mut().resource_mut::<IsPositive>() = IsPositive(true);
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
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
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
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
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(
            current_state,
            vec![-1],
            "State after flipping first filter back is incorrect."
        );

        // --- 5. Test source vec modification --- Add an item that should pass both
        // current filters: IsPositive(false) and IsEven(false) -> Add -3. It should be
        // pushed to the end of the filtered list.
        source_vec.write().push(-3);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<i32>(app.world_mut()));
        assert_eq!(current_state, vec![-1, -3], "State after pushing -3 is incorrect.");

        // --- Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_to_signal() {
        // A comprehensive test for `SignalVecExt::to_signal`, verifying that it
        // correctly reconstructs the full vector state from all `VecDiff` types.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<FinalSignalOutput<Vec<String>>>();
        let source_vec = MutableVec::from(vec!["a".to_string(), "b".to_string()]);

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
        source_vec.write().push("c".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            "State after Push is incorrect."
        );

        // Test RemoveAt
        source_vec.write().remove(1); // removes "b"
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec!["a".to_string(), "c".to_string()]),
            "State after RemoveAt is incorrect."
        );

        // Test InsertAt
        source_vec.write().insert(0, "x".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec!["x".to_string(), "a".to_string(), "c".to_string()]),
            "State after InsertAt is incorrect."
        );

        // Test UpdateAt
        source_vec.write().set(2, "C".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec!["x".to_string(), "a".to_string(), "C".to_string()]),
            "State after UpdateAt is incorrect."
        );

        // Test Move
        source_vec.write().move_item(2, 0); // moves "C" to the front
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec!["C".to_string(), "x".to_string(), "a".to_string()]),
            "State after Move is incorrect."
        );

        // Test Pop
        source_vec.write().pop(); // removes "a"
        source_vec.flush_into_world(app.world_mut());
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
            let mut writer = source_vec.write();
            writer.push("y".to_string()); // State: ["C", "x", "y"]
            writer.remove(0); // State: ["x", "y"]
        }
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec!["x".to_string(), "y".to_string()]),
            "State after batched mutations is incorrect."
        );

        // --- 6. Test Clear ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<Vec<String>>(app.world_mut()),
            Some(vec![]),
            "State after Clear is incorrect."
        );

        // --- 7. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_is_empty() {
        // A comprehensive test for `SignalVecExt::is_empty`, verifying that it
        // correctly emits true/false and re-emits on every source change, as intended.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<FinalSignalOutput<bool>>();
        // Start with a non-empty vec to test the `false` state first.
        let source_vec = MutableVec::from(vec!["a".to_string(), "b".to_string()]);

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
        source_vec.write().push("c".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(false),
            "Pushing to a non-empty vec should re-emit `false`."
        );

        // UpdateAt: size 3 -> 3. Still not empty.
        source_vec.write().set(0, "A".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(false),
            "Updating a non-empty vec should re-emit `false`."
        );

        // Move: size 3 -> 3. Still not empty.
        source_vec.write().move_item(0, 2);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(false),
            "Moving within a non-empty vec should re-emit `false`."
        );

        // --- 4. Test Transition to Empty ---
        // Current size is 3. We pop three times.
        source_vec.write().pop(); // size 3 -> 2
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(false),
            "Popping to size 2 should re-emit `false`."
        );

        source_vec.write().pop(); // size 2 -> 1
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(false),
            "Popping to size 1 should re-emit `false`."
        );

        // This is the pop that makes it empty. The state changes from false to true.
        source_vec.write().pop(); // size 1 -> 0
        source_vec.flush_into_world(app.world_mut());
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
        source_vec.write().push("z".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(false),
            "Pushing to an empty vec should emit `false`."
        );

        // --- 7. Test Clear Operation ---
        // The state changes from false to true.
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<bool>(app.world_mut()),
            Some(true),
            "Clear on a non-empty vec should emit `true`."
        );

        // --- 8. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_len() {
        // A comprehensive test for `SignalVecExt::len`, verifying that it correctly
        // reports the vector's length and re-emits on every source change.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<FinalSignalOutput<usize>>();
        // Start with a non-empty vec.
        let source_vec = MutableVec::from(vec!["a".to_string(), "b".to_string()]);

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
        source_vec.write().push("c".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(3),
            "Length after Push should be 3."
        );

        // InsertAt: size 3 -> 4
        source_vec.write().insert(1, "x".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(4),
            "Length after InsertAt should be 4."
        );

        // RemoveAt: size 4 -> 3
        source_vec.write().remove(2); // removes "b"
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(3),
            "Length after RemoveAt should be 3."
        );

        // Pop: size 3 -> 2
        source_vec.write().pop(); // removes "c"
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(2),
            "Length after Pop should be 2."
        );

        // --- 4. Test Length-Preserving Operations ---
        // These should still re-emit the current length because the underlying vec changed.

        // UpdateAt: size 2 -> 2
        source_vec.write().set(0, "A".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(2),
            "UpdateAt should re-emit length 2."
        );

        // Move: size 2 -> 2
        source_vec.write().move_item(0, 1);
        source_vec.flush_into_world(app.world_mut());
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
            let mut writer = source_vec.write();
            writer.push("y".to_string()); // len -> 3
            writer.remove(0); // len -> 2
        }
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(2),
            "Batched mutations with net-zero length change should emit final length."
        );

        // --- 7. Test Clear Operation ---
        // State changes from 2 to 0.
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(0),
            "Length after Clear should be 0."
        );

        // --- 8. Test Transition from Empty ---
        source_vec.write().push("z".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<usize>(app.world_mut()),
            Some(1),
            "Length after pushing to an empty vec should be 1."
        );

        // --- 9. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_sum() {
        // A comprehensive test for `SignalVecExt::sum`, verifying that it correctly
        // calculates the sum and re-emits on every source change, as intended.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<FinalSignalOutput<i32>>();
        let source_vec = MutableVec::from(vec![10, 20]);

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
        source_vec.write().push(5);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(35),
            "Sum after Push should be 35."
        );

        // UpdateAt: 35 -> 45 (updates 10 to 20)
        source_vec.write().set(0, 20);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(45),
            "Sum after UpdateAt should be 45."
        );

        // RemoveAt: 45 -> 25 (removes 20 at index 1)
        source_vec.write().remove(1);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(25),
            "Sum after RemoveAt should be 25."
        );

        // Pop: 25 -> 5 (removes 20 at the end)
        source_vec.write().pop(); // vec is now [20]
        source_vec.write().pop(); // vec is now []
        source_vec.write().push(5); // vec is now [5]
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(5),
            "Sum after multiple operations should be 5."
        );

        // --- 4. Test Sum-Preserving Operation (`Move`) ---
        // Add another element to make move meaningful
        source_vec.write().push(15); // vec is now [5, 15], sum is 20
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(20),
            "Sum after adding item for move test should be 20."
        );

        // Move does not change the sum, but should still re-emit the value.
        source_vec.write().move_item(0, 1); // vec is now [15, 5], sum is still 20
        source_vec.flush_into_world(app.world_mut());
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
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(0),
            "Sum after Clear should be 0."
        );

        // --- 7. Test Transition from Empty ---
        source_vec.write().push(-10);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        assert_eq!(
            get_and_clear_final_output::<i32>(app.world_mut()),
            Some(-10),
            "Sum after pushing to an empty vec should be -10."
        );

        // --- 8. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_chain() {
        // A comprehensive test for `SignalVecExt::chain`, covering all `VecDiff`
        // types on both the left and right sides, as well as edge cases like empty
        // vectors.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<String>>();

        let left_vec = MutableVec::from(vec!["L1".to_string(), "L2".to_string()]);
        let right_vec = MutableVec::from(vec!["R1".to_string(), "R2".to_string()]);

        let chained_signal = left_vec.signal_vec().chain(right_vec.signal_vec());
        let handle = chained_signal
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut current_state: Vec<String> = vec![];

        // --- 2. Initial State ---
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L1", "L2", "R1", "R2"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "Initial state was not correctly concatenated."
        );
        let left_len = || left_vec.read().len();

        // --- 3. Left-Side Mutations ---

        // Push
        left_vec.write().push("L3".to_string());
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::InsertAt {
                index: 2,
                value: "L3".to_string()
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L1", "L2", "L3", "R1", "R2"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // UpdateAt
        left_vec.write().set(0, "L1-new".to_string());
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::UpdateAt {
                index: 0,
                value: "L1-new".to_string()
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L1-new", "L2", "L3", "R1", "R2"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Move
        left_vec.write().move_item(2, 0); // "L3" to front
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::Move {
                old_index: 2,
                new_index: 0
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L3", "L1-new", "L2", "R1", "R2"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Pop
        left_vec.write().pop(); // removes "L2"
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::RemoveAt { index: 2 }]);
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L3", "L1-new", "R1", "R2"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // --- 4. Right-Side Mutations --- (left_len() is now 2)

        // InsertAt
        right_vec.write().insert(1, "R-insert".to_string());
        right_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::InsertAt {
                index: left_len() + 1,
                value: "R-insert".to_string()
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L3", "L1-new", "R1", "R-insert", "R2"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Move
        right_vec.write().move_item(0, 2); // "R1" to end
        right_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::Move {
                old_index: left_len(),
                new_index: left_len() + 2
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L3", "L1-new", "R-insert", "R2", "R1"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // --- 5. `Replace` and `Clear` ---

        // Replace Left
        left_vec.write().replace(vec!["LX".to_string()]);
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs); // apply diffs to check final state
        assert_eq!(
            current_state,
            ["LX", "R-insert", "R2", "R1"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "State after replacing left is incorrect"
        );

        // Replace Right
        right_vec.write().replace(vec!["RY".to_string(), "RZ".to_string()]);
        right_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["LX", "RY", "RZ"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "State after replacing right is incorrect"
        );

        // Clear Left
        left_vec.write().clear();
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["RY", "RZ"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "State after clearing left is incorrect"
        );
        assert_eq!(left_len(), 0);

        // Push to Left when it's empty
        left_vec.write().push("L-again".to_string());
        left_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::InsertAt {
                index: 0,
                value: "L-again".to_string()
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L-again", "RY", "RZ"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Clear Right
        right_vec.write().clear();
        right_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L-again"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "State after clearing right is incorrect"
        );
        assert_eq!(right_vec.read().len(), 0);

        // Push to Right when it's empty
        right_vec.write().push("R-again".to_string());
        right_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::Push {
                value: "R-again".to_string()
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["L-again", "R-again"].iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );

        // --- 6. Batched Diffs ---
        left_vec.write().pop(); // L-again is gone
        right_vec.write().push("R-batch".to_string());
        left_vec.flush_into_world(app.world_mut());
        right_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["R-again", "R-batch"].iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );

        // --- 7. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_intersperse() {
        // A comprehensive test for `SignalVecExt::intersperse`, covering all `VecDiff`
        // types and various edge cases.

        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<String>>();

        let separator = "sep".to_string();
        let source_vec = MutableVec::new(); // Start empty for initial tests

        let interspersed_signal = source_vec.signal_vec().intersperse(separator.clone());
        let handle = interspersed_signal
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        let mut current_state: Vec<String> = vec![];

        // --- 2. Test Initial and Edge Cases ---

        // Initial state is empty.
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert!(diffs.is_empty(), "Initial empty vec should produce no diffs.");
        assert!(current_state.is_empty());

        // Replace with empty
        source_vec.write().replace(Vec::<String>::new());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Replace { values: vec![] }]);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // Replace with one item (no separators)
        source_vec.write().replace(vec!["A".to_string()]);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::Replace {
                values: vec!["A".to_string()]
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["A".to_string()]);

        // Replace with multiple items
        source_vec.write().replace(vec!["A".to_string(), "B".to_string()]);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::Replace {
                values: vec!["A".to_string(), separator.clone(), "B".to_string()]
            }]
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["A".to_string(), separator.clone(), "B".to_string()]);

        // --- 3. Test Push & Pop ---

        // Push to multi-item vec
        source_vec.write().push("C".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["A", "sep", "B", "sep", "C"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Pop from multi-item vec
        source_vec.write().pop();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["A", "sep", "B"].iter().map(|&s| s.to_string()).collect::<Vec<_>>()
        );

        // Pop until one item left
        source_vec.write().pop();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["A".to_string()]);

        // Pop the last item
        source_vec.write().pop();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // Push to empty vec
        source_vec.write().push("Z".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec!["Z".to_string()]);

        // --- 4. Test Insert & Remove ---
        source_vec.write().replace(vec!["A".to_string(), "C".to_string()]);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<String>(app.world_mut()));
        assert_eq!(
            current_state,
            ["A", "sep", "C"].iter().map(|&s| s.to_string()).collect::<Vec<_>>()
        );

        // Insert in middle
        source_vec.write().insert(1, "B".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["A", "sep", "B", "sep", "C"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Insert at beginning
        source_vec.write().insert(0, "S".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["S", "sep", "A", "sep", "B", "sep", "C"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Insert at end (should behave like push)
        source_vec.write().insert(4, "E".to_string());
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["S", "sep", "A", "sep", "B", "sep", "C", "sep", "E"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>(),
            "InsertAt at end failed"
        );

        // Remove from middle ("A")
        source_vec.write().remove(1);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["S", "sep", "B", "sep", "C", "sep", "E"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>()
        );

        // Remove from end ("E")
        source_vec.write().remove(3);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["S", "sep", "B", "sep", "C"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>()
        );

        // --- 5. Test Update & Move ---
        // Update
        source_vec.write().set(1, "Beta".to_string()); // "B" -> "Beta"
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(
            diffs,
            vec![VecDiff::UpdateAt {
                index: 2,
                value: "Beta".to_string()
            }]
        );
        apply_diffs(&mut current_state, &diffs);
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
        source_vec.write().move_item(0, 2);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        // The exact diffs for Move can be complex; we verify the final state.
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            ["Beta", "sep", "C", "sep", "S"]
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>(),
            "Move failed"
        );

        // --- 6. Clear ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<String>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Clear]);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 7. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    /// A helper enum to represent the items in our final interspersed vector,
    /// allowing us to distinguish between original items and generated separators.
    #[derive(Clone)]
    enum ItemOrSep {
        Item(u32),
        // The separator variant holds the *unregistered* index signal.
        Sep(crate::signal::Source<Option<usize>>),
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

    /// Helper to poll the current value of a separator's index signal.
    fn poll_index(world: &mut World, handle: &SignalHandle) -> Option<usize> {
        poll_signal(world, **handle)
            .and_then(downcast_any_clone::<Option<usize>>)
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
    ///    signal, registers it ephemerally during assertions, and uses `poll_signal` to prove that
    ///    the separator's perceived index updates correctly after structural changes (`RemoveAt`,
    ///    `Move`, etc.) occur in the source vector.
    #[test]
    fn test_intersperse_with() {
        // --- 1. Setup ---
        let mut app = create_test_app();
        // Use `insert_resource` with an explicit value, as `ItemOrSep` doesn't have a Default.
        app.insert_resource(SignalVecOutput::<ItemOrSep>(Vec::new()));

        let source_vec = MutableVec::from(vec![10u32, 20, 30]);

        // CORRECTION: The closure's input parameter must be the concrete type
        // `signal::Source<Option<usize>>`, not a generic `impl Signal`. This satisfies
        // the `IntoSystem` trait bound.
        let separator_factory =
            |In(index_signal): In<crate::signal::Source<Option<usize>>>| ItemOrSep::Sep(index_signal);

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
        apply_diffs(&mut current_state, &get_and_clear_output::<ItemOrSep>(app.world_mut()));

        // CORRECTION: Use the correct `signal::Source` for the placeholder.
        // We also need to provide the type hint for the closure to resolve ambiguity.
        let placeholder_sep = ItemOrSep::Sep(SignalBuilder::from_system(|_: In<()>| None::<usize>));
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
        source_vec.write().push(40);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<ItemOrSep>(app.world_mut()));

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
        source_vec.write().remove(1); // Removes 20
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<ItemOrSep>(app.world_mut()));

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
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<ItemOrSep>(app.world_mut()));
        assert!(current_state.is_empty(), "State after Clear should be empty.");

        // --- 6. Final Cleanup ---
        handle.cleanup(app.world_mut());
        for h in handles_to_clean {
            h.cleanup(app.world_mut());
        }
        app.update(); // Run one last time to process cleanup commands.
    }

    #[test]
    fn test_sort_by() {
        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<(u32, char)>>();
        let source_vec = MutableVec::from(vec![(3, 'c'), (1, 'a'), (4, 'd')]);
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec![(1, 'a'), (3, 'c'), (4, 'd')],
            "Initial state is not correctly sorted."
        );

        // --- 3. Test Push ---
        source_vec.write().push((2, 'b'));
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
        assert_eq!(
            diffs[0],
            VecDiff::InsertAt {
                index: 1,
                value: (2, 'b')
            }
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]);

        // --- 4. Test RemoveAt ---
        source_vec.write().remove(1); // Removes (1, 'a')
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 });
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![(2, 'b'), (3, 'c'), (4, 'd')]);

        // --- 5. Test UpdateAt ---
        // Case A: Update without changing sort order.
        source_vec.write().set(0, (3, 'C')); // Was (3, 'c')
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![(2, 'b'), (3, 'C'), (4, 'd')]);

        // Case B: Update that changes sort order.
        // Source is now [(3, 'C'), (4, 'd'), (2, 'b')]. Sorted is [(2, 'b'), (3, 'C'), (4, 'd')]
        // Update (4, 'd') at source index 1 to (0, 'z').
        // The old item was at sorted index 2. The new item should be at sorted index 0.
        source_vec.write().set(1, (0, 'z'));
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec![(0, 'z'), (2, 'b'), (3, 'C')],
            "State after moving update is incorrect."
        );

        // --- 6. Test Move (in source) ---
        // Source: [(3, 'C'), (0, 'z'), (2, 'b')] -> move 0 to 2 -> [(0, 'z'), (2, 'b'), (3, 'C')]
        // Sorted output should not change at all.
        source_vec.write().move_item(0, 2);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
        assert!(diffs.is_empty(), "Moving in source should not affect sorted output.");
        assert_eq!(current_state, vec![(0, 'z'), (2, 'b'), (3, 'C')]);

        // --- 7. Test Stability (Duplicate Keys) ---
        source_vec.write().push((3, 'c'));
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(
            &mut current_state,
            &get_and_clear_output::<(u32, char)>(app.world_mut()),
        );
        assert_eq!(
            current_state,
            vec![(0, 'z'), (2, 'b'), (3, 'C'), (3, 'c')],
            "Stability with duplicate keys failed."
        );

        // --- 8. Test Clear ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<(u32, char)>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Clear]);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 9. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_sort_by_cmp() {
        // --- 1. Setup ---
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<i32>>();

        // The source vector, intentionally unsorted.
        let source_vec = MutableVec::from(vec![5, 1, 4, -2, 3]);

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
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec![-2, 1, 3, 4, 5],
            "Initial state is not correctly sorted by cmp."
        );

        // --- 3. Test Push ---
        // Push an item that should be inserted in the middle.
        source_vec.write().push(2);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Push should produce one diff.");
        assert_eq!(
            diffs[0],
            VecDiff::InsertAt { index: 2, value: 2 },
            "Pushing 2 should insert at sorted index 2."
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![-2, 1, 2, 3, 4, 5], "State after Push is incorrect.");

        // --- 4. Test RemoveAt ---
        // Remove '4' from the source vec.
        // Source is now: `[5, 1, 4, -2, 3, 2]` -> remove at index 2 (`4`)
        // Source becomes: `[5, 1, -2, 3, 2]`
        // Sorted output before was: `[-2, 1, 2, 3, 4, 5]`
        // The item '4' is at sorted index 4.
        source_vec.write().remove(2);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "RemoveAt should produce one diff.");
        assert_eq!(
            diffs[0],
            VecDiff::RemoveAt { index: 4 },
            "Removing 4 should remove at sorted index 4."
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec![-2, 1, 2, 3, 5],
            "State after RemoveAt is incorrect."
        );

        // --- 5. Test Clear ---
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<i32>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Clear]);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty(), "State after clear should be empty.");

        // --- 6. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    #[test]
    fn test_sort_by_key() {
        // --- 1. Setup ---
        let mut app = create_test_app();

        #[derive(Clone, Debug, PartialEq, Default)]
        struct DataItem {
            id: i32,
            name: &'static str,
        }

        app.init_resource::<SignalVecOutput<DataItem>>();

        let source_vec = MutableVec::from(vec![
            DataItem { id: 3, name: "C" },
            DataItem { id: 1, name: "A" },
            DataItem { id: 4, name: "D" },
        ]);

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
        apply_diffs(&mut current_state, &diffs);
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
        source_vec.write().push(DataItem { id: 2, name: "B" });
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);

        // --- 4. Test RemoveAt ---
        source_vec.write().remove(1); // Removes (1, 'A')
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<DataItem>(app.world_mut());
        assert_eq!(diffs.len(), 1, "RemoveAt should produce one diff.");
        assert_eq!(diffs[0], VecDiff::RemoveAt { index: 0 });
        apply_diffs(&mut current_state, &diffs);

        // --- 5. Test UpdateAt ---
        // Case A: Update without changing the key.
        source_vec.write().set(0, DataItem { id: 3, name: "C-new" });
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);

        // Case B: Update that changes the key and sort order.
        source_vec.write().set(1, DataItem { id: 0, name: "Z" }); // was (4, 'D')
        source_vec.flush_into_world(app.world_mut());
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
        apply_diffs(&mut current_state, &diffs);

        // --- 6. Test Move (in source) ---
        source_vec.write().move_item(0, 2);
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<DataItem>(app.world_mut());
        assert!(diffs.is_empty(), "Moving in source should not affect sorted output.");

        // --- 7. Test Stability (Duplicate Keys) ---
        source_vec.write().push(DataItem { id: 3, name: "c-dup" });
        source_vec.flush_into_world(app.world_mut());
        app.update();
        apply_diffs(&mut current_state, &get_and_clear_output::<DataItem>(app.world_mut()));
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
        source_vec.write().clear();
        source_vec.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<DataItem>(app.world_mut());
        assert_eq!(diffs, vec![VecDiff::Clear]);
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty());

        // --- 9. Cleanup ---
        handle.cleanup(app.world_mut());
    }

    /* #[test]
    fn test_flatten() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<u32>>();

        // --- Setup ---
        // Create the inner vectors that will be the items of our outer vector.
        let vec_a = MutableVec::from([10u32, 11]);
        let vec_b = MutableVec::from([20u32]);
        let vec_c = MutableVec::from([]); // Start with an empty inner vec

        // Create the outer vector, which holds SignalVecs.
        let source_of_vecs = MutableVec::from([vec_a.signal_vec(), vec_b.signal_vec(), vec_c.signal_vec()]);

        // The signal chain under test.
        let flattened_signal = source_of_vecs.signal_vec().flatten();

        let handle = flattened_signal
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        // --- 1. Initial State ---
        // The first update should replay the initial state of all inner vectors.
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Initial update should produce one Replace diff.");
        assert_eq!(
            diffs[0],
            VecDiff::Replace {
                values: vec![10, 11, 20]
            },
            "Initial state should be a Replace with combined contents."
        );

        let mut current_state = vec![10, 11, 20];

        // --- 2. Inner Vec Change: Push to `vec_a` ---
        // A push to the _first_ inner vector should insert at the correct global index.
        vec_a.write().push(12);
        vec_a.flush_into_world(app.world_mut());
        app.update();

        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Push to inner vec_a should produce one diff.");
        assert_eq!(
            diffs[0],
            VecDiff::InsertAt { index: 2, value: 12 },
            "Should insert at index 2 (end of vec_a's block)."
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![10, 11, 12, 20]);

        // --- 3. Inner Vec Change: Push to `vec_b` ---
        // A push to the _second_ inner vector should insert at an offset index.
        vec_b.write().push(21);
        vec_b.flush_into_world(app.world_mut());
        app.update();

        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Push to inner vec_b should produce one diff.");
        // Global index = len(vec_a) + new_index_in_b = 3 + 1 = 4
        assert_eq!(
            diffs[0],
            VecDiff::InsertAt { index: 4, value: 21 },
            "Should insert at index 4 (end of vec_b's block)."
        );
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(current_state, vec![10, 11, 12, 20, 21]);

        // --- 4. Outer Vec Change: Remove `vec_b` ---
        // Removing an inner vector should remove its entire block of items.
        source_of_vecs.write().remove(1); // Removes vec_b
        source_of_vecs.flush_into_world(app.world_mut());
        app.update();

        let diffs = get_and_clear_output::<u32>(app.world_mut());
        // The implementation should produce two RemoveAt diffs for vec_b's two items.
        assert_eq!(diffs.len(), 2, "Removing vec_b should produce 2 diffs.");
        apply_diffs(&mut current_state, &diffs);
        // The final state should no longer contain 20 or 21.
        assert_eq!(
            current_state,
            vec![10, 11, 12],
            "State after removing vec_b is incorrect."
        );
        // Note: vec_c is now at index 1 in the outer vec.

        // --- 5. Cleanup Verification ---
        // Updates to the removed `vec_b` should now be ignored.
        vec_b.write().push(99);
        vec_b.flush_into_world(app.world_mut());
        app.update();
        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert!(diffs.is_empty(), "Should ignore diffs from the removed vec_b.");

        // --- 6. Outer Vec Change: Push a new vector ---
        let vec_d = MutableVec::from([40u32, 41]);
        source_of_vecs.write().push(vec_d.signal_vec());
        source_of_vecs.flush_into_world(app.world_mut());
        app.update();

        let diffs = get_and_clear_output::<u32>(app.world_mut());
        // Pushing a new inner vector should add its initial items to the end.
        assert_eq!(diffs.len(), 2, "Pushing new vec_d should produce 2 diffs.");
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec![10, 11, 12, 40, 41],
            "State after pushing vec_d is incorrect."
        );

        // --- 7. Outer Vec Change: Move ---
        // Current outer vec: [vec_a, vec_c, vec_d].
        // Current state: [10, 11, 12, (empty), 40, 41].
        // Move vec_d (index 2) to the front (index 0).
        source_of_vecs.write().move_item(2, 0);
        source_of_vecs.flush_into_world(app.world_mut());
        app.update();

        let diffs = get_and_clear_output::<u32>(app.world_mut());
        apply_diffs(&mut current_state, &diffs);
        assert_eq!(
            current_state,
            vec![40, 41, 10, 11, 12],
            "State after moving vec_d to front is incorrect."
        );

        // --- 8. Outer Vec Change: Clear ---
        // Clearing the outer vector should remove all items.
        source_of_vecs.write().clear();
        source_of_vecs.flush_into_world(app.world_mut());
        app.update();

        let diffs = get_and_clear_output::<u32>(app.world_mut());
        assert_eq!(diffs.len(), 1, "Clear on outer vec should produce one diff.");
        assert_eq!(diffs[0], VecDiff::Clear, "Expected a Clear diff.");
        apply_diffs(&mut current_state, &diffs);
        assert!(current_state.is_empty(), "Final state should be empty after clear.");

        handle.cleanup(app.world_mut());
    } */

    #[test]
    fn test_mutable_vec_push_and_flush() {
        let mut app = create_test_app();
        app.init_resource::<SignalVecOutput<u32>>();
        let mutable_vec = MutableVec::new();
        let signal_handle = mutable_vec
            .signal_vec()
            .for_each(capture_signal_vec_output)
            .register(app.world_mut());

        // Initial flush (should be empty, no Replace diff)
        app.update();
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
}
