//! Integration coverage for Move-First Checkpoint 4:
//! collection `Clone` bound relaxation, purely structural `to_signal_vec()`,
//! `OutputDelivery::move_only()` defaults on ordinary collection combinators,
//! and linear (`!Clone`) `Builder` composition and terminal consumers.

extern crate alloc;

use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use bevy_platform::sync::Mutex;
use core::{
    cmp::Ordering as CmpOrdering,
    iter::Sum,
    sync::atomic::{AtomicUsize, Ordering},
};
use jonmo::{
    MapSignalMaterializer, OutputDelivery, ReplayCacheInspection, ScalarSignalMaterializer, SignalRegistrationError,
    VecSignalMaterializer, inspect_map_replay_cache, inspect_vec_replay_cache,
    prelude::*,
    signal,
    signal_map::{IntoSignalMapEither, MapDiff, MutableBTreeMap, SignalMapExt},
    signal_vec::{IntoSignalVecEither, MutableVec, SignalVecExt, VecDiff},
};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
struct Producer;

type IndexSignal = Broadcast<Option<usize>>;
type SharedLog<T> = Arc<Mutex<Vec<T>>>;
type DiffQueue<D> = Arc<Mutex<Option<Vec<D>>>>;

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<Producer>());
    app
}

struct CustomSignalVec<T: 'static>(VecSignalMaterializer<T>);

impl<T: 'static> SignalVec for CustomSignalVec<T> {
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.0
    }
}

struct CustomSignalMap<K: 'static, V: 'static>(MapSignalMaterializer<K, V>);

impl<K: 'static, V: 'static> SignalMap for CustomSignalMap<K, V> {
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.0
    }
}

/// Strictly move-only (`!Clone`, `!PartialEq`, `!Ord`) payload with drop accounting.
#[derive(Debug)]
struct NonCloneOpaque {
    id: usize,
    drops: Arc<AtomicUsize>,
}

impl NonCloneOpaque {
    fn new(id: usize, drops: &Arc<AtomicUsize>) -> Self {
        Self {
            id,
            drops: Arc::clone(drops),
        }
    }
}

impl Drop for NonCloneOpaque {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

/// Strictly move-only (`!Clone`) payload with label equality and drop accounting.
#[derive(Debug)]
struct NonClone {
    label: String,
    drops: Arc<AtomicUsize>,
}

impl NonClone {
    fn new(label: impl Into<String>, drops: &Arc<AtomicUsize>) -> Self {
        Self {
            label: label.into(),
            drops: Arc::clone(drops),
        }
    }
}

impl PartialEq for NonClone {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
    }
}

impl Eq for NonClone {}

impl Drop for NonClone {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

/// Move-only numeric wrapper supporting `for<'a> Sum<&'a Self>` without `Clone`.
#[derive(Debug)]
struct NonCloneNumber {
    value: i32,
    drops: Option<Arc<AtomicUsize>>,
}

impl NonCloneNumber {
    fn new(value: i32, drops: &Arc<AtomicUsize>) -> Self {
        Self {
            value,
            drops: Some(Arc::clone(drops)),
        }
    }
}

impl Drop for NonCloneNumber {
    fn drop(&mut self) {
        if let Some(drops) = &self.drops {
            drops.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl<'a> Sum<&'a NonCloneNumber> for NonCloneNumber {
    fn sum<I: Iterator<Item = &'a NonCloneNumber>>(iter: I) -> Self {
        Self {
            value: iter.map(|item| item.value).sum(),
            drops: None,
        }
    }
}

/// Move-only (`!Clone`) ordered key for `SignalVecExt::sort_by_key`.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NonCloneOrdKey(i32);

/// Cloneable ordered map key that counts every `.clone()` call.
#[derive(Debug)]
struct CloneCountingKey {
    key: i32,
    clones: Arc<AtomicUsize>,
}

impl CloneCountingKey {
    fn new(key: i32, clones: &Arc<AtomicUsize>) -> Self {
        Self {
            key,
            clones: Arc::clone(clones),
        }
    }
}

impl Clone for CloneCountingKey {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            key: self.key,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PartialEq for CloneCountingKey {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for CloneCountingKey {}

impl PartialOrd for CloneCountingKey {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for CloneCountingKey {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.key.cmp(&other.key)
    }
}

/// Clone-counting value probe for verifying zero unnecessary clones.
#[derive(Debug)]
struct CloneProbe {
    value: i32,
    clones: Arc<AtomicUsize>,
}

impl CloneProbe {
    fn new(value: i32, clones: &Arc<AtomicUsize>) -> Self {
        Self {
            value,
            clones: Arc::clone(clones),
        }
    }
}

impl Clone for CloneProbe {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            value: self.value,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PartialEq for CloneProbe {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for CloneProbe {}

#[test]
fn to_signal_vec_is_purely_structural_and_composes_explicit_dedupe_and_cache() {
    let mut app = test_app();

    // 1. `Vec<NonCloneOpaque>` (`!Clone`, `!PartialEq`) flows through `.to_signal_vec()` linearly
    // and preserves repeated identical-id emissions without hidden dedupe or cache bounds.
    let drops = Arc::new(AtomicUsize::new(0));
    let non_clone_queue: Arc<Mutex<Option<Vec<NonCloneOpaque>>>> = Arc::new(Mutex::new(None));
    let observed_batches: Arc<Mutex<Vec<Vec<usize>>>> = Arc::new(Mutex::new(Vec::new()));

    let claim = signal::from_system({
        let non_clone_queue = Arc::clone(&non_clone_queue);
        move |In(())| non_clone_queue.lock().unwrap().take()
    })
    .to_signal_vec()
    .for_each({
        let observed_batches = Arc::clone(&observed_batches);
        move |In(diffs): In<Vec<VecDiff<NonCloneOpaque>>>| {
            for diff in diffs {
                match diff {
                    VecDiff::Replace { values } => {
                        observed_batches
                            .lock()
                            .unwrap()
                            .push(values.into_iter().map(|item| item.id).collect());
                    }
                    other => panic!("expected VecDiff::Replace from to_signal_vec, got {other:?}"),
                }
            }
        }
    })
    .register(app.world_mut());

    *non_clone_queue.lock().unwrap() = Some(vec![NonCloneOpaque::new(1, &drops), NonCloneOpaque::new(2, &drops)]);
    app.update();
    *non_clone_queue.lock().unwrap() = Some(vec![NonCloneOpaque::new(1, &drops), NonCloneOpaque::new(2, &drops)]);
    app.update();

    assert_eq!(*observed_batches.lock().unwrap(), vec![vec![1, 2], vec![1, 2]]);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        4,
        "all 4 NonCloneOpaque items must drop immediately after linear consumption"
    );
    claim.release(app.world_mut());

    // 2. `.dedupe().to_signal_vec()` suppresses consecutive equal vectors explicitly.
    let dedupe_queue: Arc<Mutex<Option<Vec<i32>>>> = Arc::new(Mutex::new(None));
    let dedupe_batches: Arc<Mutex<Vec<Vec<i32>>>> = Arc::new(Mutex::new(Vec::new()));
    let dedupe_claim = signal::from_system({
        let dedupe_queue = Arc::clone(&dedupe_queue);
        move |In(())| dedupe_queue.lock().unwrap().take()
    })
    .dedupe()
    .to_signal_vec()
    .for_each({
        let dedupe_batches = Arc::clone(&dedupe_batches);
        move |In(diffs): In<Vec<VecDiff<i32>>>| {
            for diff in diffs {
                if let VecDiff::Replace { values } = diff {
                    dedupe_batches.lock().unwrap().push(values);
                }
            }
        }
    })
    .register(app.world_mut());

    for emission in [vec![10, 20], vec![10, 20], vec![30]] {
        *dedupe_queue.lock().unwrap() = Some(emission);
        app.update();
    }
    assert_eq!(*dedupe_batches.lock().unwrap(), vec![vec![10, 20], vec![30]]);
    dedupe_claim.release(app.world_mut());

    // 3. `.to_signal_vec().cache()` adds replay without reintroducing implicit deduplication.
    let cache_queue: Arc<Mutex<Option<Vec<i32>>>> = Arc::new(Mutex::new(None));
    let cached_vec = signal::from_system({
        let cache_queue = Arc::clone(&cache_queue);
        move |In(())| cache_queue.lock().unwrap().take()
    })
    .to_signal_vec()
    .cache()
    .broadcast();

    let early_seen: Arc<Mutex<Vec<Vec<i32>>>> = Arc::new(Mutex::new(Vec::new()));
    let early_claim = cached_vec
        .clone()
        .for_each({
            let early_seen = Arc::clone(&early_seen);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    if let VecDiff::Replace { values } = diff {
                        early_seen.lock().unwrap().push(values);
                    }
                }
            }
        })
        .register(app.world_mut());

    let probe_claim = cached_vec.clone().register(app.world_mut());
    assert_eq!(
        inspect_vec_replay_cache::<i32>(app.world(), probe_claim.entity()),
        Some(ReplayCacheInspection::Warming)
    );
    probe_claim.release(app.world_mut());

    *cache_queue.lock().unwrap() = Some(vec![7, 8]);
    app.update();
    *cache_queue.lock().unwrap() = Some(vec![7, 8]);
    app.update();
    assert_eq!(
        *early_seen.lock().unwrap(),
        vec![vec![7, 8], vec![7, 8]],
        "caching to_signal_vec must preserve repeated equal upstream emissions"
    );

    let late_seen: Arc<Mutex<Vec<Vec<i32>>>> = Arc::new(Mutex::new(Vec::new()));
    let late_claim = cached_vec
        .for_each({
            let late_seen = Arc::clone(&late_seen);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    if let VecDiff::Replace { values } = diff {
                        late_seen.lock().unwrap().push(values);
                    }
                }
            }
        })
        .register(app.world_mut());
    app.update();
    assert_eq!(
        *late_seen.lock().unwrap(),
        vec![vec![7, 8]],
        "late subscriber on cached to_signal_vec must receive the replayed vector"
    );

    early_claim.release(app.world_mut());
    late_claim.release(app.world_mut());
}

#[test]
fn linear_signal_vec_combinators_transport_non_clone_items_and_projections() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));

    // 1. `map_in` -> `map_in_ref` -> `map` -> `filter_map` -> `enumerate` -> `boxed` -> `either` ->
    //    `for_each`
    let source = MutableVec::builder().values([1, 2, 3, 4]).spawn(app.world_mut());
    let observed: SharedLog<(Option<IndexSignal>, String)> = Arc::new(Mutex::new(Vec::new()));

    let branch = core::hint::black_box(true);
    let linear_vec = if branch {
        let chain = source
            .signal_vec()
            .map_in({
                let drops = Arc::clone(&drops);
                move |n: i32| NonClone::new(format!("v{n}"), &drops)
            })
            .map_in_ref({
                let drops = Arc::clone(&drops);
                move |item: &NonClone| NonClone::new(format!("{}-ref", item.label), &drops)
            })
            .filter_map({
                let drops = Arc::clone(&drops);
                move |In(item): In<NonClone>| {
                    (!item.label.starts_with("v2")).then(|| NonClone::new(format!("{}-ok", item.label), &drops))
                }
            });
        #[cfg(feature = "tracing")]
        let chain = chain.debug();
        chain.enumerate().boxed().left_either()
    } else {
        signal::once(Vec::<(Broadcast<Option<usize>>, NonClone)>::new())
            .to_signal_vec()
            .right_either()
    };

    let vec_claim = linear_vec
        .schedule::<Last>()
        .for_each(move |In(diffs): In<Vec<VecDiff<(IndexSignal, NonClone)>>>| {
            let mut batch = Vec::new();
            for diff in diffs {
                match diff {
                    VecDiff::Replace { values } => {
                        for (index_sig, item) in values {
                            batch.push((Some(index_sig), item.label.clone()));
                        }
                    }
                    VecDiff::Push {
                        value: (index_sig, item),
                    } => {
                        batch.push((Some(index_sig), format!("push:{}", item.label)));
                    }
                    VecDiff::RemoveAt { index } => {
                        batch.push((None, format!("remove:{index}")));
                    }
                    _ => {}
                }
            }
            Some(batch)
        })
        .map({
            let observed = Arc::clone(&observed);
            move |In(batch): In<Vec<(Option<IndexSignal>, String)>>| {
                observed.lock().unwrap().extend(batch);
            }
        })
        .register(app.world_mut());

    app.update();
    {
        let guard = observed.lock().unwrap();
        let labels: Vec<&str> = guard.iter().map(|(_, label)| label.as_str()).collect();
        assert_eq!(labels, vec!["v1-ref-ok", "v3-ref-ok", "v4-ref-ok"]);
    }
    // 4 items via `map_in` + 4 via `map_in_ref` + 3 accepted via `filter_map` = 11 created and
    // dropped.
    assert_eq!(drops.load(Ordering::SeqCst), 11);

    // Verify the enumerated index signals update when removing index 0 (`1`).
    let third_index_sig = observed.lock().unwrap()[1].0.clone().expect("enumerated index signal");
    let latest_index: Arc<Mutex<Vec<Option<usize>>>> = Arc::new(Mutex::new(Vec::new()));
    let idx_claim = third_index_sig
        .dedupe()
        .map({
            let latest_index = Arc::clone(&latest_index);
            move |In(idx): In<Option<usize>>| {
                latest_index.lock().unwrap().push(idx);
            }
        })
        .register(app.world_mut());
    app.update();
    assert_eq!(*latest_index.lock().unwrap(), vec![Some(1)]);

    source.write(app.world_mut()).remove(0);
    source.write(app.world_mut()).push(5);
    app.update();
    app.update();
    assert_eq!(*latest_index.lock().unwrap(), vec![Some(1), Some(0)]);

    idx_claim.release(app.world_mut());
    vec_claim.release(app.world_mut());
    observed.lock().unwrap().clear();
    app.world_mut().flush();
    // Push(5) added 3 more NonClone instances (map_in + map_in_ref + filter_map) -> 14 total drops.
    assert_eq!(drops.load(Ordering::SeqCst), 14);

    // 2. `len`, `is_empty`, and `sum` work on move-only (`!Clone`) items!
    let len_queue: Arc<Mutex<Option<Vec<VecDiff<NonCloneOpaque>>>>> = Arc::new(Mutex::new(None));
    let len_seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let len_drops = Arc::new(AtomicUsize::new(0));
    let len_claim = CustomSignalVec(VecSignalMaterializer::new({
        let len_queue = Arc::clone(&len_queue);
        move |registrar| {
            registrar.signal_vec_entity(
                (),
                move |In(())| len_queue.lock().unwrap().take(),
                OutputDelivery::move_only(),
            )
        }
    }))
    .len()
    .map({
        let len_seen = Arc::clone(&len_seen);
        move |In(n): In<usize>| len_seen.lock().unwrap().push(n)
    })
    .register(app.world_mut());

    *len_queue.lock().unwrap() = Some(vec![
        VecDiff::Replace {
            values: vec![NonCloneOpaque::new(1, &len_drops), NonCloneOpaque::new(2, &len_drops)],
        },
        VecDiff::Push {
            value: NonCloneOpaque::new(3, &len_drops),
        },
    ]);
    app.update();
    assert_eq!(
        len_drops.load(Ordering::SeqCst),
        3,
        "len() must drop move-only items immediately without retaining them"
    );

    *len_queue.lock().unwrap() = Some(vec![
        VecDiff::InsertAt {
            index: 1,
            value: NonCloneOpaque::new(4, &len_drops),
        },
        VecDiff::UpdateAt {
            index: 0,
            value: NonCloneOpaque::new(5, &len_drops),
        },
        VecDiff::Move {
            old_index: 0,
            new_index: 2,
        },
        VecDiff::RemoveAt { index: 1 },
        VecDiff::Pop,
    ]);
    app.update();
    *len_queue.lock().unwrap() = Some(vec![VecDiff::Clear]);
    app.update();

    assert_eq!(*len_seen.lock().unwrap(), vec![3, 2, 0]);
    assert_eq!(len_drops.load(Ordering::SeqCst), 5);
    len_claim.release(app.world_mut());

    let empty_queue: Arc<Mutex<Option<Vec<NonCloneOpaque>>>> = Arc::new(Mutex::new(None));
    let empty_seen: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let empty_claim = signal::from_system({
        let empty_queue = Arc::clone(&empty_queue);
        move |In(())| empty_queue.lock().unwrap().take()
    })
    .to_signal_vec()
    .is_empty()
    .map({
        let empty_seen = Arc::clone(&empty_seen);
        move |In(empty): In<bool>| empty_seen.lock().unwrap().push(empty)
    })
    .register(app.world_mut());

    *empty_queue.lock().unwrap() = Some(vec![NonCloneOpaque::new(1, &len_drops)]);
    app.update();
    *empty_queue.lock().unwrap() = Some(Vec::new());
    app.update();
    assert_eq!(*empty_seen.lock().unwrap(), vec![false, true]);
    empty_claim.release(app.world_mut());

    // 3. `sum()` on `NonCloneNumber` (`!Clone`) and `sort_by_key` with `NonCloneOrdKey` (`!Clone`).
    let sum_drops = Arc::new(AtomicUsize::new(0));
    let sum_source = MutableVec::builder().values([30, 10, 20]).spawn(app.world_mut());
    let sums: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let sum_claim = sum_source
        .signal_vec()
        .sort_by_key(|In(v): In<i32>| NonCloneOrdKey(v))
        .map_in({
            let sum_drops = Arc::clone(&sum_drops);
            move |v: i32| NonCloneNumber::new(v, &sum_drops)
        })
        .sum()
        .map({
            let sums = Arc::clone(&sums);
            move |In(total): In<NonCloneNumber>| sums.lock().unwrap().push(total.value)
        })
        .register(app.world_mut());

    app.update();
    sum_source.write(app.world_mut()).push(5); // [30, 10, 20, 5] -> sorted [5, 10, 20, 30]
    app.update();
    sum_source.write(app.world_mut()).remove(1); // removes 10 -> [30, 20, 5]
    app.update();
    sum_source.write(app.world_mut()).insert(1, 15); // [30, 15, 20, 5]
    sum_source.write(app.world_mut()).set(0, 12); // [12, 15, 20, 5]
    sum_source.write(app.world_mut()).move_item(0, 2); // [15, 20, 12, 5]
    sum_source.write(app.world_mut()).pop(); // removes 5 -> [15, 20, 12]
    app.update();
    sum_source.write(app.world_mut()).clear(); // []
    app.update();

    assert_eq!(*sums.lock().unwrap(), vec![60, 65, 55, 47, 0]);
    sum_claim.release(app.world_mut());
    app.world_mut().flush();
    assert_eq!(
        sum_drops.load(Ordering::SeqCst),
        6,
        "all 6 mapped NonCloneNumber items stored in sum's Local must have dropped cleanly"
    );
}

#[test]
fn ordinary_collection_combinators_default_to_move_only_and_reject_fanout_without_broadcast() {
    fn assert_vec_move_only_fanout<S>(world: &mut World, signal_vec: S, label: &str)
    where
        S: SignalVec<Item = i32>,
    {
        let shared = Box::new(signal_vec).into_signal_vec_materializer();
        let left = ScalarSignalMaterializer::new({
            let mut inner = Some(shared);
            move |registrar| {
                // Register via a shared entity handle on first call by materializing two consumers
                // inside one candidate preflight!
                let vec_entity = registrar.materialize_signal_vec(inner.take().expect("one-shot"))?;
                let a = registrar.signal_entity(
                    vec_entity,
                    |In(diffs): In<Vec<VecDiff<i32>>>| diffs.len(),
                    OutputDelivery::move_only(),
                )?;
                let b = registrar.signal_entity(
                    vec_entity,
                    |In(diffs): In<Vec<VecDiff<i32>>>| diffs.len(),
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_entity([a, b], |In(n): In<usize>| n, OutputDelivery::move_only())
            }
        });
        let err = left
            .try_register(world)
            .expect_err(&format!("{label} must reject implicit multi-consumer fan-out"));
        assert_eq!(err, SignalRegistrationError::MoveOnlyFanout, "{label}");
    }

    let mut app = test_app();
    let vec_src = MutableVec::builder().values([1, 2, 3]).spawn(app.world_mut());
    let vec_src_b = MutableVec::builder().values([4, 5]).spawn(app.world_mut());
    let vec_anchor = vec_src.signal_vec().register(app.world_mut());
    let vec_anchor_b = vec_src_b.signal_vec().register(app.world_mut());

    assert_vec_move_only_fanout(
        app.world_mut(),
        signal::once(vec![1, 2]).to_signal_vec(),
        "to_signal_vec",
    );
    assert_vec_move_only_fanout(app.world_mut(), vec_src.signal_vec().map_in(|x| x + 1), "map_in");
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().map_in_ref(|x: &i32| *x + 1),
        "map_in_ref",
    );
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().filter(|In(x): In<i32>| x > 1),
        "filter",
    );
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().filter_map(|In(x): In<i32>| Some(x)),
        "filter_map",
    );
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().enumerate().map_in(|(_idx, value)| value),
        "enumerate -> map_in",
    );
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().chain(vec_src_b.signal_vec()),
        "chain",
    );
    assert_vec_move_only_fanout(app.world_mut(), vec_src.signal_vec().intersperse(0), "intersperse");
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src
            .signal_vec()
            .intersperse_with(|_: In<Broadcast<Option<usize>>>| 0),
        "intersperse_with",
    );
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().sort_by(|In((a, b)): In<(i32, i32)>| a.cmp(&b)),
        "sort_by",
    );
    assert_vec_move_only_fanout(app.world_mut(), vec_src.signal_vec().sort_by_cmp(), "sort_by_cmp");
    assert_vec_move_only_fanout(
        app.world_mut(),
        vec_src.signal_vec().sort_by_key(|In(x): In<i32>| -x),
        "sort_by_key",
    );
    #[cfg(feature = "tracing")]
    assert_vec_move_only_fanout(app.world_mut(), vec_src.signal_vec().debug(), "SignalVecExt::debug");

    // Verify `enumerate` itself is move-only as well.
    let enum_fanout = ScalarSignalMaterializer::new({
        let enumerated = vec_src.signal_vec().enumerate();
        move |registrar| {
            let e = registrar.materialize_signal_vec(Box::new(enumerated).into_signal_vec_materializer())?;
            let a = registrar.signal_entity(
                e,
                |In(d): In<Vec<VecDiff<(IndexSignal, i32)>>>| d.len(),
                OutputDelivery::move_only(),
            )?;
            let b = registrar.signal_entity(
                e,
                |In(d): In<Vec<VecDiff<(IndexSignal, i32)>>>| d.len(),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity([a, b], |In(n): In<usize>| n, OutputDelivery::move_only())
        }
    });
    assert_eq!(
        enum_fanout.try_register(app.world_mut()).unwrap_err(),
        SignalRegistrationError::MoveOnlyFanout
    );

    // Verify `SignalVecExt::for_each` and `SignalMapExt::for_each` and `SignalMapExt::map_value`
    // are also `OutputDelivery::move_only()`, and `map_value` does NOT install a replay cache.
    let map_src = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
    let map_anchor = map_src.signal_map().register(app.world_mut());
    let map_val_claim = map_src
        .signal_map()
        .map_value(|In(v): In<i32>| v * 2)
        .register(app.world_mut());
    assert_eq!(
        inspect_map_replay_cache::<i32, i32>(app.world(), map_val_claim.entity()),
        None,
        "SignalMapExt::map_value must be linear without an implicit replay cache"
    );
    map_val_claim.release(app.world_mut());

    for (map_label, map_candidate) in [
        (
            "SignalMapExt::map_value",
            Box::new(map_src.signal_map().map_value(|In(v): In<i32>| v * 2)).into_signal_map_materializer(),
        ),
        #[cfg(feature = "tracing")]
        (
            "SignalMapExt::debug",
            Box::new(map_src.signal_map().debug()).into_signal_map_materializer(),
        ),
    ] {
        let map_fanout = ScalarSignalMaterializer::new(move |registrar| {
            let m = registrar.materialize_signal_map(map_candidate)?;
            let a = registrar.signal_entity(
                m,
                |In(d): In<Vec<MapDiff<i32, i32>>>| d.len(),
                OutputDelivery::move_only(),
            )?;
            let b = registrar.signal_entity(
                m,
                |In(d): In<Vec<MapDiff<i32, i32>>>| d.len(),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity([a, b], |In(n): In<usize>| n, OutputDelivery::move_only())
        });
        assert_eq!(
            map_fanout.try_register(app.world_mut()).unwrap_err(),
            SignalRegistrationError::MoveOnlyFanout,
            "{map_label} must default to move-only delivery"
        );
    }

    // Verify scalar collection projections (`for_each`, `len`, `is_empty`, `sum`, `key`) are
    // move-only too.
    for (label, scalar) in [
        (
            "SignalVecExt::for_each",
            Box::new(vec_src.signal_vec().for_each(|In(d): In<Vec<VecDiff<i32>>>| d.len())).into_signal_materializer(),
        ),
        (
            "SignalVecExt::len",
            Box::new(vec_src.signal_vec().len()).into_signal_materializer(),
        ),
        (
            "SignalVecExt::is_empty",
            Box::new(vec_src.signal_vec().is_empty().map_in(usize::from)).into_signal_materializer(),
        ),
        (
            "SignalVecExt::sum",
            Box::new(vec_src.signal_vec().map_in(|x: i32| x as usize).sum()).into_signal_materializer(),
        ),
        (
            "SignalMapExt::for_each",
            Box::new(
                map_src
                    .signal_map()
                    .for_each(|In(d): In<Vec<MapDiff<i32, i32>>>| d.len()),
            )
            .into_signal_materializer(),
        ),
    ] {
        let fanout = ScalarSignalMaterializer::new(move |registrar| {
            let s = registrar.materialize_signal(scalar)?;
            let a = registrar.signal_entity(s, |In(n): In<usize>| n, OutputDelivery::move_only())?;
            let b = registrar.signal_entity(s, |In(n): In<usize>| n, OutputDelivery::move_only())?;
            registrar.signal_entity([a, b], |In(n): In<usize>| n, OutputDelivery::move_only())
        });
        assert_eq!(
            fanout.try_register(app.world_mut()).unwrap_err(),
            SignalRegistrationError::MoveOnlyFanout,
            "{label} must default to move-only delivery"
        );
    }

    let key_fanout = ScalarSignalMaterializer::new({
        let key_sig = map_src.signal_map().key(1);
        move |registrar| {
            let s = registrar.materialize_signal(Box::new(key_sig).into_signal_materializer())?;
            let a = registrar.signal_entity(
                s,
                |In(v): In<Option<i32>>| usize::from(v.is_some()),
                OutputDelivery::move_only(),
            )?;
            let b = registrar.signal_entity(
                s,
                |In(v): In<Option<i32>>| usize::from(v.is_some()),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity([a, b], |In(n): In<usize>| n, OutputDelivery::move_only())
        }
    });
    assert_eq!(
        key_fanout.try_register(app.world_mut()).unwrap_err(),
        SignalRegistrationError::MoveOnlyFanout,
        "SignalMapExt::key must default to move-only delivery"
    );

    map_anchor.release(app.world_mut());
    vec_anchor.release(app.world_mut());
    vec_anchor_b.release(app.world_mut());
}

#[test]
fn linear_signal_map_transports_non_clone_non_ord_keys_and_values_without_map_value_key_clones() {
    let mut app = test_app();
    let key_drops = Arc::new(AtomicUsize::new(0));
    let val_drops = Arc::new(AtomicUsize::new(0));

    // 1. Custom linear `SignalMap<Key = NonCloneOpaque, Value = NonCloneOpaque>` (`!Clone`, `!Ord`,
    //    `!PartialEq`)
    // flows through `map_value`, `debug`, `schedule`, `boxed`, `either`, and `for_each`.
    let pending: DiffQueue<MapDiff<NonCloneOpaque, NonCloneOpaque>> = Arc::new(Mutex::new(None));
    let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let custom_map = CustomSignalMap(MapSignalMaterializer::new({
        let pending = Arc::clone(&pending);
        move |registrar| {
            registrar.signal_map_entity(
                (),
                move |In(())| pending.lock().unwrap().take(),
                OutputDelivery::move_only(),
            )
        }
    }));

    let mapped = custom_map.map_value({
        let val_drops = Arc::clone(&val_drops);
        move |In(val): In<NonCloneOpaque>| NonCloneOpaque::new(val.id * 10, &val_drops)
    });
    #[cfg(feature = "tracing")]
    let mapped = mapped.debug();
    let use_left = core::hint::black_box(true);
    let mapped = if use_left {
        mapped.schedule::<Last>().boxed().left_either()
    } else {
        CustomSignalMap(MapSignalMaterializer::new(|registrar| {
            registrar.signal_map_entity(
                (),
                |In(())| None::<Vec<MapDiff<NonCloneOpaque, NonCloneOpaque>>>,
                OutputDelivery::move_only(),
            )
        }))
        .right_either()
    };

    let claim = mapped
        .for_each({
            let val_drops = Arc::clone(&val_drops);
            move |In(diffs): In<Vec<MapDiff<NonCloneOpaque, NonCloneOpaque>>>| {
                let mut tags = Vec::new();
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            for (k, v) in entries {
                                tags.push(format!("R({}:{})", k.id, v.id));
                            }
                        }
                        MapDiff::Insert { key, value } => tags.push(format!("I({}:{})", key.id, value.id)),
                        MapDiff::Update { key, value } => tags.push(format!("U({}:{})", key.id, value.id)),
                        MapDiff::Remove { key } => tags.push(format!("D({})", key.id)),
                        MapDiff::Clear => tags.push("C".into()),
                    }
                }
                (tags, NonCloneOpaque::new(999, &val_drops))
            }
        })
        .map({
            let observed = Arc::clone(&observed);
            move |In((tags, _token)): In<(Vec<String>, NonCloneOpaque)>| {
                observed.lock().unwrap().extend(tags);
            }
        })
        .register(app.world_mut());

    *pending.lock().unwrap() = Some(vec![
        MapDiff::Replace {
            entries: vec![(NonCloneOpaque::new(1, &key_drops), NonCloneOpaque::new(2, &val_drops))],
        },
        MapDiff::Insert {
            key: NonCloneOpaque::new(3, &key_drops),
            value: NonCloneOpaque::new(4, &val_drops),
        },
        MapDiff::Update {
            key: NonCloneOpaque::new(1, &key_drops),
            value: NonCloneOpaque::new(5, &val_drops),
        },
        MapDiff::Remove {
            key: NonCloneOpaque::new(3, &key_drops),
        },
        MapDiff::Clear,
    ]);
    app.update();

    assert_eq!(
        *observed.lock().unwrap(),
        vec!["R(1:20)", "I(3:40)", "U(1:50)", "D(3)", "C"]
    );
    assert_eq!(key_drops.load(Ordering::SeqCst), 4);
    // 3 input values + 3 mapped values + 1 token = 7 drops.
    assert_eq!(val_drops.load(Ordering::SeqCst), 7);
    claim.release(app.world_mut());

    // 2. `MutableBTreeMap<CloneCountingKey, i32>` -> `.map_value(...)` moves unchanged keys through
    // `MapDiff::map_value` without cloning keys inside `map_value`!
    let key_clones = Arc::new(AtomicUsize::new(0));
    let btree_map = MutableBTreeMap::builder()
        .values([(CloneCountingKey::new(1, &key_clones), 10)])
        .spawn(app.world_mut());
    let seen_entries: Arc<Mutex<Vec<(i32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let btree_claim = btree_map
        .signal_map()
        .map_value({
            let val_drops = Arc::clone(&val_drops);
            move |In(v): In<i32>| NonCloneOpaque::new(v as usize + 1, &val_drops)
        })
        .for_each({
            let seen_entries = Arc::clone(&seen_entries);
            move |In(diffs): In<Vec<MapDiff<CloneCountingKey, NonCloneOpaque>>>| {
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => {
                            for (k, v) in entries {
                                seen_entries.lock().unwrap().push((k.key, v.id));
                            }
                        }
                        MapDiff::Update { key, value } => {
                            seen_entries.lock().unwrap().push((key.key, value.id));
                        }
                        _ => {}
                    }
                }
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(*seen_entries.lock().unwrap(), vec![(1, 11)]);

    btree_map
        .write(app.world_mut())
        .insert(CloneCountingKey::new(1, &key_clones), 20);
    // Reset clone counter AFTER `MutableBTreeMap::write` queues its diff so we measure strictly
    // runtime transport through `signal_map() -> map_value() -> for_each()`.
    key_clones.store(0, Ordering::SeqCst);
    app.update();
    assert_eq!(*seen_entries.lock().unwrap(), vec![(1, 11), (1, 21)]);
    assert_eq!(
        key_clones.load(Ordering::SeqCst),
        0,
        "neither MutableBTreeMap's replay cache nor map_value clones the key on MapDiff::Update"
    );
    btree_claim.release(app.world_mut());

    // 3. `SignalMapExt::key` accepts `!Clone + PartialEq` keys and performs 0 value clones on
    // unrelated key updates.
    let val_clones = Arc::new(AtomicUsize::new(0));
    let key_lookup_queue: DiffQueue<MapDiff<NonClone, CloneProbe>> = Arc::new(Mutex::new(None));
    let key_lookup_seen: Arc<Mutex<Vec<Option<i32>>>> = Arc::new(Mutex::new(Vec::new()));
    let lookup_drops = Arc::new(AtomicUsize::new(0));

    let lookup_claim = CustomSignalMap(MapSignalMaterializer::new({
        let key_lookup_queue = Arc::clone(&key_lookup_queue);
        move |registrar| {
            registrar.signal_map_entity(
                (),
                move |In(())| key_lookup_queue.lock().unwrap().take(),
                OutputDelivery::move_only(),
            )
        }
    }))
    .key(NonClone::new("target", &lookup_drops))
    .map({
        let key_lookup_seen = Arc::clone(&key_lookup_seen);
        move |In(found): In<Option<CloneProbe>>| {
            key_lookup_seen.lock().unwrap().push(found.map(|p| p.value));
        }
    })
    .register(app.world_mut());

    *key_lookup_queue.lock().unwrap() = Some(vec![MapDiff::Insert {
        key: NonClone::new("target", &lookup_drops),
        value: CloneProbe::new(42, &val_clones),
    }]);
    app.update();
    assert_eq!(*key_lookup_seen.lock().unwrap(), vec![Some(42)]);
    assert_eq!(
        val_clones.load(Ordering::SeqCst),
        2,
        "initial Warming -> Ready publish clones once into the replay cache and once for Pending snapshot delivery"
    );

    val_clones.store(0, Ordering::SeqCst);
    *key_lookup_queue.lock().unwrap() = Some(vec![MapDiff::Update {
        key: NonClone::new("target", &lookup_drops),
        value: CloneProbe::new(43, &val_clones),
    }]);
    app.update();
    assert_eq!(*key_lookup_seen.lock().unwrap(), vec![Some(42), Some(43)]);
    assert_eq!(
        val_clones.load(Ordering::SeqCst),
        1,
        "once Published, key() clones once into its replay cache and moves the accepted output without extra clones"
    );

    val_clones.store(0, Ordering::SeqCst);
    *key_lookup_queue.lock().unwrap() = Some(vec![MapDiff::Insert {
        key: NonClone::new("unrelated", &lookup_drops),
        value: CloneProbe::new(99, &val_clones),
    }]);
    app.update();
    assert_eq!(*key_lookup_seen.lock().unwrap(), vec![Some(42), Some(43)]);
    assert_eq!(
        val_clones.load(Ordering::SeqCst),
        0,
        "unrelated key updates must perform 0 clones of the retained target value"
    );
    lookup_claim.release(app.world_mut());
    app.world_mut().flush();
    // 1 captured lookup target key + 3 diff keys = 4 NonClone keys dropped cleanly.
    assert_eq!(lookup_drops.load(Ordering::SeqCst), 4);
}

#[cfg(feature = "builder")]
mod builder_linearity {
    use super::*;
    use bevy_ecs::change_detection::Mut;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct Tag(String);

    #[derive(Component, Debug, PartialEq, Eq)]
    struct HeaderTag(String);

    #[derive(Component, Debug, PartialEq, Eq)]
    struct FooterTag(String);

    #[derive(Component)]
    struct NonCloneComp(NonClone);

    #[derive(Component)]
    struct Accumulator(Vec<String>);

    #[derive(Resource, Clone, PartialEq, Eq, Default)]
    struct DashboardState {
        title: Option<&'static str>,
        items: Vec<i32>,
    }

    #[test]
    fn builder_child_signal_and_children_signal_vec_work_with_linear_builders_and_upstream_broadcast() {
        let mut app = test_app();
        app.init_resource::<DashboardState>();
        let despawned_tags: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let state = signal::from_resource::<DashboardState>().dedupe().broadcast();

        let make_badged_child = {
            let despawned_tags = Arc::clone(&despawned_tags);
            move |prefix: &'static str, text: String| {
                let tag = format!("{prefix}:{text}");
                let on_despawn_tag = tag.clone();
                let despawned_tags = Arc::clone(&despawned_tags);
                jonmo::Builder::new()
                    .insert(Tag(tag))
                    .on_despawn(move |_world, _entity| {
                        despawned_tags.lock().unwrap().push(on_despawn_tag);
                    })
            }
        };

        let vec_items = MutableVec::builder().values([10, 20]).spawn(app.world_mut());

        // Root spawns two independent reactive child branches from the broadcast upstream `state`,
        // plus an `enumerate -> map_in` `children_signal_vec` and a `to_signal_vec`
        // `children_signal_vec`.
        let root = jonmo::Builder::new()
            .child_signal(state.clone().map_in({
                let make_badged_child = make_badged_child.clone();
                move |s: DashboardState| {
                    s.title
                        .map(|title| make_badged_child("header", title.into()).insert(HeaderTag(title.into())))
                }
            }))
            .children_signal_vec(vec_items.signal_vec().enumerate().map_in({
                let make_badged_child = make_badged_child.clone();
                move |(index_sig, value): (Broadcast<Option<usize>>, i32)| {
                    make_badged_child("vec", format!("{value}")).component_signal(
                        index_sig
                            .dedupe()
                            .map_in(move |idx: Option<usize>| idx.map(|i| FooterTag(format!("idx={i}:val={value}")))),
                    )
                }
            }))
            .children_signal_vec(
                state
                    .map_in({
                        let make_badged_child = make_badged_child.clone();
                        move |s: DashboardState| {
                            s.items
                                .into_iter()
                                .map(|item| make_badged_child("snap", format!("{item}")))
                                .collect::<Vec<jonmo::Builder>>()
                        }
                    })
                    .to_signal_vec(),
            )
            .spawn(app.world_mut());

        let read_child_tags = |world: &World| -> Vec<String> {
            let Some(children) = world.get::<Children>(root) else {
                return Vec::new();
            };
            children
                .iter()
                .filter_map(|child| world.get::<Tag>(child).map(|t| t.0.clone()))
                .collect()
        };

        app.update();
        assert_eq!(read_child_tags(app.world()), vec!["vec:10", "vec:20"]);

        // Activate header + snapshot children via the broadcast upstream resource!
        *app.world_mut().resource_mut::<DashboardState>() = DashboardState {
            title: Some("alpha"),
            items: vec![1, 2],
        };
        vec_items.write(app.world_mut()).insert(0, 5);
        app.update();
        app.update();

        assert_eq!(
            read_child_tags(app.world()),
            vec!["header:alpha", "vec:5", "vec:10", "vec:20", "snap:1", "snap:2"]
        );

        // Check that `enumerate` index signals updated all three `vec:*` children accurately.
        let children: Vec<Entity> = app.world().get::<Children>(root).unwrap().iter().collect();
        assert_eq!(
            app.world().get::<FooterTag>(children[1]),
            Some(&FooterTag("idx=0:val=5".into()))
        );
        assert_eq!(
            app.world().get::<FooterTag>(children[2]),
            Some(&FooterTag("idx=1:val=10".into()))
        );
        assert_eq!(
            app.world().get::<FooterTag>(children[3]),
            Some(&FooterTag("idx=2:val=20".into()))
        );

        // Replace header (`Some("alpha")` -> `Some("beta")`), then remove (`None`).
        *app.world_mut().resource_mut::<DashboardState>() = DashboardState {
            title: Some("beta"),
            items: vec![3],
        };
        app.update();
        assert_eq!(
            read_child_tags(app.world()),
            vec!["header:beta", "vec:5", "vec:10", "vec:20", "snap:3"]
        );
        assert!(despawned_tags.lock().unwrap().contains(&String::from("header:alpha")));

        app.world_mut().entity_mut(root).despawn();
        app.update();
        assert!(app.world().get_entity(root).is_err());
    }

    #[test]
    fn builder_terminal_consumers_accept_non_clone_payloads_and_components_with_exact_drop_accounting() {
        let mut app = test_app();
        let drops = Arc::new(AtomicUsize::new(0));

        let on_signal_queue: Arc<Mutex<Option<NonClone>>> = Arc::new(Mutex::new(None));
        let on_entity_queue: Arc<Mutex<Option<NonClone>>> = Arc::new(Mutex::new(None));
        let on_comp_present_queue: Arc<Mutex<Option<NonClone>>> = Arc::new(Mutex::new(None));
        let on_comp_missing_queue: Arc<Mutex<Option<NonClone>>> = Arc::new(Mutex::new(None));
        let comp_signal_queue: Arc<Mutex<Option<Option<NonCloneComp>>>> = Arc::new(Mutex::new(None));
        let task_vec_queue: Arc<Mutex<Option<Vec<NonCloneOpaque>>>> = Arc::new(Mutex::new(None));
        let task_map_queue: DiffQueue<MapDiff<NonCloneOpaque, NonCloneOpaque>> = Arc::new(Mutex::new(None));

        let recorded: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let task_drops = Arc::new(AtomicUsize::new(0));

        let vec_task = signal::from_system({
            let task_vec_queue = Arc::clone(&task_vec_queue);
            move |In(())| task_vec_queue.lock().unwrap().take()
        })
        .to_signal_vec()
        .task();

        let map_task = CustomSignalMap(MapSignalMaterializer::new({
            let task_map_queue = Arc::clone(&task_map_queue);
            move |registrar| {
                registrar.signal_map_entity(
                    (),
                    move |In(())| task_map_queue.lock().unwrap().take(),
                    OutputDelivery::move_only(),
                )
            }
        }))
        .task();

        let entity = jonmo::Builder::new()
            .insert(Accumulator(Vec::new()))
            .hold_tasks([vec_task, map_task])
            .on_signal(
                signal::from_system({
                    let on_signal_queue = Arc::clone(&on_signal_queue);
                    move |In(())| on_signal_queue.lock().unwrap().take()
                }),
                {
                    let recorded = Arc::clone(&recorded);
                    move |In((e, payload)): In<(Entity, NonClone)>| {
                        recorded
                            .lock()
                            .unwrap()
                            .push(format!("on_signal:{}:{e}", payload.label));
                    }
                },
            )
            .on_signal_with_entity(
                signal::from_system({
                    let on_entity_queue = Arc::clone(&on_entity_queue);
                    move |In(())| on_entity_queue.lock().unwrap().take()
                }),
                move |mut entity_mut: EntityWorldMut, payload: NonClone| {
                    entity_mut
                        .get_mut::<Accumulator>()
                        .unwrap()
                        .0
                        .push(format!("entity:{}", payload.label));
                },
            )
            .on_signal_with_component(
                signal::from_system({
                    let on_comp_present_queue = Arc::clone(&on_comp_present_queue);
                    move |In(())| on_comp_present_queue.lock().unwrap().take()
                }),
                move |mut acc: Mut<Accumulator>, payload: NonClone| {
                    acc.0.push(format!("comp:{}", payload.label));
                },
            )
            .component_signal::<NonCloneComp>(signal::from_system({
                let comp_signal_queue = Arc::clone(&comp_signal_queue);
                move |In(())| comp_signal_queue.lock().unwrap().take()
            }))
            .spawn(app.world_mut());

        // Separate entity without `Accumulator` to verify `on_signal_with_component` drops
        // unconsumed `NonClone`.
        let missing_comp_entity = jonmo::Builder::new()
            .on_signal_with_component(
                signal::from_system({
                    let on_comp_missing_queue = Arc::clone(&on_comp_missing_queue);
                    move |In(())| on_comp_missing_queue.lock().unwrap().take()
                }),
                move |mut acc: Mut<Accumulator>, payload: NonClone| {
                    acc.0.push(payload.label.clone());
                },
            )
            .spawn(app.world_mut());

        *on_signal_queue.lock().unwrap() = Some(NonClone::new("s1", &drops));
        *on_entity_queue.lock().unwrap() = Some(NonClone::new("e1", &drops));
        *on_comp_present_queue.lock().unwrap() = Some(NonClone::new("c1", &drops));
        *on_comp_missing_queue.lock().unwrap() = Some(NonClone::new("missing", &drops));
        *comp_signal_queue.lock().unwrap() = Some(Some(NonCloneComp(NonClone::new("comp_v1", &drops))));
        *task_vec_queue.lock().unwrap() = Some(vec![NonCloneOpaque::new(1, &task_drops)]);
        *task_map_queue.lock().unwrap() = Some(vec![MapDiff::Insert {
            key: NonCloneOpaque::new(2, &task_drops),
            value: NonCloneOpaque::new(3, &task_drops),
        }]);

        app.update();

        assert_eq!(recorded.lock().unwrap().as_slice(), &[format!("on_signal:s1:{entity}")]);
        let mut acc_entries = app.world().get::<Accumulator>(entity).unwrap().0.clone();
        acc_entries.sort();
        assert_eq!(acc_entries, vec![String::from("comp:c1"), String::from("entity:e1")]);
        assert_eq!(app.world().get::<NonCloneComp>(entity).unwrap().0.label, "comp_v1");
        // The 4 `on_signal*` payloads dropped; `comp_v1` is currently stored on `entity`!
        assert_eq!(drops.load(Ordering::SeqCst), 4);
        assert_eq!(task_drops.load(Ordering::SeqCst), 3);

        // Replace `comp_v1` with `comp_v2` -> `comp_v1` must drop!
        *comp_signal_queue.lock().unwrap() = Some(Some(NonCloneComp(NonClone::new("comp_v2", &drops))));
        app.update();
        assert_eq!(app.world().get::<NonCloneComp>(entity).unwrap().0.label, "comp_v2");
        assert_eq!(drops.load(Ordering::SeqCst), 5);

        // Remove `comp_v2` via `Some(None)` -> `comp_v2` must drop!
        *comp_signal_queue.lock().unwrap() = Some(None);
        app.update();
        assert!(app.world().get::<NonCloneComp>(entity).is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 6);

        // Reinsert `comp_v3`, then despawn `entity` -> `comp_v3` drops on despawn!
        *comp_signal_queue.lock().unwrap() = Some(Some(NonCloneComp(NonClone::new("comp_v3", &drops))));
        app.update();
        assert_eq!(drops.load(Ordering::SeqCst), 6);

        app.world_mut().entity_mut(entity).despawn();
        app.world_mut().entity_mut(missing_comp_entity).despawn();
        app.update();
        assert_eq!(drops.load(Ordering::SeqCst), 7);
    }
}
