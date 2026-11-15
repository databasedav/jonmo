//! Integration coverage for Move-First Checkpoint 5:
//! dynamic move-first parity across built-in and custom `DynamicSignalOperatorSpec`
//! operators (ordinary non-replayable `*_output` vs. managed `replayable_*_output`),
//! `!Clone` queue/output payloads, linear inner chains, dynamic `MoveOnlyFanout`
//! cohort discard, `Builder` dynamic composition, and opt-in imperative evaluation.

#![allow(missing_docs)]

extern crate alloc;

use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    format,
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use bevy_platform::sync::Mutex;
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand, KeyedSlotCommand,
    MapSignalMaterializer, OutputDelivery, ReplayCacheInspection, ReplayInitialization, SameSignalIdentity,
    ScalarSignalMaterializer, ScalarSlotCommand, SignalRegistrationError, VecSignalMaterializer,
    inspect_map_replay_cache, inspect_scalar_replay_cache, inspect_vec_replay_cache,
    prelude::*,
    signal::{self, BoxedReplayableSignal, BoxedSignal},
    signal_map::{MapDiff, MutableBTreeMap, SignalMapExt},
    signal_vec::{MutableVec, SignalVecExt, VecDiff},
};

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app
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

/// Clone-counting payload used to verify exact `N - 1` broadcast fan-out vs. replay clones.
#[derive(Debug)]
struct CloneCounter {
    id: i32,
    clones: Arc<AtomicUsize>,
}

impl CloneCounter {
    fn new(id: i32, clones: &Arc<AtomicUsize>) -> Self {
        Self {
            id,
            clones: Arc::clone(clones),
        }
    }
}

impl Clone for CloneCounter {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            id: self.id,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PartialEq for CloneCounter {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CloneCounter {}

struct CustomScalar<T: 'static>(ScalarSignalMaterializer<T>);

impl<T: 'static> Signal for CustomScalar<T> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.0
    }
}

struct CustomVec<T: 'static>(VecSignalMaterializer<T>);

impl<T: 'static> SignalVec for CustomVec<T> {
    type Item = T;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        self.0
    }
}

struct CustomMap<K: 'static, V: 'static>(MapSignalMaterializer<K, V>);

impl<K: 'static, V: 'static> SignalMap for CustomMap<K, V> {
    type Key = K;
    type Value = V;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        self.0
    }
}

fn map_vec_diff<T, U>(diff: VecDiff<T>, mut f: impl FnMut(T) -> U) -> VecDiff<U> {
    match diff {
        VecDiff::Replace { values } => VecDiff::Replace {
            values: values.into_iter().map(f).collect(),
        },
        VecDiff::InsertAt { index, value } => VecDiff::InsertAt { index, value: f(value) },
        VecDiff::UpdateAt { index, value } => VecDiff::UpdateAt { index, value: f(value) },
        VecDiff::RemoveAt { index } => VecDiff::RemoveAt { index },
        VecDiff::Move { old_index, new_index } => VecDiff::Move { old_index, new_index },
        VecDiff::Push { value } => VecDiff::Push { value: f(value) },
        VecDiff::Pop => VecDiff::Pop,
        VecDiff::Clear => VecDiff::Clear,
    }
}

fn apply_vec_diffs<T>(model: &mut Vec<T>, diffs: Vec<VecDiff<T>>) {
    for diff in diffs {
        diff.apply_to_vec(model);
    }
}

fn apply_map_diffs<K: Ord, V>(model: &mut BTreeMap<K, V>, diffs: Vec<MapDiff<K, V>>) {
    for diff in diffs {
        match diff {
            MapDiff::Replace { entries } => {
                model.clear();
                model.extend(entries);
            }
            MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                model.insert(key, value);
            }
            MapDiff::Remove { key } => {
                model.remove(&key);
            }
            MapDiff::Clear => {
                model.clear();
            }
        }
    }
}

// ============================================================================
// 1. Built-in dynamic operators with linear inner chains and `!Clone` payloads
// ============================================================================

#[test]
fn builtin_dynamic_operators_support_linear_inner_chains_and_non_clone_projections() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));

    // 1a. `signal::once(move_only_boxed_replayable).flatten()` consumes the linear boxed inner
    //     replayable signal without requiring `S: Clone` and projects downstream into `NonClone`.
    let inner_boxed: BoxedReplayableSignal<i32> = signal::once(42).cache().boxed_replayable();
    let flattened_seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let flatten_claim = signal::once(inner_boxed)
        .flatten()
        .map_in({
            let drops = Arc::clone(&drops);
            move |value: i32| NonClone::new(format!("flat:{value}"), &drops)
        })
        .map({
            let flattened_seen = Arc::clone(&flattened_seen);
            move |In(item): In<NonClone>| {
                flattened_seen.lock().unwrap().push(item.label.clone());
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(flattened_seen.lock().unwrap().as_slice(), &["flat:42"]);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    flatten_claim.release(app.world_mut());

    // 1b. `switch` with `Option::take()` + `signal::option(...)` + `.cache()` consumes a one-shot
    //     linear `BoxedSignal<i32>` and projects into `Option<NonClone>`.
    let switch_selector: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(Some(false)));
    let mut one_shot_inner: Option<BoxedSignal<i32>> = Some(signal::once(99).boxed());
    let switch_seen: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let switch_claim = signal::from_system({
        let switch_selector = Arc::clone(&switch_selector);
        move |In(())| switch_selector.lock().unwrap().take()
    })
    .switch(move |In(use_inner): In<bool>| {
        let picked = if use_inner { one_shot_inner.take() } else { None };
        signal::option(picked).cache()
    })
    .map_in({
        let drops = Arc::clone(&drops);
        move |opt: Option<i32>| opt.map(|v| NonClone::new(format!("opt:{v}"), &drops))
    })
    .map({
        let switch_seen = Arc::clone(&switch_seen);
        move |In(opt): In<Option<NonClone>>| {
            switch_seen.lock().unwrap().push(opt.map(|item| item.label.clone()));
        }
    })
    .register(app.world_mut());

    app.update();
    assert_eq!(switch_seen.lock().unwrap().as_slice(), &[None]);

    *switch_selector.lock().unwrap() = Some(true);
    app.update();
    assert_eq!(
        switch_seen.lock().unwrap().as_slice(),
        &[None, Some(String::from("opt:99"))]
    );
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    switch_claim.release(app.world_mut());

    // 1c. `switch_signal_vec` with purely structural `to_signal_vec().cache()` preserves equal
    //     consecutive vector emissions and maps downstream into `NonClone` items.
    let vec_selector: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(Some(0)));
    let stream_zero_batches: Arc<Mutex<VecDeque<Vec<i32>>>> =
        Arc::new(Mutex::new(VecDeque::from([vec![1, 2], vec![1, 2], vec![3]])));
    let mutable_vec_one = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
    let vec_replace_count = Arc::new(AtomicUsize::new(0));
    let reconstructed_vec: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let switch_vec_claim = signal::from_system({
        let vec_selector = Arc::clone(&vec_selector);
        move |In(())| vec_selector.lock().unwrap().take()
    })
    .switch_signal_vec({
        let stream_zero_batches = Arc::clone(&stream_zero_batches);
        move |In(which): In<usize>| {
            if which == 0 {
                let stream_zero_batches = Arc::clone(&stream_zero_batches);
                signal::from_system(move |In(())| stream_zero_batches.lock().unwrap().pop_front())
                    .to_signal_vec()
                    .cache()
                    .boxed_replayable()
            } else {
                mutable_vec_one.signal_vec().boxed_replayable()
            }
        }
    })
    .map_in({
        let drops = Arc::clone(&drops);
        move |value: i32| NonClone::new(format!("v:{value}"), &drops)
    })
    .for_each({
        let vec_replace_count = Arc::clone(&vec_replace_count);
        let reconstructed_vec = Arc::clone(&reconstructed_vec);
        move |In(diffs): In<Vec<VecDiff<NonClone>>>| {
            let string_diffs = diffs
                .into_iter()
                .map(|diff| {
                    if matches!(diff, VecDiff::Replace { .. }) {
                        vec_replace_count.fetch_add(1, Ordering::SeqCst);
                    }
                    map_vec_diff(diff, |item| item.label.clone())
                })
                .collect::<Vec<_>>();
            apply_vec_diffs(&mut reconstructed_vec.lock().unwrap(), string_diffs);
        }
    })
    .register(app.world_mut());

    app.update();
    assert_eq!(reconstructed_vec.lock().unwrap().as_slice(), &["v:1", "v:2"]);
    assert_eq!(vec_replace_count.load(Ordering::SeqCst), 1);

    // Equal consecutive `vec![1, 2]` on `to_signal_vec().cache()` still delivers `Replace`!
    app.update();
    assert_eq!(reconstructed_vec.lock().unwrap().as_slice(), &["v:1", "v:2"]);
    assert_eq!(vec_replace_count.load(Ordering::SeqCst), 2);

    app.update();
    assert_eq!(reconstructed_vec.lock().unwrap().as_slice(), &["v:3"]);
    assert_eq!(vec_replace_count.load(Ordering::SeqCst), 3);

    *vec_selector.lock().unwrap() = Some(1);
    app.update();
    assert_eq!(reconstructed_vec.lock().unwrap().as_slice(), &["v:10", "v:20"]);
    switch_vec_claim.release(app.world_mut());

    // 1d. Nested `switch_signal_map` over linear `.map_value(...).cache()` and downstream
    //     move-only `.map_value(...)` producing `NonClone`.
    let outer_map_sel: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(Some(false)));
    let map_a = MutableBTreeMap::builder()
        .values([(1, 10), (2, 20)])
        .spawn(app.world_mut());
    let map_b = MutableBTreeMap::builder().values([(3, 30)]).spawn(app.world_mut());
    let reconstructed_map: Arc<Mutex<BTreeMap<i32, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

    let nested_map_claim = signal::from_system({
        let outer_map_sel = Arc::clone(&outer_map_sel);
        move |In(())| outer_map_sel.lock().unwrap().take()
    })
    .switch_signal_map(move |In(flag): In<bool>| {
        signal::once(flag).switch_signal_map({
            let map_a = map_a.clone();
            let map_b = map_b.clone();
            move |In(inner_flag): In<bool>| {
                if !inner_flag {
                    map_a
                        .signal_map()
                        .map_value(|In(v): In<i32>| v + 1)
                        .cache()
                        .boxed_replayable()
                } else {
                    map_b
                        .signal_map()
                        .map_value(|In(v): In<i32>| v + 2)
                        .cache()
                        .boxed_replayable()
                }
            }
        })
    })
    .map_value({
        let drops = Arc::clone(&drops);
        move |In(v): In<i32>| NonClone::new(format!("m:{v}"), &drops)
    })
    .for_each({
        let reconstructed_map = Arc::clone(&reconstructed_map);
        move |In(diffs): In<Vec<MapDiff<i32, NonClone>>>| {
            let string_diffs = diffs
                .into_iter()
                .map(|diff| match diff {
                    MapDiff::Replace { entries } => MapDiff::Replace {
                        entries: entries.into_iter().map(|(k, v)| (k, v.label.clone())).collect(),
                    },
                    MapDiff::Insert { key, value } => MapDiff::Insert {
                        key,
                        value: value.label.clone(),
                    },
                    MapDiff::Update { key, value } => MapDiff::Update {
                        key,
                        value: value.label.clone(),
                    },
                    MapDiff::Remove { key } => MapDiff::Remove { key },
                    MapDiff::Clear => MapDiff::Clear,
                })
                .collect();
            apply_map_diffs(&mut reconstructed_map.lock().unwrap(), string_diffs);
        }
    })
    .register(app.world_mut());

    app.update();
    assert_eq!(
        *reconstructed_map.lock().unwrap(),
        BTreeMap::from([(1, String::from("m:11")), (2, String::from("m:21"))])
    );

    *outer_map_sel.lock().unwrap() = Some(true);
    app.update();
    assert_eq!(
        *reconstructed_map.lock().unwrap(),
        BTreeMap::from([(3, String::from("m:32"))])
    );
    nested_map_claim.release(app.world_mut());

    // 1e. `map_signal`, `filter_signal`, and `map_value_signal` consuming `NonClone` upstream
    //     items/values (where semantically linear) and projecting into `NonClone` outputs.
    let map_sig_batches: Arc<Mutex<Option<Vec<NonClone>>>> = Arc::new(Mutex::new(Some(vec![
        NonClone::new("alpha", &drops),
        NonClone::new("beta", &drops),
    ])));
    let map_sig_model: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let map_sig_claim = signal::from_system({
        let map_sig_batches = Arc::clone(&map_sig_batches);
        move |In(())| map_sig_batches.lock().unwrap().take()
    })
    .to_signal_vec()
    .map_signal(|In(item): In<NonClone>| {
        let len = item.label.len();
        signal::once(format!("{}:{len}", item.label)).cache()
    })
    .map_in({
        let drops = Arc::clone(&drops);
        move |summary: String| NonClone::new(summary, &drops)
    })
    .for_each({
        let map_sig_model = Arc::clone(&map_sig_model);
        move |In(diffs): In<Vec<VecDiff<NonClone>>>| {
            let mapped = diffs
                .into_iter()
                .map(|d| map_vec_diff(d, |item| item.label.clone()))
                .collect();
            apply_vec_diffs(&mut map_sig_model.lock().unwrap(), mapped);
        }
    })
    .register(app.world_mut());

    let filter_source = MutableVec::builder().values([1, 2, 3, 4]).spawn(app.world_mut());
    let filter_sig_model: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let filter_sig_claim = filter_source
        .signal_vec()
        .filter_signal(|In(n): In<i32>| signal::once(n % 2 == 0).cache())
        .map_in({
            let drops = Arc::clone(&drops);
            move |even: i32| NonClone::new(format!("even:{even}"), &drops)
        })
        .for_each({
            let filter_sig_model = Arc::clone(&filter_sig_model);
            move |In(diffs): In<Vec<VecDiff<NonClone>>>| {
                let mapped = diffs
                    .into_iter()
                    .map(|d| map_vec_diff(d, |item| item.label.clone()))
                    .collect();
                apply_vec_diffs(&mut filter_sig_model.lock().unwrap(), mapped);
            }
        })
        .register(app.world_mut());

    let val_sig_source = MutableBTreeMap::builder()
        .values([(10, 5), (20, 7)])
        .spawn(app.world_mut());
    let val_sig_model: Arc<Mutex<BTreeMap<i32, String>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let val_sig_claim = val_sig_source
        .signal_map()
        .map_value({
            let drops = Arc::clone(&drops);
            move |In(v): In<i32>| NonClone::new(format!("in:{v}"), &drops)
        })
        .map_value_signal(|In(item): In<NonClone>| signal::once(format!("dyn({})", item.label)).cache())
        .map_value({
            let drops = Arc::clone(&drops);
            move |In(text): In<String>| NonClone::new(text, &drops)
        })
        .for_each({
            let val_sig_model = Arc::clone(&val_sig_model);
            move |In(diffs): In<Vec<MapDiff<i32, NonClone>>>| {
                let mapped = diffs
                    .into_iter()
                    .map(|diff| match diff {
                        MapDiff::Replace { entries } => MapDiff::Replace {
                            entries: entries.into_iter().map(|(k, v)| (k, v.label.clone())).collect(),
                        },
                        MapDiff::Insert { key, value } => MapDiff::Insert {
                            key,
                            value: value.label.clone(),
                        },
                        MapDiff::Update { key, value } => MapDiff::Update {
                            key,
                            value: value.label.clone(),
                        },
                        MapDiff::Remove { key } => MapDiff::Remove { key },
                        MapDiff::Clear => MapDiff::Clear,
                    })
                    .collect();
                apply_map_diffs(&mut val_sig_model.lock().unwrap(), mapped);
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(map_sig_model.lock().unwrap().as_slice(), &["alpha:5", "beta:4"]);
    assert_eq!(filter_sig_model.lock().unwrap().as_slice(), &["even:2", "even:4"]);
    assert_eq!(
        *val_sig_model.lock().unwrap(),
        BTreeMap::from([(10, String::from("dyn(in:5)")), (20, String::from("dyn(in:7)"))])
    );

    map_sig_claim.release(app.world_mut());
    filter_sig_claim.release(app.world_mut());
    val_sig_claim.release(app.world_mut());
    app.update();
}

// ============================================================================
// 2. Dynamic switching into linear `!Clone` `Builder` entity trees
// ============================================================================

#[cfg(feature = "builder")]
mod builder_dynamic_tests {
    use super::*;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct Tag(String);

    fn child_tags(world: &World, parent: Entity) -> Vec<String> {
        let Some(children) = world.get::<Children>(parent) else {
            return Vec::new();
        };
        children
            .iter()
            .filter_map(|child| world.get::<Tag>(child).map(|tag| tag.0.clone()))
            .collect()
    }

    #[test]
    fn builtin_dynamic_operators_drive_linear_builder_trees() {
        let mut app = test_app();

        let child_mode: Arc<Mutex<Option<Option<&'static str>>>> = Arc::new(Mutex::new(Some(Some("solo-a"))));
        let list_mode: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(Some(0)));
        let vec_a = MutableVec::builder().values(["v0", "v1"]).spawn(app.world_mut());
        let vec_b = MutableVec::builder().values(["w0"]).spawn(app.world_mut());

        let solo_parent = jonmo::Builder::new()
            .child_signal(
                signal::from_system({
                    let child_mode = Arc::clone(&child_mode);
                    move |In(())| child_mode.lock().unwrap().take()
                })
                .switch(|In(label): In<Option<&'static str>>| signal::once(label).cache())
                .map_in(|opt: Option<&'static str>| {
                    opt.map(|name| jonmo::Builder::new().insert(Tag(String::from(name))))
                }),
            )
            .spawn(app.world_mut());

        let list_parent = jonmo::Builder::new()
            .children_signal_vec(
                signal::from_system({
                    let list_mode = Arc::clone(&list_mode);
                    move |In(())| list_mode.lock().unwrap().take()
                })
                .switch_signal_vec({
                    let vec_b = vec_b.clone();
                    move |In(which): In<usize>| {
                        if which == 0 {
                            vec_a.signal_vec().boxed_replayable()
                        } else {
                            vec_b.signal_vec().boxed_replayable()
                        }
                    }
                })
                .map_in(|label: &'static str| jonmo::Builder::new().insert(Tag(String::from(label)))),
            )
            .spawn(app.world_mut());

        app.update();
        assert_eq!(child_tags(app.world(), solo_parent), vec![String::from("solo-a")]);
        assert_eq!(
            child_tags(app.world(), list_parent),
            vec![String::from("v0"), String::from("v1")]
        );

        *child_mode.lock().unwrap() = Some(Some("solo-b"));
        *list_mode.lock().unwrap() = Some(1);
        app.update();
        assert_eq!(child_tags(app.world(), solo_parent), vec![String::from("solo-b")]);
        assert_eq!(child_tags(app.world(), list_parent), vec![String::from("w0")]);

        *child_mode.lock().unwrap() = Some(None);
        app.world_mut().flush();
        vec_b.write(app.world_mut()).push("w1");
        app.update();
        assert!(child_tags(app.world(), solo_parent).is_empty());
        assert_eq!(
            child_tags(app.world(), list_parent),
            vec![String::from("w0"), String::from("w1")]
        );

        app.world_mut().entity_mut(solo_parent).despawn();
        app.world_mut().entity_mut(list_parent).despawn();
        app.update();
    }
}

// ============================================================================
// 3. External custom ordinary dynamic operators (`!Clone` queues & outputs)
// ============================================================================

#[test]
fn custom_ordinary_dynamic_operators_support_non_clone_queues_and_warming_cohorts() {
    type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<BoxedReplayableSignal<i32>>, NonClone>;
    type KeyedMapUpdate = DynamicManagerUpdate<
        KeyedSlotCommand<&'static str, BoxedReplayableSignal<i32>>,
        MapDiff<NonCloneOpaque, NonClone>,
    >;

    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let key_drops = Arc::new(AtomicUsize::new(0));

    // Gate for a warming candidate in cohort 2 so we prove generation 1 stays active while
    // cohort 2 warms, and cohort 2 publishes atomically once all members become ready.
    let gen1_slot0: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(Some(10)));
    let gen1_slot1: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(Some(20)));
    let gen2_ready_slot0: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(Some(100)));
    let gen2_warming_slot1: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(None));

    let make_gate_signal = |gate: Arc<Mutex<Option<i32>>>| -> BoxedReplayableSignal<i32> {
        signal::from_system(move |In(())| gate.lock().unwrap().take())
            .cache()
            .boxed_replayable()
    };

    let indexed_updates: Arc<Mutex<VecDeque<IndexedUpdate>>> = Arc::new(Mutex::new(VecDeque::from([
        // Frame 1: install generation 1 (`[10, 20]`).
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![
                    make_gate_signal(Arc::clone(&gen1_slot0)),
                    make_gate_signal(Arc::clone(&gen1_slot1)),
                ],
            }],
            DynamicEmissions::None,
        ),
        // Frame 2: stage cohort 2 where slot 0 is ready (`100`) and slot 1 is still `Warming`.
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![
                    make_gate_signal(Arc::clone(&gen2_ready_slot0)),
                    make_gate_signal(Arc::clone(&gen2_warming_slot1)),
                ],
            }],
            DynamicEmissions::None,
        ),
    ])));

    let indexed_batches: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let indexed_dynamic = CustomScalar(ScalarSignalMaterializer::<Vec<NonClone>>::new({
        let indexed_updates = Arc::clone(&indexed_updates);
        let drops = Arc::clone(&drops);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| indexed_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(move |index, value| {
                        DynamicEmissions::One(NonClone::new(format!("slot{index}={value}"), &drops))
                    })
                    .scalar_output(|queued: Vec<NonClone>| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        }
    }));

    let indexed_claim = indexed_dynamic
        .map({
            let indexed_batches = Arc::clone(&indexed_batches);
            move |In(batch): In<Vec<NonClone>>| {
                indexed_batches
                    .lock()
                    .unwrap()
                    .push(batch.into_iter().map(|item| item.label.clone()).collect());
            }
        })
        .register(app.world_mut());

    // Frame 1: generation 1 becomes ready immediately on its scheduled pass.
    app.update();
    assert_eq!(
        indexed_batches.lock().unwrap().as_slice(),
        &[vec![String::from("slot0=10"), String::from("slot1=20")]]
    );

    // Frame 2: cohort 2 stages, but slot 1 is still warming; simultaneously push `11` on gen 1
    // slot 0 to prove generation 1 remains current while cohort 2 warms!
    *gen1_slot0.lock().unwrap() = Some(11);
    app.update();
    assert_eq!(
        indexed_batches.lock().unwrap().as_slice(),
        &[
            vec![String::from("slot0=10"), String::from("slot1=20")],
            vec![String::from("slot0=11")],
        ]
    );

    // Frame 3: warm up cohort 2's slot 1 (`200`) AND push stale `12` on generation 1 -> cohort 2
    // publishes `[slot0=100, slot1=200]` atomically and cancels generation 1!
    *gen2_warming_slot1.lock().unwrap() = Some(200);
    *gen1_slot0.lock().unwrap() = Some(12);
    app.update();
    assert_eq!(
        indexed_batches.lock().unwrap().as_slice(),
        &[
            vec![String::from("slot0=10"), String::from("slot1=20")],
            vec![String::from("slot0=11")],
            vec![String::from("slot0=100"), String::from("slot1=200")],
        ]
    );

    // Frame 4: empty `ReplaceAll` cancels cohort 2 cleanly and emits the manager emission.
    indexed_updates.lock().unwrap().push_back(IndexedUpdate::new(
        [IndexedSlotCommand::ReplaceAll { signals: Vec::new() }],
        DynamicEmissions::One(NonClone::new("cleared", &drops)),
    ));
    app.update();
    assert_eq!(
        indexed_batches.lock().unwrap().last().unwrap(),
        &vec![String::from("cleared")]
    );
    indexed_claim.release(app.world_mut());

    // Ordinary `.scalar_output`, `.signal_vec_output`, and `.signal_map_output` install no replay
    // cache (`inspect_*_replay_cache == None`), and `.signal_map_output` supports `!Clone + !Ord`
    // keys (`NonCloneOpaque`).
    let probe_scalar_claim = CustomScalar(ScalarSignalMaterializer::<i32>::new(|registrar| {
        let manager = registrar.signal_entity(
            (),
            |In(())| None::<DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, i32>>,
            OutputDelivery::move_only(),
        )?;
        registrar.dynamic_operator(
            DynamicSignalOperatorSpec::scalar(manager)
                .replay_scalar::<i32>()
                .adapter(|(), v| DynamicEmissions::One(v))
                .scalar_output(|q| q.into_iter().last())
                .same_signal_identity(SameSignalIdentity::KeepCurrent),
        )
    }))
    .register(app.world_mut());
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), probe_scalar_claim.entity()),
        None
    );
    probe_scalar_claim.release(app.world_mut());

    let probe_vec_claim = CustomVec(VecSignalMaterializer::<i32>::new(|registrar| {
        let manager = registrar.signal_entity(
            (),
            |In(())| None::<DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, VecDiff<i32>>>,
            OutputDelivery::move_only(),
        )?;
        registrar.dynamic_operator(
            DynamicSignalOperatorSpec::scalar(manager)
                .replay_scalar::<i32>()
                .adapter(|(), v| DynamicEmissions::One(VecDiff::Push { value: v }))
                .signal_vec_output(|q| (!q.is_empty()).then_some(q))
                .same_signal_identity(SameSignalIdentity::KeepCurrent),
        )
    }))
    .register(app.world_mut());
    assert_eq!(
        inspect_vec_replay_cache::<i32>(app.world(), probe_vec_claim.entity()),
        None
    );
    probe_vec_claim.release(app.world_mut());

    let probe_map_claim = CustomMap(MapSignalMaterializer::<i32, i32>::new(|registrar| {
        let manager = registrar.signal_entity(
            (),
            |In(())| None::<DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, MapDiff<i32, i32>>>,
            OutputDelivery::move_only(),
        )?;
        registrar.dynamic_operator(
            DynamicSignalOperatorSpec::scalar(manager)
                .replay_scalar::<i32>()
                .adapter(|(), v| DynamicEmissions::One(MapDiff::Insert { key: v, value: v }))
                .signal_map_output(|q| (!q.is_empty()).then_some(q))
                .same_signal_identity(SameSignalIdentity::KeepCurrent),
        )
    }))
    .register(app.world_mut());
    assert_eq!(
        inspect_map_replay_cache::<i32, i32>(app.world(), probe_map_claim.entity()),
        None
    );
    probe_map_claim.release(app.world_mut());

    // Keyed ordinary map dynamic operator emitting `MapDiff<NonCloneOpaque, NonClone>` (`!Ord`
    // key).
    let keyed_updates: Arc<Mutex<VecDeque<KeyedMapUpdate>>> =
        Arc::new(Mutex::new(VecDeque::from([KeyedMapUpdate::new(
            [KeyedSlotCommand::ReplaceAll {
                entries: vec![
                    ("k1", signal::once(5).cache().boxed_replayable()),
                    ("k2", signal::once(9).cache().boxed_replayable()),
                ],
            }],
            DynamicEmissions::None,
        )])));
    let keyed_seen: Arc<Mutex<Vec<(usize, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let keyed_claim = CustomMap(MapSignalMaterializer::<NonCloneOpaque, NonClone>::new({
        let keyed_updates = Arc::clone(&keyed_updates);
        let key_drops = Arc::clone(&key_drops);
        let drops = Arc::clone(&drops);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| keyed_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(move |slot_key, value| {
                        let numeric_id = if slot_key == "k1" { 1 } else { 2 };
                        DynamicEmissions::One(MapDiff::Insert {
                            key: NonCloneOpaque::new(numeric_id, &key_drops),
                            value: NonClone::new(format!("{slot_key}:{value}"), &drops),
                        })
                    })
                    .signal_map_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        }
    }))
    .for_each({
        let keyed_seen = Arc::clone(&keyed_seen);
        move |In(diffs): In<Vec<MapDiff<NonCloneOpaque, NonClone>>>| {
            for diff in diffs {
                if let MapDiff::Insert { key, value } = diff {
                    keyed_seen.lock().unwrap().push((key.id, value.label.clone()));
                }
            }
        }
    })
    .register(app.world_mut());

    app.update();
    let mut pairs = keyed_seen.lock().unwrap().clone();
    pairs.sort();
    assert_eq!(pairs, vec![(1, String::from("k1:5")), (2, String::from("k2:9"))]);
    assert_eq!(key_drops.load(Ordering::SeqCst), 2);
    keyed_claim.release(app.world_mut());
    app.update();
}

// ============================================================================
// 4. External custom replayable dynamic operators & broadcast `N - 1` parity
// ============================================================================

#[test]
fn custom_replayable_dynamic_operators_match_builtin_replay_cache_and_broadcast_parity() {
    type ScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<CloneCounter>>, CloneCounter>;
    type VecUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, VecDiff<i32>>;
    type MapUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, MapDiff<i32, i32>>;

    let mut app = test_app();
    let clones = Arc::new(AtomicUsize::new(0));
    let source_queue: Arc<Mutex<Option<CloneCounter>>> = Arc::new(Mutex::new(Some(CloneCounter::new(7, &clones))));
    let inner_source = CustomScalar(ScalarSignalMaterializer::new({
        let source_queue = Arc::clone(&source_queue);
        move |registrar| {
            registrar.replayable_signal_entity(
                (),
                move |In(())| source_queue.lock().unwrap().take(),
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        }
    }))
    .cache()
    .boxed_replayable();

    let scalar_updates: Arc<Mutex<VecDeque<ScalarUpdate>>> = Arc::new(Mutex::new(VecDeque::from([ScalarUpdate::new(
        [ScalarSlotCommand::Select { signal: inner_source }],
        DynamicEmissions::None,
    )])));

    let custom_replayable_scalar = CustomScalar(ScalarSignalMaterializer::new({
        let scalar_updates = Arc::clone(&scalar_updates);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| scalar_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<CloneCounter>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .replayable_scalar_output(|mut queued| queued.pop())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        }
    }))
    .cache()
    .broadcast();

    // Direct `.cache()` on a replayable broadcast normalizes to the exact same terminal entity!
    let direct_claim = custom_replayable_scalar.clone().register(app.world_mut());
    let cached_claim = custom_replayable_scalar.clone().cache().register(app.world_mut());
    let owner_entity = direct_claim.entity();
    assert_eq!(cached_claim.entity(), owner_entity);
    assert_eq!(
        inspect_scalar_replay_cache::<CloneCounter>(app.world(), owner_entity),
        Some(ReplayCacheInspection::Warming),
        "replayable dynamic stable output must start Warming without executing during registration"
    );
    direct_claim.release(app.world_mut());
    cached_claim.release(app.world_mut());

    // Subscribe 3 distinct active dependents (`N = 3`) and verify readiness + exact clones.
    let branch_outputs: Arc<Mutex<Vec<(usize, i32)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut branch_claims = Vec::new();
    for branch_id in 0..3 {
        let branch_outputs = Arc::clone(&branch_outputs);
        branch_claims.push(
            custom_replayable_scalar
                .clone()
                .map(move |In(counter): In<CloneCounter>| {
                    branch_outputs.lock().unwrap().push((branch_id, counter.id));
                })
                .register(app.world_mut()),
        );
    }

    clones.store(0, Ordering::SeqCst);
    app.update();
    assert_eq!(branch_outputs.lock().unwrap().len(), 3);
    assert!(matches!(
        inspect_scalar_replay_cache::<CloneCounter>(app.world(), owner_entity),
        Some(ReplayCacheInspection::Ready { current, revision })
            if current.id == 7 && revision == 1
    ));

    // Once all 3 relationships are established (`Published`), a steady-state emission on the
    // broadcast replayable dynamic output performs:
    // - 1 clone for `inner_source`'s replay cache retention,
    // - 1 clone for `custom_replayable_scalar`'s replay cache retention, and
    // - exact `N - 1 = 2` clones for forwarding to 3 live active dependents (`total = 4`).
    clones.store(0, Ordering::SeqCst);
    *source_queue.lock().unwrap() = Some(CloneCounter::new(8, &clones));
    app.update();
    assert_eq!(branch_outputs.lock().unwrap().len(), 6);
    assert_eq!(
        clones.load(Ordering::SeqCst),
        4,
        "expected 1 inner cache clone + 1 dynamic output cache clone + (3 - 1) broadcast clones"
    );

    for claim in branch_claims {
        claim.release(app.world_mut());
    }
    drop(custom_replayable_scalar);

    // Custom `.replayable_signal_vec_output` and `.replayable_signal_map_output` start Warming,
    // normalize under `.cache()`, and compose directly inside `switch_signal_vec` /
    // `switch_signal_map`.
    let custom_replay_vec = CustomVec(VecSignalMaterializer::<i32>::new(|registrar| {
        let updates = Arc::new(Mutex::new(VecDeque::from([VecUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::once(33).cache().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
        let manager = registrar.signal_entity(
            (),
            move |In(())| updates.lock().unwrap().pop_front(),
            OutputDelivery::move_only(),
        )?;
        registrar.dynamic_operator(
            DynamicSignalOperatorSpec::scalar(manager)
                .replay_scalar::<i32>()
                .adapter(|(), v| DynamicEmissions::One(VecDiff::Replace { values: vec![v] }))
                .replayable_signal_vec_output(|q| (!q.is_empty()).then_some(q))
                .same_signal_identity(SameSignalIdentity::KeepCurrent),
        )
    }))
    .cache();

    let mut one_shot_vec = Some(custom_replay_vec);
    let switched_vec_seen: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let switched_vec_claim = signal::once(())
        .switch_signal_vec(move |In(())| one_shot_vec.take().unwrap())
        .for_each({
            let switched_vec_seen = Arc::clone(&switched_vec_seen);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                apply_vec_diffs(&mut switched_vec_seen.lock().unwrap(), diffs);
            }
        })
        .register(app.world_mut());

    let custom_replay_map = CustomMap(MapSignalMaterializer::<i32, i32>::new(|registrar| {
        let updates = Arc::new(Mutex::new(VecDeque::from([MapUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::once(44).cache().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
        let manager = registrar.signal_entity(
            (),
            move |In(())| updates.lock().unwrap().pop_front(),
            OutputDelivery::move_only(),
        )?;
        registrar.dynamic_operator(
            DynamicSignalOperatorSpec::scalar(manager)
                .replay_scalar::<i32>()
                .adapter(|(), v| DynamicEmissions::One(MapDiff::Replace { entries: vec![(1, v)] }))
                .replayable_signal_map_output(|q| (!q.is_empty()).then_some(q))
                .same_signal_identity(SameSignalIdentity::KeepCurrent),
        )
    }))
    .cache();

    let mut one_shot_map = Some(custom_replay_map);
    let switched_map_seen: Arc<Mutex<BTreeMap<i32, i32>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let switched_map_claim = signal::once(())
        .switch_signal_map(move |In(())| one_shot_map.take().unwrap())
        .for_each({
            let switched_map_seen = Arc::clone(&switched_map_seen);
            move |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                apply_map_diffs(&mut switched_map_seen.lock().unwrap(), diffs);
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(switched_vec_seen.lock().unwrap().as_slice(), &[33]);
    assert_eq!(*switched_map_seen.lock().unwrap(), BTreeMap::from([(1, 44)]));

    switched_vec_claim.release(app.world_mut());
    switched_map_claim.release(app.world_mut());
    app.update();
}

// ============================================================================
// 5. Public dynamic candidate `MoveOnlyFanout` rejection & cohort rollback
// ============================================================================

#[test]
fn external_dynamic_candidate_move_only_fanout_discards_cohort_and_preserves_old_generation() {
    type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<BoxedReplayableSignal<i32>>, (usize, i32)>;

    let invalid_move_only_fanout_candidate = || -> BoxedReplayableSignal<i32> {
        CustomScalar(ScalarSignalMaterializer::new(|registrar| {
            let move_only_src = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())?;
            let branch_a = registrar.signal_entity(
                move_only_src,
                |In(v): In<i32>| Some(v),
                OutputDelivery::<i32>::move_only(),
            )?;
            let branch_b = registrar.signal_entity(
                move_only_src,
                |In(v): In<i32>| Some(v),
                OutputDelivery::<i32>::move_only(),
            )?;
            // Activating this replayable terminal activates BOTH `branch_a` and `branch_b` on
            // `move_only_src`, triggering `SignalRegistrationError::MoveOnlyFanout` during
            // dynamic candidate preflight!
            registrar.replayable_signal_entity(
                [branch_a, branch_b],
                |In(v): In<i32>| Some(v),
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        }))
        .cache()
        .boxed_replayable()
    };

    let mut app = test_app();
    let gen1_feed: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(Some(10)));
    let errors: Arc<Mutex<Vec<SignalRegistrationError>>> = Arc::new(Mutex::new(Vec::new()));
    let outputs = Arc::new(Mutex::new(Vec::new()));

    let updates: Arc<Mutex<VecDeque<IndexedUpdate>>> = Arc::new(Mutex::new(VecDeque::from([
        // Frame 1: establish active generation 1.
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![
                    signal::from_system({
                        let gen1_feed = Arc::clone(&gen1_feed);
                        move |In(())| gen1_feed.lock().unwrap().take()
                    })
                    .cache()
                    .boxed_replayable(),
                ],
            }],
            DynamicEmissions::None,
        ),
        // Frame 2: attempt `ReplaceAll` where slot 0 is a valid warming candidate and slot 1
        // triggers `MoveOnlyFanout` -> entire cohort must roll back and preserve generation 1!
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![
                    signal::from_system(|In(())| None::<i32>).cache().boxed_replayable(),
                    invalid_move_only_fanout_candidate(),
                ],
            }],
            DynamicEmissions::None,
        ),
    ])));

    let claim = CustomScalar(ScalarSignalMaterializer::new({
        let updates = Arc::clone(&updates);
        let errors = Arc::clone(&errors);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |err| {
                        errors.lock().unwrap().push(err);
                    }),
            )
        }
    }))
    .map({
        let outputs = Arc::clone(&outputs);
        move |In(batch): In<Vec<(usize, i32)>>| {
            outputs.lock().unwrap().push(batch);
        }
    })
    .register(app.world_mut());

    app.update();
    assert_eq!(outputs.lock().unwrap().as_slice(), &[vec![(0, 10)]]);
    let entity_count_before_rejection = app.world().entities().len();

    app.update();
    assert_eq!(
        errors.lock().unwrap().as_slice(),
        &[SignalRegistrationError::MoveOnlyFanout]
    );
    // Drain world cleanup so rolled-back candidate entities reap cleanly.
    app.update();
    assert_eq!(app.world().entities().len(), entity_count_before_rejection);

    // Generation 1 remains alive and continues delivering new values!
    *gen1_feed.lock().unwrap() = Some(25);
    app.update();
    assert_eq!(outputs.lock().unwrap().as_slice(), &[vec![(0, 10)], vec![(0, 25)]]);
    claim.release(app.world_mut());

    // Default error handler panics actionably with `MoveOnlyFanout` after discarding the candidate.
    let panic_updates: Arc<Mutex<VecDeque<IndexedUpdate>>> =
        Arc::new(Mutex::new(VecDeque::from([IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![invalid_move_only_fanout_candidate()],
            }],
            DynamicEmissions::None,
        )])));
    let default_claim = CustomScalar(ScalarSignalMaterializer::<Vec<(usize, i32)>>::new({
        let panic_updates = Arc::clone(&panic_updates);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| panic_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .scalar_output(|queued| (!queued.is_empty()).then_some(queued))
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        }
    }))
    .register(app.world_mut());

    let panic = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
        app.update();
    }))
    .expect_err("default dynamic candidate policy must panic on MoveOnlyFanout");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    assert!(
        message.contains("MoveOnlyFanout"),
        "expected MoveOnlyFanout in panic message, got: {message}"
    );
    default_claim.release(app.world_mut());
}

// ============================================================================
// 5b. Multi-update queued dynamic depth transitions & nested dynamic level sync
// ============================================================================

#[test]
fn dynamic_depth_changes_across_queued_updates_and_nested_chains_keep_levels_consistent() {
    #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
    struct UpstreamSchedule;

    type ScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, i32>;

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<UpstreamSchedule>());

    // Part A: Cross-schedule queued `DynamicManagerUpdate`s processed in a single ordered
    // `manager_bridge` drain where Update 1 raises `owner` depth (`once -> map_in -> map_in ->
    // cache -> adapter -> owner`), Update 2 lowers it (`Clear`), and Update 3 raises it again.
    let queued_updates: Arc<Mutex<VecDeque<ScalarUpdate>>> = Arc::new(Mutex::new(VecDeque::new()));
    let observed = Arc::new(Mutex::new(Vec::new()));

    let custom_claim = CustomScalar(ScalarSignalMaterializer::<i32>::new({
        let queued_updates = Arc::clone(&queued_updates);
        move |registrar| {
            let manager_signal = signal::from_system(move |In(())| queued_updates.lock().unwrap().pop_front())
                .schedule::<UpstreamSchedule>();
            let manager = registrar.materialize_signal(Box::new(manager_signal).into_signal_materializer())?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), v| DynamicEmissions::One(v))
                    .scalar_output(|mut q| q.pop())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        }
    }))
    .schedule::<Last>()
    .map({
        let observed = Arc::clone(&observed);
        move |In(v): In<i32>| {
            observed.lock().unwrap().push(v);
        }
    })
    .register(app.world_mut());

    // Pre-warm two deep replayable chains at different topological depths (`level 3` and
    // `level 2`) so selecting them in `manager_bridge` immediately commits and releases live
    // `DynamicSignalConnection`s inside the same `execute_signal_node` drain!
    let ready_deep_a = signal::once(10).map_in(|v| v + 1).map_in(|v| v + 1).cache().broadcast();
    let ready_deep_b = signal::once(30).map_in(|v| v + 5).cache().broadcast();
    let hold_a = ready_deep_a.clone().register(app.world_mut());
    let hold_b = ready_deep_b.clone().register(app.world_mut());
    app.world_mut().run_schedule(Last);

    // Push THREE updates in `UpstreamSchedule` before running `Last`, so `manager_bridge` in `Last`
    // drains all three updates back-to-back inside one `execute_signal_node` pass!
    queued_updates.lock().unwrap().push_back(ScalarUpdate::new(
        [ScalarSlotCommand::Select {
            signal: ready_deep_a.boxed_replayable(),
        }],
        DynamicEmissions::One(1),
    ));
    queued_updates
        .lock()
        .unwrap()
        .push_back(ScalarUpdate::new([ScalarSlotCommand::Clear], DynamicEmissions::One(2)));
    queued_updates.lock().unwrap().push_back(ScalarUpdate::new(
        [ScalarSlotCommand::Select {
            signal: ready_deep_b.boxed_replayable(),
        }],
        DynamicEmissions::One(3),
    ));

    for _ in 0..3 {
        app.world_mut().run_schedule(UpstreamSchedule);
    }
    // Because `sync_dynamic_owner_levels` raised `owner` to level 4 (`adapter_b` is at level 3)
    // as soon as Update 3 attached `ready_deep_b`, `Last` executes `adapter_b` BEFORE `owner` in
    // Phase 1, coalescing `[2, 3, 35]` into the final authoritative scalar `35` with zero glitch!
    app.world_mut().run_schedule(Last);
    assert_eq!(observed.lock().unwrap().as_slice(), &[35]);

    hold_a.release(app.world_mut());
    hold_b.release(app.world_mut());

    // Part B: Cross-schedule `MutableVec` pushing multiple batches before `Last` into
    // `.map_signal(...)` with deep inner chains.
    let vec_src = MutableVec::builder().values([100, 200]).spawn(app.world_mut());
    let vec_observed = Arc::new(Mutex::new(Vec::new()));
    let map_sig_claim = vec_src
        .signal_vec()
        .schedule::<UpstreamSchedule>()
        .map_signal(|In(v): In<i32>| signal::once(v).map_in(|x| x + 1).map_in(|x| x * 2).cache())
        .schedule::<Last>()
        .for_each({
            let vec_observed = Arc::clone(&vec_observed);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                vec_observed.lock().unwrap().extend(diffs);
            }
        })
        .register(app.world_mut());

    app.world_mut().run_schedule(UpstreamSchedule);
    vec_src.write(app.world_mut()).push(300);
    app.world_mut().run_schedule(UpstreamSchedule);
    vec_src.write(app.world_mut()).remove(0);
    app.world_mut().run_schedule(UpstreamSchedule);
    app.world_mut().run_schedule(Last);
    assert!(!vec_observed.lock().unwrap().is_empty());

    map_sig_claim.release(app.world_mut());
    custom_claim.release(app.world_mut());
    app.update();
}

// ============================================================================
// 6. Dedicated opt-in imperative evaluation on move-only static & dynamic graphs
// ============================================================================

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn imperative_evaluation_steps_move_only_static_and_dynamic_subgraphs_with_documented_mutations() {
    type ScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, NonClone>;

    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));

    // 6a. Move-only linear subgraph (`once(NonClone)` -> stateful `Local<usize>` -> custom
    //     `OutputDelivery::move_only()`): imperative evaluation mutates `once` and `Local`, and
    //     a mismatched downcast cleanly drops the boxed value without leaking.
    let mismatch_claim = signal::once(NonClone::new("mismatch", &drops)).register(app.world_mut());
    let mismatch_boxed = run_signal_subgraph_once(app.world_mut(), mismatch_claim.entity())
        .expect("imperative evaluation should produce boxed NonClone");
    assert!(
        downcast_signal_value::<i32>(mismatch_boxed).is_none(),
        "wrong type downcast must return None"
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    mismatch_claim.release(app.world_mut());

    let linear_claim = CustomScalar(ScalarSignalMaterializer::<NonClone>::new({
        let drops = Arc::clone(&drops);
        move |registrar| {
            let chain = signal::once(NonClone::new("seed", &drops)).map(
                move |In(item): In<NonClone>, mut runs: Local<usize>| {
                    *runs += 1;
                    NonClone::new(format!("{}:run{}", item.label, *runs), &drops)
                },
            );
            let upstream = registrar.materialize_signal(Box::new(chain).into_signal_materializer())?;
            registrar.signal_entity(
                upstream,
                |In(item): In<NonClone>| Some(item),
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    let boxed = run_signal_subgraph_once(app.world_mut(), linear_claim.entity())
        .expect("imperative evaluation should return the one-shot move-only output");
    let item =
        downcast_signal_value::<NonClone>(boxed).expect("right type downcast must recover the owned NonClone value");
    assert_eq!(item.label, "seed:run1");
    drop(item);

    // Because `run_signal_subgraph_once` already consumed `signal::once`, subsequent evaluation
    // produces `None`.
    assert!(run_signal_subgraph_once(app.world_mut(), linear_claim.entity()).is_none());
    linear_claim.release(app.world_mut());

    // 6b. Imperatively stepping an ordinary move-only dynamic operator AND a replayable dynamic
    //     operator advances dynamic slots/generations and replay revisions.
    let dyn_updates: Arc<Mutex<VecDeque<ScalarUpdate>>> = Arc::new(Mutex::new(VecDeque::from([ScalarUpdate::new(
        [ScalarSlotCommand::Select {
            signal: signal::once(55).cache().boxed_replayable(),
        }],
        DynamicEmissions::None,
    )])));
    let dyn_claim = CustomScalar(ScalarSignalMaterializer::<NonClone>::new({
        let dyn_updates = Arc::clone(&dyn_updates);
        let drops = Arc::clone(&drops);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| dyn_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(move |(), v| DynamicEmissions::One(NonClone::new(format!("imp:{v}"), &drops)))
                    .scalar_output(|mut queued| queued.pop())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        }
    }))
    .register(app.world_mut());

    let dyn_boxed = run_signal_subgraph_once(app.world_mut(), dyn_claim.entity())
        .expect("imperative evaluation should step the dynamic subgraph");
    let dyn_item = downcast_signal_value::<NonClone>(dyn_boxed).expect("dynamic output should downcast to NonClone");
    assert_eq!(dyn_item.label, "imp:55");
    drop(dyn_item);
    dyn_claim.release(app.world_mut());

    // Step a `.replayable_scalar_output(...)` dynamic operator imperatively and verify its managed
    // replay cache transitions `Warming -> Ready { current: 88, revision: 1 }`, plus nested dynamic
    // switching referencing a shared broadcast upstream inside a single imperative pass!
    type ReplayScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<i32>>, i32>;
    let replay_dyn_updates: Arc<Mutex<VecDeque<ReplayScalarUpdate>>> =
        Arc::new(Mutex::new(VecDeque::from([ReplayScalarUpdate::new(
            [ScalarSlotCommand::Select {
                signal: signal::once(80).map_in(|v| v + 8).cache().boxed_replayable(),
            }],
            DynamicEmissions::None,
        )])));
    let replay_dyn_shared = CustomScalar(ScalarSignalMaterializer::<i32>::new({
        let replay_dyn_updates = Arc::clone(&replay_dyn_updates);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| replay_dyn_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), v| DynamicEmissions::One(v))
                    .replayable_scalar_output(|mut queued| queued.pop())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        }
    }))
    .cache()
    .broadcast();

    let nested_claim = replay_dyn_shared
        .clone()
        .switch({
            let replay_dyn_shared = replay_dyn_shared.clone();
            move |In(v): In<i32>| replay_dyn_shared.clone().map_in(move |cur| cur + v).cache()
        })
        .broadcast()
        .register(app.world_mut());

    let nested_boxed = run_signal_subgraph_once(app.world_mut(), nested_claim.entity())
        .expect("nested dynamic subgraph should evaluate in one imperative call without stale levels");
    let nested_val = downcast_signal_value::<i32>(nested_boxed).expect("nested output should downcast to i32");
    assert_eq!(nested_val, 176);

    let replay_probe = replay_dyn_shared.register(app.world_mut());
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), replay_probe.entity()),
        Some(ReplayCacheInspection::Ready {
            current: 88,
            revision: 1,
        })
    );

    replay_probe.release(app.world_mut());
    nested_claim.release(app.world_mut());
    app.update();
}
