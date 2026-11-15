#![allow(missing_docs)]

extern crate alloc;

use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};
use bevy_app::App;
use bevy_ecs::prelude::*;
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::graph::{BoxedSignalValue, downcast_signal_value, run_signal_subgraph_once};
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand, MapSignalMaterializer,
    OutputDelivery, ReplayCacheInspection, ReplayInitialization, SameSignalIdentity, ScalarSignalMaterializer,
    SignalEntity, SignalRegistrationError, VecSignalMaterializer, inspect_map_replay_cache,
    inspect_scalar_replay_cache, inspect_vec_replay_cache, prelude::*,
};
use std::sync::Mutex;

#[derive(Debug, PartialEq, Eq)]
struct NonClone(String);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NonCloneKey(String);

struct CustomScalar<T: 'static>(ScalarSignalMaterializer<T>);

impl<T: 'static> Signal for CustomScalar<T> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.0
    }
}

#[cfg(feature = "imperative-signal-evaluation")]
struct CustomVec<T: 'static>(VecSignalMaterializer<T>);

#[cfg(feature = "imperative-signal-evaluation")]
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

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app
}

fn sorted_world_entities(world: &mut World) -> Vec<Entity> {
    let mut entities = world.query::<Entity>().iter(world).collect::<Vec<_>>();
    entities.sort();
    entities
}

// ============================================================================
// 1. Custom Move-Only (`NonClone`) Scalar, SignalVec, and SignalMap Chains
// ============================================================================

struct MoveOnlyScalarChain {
    queue: Arc<Mutex<VecDeque<NonClone>>>,
    source_ptrs: Arc<Mutex<Vec<usize>>>,
}

impl Signal for MoveOnlyScalarChain {
    type Item = NonClone;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { queue, source_ptrs } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let source = registrar.signal_entity(
                (),
                move |In(())| {
                    let item = queue.lock().expect("queue lock").pop_front()?;
                    source_ptrs.lock().expect("ptr lock").push(item.0.as_ptr() as usize);
                    Some(item)
                },
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(source, |In(item): In<NonClone>| Some(item), OutputDelivery::move_only())
        })
    }
}

struct MoveOnlyVecChain {
    batches: Arc<Mutex<VecDeque<Vec<VecDiff<NonClone>>>>>,
}

impl SignalVec for MoveOnlyVecChain {
    type Item = NonClone;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        let batches = self.batches;
        VecSignalMaterializer::new(move |registrar| {
            let source = registrar.signal_vec_entity(
                (),
                move |In(())| batches.lock().expect("batches lock").pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_vec_entity(
                source,
                |In(diffs): In<Vec<VecDiff<NonClone>>>| {
                    let mapped = diffs
                        .into_iter()
                        .map(|diff| {
                            let map_item = |mut item: NonClone| {
                                item.0.push_str("-mapped");
                                item
                            };
                            match diff {
                                VecDiff::Replace { values } => VecDiff::Replace {
                                    values: values.into_iter().map(map_item).collect(),
                                },
                                VecDiff::InsertAt { index, value } => VecDiff::InsertAt {
                                    index,
                                    value: map_item(value),
                                },
                                VecDiff::UpdateAt { index, value } => VecDiff::UpdateAt {
                                    index,
                                    value: map_item(value),
                                },
                                VecDiff::RemoveAt { index } => VecDiff::RemoveAt { index },
                                VecDiff::Move { old_index, new_index } => VecDiff::Move { old_index, new_index },
                                VecDiff::Push { value } => VecDiff::Push { value: map_item(value) },
                                VecDiff::Pop => VecDiff::Pop,
                                VecDiff::Clear => VecDiff::Clear,
                            }
                        })
                        .collect::<Vec<_>>();
                    Some(mapped)
                },
                OutputDelivery::move_only(),
            )
        })
    }
}

type NonCloneMapBatch = Vec<MapDiff<NonCloneKey, NonClone>>;

struct MoveOnlyMapChain {
    batches: Arc<Mutex<VecDeque<NonCloneMapBatch>>>,
}

impl SignalMap for MoveOnlyMapChain {
    type Key = NonCloneKey;
    type Value = NonClone;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        let batches = self.batches;
        MapSignalMaterializer::new(move |registrar| {
            let source = registrar.signal_map_entity(
                (),
                move |In(())| batches.lock().expect("batches lock").pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.signal_map_entity(
                source,
                |In(diffs): In<Vec<MapDiff<NonCloneKey, NonClone>>>| {
                    let mapped = diffs
                        .into_iter()
                        .map(|diff| {
                            diff.map_value(|mut value| {
                                value.0.push_str("-mapped");
                                value
                            })
                        })
                        .collect::<Vec<_>>();
                    Some(mapped)
                },
                OutputDelivery::move_only(),
            )
        })
    }
}

#[test]
fn custom_move_only_non_clone_scalar_vec_and_map_chains_execute_via_schedule() {
    let mut app = test_app();

    // 1a. Move-only scalar chain preserves exact heap allocation without Clone.
    let scalar_queue = Arc::new(Mutex::new(VecDeque::from([
        NonClone(String::from("alpha")),
        NonClone(String::from("beta")),
    ])));
    let source_ptrs = Arc::new(Mutex::new(Vec::new()));
    let observed_scalars = Arc::new(Mutex::new(Vec::<(NonClone, usize)>::new()));
    let scalar_claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let chain = MoveOnlyScalarChain {
            queue: scalar_queue,
            source_ptrs: source_ptrs.clone(),
        };
        let observed_scalars = observed_scalars.clone();
        move |registrar| {
            let upstream = registrar.materialize_signal(Box::new(chain).into_signal_materializer())?;
            registrar.signal_entity(
                upstream,
                move |In(value): In<NonClone>| {
                    let ptr = value.0.as_ptr() as usize;
                    observed_scalars.lock().expect("observed lock").push((value, ptr));
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    app.update();
    app.update();
    app.update();

    {
        let observed = observed_scalars.lock().expect("observed lock");
        let ptrs = source_ptrs.lock().expect("ptr lock");
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[0].0, NonClone(String::from("alpha")));
        assert_eq!(observed[1].0, NonClone(String::from("beta")));
        assert_eq!(
            [observed[0].1, observed[1].1],
            [ptrs[0], ptrs[1]],
            "move-only scalar transport must preserve the exact heap allocation"
        );
    }
    scalar_claim.release(app.world_mut());

    // 1b. Move-only SignalVec chain reconstructs exact state without Clone.
    let vec_batches = Arc::new(Mutex::new(VecDeque::from([
        vec![VecDiff::Replace {
            values: vec![NonClone(String::from("v0")), NonClone(String::from("v1"))],
        }],
        vec![
            VecDiff::Push {
                value: NonClone(String::from("v2")),
            },
            VecDiff::RemoveAt { index: 0 },
        ],
    ])));
    let reconstructed_vec = Arc::new(Mutex::new(Vec::<NonClone>::new()));
    let vec_claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let chain = MoveOnlyVecChain { batches: vec_batches };
        let reconstructed_vec = reconstructed_vec.clone();
        move |registrar| {
            let upstream = registrar.materialize_signal_vec(Box::new(chain).into_signal_vec_materializer())?;
            registrar.signal_entity(
                upstream,
                move |In(diffs): In<Vec<VecDiff<NonClone>>>| {
                    let mut state = reconstructed_vec.lock().expect("vec state lock");
                    for diff in diffs {
                        diff.apply_to_vec(&mut state);
                    }
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    app.update();
    app.update();

    assert_eq!(
        *reconstructed_vec.lock().expect("vec state lock"),
        vec![NonClone(String::from("v1-mapped")), NonClone(String::from("v2-mapped")),]
    );
    vec_claim.release(app.world_mut());

    // 1c. Move-only SignalMap chain reconstructs exact keyed state without Clone on key or value.
    let map_batches = Arc::new(Mutex::new(VecDeque::from([
        vec![MapDiff::Replace {
            entries: vec![
                (NonCloneKey(String::from("k1")), NonClone(String::from("one"))),
                (NonCloneKey(String::from("k2")), NonClone(String::from("two"))),
            ],
        }],
        vec![
            MapDiff::Update {
                key: NonCloneKey(String::from("k1")),
                value: NonClone(String::from("uno")),
            },
            MapDiff::Remove {
                key: NonCloneKey(String::from("k2")),
            },
            MapDiff::Insert {
                key: NonCloneKey(String::from("k3")),
                value: NonClone(String::from("three")),
            },
        ],
    ])));
    let reconstructed_map = Arc::new(Mutex::new(BTreeMap::<NonCloneKey, NonClone>::new()));
    let map_claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let chain = MoveOnlyMapChain { batches: map_batches };
        let reconstructed_map = reconstructed_map.clone();
        move |registrar| {
            let upstream = registrar.materialize_signal_map(Box::new(chain).into_signal_map_materializer())?;
            registrar.signal_entity(
                upstream,
                move |In(diffs): In<Vec<MapDiff<NonCloneKey, NonClone>>>| {
                    let mut state = reconstructed_map.lock().expect("map state lock");
                    for diff in diffs {
                        match diff {
                            MapDiff::Replace { entries } => {
                                *state = entries.into_iter().collect();
                            }
                            MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                                state.insert(key, value);
                            }
                            MapDiff::Remove { key } => {
                                state.remove(&key);
                            }
                            MapDiff::Clear => state.clear(),
                        }
                    }
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    app.update();
    app.update();

    assert_eq!(
        *reconstructed_map.lock().expect("map state lock"),
        BTreeMap::from([
            (NonCloneKey(String::from("k1")), NonClone(String::from("uno-mapped"))),
            (NonCloneKey(String::from("k3")), NonClone(String::from("three-mapped"))),
        ])
    );
    map_claim.release(app.world_mut());
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn custom_move_only_non_clone_chains_work_with_imperative_subgraph_evaluation() {
    let mut app = test_app();

    // Scalar NonClone terminal evaluated imperatively with zero clones.
    let scalar_queue = Arc::new(Mutex::new(VecDeque::from([NonClone(String::from(
        "imperative-scalar",
    ))])));
    let source_ptrs = Arc::new(Mutex::new(Vec::new()));
    let scalar_claim = MoveOnlyScalarChain {
        queue: scalar_queue,
        source_ptrs: source_ptrs.clone(),
    }
    .register(app.world_mut());

    let boxed: BoxedSignalValue =
        run_signal_subgraph_once(app.world_mut(), scalar_claim.entity()).expect("scalar output");
    let value = downcast_signal_value::<NonClone>(boxed).expect("downcast NonClone");
    assert_eq!(value, NonClone(String::from("imperative-scalar")));
    assert_eq!(
        value.0.as_ptr() as usize,
        source_ptrs.lock().expect("ptr lock")[0],
        "run_signal_subgraph_once must move terminal NonClone output without cloning"
    );
    assert!(run_signal_subgraph_once(app.world_mut(), scalar_claim.entity()).is_none());
    scalar_claim.release(app.world_mut());

    // SignalVec NonClone terminal concatenates multiple accepted batches by move.
    let trigger = CustomScalar(ScalarSignalMaterializer::<usize>::new(|registrar| {
        registrar.signal_entity(
            (),
            |In(()), mut seq: Local<usize>| {
                *seq += 1;
                Some(*seq)
            },
            OutputDelivery::move_only(),
        )
    }))
    .broadcast();
    let trigger_for_vec = trigger.clone();
    let root_claim = trigger.register(app.world_mut());
    let vec_claim = CustomVec(VecSignalMaterializer::<NonClone>::new(move |registrar| {
        let trigger = registrar.materialize_signal(Box::new(trigger_for_vec).into_signal_materializer())?;
        registrar.signal_vec_entity(
            trigger,
            |In(step): In<usize>| {
                Some(vec![VecDiff::Push {
                    value: NonClone(alloc::format!("item-{step}")),
                }])
            },
            OutputDelivery::move_only(),
        )
    }))
    .register(app.world_mut());

    // Step only the root twice to queue two inputs into the NonClone vector node, then step the
    // vector subgraph (which runs the root a third time and concatenates all three NonClone batches
    // by move).
    assert_eq!(
        run_signal_subgraph_once(app.world_mut(), root_claim.entity()).and_then(downcast_signal_value::<usize>),
        Some(1)
    );
    assert_eq!(
        run_signal_subgraph_once(app.world_mut(), root_claim.entity()).and_then(downcast_signal_value::<usize>),
        Some(2)
    );
    let vec_diffs = run_signal_subgraph_once(app.world_mut(), vec_claim.entity())
        .and_then(downcast_signal_value::<Vec<VecDiff<NonClone>>>)
        .expect("concatenated move-only vec diffs");
    let mut vec_state = Vec::new();
    for diff in vec_diffs {
        diff.apply_to_vec(&mut vec_state);
    }
    assert_eq!(
        vec_state,
        vec![
            NonClone(String::from("item-1")),
            NonClone(String::from("item-2")),
            NonClone(String::from("item-3")),
        ]
    );
    vec_claim.release(app.world_mut());
    root_claim.release(app.world_mut());

    // SignalMap NonClone terminal evaluated imperatively.
    let map_batches = Arc::new(Mutex::new(VecDeque::from([vec![MapDiff::Insert {
        key: NonCloneKey(String::from("mk")),
        value: NonClone(String::from("mv")),
    }]])));
    let map_claim = MoveOnlyMapChain { batches: map_batches }.register(app.world_mut());
    let map_diffs = run_signal_subgraph_once(app.world_mut(), map_claim.entity())
        .and_then(downcast_signal_value::<Vec<MapDiff<NonCloneKey, NonClone>>>)
        .expect("move-only map diffs");
    assert_eq!(map_diffs.len(), 1);
    assert!(matches!(
        &map_diffs[0],
        MapDiff::Insert { key, value }
            if key == &NonCloneKey(String::from("mk")) && value == &NonClone(String::from("mv-mapped"))
    ));
    map_claim.release(app.world_mut());
}

// ============================================================================
// 2. Exact Clone-Count Tests (`CloneCounter`)
// ============================================================================

#[derive(Debug)]
struct CloneCounter {
    payload: String,
    clones: Arc<AtomicUsize>,
    is_clone: bool,
}

impl CloneCounter {
    fn new(payload: impl Into<String>, clones: Arc<AtomicUsize>) -> Self {
        Self {
            payload: payload.into(),
            clones,
            is_clone: false,
        }
    }
}

impl Clone for CloneCounter {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            payload: self.payload.clone(),
            clones: self.clones.clone(),
            is_clone: true,
        }
    }
}

#[test]
fn broadcast_with_one_active_dependent_moves_with_zero_clones() {
    let mut app = test_app();
    let clones = Arc::new(AtomicUsize::new(0));
    let original_ptr = Arc::new(AtomicUsize::new(0));
    let source = CustomScalar(ScalarSignalMaterializer::<CloneCounter>::new({
        let clones = clones.clone();
        let original_ptr = original_ptr.clone();
        move |registrar| {
            registrar.signal_entity(
                (),
                move |In(()), mut emitted: Local<bool>| {
                    if core::mem::replace(&mut *emitted, true) {
                        return None;
                    }
                    let item = CloneCounter::new("single-recipient", clones.clone());
                    original_ptr.store(item.payload.as_ptr() as usize, Ordering::SeqCst);
                    Some(item)
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .broadcast();

    // Materialize a dormant (count-0) dependent as well to prove inactive dependents cause 0
    // clones.
    let dormant_branch = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let source = source.clone();
        move |registrar| {
            let source = registrar.materialize_signal(Box::new(source).into_signal_materializer())?;
            registrar.signal_entity(
                source,
                |In(_): In<CloneCounter>| None::<()>,
                OutputDelivery::move_only(),
            )
        }
    }))
    .broadcast();
    let dormant_claim = dormant_branch.clone().register(app.world_mut());
    dormant_claim.release(app.world_mut());

    let observed = Arc::new(Mutex::new(Vec::<(String, bool, usize)>::new()));
    let active_claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let observed = observed.clone();
        move |registrar| {
            let source = registrar.materialize_signal(Box::new(source).into_signal_materializer())?;
            registrar.signal_entity(
                source,
                move |In(item): In<CloneCounter>| {
                    let ptr = item.payload.as_ptr() as usize;
                    observed
                        .lock()
                        .expect("observed lock")
                        .push((item.payload, item.is_clone, ptr));
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    app.update();

    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "broadcast entity with 1 live active dependent must perform 0 clones"
    );
    let recorded = observed.lock().expect("observed lock");
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].0, "single-recipient");
    assert!(!recorded[0].1, "single recipient must receive the original moved value");
    assert_eq!(recorded[0].2, original_ptr.load(Ordering::SeqCst));

    active_claim.release(app.world_mut());
    drop(dormant_branch);
}

#[test]
fn broadcast_with_n_dependents_clones_n_minus_one_times_and_moves_original_to_highest_entity() {
    for n in [2usize, 3, 4] {
        let mut app = test_app();
        let clones = Arc::new(AtomicUsize::new(0));
        let original_ptr = Arc::new(AtomicUsize::new(0));
        let source = CustomScalar(ScalarSignalMaterializer::<CloneCounter>::new({
            let clones = clones.clone();
            let original_ptr = original_ptr.clone();
            move |registrar| {
                registrar.signal_entity(
                    (),
                    move |In(()), mut emitted: Local<bool>| {
                        if core::mem::replace(&mut *emitted, true) {
                            return None;
                        }
                        let item = CloneCounter::new(alloc::format!("fanout-{n}"), clones.clone());
                        original_ptr.store(item.payload.as_ptr() as usize, Ordering::SeqCst);
                        Some(item)
                    },
                    OutputDelivery::move_only(),
                )
            }
        }))
        .broadcast();

        // Also keep a dormant count-0 materialized dependent to verify it is excluded from N.
        let dormant = CustomScalar(ScalarSignalMaterializer::<()>::new({
            let source = source.clone();
            move |registrar| {
                let source = registrar.materialize_signal(Box::new(source).into_signal_materializer())?;
                registrar.signal_entity(
                    source,
                    |In(_): In<CloneCounter>| None::<()>,
                    OutputDelivery::move_only(),
                )
            }
        }))
        .broadcast();
        let dormant_claim = dormant.clone().register(app.world_mut());
        dormant_claim.release(app.world_mut());

        let deliveries = Arc::new(Mutex::new(BTreeMap::<SignalEntity, (String, bool, usize)>::new()));
        let mut claims = Vec::with_capacity(n);
        for _ in 0..n {
            let slot_entity = Arc::new(Mutex::new(None::<SignalEntity>));
            let claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
                let source = source.clone();
                let deliveries = deliveries.clone();
                let slot_entity = slot_entity.clone();
                move |registrar| {
                    let source = registrar.materialize_signal(Box::new(source).into_signal_materializer())?;
                    registrar.signal_entity(
                        source,
                        move |In(item): In<CloneCounter>| {
                            let entity = slot_entity
                                .lock()
                                .expect("slot lock")
                                .expect("entity recorded before update");
                            let ptr = item.payload.as_ptr() as usize;
                            deliveries
                                .lock()
                                .expect("deliveries lock")
                                .insert(entity, (item.payload, item.is_clone, ptr));
                            None::<()>
                        },
                        OutputDelivery::move_only(),
                    )
                }
            }))
            .register(app.world_mut());
            *slot_entity.lock().expect("slot lock") = Some(claim.entity());
            claims.push(claim);
        }

        let highest_entity = claims
            .iter()
            .map(SignalClaim::entity)
            .max()
            .expect("at least 2 dependents");

        app.update();

        assert_eq!(
            clones.load(Ordering::SeqCst),
            n - 1,
            "broadcast to {n} live active dependents must perform exactly {n} - 1 clones"
        );
        let deliveries = deliveries.lock().expect("deliveries lock");
        assert_eq!(deliveries.len(), n);
        for (&entity, (payload, is_clone, ptr)) in deliveries.iter() {
            assert_eq!(payload, &alloc::format!("fanout-{n}"));
            if entity == highest_entity {
                assert!(
                    !is_clone,
                    "highest-ordered entity {entity:?} must receive the moved original"
                );
                assert_eq!(
                    *ptr,
                    original_ptr.load(Ordering::SeqCst),
                    "highest-ordered entity must receive the original heap buffer"
                );
            } else {
                assert!(
                    *is_clone,
                    "non-final sorted entity {entity:?} must receive a cloned value"
                );
            }
        }

        for claim in claims {
            claim.release(app.world_mut());
        }
        drop(dormant);
    }
}

struct NonReplayVecSource {
    batches: Arc<Mutex<VecDeque<Vec<VecDiff<CloneCounter>>>>>,
}

impl SignalVec for NonReplayVecSource {
    type Item = CloneCounter;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        let batches = self.batches;
        VecSignalMaterializer::new(move |registrar| {
            registrar.signal_vec_entity(
                (),
                move |In(())| batches.lock().expect("batches lock").pop_front(),
                OutputDelivery::move_only(),
            )
        })
    }
}

#[test]
fn ordered_single_dependent_chain_taggers_perform_zero_clones() {
    let mut app = test_app();
    let clones = Arc::new(AtomicUsize::new(0));

    let left_item = CloneCounter::new("left-0", clones.clone());
    let right_item = CloneCounter::new("right-0", clones.clone());
    let left_ptr = left_item.payload.as_ptr() as usize;
    let right_ptr = right_item.payload.as_ptr() as usize;

    let left = NonReplayVecSource {
        batches: Arc::new(Mutex::new(VecDeque::from([vec![VecDiff::Push { value: left_item }]]))),
    };
    let right = NonReplayVecSource {
        batches: Arc::new(Mutex::new(VecDeque::from([vec![VecDiff::Push { value: right_item }]]))),
    };

    let chained = left.chain(right);
    let received = Arc::new(Mutex::new(Vec::<(String, bool, usize)>::new()));
    let claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let received = received.clone();
        move |registrar| {
            let chained = registrar.materialize_signal_vec(Box::new(chained).into_signal_vec_materializer())?;
            registrar.signal_entity(
                chained,
                move |In(diffs): In<Vec<VecDiff<CloneCounter>>>| {
                    let mut log = received.lock().expect("received lock");
                    for diff in diffs {
                        match diff {
                            VecDiff::Push { value } | VecDiff::InsertAt { value, .. } => {
                                let ptr = value.payload.as_ptr() as usize;
                                log.push((value.payload, value.is_clone, ptr));
                            }
                            other => panic!("unexpected diff: {other:?}"),
                        }
                    }
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    app.update();

    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "ordered single-dependent chain taggers must move items with 0 clones"
    );
    let mut items = received.lock().expect("received lock").clone();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        items,
        vec![
            (String::from("left-0"), false, left_ptr),
            (String::from("right-0"), false, right_ptr),
        ]
    );

    claim.release(app.world_mut());
}

// ============================================================================
// 3. Fan-Out Preflight Rejection (`MoveOnlyFanout`) & Controls
// ============================================================================

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn second_static_dependent_on_move_only_entity_rejects_atomically_with_move_only_fanout() {
    let mut app = test_app();

    // 3a-i: Diamond within one materialization plan sharing a pending move_only() source.
    let drops = Arc::new(AtomicUsize::new(0));
    let drop_probe = DropCounter(drops.clone());
    let entities_before = sorted_world_entities(app.world_mut());
    let diamond = CustomScalar(ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let source = registrar.signal_entity(
            (),
            move |In(())| {
                let _keep = &drop_probe;
                Some(1)
            },
            OutputDelivery::move_only(),
        )?;
        let left = registrar.signal_entity(source, |In(v): In<i32>| Some(v + 1), OutputDelivery::move_only())?;
        let right = registrar.signal_entity(source, |In(v): In<i32>| Some(v + 2), OutputDelivery::move_only())?;
        registrar.signal_entity([left, right], |In(v): In<i32>| Some(v), OutputDelivery::move_only())
    }));

    assert_eq!(
        diamond.try_register(app.world_mut()).err(),
        Some(SignalRegistrationError::MoveOnlyFanout)
    );
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "rejected candidate system must be dropped immediately without world allocation"
    );
    assert_eq!(sorted_world_entities(app.world_mut()), entities_before);

    // 3a-ii: Rejected downstream intra-plan move_only() fan-out leaves an already-active
    // upstream untouched. (`Existing` move_only() entity fan-out backstops are covered in
    // `src/registration.rs` via crate-internal `shared_clone()`.)
    let shared_source = CustomScalar(ScalarSignalMaterializer::<i32>::new(|registrar| {
        registrar.signal_entity((), |In(())| Some(10), OutputDelivery::move_only())
    }))
    .broadcast();
    let first_values = Arc::new(Mutex::new(Vec::new()));
    let first_claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let shared_source = shared_source.clone();
        let first_values = first_values.clone();
        move |registrar| {
            let source = registrar.materialize_signal(Box::new(shared_source).into_signal_materializer())?;
            registrar.signal_entity(
                source,
                move |In(v): In<i32>| {
                    first_values.lock().expect("first lock").push(v);
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    let entities_after_first = sorted_world_entities(app.world_mut());
    let second_drops = Arc::new(AtomicUsize::new(0));
    let second_probe = DropCounter(second_drops.clone());
    let second_consumer = CustomScalar(ScalarSignalMaterializer::<i32>::new({
        let shared_source = shared_source.clone();
        move |registrar| {
            let source = registrar.materialize_signal(Box::new(shared_source).into_signal_materializer())?;
            let linear_mid = registrar.signal_entity(source, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
            let left = registrar.signal_entity(linear_mid, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
            let right = registrar.signal_entity(
                linear_mid,
                move |In(v): In<i32>| {
                    let _keep = &second_probe;
                    Some(v)
                },
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity([left, right], |In(v): In<i32>| Some(v), OutputDelivery::move_only())
        }
    }));

    assert_eq!(
        second_consumer.try_register(app.world_mut()).err(),
        Some(SignalRegistrationError::MoveOnlyFanout)
    );
    assert_eq!(second_drops.load(Ordering::SeqCst), 1);
    assert_eq!(sorted_world_entities(app.world_mut()), entities_after_first);

    app.update();
    assert_eq!(*first_values.lock().expect("first lock"), vec![10]);

    first_claim.release(app.world_mut());
    drop(shared_source);
}

#[test]
fn second_pending_replay_consumer_on_warming_move_only_entity_rejects_immediately() {
    // Public intra-plan Warming move_only() fan-out rejection (`Existing` Warming replay fan-out
    // backstop is covered in `src/registration.rs` via crate-internal `shared_clone()`).
    let mut app = test_app();

    let entities_before = sorted_world_entities(app.world_mut());
    let b_drops = Arc::new(AtomicUsize::new(0));
    let b_probe = DropCounter(b_drops.clone());
    let rejected_warming_fanout = CustomScalar(ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let warming = registrar.replayable_signal_entity(
            (),
            |In(())| Some(77),
            ReplayInitialization::Warming,
            OutputDelivery::move_only(),
        )?;
        let consumer_a = registrar.signal_entity(warming, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        let consumer_b = registrar.signal_entity(
            warming,
            move |In(v): In<i32>| {
                let _keep = &b_probe;
                Some(v)
            },
            OutputDelivery::move_only(),
        )?;
        registrar.signal_entity(
            [consumer_a, consumer_b],
            |In(v): In<i32>| Some(v),
            OutputDelivery::move_only(),
        )
    }));
    assert_eq!(
        rejected_warming_fanout.try_register(app.world_mut()).err(),
        Some(SignalRegistrationError::MoveOnlyFanout)
    );
    assert_eq!(b_drops.load(Ordering::SeqCst), 1);
    assert_eq!(sorted_world_entities(app.world_mut()), entities_before);

    // Single consumer on a Warming move_only() replayable entity succeeds and warms on update.
    let consumer_a_values = Arc::new(Mutex::new(Vec::new()));
    let consumer_a = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let consumer_a_values = consumer_a_values.clone();
        move |registrar| {
            let source = registrar.replayable_signal_entity(
                (),
                |In(()), mut emitted: Local<bool>| {
                    if core::mem::replace(&mut *emitted, true) {
                        None
                    } else {
                        Some(77)
                    }
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )?;
            registrar.signal_entity(
                source,
                move |In(v): In<i32>| {
                    consumer_a_values.lock().expect("a lock").push(v);
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    // Scheduled execution warms the cache and delivers cleanly to Consumer A.
    app.update();
    assert_eq!(*consumer_a_values.lock().expect("a lock"), vec![77]);

    consumer_a.release(app.world_mut());
}

#[test]
fn dynamic_operator_move_only_fanout_rejects_candidate_atomically_and_preserves_old_generation() {
    // Public dynamic candidate fan-out rejection (`Existing` cross-slot and external-vs-dynamic
    // move_only() collision backstops are covered in `src/registration.rs`).
    type SelectedSignal = BoxedReplayableSignal<i32>;
    type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<SelectedSignal>, (usize, i32)>;

    let mut app = test_app();

    // First verify `.output_delivery(OutputDelivery::move_only())` on `DynamicSignalOperatorSpec`
    // rejects a second consumer on the dynamic stable output itself!
    let rejected_dynamic_output_fanout = CustomScalar(ScalarSignalMaterializer::<()>::new(|registrar| {
        let manager = registrar.signal_entity((), |In(())| None::<IndexedUpdate>, OutputDelivery::move_only())?;
        let owner = registrar.dynamic_operator(
            DynamicSignalOperatorSpec::indexed(manager)
                .replay_scalar::<i32>()
                .adapter(|index, value| DynamicEmissions::One((index, value)))
                .scalar_output(Some)
                .output_delivery(OutputDelivery::move_only())
                .same_signal_identity(SameSignalIdentity::KeepCurrent),
        )?;
        let left = registrar.signal_entity(
            owner,
            |In(_): In<Vec<(usize, i32)>>| Some(()),
            OutputDelivery::move_only(),
        )?;
        let right = registrar.signal_entity(
            owner,
            |In(_): In<Vec<(usize, i32)>>| Some(()),
            OutputDelivery::move_only(),
        )?;
        registrar.signal_entity([left, right], |In(())| None::<()>, OutputDelivery::move_only())
    }));
    assert_eq!(
        rejected_dynamic_output_fanout.try_register(app.world_mut()).err(),
        Some(SignalRegistrationError::MoveOnlyFanout)
    );

    let initial_queue = Arc::new(Mutex::new(VecDeque::new()));
    let initial = CustomScalar(ScalarSignalMaterializer::new({
        let initial_queue = initial_queue.clone();
        move |registrar| {
            registrar.replayable_signal_source(
                10,
                move |In(())| initial_queue.lock().expect("initial queue").pop_front(),
                OutputDelivery::move_only(),
            )
        }
    }))
    .cache()
    .boxed_replayable();

    let invalid_ready_fanout_candidate = CustomScalar(ScalarSignalMaterializer::new(|registrar| {
        let source = registrar.replayable_signal_source(99, |In(())| None::<i32>, OutputDelivery::move_only())?;
        let left = registrar.signal_entity(source, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        let right = registrar.signal_entity(source, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        registrar.replayable_signal_entity(
            [left, right],
            |In(v): In<i32>| Some(v),
            ReplayInitialization::Warming,
            OutputDelivery::move_only(),
        )
    }))
    .cache()
    .boxed_replayable();

    let invalid_warming_fanout_candidate = CustomScalar(ScalarSignalMaterializer::new(|registrar| {
        let warming = registrar.replayable_signal_entity(
            (),
            |In(())| None::<i32>,
            ReplayInitialization::Warming,
            OutputDelivery::move_only(),
        )?;
        let left = registrar.signal_entity(warming, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        let right = registrar.signal_entity(warming, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        registrar.replayable_signal_entity(
            [left, right],
            |In(v): In<i32>| Some(v),
            ReplayInitialization::Warming,
            OutputDelivery::move_only(),
        )
    }))
    .cache()
    .boxed_replayable();

    let updates = Arc::new(Mutex::new(VecDeque::from([
        // Frame 1: valid single-slot selection of `initial` (generation 1).
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll { signals: vec![initial] }],
            DynamicEmissions::None,
        ),
        // Frame 2: candidate with internal move_only() fan-out rejected atomically.
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![invalid_ready_fanout_candidate],
            }],
            DynamicEmissions::One((99, 999)),
        ),
        // Frame 3: candidate with warming move_only() replay fan-out rejected atomically.
        IndexedUpdate::new(
            [IndexedSlotCommand::ReplaceAll {
                signals: vec![invalid_warming_fanout_candidate],
            }],
            DynamicEmissions::One((98, 998)),
        ),
    ])));

    let candidate_errors = Arc::new(Mutex::new(Vec::<SignalRegistrationError>::new()));
    let delivered = Arc::new(Mutex::new(Vec::<Vec<(usize, i32)>>::new()));
    let observer_claim = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let updates = updates.clone();
        let candidate_errors = candidate_errors.clone();
        let delivered = delivered.clone();
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| updates.lock().expect("updates lock").pop_front(),
                OutputDelivery::move_only(),
            )?;
            let owner = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .scalar_output(Some)
                    .output_delivery(OutputDelivery::move_only())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        candidate_errors.lock().expect("errors lock").push(error);
                    }),
            )?;
            registrar.signal_entity(
                owner,
                move |In(batch): In<Vec<(usize, i32)>>| {
                    delivered.lock().expect("delivered lock").push(batch);
                    None::<()>
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    // Frame 1 establishes generation 1 (initial snapshot 10).
    app.update();
    assert_eq!(*delivered.lock().expect("delivered lock"), vec![vec![(0, 10)]]);
    assert!(candidate_errors.lock().expect("errors lock").is_empty());

    let entities_after_gen1 = sorted_world_entities(app.world_mut());

    // Frame 2 rejects move-only fan-out candidate atomically.
    app.update();
    assert_eq!(
        *candidate_errors.lock().expect("errors lock"),
        vec![SignalRegistrationError::MoveOnlyFanout]
    );
    assert_eq!(sorted_world_entities(app.world_mut()), entities_after_gen1);

    // Frame 3 rejects warming move-only fan-out candidate atomically.
    app.update();
    assert_eq!(
        *candidate_errors.lock().expect("errors lock"),
        vec![
            SignalRegistrationError::MoveOnlyFanout,
            SignalRegistrationError::MoveOnlyFanout,
        ]
    );

    // Generation 1 (`initial`) is preserved and still delivers new emissions!
    initial_queue.lock().expect("initial queue").push_back(30);
    app.update();
    assert_eq!(
        *delivered.lock().expect("delivered lock"),
        vec![vec![(0, 10)], vec![(0, 30)]]
    );

    observer_claim.release(app.world_mut());
}

#[test]
fn move_only_controls_allow_repeated_claims_duplicate_static_deps_and_dormant_dependents() {
    // Public duplicate `[source, source]`, repeated direct claims (`1 -> 2 -> 3`), and dormant
    // reactivation controls (`Existing` dormant sibling dependent coexistence is covered in
    // `src/registration.rs`).
    let mut app = test_app();

    // Active dependent uses duplicate `[source, source]` static dependencies on a move_only()
    // source, and its broadcast terminal acquires 3 direct claims (`1 -> 2 -> 3`) without
    // triggering MoveOnlyFanout on `source` or `active_dependent`.
    let values = Arc::new(Mutex::new(Vec::new()));
    let active_dependent = CustomScalar(ScalarSignalMaterializer::<()>::new({
        let values = values.clone();
        move |registrar| {
            let source = registrar.signal_entity((), |In(())| Some(5), OutputDelivery::move_only())?;
            registrar.signal_entity(
                [source, source],
                move |In(v): In<i32>| {
                    values.lock().expect("values lock").push(v);
                    Some(())
                },
                OutputDelivery::move_only(),
            )
        }
    }))
    .broadcast();

    let claim_1 = active_dependent
        .clone()
        .try_register(app.world_mut())
        .expect("1 active dependent with canonicalized [s, s] must succeed");
    let claim_2 = active_dependent
        .clone()
        .try_register(app.world_mut())
        .expect("second direct claim (1 -> 2) on same terminal must succeed");
    let claim_3 = active_dependent
        .clone()
        .try_register(app.world_mut())
        .expect("third direct claim (2 -> 3) on same terminal must succeed");
    assert_eq!(claim_1.entity(), claim_2.entity());
    assert_eq!(claim_2.entity(), claim_3.entity());

    app.update();
    assert_eq!(*values.lock().expect("values lock"), vec![5]);

    claim_3.release(app.world_mut());
    claim_2.release(app.world_mut());
    claim_1.release(app.world_mut());

    // Dormant retained chain can reactivate cleanly after balanced release.
    let reactivated = active_dependent
        .try_register(app.world_mut())
        .expect("dormant retained chain must reactivate cleanly");
    reactivated.release(app.world_mut());
}

// ============================================================================
// 4. Canonical `OutputDeliveryKey` Mismatch Rejects With `ConflictingCanonicalState`
// ============================================================================

#[test]
fn canonical_output_delivery_key_mismatch_rejects_with_conflicting_canonical_state() {
    // Establish a Ready single-entity chain whose cached canonical key is `(Broadcast, i32)`.
    let mut app_a = test_app();
    let broadcast_chain = CustomScalar(ScalarSignalMaterializer::<i32>::new(|registrar| {
        registrar.replayable_signal_source(1, |In(())| None::<i32>, OutputDelivery::broadcast())
    }))
    .cache()
    .broadcast();
    let claim_a = broadcast_chain.clone().register(app_a.world_mut());

    // Construct an entity at the identical generational Entity identity with `(Move, i32)`.
    let mut app_b = test_app();
    let move_claim = CustomScalar(ScalarSignalMaterializer::<i32>::new(|registrar| {
        registrar.signal_entity((), |In(())| Some(1), OutputDelivery::move_only())
    }))
    .register(app_b.world_mut());
    assert_eq!(claim_a.entity(), move_claim.entity());

    let entities_before = sorted_world_entities(app_b.world_mut());
    assert_eq!(
        broadcast_chain.clone().try_register(app_b.world_mut()).err(),
        Some(SignalRegistrationError::ConflictingCanonicalState),
        "rematerializing an entity whose OutputDeliveryKind differs must reject with ConflictingCanonicalState"
    );
    assert_eq!(sorted_world_entities(app_b.world_mut()), entities_before);

    // Also verify output_type mismatch (`(Broadcast, i32)` vs `(Broadcast, u64)`) rejects with
    // ConflictingCanonicalState.
    let mut app_c = test_app();
    let u64_claim = CustomScalar(ScalarSignalMaterializer::<u64>::new(|registrar| {
        registrar.replayable_signal_source(1_u64, |In(())| None::<u64>, OutputDelivery::broadcast())
    }))
    .register(app_c.world_mut());
    assert_eq!(claim_a.entity(), u64_claim.entity());

    let entities_before_c = sorted_world_entities(app_c.world_mut());
    assert_eq!(
        broadcast_chain.try_register(app_c.world_mut()).err(),
        Some(SignalRegistrationError::ConflictingCanonicalState),
        "rematerializing an entity whose output TypeId differs must reject with ConflictingCanonicalState"
    );
    assert_eq!(sorted_world_entities(app_c.world_mut()), entities_before_c);

    claim_a.release(app_a.world_mut());
    move_claim.release(app_b.world_mut());
    u64_claim.release(app_c.world_mut());
}

// ============================================================================
// 5. Public Non-Executing Replay Cache Inspection API
// ============================================================================

#[test]
fn public_replay_cache_inspection_reports_warming_and_ready_without_executing_systems() {
    let mut app = test_app();

    let scalar_runs = Arc::new(AtomicUsize::new(0));
    let vec_runs = Arc::new(AtomicUsize::new(0));
    let map_runs = Arc::new(AtomicUsize::new(0));

    let warming_scalar = CustomScalar(ScalarSignalMaterializer::<i32>::new({
        let scalar_runs = scalar_runs.clone();
        move |registrar| {
            registrar.replayable_signal_entity(
                (),
                move |In(()), mut local_counter: Local<i32>| {
                    scalar_runs.fetch_add(1, Ordering::SeqCst);
                    *local_counter += 10;
                    Some(*local_counter)
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    let warming_vec = VecSignalMaterializer::<i32>::new({
        let vec_runs = vec_runs.clone();
        move |registrar| {
            registrar.replayable_signal_vec_entity(
                (),
                move |In(()), mut step: Local<i32>| {
                    vec_runs.fetch_add(1, Ordering::SeqCst);
                    *step += 1;
                    if *step == 1 {
                        Some(vec![VecDiff::Replace { values: vec![1, 2] }])
                    } else {
                        Some(vec![VecDiff::Push { value: *step + 1 }])
                    }
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        }
    })
    .register(app.world_mut());

    let warming_map = CustomMap(MapSignalMaterializer::<i32, String>::new({
        let map_runs = map_runs.clone();
        move |registrar| {
            registrar.replayable_signal_map_entity(
                (),
                move |In(()), mut step: Local<i32>| {
                    map_runs.fetch_add(1, Ordering::SeqCst);
                    *step += 1;
                    if *step == 1 {
                        Some(vec![MapDiff::Replace {
                            entries: vec![(1, String::from("a"))],
                        }])
                    } else {
                        Some(vec![MapDiff::Insert {
                            key: *step,
                            value: alloc::format!("v{}", *step),
                        }])
                    }
                },
                ReplayInitialization::Warming,
                OutputDelivery::move_only(),
            )
        }
    }))
    .register(app.world_mut());

    let non_replayable = CustomScalar(ScalarSignalMaterializer::<i32>::new(|registrar| {
        registrar.signal_entity((), |In(())| Some(123), OutputDelivery::move_only())
    }))
    .register(app.world_mut());

    // Before scheduled execution: all three derived caches report Warming, 0 system runs!
    for _ in 0..3 {
        assert_eq!(
            inspect_scalar_replay_cache::<i32>(app.world(), warming_scalar.entity()),
            Some(ReplayCacheInspection::Warming)
        );
        assert_eq!(
            inspect_vec_replay_cache::<i32>(app.world(), warming_vec.entity()),
            Some(ReplayCacheInspection::Warming)
        );
        assert_eq!(
            inspect_map_replay_cache::<i32, String>(app.world(), warming_map.entity()),
            Some(ReplayCacheInspection::Warming)
        );
    }
    assert_eq!(scalar_runs.load(Ordering::SeqCst), 0);
    assert_eq!(vec_runs.load(Ordering::SeqCst), 0);
    assert_eq!(map_runs.load(Ordering::SeqCst), 0);

    // Non-replayable, wrong type, and scalar-vs-vec event-type collisions return None.
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), non_replayable.entity()),
        None
    );
    assert_eq!(
        inspect_scalar_replay_cache::<u64>(app.world(), warming_scalar.entity()),
        None
    );
    assert!(
        inspect_scalar_replay_cache::<Vec<VecDiff<i32>>>(app.world(), warming_vec.entity()).is_none(),
        "scalar inspection must not match a VecReplayCache even when T = Vec<VecDiff<i32>>"
    );
    assert_eq!(inspect_vec_replay_cache::<u64>(app.world(), warming_vec.entity()), None);
    assert_eq!(
        inspect_map_replay_cache::<i32, i32>(app.world(), warming_map.entity()),
        None
    );

    // First scheduled frame establishes Ready at revision 1.
    app.update();
    assert_eq!(scalar_runs.load(Ordering::SeqCst), 1);
    assert_eq!(vec_runs.load(Ordering::SeqCst), 1);
    assert_eq!(map_runs.load(Ordering::SeqCst), 1);

    for _ in 0..3 {
        assert_eq!(
            inspect_scalar_replay_cache::<i32>(app.world(), warming_scalar.entity()),
            Some(ReplayCacheInspection::Ready {
                current: 10,
                revision: 1,
            })
        );
        assert_eq!(
            inspect_vec_replay_cache::<i32>(app.world(), warming_vec.entity()),
            Some(ReplayCacheInspection::Ready {
                current: vec![1, 2],
                revision: 1,
            })
        );
        assert_eq!(
            inspect_map_replay_cache::<i32, String>(app.world(), warming_map.entity()),
            Some(ReplayCacheInspection::Ready {
                current: BTreeMap::from([(1, String::from("a"))]),
                revision: 1,
            })
        );
    }
    // Repeated inspections never run systems or advance Local state!
    assert_eq!(scalar_runs.load(Ordering::SeqCst), 1);
    assert_eq!(vec_runs.load(Ordering::SeqCst), 1);
    assert_eq!(map_runs.load(Ordering::SeqCst), 1);

    // Second scheduled frame advances to revision 2.
    app.update();
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), warming_scalar.entity()),
        Some(ReplayCacheInspection::Ready {
            current: 20,
            revision: 2,
        })
    );
    assert_eq!(
        inspect_vec_replay_cache::<i32>(app.world(), warming_vec.entity()),
        Some(ReplayCacheInspection::Ready {
            current: vec![1, 2, 3],
            revision: 2,
        })
    );
    assert_eq!(
        inspect_map_replay_cache::<i32, String>(app.world(), warming_map.entity()),
        Some(ReplayCacheInspection::Ready {
            current: BTreeMap::from([(1, String::from("a")), (2, String::from("v2"))]),
            revision: 2,
        })
    );

    warming_scalar.release(app.world_mut());
    warming_vec.release(app.world_mut());
    warming_map.release(app.world_mut());
    non_replayable.release(app.world_mut());
}
