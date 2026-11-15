//! Integration coverage for Move-First Checkpoint 2:
//! explicit `.broadcast()` wrappers, `.cache()` / `.broadcast()` ordering and normalization,
//! linear chain consumption, move-only boxed traits, and shared `MutableVec` / `MutableBTreeMap`
//! source terminals.

extern crate alloc;

use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec, vec::Vec};
use bevy_app::{App, Last, Update};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel, system::SystemIdMarker};
use core::sync::atomic::{AtomicUsize, Ordering};
use jonmo::{
    MapSignalMaterializer, OutputDelivery, ReplayCacheInspection, ScalarSignalMaterializer, SignalRegistrationError,
    VecSignalMaterializer, inspect_map_replay_cache, inspect_scalar_replay_cache, inspect_vec_replay_cache, prelude::*,
    signal, signal_map::MapDiff, signal_vec::VecDiff,
};
use std::sync::Mutex;

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app
}

fn system_entity_count(world: &mut World) -> usize {
    world
        .query_filtered::<Entity, With<SystemIdMarker>>()
        .iter(world)
        .count()
}

#[derive(Debug)]
struct CloneProbe {
    value: i32,
    clones: Arc<AtomicUsize>,
}

impl PartialEq for CloneProbe {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for CloneProbe {}

impl PartialOrd for CloneProbe {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CloneProbe {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
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

#[test]
fn scalar_vec_and_map_broadcast_share_canonical_identity_and_count_exact_clones() {
    // 1. Scalar Broadcast<T>: single upstream execution, 0 clones for 1 dependent, N - 1 for N.
    {
        let mut app = test_app();
        let clones = Arc::new(AtomicUsize::new(0));
        let upstream_runs = Arc::new(AtomicUsize::new(0));
        let sink_a = Arc::new(Mutex::new(Vec::new()));
        let sink_b = Arc::new(Mutex::new(Vec::new()));
        let sink_c = Arc::new(Mutex::new(Vec::new()));

        let broadcast = signal::from_system({
            let clones = Arc::clone(&clones);
            let upstream_runs = Arc::clone(&upstream_runs);
            move |In(()), mut step: Local<i32>| {
                *step += 1;
                upstream_runs.fetch_add(1, Ordering::SeqCst);
                Some(CloneProbe::new(*step * 10, &clones))
            }
        })
        .broadcast();

        // Cloning the Broadcast handle before registration spawns zero entities and runs nothing.
        let unconsumed_clone = broadcast.clone();
        assert_eq!(system_entity_count(app.world_mut()), 0);

        let direct_a = broadcast.clone().register(app.world_mut());
        let direct_b = unconsumed_clone.register(app.world_mut());
        assert_eq!(
            direct_a.entity(),
            direct_b.entity(),
            "clones of Broadcast<T> must share one canonical broadcast terminal entity"
        );
        // Source + canonical identity broadcast entity = 2 system entities.
        assert_eq!(system_entity_count(app.world_mut()), 2);
        direct_a.release(app.world_mut());
        direct_b.release(app.world_mut());

        // 1 active downstream dependent -> 0 payload clones.
        let claim_a = broadcast
            .clone()
            .map({
                let sink_a = Arc::clone(&sink_a);
                move |In(item): In<CloneProbe>| {
                    sink_a.lock().unwrap().push(item.value);
                }
            })
            .register(app.world_mut());

        clones.store(0, Ordering::SeqCst);
        app.update();
        assert_eq!(upstream_runs.load(Ordering::SeqCst), 1);
        assert_eq!(
            clones.load(Ordering::SeqCst),
            0,
            "1 active dependent must move with 0 clones"
        );
        assert_eq!(*sink_a.lock().unwrap(), vec![10]);

        // 3 active downstream dependents -> N - 1 = 2 payload clones and still 1 upstream run.
        let claim_b = broadcast
            .clone()
            .map({
                let sink_b = Arc::clone(&sink_b);
                move |In(item): In<CloneProbe>| {
                    sink_b.lock().unwrap().push(item.value);
                }
            })
            .register(app.world_mut());
        let claim_c = broadcast
            .clone()
            .map({
                let sink_c = Arc::clone(&sink_c);
                move |In(item): In<CloneProbe>| {
                    sink_c.lock().unwrap().push(item.value);
                }
            })
            .register(app.world_mut());

        clones.store(0, Ordering::SeqCst);
        app.update();
        assert_eq!(upstream_runs.load(Ordering::SeqCst), 2);
        assert_eq!(
            clones.load(Ordering::SeqCst),
            2,
            "3 active dependents must clone exactly N - 1 = 2 times"
        );
        assert_eq!(*sink_a.lock().unwrap(), vec![10, 20]);
        assert_eq!(*sink_b.lock().unwrap(), vec![20]);
        assert_eq!(*sink_c.lock().unwrap(), vec![20]);

        // Releasing 2 of 3 consumers drops back to 1 active dependent -> 0 clones.
        claim_b.release(app.world_mut());
        claim_c.release(app.world_mut());
        clones.store(0, Ordering::SeqCst);
        app.update();
        assert_eq!(upstream_runs.load(Ordering::SeqCst), 3);
        assert_eq!(clones.load(Ordering::SeqCst), 0);
        assert_eq!(*sink_a.lock().unwrap(), vec![10, 20, 30]);

        // Releasing final consumer transitions broadcast 1 -> 0 and deactivates upstream.
        claim_a.release(app.world_mut());
        app.update();
        assert_eq!(upstream_runs.load(Ordering::SeqCst), 3);
        drop(broadcast);
        app.update();
        assert_eq!(system_entity_count(app.world_mut()), 0);
    }

    // 2. Non-replayable BroadcastVec<T> & BroadcastMap<K, V> exact clone counts (0 for 1, N - 1 for
    //    N).
    {
        let mut app = test_app();
        let vec_clones = Arc::new(AtomicUsize::new(0));
        let map_clones = Arc::new(AtomicUsize::new(0));

        struct CustomEventVec(Arc<AtomicUsize>);
        impl SignalVec for CustomEventVec {
            type Item = CloneProbe;
            fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
                let clones = self.0;
                VecSignalMaterializer::new(move |registrar| {
                    registrar.signal_vec_entity(
                        (),
                        move |In(()), mut step: Local<i32>| {
                            *step += 1;
                            Some(vec![VecDiff::Push {
                                value: CloneProbe::new(*step, &clones),
                            }])
                        },
                        OutputDelivery::move_only(),
                    )
                })
            }
        }

        struct CustomEventMap(Arc<AtomicUsize>);
        impl SignalMap for CustomEventMap {
            type Key = CloneProbe;
            type Value = CloneProbe;
            fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
                let clones = self.0;
                MapSignalMaterializer::new(move |registrar| {
                    registrar.signal_map_entity(
                        (),
                        move |In(()), mut step: Local<i32>| {
                            *step += 1;
                            Some(vec![MapDiff::Insert {
                                key: CloneProbe::new(*step, &clones),
                                value: CloneProbe::new(*step * 10, &clones),
                            }])
                        },
                        OutputDelivery::move_only(),
                    )
                })
            }
        }

        let b_vec = CustomEventVec(Arc::clone(&vec_clones)).broadcast();
        let b_map = CustomEventMap(Arc::clone(&map_clones)).broadcast();

        let vec_c1 = b_vec
            .clone()
            .for_each(|In(_): In<Vec<VecDiff<CloneProbe>>>| {})
            .register(app.world_mut());
        let map_c1 = b_map
            .clone()
            .for_each(|In(_): In<Vec<MapDiff<CloneProbe, CloneProbe>>>| {})
            .register(app.world_mut());

        vec_clones.store(0, Ordering::SeqCst);
        map_clones.store(0, Ordering::SeqCst);
        app.update();
        assert_eq!(vec_clones.load(Ordering::SeqCst), 0, "1 vec consumer -> 0 item clones");
        assert_eq!(map_clones.load(Ordering::SeqCst), 0, "1 map consumer -> 0 entry clones");

        let vec_c2 = b_vec
            .clone()
            .for_each(|In(_): In<Vec<VecDiff<CloneProbe>>>| {})
            .register(app.world_mut());
        let vec_c3 = b_vec
            .for_each(|In(_): In<Vec<VecDiff<CloneProbe>>>| {})
            .register(app.world_mut());
        let map_c2 = b_map
            .clone()
            .for_each(|In(_): In<Vec<MapDiff<CloneProbe, CloneProbe>>>| {})
            .register(app.world_mut());
        let map_c3 = b_map
            .for_each(|In(_): In<Vec<MapDiff<CloneProbe, CloneProbe>>>| {})
            .register(app.world_mut());

        vec_clones.store(0, Ordering::SeqCst);
        map_clones.store(0, Ordering::SeqCst);
        app.update();
        assert_eq!(vec_clones.load(Ordering::SeqCst), 2, "3 vec consumers -> 2 item clones");
        // Each MapDiff::Insert clone clones both key and value (2 CloneProbes per batch clone × 2
        // batch clones = 4).
        assert_eq!(
            map_clones.load(Ordering::SeqCst),
            4,
            "3 map consumers -> 2 batch clones (4 key+value clones)"
        );

        vec_c1.release(app.world_mut());
        vec_c2.release(app.world_mut());
        vec_c3.release(app.world_mut());
        map_c1.release(app.world_mut());
        map_c2.release(app.world_mut());
        map_c3.release(app.world_mut());
    }
}

#[test]
fn cache_then_broadcast_normalizes_in_place_and_trailing_cache_is_idempotent() {
    // Scalar: .cache().broadcast() has 1 shared replay terminal (source + cache = 2 entities, NOT
    // 3) and .cache().broadcast().cache() is graph-idempotent in both Warming and Ready states.
    {
        let mut app = test_app();
        #[derive(Resource, Default)]
        struct Emit(Option<i32>);
        app.init_resource::<Emit>();

        let shared = signal::from_system(|In(()), mut emit: ResMut<Emit>| emit.0.take())
            .cache()
            .broadcast();

        // Register while Warming: source + 1 shared replay cache = 2 system entities.
        let early_seen = Arc::new(Mutex::new(Vec::new()));
        let direct_warming = shared.clone().register(app.world_mut());
        let cached_while_warming = shared.clone().cache().register(app.world_mut());
        assert_eq!(
            direct_warming.entity(),
            cached_while_warming.entity(),
            ".cache() on a Warming ReplayableBroadcast must normalize to the exact shared replay terminal"
        );
        assert_eq!(system_entity_count(app.world_mut()), 2);
        assert_eq!(
            inspect_scalar_replay_cache::<i32>(app.world(), direct_warming.entity()),
            Some(ReplayCacheInspection::Warming)
        );

        let early_consumer = shared
            .clone()
            .map({
                let early_seen = Arc::clone(&early_seen);
                move |In(v): In<i32>| early_seen.lock().unwrap().push(v)
            })
            .register(app.world_mut());
        // Source + shared replay cache + early_consumer = 3 system entities.
        assert_eq!(system_entity_count(app.world_mut()), 3);

        app.world_mut().resource_mut::<Emit>().0 = Some(42);
        app.update();
        assert_eq!(*early_seen.lock().unwrap(), vec![42]);
        assert!(matches!(
            inspect_scalar_replay_cache::<i32>(app.world(), direct_warming.entity()),
            Some(ReplayCacheInspection::Ready { current: 42, .. })
        ));

        // Direct .cache() while Ready is also graph-idempotent and late consumer gets targeted
        // snapshot!
        let cached_while_ready = shared.clone().cache().register(app.world_mut());
        assert_eq!(cached_while_ready.entity(), direct_warming.entity());
        assert_eq!(system_entity_count(app.world_mut()), 3);

        let late_seen = Arc::new(Mutex::new(Vec::new()));
        let late_consumer = shared
            .clone()
            .cache()
            .map({
                let late_seen = Arc::clone(&late_seen);
                move |In(v): In<i32>| late_seen.lock().unwrap().push(v)
            })
            .register(app.world_mut());

        app.update();
        assert_eq!(
            *early_seen.lock().unwrap(),
            vec![42],
            "attaching a late consumer to a shared replay broadcast must not re-emit to early consumers"
        );
        assert_eq!(
            *late_seen.lock().unwrap(),
            vec![42],
            "late consumer must receive the authoritative targeted snapshot"
        );

        app.world_mut().resource_mut::<Emit>().0 = Some(99);
        app.update();
        assert_eq!(*early_seen.lock().unwrap(), vec![42, 99]);
        assert_eq!(*late_seen.lock().unwrap(), vec![42, 99]);

        direct_warming.release(app.world_mut());
        cached_while_warming.release(app.world_mut());
        cached_while_ready.release(app.world_mut());
        early_consumer.release(app.world_mut());
        late_consumer.release(app.world_mut());
        drop(shared);
        app.update();
        assert_eq!(system_entity_count(app.world_mut()), 0);
    }

    // Vec & Map: .cache().broadcast() + .cache() idempotence in Warming and Ready.
    {
        let mut app = test_app();
        #[derive(Resource, Default)]
        struct Collections {
            vec_batch: Option<Vec<VecDiff<i32>>>,
            map_batch: Option<Vec<MapDiff<i32, i32>>>,
        }
        app.init_resource::<Collections>();

        struct EventVec;
        impl SignalVec for EventVec {
            type Item = i32;
            fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
                VecSignalMaterializer::new(|registrar| {
                    registrar.signal_vec_entity(
                        (),
                        |In(()), mut c: ResMut<Collections>| c.vec_batch.take(),
                        OutputDelivery::move_only(),
                    )
                })
            }
        }

        struct EventMap;
        impl SignalMap for EventMap {
            type Key = i32;
            type Value = i32;
            fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
                MapSignalMaterializer::new(|registrar| {
                    registrar.signal_map_entity(
                        (),
                        |In(()), mut c: ResMut<Collections>| c.map_batch.take(),
                        OutputDelivery::move_only(),
                    )
                })
            }
        }

        let shared_vec = EventVec.cache().broadcast();
        let shared_map = EventMap.cache().broadcast();

        let v1 = shared_vec.clone().register(app.world_mut());
        let v2 = shared_vec.clone().cache().register(app.world_mut());
        assert_eq!(v1.entity(), v2.entity());
        assert_eq!(
            inspect_vec_replay_cache::<i32>(app.world(), v1.entity()),
            Some(ReplayCacheInspection::Warming)
        );

        let m1 = shared_map.clone().register(app.world_mut());
        let m2 = shared_map.clone().cache().register(app.world_mut());
        assert_eq!(m1.entity(), m2.entity());
        assert_eq!(
            inspect_map_replay_cache::<i32, i32>(app.world(), m1.entity()),
            Some(ReplayCacheInspection::Warming)
        );
        // 2 sources + 2 shared caches = 4 system entities total.
        assert_eq!(system_entity_count(app.world_mut()), 4);

        app.world_mut().resource_mut::<Collections>().vec_batch = Some(vec![VecDiff::Replace { values: vec![1, 2] }]);
        app.world_mut().resource_mut::<Collections>().map_batch =
            Some(vec![MapDiff::Replace { entries: vec![(1, 10)] }]);
        app.update();

        let v3 = shared_vec.cache().register(app.world_mut());
        let m3 = shared_map.cache().register(app.world_mut());
        assert_eq!(v3.entity(), v1.entity());
        assert_eq!(m3.entity(), m1.entity());
        assert_eq!(system_entity_count(app.world_mut()), 4);

        v1.release(app.world_mut());
        v2.release(app.world_mut());
        v3.release(app.world_mut());
        m1.release(app.world_mut());
        m2.release(app.world_mut());
        m3.release(app.world_mut());
    }
}

#[test]
fn broadcast_then_cache_creates_branch_local_warming_caches_and_suppresses_pre_replace_incrementals() {
    let mut app = test_app();
    #[derive(Resource, Default)]
    struct Driver {
        scalar: Option<i32>,
        vec_batch: Option<Vec<VecDiff<i32>>>,
        map_batch: Option<Vec<MapDiff<i32, i32>>>,
    }
    app.init_resource::<Driver>();

    struct RawVec;
    impl SignalVec for RawVec {
        type Item = i32;
        fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
            VecSignalMaterializer::new(|registrar| {
                registrar.signal_vec_entity(
                    (),
                    |In(()), mut d: ResMut<Driver>| d.vec_batch.take(),
                    OutputDelivery::move_only(),
                )
            })
        }
    }

    struct RawMap;
    impl SignalMap for RawMap {
        type Key = i32;
        type Value = i32;
        fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
            MapSignalMaterializer::new(|registrar| {
                registrar.signal_map_entity(
                    (),
                    |In(()), mut d: ResMut<Driver>| d.map_batch.take(),
                    OutputDelivery::move_only(),
                )
            })
        }
    }

    let scalar_b = signal::from_system(|In(()), mut d: ResMut<Driver>| d.scalar.take()).broadcast();
    let vec_b = RawVec.broadcast();
    let map_b = RawMap.broadcast();

    // Early consumers keep the broadcast terminals active while initial replacements flow through.
    let early_scalar = scalar_b.clone().register(app.world_mut());
    let early_vec = vec_b.clone().register(app.world_mut());
    let early_map = map_b.clone().register(app.world_mut());

    app.world_mut().resource_mut::<Driver>().scalar = Some(100);
    app.world_mut().resource_mut::<Driver>().vec_batch = Some(vec![VecDiff::Replace { values: vec![1, 2] }]);
    app.world_mut().resource_mut::<Driver>().map_batch = Some(vec![MapDiff::Replace { entries: vec![(1, 10)] }]);
    app.update();

    // Now attach late branch-local .broadcast().cache() consumers AFTER initial emissions already
    // happened!
    let late_scalar_cache = scalar_b.clone().cache();
    let late_vec_cache = vec_b.clone().cache();
    let late_map_cache = map_b.clone().cache();

    let scalar_out = Arc::new(Mutex::new(Vec::new()));
    let vec_out = Arc::new(Mutex::new(Vec::new()));
    let map_out = Arc::new(Mutex::new(Vec::new()));

    // Inspect branch-local cache identities via custom registrar wrapper or direct claim before
    // mapping: here we register late_scalar_cache directly first to verify distinct entity +
    // Warming state, and create fresh branch-local caches for downstream sinks.
    let late_scalar_probe = late_scalar_cache.register(app.world_mut());
    let late_vec_probe = late_vec_cache.register(app.world_mut());
    let late_map_probe = late_map_cache.register(app.world_mut());

    assert_ne!(late_scalar_probe.entity(), early_scalar.entity());
    assert_ne!(late_vec_probe.entity(), early_vec.entity());
    assert_ne!(late_map_probe.entity(), early_map.entity());
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), early_scalar.entity()),
        None
    );
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), late_scalar_probe.entity()),
        Some(ReplayCacheInspection::Warming)
    );
    assert_eq!(
        inspect_vec_replay_cache::<i32>(app.world(), late_vec_probe.entity()),
        Some(ReplayCacheInspection::Warming)
    );
    assert_eq!(
        inspect_map_replay_cache::<i32, i32>(app.world(), late_map_probe.entity()),
        Some(ReplayCacheInspection::Warming)
    );

    let s_sink = scalar_b
        .cache()
        .map({
            let scalar_out = Arc::clone(&scalar_out);
            move |In(v): In<i32>| scalar_out.lock().unwrap().push(v)
        })
        .register(app.world_mut());
    let v_sink = vec_b
        .cache()
        .for_each({
            let vec_out = Arc::clone(&vec_out);
            move |In(diffs): In<Vec<VecDiff<i32>>>| vec_out.lock().unwrap().push(diffs)
        })
        .register(app.world_mut());
    let m_sink = map_b
        .cache()
        .for_each({
            let map_out = Arc::clone(&map_out);
            move |In(diffs): In<Vec<MapDiff<i32, i32>>>| map_out.lock().unwrap().push(diffs)
        })
        .register(app.world_mut());

    // Frame with incremental-only vec/map diffs and no scalar output: all branch-local caches stay
    // Warming!
    app.world_mut().resource_mut::<Driver>().vec_batch = Some(vec![VecDiff::Push { value: 999 }]);
    app.world_mut().resource_mut::<Driver>().map_batch = Some(vec![MapDiff::Insert { key: 9, value: 90 }]);
    app.update();

    assert!(scalar_out.lock().unwrap().is_empty());
    assert!(vec_out.lock().unwrap().is_empty());
    assert!(map_out.lock().unwrap().is_empty());
    assert_eq!(
        inspect_vec_replay_cache::<i32>(app.world(), late_vec_probe.entity()),
        Some(ReplayCacheInspection::Warming)
    );
    assert_eq!(
        inspect_map_replay_cache::<i32, i32>(app.world(), late_map_probe.entity()),
        Some(ReplayCacheInspection::Warming)
    );

    // Mixed batch: incremental prefix BEFORE first Replace must be discarded, and ordered suffix
    // AFTER Replace applied!
    app.world_mut().resource_mut::<Driver>().scalar = Some(200);
    app.world_mut().resource_mut::<Driver>().vec_batch = Some(vec![
        VecDiff::Push { value: 888 },
        VecDiff::Replace { values: vec![10, 20] },
        VecDiff::Push { value: 30 },
    ]);
    app.world_mut().resource_mut::<Driver>().map_batch = Some(vec![
        MapDiff::Insert { key: 88, value: 880 },
        MapDiff::Replace {
            entries: vec![(1, 100), (2, 200)],
        },
        MapDiff::Insert { key: 3, value: 300 },
    ]);
    app.update();

    assert_eq!(*scalar_out.lock().unwrap(), vec![200]);
    assert!(matches!(
        inspect_vec_replay_cache::<i32>(app.world(), late_vec_probe.entity()),
        Some(ReplayCacheInspection::Ready { ref current, .. }) if current.as_slice() == [10, 20, 30]
    ));
    assert!(matches!(
        inspect_map_replay_cache::<i32, i32>(app.world(), late_map_probe.entity()),
        Some(ReplayCacheInspection::Ready { ref current, .. })
            if *current == BTreeMap::from([(1, 100), (2, 200), (3, 300)])
    ));
    let mut observed_vec = Vec::new();
    for batch in vec_out.lock().unwrap().iter() {
        for diff in batch {
            diff.clone().apply_to_vec(&mut observed_vec);
        }
    }
    assert_eq!(observed_vec, vec![10, 20, 30]);
    assert_eq!(map_out.lock().unwrap().len(), 1);

    early_scalar.release(app.world_mut());
    early_vec.release(app.world_mut());
    early_map.release(app.world_mut());
    late_scalar_probe.release(app.world_mut());
    late_vec_probe.release(app.world_mut());
    late_map_probe.release(app.world_mut());
    s_sink.release(app.world_mut());
    v_sink.release(app.world_mut());
    m_sink.release(app.world_mut());
}

#[test]
fn linear_replayable_rejects_second_dependent_and_conflicting_post_commit_broadcast_normalization() {
    let mut app = test_app();

    // 1. Linear .cache() terminal rejects 2 distinct dependents via registrar with MoveOnlyFanout
    //    and zero partial world state.
    let baseline = system_entity_count(app.world_mut());
    let fanout_attempt = ScalarSignalMaterializer::<i32>::new(|registrar| {
        let cached = registrar.materialize_signal(Box::new(signal::once(7).cache()).into_signal_materializer())?;
        let left = registrar.signal_entity(cached, |In(v): In<i32>| Some(v + 1), OutputDelivery::move_only())?;
        let right = registrar.signal_entity(cached, |In(v): In<i32>| Some(v + 2), OutputDelivery::move_only())?;
        registrar.signal_entity(
            [left, right],
            |In(value): In<i32>| Some(value),
            OutputDelivery::move_only(),
        )
    })
    .try_register(app.world_mut());

    assert_eq!(fanout_attempt.unwrap_err(), SignalRegistrationError::MoveOnlyFanout);
    assert_eq!(
        system_entity_count(app.world_mut()),
        baseline,
        "rejected MoveOnlyFanout plan must publish zero entities"
    );

    // 2. Downstream combinator on a Broadcast clone yields a linear move-only chain again.
    let b = signal::once(10).broadcast();
    let linear_after_broadcast = ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let mapped = registrar.materialize_signal(Box::new(b.map_in(|v: i32| v * 2)).into_signal_materializer())?;
        let left = registrar.signal_entity(mapped, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        let right = registrar.signal_entity(mapped, |In(v): In<i32>| Some(v), OutputDelivery::move_only())?;
        registrar.signal_entity(
            [left, right],
            |In(value): In<i32>| Some(value),
            OutputDelivery::move_only(),
        )
    })
    .try_register(app.world_mut());

    assert_eq!(
        linear_after_broadcast.unwrap_err(),
        SignalRegistrationError::MoveOnlyFanout
    );
    assert_eq!(system_entity_count(app.world_mut()), baseline);

    // 3. Re-normalizing an already-committed ReplayableBroadcast terminal via .broadcast() hits the
    // Existing(entity) branch in planned_broadcast_replayable and preserves the exact canonical
    // entity.
    let shared_replay = signal::once(5).cache().broadcast();
    let c1 = shared_replay.clone().register(app.world_mut());
    let c2 = shared_replay.broadcast().register(app.world_mut());
    assert_eq!(c1.entity(), c2.entity());
    c1.release(app.world_mut());
    c2.release(app.world_mut());
    assert_eq!(system_entity_count(app.world_mut()), baseline);
}

#[test]
fn mutable_vec_and_mutable_btree_map_coexist_across_multiple_linear_chains_without_user_broadcast() {
    let mut app = test_app();

    let vec_source = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let vec_a = Arc::new(Mutex::new(Vec::new()));
    let vec_b = Arc::new(Mutex::new(Vec::new()));

    let v_claim_a = vec_source
        .signal_vec()
        .map_in(|x: i32| x * 10)
        .for_each({
            let vec_a = Arc::clone(&vec_a);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    diff.apply_to_vec(&mut vec_a.lock().unwrap());
                }
            }
        })
        .register(app.world_mut());

    let v_claim_b = vec_source
        .signal_vec()
        .for_each({
            let vec_b = Arc::clone(&vec_b);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    diff.apply_to_vec(&mut vec_b.lock().unwrap());
                }
            }
        })
        .register(app.world_mut());

    let map_source = MutableBTreeMap::builder()
        .values([(1, 10), (2, 20)])
        .spawn(app.world_mut());
    let map_entries_seen = Arc::new(Mutex::new(BTreeMap::new()));
    let vec_keys_seen = Arc::new(Mutex::new(Vec::new()));
    let vec_entries_seen = Arc::new(Mutex::new(Vec::new()));
    let key_lookup_seen = Arc::new(Mutex::new(Vec::new()));

    let m_claim = map_source
        .signal_map()
        .for_each({
            let map_entries_seen = Arc::clone(&map_entries_seen);
            move |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                let mut guard = map_entries_seen.lock().unwrap();
                for diff in diffs {
                    match diff {
                        MapDiff::Replace { entries } => *guard = entries.into_iter().collect(),
                        MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                            guard.insert(key, value);
                        }
                        MapDiff::Remove { key } => {
                            guard.remove(&key);
                        }
                        MapDiff::Clear => guard.clear(),
                    }
                }
            }
        })
        .register(app.world_mut());

    let keys_claim = map_source
        .signal_vec_keys()
        .for_each({
            let vec_keys_seen = Arc::clone(&vec_keys_seen);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    diff.apply_to_vec(&mut vec_keys_seen.lock().unwrap());
                }
            }
        })
        .register(app.world_mut());

    let entries_claim = map_source
        .signal_vec_entries()
        .for_each({
            let vec_entries_seen = Arc::clone(&vec_entries_seen);
            move |In(diffs): In<Vec<VecDiff<(i32, i32)>>>| {
                for diff in diffs {
                    diff.apply_to_vec(&mut vec_entries_seen.lock().unwrap());
                }
            }
        })
        .register(app.world_mut());

    let key_claim = map_source
        .signal_map()
        .key(2)
        .map({
            let key_lookup_seen = Arc::clone(&key_lookup_seen);
            move |In(v): In<Option<i32>>| key_lookup_seen.lock().unwrap().push(v)
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(*vec_a.lock().unwrap(), vec![10, 20]);
    assert_eq!(*vec_b.lock().unwrap(), vec![1, 2]);
    assert_eq!(*map_entries_seen.lock().unwrap(), BTreeMap::from([(1, 10), (2, 20)]));
    assert_eq!(*vec_keys_seen.lock().unwrap(), vec![1, 2]);
    assert_eq!(*vec_entries_seen.lock().unwrap(), vec![(1, 10), (2, 20)]);
    assert_eq!(*key_lookup_seen.lock().unwrap(), vec![Some(20)]);

    vec_source.write(app.world_mut()).push(3);
    map_source.write(app.world_mut()).insert(3, 30);
    map_source.write(app.world_mut()).insert(2, 200);
    app.update();

    assert_eq!(*vec_a.lock().unwrap(), vec![10, 20, 30]);
    assert_eq!(*vec_b.lock().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        *map_entries_seen.lock().unwrap(),
        BTreeMap::from([(1, 10), (2, 200), (3, 30)])
    );
    assert_eq!(*vec_keys_seen.lock().unwrap(), vec![1, 2, 3]);
    assert_eq!(*vec_entries_seen.lock().unwrap(), vec![(1, 10), (2, 200), (3, 30)]);
    assert_eq!(*key_lookup_seen.lock().unwrap(), vec![Some(20), Some(200)]);

    v_claim_a.release(app.world_mut());
    v_claim_b.release(app.world_mut());
    m_claim.release(app.world_mut());
    keys_claim.release(app.world_mut());
    entries_claim.release(app.world_mut());
    key_claim.release(app.world_mut());
}

#[test]
fn move_only_boxed_signals_and_signal_option_work_through_switch_and_flatten() {
    let mut app = test_app();

    #[derive(Resource, Clone, PartialEq)]
    struct Branch(usize);
    app.insert_resource(Branch(0));

    let scalar_seen = Arc::new(Mutex::new(Vec::new()));
    let vec_seen = Arc::new(Mutex::new(Vec::new()));
    let map_seen = Arc::new(Mutex::new(BTreeMap::new()));

    let scalar_claim = signal::from_resource::<Branch>()
        .dedupe()
        .map_in(|Branch(mode)| match mode {
            0 => signal::option::<BoxedSignal<i32>>(None).cache().boxed_replayable(),
            1 => {
                let inner: BoxedSignal<i32> = signal::once(11).boxed();
                signal::option(Some(inner)).cache().boxed_replayable()
            }
            _ => {
                let inner: BoxedSignal<i32> = signal::once(22).map_in(|x: i32| x + 1).boxed();
                signal::option(Some(inner)).cache().boxed_replayable()
            }
        })
        .flatten()
        .map({
            let scalar_seen = Arc::clone(&scalar_seen);
            move |In(v): In<Option<i32>>| scalar_seen.lock().unwrap().push(v)
        })
        .register(app.world_mut());

    let vec_a = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let vec_b = MutableVec::builder().values([10]).spawn(app.world_mut());
    let vec_claim = signal::from_resource::<Branch>()
        .dedupe()
        .switch_signal_vec(move |In(Branch(mode))| -> BoxedReplayableSignalVec<i32> {
            if mode == 0 {
                vec_a.signal_vec().boxed_replayable()
            } else {
                vec_b.signal_vec().map_in(|x: i32| x * 3).cache().boxed_replayable()
            }
        })
        .for_each({
            let vec_seen = Arc::clone(&vec_seen);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    diff.apply_to_vec(&mut vec_seen.lock().unwrap());
                }
            }
        })
        .register(app.world_mut());

    let map_a = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
    let map_b = MutableBTreeMap::builder().values([(2, 20)]).spawn(app.world_mut());
    let map_claim = signal::from_resource::<Branch>()
        .dedupe()
        .switch_signal_map(move |In(Branch(mode))| -> BoxedReplayableSignalMap<i32, i32> {
            if mode == 0 {
                map_a.signal_map().boxed_replayable()
            } else {
                map_b
                    .signal_map()
                    .map_value(|In(v): In<i32>| v + 5)
                    .cache()
                    .boxed_replayable()
            }
        })
        .for_each({
            let map_seen = Arc::clone(&map_seen);
            move |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                let mut guard = map_seen.lock().unwrap();
                for diff in diffs {
                    if let MapDiff::Replace { entries } = diff {
                        *guard = entries.into_iter().collect();
                    }
                }
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(*scalar_seen.lock().unwrap(), vec![None]);
    assert_eq!(*vec_seen.lock().unwrap(), vec![1, 2]);
    assert_eq!(*map_seen.lock().unwrap(), BTreeMap::from([(1, 10)]));

    app.world_mut().resource_mut::<Branch>().0 = 1;
    app.update();
    assert_eq!(*scalar_seen.lock().unwrap(), vec![None, Some(11)]);
    assert_eq!(*vec_seen.lock().unwrap(), vec![30]);
    assert_eq!(*map_seen.lock().unwrap(), BTreeMap::from([(2, 25)]));

    app.world_mut().resource_mut::<Branch>().0 = 2;
    app.update();
    assert_eq!(*scalar_seen.lock().unwrap(), vec![None, Some(11), Some(23)]);

    scalar_claim.release(app.world_mut());
    vec_claim.release(app.world_mut());
    map_claim.release(app.world_mut());
}

#[test]
fn replayable_schedule_and_option_preserve_in_place_broadcast_normalization() {
    fn assert_replayable_scalar<S: ReplayableSignal>(_: &S) {}
    fn assert_replayable_vec<S: ReplayableSignalVec>(_: &S) {}
    fn assert_replayable_map<S: ReplayableSignalMap>(_: &S) {}

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<Update>());

    // 1. `.cache().schedule::<Update>().broadcast()` and
    //    `.dedupe().schedule::<Update>().broadcast()` preserve `ReplayableBroadcast<T>` and add 0
    //    extra entities.
    let cached_sched: ReplayableBroadcast<i32> = signal::from_system(|In(())| Some(10))
        .cache()
        .schedule::<Update>()
        .broadcast();
    assert_replayable_scalar(&cached_sched);
    let c1 = cached_sched.clone().register(app.world_mut());
    let c2 = cached_sched.clone().cache().register(app.world_mut());
    assert_eq!(c1.entity(), c2.entity());
    // Source + shared replay cache = 2 system entities.
    assert_eq!(system_entity_count(app.world_mut()), 2);
    c1.release(app.world_mut());
    c2.release(app.world_mut());
    drop(cached_sched);
    app.update();
    assert_eq!(system_entity_count(app.world_mut()), 0);

    let deduped_sched: ReplayableBroadcast<i32> = signal::from_system(|In(())| Some(20))
        .dedupe()
        .schedule::<Update>()
        .broadcast();
    assert_replayable_scalar(&deduped_sched);
    let d1 = deduped_sched.clone().register(app.world_mut());
    let d2 = deduped_sched.clone().cache().register(app.world_mut());
    assert_eq!(d1.entity(), d2.entity());
    assert_eq!(system_entity_count(app.world_mut()), 2);
    d1.release(app.world_mut());
    d2.release(app.world_mut());
    drop(deduped_sched);
    app.update();
    assert_eq!(system_entity_count(app.world_mut()), 0);

    // 2. `signal::option(...).cache().broadcast()` returns `ReplayableBroadcast<Option<T>>` and
    //    normalizes in place without adding an identity broadcast entity.
    let opt_some: ReplayableBroadcast<Option<i32>> = signal::option(Some(signal::once(7))).cache().broadcast();
    assert_replayable_scalar(&opt_some);
    let o1 = opt_some.clone().register(app.world_mut());
    let o2 = opt_some.clone().cache().register(app.world_mut());
    assert_eq!(o1.entity(), o2.entity());
    o1.release(app.world_mut());
    o2.release(app.world_mut());
    drop(opt_some);
    app.update();

    let opt_none: ReplayableBroadcast<Option<i32>> = signal::option::<BoxedSignal<i32>>(None)
        .cache()
        .schedule::<Update>()
        .broadcast();
    assert_replayable_scalar(&opt_none);
    let n1 = opt_none.clone().register(app.world_mut());
    let n2 = opt_none.clone().cache().register(app.world_mut());
    assert_eq!(n1.entity(), n2.entity());
    n1.release(app.world_mut());
    n2.release(app.world_mut());
    drop(opt_none);
    app.update();

    // 3. `MutableVec::signal_vec().schedule::<Update>().broadcast()` and
    //    `MutableBTreeMap::signal_map().schedule::<Update>().broadcast()` preserve
    //    `ReplayableBroadcastVec` / `ReplayableBroadcastMap` and reuse the intrinsic source
    //    terminal.
    let vec_src = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let map_src = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());

    let direct_vec = vec_src.signal_vec().register(app.world_mut());
    let sched_vec: ReplayableBroadcastVec<i32> = vec_src.signal_vec().schedule::<Update>().broadcast();
    assert_replayable_vec(&sched_vec);
    let sv1 = sched_vec.clone().register(app.world_mut());
    let sv2 = sched_vec.cache().register(app.world_mut());
    assert_eq!(sv1.entity(), direct_vec.entity());
    assert_eq!(sv2.entity(), direct_vec.entity());

    let direct_map = map_src.signal_map().register(app.world_mut());
    let sched_map: ReplayableBroadcastMap<i32, i32> = map_src.signal_map().schedule::<Update>().broadcast();
    assert_replayable_map(&sched_map);
    let sm1 = sched_map.clone().register(app.world_mut());
    let sm2 = sched_map.cache().register(app.world_mut());
    assert_eq!(sm1.entity(), direct_map.entity());
    assert_eq!(sm2.entity(), direct_map.entity());

    direct_vec.release(app.world_mut());
    sv1.release(app.world_mut());
    sv2.release(app.world_mut());
    direct_map.release(app.world_mut());
    sm1.release(app.world_mut());
    sm2.release(app.world_mut());
}

#[test]
fn cache_after_non_replayable_transform_on_replayable_broadcast_starts_warming() {
    let mut app = test_app();

    #[derive(Resource, Default)]
    struct Emit(Option<i32>);
    app.init_resource::<Emit>();

    let shared_scalar = signal::from_system(|In(()), mut emit: ResMut<Emit>| emit.0.take())
        .cache()
        .broadcast();
    let vec_src = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let map_src = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
    let shared_vec = vec_src.signal_vec().broadcast();
    let shared_map = map_src.signal_map().broadcast();

    // Establish Ready state on `shared_scalar`, `shared_vec`, and `shared_map`.
    let anchor_scalar = shared_scalar.clone().register(app.world_mut());
    let anchor_vec = shared_vec.clone().register(app.world_mut());
    let anchor_map = shared_map.clone().register(app.world_mut());

    app.world_mut().resource_mut::<Emit>().0 = Some(5);
    app.update();
    assert!(matches!(
        inspect_scalar_replay_cache::<i32>(app.world(), anchor_scalar.entity()),
        Some(ReplayCacheInspection::Ready { current: 5, .. })
    ));

    // Non-replayable downstream transforms followed by `.cache()` must create distinct `Warming`
    // caches rather than synchronously transferring ancestor replay state during registration!
    struct NonReplayableMapTransform(ReplayableBroadcastMap<i32, i32>);
    impl SignalMap for NonReplayableMapTransform {
        type Key = i32;
        type Value = i32;
        fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
            let upstream = Box::new(self.0).into_signal_map_materializer();
            MapSignalMaterializer::new(move |registrar| {
                let upstream = registrar.materialize_signal_map(upstream)?;
                registrar.signal_map_entity(
                    upstream,
                    |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                        Some(
                            diffs
                                .into_iter()
                                .map(|diff| diff.map_value(|v| v * 10))
                                .collect::<Vec<_>>(),
                        )
                    },
                    OutputDelivery::move_only(),
                )
            })
        }
    }

    let derived_scalar_cache = shared_scalar
        .clone()
        .map_in(|v: i32| v * 10)
        .cache()
        .register(app.world_mut());
    let derived_vec_cache = shared_vec
        .clone()
        .map_in(|v: i32| v * 10)
        .cache()
        .register(app.world_mut());
    let derived_map_cache = NonReplayableMapTransform(shared_map.clone())
        .cache()
        .register(app.world_mut());

    assert_ne!(derived_scalar_cache.entity(), anchor_scalar.entity());
    assert_ne!(derived_vec_cache.entity(), anchor_vec.entity());
    assert_ne!(derived_map_cache.entity(), anchor_map.entity());
    assert_eq!(
        inspect_scalar_replay_cache::<i32>(app.world(), derived_scalar_cache.entity()),
        Some(ReplayCacheInspection::Warming)
    );
    assert_eq!(
        inspect_vec_replay_cache::<i32>(app.world(), derived_vec_cache.entity()),
        Some(ReplayCacheInspection::Warming)
    );
    assert_eq!(
        inspect_map_replay_cache::<i32, i32>(app.world(), derived_map_cache.entity()),
        Some(ReplayCacheInspection::Warming)
    );

    // Scheduled execution delivers ancestor replay snapshots through the non-replayable transforms
    // and warms all three downstream caches!
    app.update();
    assert!(matches!(
        inspect_scalar_replay_cache::<i32>(app.world(), derived_scalar_cache.entity()),
        Some(ReplayCacheInspection::Ready { current: 50, .. })
    ));
    assert!(matches!(
        inspect_vec_replay_cache::<i32>(app.world(), derived_vec_cache.entity()),
        Some(ReplayCacheInspection::Ready { ref current, .. }) if current.as_slice() == [10, 20]
    ));
    assert!(matches!(
        inspect_map_replay_cache::<i32, i32>(app.world(), derived_map_cache.entity()),
        Some(ReplayCacheInspection::Ready { ref current, .. }) if *current == BTreeMap::from([(1, 100)])
    ));

    derived_scalar_cache.release(app.world_mut());
    derived_vec_cache.release(app.world_mut());
    derived_map_cache.release(app.world_mut());
    anchor_scalar.release(app.world_mut());
    anchor_vec.release(app.world_mut());
    anchor_map.release(app.world_mut());
}

#[test]
fn chain_followed_by_broadcast_repairs_canceled_baseline_across_full_deactivation_and_reactivation() {
    #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
    struct Producer;

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<Producer>());

    let left = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let right = MutableVec::builder().values([10]).spawn(app.world_mut());

    // Direct `left.chain(right).broadcast()` structurally retains `Chain` across `2 -> 1 -> 0`
    // deactivation and `0 -> 1 -> 2` reactivation.
    let shared_chain = left
        .signal_vec()
        .schedule::<Producer>()
        .chain(right.signal_vec().schedule::<Producer>())
        .schedule::<Last>()
        .broadcast();

    let branch_a = Arc::new(Mutex::new(Vec::new()));
    let branch_b = Arc::new(Mutex::new(Vec::new()));

    let register_branch = |world: &mut World, shared: BroadcastVec<i32>, target: &Arc<Mutex<Vec<i32>>>| {
        let target = Arc::clone(target);
        shared
            .for_each(move |In(diffs): In<Vec<VecDiff<i32>>>| {
                let mut guard = target.lock().unwrap();
                for diff in diffs {
                    diff.apply_to_vec(&mut guard);
                }
            })
            .register(world)
    };

    let claim_a = register_branch(app.world_mut(), shared_chain.clone(), &branch_a);
    let claim_b = register_branch(app.world_mut(), shared_chain.clone(), &branch_b);

    app.world_mut().run_schedule(Producer);
    app.update();
    assert_eq!(*branch_a.lock().unwrap(), vec![1, 2, 10]);
    assert_eq!(*branch_b.lock().unwrap(), vec![1, 2, 10]);

    // Queue a batch into `Chain`'s input buffer in `Producer`, then release both branches (`2 -> 1
    // -> 0`) BEFORE `Last` runs so `Chain`'s queued batch is canceled by deactivation
    // (`signal_input_epoch` bump)!
    left.write(app.world_mut()).push(3);
    app.world_mut().run_schedule(Producer);
    claim_a.release(app.world_mut());
    claim_b.release(app.world_mut());

    // Mutate while `Chain` is inactive (`0` claims) and step `Producer` + `Last`.
    right.write(app.world_mut()).push(20);
    app.world_mut().run_schedule(Producer);
    app.update();

    // Reactivate (`0 -> 1 -> 2`) on new consumer state vectors: `Chain` must detect the epoch bump
    // and repair via `ChainState::published` + fresh `MutableVec` replay snapshots so both branches
    // converge to the exact authoritative `[1, 2, 3, 10, 20]` sequence!
    let reactivated_a = Arc::new(Mutex::new(Vec::new()));
    let reactivated_b = Arc::new(Mutex::new(Vec::new()));
    let claim_a2 = register_branch(app.world_mut(), shared_chain.clone(), &reactivated_a);
    let claim_b2 = register_branch(app.world_mut(), shared_chain.clone(), &reactivated_b);

    app.world_mut().run_schedule(Producer);
    app.update();
    assert_eq!(*reactivated_a.lock().unwrap(), vec![1, 2, 3, 10, 20]);
    assert_eq!(*reactivated_b.lock().unwrap(), vec![1, 2, 3, 10, 20]);

    claim_a2.release(app.world_mut());
    claim_b2.release(app.world_mut());
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn imperative_signal_evaluation_rejects_move_target_with_live_downstream_consumer_before_running() {
    use core::panic::AssertUnwindSafe;
    use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
    use std::panic::catch_unwind;

    // 1. Leaf `Move` target (`0` live downstream consumers) returns its output by move, and
    //    `downcast_signal_value` returns `None` on wrong type and `Some` on exact type.
    {
        let mut app = test_app();
        let runs = Arc::new(AtomicUsize::new(0));
        let leaf = ScalarSignalMaterializer::<i32>::new({
            let runs = Arc::clone(&runs);
            move |registrar| {
                registrar.signal_entity(
                    (),
                    move |In(())| {
                        runs.fetch_add(1, Ordering::SeqCst);
                        Some(42)
                    },
                    OutputDelivery::move_only(),
                )
            }
        })
        .register(app.world_mut());

        let wrong_type = run_signal_subgraph_once(app.world_mut(), leaf.entity()).expect("leaf output");
        assert_eq!(downcast_signal_value::<u64>(wrong_type), None);

        let right_type = run_signal_subgraph_once(app.world_mut(), leaf.entity()).expect("leaf output");
        assert_eq!(downcast_signal_value::<i32>(right_type), Some(42));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        leaf.release(app.world_mut());
    }

    // 2. `Move` target with 1 active ordinary dependent panics upfront BEFORE executing upstream!
    {
        let mut app = test_app();
        let runs = Arc::new(AtomicUsize::new(0));
        let consumer = ScalarSignalMaterializer::<()>::new({
            let runs = Arc::clone(&runs);
            move |registrar| {
                let src = registrar.signal_entity(
                    (),
                    move |In(())| {
                        runs.fetch_add(1, Ordering::SeqCst);
                        Some(7)
                    },
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_entity(src, |In(_): In<i32>| None::<()>, OutputDelivery::move_only())
            }
        })
        .register(app.world_mut());

        let move_entity = app
            .world_mut()
            .query_filtered::<Entity, With<SystemIdMarker>>()
            .iter(app.world())
            .find(|&e| e != consumer.entity().entity())
            .map(jonmo::graph::SignalEntity::from)
            .expect("upstream move entity");

        let panic_res = catch_unwind(AssertUnwindSafe(|| {
            run_signal_subgraph_once(app.world_mut(), move_entity)
        }));
        assert!(panic_res.is_err(), "Move target with 1 ordinary dependent must panic");
        assert_eq!(
            runs.load(Ordering::SeqCst),
            0,
            "upfront precheck must panic BEFORE executing upstream system"
        );
        consumer.release(app.world_mut());
    }

    // 3. `Move` target with 1 ordered dependent (`Switch` manager -> `ManagerBridge`) panics
    //    upfront BEFORE executing upstream!
    {
        let mut app = test_app();
        let runs = Arc::new(AtomicUsize::new(0));
        let switch_claim = signal::from_system({
            let runs = Arc::clone(&runs);
            move |In(())| {
                runs.fetch_add(1, Ordering::SeqCst);
                Some(1)
            }
        })
        .switch(|In(v): In<i32>| signal::once(v).cache())
        .register(app.world_mut());

        let system_entities: Vec<jonmo::graph::SignalEntity> = app
            .world_mut()
            .query_filtered::<Entity, With<SystemIdMarker>>()
            .iter(app.world())
            .filter(|&e| e != switch_claim.entity().entity())
            .map(jonmo::graph::SignalEntity::from)
            .collect();
        let mut saw_ordered_or_move_panic = false;
        for candidate in system_entities {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_signal_subgraph_once(app.world_mut(), candidate)
            }));
            if res.is_err() {
                saw_ordered_or_move_panic = true;
                assert_eq!(
                    runs.load(Ordering::SeqCst),
                    0,
                    "Move manager / adapter entity must panic BEFORE running upstream"
                );
            }
        }
        assert!(saw_ordered_or_move_panic);
        switch_claim.release(app.world_mut());
    }

    // 4. `Move` replayable target (`.cache()` without `.broadcast()`) with 1 `Pending` (`Warming`)
    //    or `Published` (`Ready`) replay dependent panics upfront BEFORE executing upstream (never
    //    silently cloning via the replay contract)!
    {
        let mut app = test_app();
        let runs = Arc::new(AtomicUsize::new(0));
        let sink_claim = ScalarSignalMaterializer::<()>::new({
            let runs = Arc::clone(&runs);
            move |registrar| {
                let cached = registrar.replayable_signal_entity(
                    (),
                    move |In(())| {
                        runs.fetch_add(1, Ordering::SeqCst);
                        Some(99)
                    },
                    jonmo::ReplayInitialization::Warming,
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_entity(cached, |In(_): In<i32>| None::<()>, OutputDelivery::move_only())
            }
        })
        .register(app.world_mut());

        let cached_entity = app
            .world_mut()
            .query_filtered::<Entity, With<SystemIdMarker>>()
            .iter(app.world())
            .find(|&e| e != sink_claim.entity().entity())
            .map(jonmo::graph::SignalEntity::from)
            .expect("cached move entity");

        assert_eq!(
            inspect_scalar_replay_cache::<i32>(app.world(), cached_entity),
            Some(ReplayCacheInspection::Warming)
        );
        let warming_panic = catch_unwind(AssertUnwindSafe(|| {
            run_signal_subgraph_once(app.world_mut(), cached_entity)
        }));
        assert!(warming_panic.is_err());
        assert_eq!(runs.load(Ordering::SeqCst), 0);

        app.update();
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(matches!(
            inspect_scalar_replay_cache::<i32>(app.world(), cached_entity),
            Some(ReplayCacheInspection::Ready { current: 99, .. })
        ));

        let ready_panic = catch_unwind(AssertUnwindSafe(|| {
            run_signal_subgraph_once(app.world_mut(), cached_entity)
        }));
        assert!(ready_panic.is_err());
        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "Ready Move replayable target must panic BEFORE running upstream again"
        );
        sink_claim.release(app.world_mut());
    }

    // 5. Positive control: explicit `.broadcast()` and `.cache().broadcast()` targets share with
    //    active downstream consumers AND return the evaluated value cleanly!
    {
        let mut app = test_app();
        let downstream_seen = Arc::new(Mutex::new(Vec::new()));
        let shared = signal::from_system(|In(()), mut n: Local<i32>| {
            *n += 10;
            Some(*n)
        })
        .cache()
        .broadcast();

        let direct_claim = shared.clone().register(app.world_mut());
        let sink_claim = shared
            .map({
                let downstream_seen = Arc::clone(&downstream_seen);
                move |In(v): In<i32>| downstream_seen.lock().unwrap().push(v)
            })
            .register(app.world_mut());

        let out = run_signal_subgraph_once(app.world_mut(), direct_claim.entity()).expect("broadcast output");
        assert_eq!(downcast_signal_value::<i32>(out), Some(10));
        app.update();
        assert_eq!(*downstream_seen.lock().unwrap(), vec![10, 20]);

        direct_claim.release(app.world_mut());
        sink_claim.release(app.world_mut());
    }
}
