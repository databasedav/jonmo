#![allow(missing_docs)]

extern crate alloc;

use alloc::sync::Arc;
use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use std::sync::Mutex;

#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
use jonmo::{OutputDelivery, ScalarSignalMaterializer, SignalRegistrationError, prelude::*};

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app
}

fn settle(app: &mut App) {
    for _ in 0..8 {
        app.update();
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct IndexProbe;

type IndexLog = Arc<Mutex<Vec<Option<usize>>>>;

fn register_index_observer(app: &mut App, index: Broadcast<Option<usize>>) -> (SignalClaim, IndexLog) {
    let log = IndexLog::default();
    let claim = index
        .map({
            let log = log.clone();
            move |In(value): In<Option<usize>>| {
                log.lock().expect("index observation lock").push(value);
            }
        })
        .schedule::<IndexProbe>()
        .register(app.world_mut());
    (claim, log)
}

fn observe_index(app: &mut App, log: &IndexLog) -> Option<Option<usize>> {
    log.lock().expect("index observation lock").clear();
    app.world_mut().run_schedule(IndexProbe);
    let mut values = log.lock().expect("index observation lock");
    let observed = values.pop();
    assert!(
        values.is_empty(),
        "one schedule pass must deliver at most one index value"
    );
    observed
}

#[cfg(feature = "imperative-signal-evaluation")]
fn evaluate_index(world: &mut World, claim: &SignalClaim) -> Option<Option<usize>> {
    run_signal_subgraph_once(world, claim.entity()).and_then(downcast_signal_value::<Option<usize>>)
}

#[derive(Resource, Default)]
struct CapturedIndices(Vec<Broadcast<Option<usize>>>);

#[derive(Resource, Default)]
struct CapturedEnumerated(Vec<(Broadcast<Option<usize>>, i32)>);

fn index_fixture(intersperse: bool) -> (App, MutableVec<i32>, SignalClaim, Vec<Broadcast<Option<usize>>>) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<IndexProbe>());
    let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());
    if intersperse {
        app.init_resource::<CapturedIndices>();
        let producer = source
            .signal_vec()
            .intersperse_with(
                |In(index): In<Broadcast<Option<usize>>>, mut captured: ResMut<CapturedIndices>| {
                    captured.0.push(index);
                    -1
                },
            )
            .register(app.world_mut());
        app.update();
        let indices = core::mem::take(&mut app.world_mut().resource_mut::<CapturedIndices>().0);
        assert_eq!(indices.len(), 2);
        (app, source, producer, indices)
    } else {
        type EnumeratedBatch = Vec<VecDiff<(Broadcast<Option<usize>>, i32)>>;
        app.init_resource::<CapturedEnumerated>();
        let enumerated = source.signal_vec().enumerate().broadcast();
        let producer = enumerated.clone().register(app.world_mut());
        let observer = enumerated
            .for_each(
                |In(diffs): In<EnumeratedBatch>, mut captured: ResMut<CapturedEnumerated>| {
                    for diff in diffs {
                        diff.apply_to_vec(&mut captured.0);
                    }
                },
            )
            .register(app.world_mut());
        app.update();
        observer.release(app.world_mut());
        let values = core::mem::take(&mut app.world_mut().resource_mut::<CapturedEnumerated>().0);
        let indices = values.into_iter().map(|(index, _)| index).collect::<Vec<_>>();
        assert_eq!(indices.len(), 3);
        (app, source, producer, indices)
    }
}

fn independent_index_lifetime(intersperse: bool) {
    let (mut app, source, producer, indices) = index_fixture(intersperse);
    let producer_entity = producer.entity().entity();
    let selected = indices[1].clone();
    let late = indices[0].clone();
    let (selected_claim, selected_log) = register_index_observer(&mut app, selected.clone());
    assert_eq!(observe_index(&mut app, &selected_log), Some(Some(1)));

    source.write(app.world_mut()).insert(0, 5);
    app.update();
    assert_eq!(observe_index(&mut app, &selected_log), Some(Some(2)));
    source.write(app.world_mut()).remove(2);
    app.update();
    assert_eq!(observe_index(&mut app, &selected_log), Some(None));

    producer.release(app.world_mut());
    drop(source);
    drop(indices);
    settle(&mut app);
    assert!(app.world().get_entity(producer_entity).is_err());
    assert_eq!(observe_index(&mut app, &selected_log), Some(None));
    let (late_claim, late_log) = register_index_observer(&mut app, late.clone());
    assert_eq!(observe_index(&mut app, &late_log), Some(Some(1)));

    selected_claim.release(app.world_mut());
    late_claim.release(app.world_mut());
    drop(selected);
    drop(late);
    settle(&mut app);
}

#[test]
fn enumerate_indices_outlive_the_producer_without_retaining_it() {
    independent_index_lifetime(false);
}

#[test]
fn separator_indices_outlive_the_producer_without_retaining_it() {
    independent_index_lifetime(true);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[derive(Resource, Default)]
struct SynchronousIndices(Vec<Option<Option<usize>>>);

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn separator_callback_can_register_and_evaluate_the_current_index_synchronously() {
    let mut app = app();
    app.init_resource::<SynchronousIndices>();
    let source = MutableVec::builder().values([10, 20, 30]).spawn(app.world_mut());
    let producer = source
        .signal_vec()
        .intersperse_with(|In(index): In<Broadcast<Option<usize>>>, world: &mut World| {
            let claim = index.register(world);
            let value = evaluate_index(world, &claim);
            world.resource_mut::<SynchronousIndices>().0.push(value);
            claim.release(world);
            -1
        })
        .register(app.world_mut());
    app.update();
    assert_eq!(
        app.world().resource::<SynchronousIndices>().0,
        [Some(Some(0)), Some(Some(1))]
    );
    source.write(app.world_mut()).insert(0, 5);
    app.update();
    assert_eq!(
        app.world().resource::<SynchronousIndices>().0,
        [Some(Some(0)), Some(Some(1)), Some(Some(0))]
    );
    producer.release(app.world_mut());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
struct Unregistered;

fn entities(world: &mut World) -> Vec<Entity> {
    let mut entities = world.query::<Entity>().iter(world).collect::<Vec<_>>();
    entities.sort();
    entities
}

#[test]
fn vector_snapshot_is_current_and_rejected_planning_does_not_materialize() {
    let mut app = app();
    let source = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
    let signal = source.signal_vec();
    source.write(app.world_mut()).insert(0, 5);
    source.write(app.world_mut()).set(2, 30);
    let before = entities(app.world_mut());
    let materializer = ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let upstream = registrar.materialize_signal_vec(Box::new(signal).into_signal_vec_materializer())?;
        let terminal = registrar.signal_entity(
            upstream,
            |In(_): In<Vec<VecDiff<i32>>>| Some(1),
            OutputDelivery::broadcast(),
        )?;
        registrar.apply_schedule(terminal, Unregistered)
    });
    assert!(matches!(
        materializer.try_register(app.world_mut()),
        Err(SignalRegistrationError::ScheduleNotRegistered)
    ));
    assert_eq!(entities(app.world_mut()), before);
    assert_eq!(&*source.read(app.world()), &[5, 10, 30]);
    source.write(app.world_mut()).push(40);
    assert_eq!(&*source.read(app.world()), &[5, 10, 30, 40]);
}

#[test]
fn map_snapshot_is_current_and_rejected_planning_does_not_materialize() {
    let mut app = app();
    let source = MutableBTreeMap::builder()
        .values([(1, 10), (2, 20)])
        .spawn(app.world_mut());
    let signal = source.signal_map();
    source.write(app.world_mut()).remove(&1);
    source.write(app.world_mut()).insert(2, 30);
    let before = entities(app.world_mut());
    let materializer = ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let upstream = registrar.materialize_signal_map(Box::new(signal).into_signal_map_materializer())?;
        let terminal = registrar.signal_entity(
            upstream,
            |In(_): In<Vec<MapDiff<i32, i32>>>| Some(1),
            OutputDelivery::broadcast(),
        )?;
        registrar.apply_schedule(terminal, Unregistered)
    });
    assert!(matches!(
        materializer.try_register(app.world_mut()),
        Err(SignalRegistrationError::ScheduleNotRegistered)
    ));
    assert_eq!(entities(app.world_mut()), before);
    assert_eq!(
        source
            .read(app.world())
            .iter()
            .map(|(key, value)| (*key, *value))
            .collect::<Vec<_>>(),
        [(2, 30)]
    );
    source.write(app.world_mut()).insert(3, 40);
    assert_eq!(source.read(app.world()).len(), 2);
}

#[test]
fn collection_and_index_public_interfaces_remain_clone_send_sync() {
    fn require<T: Clone + Send + Sync + 'static>(_: &T) {}
    fn require_send_sync<T: Send + Sync + 'static>(_: &T) {}
    fn require_component<T: Component>() {}
    let mut app = app();
    let vector = MutableVec::builder().values([1]).spawn(app.world_mut());
    let map = MutableBTreeMap::builder().values([(1, 2)]).spawn(app.world_mut());
    require(&vector);
    require(&map);
    require_send_sync(&vector.signal_vec());
    require_send_sync(&map.signal_map());
    require_send_sync(&vector.signal_vec().enumerate());
    require_send_sync(
        &vector
            .signal_vec()
            .intersperse_with(|In(index): In<Broadcast<Option<usize>>>| {
                require(&index);
                0
            }),
    );
    require(&vector.signal_vec().broadcast());
    require(&map.signal_map().broadcast());
    require(&vector.signal_vec().enumerate().broadcast());
    require(
        &vector
            .signal_vec()
            .intersperse_with(|In(index): In<Broadcast<Option<usize>>>| {
                require(&index);
                0
            })
            .broadcast(),
    );
    require_component::<MutableVecData<i32>>();
    require_component::<MutableBTreeMapData<i32, i32>>();
}

#[test]
fn vector_rejects_a_missing_data_component_before_materializing() {
    let mut app = app();
    let source = MutableVec::builder().values([10]).spawn(app.world_mut());
    let entity = app
        .world_mut()
        .query_filtered::<Entity, With<MutableVecData<i32>>>()
        .single(app.world())
        .expect("source data");
    app.world_mut().entity_mut(entity).remove::<MutableVecData<i32>>();
    let before = entities(app.world_mut());
    assert!(matches!(
        source.signal_vec().try_register(app.world_mut()),
        Err(SignalRegistrationError::EntityUnavailable)
    ));
    assert_eq!(entities(app.world_mut()), before);
}

#[test]
fn map_rejects_a_missing_data_component_before_materializing() {
    let mut app = app();
    let source = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());
    let entity = app
        .world_mut()
        .query_filtered::<Entity, With<MutableBTreeMapData<i32, i32>>>()
        .single(app.world())
        .expect("source data");
    app.world_mut()
        .entity_mut(entity)
        .remove::<MutableBTreeMapData<i32, i32>>();
    let before = entities(app.world_mut());
    assert!(matches!(
        source.signal_map().try_register(app.world_mut()),
        Err(SignalRegistrationError::EntityUnavailable)
    ));
    assert_eq!(entities(app.world_mut()), before);
}

#[test]
fn vector_rejects_a_foreign_world_even_with_colliding_typed_data() {
    let mut origin = app();
    let mut foreign = app();
    let source = MutableVec::builder().values([10]).spawn(origin.world_mut());
    let neighbor = MutableVec::builder().values([99]).spawn(foreign.world_mut());
    assert_eq!(entities(origin.world_mut()), entities(foreign.world_mut()));
    let before = entities(foreign.world_mut());
    assert!(matches!(
        source.signal_vec().try_register(foreign.world_mut()),
        Err(SignalRegistrationError::EntityUnavailable)
    ));
    assert_eq!(entities(foreign.world_mut()), before);
    assert_eq!(entities(origin.world_mut()), before);
    assert!(
        matches!(
            source.signal_vec().try_register(origin.world_mut()),
            Err(SignalRegistrationError::SharedMaterializerRejected)
        ),
        "the failed foreign-World attempt permanently rejects the shared one-shot initializer"
    );
    assert_eq!(entities(origin.world_mut()), before);
    assert_eq!(entities(foreign.world_mut()), before);
    assert_eq!(&*neighbor.read(foreign.world()), &[99]);
    assert_eq!(&*source.read(origin.world()), &[10]);
}

#[test]
fn map_rejects_a_foreign_world_even_with_colliding_typed_data() {
    let mut origin = app();
    let mut foreign = app();
    let source = MutableBTreeMap::builder().values([(1, 10)]).spawn(origin.world_mut());
    let neighbor = MutableBTreeMap::builder().values([(1, 99)]).spawn(foreign.world_mut());
    assert_eq!(entities(origin.world_mut()), entities(foreign.world_mut()));
    let before = entities(foreign.world_mut());
    assert!(matches!(
        source.signal_map().try_register(foreign.world_mut()),
        Err(SignalRegistrationError::EntityUnavailable)
    ));
    assert_eq!(entities(foreign.world_mut()), before);
    assert_eq!(entities(origin.world_mut()), before);
    assert!(
        matches!(
            source.signal_map().try_register(origin.world_mut()),
            Err(SignalRegistrationError::SharedMaterializerRejected)
        ),
        "the failed foreign-World attempt permanently rejects the shared one-shot initializer"
    );
    assert_eq!(entities(origin.world_mut()), before);
    assert_eq!(entities(foreign.world_mut()), before);
    assert_eq!(neighbor.read(foreign.world()).get(&1), Some(&99));
    assert_eq!(source.read(origin.world()).get(&1), Some(&10));
}

#[test]
fn lazy_vector_retains_only_canonical_contents_and_pending_diffs() {
    let mut app = app();
    let initial = Arc::new(());
    let pushed = Arc::new(());
    let source = MutableVec::builder().values([initial.clone()]).spawn(app.world_mut());
    let lazy = source.signal_vec();
    assert_eq!(
        Arc::strong_count(&initial),
        2,
        "no bootstrap mirror before registration"
    );
    source.write(app.world_mut()).push(pushed.clone());
    assert_eq!(
        Arc::strong_count(&pushed),
        3,
        "one canonical value and one pending diff"
    );
    drop(source);
    settle(&mut app);
    assert_eq!(Arc::strong_count(&initial), 2, "lazy initializer retains source data");
    drop(lazy);
    settle(&mut app);
    assert_eq!(Arc::strong_count(&initial), 1);
    assert_eq!(Arc::strong_count(&pushed), 1);
}

#[test]
fn lazy_map_retains_only_canonical_contents_and_pending_diffs() {
    let mut app = app();
    let initial = Arc::new(());
    let inserted = Arc::new(());
    let source = MutableBTreeMap::builder()
        .values([(1, initial.clone())])
        .spawn(app.world_mut());
    let lazy = source.signal_map();
    assert_eq!(
        Arc::strong_count(&initial),
        2,
        "no bootstrap mirror before registration"
    );
    source.write(app.world_mut()).insert(2, inserted.clone());
    assert_eq!(
        Arc::strong_count(&inserted),
        3,
        "one canonical value and one pending diff"
    );
    drop(source);
    settle(&mut app);
    assert_eq!(Arc::strong_count(&initial), 2, "lazy initializer retains source data");
    drop(lazy);
    settle(&mut app);
    assert_eq!(Arc::strong_count(&initial), 1);
    assert_eq!(Arc::strong_count(&inserted), 1);
}

#[derive(Resource, Default)]
struct VectorBatches(Vec<Vec<VecDiff<i32>>>);

#[derive(Resource, Default)]
struct MapBatches(Vec<Vec<MapDiff<i32, i32>>>);

#[test]
fn vector_bootstrap_and_cursor_capture_the_same_current_component_state() {
    let mut app = app();
    app.init_resource::<VectorBatches>();
    let source = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
    let lazy = source.signal_vec();
    source.write(app.world_mut()).insert(0, 5);
    source.write(app.world_mut()).set(2, 30);
    let claim = lazy
        .for_each(|In(diffs), mut batches: ResMut<VectorBatches>| batches.0.push(diffs))
        .register(app.world_mut());
    app.update();
    let batches = core::mem::take(&mut app.world_mut().resource_mut::<VectorBatches>().0);
    assert_eq!(batches.len(), 1);
    assert!(matches!(batches[0].as_slice(), [VecDiff::Replace { values }] if values == &[5, 10, 30]));
    source.write(app.world_mut()).push(40);
    app.update();
    let batches = core::mem::take(&mut app.world_mut().resource_mut::<VectorBatches>().0);
    assert_eq!(batches.len(), 1);
    assert!(matches!(batches[0].as_slice(), [VecDiff::Push { value: 40 }]));
    claim.release(app.world_mut());
}

#[test]
fn map_bootstrap_and_cursor_capture_the_same_current_component_state() {
    let mut app = app();
    app.init_resource::<MapBatches>();
    let source = MutableBTreeMap::builder()
        .values([(1, 10), (2, 20)])
        .spawn(app.world_mut());
    let lazy = source.signal_map();
    source.write(app.world_mut()).remove(&1);
    source.write(app.world_mut()).insert(2, 30);
    let claim = lazy
        .for_each(|In(diffs), mut batches: ResMut<MapBatches>| batches.0.push(diffs))
        .register(app.world_mut());
    app.update();
    let batches = core::mem::take(&mut app.world_mut().resource_mut::<MapBatches>().0);
    assert_eq!(batches.len(), 1);
    assert!(matches!(batches[0].as_slice(), [MapDiff::Replace { entries }] if entries == &[(2, 30)]));
    source.write(app.world_mut()).insert(3, 40);
    app.update();
    let batches = core::mem::take(&mut app.world_mut().resource_mut::<MapBatches>().0);
    assert_eq!(batches.len(), 1);
    assert!(matches!(batches[0].as_slice(), [MapDiff::Insert { key: 3, value: 40 }]));
    claim.release(app.world_mut());
}
