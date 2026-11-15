use super::{Shape, app, apply};
use alloc::{boxed::Box, collections::BTreeMap, format, vec, vec::Vec};
use bevy_ecs::prelude::*;
use jonmo::{
    DynamicEmissions, DynamicManagerStep, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand,
    KeyedSlotCommand, MapSignalMaterializer, OutputDelivery, ReplayInitialization, SameSignalIdentity,
    ScalarSignalMaterializer, ScalarSlotCommand, VecSignalMaterializer, prelude::*,
};

#[derive(Resource, Default)]
struct Control {
    release_second: bool,
    vector_delta: Option<Vec<VecDiff<i32>>>,
    map_delta: Option<Vec<MapDiff<i32, i32>>>,
}
#[derive(Resource, Default, Debug)]
struct VectorEvents(Vec<(usize, Vec<VecDiff<i32>>)>);
type MapEventBatch = Vec<(usize, Vec<MapDiff<i32, i32>>)>;

#[derive(Resource, Default, Debug)]
struct MapEvents(MapEventBatch);

#[derive(Clone)]
struct VectorSource {
    warming: bool,
}
impl SignalVec for VectorSource {
    type Item = i32;
    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
        VecSignalMaterializer::new(move |registrar| {
            if self.warming {
                registrar.replayable_signal_vec_entity(
                    (),
                    |In(()), control: Res<Control>, mut emitted: Local<bool>| {
                        if control.release_second && !*emitted {
                            *emitted = true;
                            Some(vec![VecDiff::Replace { values: vec![9] }])
                        } else {
                            None
                        }
                    },
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            } else {
                registrar.replayable_signal_vec_source(
                    vec![1],
                    |In(()), mut control: ResMut<Control>| control.vector_delta.take(),
                    OutputDelivery::broadcast(),
                )
            }
        })
    }
}
#[derive(Clone)]
struct MapSource {
    warming: bool,
}
impl SignalMap for MapSource {
    type Key = i32;
    type Value = i32;
    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
        MapSignalMaterializer::new(move |registrar| {
            if self.warming {
                registrar.replayable_signal_map_entity(
                    (),
                    |In(()), control: Res<Control>, mut emitted: Local<bool>| {
                        if control.release_second && !*emitted {
                            *emitted = true;
                            Some(vec![MapDiff::Replace { entries: vec![(9, 90)] }])
                        } else {
                            None
                        }
                    },
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            } else {
                registrar.replayable_signal_map_source(
                    BTreeMap::from([(1, 10)]),
                    |In(()), mut control: ResMut<Control>| control.map_delta.take(),
                    OutputDelivery::broadcast(),
                )
            }
        })
    }
}

struct VectorCohort {
    keyed: bool,
}
impl Signal for VectorCohort {
    type Item = Vec<(usize, Vec<VecDiff<i32>>)>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(move |registrar| {
            let first = VectorSource { warming: false }.cache().broadcast();
            let second = VectorSource { warming: true }.cache().broadcast();
            macro_rules! build {
                ($shape:ident, $command:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut emitted: Local<bool>| {
                            if *emitted {
                                return None;
                            }
                            *emitted = true;
                            Some(DynamicManagerUpdate::new(
                                [$command(first.clone(), second.clone())],
                                DynamicEmissions::<(usize, Vec<VecDiff<i32>>)>::None,
                            ))
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$shape(manager)
                            .replay_signal_vec::<i32>()
                            .adapter(|slot, diffs| DynamicEmissions::One((slot, diffs)))
                            .scalar_output(Some)
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }};
            }
            if self.keyed {
                build!(keyed, |first, second| KeyedSlotCommand::ReplaceAll {
                    entries: vec![(0usize, first), (1, second)]
                })
            } else {
                build!(indexed, |first, second| IndexedSlotCommand::ReplaceAll {
                    signals: vec![first, second]
                })
            }
        })
    }
}
struct MapCohort {
    keyed: bool,
}
impl Signal for MapCohort {
    type Item = Vec<(usize, Vec<MapDiff<i32, i32>>)>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(move |registrar| {
            let first = MapSource { warming: false }.cache().broadcast();
            let second = MapSource { warming: true }.cache().broadcast();
            macro_rules! build {
                ($shape:ident, $command:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut emitted: Local<bool>| {
                            if *emitted {
                                return None;
                            }
                            *emitted = true;
                            Some(DynamicManagerUpdate::new(
                                [$command(first.clone(), second.clone())],
                                DynamicEmissions::<(usize, Vec<MapDiff<i32, i32>>)>::None,
                            ))
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$shape(manager)
                            .replay_signal_map::<i32, i32>()
                            .adapter(|slot, diffs| DynamicEmissions::One((slot, diffs)))
                            .scalar_output(Some)
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }};
            }
            if self.keyed {
                build!(keyed, |first, second| KeyedSlotCommand::ReplaceAll {
                    entries: vec![(0usize, first), (1, second)]
                })
            } else {
                build!(indexed, |first, second| IndexedSlotCommand::ReplaceAll {
                    signals: vec![first, second]
                })
            }
        })
    }
}
fn vector_case(keyed: bool, pending_delta: bool) {
    let mut app = app();
    app.init_resource::<Control>();
    app.init_resource::<VectorEvents>();
    let registration = VectorCohort { keyed }
        .map(
            |In(events): In<Vec<(usize, Vec<VecDiff<i32>>)>>, mut output: ResMut<VectorEvents>| output.0.extend(events),
        )
        .register(app.world_mut());
    app.update();
    assert!(app.world().resource::<VectorEvents>().0.is_empty());
    if pending_delta {
        app.world_mut().resource_mut::<Control>().vector_delta = Some(vec![VecDiff::Push { value: 2 }]);
    }
    app.update();
    assert!(
        app.world().resource::<VectorEvents>().0.is_empty(),
        "second candidate must still block cohort"
    );
    app.world_mut().resource_mut::<Control>().release_second = true;
    for _ in 0..3 {
        app.update();
    }
    let events = &app.world().resource::<VectorEvents>().0;
    println!("ReplaySignalVec keyed={keyed}, pending_delta={pending_delta}: {events:?}");
    let mut states = BTreeMap::<usize, Vec<i32>>::new();
    for (slot, diffs) in events {
        apply(states.entry(*slot).or_default(), diffs);
    }
    println!("ReplaySignalVec reconstructed: {states:?}");
    registration.release(app.world_mut());
    assert_eq!(
        states.get(&1),
        Some(&vec![9]),
        "warming candidate should publish normally"
    );
    let expected = if pending_delta { vec![1, 2] } else { vec![1] };
    assert_eq!(
        states.get(&0),
        Some(&expected),
        "pending delta must not erase initial replay baseline"
    );
}
fn map_case(keyed: bool, pending_delta: bool) {
    let mut app = app();
    app.init_resource::<Control>();
    app.init_resource::<MapEvents>();
    let registration = MapCohort { keyed }
        .map(|In(events): In<MapEventBatch>, mut output: ResMut<MapEvents>| output.0.extend(events))
        .register(app.world_mut());
    app.update();
    assert!(app.world().resource::<MapEvents>().0.is_empty());
    if pending_delta {
        app.world_mut().resource_mut::<Control>().map_delta = Some(vec![MapDiff::Insert { key: 2, value: 20 }]);
    }
    app.update();
    assert!(
        app.world().resource::<MapEvents>().0.is_empty(),
        "second candidate must still block cohort"
    );
    app.world_mut().resource_mut::<Control>().release_second = true;
    for _ in 0..3 {
        app.update();
    }
    let events = &app.world().resource::<MapEvents>().0;
    println!("ReplaySignalMap keyed={keyed}, pending_delta={pending_delta}: {events:?}");
    let mut states = BTreeMap::<usize, BTreeMap<i32, i32>>::new();
    for (slot, diffs) in events {
        let state = states.entry(*slot).or_default();
        for diff in diffs {
            match diff {
                MapDiff::Replace { entries } => *state = entries.iter().copied().collect(),
                MapDiff::Insert { key, value } => {
                    state.insert(*key, *value);
                }
                MapDiff::Update { key, value } => {
                    *state.get_mut(key).expect("update requires an existing key") = *value;
                }
                MapDiff::Remove { key } => {
                    state.remove(key);
                }
                MapDiff::Clear => state.clear(),
            }
        }
    }
    println!("ReplaySignalMap reconstructed: {states:?}");
    registration.release(app.world_mut());
    assert_eq!(
        states.get(&1),
        Some(&BTreeMap::from([(9, 90)])),
        "warming candidate should publish normally"
    );
    let expected = if pending_delta {
        BTreeMap::from([(1, 10), (2, 20)])
    } else {
        BTreeMap::from([(1, 10)])
    };
    assert_eq!(
        states.get(&0),
        Some(&expected),
        "pending delta must not erase initial replay baseline"
    );
}
#[test]
fn vector_indexed_pending_delta() {
    vector_case(false, true);
}
#[test]
fn vector_keyed_pending_delta() {
    vector_case(true, true);
}
#[test]
fn map_indexed_pending_delta() {
    map_case(false, true);
}
#[test]
fn map_keyed_pending_delta() {
    map_case(true, true);
}
#[test]
fn control_vector_indexed_no_delta() {
    vector_case(false, false);
}
#[test]
fn control_vector_keyed_no_delta() {
    vector_case(true, false);
}
#[test]
fn control_map_indexed_no_delta() {
    map_case(false, false);
}
#[test]
fn control_map_keyed_no_delta() {
    map_case(true, false);
}

#[derive(Resource, Default)]
struct HistoryControl {
    vectors: BTreeMap<usize, Vec<VecDiff<i32>>>,
    maps: BTreeMap<usize, Vec<MapDiff<i32, i32>>>,
    scalars: BTreeMap<usize, Option<i32>>,
}

struct HistoryVector {
    id: usize,
    initial: Option<Vec<i32>>,
}
impl SignalVec for HistoryVector {
    type Item = i32;
    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
        VecSignalMaterializer::new(move |registrar| {
            let Self { id, initial } = *self;
            let system = move |In(()), mut control: ResMut<HistoryControl>| control.vectors.remove(&id);
            match initial {
                Some(initial) => registrar.replayable_signal_vec_source(initial, system, OutputDelivery::broadcast()),
                None => registrar.replayable_signal_vec_entity(
                    (),
                    system,
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                ),
            }
        })
    }
}
struct HistoryMap {
    id: usize,
    initial: Option<BTreeMap<i32, i32>>,
}
impl SignalMap for HistoryMap {
    type Key = i32;
    type Value = i32;
    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
        MapSignalMaterializer::new(move |registrar| {
            let Self { id, initial } = *self;
            let system = move |In(()), mut control: ResMut<HistoryControl>| control.maps.remove(&id);
            match initial {
                Some(initial) => registrar.replayable_signal_map_source(initial, system, OutputDelivery::broadcast()),
                None => registrar.replayable_signal_map_entity(
                    (),
                    system,
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                ),
            }
        })
    }
}
struct HistoryScalar {
    id: usize,
    initial: Option<Option<i32>>,
}
impl Signal for HistoryScalar {
    type Item = Option<i32>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(move |registrar| {
            let Self { id, initial } = *self;
            let system = move |In(()), mut control: ResMut<HistoryControl>| control.scalars.remove(&id);
            match initial {
                Some(initial) => registrar.replayable_signal_source(initial, system, OutputDelivery::broadcast()),
                None => registrar.replayable_signal_entity(
                    (),
                    system,
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                ),
            }
        })
    }
}

#[derive(Clone, Debug)]
enum ReplayEvent {
    Manager(usize),
    Vector(usize, Vec<VecDiff<i32>>),
    Map(usize, Vec<MapDiff<i32, i32>>),
    Scalar(usize, Option<i32>),
}
impl PartialEq for ReplayEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Manager(left), Self::Manager(right)) => left == right,
            (Self::Scalar(left_slot, left), Self::Scalar(right_slot, right)) => {
                left_slot == right_slot && left == right
            }
            // Public collection diffs do not implement PartialEq; Debug includes every field.
            (Self::Vector(left_slot, left), Self::Vector(right_slot, right)) => {
                left_slot == right_slot && format!("{left:?}") == format!("{right:?}")
            }
            (Self::Map(left_slot, left), Self::Map(right_slot, right)) => {
                left_slot == right_slot && format!("{left:?}") == format!("{right:?}")
            }
            _ => false,
        }
    }
}
#[derive(Resource, Default, Debug)]
struct ReplayJournal(Vec<ReplayEvent>);

macro_rules! history_operator {
    ($name:ident, $replay:ident::<$($item:ty),+>, $variant:ident, $sources:expr) => {
        struct $name { shape: Shape }
        impl Signal for $name {
            type Item = Vec<ReplayEvent>;
            fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
                ScalarSignalMaterializer::new(move |registrar| {
                    let sources = $sources;
                    macro_rules! build {
                        ($shape:ident, $command:expr, $adapter:expr) => {{
                            let manager = registrar.signal_entity((), move |In(()), mut emitted: Local<bool>| {
                                if core::mem::replace(&mut *emitted, true) { return None; }
                                Some(DynamicManagerUpdate::sequenced(sources.iter().enumerate().map(|(index, signal)| {
                                    DynamicManagerStep::command($command(index, signal.clone()), DynamicEmissions::One(ReplayEvent::Manager(index)))
                                })))
                            }, OutputDelivery::move_only())?;
                            registrar.dynamic_operator(DynamicSignalOperatorSpec::$shape(manager)
                                .$replay::<$($item),+>().adapter($adapter).scalar_output(Some)
                                .same_signal_identity(SameSignalIdentity::KeepCurrent))
                        }};
                    }
                    match self.shape {
                        Shape::Scalar => build!(scalar,
                            |_index, signal| ScalarSlotCommand::Select { signal },
                            |(), value| DynamicEmissions::One(ReplayEvent::$variant(0, value))),
                        Shape::Indexed => build!(indexed,
                            |index, signal| IndexedSlotCommand::Insert { index, signal },
                            |index, value| DynamicEmissions::One(ReplayEvent::$variant(index, value))),
                        Shape::Keyed => build!(keyed,
                            |index, signal| KeyedSlotCommand::Insert { key: 9 - index, signal },
                            |key, value| DynamicEmissions::One(ReplayEvent::$variant(key, value))),
                    }
                })
            }
        }
    };
}

history_operator!(
    VectorHistory,
    replay_signal_vec::<i32>,
    Vector,
    [
        HistoryVector {
            id: 0,
            initial: Some(vec![1])
        }
        .cache()
        .broadcast(),
        HistoryVector { id: 1, initial: None }.cache().broadcast(),
        HistoryVector {
            id: 2,
            initial: Some(vec![])
        }
        .cache()
        .broadcast(),
    ]
);
history_operator!(
    MapHistory,
    replay_signal_map::<i32, i32>,
    Map,
    [
        HistoryMap {
            id: 0,
            initial: Some(BTreeMap::from([(1, 10)]))
        }
        .cache()
        .broadcast(),
        HistoryMap { id: 1, initial: None }.cache().broadcast(),
        HistoryMap {
            id: 2,
            initial: Some(BTreeMap::new())
        }
        .cache()
        .broadcast(),
    ]
);
history_operator!(
    ScalarHistory,
    replay_scalar::<Option<i32>>,
    Scalar,
    [
        HistoryScalar {
            id: 0,
            initial: Some(Some(1))
        }
        .cache()
        .broadcast(),
        HistoryScalar { id: 1, initial: None }.cache().broadcast(),
        HistoryScalar {
            id: 2,
            initial: Some(None)
        }
        .cache()
        .broadcast(),
    ]
);

fn register_history(app: &mut bevy_app::App, signal: impl Signal<Item = Vec<ReplayEvent>>) -> SignalClaim {
    app.init_resource::<HistoryControl>();
    app.init_resource::<ReplayJournal>();
    signal
        .map(|In(events): In<Vec<ReplayEvent>>, mut journal: ResMut<ReplayJournal>| journal.0.extend(events))
        .register(app.world_mut())
}

#[derive(Clone, Copy, Debug)]
enum History {
    Deltas,
    ReplaceSuffix,
    ClearSuffix,
    EmptyReplace,
    EmptyClear,
}
fn vector_history(history: History) -> Vec<Vec<VecDiff<i32>>> {
    let mut batches = vec![vec![VecDiff::Push { value: 2 }], vec![VecDiff::Push { value: 3 }]];
    match history {
        History::Deltas => batches.push(vec![VecDiff::Push { value: 4 }, VecDiff::Push { value: 5 }]),
        History::ReplaceSuffix => batches.extend([
            vec![VecDiff::Replace { values: vec![6, 7] }, VecDiff::Push { value: 8 }],
            vec![VecDiff::Push { value: 10 }],
        ]),
        History::ClearSuffix => batches.extend([
            vec![VecDiff::Clear, VecDiff::Push { value: 8 }],
            vec![VecDiff::Push { value: 10 }],
        ]),
        History::EmptyReplace => batches.push(vec![VecDiff::Replace { values: vec![] }]),
        History::EmptyClear => batches.push(vec![VecDiff::Clear]),
    }
    batches
}
fn map_history(history: History) -> Vec<Vec<MapDiff<i32, i32>>> {
    let mut batches = vec![
        vec![MapDiff::Insert { key: 2, value: 20 }],
        vec![MapDiff::Insert { key: 3, value: 30 }],
    ];
    match history {
        History::Deltas => batches.push(vec![
            MapDiff::Insert { key: 4, value: 40 },
            MapDiff::Insert { key: 5, value: 50 },
        ]),
        History::ReplaceSuffix => batches.extend([
            vec![
                MapDiff::Replace {
                    entries: vec![(6, 60), (7, 70)],
                },
                MapDiff::Insert { key: 8, value: 80 },
            ],
            vec![MapDiff::Insert { key: 10, value: 100 }],
        ]),
        History::ClearSuffix => batches.extend([
            vec![MapDiff::Clear, MapDiff::Insert { key: 8, value: 80 }],
            vec![MapDiff::Insert { key: 10, value: 100 }],
        ]),
        History::EmptyReplace => batches.push(vec![MapDiff::Replace { entries: vec![] }]),
        History::EmptyClear => batches.push(vec![MapDiff::Clear]),
    }
    batches
}
fn apply_map(state: &mut BTreeMap<i32, i32>, diffs: &[MapDiff<i32, i32>]) {
    for diff in diffs {
        match diff {
            MapDiff::Replace { entries } => *state = entries.iter().copied().collect(),
            MapDiff::Insert { key, value } => {
                state.insert(*key, *value);
            }
            MapDiff::Update { key, value } => {
                *state.get_mut(key).expect("update requires existing key") = *value;
            }
            MapDiff::Remove { key } => {
                assert!(state.remove(key).is_some(), "remove requires existing key");
            }
            MapDiff::Clear => state.clear(),
        }
    }
}
fn replay_slot(shape: Shape, command: usize) -> usize {
    match shape {
        Shape::Scalar => 0,
        Shape::Indexed => command,
        Shape::Keyed => 9 - command,
    }
}

fn collection_history_neighbor(shape: Shape, map: bool, histories: &[History]) {
    let mut failures = Vec::new();
    for history in histories {
        let mut app = app();
        let registration = if map {
            register_history(&mut app, MapHistory { shape })
        } else {
            register_history(&mut app, VectorHistory { shape })
        };
        app.update();
        assert!(
            app.world().resource::<ReplayJournal>().0.is_empty(),
            "warming middle command blocks whole source-ordered journal"
        );
        let mut expected_vector = vec![1];
        let mut expected_map = BTreeMap::from([(1, 10)]);
        if map {
            for batch in map_history(*history) {
                apply_map(&mut expected_map, &batch);
                app.world_mut().resource_mut::<HistoryControl>().maps.insert(0, batch);
                app.update();
                assert!(app.world().resource::<ReplayJournal>().0.is_empty());
            }
            app.world_mut()
                .resource_mut::<HistoryControl>()
                .maps
                .insert(1, vec![MapDiff::Replace { entries: vec![(9, 90)] }]);
        } else {
            for batch in vector_history(*history) {
                apply(&mut expected_vector, &batch);
                app.world_mut()
                    .resource_mut::<HistoryControl>()
                    .vectors
                    .insert(0, batch);
                app.update();
                assert!(app.world().resource::<ReplayJournal>().0.is_empty());
            }
            app.world_mut()
                .resource_mut::<HistoryControl>()
                .vectors
                .insert(1, vec![VecDiff::Replace { values: vec![9] }]);
        }
        for _ in 0..3 {
            app.update();
        }
        let events = &app.world().resource::<ReplayJournal>().0;
        println!("{shape:?} map={map} history={history:?}: {events:?}");
        let expected = if map {
            vec![
                ReplayEvent::Manager(0),
                ReplayEvent::Map(
                    replay_slot(shape, 0),
                    vec![MapDiff::Replace {
                        entries: expected_map.into_iter().collect(),
                    }],
                ),
                ReplayEvent::Manager(1),
                ReplayEvent::Map(replay_slot(shape, 1), vec![MapDiff::Replace { entries: vec![(9, 90)] }]),
                ReplayEvent::Manager(2),
                ReplayEvent::Map(replay_slot(shape, 2), vec![MapDiff::Replace { entries: vec![] }]),
            ]
        } else {
            vec![
                ReplayEvent::Manager(0),
                ReplayEvent::Vector(
                    replay_slot(shape, 0),
                    vec![VecDiff::Replace {
                        values: expected_vector,
                    }],
                ),
                ReplayEvent::Manager(1),
                ReplayEvent::Vector(replay_slot(shape, 1), vec![VecDiff::Replace { values: vec![9] }]),
                ReplayEvent::Manager(2),
                ReplayEvent::Vector(replay_slot(shape, 2), vec![VecDiff::Replace { values: vec![] }]),
            ]
        };
        // Initialize once per command, not per batch: a candidate may publish a baseline followed
        // by several deltas. Scalar slots reuse context 0, so slot identity cannot delimit
        // commands.
        let normalize = |events: &[ReplayEvent]| -> Vec<ReplayEvent> {
            let mut commands: Vec<(usize, Vec<&ReplayEvent>)> = Vec::new();
            for event in events {
                match event {
                    ReplayEvent::Manager(command) => commands.push((*command, Vec::new())),
                    event => commands.last_mut().expect("manager precedes candidate").1.push(event),
                }
            }
            let mut normalized = Vec::new();
            for (command, batches) in commands {
                assert!(!batches.is_empty(), "every candidate must initialize");
                let slot = replay_slot(shape, command);
                let mut vector = vec![-999];
                let mut entries = BTreeMap::from([(-999, -999)]);
                for batch in batches {
                    match batch {
                        ReplayEvent::Vector(actual_slot, diffs) if !map => {
                            assert_eq!(*actual_slot, slot);
                            apply(&mut vector, diffs);
                        }
                        ReplayEvent::Map(actual_slot, diffs) if map => {
                            assert_eq!(*actual_slot, slot);
                            apply_map(&mut entries, diffs);
                        }
                        event => panic!("unexpected candidate event: {event:?}"),
                    }
                }
                normalized.push(ReplayEvent::Manager(command));
                normalized.push(if map {
                    ReplayEvent::Map(
                        slot,
                        vec![MapDiff::Replace {
                            entries: entries.into_iter().collect(),
                        }],
                    )
                } else {
                    ReplayEvent::Vector(slot, vec![VecDiff::Replace { values: vector }])
                });
            }
            normalized
        };
        if normalize(events) != expected {
            failures.push(format!(
                "{history:?}: {events:?}; expected reconstructed journal {expected:?}"
            ));
        }
        let count = events.len();
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<ReplayJournal>().0.len(),
            count,
            "no duplicate replay on idle updates"
        );
        registration.release(app.world_mut());
    }
    assert!(failures.is_empty(), "{shape:?}, map={map}\n{}", failures.join("\n"));
}

const NONEMPTY_HISTORIES: &[History] = &[History::Deltas, History::ReplaceSuffix, History::ClearSuffix];
const EMPTY_HISTORIES: &[History] = &[History::EmptyReplace, History::EmptyClear];
#[test]
fn neighbor_vector_scalar_multicommand_pending_histories() {
    collection_history_neighbor(Shape::Scalar, false, NONEMPTY_HISTORIES);
}
#[test]
fn neighbor_vector_indexed_pending_histories() {
    collection_history_neighbor(Shape::Indexed, false, NONEMPTY_HISTORIES);
}
#[test]
fn neighbor_vector_keyed_pending_histories_source_order() {
    collection_history_neighbor(Shape::Keyed, false, NONEMPTY_HISTORIES);
}
#[test]
fn neighbor_map_scalar_multicommand_pending_histories() {
    collection_history_neighbor(Shape::Scalar, true, NONEMPTY_HISTORIES);
}
#[test]
fn neighbor_map_indexed_pending_histories() {
    collection_history_neighbor(Shape::Indexed, true, NONEMPTY_HISTORIES);
}
#[test]
fn neighbor_map_keyed_pending_histories_source_order() {
    collection_history_neighbor(Shape::Keyed, true, NONEMPTY_HISTORIES);
}
#[test]
fn neighbor_vector_scalar_authoritative_empty_control() {
    collection_history_neighbor(Shape::Scalar, false, EMPTY_HISTORIES);
}
#[test]
fn neighbor_vector_indexed_authoritative_empty_control() {
    collection_history_neighbor(Shape::Indexed, false, EMPTY_HISTORIES);
}
#[test]
fn neighbor_vector_keyed_authoritative_empty_control() {
    collection_history_neighbor(Shape::Keyed, false, EMPTY_HISTORIES);
}
#[test]
fn neighbor_map_scalar_authoritative_empty_control() {
    collection_history_neighbor(Shape::Scalar, true, EMPTY_HISTORIES);
}
#[test]
fn neighbor_map_indexed_authoritative_empty_control() {
    collection_history_neighbor(Shape::Indexed, true, EMPTY_HISTORIES);
}
#[test]
fn neighbor_map_keyed_authoritative_empty_control() {
    collection_history_neighbor(Shape::Keyed, true, EMPTY_HISTORIES);
}

fn scalar_none_neighbor(shape: Shape) {
    let mut app = app();
    let registration = register_history(&mut app, ScalarHistory { shape });
    app.update();
    assert!(app.world().resource::<ReplayJournal>().0.is_empty());
    for value in [Some(2), Some(3), None] {
        app.world_mut()
            .resource_mut::<HistoryControl>()
            .scalars
            .insert(0, value);
        app.update();
        assert!(app.world().resource::<ReplayJournal>().0.is_empty());
    }
    app.world_mut().resource_mut::<HistoryControl>().scalars.insert(1, None);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        app.world().resource::<ReplayJournal>().0,
        vec![
            ReplayEvent::Manager(0),
            ReplayEvent::Scalar(replay_slot(shape, 0), None),
            ReplayEvent::Manager(1),
            ReplayEvent::Scalar(replay_slot(shape, 1), None),
            ReplayEvent::Manager(2),
            ReplayEvent::Scalar(replay_slot(shape, 2), None),
        ],
        "Ready(None) must complete pending work, not remain warming; command order wins over readiness and key order"
    );
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(app.world().resource::<ReplayJournal>().0.len(), 6);
    registration.release(app.world_mut());
}
#[test]
fn neighbor_scalar_slots_ready_none_source_order_control() {
    scalar_none_neighbor(Shape::Scalar);
}
#[test]
fn neighbor_indexed_slots_ready_none_source_order_control() {
    scalar_none_neighbor(Shape::Indexed);
}
#[test]
fn neighbor_keyed_slots_ready_none_source_order_control() {
    scalar_none_neighbor(Shape::Keyed);
}

macro_rules! transient_collection_operator {
    ($name:ident, $replay:ident::<$($item:ty),+>, $variant:ident, $sources:expr) => {
        struct $name {
            shape: Shape,
            intervening_updates: bool,
        }
        impl Signal for $name {
            type Item = Vec<ReplayEvent>;
            fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
                ScalarSignalMaterializer::new(move |registrar| {
                    let [transient, warming] = $sources;
                    macro_rules! build {
                        ($policy:ident, $insert:expr, $remove:expr, $adapter:expr) => {{
                            let manager = registrar.signal_entity((), move |In(()), mut update: Local<usize>| {
                                let phase = *update;
                                *update += 1;
                                match phase {
                                    0 => Some(DynamicManagerUpdate::sequenced([
                                        DynamicManagerStep::command($insert(0, transient.clone()), DynamicEmissions::One(ReplayEvent::Manager(0))),
                                        DynamicManagerStep::command($remove, DynamicEmissions::One(ReplayEvent::Manager(1))),
                                        DynamicManagerStep::command($insert(1, warming.clone()), DynamicEmissions::One(ReplayEvent::Manager(2))),
                                    ])),
                                    2 if self.intervening_updates => Some(DynamicManagerUpdate::sequenced([
                                        DynamicManagerStep::emissions(DynamicEmissions::One(ReplayEvent::Manager(3))),
                                    ])),
                                    3 if self.intervening_updates => Some(DynamicManagerUpdate::new([], DynamicEmissions::None)),
                                    4 if self.intervening_updates => Some(DynamicManagerUpdate::sequenced([])),
                                    _ => None,
                                }
                            }, OutputDelivery::move_only())?;
                            registrar.dynamic_operator(DynamicSignalOperatorSpec::$policy(manager)
                                .$replay::<$($item),+>()
                                .adapter($adapter)
                                .scalar_output(Some)
                                .same_signal_identity(SameSignalIdentity::KeepCurrent))
                        }};
                    }
                    match self.shape {
                        Shape::Scalar => build!(scalar,
                            |_command, signal| ScalarSlotCommand::Select { signal },
                            ScalarSlotCommand::Clear,
                            |(), value| DynamicEmissions::One(ReplayEvent::$variant(0, value))),
                        Shape::Indexed => build!(indexed,
                            |_command, signal| IndexedSlotCommand::Insert { index: 0, signal },
                            IndexedSlotCommand::Remove { index: 0 },
                            |index, value| DynamicEmissions::One(ReplayEvent::$variant(index, value))),
                        Shape::Keyed => build!(keyed,
                            |command, signal| KeyedSlotCommand::Insert { key: 7usize + command, signal },
                            KeyedSlotCommand::Remove { key: 7usize },
                            |key, value| DynamicEmissions::One(ReplayEvent::$variant(key, value))),
                    }
                })
            }
        }
    };
}
transient_collection_operator!(
    VectorTransient,
    replay_signal_vec::<i32>,
    Vector,
    [
        HistoryVector {
            id: 0,
            initial: Some(vec![1])
        }
        .cache()
        .broadcast(),
        HistoryVector { id: 1, initial: None }.cache().broadcast(),
    ]
);
transient_collection_operator!(
    MapTransient,
    replay_signal_map::<i32, i32>,
    Map,
    [
        HistoryMap {
            id: 0,
            initial: Some(BTreeMap::from([(1, 10)]))
        }
        .cache()
        .broadcast(),
        HistoryMap { id: 1, initial: None }.cache().broadcast(),
    ]
);

fn collection_transient_case(shape: Shape, map: bool, intervening_updates: bool) {
    let mut app = app();
    let registration = if map {
        register_history(
            &mut app,
            MapTransient {
                shape,
                intervening_updates,
            },
        )
    } else {
        register_history(
            &mut app,
            VectorTransient {
                shape,
                intervening_updates,
            },
        )
    };
    let transient_slot = if matches!(shape, Shape::Keyed) { 7 } else { 0 };
    let current_slot = if matches!(shape, Shape::Keyed) { 8 } else { 0 };
    let mut expected = vec![ReplayEvent::Manager(0)];
    expected.push(if map {
        ReplayEvent::Map(transient_slot, vec![MapDiff::Replace { entries: vec![(1, 10)] }])
    } else {
        ReplayEvent::Vector(transient_slot, vec![VecDiff::Replace { values: vec![1] }])
    });
    app.update();
    assert!(app.world().resource::<ReplayJournal>().0.is_empty());
    for phase in 1..=4 {
        if phase == 1 || phase == 3 {
            if map {
                let batch = if phase == 1 {
                    vec![
                        MapDiff::Insert { key: 2, value: 20 },
                        MapDiff::Insert { key: 3, value: 30 },
                    ]
                } else {
                    vec![MapDiff::Remove { key: 1 }, MapDiff::Insert { key: 2, value: 21 }]
                };
                app.world_mut()
                    .resource_mut::<HistoryControl>()
                    .maps
                    .insert(0, batch.clone());
                expected.push(ReplayEvent::Map(transient_slot, batch));
            } else {
                let batch = if phase == 1 {
                    vec![VecDiff::Push { value: 2 }, VecDiff::Push { value: 3 }]
                } else {
                    vec![
                        VecDiff::RemoveAt { index: 0 },
                        VecDiff::UpdateAt { index: 0, value: 21 },
                    ]
                };
                app.world_mut()
                    .resource_mut::<HistoryControl>()
                    .vectors
                    .insert(0, batch.clone());
                expected.push(ReplayEvent::Vector(transient_slot, batch));
            }
        }
        app.update();
        assert!(
            app.world().resource::<ReplayJournal>().0.is_empty(),
            "other warming selection still blocks publication"
        );
    }
    expected.extend([ReplayEvent::Manager(1), ReplayEvent::Manager(2)]);
    if map {
        let batch = vec![MapDiff::Replace { entries: vec![(9, 90)] }];
        app.world_mut()
            .resource_mut::<HistoryControl>()
            .maps
            .insert(1, batch.clone());
        expected.push(ReplayEvent::Map(current_slot, batch));
    } else {
        let batch = vec![VecDiff::Replace { values: vec![9] }];
        app.world_mut()
            .resource_mut::<HistoryControl>()
            .vectors
            .insert(1, batch.clone());
        expected.push(ReplayEvent::Vector(current_slot, batch));
    }
    if intervening_updates {
        expected.push(ReplayEvent::Manager(3));
    }
    for _ in 0..3 {
        app.update();
    }
    let initialized = app.world().resource::<ReplayJournal>().0.clone();
    println!("{shape:?} map={map} intervening_updates={intervening_updates} transient history: {initialized:?}");
    let live_event = if map {
        app.world_mut().resource_mut::<HistoryControl>().maps.extend([
            (0, vec![MapDiff::Insert { key: 4, value: 40 }]),
            (1, vec![MapDiff::Insert { key: 10, value: 100 }]),
        ]);
        ReplayEvent::Map(current_slot, vec![MapDiff::Insert { key: 10, value: 100 }])
    } else {
        app.world_mut().resource_mut::<HistoryControl>().vectors.extend([
            (0, vec![VecDiff::Push { value: 4 }]),
            (1, vec![VecDiff::Push { value: 10 }]),
        ]);
        ReplayEvent::Vector(current_slot, vec![VecDiff::Push { value: 10 }])
    };
    for _ in 0..3 {
        app.update();
    }
    let live = app.world().resource::<ReplayJournal>().0.clone();
    registration.release(app.world_mut());
    assert_eq!(
        initialized, expected,
        "removed transient retains baseline and exact delta batches at its command-time position"
    );
    expected.push(live_event);
    assert_eq!(
        live, expected,
        "removed transient stops forwarding after initialization"
    );
}
macro_rules! collection_transient_test {
    ($name:ident, $shape:expr, $map:expr, $updates:expr) => {
        #[test]
        fn $name() {
            collection_transient_case($shape, $map, $updates);
        }
    };
}
collection_transient_test!(vector_scalar_transient_pending_history, Shape::Scalar, false, true);
collection_transient_test!(vector_indexed_transient_pending_history, Shape::Indexed, false, true);
collection_transient_test!(vector_keyed_transient_pending_history, Shape::Keyed, false, true);
collection_transient_test!(map_scalar_transient_pending_history, Shape::Scalar, true, true);
collection_transient_test!(map_indexed_transient_pending_history, Shape::Indexed, true, true);
collection_transient_test!(map_keyed_transient_pending_history, Shape::Keyed, true, true);
collection_transient_test!(
    vector_scalar_transient_pending_history_control,
    Shape::Scalar,
    false,
    false
);
collection_transient_test!(
    vector_indexed_transient_pending_history_control,
    Shape::Indexed,
    false,
    false
);
collection_transient_test!(
    vector_keyed_transient_pending_history_control,
    Shape::Keyed,
    false,
    false
);
collection_transient_test!(map_scalar_transient_pending_history_control, Shape::Scalar, true, false);
collection_transient_test!(
    map_indexed_transient_pending_history_control,
    Shape::Indexed,
    true,
    false
);
collection_transient_test!(map_keyed_transient_pending_history_control, Shape::Keyed, true, false);
