//! Public regressions for dynamic cancellation, identity normalization, and pending replay.

extern crate alloc;

#[path = "dynamic_semantics/replay.rs"]
mod replay;
use alloc::{boxed::Box, collections::BTreeMap, format, string::String, vec, vec::Vec};
use bevy_app::App;
use bevy_ecs::prelude::*;
use jonmo::{
    DynamicEmissions, DynamicManagerStep, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand,
    KeyedSlotCommand, OutputDelivery, SameSignalIdentity, ScalarSignalMaterializer, ScalarSlotCommand, prelude::*,
};

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Readiness>();
    app.init_resource::<VectorOutput>();
    app.init_resource::<ScalarOutput>();
    app
}

#[derive(Resource, Default)]
struct Readiness(BTreeMap<i32, i32>);
#[derive(Resource, Default, Debug)]
struct VectorOutput {
    state: Vec<i32>,
    batches: Vec<Vec<VecDiff<i32>>>,
}
#[derive(Resource, Default, Debug)]
struct ScalarOutput(Vec<i32>);

fn apply(values: &mut Vec<i32>, diffs: &[VecDiff<i32>]) {
    for diff in diffs {
        match diff {
            VecDiff::Replace { values: replacement } => *values = replacement.clone(),
            VecDiff::InsertAt { index, value } => values.insert(*index, *value),
            VecDiff::UpdateAt { index, value } => values[*index] = *value,
            VecDiff::RemoveAt { index } => {
                values.remove(*index);
            }
            VecDiff::Move { old_index, new_index } => {
                let value = values.remove(*old_index);
                values.insert(*new_index, value);
            }
            VecDiff::Push { value } => values.push(*value),
            VecDiff::Pop => {
                assert!(values.pop().is_some());
            }
            VecDiff::Clear => values.clear(),
        }
    }
}
fn capture(In(diffs): In<Vec<VecDiff<i32>>>, mut output: ResMut<VectorOutput>) {
    apply(&mut output.state, &diffs);
    output.batches.push(diffs);
}
fn scalar_capture(In(values): In<Vec<i32>>, mut output: ResMut<ScalarOutput>) {
    output.0.extend(values);
}
fn warming(key: i32) -> ReplayableBroadcast<i32> {
    signal::from_system(move |In(()), readiness: Res<Readiness>| readiness.0.get(&key).copied())
        .dedupe()
        .cache()
        .broadcast()
}
fn ready(value: i32) -> ReplayableBroadcast<i32> {
    signal::once(value).cache().broadcast()
}

#[test]
fn map_warming_insert_then_remove_must_preserve_published_x() {
    let mut app = app();
    app.world_mut().resource_mut::<Readiness>().0.insert(10, 10);
    let source = MutableVec::builder().values([10]).spawn(app.world_mut());
    let registration = source
        .signal_vec()
        .map_signal(|In(value): In<i32>| warming(value))
        .for_each(capture)
        .register(app.world_mut());
    app.update();
    println!("map initial: {:?}", app.world().resource::<VectorOutput>());
    assert_eq!(app.world().resource::<VectorOutput>().state, [10]);
    source.write(app.world_mut()).insert(0, 20);
    app.update();
    println!("map warming A inserted: {:?}", app.world().resource::<VectorOutput>());
    assert_eq!(app.world().resource::<VectorOutput>().state, [10]);
    source.write(app.world_mut()).remove(0);
    app.update();
    println!(
        "map A removed, source={:?}, output={:?}",
        &*source.read(app.world()),
        app.world().resource::<VectorOutput>()
    );
    for _ in 0..3 {
        app.update();
    }
    registration.release(app.world_mut());
    assert_eq!(
        app.world().resource::<VectorOutput>().state,
        [10],
        "removing unpublished warming A must not remove published X"
    );
}

#[test]
fn filter_warming_replacement_then_remove_must_not_leave_ghost() {
    let mut app = app();
    app.world_mut().resource_mut::<Readiness>().0.insert(1, 1);
    let source = MutableVec::builder().values([1]).spawn(app.world_mut());
    let registration = source
        .signal_vec()
        .filter_signal(|In(value): In<i32>| {
            signal::from_system(move |In(()), readiness: Res<Readiness>| readiness.0.get(&value).map(|_| true))
                .dedupe()
                .cache()
        })
        .for_each(capture)
        .register(app.world_mut());
    app.update();
    println!("filter initial: {:?}", app.world().resource::<VectorOutput>());
    assert_eq!(app.world().resource::<VectorOutput>().state, [1]);
    source.write(app.world_mut()).set(0, 2);
    app.update();
    println!(
        "filter warming predicate replacement: {:?}",
        app.world().resource::<VectorOutput>()
    );
    source.write(app.world_mut()).remove(0);
    app.update();
    println!(
        "filter removed, source={:?}, output={:?}",
        &*source.read(app.world()),
        app.world().resource::<VectorOutput>()
    );
    for _ in 0..3 {
        app.update();
    }
    registration.release(app.world_mut());
    assert!(
        app.world().resource::<VectorOutput>().state.is_empty(),
        "empty source must not retain published ghost"
    );
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Scalar,
    Indexed,
    Keyed,
}
struct KeepCurrent {
    shape: Shape,
    long: bool,
    warming_b: bool,
}
impl Signal for KeepCurrent {
    type Item = Vec<i32>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(move |registrar| {
            let current = ready(10);
            let second = if self.warming_b { warming(20) } else { ready(20) };
            let third = ready(30);
            macro_rules! build {
                ($shape:ident, $initial:expr, $replace:expr, $adapter:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut update: Local<usize>| {
                            let commands = match *update {
                                0 => Some(vec![$initial(current.clone())]),
                                1 => {
                                    let mut commands = vec![$replace(second.clone())];
                                    if self.long {
                                        commands.push($replace(third.clone()));
                                    }
                                    commands.push($replace(current.clone()));
                                    Some(commands)
                                }
                                _ => None,
                            };
                            *update += 1;
                            commands.map(|commands| DynamicManagerUpdate::new(commands, DynamicEmissions::<i32>::None))
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$shape(manager)
                            .replay_scalar::<i32>()
                            .adapter($adapter)
                            .scalar_output(Some)
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }};
            }
            match self.shape {
                Shape::Scalar => build!(
                    scalar,
                    |signal| ScalarSlotCommand::Select { signal },
                    |signal| ScalarSlotCommand::Select { signal },
                    |(), value| DynamicEmissions::One(value)
                ),
                Shape::Indexed => build!(
                    indexed,
                    |signal| IndexedSlotCommand::Insert { index: 0, signal },
                    |signal| IndexedSlotCommand::Replace { index: 0, signal },
                    |_: usize, value| DynamicEmissions::One(value)
                ),
                Shape::Keyed => build!(
                    keyed,
                    |signal| KeyedSlotCommand::Insert { key: 0usize, signal },
                    |signal| KeyedSlotCommand::Replace { key: 0usize, signal },
                    |_: usize, value| DynamicEmissions::One(value)
                ),
            }
        })
    }
}
fn keep_current_case(shape: Shape, long: bool, warming_b: bool) {
    let mut app = app();
    let registration = KeepCurrent { shape, long, warming_b }
        .map(scalar_capture)
        .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<ScalarOutput>().0, [10]);
    app.update();
    println!(
        "KeepCurrent {shape:?}, long={long}, warming_b={warming_b}, after batch: {:?}",
        app.world().resource::<ScalarOutput>()
    );
    app.world_mut().resource_mut::<Readiness>().0.insert(20, 20);
    for _ in 0..4 {
        app.update();
    }
    println!(
        "KeepCurrent {shape:?}, after late B and four updates: {:?}",
        app.world().resource::<ScalarOutput>()
    );
    registration.release(app.world_mut());
    assert_eq!(
        app.world().resource::<ScalarOutput>().0,
        [10],
        "same-batch return to current A must discard transient candidates"
    );
}
#[test]
fn keep_scalar_long_ready() {
    keep_current_case(Shape::Scalar, true, false);
}
#[test]
fn keep_indexed_long_ready() {
    keep_current_case(Shape::Indexed, true, false);
}
#[test]
fn keep_keyed_long_ready() {
    keep_current_case(Shape::Keyed, true, false);
}
#[test]
fn keep_scalar_long_warming_b() {
    keep_current_case(Shape::Scalar, true, true);
}
#[test]
fn keep_indexed_long_warming_b() {
    keep_current_case(Shape::Indexed, true, true);
}
#[test]
fn keep_keyed_long_warming_b() {
    keep_current_case(Shape::Keyed, true, true);
}
#[test]
fn control_keep_scalar_short() {
    keep_current_case(Shape::Scalar, false, false);
}
#[test]
fn control_keep_indexed_short() {
    keep_current_case(Shape::Indexed, false, false);
}
#[test]
fn control_keep_keyed_short() {
    keep_current_case(Shape::Keyed, false, false);
}

fn check_vector_phase(app: &mut App, label: &str, state: &[i32], diffs: &[VecDiff<i32>], failures: &mut Vec<String>) {
    let mut output = app.world_mut().resource_mut::<VectorOutput>();
    let actual_diffs: Vec<_> = core::mem::take(&mut output.batches).into_iter().flatten().collect();
    if output.state != state || format!("{actual_diffs:?}") != format!("{diffs:?}") {
        failures.push(format!(
            "{label}: state {:?}, diffs {actual_diffs:?}; expected state {state:?}, diffs {diffs:?}",
            output.state
        ));
    }
}

#[test]
fn neighbor_map_multiple_pending_edits_cancel_insertion_preserve_siblings_and_late_readiness() {
    let mut app = app();
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .extend([(10, 10), (40, 40), (70, 70)]);
    let source = MutableVec::builder().values([10, 40, 70]).spawn(app.world_mut());
    let registration = source
        .signal_vec()
        .map_signal(|In(value): In<i32>| warming(value))
        .for_each(capture)
        .register(app.world_mut());
    let mut failures = Vec::new();
    app.update();
    check_vector_phase(
        &mut app,
        "initial",
        &[10, 40, 70],
        &[VecDiff::Replace {
            values: vec![10, 40, 70],
        }],
        &mut failures,
    );
    {
        let mut values = source.write(app.world_mut());
        values.insert(0, 20);
        values.insert(2, 30);
    }
    app.update();
    check_vector_phase(&mut app, "two warming insertions", &[10, 40, 70], &[], &mut failures);
    source.write(app.world_mut()).set(0, 21);
    app.update();
    check_vector_phase(
        &mut app,
        "replace unpublished insertion",
        &[10, 40, 70],
        &[],
        &mut failures,
    );
    source.write(app.world_mut()).remove(0);
    app.update();
    check_vector_phase(&mut app, "cancel first insertion", &[10, 40, 70], &[], &mut failures);
    app.world_mut().resource_mut::<Readiness>().0.insert(30, 30);
    for _ in 0..3 {
        app.update();
    }
    check_vector_phase(
        &mut app,
        "surviving insertion ready",
        &[10, 30, 40, 70],
        &[VecDiff::InsertAt { index: 1, value: 30 }],
        &mut failures,
    );
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .extend([(20, 200), (21, 210)]);
    for _ in 0..3 {
        app.update();
    }
    check_vector_phase(&mut app, "cancelled readiness", &[10, 30, 40, 70], &[], &mut failures);
    registration.release(app.world_mut());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn filter_neighbor(initially_included: bool, replacement: Option<bool>, shift_while_warming: bool) {
    let mut app = app();
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .extend([(1, i32::from(initially_included)), (3, 0), (4, 1), (5, 1)]);
    let source = MutableVec::builder().values([1, 3, 5]).spawn(app.world_mut());
    let registration = source
        .signal_vec()
        .filter_signal(|In(value): In<i32>| {
            signal::from_system(move |In(()), readiness: Res<Readiness>| {
                readiness.0.get(&value).map(|value| *value != 0)
            })
            .dedupe()
            .cache()
        })
        .for_each(capture)
        .register(app.world_mut());
    let mut failures = Vec::new();
    let mut expected = if initially_included { vec![1, 5] } else { vec![5] };
    app.update();
    check_vector_phase(
        &mut app,
        "initial",
        &expected,
        &[VecDiff::Replace {
            values: expected.clone(),
        }],
        &mut failures,
    );
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .insert(1, i32::from(!initially_included));
    app.update();
    let transition = if initially_included {
        expected.remove(0);
        VecDiff::RemoveAt { index: 0 }
    } else {
        expected.insert(0, 1);
        VecDiff::InsertAt { index: 0, value: 1 }
    };
    check_vector_phase(
        &mut app,
        "predicate transition",
        &expected,
        &[transition],
        &mut failures,
    );
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .insert(1, i32::from(initially_included));
    app.update();
    let transition = if initially_included {
        expected.insert(0, 1);
        VecDiff::InsertAt { index: 0, value: 1 }
    } else {
        expected.remove(0);
        VecDiff::RemoveAt { index: 0 }
    };
    check_vector_phase(&mut app, "predicate restored", &expected, &[transition], &mut failures);
    source.write(app.world_mut()).remove(1);
    app.update();
    check_vector_phase(&mut app, "remove hidden sibling", &expected, &[], &mut failures);
    if shift_while_warming {
        source.write(app.world_mut()).set(0, 2);
        app.update();
        check_vector_phase(&mut app, "warming before shift", &expected, &[], &mut failures);
    }
    source.write(app.world_mut()).insert(0, 4);
    app.update();
    // A pending cohort may defer this insertion until its warming member resolves or is removed.
    if !shift_while_warming {
        expected.insert(0, 4);
        check_vector_phase(
            &mut app,
            "insert visible sibling",
            &expected,
            &[VecDiff::InsertAt { index: 0, value: 4 }],
            &mut failures,
        );
        source.write(app.world_mut()).set(1, 2);
        app.update();
        check_vector_phase(&mut app, "warming replacement", &expected, &[], &mut failures);
    }
    let mut final_diffs = Vec::new();
    match replacement {
        Some(included) => {
            app.world_mut()
                .resource_mut::<Readiness>()
                .0
                .insert(2, i32::from(included));
            let index = usize::from(!shift_while_warming);
            match (initially_included, included) {
                (true, true) => final_diffs.push(VecDiff::UpdateAt { index, value: 2 }),
                (true, false) => final_diffs.push(VecDiff::RemoveAt { index }),
                (false, true) => final_diffs.push(VecDiff::InsertAt { index, value: 2 }),
                (false, false) => {}
            }
            if shift_while_warming {
                final_diffs.push(VecDiff::InsertAt { index: 0, value: 4 });
            }
            expected = if included { vec![4, 2, 5] } else { vec![4, 5] };
        }
        None => {
            source.write(app.world_mut()).remove(1);
            if shift_while_warming {
                final_diffs.push(VecDiff::InsertAt { index: 0, value: 4 });
            }
            if initially_included {
                final_diffs.push(VecDiff::RemoveAt { index: 1 });
            }
            expected = vec![4, 5];
        }
    }
    for _ in 0..3 {
        app.update();
    }
    check_vector_phase(
        &mut app,
        "resolve or remove replacement",
        &expected,
        &final_diffs,
        &mut failures,
    );
    if replacement.is_none() {
        for included in [true, false, true] {
            app.world_mut()
                .resource_mut::<Readiness>()
                .0
                .insert(2, i32::from(included));
            app.update();
            check_vector_phase(
                &mut app,
                "removed predicate stale readiness",
                &expected,
                &[],
                &mut failures,
            );
        }
    }
    registration.release(app.world_mut());
    assert!(
        failures.is_empty(),
        "initial={initially_included}, replacement={replacement:?}, shift_while_warming={shift_while_warming}\n{}",
        failures.join("\n")
    );
}

#[test]
fn neighbor_filter_true_warming_remove_shifted_siblings() {
    filter_neighbor(true, None, false);
}
#[test]
fn neighbor_filter_false_warming_remove_shifted_siblings_control() {
    filter_neighbor(false, None, false);
}
#[test]
fn neighbor_filter_true_to_true_warming_shifted_siblings_control() {
    filter_neighbor(true, Some(true), false);
}
#[test]
fn neighbor_filter_true_to_false_warming_shifted_siblings_control() {
    filter_neighbor(true, Some(false), false);
}
#[test]
fn neighbor_filter_false_to_true_warming_shifted_siblings_control() {
    filter_neighbor(false, Some(true), false);
}
#[test]
fn neighbor_filter_false_to_false_warming_shifted_siblings_control() {
    filter_neighbor(false, Some(false), false);
}
#[test]
fn neighbor_filter_index_shift_during_warming_then_remove() {
    filter_neighbor(true, None, true);
}
#[test]
fn neighbor_filter_index_shift_during_warming_true_to_true() {
    filter_neighbor(true, Some(true), true);
}
#[test]
fn neighbor_filter_index_shift_during_warming_true_to_false() {
    filter_neighbor(true, Some(false), true);
}
#[test]
fn neighbor_filter_index_shift_during_warming_false_to_true() {
    filter_neighbor(false, Some(true), true);
}
#[test]
fn neighbor_filter_index_shift_during_warming_false_to_false_control() {
    filter_neighbor(false, Some(false), true);
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum JournalItem {
    Manager(usize),
    Value(usize, i32),
}
#[derive(Resource, Default)]
struct Journal(Vec<JournalItem>);

struct IdentityNeighbor {
    shape: Shape,
    length: usize,
    warming: bool,
    restart: bool,
}
impl Signal for IdentityNeighbor {
    type Item = Vec<JournalItem>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            shape,
            length,
            warming: warming_candidates,
            restart,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let current = warming(10);
            let sibling = ready(90);
            let next_sibling = ready(91);
            let candidates: Vec<_> = (0..length)
                .map(|index| {
                    let value = 20 + i32::try_from(index).expect("bounded test length");
                    if warming_candidates {
                        warming(value)
                    } else {
                        ready(value)
                    }
                })
                .collect();
            macro_rules! build {
                ($shape:ident, $initial:expr, $replace:expr, $sibling:expr, $adapter:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut update: Local<usize>| {
                            let steps = match *update {
                                0 => $initial(current.clone(), sibling.clone())
                                    .into_iter()
                                    .map(|command| DynamicManagerStep::command(command, DynamicEmissions::None))
                                    .collect(),
                                1 => {
                                    let mut steps = Vec::new();
                                    for (index, candidate) in candidates.iter().enumerate() {
                                        steps.push(DynamicManagerStep::command(
                                            $replace(candidate.clone()),
                                            DynamicEmissions::One(JournalItem::Manager(index)),
                                        ));
                                        if index == 0 {
                                            if let Some(command) = $sibling(next_sibling.clone()) {
                                                steps.push(DynamicManagerStep::command(
                                                    command,
                                                    DynamicEmissions::One(JournalItem::Manager(100)),
                                                ));
                                            }
                                        }
                                    }
                                    steps.push(DynamicManagerStep::command(
                                        $replace(current.clone()),
                                        DynamicEmissions::One(JournalItem::Manager(length)),
                                    ));
                                    steps
                                }
                                _ => {
                                    *update += 1;
                                    return None;
                                }
                            };
                            *update += 1;
                            Some(DynamicManagerUpdate::sequenced(steps))
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$shape(manager)
                            .replay_scalar::<i32>()
                            .adapter($adapter)
                            .scalar_output(Some)
                            .same_signal_identity(if restart {
                                SameSignalIdentity::Restart
                            } else {
                                SameSignalIdentity::KeepCurrent
                            }),
                    )
                }};
            }
            match shape {
                Shape::Scalar => build!(
                    scalar,
                    |current, _sibling| vec![ScalarSlotCommand::Select { signal: current }],
                    |signal| ScalarSlotCommand::Select { signal },
                    |_signal: ReplayableBroadcast<i32>| None::<ScalarSlotCommand<ReplayableBroadcast<i32>>>,
                    |(), value| DynamicEmissions::One(JournalItem::Value(0, value))
                ),
                Shape::Indexed => build!(
                    indexed,
                    |current, sibling| vec![IndexedSlotCommand::ReplaceAll {
                        signals: vec![current, sibling]
                    }],
                    |signal| IndexedSlotCommand::Replace { index: 0, signal },
                    |signal| Some(IndexedSlotCommand::Replace { index: 1, signal }),
                    |index, value| DynamicEmissions::One(JournalItem::Value(index, value))
                ),
                Shape::Keyed => build!(
                    keyed,
                    |current, sibling| vec![KeyedSlotCommand::ReplaceAll {
                        entries: vec![(0usize, current), (1, sibling)]
                    }],
                    |signal| KeyedSlotCommand::Replace { key: 0usize, signal },
                    |signal| Some(KeyedSlotCommand::Replace { key: 1usize, signal }),
                    |key, value| DynamicEmissions::One(JournalItem::Value(key, value))
                ),
            }
        })
    }
}

fn identity_neighbor(shape: Shape, warming_candidates: bool, restart: bool) {
    let mut failures = Vec::new();
    for length in [1, 2, 3, 4, 7, 16] {
        let mut app = app();
        app.init_resource::<Journal>();
        app.world_mut().resource_mut::<Readiness>().0.insert(10, 10);
        let registration = IdentityNeighbor {
            shape,
            length,
            warming: warming_candidates,
            restart,
        }
        .map(|In(items): In<Vec<JournalItem>>, mut journal: ResMut<Journal>| journal.0.extend(items))
        .register(app.world_mut());
        app.update();
        let has_sibling = !matches!(shape, Shape::Scalar);
        let mut expected = vec![JournalItem::Value(0, 10)];
        if has_sibling {
            expected.push(JournalItem::Value(1, 90));
        }
        assert_eq!(app.world().resource::<Journal>().0, expected);
        for index in 0..length {
            expected.push(JournalItem::Manager(index));
            if restart {
                expected.push(JournalItem::Value(
                    0,
                    20 + i32::try_from(index).expect("bounded test length"),
                ));
            }
            if index == 0 && has_sibling {
                expected.extend([JournalItem::Manager(100), JournalItem::Value(1, 91)]);
            }
        }
        expected.push(JournalItem::Manager(length));
        if restart {
            expected.push(JournalItem::Value(0, 10));
        }
        app.update();
        if !restart && app.world().resource::<Journal>().0 != expected {
            failures.push(format!(
                "length={length} before late readiness: {:?}; expected {expected:?}",
                app.world().resource::<Journal>().0
            ));
        }
        for index in (0..length).rev() {
            let value = 20 + i32::try_from(index).expect("bounded test length");
            app.world_mut().resource_mut::<Readiness>().0.insert(value, value);
            app.update();
        }
        for _ in 0..3 {
            app.update();
        }
        if app.world().resource::<Journal>().0 != expected {
            failures.push(format!(
                "length={length} after reverse readiness: {:?}; expected {expected:?}",
                app.world().resource::<Journal>().0
            ));
        }
        app.world_mut().resource_mut::<Readiness>().0.insert(10, 11);
        for _ in 0..3 {
            app.update();
        }
        expected.push(JournalItem::Value(0, 11));
        if app.world().resource::<Journal>().0 != expected {
            failures.push(format!(
                "length={length} current A update: {:?}; expected {expected:?}",
                app.world().resource::<Journal>().0
            ));
        }
        registration.release(app.world_mut());
    }
    assert!(
        failures.is_empty(),
        "{shape:?}, warming={warming_candidates}, restart={restart}\n{}",
        failures.join("\n")
    );
}

#[test]
fn neighbor_keep_scalar_arbitrary_ready_journal() {
    identity_neighbor(Shape::Scalar, false, false);
}
#[test]
fn neighbor_keep_indexed_arbitrary_ready_unrelated_slot_journal() {
    identity_neighbor(Shape::Indexed, false, false);
}
#[test]
fn neighbor_keep_keyed_arbitrary_ready_unrelated_slot_journal() {
    identity_neighbor(Shape::Keyed, false, false);
}
#[test]
fn neighbor_keep_scalar_arbitrary_warming_journal() {
    identity_neighbor(Shape::Scalar, true, false);
}
#[test]
fn neighbor_keep_indexed_arbitrary_warming_unrelated_slot_journal() {
    identity_neighbor(Shape::Indexed, true, false);
}
#[test]
fn neighbor_keep_keyed_arbitrary_warming_unrelated_slot_journal() {
    identity_neighbor(Shape::Keyed, true, false);
}
#[test]
fn neighbor_restart_scalar_arbitrary_ready_control() {
    identity_neighbor(Shape::Scalar, false, true);
}
#[test]
fn neighbor_restart_indexed_arbitrary_ready_control() {
    identity_neighbor(Shape::Indexed, false, true);
}
#[test]
fn neighbor_restart_keyed_arbitrary_ready_control() {
    identity_neighbor(Shape::Keyed, false, true);
}
#[test]
fn neighbor_restart_scalar_arbitrary_warming_control() {
    identity_neighbor(Shape::Scalar, true, true);
}
#[test]
fn neighbor_restart_indexed_arbitrary_warming_control() {
    identity_neighbor(Shape::Indexed, true, true);
}
#[test]
fn neighbor_restart_keyed_arbitrary_warming_control() {
    identity_neighbor(Shape::Keyed, true, true);
}

struct InsertRemoveNeighbor {
    keyed: bool,
    warming: bool,
}
impl Signal for InsertRemoveNeighbor {
    type Item = Vec<JournalItem>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            keyed,
            warming: warming_candidates,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let current = warming(10);
            let candidates: Vec<_> = [20, 30, 40]
                .into_iter()
                .map(|value| {
                    if warming_candidates {
                        warming(value)
                    } else {
                        ready(value)
                    }
                })
                .collect();
            macro_rules! build {
                ($shape:ident, $initial:expr, $insert:expr, $remove:expr, $adapter:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut update: Local<usize>| {
                            let steps = match *update {
                                0 => vec![DynamicManagerStep::command(
                                    $initial(current.clone()),
                                    DynamicEmissions::None,
                                )],
                                1 => candidates
                                    .iter()
                                    .enumerate()
                                    .flat_map(|(index, signal)| {
                                        [
                                            DynamicManagerStep::command(
                                                $insert(signal.clone()),
                                                DynamicEmissions::One(JournalItem::Manager(index * 2)),
                                            ),
                                            DynamicManagerStep::command(
                                                $remove,
                                                DynamicEmissions::One(JournalItem::Manager(index * 2 + 1)),
                                            ),
                                        ]
                                    })
                                    .collect(),
                                _ => return None,
                            };
                            *update += 1;
                            Some(DynamicManagerUpdate::sequenced(steps))
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$shape(manager)
                            .replay_scalar::<i32>()
                            .adapter($adapter)
                            .scalar_output(Some)
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }};
            }
            if keyed {
                build!(
                    keyed,
                    |signal| KeyedSlotCommand::Insert { key: 0usize, signal },
                    |signal| KeyedSlotCommand::Insert { key: 7usize, signal },
                    KeyedSlotCommand::Remove { key: 7usize },
                    |key, value| DynamicEmissions::One(JournalItem::Value(key, value))
                )
            } else {
                build!(
                    indexed,
                    |signal| IndexedSlotCommand::Insert { index: 0, signal },
                    |signal| IndexedSlotCommand::Insert { index: 1, signal },
                    IndexedSlotCommand::Remove { index: 1 },
                    |index, value| DynamicEmissions::One(JournalItem::Value(index, value))
                )
            }
        })
    }
}
fn insert_remove_neighbor(keyed: bool, warming_candidates: bool) {
    let mut app = app();
    app.init_resource::<Journal>();
    app.world_mut().resource_mut::<Readiness>().0.insert(10, 10);
    let registration = InsertRemoveNeighbor {
        keyed,
        warming: warming_candidates,
    }
    .map(|In(items): In<Vec<JournalItem>>, mut journal: ResMut<Journal>| journal.0.extend(items))
    .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Journal>().0, [JournalItem::Value(0, 10)]);
    app.update();
    let mut expected = vec![JournalItem::Value(0, 10)];
    for (index, value) in [20, 30, 40].into_iter().enumerate() {
        expected.push(JournalItem::Manager(index * 2));
        expected.push(JournalItem::Value(if keyed { 7 } else { 1 }, value));
        expected.push(JournalItem::Manager(index * 2 + 1));
    }
    if warming_candidates {
        assert_eq!(
            app.world().resource::<Journal>().0,
            [JournalItem::Value(0, 10)],
            "atomic transient journal waits for all selected replay values"
        );
        for value in [40, 30, 20] {
            app.world_mut().resource_mut::<Readiness>().0.insert(value, value);
            app.update();
        }
        for _ in 0..3 {
            app.update();
        }
    }
    assert_eq!(
        app.world().resource::<Journal>().0,
        expected,
        "same-batch Insert+Remove transients retain source-command order, independent of readiness order"
    );
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .extend([(20, 200), (30, 300), (40, 400), (10, 11)]);
    for _ in 0..3 {
        app.update();
    }
    expected.push(JournalItem::Value(0, 11));
    assert_eq!(
        app.world().resource::<Journal>().0,
        expected,
        "removed slots must not forward late readiness and the unrelated current slot remains live"
    );
    registration.release(app.world_mut());
}
#[test]
fn neighbor_indexed_insert_remove_ready_transients_control() {
    insert_remove_neighbor(false, false);
}
#[test]
fn neighbor_keyed_insert_remove_ready_transients_control() {
    insert_remove_neighbor(true, false);
}
#[test]
fn neighbor_indexed_insert_remove_warming_transients_control() {
    insert_remove_neighbor(false, true);
}
#[test]
fn neighbor_keyed_insert_remove_warming_transients_control() {
    insert_remove_neighbor(true, true);
}

#[derive(Clone, Copy, Debug)]
enum TransientFollowup {
    None,
    Emission,
    Empty,
    EmptySequence,
    Mixed,
    Replace,
    Remove,
    Clear,
}
impl TransientFollowup {
    fn supersedes(self) -> bool {
        matches!(self, Self::Replace | Self::Remove | Self::Clear)
    }
}

struct Transient {
    shape: Shape,
    followup: TransientFollowup,
    updates: usize,
    candidate: ReplayableBroadcast<i32>,
}
impl Signal for Transient {
    type Item = Vec<i32>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            shape,
            followup,
            updates,
            candidate,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let replacement = ready(99);
            macro_rules! build {
                ($policy:ident, $select:expr, $replace:expr, $remove:expr, $clear:expr, $adapter:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut update: Local<usize>| {
                            let phase = *update;
                            *update += 1;
                            if phase == 0 {
                                let mut steps = vec![DynamicManagerStep::command(
                                    $select(candidate.clone()),
                                    DynamicEmissions::One(1),
                                )];
                                if !followup.supersedes() {
                                    steps.push(DynamicManagerStep::command($remove, DynamicEmissions::One(2)));
                                }
                                return Some(DynamicManagerUpdate::sequenced(steps));
                            }
                            if phase > updates {
                                return None;
                            }
                            let followup = match followup {
                                TransientFollowup::Mixed => match (phase - 1) % 3 {
                                    0 => TransientFollowup::Emission,
                                    1 => TransientFollowup::Empty,
                                    _ => TransientFollowup::EmptySequence,
                                },
                                followup => followup,
                            };
                            match followup {
                                TransientFollowup::None => None,
                                TransientFollowup::Emission => {
                                    Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::emissions(
                                        DynamicEmissions::One(3),
                                    )]))
                                }
                                TransientFollowup::Empty => Some(DynamicManagerUpdate::new([], DynamicEmissions::None)),
                                TransientFollowup::EmptySequence => Some(DynamicManagerUpdate::sequenced([])),
                                TransientFollowup::Replace => {
                                    Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                                        $replace(replacement.clone()),
                                        DynamicEmissions::One(3),
                                    )]))
                                }
                                TransientFollowup::Remove => {
                                    Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                                        $remove,
                                        DynamicEmissions::One(3),
                                    )]))
                                }
                                TransientFollowup::Clear => {
                                    Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                                        $clear,
                                        DynamicEmissions::One(3),
                                    )]))
                                }
                                TransientFollowup::Mixed => unreachable!("mixed updates resolved above"),
                            }
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$policy(manager)
                            .replay_scalar::<i32>()
                            .adapter($adapter)
                            .scalar_output(Some)
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }};
            }
            match shape {
                Shape::Scalar => build!(
                    scalar,
                    |signal| ScalarSlotCommand::Select { signal },
                    |signal| ScalarSlotCommand::Select { signal },
                    ScalarSlotCommand::Clear,
                    ScalarSlotCommand::Clear,
                    |(), value| DynamicEmissions::One(value)
                ),
                Shape::Indexed => build!(
                    indexed,
                    |signal| IndexedSlotCommand::Insert { index: 0, signal },
                    |signal| IndexedSlotCommand::Replace { index: 0, signal },
                    IndexedSlotCommand::Remove { index: 0 },
                    IndexedSlotCommand::Clear,
                    |_: usize, value| DynamicEmissions::One(value)
                ),
                Shape::Keyed => build!(
                    keyed,
                    |signal| KeyedSlotCommand::Insert { key: 7usize, signal },
                    |signal| KeyedSlotCommand::Replace { key: 7usize, signal },
                    KeyedSlotCommand::Remove { key: 7usize },
                    KeyedSlotCommand::Clear,
                    |_: usize, value| DynamicEmissions::One(value)
                ),
            }
        })
    }
}

fn transient_case(shape: Shape, followup: TransientFollowup, updates: usize) {
    let mut app = app();
    let registration = Transient {
        shape,
        followup,
        updates,
        candidate: warming(20),
    }
    .map(scalar_capture)
    .register(app.world_mut());
    app.update();
    assert!(app.world().resource::<ScalarOutput>().0.is_empty());
    for _ in 0..updates {
        app.update();
    }
    let before_readiness = app.world().resource::<ScalarOutput>().0.clone();
    println!("{shape:?} {followup:?} updates={updates} before readiness: {before_readiness:?}");
    app.world_mut().resource_mut::<Readiness>().0.insert(20, 20);
    for _ in 0..4 {
        app.update();
    }
    let actual = app.world().resource::<ScalarOutput>().0.clone();
    registration.release(app.world_mut());
    let mut expected = vec![1, 20, 2];
    match followup {
        TransientFollowup::Emission => expected.extend(core::iter::repeat_n(3, updates)),
        TransientFollowup::Mixed => expected.extend(core::iter::repeat_n(3, updates.div_ceil(3))),
        TransientFollowup::Replace => expected = vec![1, 3, 99],
        TransientFollowup::Remove | TransientFollowup::Clear => expected = vec![1, 3],
        _ => {}
    }
    assert_eq!(
        actual, expected,
        "{shape:?} {followup:?}: only superseding commands may cancel pending intent"
    );
    if followup.supersedes() {
        assert_eq!(
            before_readiness, expected,
            "superseded readiness must not block publication"
        );
    } else {
        assert!(
            before_readiness.is_empty(),
            "required transient must block the entire journal"
        );
    }
}

#[test]
fn scalar_transient_no_intervening_update_control() {
    transient_case(Shape::Scalar, TransientFollowup::None, 1);
}
#[test]
fn indexed_transient_no_intervening_update_control() {
    transient_case(Shape::Indexed, TransientFollowup::None, 1);
}
#[test]
fn keyed_transient_no_intervening_update_control() {
    transient_case(Shape::Keyed, TransientFollowup::None, 1);
}
#[test]
fn scalar_emission_only_preserves_transient() {
    transient_case(Shape::Scalar, TransientFollowup::Emission, 1);
}
#[test]
fn indexed_emission_only_preserves_transient() {
    transient_case(Shape::Indexed, TransientFollowup::Emission, 1);
}
#[test]
fn keyed_emission_only_preserves_transient() {
    transient_case(Shape::Keyed, TransientFollowup::Emission, 1);
}
#[test]
fn scalar_empty_update_preserves_transient() {
    transient_case(Shape::Scalar, TransientFollowup::Empty, 1);
}
#[test]
fn indexed_empty_update_preserves_transient() {
    transient_case(Shape::Indexed, TransientFollowup::Empty, 1);
}
#[test]
fn keyed_empty_update_preserves_transient() {
    transient_case(Shape::Keyed, TransientFollowup::Empty, 1);
}

macro_rules! transient_neighbors {
    ($name:ident, $shape:expr, $followup:ident, $updates:expr) => {
        #[test]
        fn $name() {
            transient_case($shape, TransientFollowup::$followup, $updates);
        }
    };
}
transient_neighbors!(scalar_transient_repeated_emissions, Shape::Scalar, Emission, 6);
transient_neighbors!(indexed_transient_repeated_emissions, Shape::Indexed, Emission, 6);
transient_neighbors!(keyed_transient_repeated_emissions, Shape::Keyed, Emission, 6);
transient_neighbors!(scalar_transient_repeated_empty_updates, Shape::Scalar, Empty, 6);
transient_neighbors!(indexed_transient_repeated_empty_updates, Shape::Indexed, Empty, 6);
transient_neighbors!(keyed_transient_repeated_empty_updates, Shape::Keyed, Empty, 6);
transient_neighbors!(
    scalar_transient_repeated_empty_sequences_control,
    Shape::Scalar,
    EmptySequence,
    6
);
transient_neighbors!(
    indexed_transient_repeated_empty_sequences_control,
    Shape::Indexed,
    EmptySequence,
    6
);
transient_neighbors!(
    keyed_transient_repeated_empty_sequences_control,
    Shape::Keyed,
    EmptySequence,
    6
);
transient_neighbors!(scalar_transient_mixed_no_command_updates, Shape::Scalar, Mixed, 6);
transient_neighbors!(indexed_transient_mixed_no_command_updates, Shape::Indexed, Mixed, 6);
transient_neighbors!(keyed_transient_mixed_no_command_updates, Shape::Keyed, Mixed, 6);
transient_neighbors!(scalar_transient_superseding_replace_control, Shape::Scalar, Replace, 1);
transient_neighbors!(
    indexed_transient_superseding_replace_control,
    Shape::Indexed,
    Replace,
    1
);
transient_neighbors!(keyed_transient_superseding_replace_control, Shape::Keyed, Replace, 1);
transient_neighbors!(indexed_transient_superseding_remove_control, Shape::Indexed, Remove, 1);
transient_neighbors!(keyed_transient_superseding_remove_control, Shape::Keyed, Remove, 1);
transient_neighbors!(scalar_transient_superseding_clear_control, Shape::Scalar, Clear, 1);
transient_neighbors!(indexed_transient_superseding_clear_control, Shape::Indexed, Clear, 1);
transient_neighbors!(keyed_transient_superseding_clear_control, Shape::Keyed, Clear, 1);

struct TransientUnrelatedEdits {
    keyed: bool,
}
impl Signal for TransientUnrelatedEdits {
    type Item = Vec<JournalItem>;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(move |registrar| {
            let current = warming(10);
            let transient = warming(20);
            let sibling = ready(90);
            let replacement = ready(91);
            macro_rules! build {
                ($policy:ident, $insert:expr, $remove:expr, $move:expr, $replace:expr) => {{
                    let manager = registrar.signal_entity(
                        (),
                        move |In(()), mut update: Local<usize>| {
                            let phase = *update;
                            *update += 1;
                            let commands = match phase {
                                0 => vec![
                                    $insert(0, current.clone()),
                                    $insert(1, transient.clone()),
                                    $remove,
                                ],
                                1 => vec![$insert(1, sibling.clone())],
                                2 => vec![$move],
                                3 => vec![$replace(replacement.clone())],
                                _ => return None,
                            };
                            let first = if phase == 0 { 0 } else { phase + 2 };
                            Some(DynamicManagerUpdate::sequenced(
                                commands.into_iter().enumerate().map(|(offset, command)| {
                                    DynamicManagerStep::command(
                                        command,
                                        DynamicEmissions::One(JournalItem::Manager(first + offset)),
                                    )
                                }),
                            ))
                        },
                        OutputDelivery::move_only(),
                    )?;
                    registrar.dynamic_operator(
                        DynamicSignalOperatorSpec::$policy(manager)
                            .replay_scalar::<i32>()
                            .adapter(|slot, value| DynamicEmissions::One(JournalItem::Value(slot, value)))
                            .scalar_output(Some)
                            .same_signal_identity(SameSignalIdentity::KeepCurrent),
                    )
                }};
            }
            if self.keyed {
                build!(
                    keyed,
                    |key, signal| KeyedSlotCommand::Insert { key, signal },
                    KeyedSlotCommand::Remove { key: 1usize },
                    KeyedSlotCommand::Rekey {
                        old_key: 0usize,
                        new_key: 9usize
                    },
                    |signal| KeyedSlotCommand::Replace { key: 1usize, signal }
                )
            } else {
                build!(
                    indexed,
                    |index, signal| IndexedSlotCommand::Insert { index, signal },
                    IndexedSlotCommand::Remove { index: 1 },
                    IndexedSlotCommand::Move {
                        old_index: 0,
                        new_index: 1
                    },
                    |signal| IndexedSlotCommand::Replace { index: 0, signal }
                )
            }
        })
    }
}
fn transient_unrelated_edits(keyed: bool) {
    let mut app = app();
    app.init_resource::<Journal>();
    app.world_mut().resource_mut::<Readiness>().0.insert(10, 10);
    let registration = TransientUnrelatedEdits { keyed }
        .map(|In(items): In<Vec<JournalItem>>, mut journal: ResMut<Journal>| journal.0.extend(items))
        .register(app.world_mut());
    app.update();
    assert!(app.world().resource::<Journal>().0.is_empty());
    for _ in 0..3 {
        app.update();
    }
    let before_readiness = app.world().resource::<Journal>().0.clone();
    println!("keyed={keyed} unrelated edits before readiness: {before_readiness:?}");
    app.world_mut().resource_mut::<Readiness>().0.insert(20, 20);
    for _ in 0..4 {
        app.update();
    }
    // The inserted sibling is genuinely superseded before publication; its manager marker remains.
    let mut expected = vec![
        JournalItem::Manager(0),
        JournalItem::Value(0, 10),
        JournalItem::Manager(1),
        JournalItem::Value(1, 20),
        JournalItem::Manager(2),
        JournalItem::Manager(3),
        JournalItem::Manager(4),
        JournalItem::Manager(5),
        JournalItem::Value(if keyed { 1 } else { 0 }, 91),
    ];
    let initialized = app.world().resource::<Journal>().0.clone();
    app.world_mut()
        .resource_mut::<Readiness>()
        .0
        .extend([(10, 11), (20, 21)]);
    for _ in 0..3 {
        app.update();
    }
    let live = app.world().resource::<Journal>().0.clone();
    registration.release(app.world_mut());
    assert_eq!(
        initialized, expected,
        "unrelated edits must preserve command-time transient context"
    );
    assert!(
        before_readiness.is_empty(),
        "unrelated edits must not unblock a warming transient"
    );
    expected.push(JournalItem::Value(if keyed { 9 } else { 1 }, 11));
    assert_eq!(live, expected, "only the moved/rekeyed surviving source remains live");
}
#[test]
fn indexed_transient_unrelated_insert_move_replace() {
    transient_unrelated_edits(false);
}
#[test]
fn keyed_transient_unrelated_insert_rekey_replace() {
    transient_unrelated_edits(true);
}

fn transient_final_owner_cleanup(shape: Shape, release_phase: usize, late_readiness: bool) {
    use alloc::sync::Arc;

    let mut app = app();
    let probe = Arc::new(());
    let captured_probe = probe.clone();
    let candidate = signal::from_system(move |In(()), readiness: Res<Readiness>| {
        assert!(Arc::strong_count(&captured_probe) > 1);
        readiness.0.get(&20).copied()
    })
    .dedupe()
    .cache()
    .broadcast();
    let registration = Transient {
        shape,
        followup: TransientFollowup::Emission,
        updates: 1,
        candidate,
    }
    .map(scalar_capture)
    .register(app.world_mut());
    let terminal = registration.entity().entity();
    app.update();
    assert!(app.world().resource::<ScalarOutput>().0.is_empty());
    if release_phase >= 1 {
        app.update();
    }
    if release_phase == 2 {
        app.world_mut().resource_mut::<Readiness>().0.insert(20, 20);
        for _ in 0..4 {
            app.update();
        }
    }
    assert!(
        Arc::strong_count(&probe) > 1,
        "source capture remains owned before release"
    );
    let before_release = app.world().resource::<ScalarOutput>().0.clone();
    println!("{shape:?} cleanup release_phase={release_phase}: {before_release:?}");
    registration
        .try_release(app.world_mut())
        .expect("final owner release must accept a required pending transient");
    if late_readiness {
        app.world_mut().resource_mut::<Readiness>().0.insert(20, 21);
    }
    for _ in 0..32 {
        app.update();
    }
    assert!(
        app.world().get_entity(terminal).is_err(),
        "final owner terminal must be reclaimed"
    );
    assert_eq!(
        Arc::strong_count(&probe),
        1,
        "{shape:?} release_phase={release_phase}: source capture must be reclaimed without dropping App"
    );
    assert_eq!(
        app.world().resource::<ScalarOutput>().0,
        before_release,
        "released owner must not forward late readiness"
    );
}
macro_rules! transient_cleanup_test {
    ($name:ident, $shape:expr, $phase:expr) => {
        #[test]
        fn $name() {
            transient_final_owner_cleanup($shape, $phase, true);
        }
    };
}
transient_cleanup_test!(scalar_transient_pending_final_owner_cleanup, Shape::Scalar, 0);
transient_cleanup_test!(indexed_transient_pending_final_owner_cleanup, Shape::Indexed, 0);
transient_cleanup_test!(keyed_transient_pending_final_owner_cleanup, Shape::Keyed, 0);
transient_cleanup_test!(scalar_transient_emission_final_owner_cleanup, Shape::Scalar, 1);
transient_cleanup_test!(indexed_transient_emission_final_owner_cleanup, Shape::Indexed, 1);
transient_cleanup_test!(keyed_transient_emission_final_owner_cleanup, Shape::Keyed, 1);
transient_cleanup_test!(scalar_transient_ready_final_owner_cleanup_control, Shape::Scalar, 2);
transient_cleanup_test!(indexed_transient_ready_final_owner_cleanup_control, Shape::Indexed, 2);
transient_cleanup_test!(keyed_transient_ready_final_owner_cleanup_control, Shape::Keyed, 2);

#[test]
fn scalar_transient_never_ready_final_owner_cleanup() {
    for phase in [0, 1] {
        transient_final_owner_cleanup(Shape::Scalar, phase, false);
    }
}

#[test]
fn indexed_transient_never_ready_final_owner_cleanup() {
    for phase in [0, 1] {
        transient_final_owner_cleanup(Shape::Indexed, phase, false);
    }
}

#[test]
fn keyed_transient_never_ready_final_owner_cleanup() {
    for phase in [0, 1] {
        transient_final_owner_cleanup(Shape::Keyed, phase, false);
    }
}
