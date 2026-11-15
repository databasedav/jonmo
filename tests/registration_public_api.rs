#![allow(missing_docs)]

extern crate alloc;

use alloc::{collections::BTreeMap, sync::Arc};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use core::sync::atomic::{AtomicUsize, Ordering};
use jonmo::{
    DynamicEmissions, DynamicManagerStep, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand,
    KeyedSlotCommand, MapSignalMaterializer, OutputDelivery, ReplayInitialization, SameSignalIdentity,
    ScalarSignalMaterializer, ScalarSlotCommand, SignalRegistrationError, VecSignalMaterializer, prelude::*,
};

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct RegistrarObservations {
    category_terminal_values: Vec<i32>,
    scheduled_terminal_values: Vec<i32>,
}

struct CustomScalar;

impl Signal for CustomScalar {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(|registrar| {
            registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())
        })
    }
}

#[derive(Clone)]
enum TaggedInput {
    Number(i32),
    Flag(bool),
}

struct CustomStatefulCombination;

impl Signal for CustomStatefulCombination {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(|registrar| {
            let source = registrar.signal_entity((), |In(())| Some(2), OutputDelivery::broadcast())?;
            let unary = registrar.signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                OutputDelivery::broadcast(),
            )?;
            let stateful_unary = registrar.signal_entity(
                unary,
                |In(value): In<i32>, mut previous: Local<Option<i32>>| {
                    let value = previous.unwrap_or_default() + value;
                    *previous = Some(value);
                    Some(value)
                },
                OutputDelivery::broadcast(),
            )?;
            let second_source = registrar.signal_entity((), |In(())| Some(4), OutputDelivery::broadcast())?;
            let homogeneous_reducer = registrar.signal_entity(
                [stateful_unary, second_source],
                |In(value): In<i32>, mut state: Local<(usize, i32)>| {
                    state.0 += 1;
                    state.1 += value;
                    (state.0 == 2).then_some(state.1)
                },
                OutputDelivery::broadcast(),
            )?;

            let flag = registrar.signal_entity((), |In(())| Some(true), OutputDelivery::broadcast())?;
            let number = registrar.signal_entity(
                homogeneous_reducer,
                |In(value): In<i32>| Some(TaggedInput::Number(value)),
                OutputDelivery::broadcast(),
            )?;
            let flag = registrar.signal_entity(
                flag,
                |In(value): In<bool>| Some(TaggedInput::Flag(value)),
                OutputDelivery::broadcast(),
            )?;
            let combined = registrar.signal_entity(
                vec![number, flag].into_boxed_slice(),
                |In(input): In<TaggedInput>, mut state: Local<(Option<i32>, Option<bool>)>| {
                    match input {
                        TaggedInput::Number(value) => state.0 = Some(value),
                        TaggedInput::Flag(value) => state.1 = Some(value),
                    }
                    match *state {
                        (Some(number), Some(true)) => Some(number),
                        _ => None,
                    }
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.signal_entity(
                combined,
                |In(value): In<i32>, mut observations: ResMut<RegistrarObservations>| {
                    observations.category_terminal_values.push(value);
                    None::<i32>
                },
                OutputDelivery::broadcast(),
            )
        })
    }
}

struct CustomReplayableScalar;

impl Signal for CustomReplayableScalar {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(|registrar| {
            let source = registrar.replayable_signal_source(5, |In(())| None::<i32>, OutputDelivery::broadcast())?;
            registrar.replayable_signal_entity(
                source,
                |In(value): In<i32>| Some(value + 1),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    }
}

struct CustomVector;

impl SignalVec for CustomVector {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        VecSignalMaterializer::new(|registrar| {
            let source = registrar.replayable_signal_vec_source(
                vec![1],
                |In(())| None::<Vec<VecDiff<i32>>>,
                OutputDelivery::broadcast(),
            )?;
            registrar.replayable_signal_vec_entity(
                source,
                |In(diffs): In<Vec<VecDiff<i32>>>| Some(diffs),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    }
}

struct CustomMap;

impl SignalMap for CustomMap {
    type Key = i32;
    type Value = i32;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        MapSignalMaterializer::new(|registrar| {
            let source = registrar.replayable_signal_map_source(
                [(1, 2)].into_iter().collect(),
                |In(())| None::<Vec<MapDiff<i32, i32>>>,
                OutputDelivery::broadcast(),
            )?;
            registrar.replayable_signal_map_entity(
                source,
                |In(diffs): In<Vec<MapDiff<i32, i32>>>| Some(diffs),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    }
}

struct CustomScheduledTerminal;

impl Signal for CustomScheduledTerminal {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(|registrar| {
            let terminal = registrar.signal_entity((), |In(())| Some(11), OutputDelivery::broadcast())?;
            let original_terminal = terminal;
            let scheduled_terminal = registrar.apply_schedule(terminal, Last)?;
            registrar.signal_entity(
                [original_terminal, scheduled_terminal],
                |In(value): In<i32>, mut observations: ResMut<RegistrarObservations>| {
                    observations.scheduled_terminal_values.push(value);
                    None::<i32>
                },
                OutputDelivery::broadcast(),
            )
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, ScheduleLabel)]
struct UnregisteredSchedule;

struct CustomUnregisteredSchedule;

impl Signal for CustomUnregisteredSchedule {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(|registrar| {
            let terminal = registrar.signal_entity((), |In(())| Some(1), OutputDelivery::broadcast())?;
            registrar.apply_schedule(terminal, UnregisteredSchedule)
        })
    }
}

fn app() -> App {
    let mut app = App::new();
    app.init_resource::<RegistrarObservations>();
    app.add_plugins(JonmoPlugin::default());
    app
}

#[test]
fn external_safe_registrar_categories_execute_through_an_opaque_terminal() {
    let mut app = app();
    let _registration = CustomStatefulCombination
        .try_register(app.world_mut())
        .expect("public source, unary, reducer, adapter, and terminal categories should register");

    app.update();

    assert_eq!(
        app.world().resource::<RegistrarObservations>().category_terminal_values,
        [7]
    );
}

#[test]
fn external_replayable_scalar_vector_and_map_categories_register() {
    let mut app = app();

    let _registrations = (
        CustomReplayableScalar
            .try_register(app.world_mut())
            .expect("public replayable scalar constructors should register"),
        CustomVector
            .try_register(app.world_mut())
            .expect("public replayable vector constructors should register"),
        CustomMap
            .try_register(app.world_mut())
            .expect("public replayable map constructors should register"),
    );
}

#[derive(Clone)]
enum DirectMaterializedInput {
    Scalar(i32),
    Vector(usize),
    Map(usize),
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct DirectMaterializedOutputs(Vec<i32>);

struct ExternalDirectMaterializedComposition;

impl Signal for ExternalDirectMaterializedComposition {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let scalar = Box::new(CustomReplayableScalar).into_signal_materializer();
        let vector = Box::new(CustomVector).into_signal_vec_materializer();
        let map = Box::new(CustomMap).into_signal_map_materializer();
        ScalarSignalMaterializer::new(move |registrar| {
            let scalar = registrar.materialize_signal(scalar)?;
            let vector = registrar.materialize_signal_vec(vector)?;
            let map = registrar.materialize_signal_map(map)?;

            let vector = registrar.signal_vec_entity(
                vector,
                |In(diffs): In<Vec<VecDiff<i32>>>| Some(diffs),
                OutputDelivery::broadcast(),
            )?;
            let map = registrar.signal_map_entity(
                map,
                |In(diffs): In<Vec<MapDiff<i32, i32>>>| Some(diffs),
                OutputDelivery::broadcast(),
            )?;
            let scalar = registrar.signal_entity(
                scalar,
                |In(value): In<i32>| Some(DirectMaterializedInput::Scalar(value)),
                OutputDelivery::broadcast(),
            )?;
            let vector = registrar.signal_entity(
                vector,
                |In(diffs): In<Vec<VecDiff<i32>>>| {
                    let length = diffs
                        .into_iter()
                        .find_map(|diff| match diff {
                            VecDiff::Replace { values } => Some(values.len()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    Some(DirectMaterializedInput::Vector(length))
                },
                OutputDelivery::broadcast(),
            )?;
            let map = registrar.signal_entity(
                map,
                |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                    let length = diffs
                        .into_iter()
                        .find_map(|diff| match diff {
                            MapDiff::Replace { entries } => Some(entries.len()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    Some(DirectMaterializedInput::Map(length))
                },
                OutputDelivery::broadcast(),
            )?;

            registrar.signal_entity(
                [scalar, vector, map],
                |In(input): In<DirectMaterializedInput>,
                 mut state: Local<(Option<i32>, Option<usize>, Option<usize>)>| {
                    match input {
                        DirectMaterializedInput::Scalar(value) => state.0 = Some(value),
                        DirectMaterializedInput::Vector(length) => state.1 = Some(length),
                        DirectMaterializedInput::Map(length) => state.2 = Some(length),
                    }
                    match *state {
                        (Some(scalar), Some(vector), Some(map)) => Some(scalar + vector as i32 + map as i32),
                        _ => None,
                    }
                },
                OutputDelivery::broadcast(),
            )
        })
    }
}

#[test]
fn external_composition_directly_materializes_and_adapts_scalar_vector_and_map_inputs() {
    let mut app = app();
    app.init_resource::<DirectMaterializedOutputs>();
    let registration = ExternalDirectMaterializedComposition
        .map(|In(value): In<i32>, mut outputs: ResMut<DirectMaterializedOutputs>| outputs.0.push(value))
        .register(app.world_mut());

    app.update();
    app.update();

    assert_eq!(app.world().resource::<DirectMaterializedOutputs>().0, [8]);
    registration.release(app.world_mut());
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AuthoritativeSnapshot {
    ScalarNone,
    VectorEmptyReplace,
    MapEmptyReplace,
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct AuthoritativeSnapshots(Vec<AuthoritativeSnapshot>);

struct ExternalAuthoritativeEmptySnapshots;

impl Signal for ExternalAuthoritativeEmptySnapshots {
    type Item = ();

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(|registrar| {
            let scalar = registrar.replayable_signal_source(
                None::<i32>,
                |In(())| None::<Option<i32>>,
                OutputDelivery::broadcast(),
            )?;
            let vector = registrar.replayable_signal_vec_source(
                Vec::<i32>::new(),
                |In(())| None::<Vec<VecDiff<i32>>>,
                OutputDelivery::broadcast(),
            )?;
            let map = registrar.replayable_signal_map_source(
                BTreeMap::<i32, i32>::new(),
                |In(())| None::<Vec<MapDiff<i32, i32>>>,
                OutputDelivery::broadcast(),
            )?;
            let scalar = registrar.signal_entity(
                scalar,
                |In(value): In<Option<i32>>, mut snapshots: ResMut<AuthoritativeSnapshots>| {
                    assert_eq!(value, None);
                    snapshots.0.push(AuthoritativeSnapshot::ScalarNone);
                    Some(())
                },
                OutputDelivery::broadcast(),
            )?;
            let vector = registrar.signal_entity(
                vector,
                |In(diffs): In<Vec<VecDiff<i32>>>, mut snapshots: ResMut<AuthoritativeSnapshots>| {
                    assert!(matches!(diffs.as_slice(), [VecDiff::Replace { values }] if values.is_empty()));
                    snapshots.0.push(AuthoritativeSnapshot::VectorEmptyReplace);
                    Some(())
                },
                OutputDelivery::broadcast(),
            )?;
            let map = registrar.signal_entity(
                map,
                |In(diffs): In<Vec<MapDiff<i32, i32>>>, mut snapshots: ResMut<AuthoritativeSnapshots>| {
                    assert!(matches!(diffs.as_slice(), [MapDiff::Replace { entries }] if entries.is_empty()));
                    snapshots.0.push(AuthoritativeSnapshot::MapEmptyReplace);
                    Some(())
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.signal_entity([scalar, vector, map], |In(())| Some(()), OutputDelivery::broadcast())
        })
    }
}

#[test]
fn external_managed_replay_delivers_authoritative_none_and_empty_collection_snapshots() {
    let mut app = app();
    app.init_resource::<AuthoritativeSnapshots>();
    let registration = ExternalAuthoritativeEmptySnapshots.register(app.world_mut());

    app.update();

    let snapshots = &app.world().resource::<AuthoritativeSnapshots>().0;
    assert_eq!(snapshots.len(), 3);
    assert!(snapshots.contains(&AuthoritativeSnapshot::ScalarNone));
    assert!(snapshots.contains(&AuthoritativeSnapshot::VectorEmptyReplace));
    assert!(snapshots.contains(&AuthoritativeSnapshot::MapEmptyReplace));
    registration.release(app.world_mut());
}

#[test]
fn registrar_schedule_application_preserves_same_terminal_behavior() {
    let mut app = app();
    let _registration = CustomScheduledTerminal
        .try_register(app.world_mut())
        .expect("the registrar should accept its registered schedule");

    app.update();

    assert_eq!(
        app.world()
            .resource::<RegistrarObservations>()
            .scheduled_terminal_values,
        [11]
    );
}

#[test]
fn registrar_rejects_an_unregistered_schedule_through_the_public_error() {
    let mut app = app();

    assert!(matches!(
        CustomUnregisteredSchedule.try_register(app.world_mut()),
        Err(SignalRegistrationError::ScheduleNotRegistered)
    ));
}

#[test]
fn builtin_source_and_map_share_materializer_registration_authority() {
    let mut app = app();
    let source = jonmo::signal::from_system(|In(())| -> i32 { 1 }).broadcast();

    let registered = source.clone().register(app.world_mut());
    let tried = source
        .clone()
        .try_register(app.world_mut())
        .expect("built-in source try_register should use its materializer");

    let mapped = source.clone().map(|In(value): In<i32>| value + 1);
    let mapped = mapped
        .try_register(app.world_mut())
        .expect("built-in map try_register should use the same registrar path");

    let cached = source
        .cache()
        .try_register(app.world_mut())
        .expect("built-in source cache should materialize through the registrar");
    let _registrations = (registered, tried, mapped, cached);
}

#[test]
fn external_cache_wrapper_registers_and_normalizes() {
    let mut app = app();
    let cached = CustomScalar
        .cache()
        .cache()
        .try_register(app.world_mut())
        .expect("repeated cache should remain a valid verified cache wrapper");
    let scalar = CustomReplayableScalar
        .cache()
        .try_register(app.world_mut())
        .expect("external managed replayable scalar cache should normalize to its terminal");
    let vector = CustomVector
        .cache()
        .try_register(app.world_mut())
        .expect("external managed replayable vector cache should normalize to its terminal");
    let map = CustomMap
        .cache()
        .try_register(app.world_mut())
        .expect("external managed replayable map cache should normalize to its terminal");
    let _registrations = (cached, scalar, vector, map);
}

#[derive(Resource, Default)]
struct DynamicObservations {
    scalar: Vec<i32>,
    vector: Vec<Vec<VecDiff<i32>>>,
}

struct ExternalScalarDynamic;

impl Signal for ExternalScalarDynamic {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let selected = jonmo::signal::always(7).dedupe().broadcast();
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    Some(DynamicManagerUpdate::new(
                        [ScalarSlotCommand::Select {
                            signal: selected.clone(),
                        }],
                        DynamicEmissions::<i32>::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            let output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )?;
            registrar.apply_schedule(output, Last)
        })
    }
}

struct ExternalIndexedDynamic;

impl SignalVec for ExternalIndexedDynamic {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        let first = jonmo::signal::always(3).dedupe().broadcast();
        let second = jonmo::signal::always(5).dedupe().broadcast();
        VecSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let result = match *update {
                        0 => Some(DynamicManagerUpdate::new(
                            [IndexedSlotCommand::ReplaceAll {
                                signals: vec![first.clone(), second.clone()],
                            }],
                            DynamicEmissions::<(usize, i32)>::None,
                        )),
                        1 => Some(DynamicManagerUpdate::new(
                            [IndexedSlotCommand::Clear],
                            DynamicEmissions::<(usize, i32)>::None,
                        )),
                        _ => None,
                    };
                    *update += 1;
                    result
                },
                OutputDelivery::move_only(),
            )?;
            let mut current = Vec::new();
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .signal_vec_output(move |values| {
                        if values.is_empty() {
                            current.clear();
                        }
                        for (index, value) in values {
                            if current.len() <= index {
                                current.resize(index + 1, 0);
                            }
                            current[index] = value;
                        }
                        Some(vec![VecDiff::Replace {
                            values: current.clone(),
                        }])
                    })
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

struct ExternalKeyedDynamic;

impl SignalMap for ExternalKeyedDynamic {
    type Key = i32;
    type Value = i32;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        let first = jonmo::signal::always(11).dedupe().broadcast();
        let second = jonmo::signal::always(13).dedupe().broadcast();
        MapSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut emitted: Local<bool>| {
                    if core::mem::replace(&mut *emitted, true) {
                        None
                    } else {
                        Some(DynamicManagerUpdate::new(
                            [KeyedSlotCommand::ReplaceAll {
                                entries: vec![(1, first.clone()), (2, second.clone())],
                            }],
                            DynamicEmissions::<(i32, i32)>::None,
                        ))
                    }
                },
                OutputDelivery::move_only(),
            )?;
            let mut current = BTreeMap::new();
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One((key, value)))
                    .signal_map_output(move |values| {
                        for (key, value) in values {
                            current.insert(key, value);
                        }
                        Some(vec![MapDiff::Replace {
                            entries: current.iter().map(|(key, value)| (*key, *value)).collect(),
                        }])
                    })
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

struct ExternalRestartDynamic;

impl Signal for ExternalRestartDynamic {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let selected = jonmo::signal::always(17).dedupe().broadcast();
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    Some(DynamicManagerUpdate::new(
                        [ScalarSlotCommand::Select {
                            signal: selected.clone(),
                        }],
                        DynamicEmissions::<i32>::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        })
    }
}

struct ExternalHandledDynamic {
    errors: Arc<AtomicUsize>,
}

impl SignalVec for ExternalHandledDynamic {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        let errors = self.errors;
        let selected = jonmo::signal::always(19).dedupe().broadcast();
        VecSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut initialized: Local<bool>| {
                    let command = if core::mem::replace(&mut *initialized, true) {
                        IndexedSlotCommand::Remove { index: 9 }
                    } else {
                        IndexedSlotCommand::ReplaceAll {
                            signals: vec![selected.clone()],
                        }
                    };
                    Some(DynamicManagerUpdate::new(
                        [command],
                        DynamicEmissions::<(usize, i32)>::None,
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One((index, value)))
                    .signal_vec_output(|values| {
                        Some(vec![VecDiff::Replace {
                            values: values.into_iter().map(|(_, value)| value).collect(),
                        }])
                    })
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        assert_eq!(error, SignalRegistrationError::InvalidDynamicCommand);
                        errors.fetch_add(1, Ordering::SeqCst);
                    }),
            )
        })
    }
}

#[test]
fn external_dynamic_scalar_indexed_keyed_restart_and_handler_use_the_public_surface() {
    let mut app = app();
    app.init_resource::<DynamicObservations>();

    let scalar = ExternalScalarDynamic
        .map(|In(value): In<i32>, mut observations: ResMut<DynamicObservations>| {
            observations.scalar.push(value);
        })
        .register(app.world_mut());
    let vector = ExternalIndexedDynamic
        .for_each(
            |In(diffs): In<Vec<VecDiff<i32>>>, mut observations: ResMut<DynamicObservations>| {
                observations.vector.push(diffs);
            },
        )
        .register(app.world_mut());
    let map = ExternalKeyedDynamic
        .for_each(|In(diffs): In<Vec<MapDiff<i32, i32>>>| {
            let _ = diffs;
        })
        .register(app.world_mut());
    let restart = ExternalRestartDynamic
        .map(|In(value): In<i32>, mut observations: ResMut<DynamicObservations>| {
            observations.scalar.push(value);
        })
        .register(app.world_mut());
    let errors = Arc::new(AtomicUsize::new(0));
    let handled = ExternalHandledDynamic { errors: errors.clone() }
        .for_each(|In(_): In<Vec<VecDiff<i32>>>| {})
        .register(app.world_mut());

    app.update();
    app.update();

    let observations = app.world().resource::<DynamicObservations>();
    assert!(observations.scalar.contains(&7));
    assert!(observations.scalar.iter().filter(|value| **value == 17).count() >= 2);
    assert!(!observations.vector.is_empty());
    assert!(
        observations
            .vector
            .iter()
            .any(|diffs| { matches!(diffs.as_slice(), [VecDiff::Replace { values }] if values.is_empty()) })
    );
    assert!(errors.load(Ordering::SeqCst) >= 1);

    scalar.release(app.world_mut());
    vector.release(app.world_mut());
    map.release(app.world_mut());
    restart.release(app.world_mut());
    handled.release(app.world_mut());
}

#[derive(Resource, Default)]
struct ExternalScalarReadiness(BTreeMap<&'static str, i32>);

#[derive(Clone)]
enum ExternalManagedScalar {
    Ready(i32),
    Warming(&'static str),
}

impl Signal for ExternalManagedScalar {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        ScalarSignalMaterializer::new(move |registrar| match *self {
            ExternalManagedScalar::Ready(value) => {
                registrar.replayable_signal_source(value, |In(())| None::<i32>, OutputDelivery::broadcast())
            }
            ExternalManagedScalar::Warming(name) => registrar.replayable_signal_entity(
                (),
                move |In(()), readiness: Res<ExternalScalarReadiness>, mut emitted: Local<bool>| {
                    if *emitted {
                        return None;
                    }
                    let value = readiness.0.get(name).copied();
                    if value.is_some() {
                        *emitted = true;
                    }
                    value
                },
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            ),
        })
    }
}

fn ready_external_scalar(value: i32) -> ReplayableBroadcast<i32> {
    ExternalManagedScalar::Ready(value).cache().broadcast()
}

fn warming_external_scalar(name: &'static str) -> ReplayableBroadcast<i32> {
    ExternalManagedScalar::Warming(name).cache().broadcast()
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct ScalarDynamicOutputs(Vec<i32>);

struct ExternalKeepCurrentCancellation {
    current: ReplayableBroadcast<i32>,
    warming: ReplayableBroadcast<i32>,
}

impl Signal for ExternalKeepCurrentCancellation {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { current, warming } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let command = match *update {
                        0 => Some(ScalarSlotCommand::Select {
                            signal: current.clone(),
                        }),
                        1 => Some(ScalarSlotCommand::Select {
                            signal: warming.clone(),
                        }),
                        2 => Some(ScalarSlotCommand::Select {
                            signal: current.clone(),
                        }),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    command.map(|command| DynamicManagerUpdate::new([command], DynamicEmissions::<i32>::None))
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

#[test]
fn scalar_keep_current_reselection_cancels_a_warming_candidate_and_ignores_its_late_replay() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<ScalarDynamicOutputs>();
    let current = ready_external_scalar(41);
    let registration = ExternalKeepCurrentCancellation {
        current: current.clone(),
        warming: warming_external_scalar("scalar-b"),
    }
    .map(|In(value): In<i32>, mut outputs: ResMut<ScalarDynamicOutputs>| {
        outputs.0.push(value);
    })
    .register(app.world_mut());

    app.update();
    app.update();
    app.update();
    assert_eq!(app.world().resource::<ScalarDynamicOutputs>().0, [41]);

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("scalar-b", 99);
    app.update();
    app.update();

    assert_eq!(app.world().resource::<ScalarDynamicOutputs>().0, [41]);
    registration.release(app.world_mut());
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PublicDynamicQueueItem {
    Manager(&'static str),
    Scalar(i32),
    Indexed { index: usize, value: i32 },
    Keyed { key: i32, value: i32 },
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct PublicDynamicQueueOutputs(Vec<Vec<PublicDynamicQueueItem>>);

#[derive(Resource, Default)]
struct ScalarSequenceControl(bool);

struct ExternalScalarSequenced {
    first: ReplayableBroadcast<i32>,
    second: ReplayableBroadcast<i32>,
}

impl Signal for ExternalScalarSequenced {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { first, second } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut control: ResMut<ScalarSequenceControl>| {
                    if !core::mem::take(&mut control.0) {
                        return None;
                    }
                    Some(DynamicManagerUpdate::sequenced([
                        DynamicManagerStep::command(
                            ScalarSlotCommand::Select { signal: first.clone() },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("scalar:first")),
                        ),
                        DynamicManagerStep::emissions(DynamicEmissions::One(PublicDynamicQueueItem::Manager(
                            "scalar:between",
                        ))),
                        DynamicManagerStep::command(
                            ScalarSlotCommand::Select { signal: second.clone() },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("scalar:second")),
                        ),
                    ]))
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(PublicDynamicQueueItem::Scalar(value)))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

fn register_public_queue_output(app: &mut App, signal: impl Signal<Item = Vec<PublicDynamicQueueItem>>) -> SignalClaim {
    signal
        .map(
            |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
                outputs.0.push(items);
            },
        )
        .register(app.world_mut())
}

#[test]
fn scalar_sequenced_steps_interleave_manager_emissions_and_multiple_ready_candidates() {
    let mut app = app();
    app.init_resource::<PublicDynamicQueueOutputs>();
    app.insert_resource(ScalarSequenceControl(true));
    let registration = register_public_queue_output(
        &mut app,
        ExternalScalarSequenced {
            first: ready_external_scalar(101),
            second: ready_external_scalar(102),
        },
    );

    app.update();
    app.update();

    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("scalar:first"),
            PublicDynamicQueueItem::Scalar(101),
            PublicDynamicQueueItem::Manager("scalar:between"),
            PublicDynamicQueueItem::Manager("scalar:second"),
            PublicDynamicQueueItem::Scalar(102),
        ]
    );
    registration.release(app.world_mut());
}

#[test]
fn scalar_sequenced_steps_wait_for_all_candidates_and_preserve_order_under_reverse_readiness() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    app.insert_resource(ScalarSequenceControl(true));
    let registration = register_public_queue_output(
        &mut app,
        ExternalScalarSequenced {
            first: warming_external_scalar("scalar-sequenced-first"),
            second: ready_external_scalar(112),
        },
    );

    app.update();
    app.update();
    assert!(app.world().resource::<PublicDynamicQueueOutputs>().0.is_empty());

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("scalar-sequenced-first", 111);
    app.update();
    app.update();

    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("scalar:first"),
            PublicDynamicQueueItem::Scalar(111),
            PublicDynamicQueueItem::Manager("scalar:between"),
            PublicDynamicQueueItem::Manager("scalar:second"),
            PublicDynamicQueueItem::Scalar(112),
        ]
    );
    registration.release(app.world_mut());
}

#[derive(Resource, Default)]
struct ScalarSupersessionControl(Option<usize>);

struct ExternalScalarSupersessionJournal {
    current: ReplayableBroadcast<i32>,
    first_pending: ReplayableBroadcast<i32>,
    replacement: ReplayableBroadcast<i32>,
    cleared_pending: ReplayableBroadcast<i32>,
}

impl Signal for ExternalScalarSupersessionJournal {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            current,
            first_pending,
            replacement,
            cleared_pending,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut control: ResMut<ScalarSupersessionControl>| {
                    let step = match control.0.take()? {
                        0 => DynamicManagerStep::command(
                            ScalarSlotCommand::Select {
                                signal: current.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:current")),
                        ),
                        1 => DynamicManagerStep::command(
                            ScalarSlotCommand::Select {
                                signal: first_pending.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:b-for-a")),
                        ),
                        2 => DynamicManagerStep::command(
                            ScalarSlotCommand::Select {
                                signal: current.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:a")),
                        ),
                        3 => DynamicManagerStep::command(
                            ScalarSlotCommand::Select {
                                signal: first_pending.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:b-for-c")),
                        ),
                        4 => DynamicManagerStep::command(
                            ScalarSlotCommand::Select {
                                signal: replacement.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:c")),
                        ),
                        5 => DynamicManagerStep::command(
                            ScalarSlotCommand::Select {
                                signal: cleared_pending.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:b-for-clear")),
                        ),
                        6 => DynamicManagerStep::command(
                            ScalarSlotCommand::Clear,
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("supersede:clear")),
                        ),
                        _ => return None,
                    };
                    Some(DynamicManagerUpdate::sequenced([step]))
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(PublicDynamicQueueItem::Scalar(value)))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

#[test]
fn scalar_supersession_preserves_manager_journal_and_discards_stale_candidate_semantics() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    app.init_resource::<ScalarSupersessionControl>();
    let current = ready_external_scalar(121);
    let registration = register_public_queue_output(
        &mut app,
        ExternalScalarSupersessionJournal {
            current: current.clone(),
            first_pending: warming_external_scalar("scalar-superseded-b"),
            replacement: warming_external_scalar("scalar-replacement-c"),
            cleared_pending: warming_external_scalar("scalar-cleared-b"),
        },
    );

    for update in 0..=2 {
        app.world_mut().resource_mut::<ScalarSupersessionControl>().0 = Some(update);
        app.update();
    }
    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("supersede:current"),
            PublicDynamicQueueItem::Scalar(121),
            PublicDynamicQueueItem::Manager("supersede:b-for-a"),
            PublicDynamicQueueItem::Manager("supersede:a"),
        ]
    );

    for update in 3..=4 {
        app.world_mut().resource_mut::<ScalarSupersessionControl>().0 = Some(update);
        app.update();
    }
    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("scalar-superseded-b", 122);
    app.update();
    assert_eq!(flattened_public_dynamic_queue(&app).len(), 4);

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("scalar-replacement-c", 123);
    app.update();
    app.update();
    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("supersede:current"),
            PublicDynamicQueueItem::Scalar(121),
            PublicDynamicQueueItem::Manager("supersede:b-for-a"),
            PublicDynamicQueueItem::Manager("supersede:a"),
            PublicDynamicQueueItem::Manager("supersede:b-for-c"),
            PublicDynamicQueueItem::Manager("supersede:c"),
            PublicDynamicQueueItem::Scalar(123),
        ]
    );

    for update in 5..=6 {
        app.world_mut().resource_mut::<ScalarSupersessionControl>().0 = Some(update);
        app.update();
    }
    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("supersede:current"),
            PublicDynamicQueueItem::Scalar(121),
            PublicDynamicQueueItem::Manager("supersede:b-for-a"),
            PublicDynamicQueueItem::Manager("supersede:a"),
            PublicDynamicQueueItem::Manager("supersede:b-for-c"),
            PublicDynamicQueueItem::Manager("supersede:c"),
            PublicDynamicQueueItem::Scalar(123),
            PublicDynamicQueueItem::Manager("supersede:b-for-clear"),
            PublicDynamicQueueItem::Manager("supersede:clear"),
        ]
    );

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("scalar-cleared-b", 124);
    app.update();
    app.update();
    assert_eq!(flattened_public_dynamic_queue(&app).len(), 9);
    registration.release(app.world_mut());
}

#[derive(Resource, Default)]
struct IndexedBatchControl(Option<usize>);

struct ExternalIndexedSameBatchSequences;

impl Signal for ExternalIndexedSameBatchSequences {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let insert_remove = ready_external_scalar(10);
        let insert_replace = ready_external_scalar(20);
        let replacement = ready_external_scalar(21);
        let insert_clear = ready_external_scalar(30);
        let first_same_index = ready_external_scalar(40);
        let second_same_index = ready_external_scalar(41);
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut control: ResMut<IndexedBatchControl>| {
                    let update = control.0.take()?;
                    match update {
                        0 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Insert {
                                    index: 0,
                                    signal: insert_remove.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("insert-remove:insert")),
                            ),
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Remove { index: 0 },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("insert-remove:remove")),
                            ),
                        ])),
                        1 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Insert {
                                    index: 0,
                                    signal: insert_replace.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("insert-replace:insert")),
                            ),
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Replace {
                                    index: 0,
                                    signal: replacement.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("insert-replace:replace")),
                            ),
                        ])),
                        2 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Insert {
                                    index: 1,
                                    signal: insert_clear.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("insert-clear:insert")),
                            ),
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Clear,
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("insert-clear:clear")),
                            ),
                        ])),
                        3 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Insert {
                                    index: 0,
                                    signal: first_same_index.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("same-index:first")),
                            ),
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Insert {
                                    index: 0,
                                    signal: second_same_index.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("same-index:second")),
                            ),
                        ])),
                        _ => None,
                    }
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One(PublicDynamicQueueItem::Indexed { index, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

fn flattened_public_dynamic_queue(app: &App) -> Vec<PublicDynamicQueueItem> {
    app.world()
        .resource::<PublicDynamicQueueOutputs>()
        .0
        .iter()
        .flatten()
        .cloned()
        .collect()
}

#[test]
fn indexed_same_batch_commands_preserve_exact_semantic_queue_order() {
    let mut app = app();
    app.init_resource::<IndexedBatchControl>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let registration = ExternalIndexedSameBatchSequences
        .map(
            |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
                outputs.0.push(items);
            },
        )
        .register(app.world_mut());

    for update in 0..4 {
        app.world_mut().resource_mut::<IndexedBatchControl>().0 = Some(update);
        app.update();
        app.update();
    }

    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("insert-remove:insert"),
            PublicDynamicQueueItem::Indexed { index: 0, value: 10 },
            PublicDynamicQueueItem::Manager("insert-remove:remove"),
            PublicDynamicQueueItem::Manager("insert-replace:insert"),
            PublicDynamicQueueItem::Indexed { index: 0, value: 20 },
            PublicDynamicQueueItem::Manager("insert-replace:replace"),
            PublicDynamicQueueItem::Indexed { index: 0, value: 21 },
            PublicDynamicQueueItem::Manager("insert-clear:insert"),
            PublicDynamicQueueItem::Indexed { index: 1, value: 30 },
            PublicDynamicQueueItem::Manager("insert-clear:clear"),
            PublicDynamicQueueItem::Manager("same-index:first"),
            PublicDynamicQueueItem::Indexed { index: 0, value: 40 },
            PublicDynamicQueueItem::Manager("same-index:second"),
            PublicDynamicQueueItem::Indexed { index: 0, value: 41 },
        ]
    );
    registration.release(app.world_mut());
}

struct ExternalPendingIndexedInsertions {
    warming: ReplayableBroadcast<i32>,
    ready: ReplayableBroadcast<i32>,
    errors: Arc<AtomicUsize>,
}

impl Signal for ExternalPendingIndexedInsertions {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { warming, ready, errors } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let result = match *update {
                        0 => Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                            IndexedSlotCommand::Insert {
                                index: 0,
                                signal: warming.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending:insert-a")),
                        )])),
                        1 => Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                            IndexedSlotCommand::Insert {
                                index: 1,
                                signal: ready.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending:insert-b")),
                        )])),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    result
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One(PublicDynamicQueueItem::Indexed { index, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        assert_eq!(error, SignalRegistrationError::InvalidDynamicCommand);
                        errors.fetch_add(1, Ordering::SeqCst);
                    }),
            )
        })
    }
}

#[test]
fn indexed_pending_insert_accepts_the_latest_desired_layout_in_source_order() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let errors = Arc::new(AtomicUsize::new(0));
    let registration = ExternalPendingIndexedInsertions {
        warming: warming_external_scalar("indexed-a"),
        ready: ready_external_scalar(52),
        errors: errors.clone(),
    }
    .map(
        |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
            outputs.0.push(items);
        },
    )
    .register(app.world_mut());

    app.update();
    app.update();
    assert!(app.world().resource::<PublicDynamicQueueOutputs>().0.is_empty());
    assert_eq!(errors.load(Ordering::SeqCst), 0);

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("indexed-a", 51);
    app.update();
    app.update();

    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("pending:insert-a"),
            PublicDynamicQueueItem::Indexed { index: 0, value: 51 },
            PublicDynamicQueueItem::Manager("pending:insert-b"),
            PublicDynamicQueueItem::Indexed { index: 1, value: 52 },
        ]
    );
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    registration.release(app.world_mut());
}

struct ExternalKeyedSameBatchSequence;

impl Signal for ExternalKeyedSameBatchSequence {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let initial = ready_external_scalar(60);
        let inserted = ready_external_scalar(61);
        let replaced = ready_external_scalar(62);
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let result = match *update {
                        0 => Some(DynamicManagerUpdate::new(
                            [KeyedSlotCommand::ReplaceAll {
                                entries: vec![(9, initial.clone())],
                            }],
                            DynamicEmissions::<PublicDynamicQueueItem>::None,
                        )),
                        1 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Insert {
                                    key: 1,
                                    signal: inserted.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("keyed:insert")),
                            ),
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Rekey { old_key: 1, new_key: 2 },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("keyed:rekey")),
                            ),
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Replace {
                                    key: 2,
                                    signal: replaced.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("keyed:replace")),
                            ),
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Remove { key: 9 },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("keyed:remove")),
                            ),
                        ])),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    result
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One(PublicDynamicQueueItem::Keyed { key, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

#[test]
fn keyed_same_batch_insert_rekey_replace_remove_preserves_exact_order_without_panicking() {
    let mut app = app();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let registration = ExternalKeyedSameBatchSequence
        .map(
            |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
                outputs.0.push(items);
            },
        )
        .register(app.world_mut());

    app.update();
    app.world_mut().resource_mut::<PublicDynamicQueueOutputs>().0.clear();
    app.update();
    app.update();

    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("keyed:insert"),
            PublicDynamicQueueItem::Keyed { key: 1, value: 61 },
            PublicDynamicQueueItem::Manager("keyed:rekey"),
            PublicDynamicQueueItem::Manager("keyed:replace"),
            PublicDynamicQueueItem::Keyed { key: 2, value: 62 },
            PublicDynamicQueueItem::Manager("keyed:remove"),
        ]
    );
    registration.release(app.world_mut());
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExternalReplaySelection {
    Scalar(i32),
    Vector(Vec<i32>),
    Map(Vec<(i32, i32)>),
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct ExternalReplaySelections(Vec<ExternalReplaySelection>);

struct ExternalManagedReplayDynamic {
    scalar: BoxedReplayableSignal<i32>,
    vector: BoxedReplayableSignalVec<i32>,
    map: BoxedReplayableSignalMap<i32, i32>,
}

impl Signal for ExternalManagedReplayDynamic {
    type Item = ();

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { scalar, vector, map } = *self;
        let mut scalar = Some(scalar);
        let mut vector = Some(vector);
        let mut map = Some(map);
        ScalarSignalMaterializer::new(move |registrar| {
            let scalar_manager = registrar.signal_entity(
                (),
                move |In(())| {
                    scalar.take().map(|signal| {
                        DynamicManagerUpdate::new(
                            [ScalarSlotCommand::Select { signal }],
                            DynamicEmissions::<ExternalReplaySelection>::None,
                        )
                    })
                },
                OutputDelivery::move_only(),
            )?;
            let scalar_output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(scalar_manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(ExternalReplaySelection::Scalar(value)))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )?;

            let vector_manager = registrar.signal_entity(
                (),
                move |In(())| {
                    vector.take().map(|signal| {
                        DynamicManagerUpdate::new(
                            [ScalarSlotCommand::Select { signal }],
                            DynamicEmissions::<ExternalReplaySelection>::None,
                        )
                    })
                },
                OutputDelivery::move_only(),
            )?;
            let mut vector_values = Vec::new();
            let vector_output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(vector_manager)
                    .replay_signal_vec::<i32>()
                    .adapter(move |(), diffs| {
                        for diff in diffs {
                            match diff {
                                VecDiff::Replace { values } => vector_values = values,
                                VecDiff::InsertAt { index, value } => {
                                    if index <= vector_values.len() {
                                        vector_values.insert(index, value);
                                    }
                                }
                                VecDiff::UpdateAt { index, value } => {
                                    if let Some(current) = vector_values.get_mut(index) {
                                        *current = value;
                                    }
                                }
                                VecDiff::RemoveAt { index } => {
                                    if index < vector_values.len() {
                                        vector_values.remove(index);
                                    }
                                }
                                VecDiff::Move { old_index, new_index } => {
                                    if old_index < vector_values.len() && new_index < vector_values.len() {
                                        let value = vector_values.remove(old_index);
                                        vector_values.insert(new_index, value);
                                    }
                                }
                                VecDiff::Push { value } => vector_values.push(value),
                                VecDiff::Pop => vector_values.truncate(vector_values.len().saturating_sub(1)),
                                VecDiff::Clear => vector_values.clear(),
                            }
                        }
                        DynamicEmissions::One(ExternalReplaySelection::Vector(vector_values.clone()))
                    })
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )?;

            let map_manager = registrar.signal_entity(
                (),
                move |In(())| {
                    map.take().map(|signal| {
                        DynamicManagerUpdate::new(
                            [ScalarSlotCommand::Select { signal }],
                            DynamicEmissions::<ExternalReplaySelection>::None,
                        )
                    })
                },
                OutputDelivery::move_only(),
            )?;
            let mut map_entries = BTreeMap::new();
            let map_output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(map_manager)
                    .replay_signal_map::<i32, i32>()
                    .adapter(move |(), diffs| {
                        for diff in diffs {
                            match diff {
                                MapDiff::Replace { entries } => map_entries = entries.into_iter().collect(),
                                MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                                    map_entries.extend([(key, value)]);
                                }
                                MapDiff::Remove { key } => {
                                    map_entries.retain(|current, _| *current != key);
                                }
                                MapDiff::Clear => map_entries.clear(),
                            }
                        }
                        DynamicEmissions::One(ExternalReplaySelection::Map(
                            map_entries.iter().map(|(key, value)| (*key, *value)).collect(),
                        ))
                    })
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )?;

            let scalar_observer = registrar.signal_entity(
                scalar_output,
                |In(selection): In<ExternalReplaySelection>, mut selections: ResMut<ExternalReplaySelections>| {
                    selections.0.push(selection);
                    Some(())
                },
                OutputDelivery::broadcast(),
            )?;
            let vector_observer = registrar.signal_entity(
                vector_output,
                |In(selection): In<ExternalReplaySelection>, mut selections: ResMut<ExternalReplaySelections>| {
                    selections.0.push(selection);
                    Some(())
                },
                OutputDelivery::broadcast(),
            )?;
            let map_observer = registrar.signal_entity(
                map_output,
                |In(selection): In<ExternalReplaySelection>, mut selections: ResMut<ExternalReplaySelections>| {
                    selections.0.push(selection);
                    Some(())
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.signal_entity(
                [scalar_observer, vector_observer, map_observer],
                |In(())| Some(()),
                OutputDelivery::broadcast(),
            )
        })
    }
}

#[test]
fn external_managed_replay_signals_normalize_and_are_selected_by_each_public_replay_policy() {
    let mut app = app();
    app.init_resource::<ExternalReplaySelections>();

    let scalar_cache = CustomReplayableScalar.cache().broadcast();
    let scalar_direct = scalar_cache.clone().register(app.world_mut());
    let scalar_boxed = scalar_cache
        .clone()
        .cache()
        .boxed_replayable()
        .register(app.world_mut());
    assert_eq!(scalar_direct.entity(), scalar_boxed.entity());
    let scalar = scalar_cache.cache().boxed_replayable();

    let vector_cache = CustomVector.cache().broadcast();
    let vector_direct = vector_cache.clone().register(app.world_mut());
    let vector_boxed = vector_cache
        .clone()
        .cache()
        .boxed_replayable()
        .register(app.world_mut());
    assert_eq!(vector_direct.entity(), vector_boxed.entity());
    let vector = vector_cache.cache().boxed_replayable();

    let map_cache = CustomMap.cache().broadcast();
    let map_direct = map_cache.clone().register(app.world_mut());
    let map_boxed = map_cache.clone().cache().boxed_replayable().register(app.world_mut());
    assert_eq!(map_direct.entity(), map_boxed.entity());
    let map = map_cache.cache().boxed_replayable();

    let dynamic = ExternalManagedReplayDynamic { scalar, vector, map }.register(app.world_mut());

    app.update();
    app.update();
    app.update();

    let selections = &app.world().resource::<ExternalReplaySelections>().0;
    assert_eq!(selections.len(), 3);
    assert!(selections.contains(&ExternalReplaySelection::Scalar(6)));
    assert!(selections.contains(&ExternalReplaySelection::Vector(vec![1])));
    assert!(selections.contains(&ExternalReplaySelection::Map(vec![(1, 2)])));

    dynamic.release(app.world_mut());
    scalar_direct.release(app.world_mut());
    scalar_boxed.release(app.world_mut());
    vector_direct.release(app.world_mut());
    vector_boxed.release(app.world_mut());
    map_direct.release(app.world_mut());
    map_boxed.release(app.world_mut());
}

#[derive(Resource, Default)]
struct ExternalCollectionReadiness {
    vectors: BTreeMap<&'static str, Vec<i32>>,
    maps: BTreeMap<&'static str, BTreeMap<i32, i32>>,
}

#[derive(Clone)]
enum ExternalManagedVector {
    Ready(Vec<i32>),
    Warming(&'static str),
}

impl SignalVec for ExternalManagedVector {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<Self::Item> {
        VecSignalMaterializer::new(move |registrar| match *self {
            ExternalManagedVector::Ready(values) => registrar.replayable_signal_vec_source(
                values,
                |In(())| None::<Vec<VecDiff<i32>>>,
                OutputDelivery::broadcast(),
            ),
            ExternalManagedVector::Warming(name) => registrar.replayable_signal_vec_entity(
                (),
                move |In(()), readiness: Res<ExternalCollectionReadiness>, mut emitted: Local<bool>| {
                    if *emitted {
                        return None;
                    }
                    let values = readiness.vectors.get(name).cloned()?;
                    *emitted = true;
                    Some(vec![VecDiff::Replace { values }])
                },
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            ),
        })
    }
}

fn ready_external_vector(values: impl Into<Vec<i32>>) -> ReplayableBroadcastVec<i32> {
    ExternalManagedVector::Ready(values.into()).cache().broadcast()
}

fn warming_external_vector(name: &'static str) -> ReplayableBroadcastVec<i32> {
    ExternalManagedVector::Warming(name).cache().broadcast()
}

#[derive(Clone)]
enum ExternalManagedMap {
    Ready(BTreeMap<i32, i32>),
    Warming(&'static str),
}

impl SignalMap for ExternalManagedMap {
    type Key = i32;
    type Value = i32;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<Self::Key, Self::Value> {
        MapSignalMaterializer::new(move |registrar| match *self {
            ExternalManagedMap::Ready(entries) => registrar.replayable_signal_map_source(
                entries,
                |In(())| None::<Vec<MapDiff<i32, i32>>>,
                OutputDelivery::broadcast(),
            ),
            ExternalManagedMap::Warming(name) => registrar.replayable_signal_map_entity(
                (),
                move |In(()), readiness: Res<ExternalCollectionReadiness>, mut emitted: Local<bool>| {
                    if *emitted {
                        return None;
                    }
                    let entries = readiness
                        .maps
                        .get(name)?
                        .iter()
                        .map(|(key, value)| (*key, *value))
                        .collect();
                    *emitted = true;
                    Some(vec![MapDiff::Replace { entries }])
                },
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            ),
        })
    }
}

fn ready_external_map(entries: impl IntoIterator<Item = (i32, i32)>) -> ReplayableBroadcastMap<i32, i32> {
    ExternalManagedMap::Ready(entries.into_iter().collect())
        .cache()
        .broadcast()
}

fn warming_external_map(name: &'static str) -> ReplayableBroadcastMap<i32, i32> {
    ExternalManagedMap::Warming(name).cache().broadcast()
}

fn apply_public_vec_diffs(values: &mut Vec<i32>, diffs: Vec<VecDiff<i32>>) {
    for diff in diffs {
        match diff {
            VecDiff::Replace { values: replacement } => *values = replacement,
            VecDiff::InsertAt { index, value } => {
                if index <= values.len() {
                    values.insert(index, value);
                }
            }
            VecDiff::UpdateAt { index, value } => {
                if let Some(current) = values.get_mut(index) {
                    *current = value;
                }
            }
            VecDiff::RemoveAt { index } => {
                if index < values.len() {
                    values.remove(index);
                }
            }
            VecDiff::Move { old_index, new_index } => {
                if old_index < values.len() && new_index < values.len() {
                    let value = values.remove(old_index);
                    values.insert(new_index, value);
                }
            }
            VecDiff::Push { value } => values.push(value),
            VecDiff::Pop => values.truncate(values.len().saturating_sub(1)),
            VecDiff::Clear => values.clear(),
        }
    }
}

fn apply_public_map_diffs(entries: &mut BTreeMap<i32, i32>, diffs: Vec<MapDiff<i32, i32>>) {
    for diff in diffs {
        match diff {
            MapDiff::Replace { entries: replacement } => *entries = replacement.into_iter().collect(),
            MapDiff::Insert { key, value } | MapDiff::Update { key, value } => entries.extend([(key, value)]),
            MapDiff::Remove { key } => entries.retain(|current, _| *current != key),
            MapDiff::Clear => entries.clear(),
        }
    }
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct VectorDynamicOutputs(Vec<Vec<i32>>);

struct ExternalVectorKeepCurrentCancellation {
    current: ReplayableBroadcastVec<i32>,
    warming: ReplayableBroadcastVec<i32>,
}

impl Signal for ExternalVectorKeepCurrentCancellation {
    type Item = Vec<i32>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { current, warming } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let command = match *update {
                        0 => Some(ScalarSlotCommand::Select {
                            signal: current.clone(),
                        }),
                        1 => Some(ScalarSlotCommand::Select {
                            signal: warming.clone(),
                        }),
                        2 => Some(ScalarSlotCommand::Select {
                            signal: current.clone(),
                        }),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    command.map(|command| DynamicManagerUpdate::new([command], DynamicEmissions::<Vec<i32>>::None))
                },
                OutputDelivery::broadcast(),
            )?;
            let mut current_values = Vec::new();
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_signal_vec::<i32>()
                    .adapter(move |(), diffs| {
                        apply_public_vec_diffs(&mut current_values, diffs);
                        DynamicEmissions::One(current_values.clone())
                    })
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

#[test]
fn scalar_vector_keep_current_reselection_cancels_a_warming_candidate_and_ignores_its_late_replay() {
    let mut app = app();
    app.init_resource::<ExternalCollectionReadiness>();
    app.init_resource::<VectorDynamicOutputs>();
    let current = ready_external_vector(vec![1, 2]);
    let registration = ExternalVectorKeepCurrentCancellation {
        current: current.clone(),
        warming: warming_external_vector("scalar-vector-b"),
    }
    .map(|In(values): In<Vec<i32>>, mut outputs: ResMut<VectorDynamicOutputs>| {
        outputs.0.push(values);
    })
    .register(app.world_mut());

    app.update();
    app.update();
    app.update();
    assert_eq!(app.world().resource::<VectorDynamicOutputs>().0, [vec![1, 2]]);

    app.world_mut()
        .resource_mut::<ExternalCollectionReadiness>()
        .vectors
        .insert("scalar-vector-b", vec![8, 9]);
    app.update();
    app.update();

    assert_eq!(app.world().resource::<VectorDynamicOutputs>().0, [vec![1, 2]]);
    registration.release(app.world_mut());
}

#[derive(Resource, Default, Debug, PartialEq, Eq)]
struct MapDynamicOutputs(Vec<Vec<(i32, i32)>>);

struct ExternalMapKeepCurrentCancellation {
    current: ReplayableBroadcastMap<i32, i32>,
    warming: ReplayableBroadcastMap<i32, i32>,
}

impl Signal for ExternalMapKeepCurrentCancellation {
    type Item = Vec<(i32, i32)>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { current, warming } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let command = match *update {
                        0 => Some(ScalarSlotCommand::Select {
                            signal: current.clone(),
                        }),
                        1 => Some(ScalarSlotCommand::Select {
                            signal: warming.clone(),
                        }),
                        2 => Some(ScalarSlotCommand::Select {
                            signal: current.clone(),
                        }),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    command
                        .map(|command| DynamicManagerUpdate::new([command], DynamicEmissions::<Vec<(i32, i32)>>::None))
                },
                OutputDelivery::broadcast(),
            )?;
            let mut current_entries = BTreeMap::new();
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_signal_map::<i32, i32>()
                    .adapter(move |(), diffs| {
                        apply_public_map_diffs(&mut current_entries, diffs);
                        DynamicEmissions::One(current_entries.iter().map(|(key, value)| (*key, *value)).collect())
                    })
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

#[test]
fn scalar_map_keep_current_reselection_cancels_a_warming_candidate_and_ignores_its_late_replay() {
    let mut app = app();
    app.init_resource::<ExternalCollectionReadiness>();
    app.init_resource::<MapDynamicOutputs>();
    let current = ready_external_map([(1, 10), (2, 20)]);
    let registration = ExternalMapKeepCurrentCancellation {
        current: current.clone(),
        warming: warming_external_map("scalar-map-b"),
    }
    .map(
        |In(entries): In<Vec<(i32, i32)>>, mut outputs: ResMut<MapDynamicOutputs>| {
            outputs.0.push(entries);
        },
    )
    .register(app.world_mut());

    app.update();
    app.update();
    app.update();
    assert_eq!(app.world().resource::<MapDynamicOutputs>().0, [vec![(1, 10), (2, 20)]]);

    app.world_mut()
        .resource_mut::<ExternalCollectionReadiness>()
        .maps
        .insert("scalar-map-b", [(8, 80), (9, 90)].into_iter().collect());
    app.update();
    app.update();

    assert_eq!(app.world().resource::<MapDynamicOutputs>().0, [vec![(1, 10), (2, 20)]]);
    registration.release(app.world_mut());
}

struct ExternalPendingKeyedIntent {
    inserted: ReplayableBroadcast<i32>,
    replacement: ReplayableBroadcast<i32>,
    final_signal: ReplayableBroadcast<i32>,
    errors: Arc<AtomicUsize>,
}

impl Signal for ExternalPendingKeyedIntent {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            inserted,
            replacement,
            final_signal,
            errors,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let result = match *update {
                        0 => Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                            KeyedSlotCommand::Insert {
                                key: 10,
                                signal: inserted.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending-keyed:insert-a")),
                        )])),
                        1 => Some(DynamicManagerUpdate::sequenced([DynamicManagerStep::command(
                            KeyedSlotCommand::Replace {
                                key: 10,
                                signal: replacement.clone(),
                            },
                            DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending-keyed:replace-a")),
                        )])),
                        2 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Rekey {
                                    old_key: 10,
                                    new_key: 11,
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending-keyed:rekey")),
                            ),
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Remove { key: 11 },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending-keyed:remove")),
                            ),
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Insert {
                                    key: 12,
                                    signal: final_signal.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("pending-keyed:insert-final")),
                            ),
                        ])),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    result
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One(PublicDynamicQueueItem::Keyed { key, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        assert_eq!(error, SignalRegistrationError::InvalidDynamicCommand);
                        errors.fetch_add(1, Ordering::SeqCst);
                    }),
            )
        })
    }
}

#[test]
fn keyed_pending_commands_follow_latest_desired_intent_and_ignore_superseded_readiness() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let errors = Arc::new(AtomicUsize::new(0));
    let registration = ExternalPendingKeyedIntent {
        inserted: warming_external_scalar("pending-keyed-stale-insert"),
        replacement: warming_external_scalar("pending-keyed-stale-replace"),
        final_signal: warming_external_scalar("pending-keyed-final"),
        errors: errors.clone(),
    }
    .map(
        |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
            outputs.0.push(items);
        },
    )
    .register(app.world_mut());

    app.update();
    app.update();
    app.update();
    assert!(app.world().resource::<PublicDynamicQueueOutputs>().0.is_empty());
    assert_eq!(errors.load(Ordering::SeqCst), 0);

    {
        let mut readiness = app.world_mut().resource_mut::<ExternalScalarReadiness>();
        readiness.0.insert("pending-keyed-stale-insert", 71);
        readiness.0.insert("pending-keyed-stale-replace", 72);
    }
    app.update();
    app.update();
    assert!(app.world().resource::<PublicDynamicQueueOutputs>().0.is_empty());
    assert_eq!(errors.load(Ordering::SeqCst), 0);

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("pending-keyed-final", 73);
    app.update();
    app.update();

    assert_eq!(
        flattened_public_dynamic_queue(&app),
        [
            PublicDynamicQueueItem::Manager("pending-keyed:insert-a"),
            PublicDynamicQueueItem::Manager("pending-keyed:replace-a"),
            PublicDynamicQueueItem::Manager("pending-keyed:rekey"),
            PublicDynamicQueueItem::Manager("pending-keyed:remove"),
            PublicDynamicQueueItem::Manager("pending-keyed:insert-final"),
            PublicDynamicQueueItem::Keyed { key: 12, value: 73 },
        ]
    );
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    registration.release(app.world_mut());
}

struct ExternalIndexedKeepCurrentReplacement {
    current: ReplayableBroadcast<i32>,
    warming: ReplayableBroadcast<i32>,
    errors: Arc<AtomicUsize>,
}

impl Signal for ExternalIndexedKeepCurrentReplacement {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            current,
            warming,
            errors,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let command = match *update {
                        0 => Some(IndexedSlotCommand::Insert {
                            index: 0,
                            signal: current.clone(),
                        }),
                        1 => Some(IndexedSlotCommand::Replace {
                            index: 0,
                            signal: warming.clone(),
                        }),
                        2 => Some(IndexedSlotCommand::Replace {
                            index: 0,
                            signal: current.clone(),
                        }),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    command.map(|command| {
                        DynamicManagerUpdate::new([command], DynamicEmissions::<PublicDynamicQueueItem>::None)
                    })
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One(PublicDynamicQueueItem::Indexed { index, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |_error| {
                        errors.fetch_add(1, Ordering::SeqCst);
                    }),
            )
        })
    }
}

#[test]
fn indexed_keep_current_reselection_cancels_a_warming_replacement_without_duplicate_output() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let errors = Arc::new(AtomicUsize::new(0));
    let current = ready_external_scalar(81);
    let registration = ExternalIndexedKeepCurrentReplacement {
        current: current.clone(),
        warming: warming_external_scalar("indexed-keep-current-b"),
        errors: errors.clone(),
    }
    .map(
        |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
            outputs.0.push(items);
        },
    )
    .register(app.world_mut());

    app.update();
    let current_entity_count = app.world().entities().len();
    app.update();
    app.update();
    app.update();

    let expected = [PublicDynamicQueueItem::Indexed { index: 0, value: 81 }];
    assert_eq!(flattened_public_dynamic_queue(&app), expected);
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    assert_eq!(app.world().entities().len(), current_entity_count);

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("indexed-keep-current-b", 82);
    app.update();
    app.update();

    assert_eq!(flattened_public_dynamic_queue(&app), expected);
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    assert_eq!(app.world().entities().len(), current_entity_count);
    registration.release(app.world_mut());
}

struct ExternalKeyedKeepCurrentReplacement {
    current: ReplayableBroadcast<i32>,
    warming: ReplayableBroadcast<i32>,
    errors: Arc<AtomicUsize>,
}

impl Signal for ExternalKeyedKeepCurrentReplacement {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            current,
            warming,
            errors,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let command = match *update {
                        0 => Some(KeyedSlotCommand::Insert {
                            key: 20,
                            signal: current.clone(),
                        }),
                        1 => Some(KeyedSlotCommand::Replace {
                            key: 20,
                            signal: warming.clone(),
                        }),
                        2 => Some(KeyedSlotCommand::Replace {
                            key: 20,
                            signal: current.clone(),
                        }),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    command.map(|command| {
                        DynamicManagerUpdate::new([command], DynamicEmissions::<PublicDynamicQueueItem>::None)
                    })
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One(PublicDynamicQueueItem::Keyed { key, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |_error| {
                        errors.fetch_add(1, Ordering::SeqCst);
                    }),
            )
        })
    }
}

#[test]
fn keyed_keep_current_reselection_cancels_a_warming_replacement_without_duplicate_output() {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let errors = Arc::new(AtomicUsize::new(0));
    let current = ready_external_scalar(91);
    let registration = ExternalKeyedKeepCurrentReplacement {
        current: current.clone(),
        warming: warming_external_scalar("keyed-keep-current-b"),
        errors: errors.clone(),
    }
    .map(
        |In(items): In<Vec<PublicDynamicQueueItem>>, mut outputs: ResMut<PublicDynamicQueueOutputs>| {
            outputs.0.push(items);
        },
    )
    .register(app.world_mut());

    app.update();
    let current_entity_count = app.world().entities().len();
    app.update();
    app.update();
    app.update();

    let expected = [PublicDynamicQueueItem::Keyed { key: 20, value: 91 }];
    assert_eq!(flattened_public_dynamic_queue(&app), expected);
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    assert_eq!(app.world().entities().len(), current_entity_count);

    app.world_mut()
        .resource_mut::<ExternalScalarReadiness>()
        .0
        .insert("keyed-keep-current-b", 92);
    app.update();
    app.update();

    assert_eq!(flattened_public_dynamic_queue(&app), expected);
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    assert_eq!(app.world().entities().len(), current_entity_count);
    registration.release(app.world_mut());
}

struct ExternalIndexedSameBatchKeepCurrent {
    current: ReplayableBroadcast<i32>,
    replacement: ReplayableBroadcast<i32>,
}

impl Signal for ExternalIndexedSameBatchKeepCurrent {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { current, replacement } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let result = match *update {
                        0 => Some(DynamicManagerUpdate::new(
                            [IndexedSlotCommand::Insert {
                                index: 0,
                                signal: current.clone(),
                            }],
                            DynamicEmissions::<PublicDynamicQueueItem>::None,
                        )),
                        1 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Replace {
                                    index: 0,
                                    signal: replacement.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("indexed-same-batch:b")),
                            ),
                            DynamicManagerStep::command(
                                IndexedSlotCommand::Replace {
                                    index: 0,
                                    signal: current.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("indexed-same-batch:a")),
                            ),
                        ])),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    result
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|index, value| DynamicEmissions::One(PublicDynamicQueueItem::Indexed { index, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

fn assert_indexed_same_batch_keep_current(replacement: ReplayableBroadcast<i32>, late_name: Option<&'static str>) {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let current = ready_external_scalar(131);
    let registration = register_public_queue_output(
        &mut app,
        ExternalIndexedSameBatchKeepCurrent {
            current: current.clone(),
            replacement,
        },
    );

    app.update();
    let current_entity_count = app.world().entities().len();
    app.update();
    app.update();
    let expected = [
        PublicDynamicQueueItem::Indexed { index: 0, value: 131 },
        PublicDynamicQueueItem::Manager("indexed-same-batch:b"),
        PublicDynamicQueueItem::Manager("indexed-same-batch:a"),
    ];
    assert_eq!(flattened_public_dynamic_queue(&app), expected);
    assert_eq!(app.world().entities().len(), current_entity_count);

    if let Some(name) = late_name {
        app.world_mut()
            .resource_mut::<ExternalScalarReadiness>()
            .0
            .insert(name, 132);
        app.update();
        app.update();
        assert_eq!(flattened_public_dynamic_queue(&app), expected);
        assert_eq!(app.world().entities().len(), current_entity_count);
    }
    registration.release(app.world_mut());
}

#[test]
fn indexed_same_batch_replacement_back_to_current_normalizes_ready_and_warming_candidates() {
    assert_indexed_same_batch_keep_current(ready_external_scalar(132), None);
    assert_indexed_same_batch_keep_current(
        warming_external_scalar("indexed-same-batch-warming"),
        Some("indexed-same-batch-warming"),
    );
}

struct ExternalKeyedSameBatchKeepCurrent {
    current: ReplayableBroadcast<i32>,
    replacement: ReplayableBroadcast<i32>,
}

impl Signal for ExternalKeyedSameBatchKeepCurrent {
    type Item = Vec<PublicDynamicQueueItem>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { current, replacement } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut update: Local<usize>| {
                    let result = match *update {
                        0 => Some(DynamicManagerUpdate::new(
                            [KeyedSlotCommand::Insert {
                                key: 30,
                                signal: current.clone(),
                            }],
                            DynamicEmissions::<PublicDynamicQueueItem>::None,
                        )),
                        1 => Some(DynamicManagerUpdate::sequenced([
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Replace {
                                    key: 30,
                                    signal: replacement.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("keyed-same-batch:b")),
                            ),
                            DynamicManagerStep::command(
                                KeyedSlotCommand::Replace {
                                    key: 30,
                                    signal: current.clone(),
                                },
                                DynamicEmissions::One(PublicDynamicQueueItem::Manager("keyed-same-batch:a")),
                            ),
                        ])),
                        _ => None,
                    };
                    *update = update.saturating_add(1);
                    result
                },
                OutputDelivery::broadcast(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::keyed(manager)
                    .replay_scalar::<i32>()
                    .adapter(|key, value| DynamicEmissions::One(PublicDynamicQueueItem::Keyed { key, value }))
                    .scalar_output(Some)
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )
        })
    }
}

fn assert_keyed_same_batch_keep_current(replacement: ReplayableBroadcast<i32>, late_name: Option<&'static str>) {
    let mut app = app();
    app.init_resource::<ExternalScalarReadiness>();
    app.init_resource::<PublicDynamicQueueOutputs>();
    let current = ready_external_scalar(141);
    let registration = register_public_queue_output(
        &mut app,
        ExternalKeyedSameBatchKeepCurrent {
            current: current.clone(),
            replacement,
        },
    );

    app.update();
    let current_entity_count = app.world().entities().len();
    app.update();
    app.update();
    let expected = [
        PublicDynamicQueueItem::Keyed { key: 30, value: 141 },
        PublicDynamicQueueItem::Manager("keyed-same-batch:b"),
        PublicDynamicQueueItem::Manager("keyed-same-batch:a"),
    ];
    assert_eq!(flattened_public_dynamic_queue(&app), expected);
    assert_eq!(app.world().entities().len(), current_entity_count);

    if let Some(name) = late_name {
        app.world_mut()
            .resource_mut::<ExternalScalarReadiness>()
            .0
            .insert(name, 142);
        app.update();
        app.update();
        assert_eq!(flattened_public_dynamic_queue(&app), expected);
        assert_eq!(app.world().entities().len(), current_entity_count);
    }
    registration.release(app.world_mut());
}

#[test]
fn keyed_same_batch_replacement_back_to_current_normalizes_ready_and_warming_candidates() {
    assert_keyed_same_batch_keep_current(ready_external_scalar(142), None);
    assert_keyed_same_batch_keep_current(
        warming_external_scalar("keyed-same-batch-warming"),
        Some("keyed-same-batch-warming"),
    );
}
