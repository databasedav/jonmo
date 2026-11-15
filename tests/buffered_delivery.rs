//! Public regressions for collection batch delivery and scalar drain coalescing.

extern crate alloc;

use alloc::{collections::BTreeMap, sync::Arc};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use core::fmt::Debug;
#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
use jonmo::{
    MapSignalMaterializer, OutputDelivery, ReplayInitialization, ScalarSignalMaterializer, VecSignalMaterializer,
    prelude::*,
};
use std::sync::Mutex;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Producer;

#[derive(Resource, Default, Debug)]
struct Observations {
    direct: Vec<i32>,
    mapped: Vec<i32>,
    mapper_calls: Vec<i32>,
    terminal_calls: Vec<i32>,
    release_order: Vec<String>,
}

#[derive(Resource, Default)]
struct SelfClaim(Option<SignalClaim>);

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<Producer>());
    app.init_resource::<Observations>();
    app.init_resource::<PendingInput>();
    app
}

fn push_and_stage(app: &mut App, values: &MutableVec<i32>, value: i32) {
    values.write(app.world_mut()).push(value);
    app.world_mut().run_schedule(Producer);
}

#[test]
fn two_source_polls_must_preserve_both_pushes_through_map() {
    let mut app = app();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let source = values.signal_vec().schedule::<Producer>().register(app.world_mut());
    let direct = values
        .signal_vec()
        .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, mut seen: ResMut<Observations>| {
            for diff in diffs {
                diff.apply_to_vec(&mut seen.direct);
            }
        })
        .schedule::<Last>()
        .register(app.world_mut());
    let mapped = values
        .signal_vec()
        .map(|In(value): In<i32>, mut seen: ResMut<Observations>| {
            seen.mapper_calls.push(value);
            value * 10
        })
        .schedule::<Last>()
        .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, mut seen: ResMut<Observations>| {
            for diff in diffs {
                diff.apply_to_vec(&mut seen.mapped);
            }
        })
        .register(app.world_mut());
    app.world_mut().run_schedule(Producer);
    app.update();
    assert!(app.world().resource::<Observations>().direct.is_empty());
    assert!(app.world().resource::<Observations>().mapped.is_empty());
    push_and_stage(&mut app, &values, 1);
    push_and_stage(&mut app, &values, 2);
    assert!(app.world().resource::<Observations>().direct.is_empty());
    assert!(app.world().resource::<Observations>().mapper_calls.is_empty());
    assert!(app.world().resource::<Observations>().mapped.is_empty());
    app.update();
    let seen = app.world().resource::<Observations>();
    println!(
        "two evaluations: direct={:?}, mapper_calls={:?}, mapped={:?}; expected mapped=[10, 20]",
        seen.direct, seen.mapper_calls, seen.mapped
    );
    assert_eq!(seen.direct, [1, 2], "direct branch control");
    assert_eq!(seen.mapper_calls, [1, 2], "mapper executed both inputs");
    let actual = seen.mapped.clone();
    direct.release(app.world_mut());
    mapped.release(app.world_mut());
    source.release(app.world_mut());
    assert_eq!(actual, [10, 20], "an item map must preserve both source diffs");
}

#[test]
fn two_source_polls_direct_branch_control() {
    let mut app = app();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let source = values.signal_vec().schedule::<Producer>().register(app.world_mut());
    let direct = values
        .signal_vec()
        .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, mut seen: ResMut<Observations>| {
            for diff in diffs {
                diff.apply_to_vec(&mut seen.direct);
            }
        })
        .schedule::<Last>()
        .register(app.world_mut());
    app.world_mut().run_schedule(Producer);
    app.update();
    push_and_stage(&mut app, &values, 1);
    push_and_stage(&mut app, &values, 2);
    assert!(app.world().resource::<Observations>().direct.is_empty());
    app.update();
    assert_eq!(app.world().resource::<Observations>().direct, [1, 2]);
    direct.release(app.world_mut());
    source.release(app.world_mut());
}

#[test]
fn two_source_polls_mapper_execution_control() {
    let mut app = app();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let source = values.signal_vec().schedule::<Producer>().register(app.world_mut());
    let mapper = values
        .signal_vec()
        .map(|In(value): In<i32>, mut seen: ResMut<Observations>| {
            seen.mapper_calls.push(value);
            value * 10
        })
        .schedule::<Last>()
        .register(app.world_mut());
    app.world_mut().run_schedule(Producer);
    app.update();
    push_and_stage(&mut app, &values, 1);
    push_and_stage(&mut app, &values, 2);
    assert!(app.world().resource::<Observations>().mapper_calls.is_empty());
    app.update();
    assert_eq!(app.world().resource::<Observations>().mapper_calls, [1, 2]);
    mapper.release(app.world_mut());
    source.release(app.world_mut());
}

#[test]
fn self_release_on_first_buffered_event_must_stop_second_callback() {
    let mut app = app();
    app.init_resource::<SelfClaim>();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let source = values.signal_vec().schedule::<Producer>().register(app.world_mut());
    let terminal = values
        .signal_vec()
        .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
            for diff in diffs {
                if let VecDiff::Push { value } = diff {
                    world.resource_mut::<Observations>().terminal_calls.push(value);
                    world
                        .resource_mut::<Observations>()
                        .release_order
                        .push(format!("callback({value})"));
                    let claim = world.resource_mut::<SelfClaim>().0.take();
                    if let Some(claim) = claim {
                        claim.try_release(world).expect("self release must succeed");
                        world
                            .resource_mut::<Observations>()
                            .release_order
                            .push("release returned Ok".into());
                    }
                }
            }
        })
        .schedule::<Last>()
        .broadcast();
    // Keep the dormant terminal alive: this is deactivation, not stale-entity execution.
    let claim = terminal.clone().register(app.world_mut());
    let terminal_entity = claim.entity();
    app.world_mut().run_schedule(Producer);
    app.update();
    app.world_mut().resource_mut::<SelfClaim>().0 = Some(claim);
    push_and_stage(&mut app, &values, 1);
    push_and_stage(&mut app, &values, 2);
    assert!(app.world().resource::<Observations>().terminal_calls.is_empty());
    app.update();
    assert!(app.world().resource::<SelfClaim>().0.is_none());
    assert!(
        app.world().get_entity(terminal_entity.entity()).is_ok(),
        "retained terminal still exists"
    );
    let actual = app.world().resource::<Observations>().terminal_calls.clone();
    println!(
        "self release: {:?}; expected only callback(1)",
        app.world().resource::<Observations>().release_order
    );
    push_and_stage(&mut app, &values, 3);
    app.update();
    assert_eq!(
        app.world().resource::<Observations>().terminal_calls,
        actual,
        "deactivation blocks future frames"
    );
    source.release(app.world_mut());
    drop(terminal);
    assert_eq!(
        actual,
        [1],
        "release must also block the remaining locally buffered callback"
    );
}

#[derive(Resource, Default)]
struct PendingInput(Option<usize>);

fn input_source() -> Broadcast<usize> {
    jonmo::signal::from_system(|In(()), mut pending: ResMut<PendingInput>| pending.0.take())
        .schedule::<Producer>()
        .broadcast()
}

fn queue_inputs(app: &mut App, source: &SignalClaim, drain: Drain, inputs: impl IntoIterator<Item = usize>) {
    let _ = source;
    for input in inputs {
        app.world_mut().resource_mut::<PendingInput>().0 = Some(input);
        match drain {
            Drain::Schedule => app.world_mut().run_schedule(Producer),
            #[cfg(feature = "imperative-signal-evaluation")]
            Drain::Poll | Drain::PollSecondLevel => {
                assert_eq!(
                    run_signal_subgraph_once(app.world_mut(), source.entity()).and_then(downcast_signal_value::<usize>),
                    Some(input),
                    "each separate root evaluation queues one downstream input"
                );
            }
        }
    }
}

type Log<T> = Arc<Mutex<Vec<T>>>;

fn record<T>(log: &Log<T>, value: T) {
    log.lock().expect("observation lock").push(value);
}

fn recorded<T: Clone>(log: &Log<T>) -> Vec<T> {
    log.lock().expect("observation lock").clone()
}

fn clear<T>(log: &Log<T>) {
    log.lock().expect("observation lock").clear();
}

fn assert_structure(actual: &impl Debug, expected: &impl Debug) {
    // Diff types only implement PartialEq in jonmo's private unit tests. Debug includes every
    // variant and field, so this also distinguishes batch boundaries and redundant operations.
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

#[derive(Clone, Copy, Debug)]
enum Drain {
    Schedule,
    #[cfg(feature = "imperative-signal-evaluation")]
    Poll,
    #[cfg(feature = "imperative-signal-evaluation")]
    PollSecondLevel,
}

macro_rules! collection_tests {
    ($signal:ident, {$($associated:item)*}, $materializer:ty, $into:ident, $materialize:ident,
     $ordinary:ident, $replayable:ident) => {
        struct ExternalCollection {
            source: Broadcast<usize>,
            outputs: Vec<Option<Batch>>,
            inputs: Log<usize>,
            replayable: bool,
        }

        impl $signal for ExternalCollection {
            $($associated)*

            fn $into(self: Box<Self>) -> $materializer {
                let Self { source, outputs, inputs, replayable } = *self;
                <$materializer>::new(move |registrar| {
                    let source = registrar.materialize_signal(Box::new(source).into_signal_materializer())?;
                    let system = move |In(input): In<usize>| {
                        record(&inputs, input);
                        outputs.get(input).expect("scripted input").clone()
                    };
                    if replayable {
                        registrar.$replayable(
                            source,
                            system,
                            ReplayInitialization::Warming,
                            OutputDelivery::broadcast(),
                        )
                    } else {
                        registrar.$ordinary(source, system, OutputDelivery::broadcast())
                    }
                })
            }
        }

        struct ExternalForward {
            source: $materializer,
            inputs: Log<Batch>,
        }

        impl $signal for ExternalForward {
            $($associated)*

            fn $into(self: Box<Self>) -> $materializer {
                let Self { source, inputs } = *self;
                <$materializer>::new(move |registrar| {
                    let source = registrar.$materialize(source)?;
                    registrar.$ordinary(
                        source,
                        move |In(batch): In<Batch>| {
                            record(&inputs, batch.clone());
                            Some(batch)
                        },
                        OutputDelivery::broadcast(),
                    )
                })
            }
        }

        fn collection_case(outputs: Vec<Option<Batch>>, drain: Drain, replayable: bool, replay_only: bool) {
            let mut app = app();
            let input = input_source();
            let source_claim = input.clone().register(app.world_mut());
            let calls = Log::default();
            let count = outputs.len();
            let expected = outputs.iter().flatten().cloned().collect::<Vec<_>>();
            // `.broadcast()` provides public `Clone` without adding a replay cache (normalizing
            // in-place when replayable, or adding one ordinary identity broadcast node otherwise).
            let collection = ExternalCollection {
                source: input,
                outputs: core::iter::once(Some(empty_snapshot())).chain(outputs).collect(),
                inputs: calls.clone(),
                replayable,
            }.schedule::<Last>().broadcast();
            let collection_claim = collection.clone().register(app.world_mut());
            let middle = Log::default();
            let forward = ExternalForward {
                source: Box::new(collection.clone()).$into(),
                inputs: middle.clone(),
            }.schedule::<Last>().broadcast();
            let forward_claim = forward.clone().register(app.world_mut());
            let terminal = Log::default();
            let terminal_claim = forward.for_each({
                let terminal = terminal.clone();
                move |In(batch): In<Batch>| record(&terminal, batch)
            }).register(app.world_mut());

            // Consume the initial snapshot before measuring deltas: pending replay snapshots are
            // intentionally replaceable by fresher full state, unlike published delta batches.
            queue_inputs(&mut app, &source_claim, drain, [0]);
            app.update();
            assert_structure(&recorded(&terminal), &vec![empty_snapshot()]);
            clear(&calls);
            clear(&middle);
            clear(&terminal);

            queue_inputs(&mut app, &source_claim, drain, 1..=count);
            let evaluated = match drain {
                Drain::Schedule => {
                    app.update();
                    None::<Batch>
                }
                #[cfg(feature = "imperative-signal-evaluation")]
                Drain::Poll => run_signal_subgraph_once(app.world_mut(), collection_claim.entity())
                    .map(|output| downcast_signal_value::<Batch>(output).expect("typed collection evaluation")),
                #[cfg(feature = "imperative-signal-evaluation")]
                Drain::PollSecondLevel => run_signal_subgraph_once(app.world_mut(), forward_claim.entity())
                    .map(|output| downcast_signal_value::<Batch>(output).expect("typed second-level evaluation")),
            };
            app.update();
            let middle_after_drain = recorded(&middle);
            let terminal_after_drain = recorded(&terminal);
            println!("{drain:?}, replayable={replayable}: evaluated={evaluated:?}, middle={middle_after_drain:?}, terminal={terminal_after_drain:?}");

            // A later evaluation and multiple schedule passes must not redeliver the consumed history.
            match drain {
                Drain::Schedule => {}
                #[cfg(feature = "imperative-signal-evaluation")]
                Drain::Poll | Drain::PollSecondLevel => {
                    assert!(run_signal_subgraph_once(app.world_mut(), collection_claim.entity()).is_none());
                    assert!(run_signal_subgraph_once(app.world_mut(), forward_claim.entity()).is_none());
                }
            }
            for _ in 0..3 {
                app.update();
            }
            assert_eq!(recorded(&calls), (1..=count).collect::<Vec<_>>(), "all callbacks ran exactly once");
            assert_structure(&recorded(&middle), &middle_after_drain);
            assert_structure(&recorded(&terminal), &terminal_after_drain);

            let late = Log::default();
            let late_claim = replayable.then(|| collection.for_each({
                let late = late.clone();
                move |In(batch): In<Batch>| record(&late, batch)
            }).register(app.world_mut()));
            if late_claim.is_some() {
                for _ in 0..3 {
                    app.update();
                }
                println!("late replay: {:?}", recorded(&late));
            }
            if let Some(claim) = late_claim {
                claim.release(app.world_mut());
            }
            terminal_claim.release(app.world_mut());
            forward_claim.release(app.world_mut());
            collection_claim.release(app.world_mut());
            source_claim.release(app.world_mut());

            if replay_only {
                assert_structure(&recorded(&late), &vec![full_snapshot()]);
            } else {
                if !matches!(drain, Drain::Schedule) {
                    assert_structure(&evaluated, &Some(expected.iter().flatten().cloned().collect::<Batch>()));
                }
                assert_structure(&middle_after_drain, &expected);
                assert_structure(&terminal_after_drain, &expected);
                assert_eq!(state(&terminal_after_drain), full_state());
            }
        }

        fn mixed_outputs() -> Vec<Option<Batch>> {
            let mut outputs = vec![None];
            for batch in batches() {
                outputs.extend([Some(batch), None]);
            }
            outputs.insert(3, Some(Vec::new()));
            outputs
        }

        #[test]
        fn three_queued_batches_cross_two_downstream_levels() {
            collection_case(batches().into_iter().map(Some).collect(), Drain::Schedule, false, false);
        }

        #[test]
        fn mixed_none_including_trailing_none_preserves_some_and_empty_batches() {
            collection_case(mixed_outputs(), Drain::Schedule, false, false);
        }

        #[test]
        fn replace_and_incrementals_keep_order_and_batch_boundaries() {
            collection_case(replacement_batches().into_iter().map(Some).collect(), Drain::Schedule, false, false);
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        #[test]
        fn poll_returns_all_diffs_but_downstream_keeps_original_batches() {
            collection_case(replacement_batches().into_iter().map(Some).collect(), Drain::Poll, false, false);
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        #[test]
        fn poll_mixed_none_preserves_some_and_empty_batches() {
            collection_case(mixed_outputs(), Drain::Poll, false, false);
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        #[test]
        fn polling_second_level_preserves_all_upstream_batches() {
            collection_case(batches().into_iter().map(Some).collect(), Drain::PollSecondLevel, false, false);
        }

        #[test]
        fn replayable_published_batches_remain_unchanged() {
            collection_case(mixed_outputs(), Drain::Schedule, true, false);
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        #[test]
        fn replayable_poll_returns_history_without_flattening_downstream_batches() {
            collection_case(mixed_outputs(), Drain::Poll, true, false);
        }

        #[test]
        fn replay_cache_contains_full_state_after_scheduled_drain() {
            collection_case(batches().into_iter().map(Some).collect(), Drain::Schedule, true, true);
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        #[test]
        fn replay_cache_contains_full_state_after_poll_drain() {
            collection_case(batches().into_iter().map(Some).collect(), Drain::Poll, true, true);
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        #[test]
        fn poll_without_dependents_returns_all_diffs_once() {
            let mut app = app();
            let input = input_source();
            let source_claim = input.clone().register(app.world_mut());
            let outputs = batches();
            let calls = Log::default();
            let collection = Box::new(ExternalCollection {
                source: input,
                outputs: outputs.iter().cloned().map(Some).collect(),
                inputs: calls.clone(),
                replayable: false,
            }).$into().register(app.world_mut());
            queue_inputs(&mut app, &source_claim, Drain::Poll, 0..outputs.len());
            let evaluated = run_signal_subgraph_once(app.world_mut(), collection.entity())
                .and_then(downcast_signal_value::<Batch>);
            assert!(run_signal_subgraph_once(app.world_mut(), collection.entity()).is_none());
            app.update();
            assert!(run_signal_subgraph_once(app.world_mut(), collection.entity()).is_none());
            assert_eq!(recorded(&calls), (0..outputs.len()).collect::<Vec<_>>());
            collection.release(app.world_mut());
            source_claim.release(app.world_mut());
            assert_structure(&evaluated, &Some(outputs.into_iter().flatten().collect::<Batch>()));
        }
    };
}

mod vector {
    use super::*;
    type Batch = Vec<VecDiff<i32>>;

    fn empty_snapshot() -> Batch {
        vec![VecDiff::Replace { values: vec![] }]
    }

    fn batches() -> Vec<Batch> {
        vec![
            vec![VecDiff::Push { value: 11 }],
            vec![VecDiff::Push { value: 20 }],
            vec![VecDiff::Push { value: 30 }],
        ]
    }

    fn replacement_batches() -> Vec<Batch> {
        vec![
            vec![VecDiff::Push { value: 99 }],
            vec![VecDiff::Replace { values: vec![10] }, VecDiff::Push { value: 20 }],
            vec![VecDiff::Push { value: 30 }, VecDiff::UpdateAt { index: 0, value: 11 }],
        ]
    }

    fn full_state() -> Vec<i32> {
        vec![11, 20, 30]
    }

    fn full_snapshot() -> Batch {
        vec![VecDiff::Replace { values: full_state() }]
    }

    fn state(batches: &[Batch]) -> Vec<i32> {
        let mut state = Vec::new();
        for diff in batches.iter().flatten().cloned() {
            diff.apply_to_vec(&mut state);
        }
        state
    }

    collection_tests!(
        SignalVec,
        {
            type Item = i32;
        },
        VecSignalMaterializer<i32>,
        into_signal_vec_materializer,
        materialize_signal_vec,
        signal_vec_entity,
        replayable_signal_vec_entity
    );
}

mod map {
    use super::*;
    type Batch = Vec<MapDiff<i32, i32>>;

    fn empty_snapshot() -> Batch {
        vec![MapDiff::Replace { entries: vec![] }]
    }

    fn batches() -> Vec<Batch> {
        vec![
            vec![MapDiff::Insert { key: 1, value: 11 }],
            vec![MapDiff::Insert { key: 2, value: 20 }],
            vec![MapDiff::Insert { key: 3, value: 30 }],
        ]
    }

    fn replacement_batches() -> Vec<Batch> {
        vec![
            vec![MapDiff::Insert { key: 99, value: 99 }],
            vec![
                MapDiff::Replace { entries: vec![(1, 10)] },
                MapDiff::Insert { key: 2, value: 20 },
            ],
            vec![
                MapDiff::Update { key: 1, value: 11 },
                MapDiff::Insert { key: 3, value: 30 },
            ],
        ]
    }

    fn full_state() -> BTreeMap<i32, i32> {
        [(1, 11), (2, 20), (3, 30)].into_iter().collect()
    }

    fn full_snapshot() -> Batch {
        vec![MapDiff::Replace {
            entries: full_state().into_iter().collect(),
        }]
    }

    fn state(batches: &[Batch]) -> BTreeMap<i32, i32> {
        let mut state = BTreeMap::new();
        for diff in batches.iter().flatten() {
            match diff {
                MapDiff::Replace { entries } => state = entries.iter().copied().collect(),
                MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                    state.insert(*key, *value);
                }
                MapDiff::Remove { key } => {
                    state.remove(key);
                }
                MapDiff::Clear => state.clear(),
            }
        }
        state
    }

    collection_tests!(SignalMap, {type Key = i32; type Value = i32;}, MapSignalMaterializer<i32, i32>,
        into_signal_map_materializer, materialize_signal_map, signal_map_entity, replayable_signal_map_entity);

    #[test]
    fn two_source_polls_must_preserve_both_inserts_through_map_value() {
        let mut app = app();
        let input = input_source();
        let source_claim = input.clone().register(app.world_mut());
        let collection = ExternalCollection {
            source: input,
            outputs: vec![
                Some(empty_snapshot()),
                Some(vec![MapDiff::Insert { key: 1, value: 1 }]),
                Some(vec![MapDiff::Insert { key: 2, value: 2 }]),
            ],
            inputs: Log::default(),
            replayable: true,
        }
        .schedule::<Producer>()
        .broadcast();
        let collection_claim = collection.clone().register(app.world_mut());
        let direct = Log::default();
        let direct_claim = collection
            .clone()
            .for_each({
                let direct = direct.clone();
                move |In(batch): In<Batch>| record(&direct, batch)
            })
            .schedule::<Last>()
            .register(app.world_mut());
        let mapper_calls = Log::default();
        let mapped = Log::default();
        let mapped_claim = collection
            .map_value({
                let mapper_calls = mapper_calls.clone();
                move |In(value): In<i32>| {
                    record(&mapper_calls, value);
                    value * 10
                }
            })
            .schedule::<Last>()
            .for_each({
                let mapped = mapped.clone();
                move |In(batch): In<Batch>| record(&mapped, batch)
            })
            .register(app.world_mut());
        queue_inputs(&mut app, &source_claim, Drain::Schedule, [0]);
        app.update();
        for input in [1, 2] {
            queue_inputs(&mut app, &source_claim, Drain::Schedule, [input]);
        }
        assert!(recorded(&mapper_calls).is_empty());
        app.update();
        assert_eq!(
            state(&recorded(&direct)),
            [(1, 1), (2, 2)].into_iter().collect(),
            "direct branch control"
        );
        assert_eq!(recorded(&mapper_calls), [1, 2], "mapper execution control");
        let actual = state(&recorded(&mapped));
        direct_claim.release(app.world_mut());
        mapped_claim.release(app.world_mut());
        collection_claim.release(app.world_mut());
        source_claim.release(app.world_mut());
        assert_eq!(actual, [(1, 10), (2, 20)].into_iter().collect());
    }
}

struct ExternalScalar<T> {
    source: Broadcast<usize>,
    outputs: Vec<Option<T>>,
    inputs: Log<usize>,
}

impl<T: Clone + Send + Sync + 'static> Signal for ExternalScalar<T> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<T> {
        let Self {
            source,
            outputs,
            inputs,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let source = registrar.materialize_signal(Box::new(source).into_signal_materializer())?;
            registrar.replayable_signal_entity(
                source,
                move |In(input): In<usize>| {
                    record(&inputs, input);
                    outputs.get(input).expect("scripted scalar input").clone()
                },
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    }
}

fn scalar_case<T: Clone + Debug + Send + Sync + 'static>(outputs: Vec<Option<T>>, expected: T, drain: Drain) {
    let mut app = app();
    let input = input_source();
    let source_claim = input.clone().register(app.world_mut());
    let count = outputs.len();
    let calls = Log::default();
    let scalar = ExternalScalar {
        source: input,
        outputs,
        inputs: calls.clone(),
    }
    .schedule::<Last>()
    .broadcast();
    let scalar_claim = scalar.clone().register(app.world_mut());
    let delivered = Log::default();
    let observer = scalar
        .clone()
        .map({
            let delivered = delivered.clone();
            move |In(value): In<T>| record(&delivered, value)
        })
        .register(app.world_mut());
    queue_inputs(&mut app, &source_claim, drain, 0..count);
    match drain {
        Drain::Schedule => {}
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::Poll => {
            assert_structure(
                &run_signal_subgraph_once(app.world_mut(), scalar_claim.entity()).and_then(downcast_signal_value::<T>),
                &Some(expected.clone()),
            );
        }
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::PollSecondLevel => unreachable!("scalar_case only uses Schedule or Poll"),
    }
    app.update();
    assert_eq!(recorded(&calls), (0..count).collect::<Vec<_>>());
    assert_structure(&recorded(&delivered), &vec![expected.clone()]);
    match drain {
        Drain::Schedule => {}
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::Poll => {
            assert!(run_signal_subgraph_once(app.world_mut(), scalar_claim.entity()).is_none());
        }
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::PollSecondLevel => unreachable!("scalar_case only uses Schedule or Poll"),
    }
    let late = Log::default();
    let late_claim = scalar
        .map({
            let late = late.clone();
            move |In(value): In<T>| record(&late, value)
        })
        .register(app.world_mut());
    for _ in 0..3 {
        app.update();
    }
    assert_structure(&recorded(&late), &vec![expected.clone()]);
    assert_structure(&recorded(&delivered), &vec![expected]);
    late_claim.release(app.world_mut());
    observer.release(app.world_mut());
    scalar_claim.release(app.world_mut());
    source_claim.release(app.world_mut());
}

#[test]
fn scalar_scheduled_drain_keeps_last_some_despite_trailing_none() {
    scalar_case(vec![None, Some(10), None, Some(20), None], 20, Drain::Schedule);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn scalar_poll_keeps_last_some_despite_trailing_none() {
    scalar_case(vec![None, Some(10), None, Some(20), None], 20, Drain::Poll);
}

#[test]
fn scalar_vec_container_is_not_a_collection_batch_in_scheduled_drain() {
    scalar_case(
        vec![Some(vec![1, 2]), None, Some(vec![3]), None],
        vec![3],
        Drain::Schedule,
    );
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn scalar_vec_container_is_not_a_collection_batch_in_poll() {
    scalar_case(vec![Some(vec![1, 2]), None, Some(vec![3]), None], vec![3], Drain::Poll);
}

#[test]
fn scalar_vec_diff_container_still_coalesces_in_scheduled_drain() {
    scalar_case(
        vec![
            Some(vec![VecDiff::Push { value: 1 }]),
            Some(vec![VecDiff::Push { value: 2 }]),
            None,
        ],
        vec![VecDiff::Push { value: 2 }],
        Drain::Schedule,
    );
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn scalar_vec_diff_container_still_coalesces_in_poll() {
    scalar_case(
        vec![
            Some(vec![VecDiff::Push { value: 1 }]),
            Some(vec![VecDiff::Push { value: 2 }]),
            None,
        ],
        vec![VecDiff::Push { value: 2 }],
        Drain::Poll,
    );
}

#[test]
fn scalar_map_diff_container_still_coalesces_in_scheduled_drain() {
    scalar_case(
        vec![
            Some(vec![MapDiff::Insert { key: 1, value: 1 }]),
            Some(vec![MapDiff::Insert { key: 2, value: 2 }]),
            None,
        ],
        vec![MapDiff::Insert { key: 2, value: 2 }],
        Drain::Schedule,
    );
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn scalar_map_diff_container_still_coalesces_in_poll() {
    scalar_case(
        vec![
            Some(vec![MapDiff::Insert { key: 1, value: 1 }]),
            Some(vec![MapDiff::Insert { key: 2, value: 2 }]),
            None,
        ],
        vec![MapDiff::Insert { key: 2, value: 2 }],
        Drain::Poll,
    );
}
