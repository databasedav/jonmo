//! Ordered dynamic-manager consumption must not change ordinary scalar cadence.

extern crate alloc;

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use core::panic::AssertUnwindSafe;
use jonmo::prelude::*;
use std::panic::{catch_unwind, resume_unwind};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
struct SourceSchedule;

#[derive(Resource, Default, Debug)]
struct Observations {
    factory_calls: Vec<i32>,
    batches: Vec<Vec<VecDiff<i32>>>,
    values: Vec<i32>,
}

#[allow(
    clippy::std_instead_of_core,
    reason = "Keep the promoted review reproduction unchanged"
)]
fn run_case(intermediate: bool) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<SourceSchedule>());
    app.init_resource::<Observations>();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let factory = |In(value): In<i32>, mut seen: ResMut<Observations>| {
        println!("factory({value})");
        seen.factory_calls.push(value);
        signal::once(value).cache()
    };
    let observer = |In(diffs): In<Vec<VecDiff<i32>>>, mut seen: ResMut<Observations>| {
        println!("observer: {diffs:?}");
        seen.batches.push(diffs.clone());
        for diff in diffs {
            diff.apply_to_vec(&mut seen.values);
        }
    };
    let branch = if intermediate {
        values
            .signal_vec()
            .map_in(std::convert::identity)
            .map_signal(factory)
            .for_each(observer)
            .register(app.world_mut())
    } else {
        values
            .signal_vec()
            .map_signal(factory)
            .for_each(observer)
            .register(app.world_mut())
    };
    let source = values
        .signal_vec()
        .schedule::<SourceSchedule>()
        .register(app.world_mut());
    app.world_mut().run_schedule(SourceSchedule);
    app.update();
    assert!(app.world().resource::<Observations>().values.is_empty());
    println!("settled empty; intermediate={intermediate}");
    for value in [1, 2] {
        values.write(app.world_mut()).push(value);
        app.world_mut().run_schedule(SourceSchedule);
    }
    assert!(app.world().resource::<Observations>().factory_calls.is_empty());
    let outcome = catch_unwind(AssertUnwindSafe(|| app.update()));
    let seen = app.world().resource::<Observations>();
    println!(
        "POST-FIX intermediate={intermediate}, update_panicked={}, observations={seen:?}; expected values=[1, 2]",
        outcome.is_err()
    );
    assert_eq!(seen.factory_calls, [1, 2], "manager factory executes both inputs");
    let actual = seen.values.clone();
    if let Err(payload) = outcome {
        resume_unwind(payload);
    }
    branch.release(app.world_mut());
    source.release(app.world_mut());
    assert_eq!(actual, [1, 2]);
}

#[test]
fn direct_map_signal_control() {
    run_case(false);
}

#[test]
fn map_in_then_map_signal_must_preserve_both_pushes() {
    run_case(true);
}

mod external {
    use alloc::{
        collections::BTreeMap,
        sync::{Arc, Weak},
    };
    use bevy_app::{App, Last, MainScheduleOrder};
    use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
    #[cfg(feature = "imperative-signal-evaluation")]
    use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
    use jonmo::{
        DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand, OutputDelivery,
        ReplayInitialization, SameSignalIdentity, ScalarSignalMaterializer, SignalRegistrationError,
        VecSignalMaterializer, prelude::*,
    };
    use std::sync::Mutex;

    #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
    struct SourceSchedule;

    #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
    struct ManagerSchedule;

    type Update = DynamicManagerUpdate<IndexedSlotCommand<ReplayableBroadcast<i32>>, Item>;
    type Manager = ReplayableBroadcast<Update>;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Item {
        Manager { value: i32, sequence: usize },
        Selected { index: usize, value: i32 },
    }

    #[derive(Clone, Copy)]
    enum Program {
        Emissions,
        RejectMiddle,
        Capture,
        CaptureReady,
    }

    #[derive(Resource, Default)]
    struct Production {
        materializations: usize,
        items: Vec<Item>,
        latest_update: Option<Update>,
        capture: Option<Weak<()>>,
        selected_runs: usize,
    }

    #[derive(Resource, Default)]
    struct Ordinary(Vec<Update>);

    #[derive(Clone, Default)]
    struct Journal(Arc<Mutex<Vec<Vec<Item>>>>);

    impl Journal {
        fn batches(&self) -> Vec<Vec<Item>> {
            self.0.lock().expect("journal lock").clone()
        }

        fn items(&self) -> Vec<Item> {
            self.batches().into_iter().flatten().collect()
        }

        fn clear(&self) {
            self.0.lock().expect("journal lock").clear();
        }
    }

    struct Ready(i32);

    impl Signal for Ready {
        type Item = i32;

        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
            ScalarSignalMaterializer::new(move |registrar| {
                registrar.replayable_signal_source(self.0, |In(())| None::<i32>, OutputDelivery::broadcast())
            })
        }
    }

    struct ExternalManager {
        source: VecSignalMaterializer<i32>,
        program: Program,
    }

    impl Signal for ExternalManager {
        type Item = Update;

        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Update> {
            ScalarSignalMaterializer::new(move |registrar| {
                let source = registrar.materialize_signal_vec(self.source)?;
                let program = self.program;
                registrar.replayable_signal_entity(
                    source,
                    move |In(diffs): In<Vec<VecDiff<i32>>>,
                          mut sequence: Local<usize>,
                          mut initialized: Local<bool>,
                          mut production: ResMut<Production>| {
                        if !core::mem::replace(&mut *initialized, true) {
                            production.materializations += 1;
                        }
                        // Empty bootstrap snapshots are not future manager commands.
                        let value = match diffs.as_slice() {
                            [VecDiff::Replace { values }] if values.is_empty() => return None,
                            [VecDiff::Push { value }] => *value,
                            _ => panic!("fixture expects separately polled pushes: {diffs:?}"),
                        };
                        // A trailing suppressed input must not erase the last accepted scalar
                        // output.
                        if value == 0 {
                            return None;
                        }
                        *sequence += 1;
                        let item = Item::Manager {
                            value,
                            sequence: *sequence,
                        };
                        production.items.push(item.clone());
                        let commands = match (program, value) {
                            (Program::RejectMiddle, 1) => vec![IndexedSlotCommand::Insert {
                                index: 0,
                                signal: Ready(10).cache().broadcast(),
                            }],
                            (Program::RejectMiddle, 2) => vec![
                                IndexedSlotCommand::Insert {
                                    index: 1,
                                    signal: Ready(20).cache().broadcast(),
                                },
                                IndexedSlotCommand::Remove { index: 99 },
                            ],
                            (Program::RejectMiddle, 3) => vec![IndexedSlotCommand::Insert {
                                index: 1,
                                signal: Ready(30).cache().broadcast(),
                            }],
                            (Program::Capture | Program::CaptureReady, 1) => {
                                let ready = matches!(program, Program::CaptureReady);
                                let capture = Arc::new(());
                                production.capture = Some(Arc::downgrade(&capture));
                                let selected =
                                    signal::from_system(move |In(()), mut production: ResMut<Production>| {
                                        let _keep_alive = &capture;
                                        production.selected_runs += 1;
                                        ready.then_some(10)
                                    })
                                    .dedupe()
                                    .cache()
                                    .broadcast();
                                vec![IndexedSlotCommand::Insert {
                                    index: 0,
                                    signal: selected,
                                }]
                            }
                            _ => Vec::new(),
                        };
                        let update = DynamicManagerUpdate::new(commands, DynamicEmissions::One(item));
                        production.latest_update = Some(update.clone());
                        Some(update)
                    },
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            })
        }
    }

    struct ExternalDynamic<M = Manager> {
        manager: M,
        journal: Journal,
        errors: Arc<Mutex<Vec<SignalRegistrationError>>>,
        scheduled: bool,
    }

    impl<M: Signal<Item = Update>> Signal for ExternalDynamic<M> {
        type Item = Vec<Item>;

        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
            let ExternalDynamic {
                manager,
                journal,
                errors,
                scheduled,
            } = *self;
            let manager = Box::new(manager).into_signal_materializer();
            ScalarSignalMaterializer::new(move |registrar| {
                let manager = registrar.materialize_signal(manager)?;
                let output = registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::indexed(manager)
                        .replay_scalar::<i32>()
                        .adapter(|index, value| DynamicEmissions::One(Item::Selected { index, value }))
                        .replayable_scalar_output(move |items| {
                            // Observe each reduction, independently of the scalar terminal's
                            // last-Some cache.
                            journal.0.lock().expect("journal lock").push(items.clone());
                            Some(items)
                        })
                        .same_signal_identity(SameSignalIdentity::KeepCurrent)
                        .on_candidate_error(move |error| {
                            errors.lock().expect("error lock").push(error);
                        }),
                )?;
                if scheduled {
                    registrar.apply_schedule(output, Last)
                } else {
                    Ok(output)
                }
            })
        }
    }

    struct Fixture {
        app: App,
        values: MutableVec<i32>,
        source: SignalClaim,
        manager: Manager,
        manager_claim: SignalClaim,
    }

    impl Fixture {
        fn new(program: Program) -> Self {
            let mut app = App::new();
            app.add_plugins(
                JonmoPlugin::default()
                    .with_schedule::<SourceSchedule>()
                    .with_schedule::<ManagerSchedule>(),
            );
            app.world_mut()
                .resource_mut::<MainScheduleOrder>()
                .insert_before(Last, ManagerSchedule);
            app.init_resource::<Production>();
            app.init_resource::<Ordinary>();
            let values = MutableVec::<i32>::builder().spawn(app.world_mut());
            let manager = ExternalManager {
                source: Box::new(values.signal_vec().map_in(core::convert::identity)).into_signal_vec_materializer(),
                program,
            }
            .cache()
            .broadcast();
            let manager_stage = manager
                .clone()
                .map(|In(_)| ())
                .schedule::<ManagerSchedule>()
                .register(app.world_mut());
            let manager_claim = manager.clone().register(app.world_mut());
            manager_stage.release(app.world_mut());
            let source = values
                .signal_vec()
                .schedule::<SourceSchedule>()
                .register(app.world_mut());
            app.world_mut().run_schedule(SourceSchedule);
            app.update();
            Self {
                app,
                values,
                source,
                manager,
                manager_claim,
            }
        }

        fn consumer(&mut self, scheduled: bool) -> (SignalClaim, Journal, Arc<Mutex<Vec<SignalRegistrationError>>>) {
            let journal = Journal::default();
            let errors = Arc::new(Mutex::new(Vec::new()));
            let claim = ExternalDynamic {
                manager: self.manager.clone(),
                journal: journal.clone(),
                errors: errors.clone(),
                scheduled,
            }
            .register(self.app.world_mut());
            (claim, journal, errors)
        }

        fn ordinary(&mut self) -> SignalClaim {
            self.manager
                .clone()
                .map(|In(update): In<Update>, mut ordinary: ResMut<Ordinary>| {
                    ordinary.0.push(update);
                })
                .register(self.app.world_mut())
        }

        fn queue(&mut self, values: &[i32]) {
            for value in values {
                self.values.write(self.app.world_mut()).push(*value);
                self.app.world_mut().run_schedule(SourceSchedule);
            }
        }

        fn step_manager(&mut self) -> Option<Update> {
            self.app.world_mut().resource_mut::<Production>().latest_update = None;
            self.app.world_mut().run_schedule(ManagerSchedule);
            self.app.world_mut().resource_mut::<Production>().latest_update.take()
        }

        #[cfg(feature = "imperative-signal-evaluation")]
        fn evaluate_manager(&mut self) -> Option<Update> {
            run_signal_subgraph_once(self.app.world_mut(), self.manager_claim.entity())
                .and_then(downcast_signal_value::<Update>)
        }

        fn finish(mut self, claims: impl IntoIterator<Item = SignalClaim>) {
            for claim in claims {
                claim.release(self.app.world_mut());
            }
            self.manager_claim.release(self.app.world_mut());
            self.source.release(self.app.world_mut());
        }
    }

    fn markers(values: &[i32], start: usize) -> Vec<Item> {
        values
            .iter()
            .enumerate()
            .map(|(offset, value)| Item::Manager {
                value: *value,
                sequence: start + offset,
            })
            .collect()
    }

    fn decode_latest(update: Update) -> Vec<Item> {
        // DynamicManagerUpdate intentionally has no public field access. Interpret exactly the
        // supplied scalar value in a fresh external consumer, never the original manager's history.
        let mut app = App::new();
        app.add_plugins(JonmoPlugin::default());
        let journal = Journal::default();
        let errors = Arc::new(Mutex::new(Vec::new()));
        let claim = ExternalDynamic {
            manager: signal::once(update).cache().broadcast(),
            journal: journal.clone(),
            errors: errors.clone(),
            scheduled: false,
        }
        .register(app.world_mut());
        app.update();
        assert!(errors.lock().expect("error lock").is_empty());
        claim.release(app.world_mut());
        journal.items()
    }

    #[test]
    fn shared_manager_orders_bridge_but_ordinary_observer_remains_last_some() {
        let mut fixture = Fixture::new(Program::Emissions);
        let ordinary = fixture.ordinary();
        let (dynamic, journal, errors) = fixture.consumer(false);
        fixture.app.update();
        assert!(journal.items().is_empty());
        fixture.queue(&[1, 2, 3]);
        fixture.app.update();
        let observed = &fixture.app.world().resource::<Ordinary>().0;
        assert_eq!(
            observed.len(),
            1,
            "ordinary scalar cadence must not become ordered delivery"
        );
        assert_eq!(decode_latest(observed[0].clone()), markers(&[3], 3));
        assert_eq!(
            fixture.app.world().resource::<Production>().items,
            markers(&[1, 2, 3], 1)
        );
        assert!(errors.lock().expect("error lock").is_empty());
        let actual = journal.items();
        fixture.finish([ordinary, dynamic]);
        assert_eq!(actual, markers(&[1, 2, 3], 1));
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    #[test]
    fn scalar_manager_poll_and_late_observer_replay_only_last_some() {
        let mut fixture = Fixture::new(Program::Emissions);
        let (dynamic, _journal, _errors) = fixture.consumer(false);
        fixture.app.update();
        fixture.queue(&[1, 2, 3, 0]);
        let polled = fixture.evaluate_manager().expect("manager produces outputs");
        assert_eq!(decode_latest(polled), markers(&[3], 3));
        assert!(
            fixture.evaluate_manager().is_none(),
            "evaluation does not replay a consumed drain"
        );
        let late = fixture.ordinary();
        fixture.app.update();
        fixture.app.update();
        let observed = &fixture.app.world().resource::<Ordinary>().0;
        assert_eq!(observed.len(), 1, "late observer gets one current scalar revision");
        assert_eq!(decode_latest(observed[0].clone()), markers(&[3], 3));
        fixture.queue(&[4, 5, 0]);
        fixture.app.update();
        fixture.app.update();
        let observed = &fixture.app.world().resource::<Ordinary>().0;
        assert_eq!(observed.len(), 2, "future scalar revision is delivered once");
        assert_eq!(decode_latest(observed[1].clone()), markers(&[5], 5));
        fixture.finish([late, dynamic]);
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    #[test]
    fn two_dynamic_consumers_receive_independent_future_outputs() {
        let mut fixture = Fixture::new(Program::Emissions);
        let (first, first_journal, first_errors) = fixture.consumer(false);
        let (second, second_journal, second_errors) = fixture.consumer(true);
        fixture.app.update();
        fixture.queue(&[1, 2]);
        let _produced = fixture
            .evaluate_manager()
            .expect("produce both commands before either consumer");
        run_signal_subgraph_once(fixture.app.world_mut(), first.entity());
        assert!(
            second_journal.items().is_empty(),
            "polling one consumer must not execute the other"
        );
        fixture.app.update();
        let first_actual = first_journal.items();
        let second_initial = second_journal.items();
        first.release(fixture.app.world_mut());
        fixture.queue(&[3, 4]);
        fixture.app.update();
        fixture.app.update();
        assert_eq!(first_journal.items(), first_actual, "released consumer stays silent");
        assert!(first_errors.lock().expect("error lock").is_empty());
        assert!(second_errors.lock().expect("error lock").is_empty());
        let second_actual = second_journal.items();
        fixture.finish([second]);
        assert_eq!(first_actual, markers(&[1, 2], 1));
        assert_eq!(second_initial, markers(&[1, 2], 1));
        assert_eq!(second_actual, markers(&[1, 2, 3, 4], 1));
    }

    #[test]
    fn materialized_manager_preserves_identity_local_state_and_latest_bootstrap() {
        let mut fixture = Fixture::new(Program::Emissions);
        fixture.queue(&[7, 8]);
        let _produced = fixture
            .step_manager()
            .expect("history produced before dynamic relationship");
        fixture.app.update();
        let identity = fixture.manager_claim.entity();
        let (dynamic, journal, errors) = fixture.consumer(false);
        let second_claim = fixture.manager.clone().register(fixture.app.world_mut());
        assert_eq!(
            identity,
            second_claim.entity(),
            "enabling delivery cannot replace a live manager"
        );
        fixture.app.update();
        assert_eq!(
            journal.items(),
            markers(&[8], 2),
            "bootstrap is latest only, never reconstructed 7,8 history"
        );
        journal.clear();
        fixture.queue(&[9, 10]);
        fixture.app.update();
        assert_eq!(fixture.app.world().resource::<Production>().materializations, 1);
        assert_eq!(
            fixture.app.world().resource::<Production>().items,
            markers(&[7, 8, 9, 10], 1)
        );
        assert!(errors.lock().expect("error lock").is_empty());
        let actual = journal.items();
        fixture.finish([second_claim, dynamic]);
        assert_eq!(
            actual,
            markers(&[9, 10], 3),
            "future outputs retain the existing Local sequence"
        );
    }

    #[test]
    fn separate_updates_reject_b_atomically_and_continue_with_c() {
        let mut fixture = Fixture::new(Program::RejectMiddle);
        let (dynamic, journal, errors) = fixture.consumer(false);
        fixture.app.update();
        fixture.queue(&[1, 2, 3]);
        fixture.app.update();
        fixture.app.update();
        let actual_errors = errors.lock().expect("error lock").clone();
        let actual = journal.items();
        println!("A/B/C errors={actual_errors:?}, journal={actual:?}");
        fixture.finish([dynamic]);
        assert_eq!(actual_errors, [SignalRegistrationError::InvalidDynamicCommand]);
        assert_eq!(
            actual,
            [
                Item::Manager { value: 1, sequence: 1 },
                Item::Selected { index: 0, value: 10 },
                Item::Manager { value: 3, sequence: 3 },
                Item::Selected { index: 1, value: 30 },
            ],
            "B's valid prefix and emissions are rejected only with B, not merged with A or C"
        );
    }

    #[test]
    fn separately_drained_updates_control_rejects_b_and_continues_with_c() {
        let mut fixture = Fixture::new(Program::RejectMiddle);
        let (dynamic, journal, errors) = fixture.consumer(false);
        fixture.app.update();
        for value in [1, 2, 3] {
            fixture.queue(&[value]);
            fixture.app.update();
            fixture.app.update();
        }
        let actual_errors = errors.lock().expect("error lock").clone();
        let actual = journal.items();
        fixture.finish([dynamic]);
        assert_eq!(actual_errors, [SignalRegistrationError::InvalidDynamicCommand]);
        assert_eq!(
            actual,
            [
                Item::Manager { value: 1, sequence: 1 },
                Item::Selected { index: 0, value: 10 },
                Item::Manager { value: 3, sequence: 3 },
                Item::Selected { index: 1, value: 30 },
            ]
        );
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    #[test]
    fn poll_then_schedule_does_not_duplicate_ordered_manager_delivery() {
        let mut fixture = Fixture::new(Program::Emissions);
        let (dynamic, journal, errors) = fixture.consumer(true);
        fixture.app.update();
        fixture.queue(&[1, 2]);
        assert_eq!(
            decode_latest(fixture.evaluate_manager().expect("manager output")),
            markers(&[2], 2)
        );
        assert!(run_signal_subgraph_once(fixture.app.world_mut(), dynamic.entity()).is_some());
        let after_poll = journal.batches();
        assert!(run_signal_subgraph_once(fixture.app.world_mut(), dynamic.entity()).is_none());
        for _ in 0..3 {
            fixture.app.update();
        }
        assert_eq!(
            journal.batches(),
            after_poll,
            "scheduled replay must not duplicate polled commands"
        );
        assert!(errors.lock().expect("error lock").is_empty());
        let actual = journal.items();
        fixture.finish([dynamic]);
        assert_eq!(actual, markers(&[1, 2], 1));
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    #[test]
    fn scheduled_delivery_then_poll_does_not_repeat_ordered_outputs() {
        let mut fixture = Fixture::new(Program::Emissions);
        let (dynamic, journal, errors) = fixture.consumer(true);
        fixture.app.update();
        fixture.queue(&[1, 2]);
        fixture.app.update();
        let after_schedule = journal.batches();
        assert!(fixture.evaluate_manager().is_none());
        assert!(run_signal_subgraph_once(fixture.app.world_mut(), dynamic.entity()).is_none());
        for _ in 0..3 {
            fixture.app.update();
        }
        assert_eq!(
            journal.batches(),
            after_schedule,
            "consumed inputs and scalar revisions are not re-delivered"
        );
        assert!(errors.lock().expect("error lock").is_empty());
        let actual = journal.items();
        fixture.finish([dynamic]);
        assert_eq!(actual, markers(&[1, 2], 1));
    }

    #[test]
    fn queued_command_release_drops_capture_without_consuming() {
        let mut fixture = Fixture::new(Program::Capture);
        let (dynamic, journal, errors) = fixture.consumer(false);
        fixture.app.update();
        fixture.queue(&[1, 2]);
        // The final scalar update holds no candidate; only the relationship's earlier queued
        // command should retain this capture while the manager remains independently owned.
        let latest = fixture
            .step_manager()
            .expect("queue commands without running the bridge");
        assert_eq!(decode_latest(latest), markers(&[2], 2));
        let capture = fixture
            .app
            .world()
            .resource::<Production>()
            .capture
            .clone()
            .expect("created capture");
        let held_while_queued = capture.upgrade().is_some();
        assert_eq!(fixture.app.world().resource::<Production>().selected_runs, 0);
        dynamic.release(fixture.app.world_mut());
        assert!(
            capture.upgrade().is_none(),
            "release must discard the queued command's capture immediately"
        );
        fixture.app.update();
        assert_eq!(fixture.app.world().resource::<Production>().selected_runs, 0);
        assert!(
            journal.items().is_empty(),
            "release cannot drain commands into user callbacks"
        );
        assert!(errors.lock().expect("error lock").is_empty());
        fixture.finish([]);
        assert!(
            held_while_queued,
            "earlier accepted command must be retained until consumption or relationship release"
        );
    }

    // These cases were added after the ordered-bridge and validator fixes; they have no pre-fix
    // baseline.
    mod post_fix {
        use super::*;

        fn settle(fixture: &mut Fixture) {
            for _ in 0..3 {
                fixture.app.update();
            }
        }

        fn assert_late_manager_snapshot(fixture: &mut Fixture, expected: Item) -> SignalClaim {
            assert!(fixture.app.world().resource::<Ordinary>().0.is_empty());
            let late = fixture.ordinary();
            settle(fixture);
            let observed = &fixture.app.world().resource::<Ordinary>().0;
            assert_eq!(
                observed.len(),
                1,
                "late ordinary observer reads one current scalar cache value"
            );
            assert_eq!(decode_latest(observed[0].clone()), [expected]);
            late
        }

        #[test]
        fn undrained_ready_bootstrap_a_precedes_future_b_c_exactly_once() {
            let mut fixture = Fixture::new(Program::Emissions);
            fixture.queue(&[1]);
            assert_eq!(
                decode_latest(fixture.step_manager().expect("produce A before attachment")),
                markers(&[1], 1)
            );
            let (dynamic, journal, errors) = fixture.consumer(false);
            assert!(
                journal.items().is_empty(),
                "registration must not drain the queued ready bootstrap"
            );
            fixture.queue(&[2, 3]);
            assert_eq!(
                decode_latest(fixture.step_manager().expect("produce B and C before bridge")),
                markers(&[3], 3)
            );
            assert!(journal.items().is_empty(), "only the manager has been drained");
            let late = assert_late_manager_snapshot(&mut fixture, Item::Manager { value: 3, sequence: 3 });
            assert_eq!(
                journal.items(),
                markers(&[1, 2, 3], 1),
                "ready A is neither overwritten by C nor replayed twice"
            );
            #[cfg(feature = "imperative-signal-evaluation")]
            assert!(run_signal_subgraph_once(fixture.app.world_mut(), dynamic.entity()).is_none());
            settle(&mut fixture);
            assert_eq!(journal.items(), markers(&[1, 2, 3], 1));
            assert!(errors.lock().expect("error lock").is_empty());
            fixture.finish([dynamic, late]);
        }

        #[test]
        fn materialized_warming_manager_first_burst_is_all_future_outputs_not_latest_bootstrap() {
            let mut fixture = Fixture::new(Program::Emissions);
            assert!(fixture.app.world().resource::<Production>().items.is_empty());
            assert!(
                fixture.step_manager().is_none(),
                "already materialized but still warming"
            );
            let identity = fixture.manager_claim.entity();
            let (dynamic, journal, errors) = fixture.consumer(false);
            // Do not run the bridge between attaching to Warming and accepting the first burst.
            fixture.queue(&[1, 2, 3]);
            assert_eq!(
                decode_latest(fixture.step_manager().expect("first accepted outputs")),
                markers(&[3], 3)
            );
            assert!(journal.items().is_empty());
            let late = assert_late_manager_snapshot(&mut fixture, Item::Manager { value: 3, sequence: 3 });
            assert_eq!(
                journal.items(),
                markers(&[1, 2, 3], 1),
                "Warming attachment has no historical bootstrap to replace future A/B/C"
            );
            let another_claim = fixture.manager.clone().register(fixture.app.world_mut());
            assert_eq!(another_claim.entity(), identity);
            assert_eq!(fixture.app.world().resource::<Production>().materializations, 1);
            fixture.queue(&[4, 5]);
            settle(&mut fixture);
            assert_eq!(journal.items(), markers(&[1, 2, 3, 4, 5], 1));
            let observed = &fixture.app.world().resource::<Ordinary>().0;
            assert_eq!(
                observed.len(),
                2,
                "ordinary observer still receives one final value per burst"
            );
            assert_eq!(decode_latest(observed[1].clone()), markers(&[5], 5));
            assert!(errors.lock().expect("error lock").is_empty());
            fixture.finish([dynamic, late, another_claim]);
        }

        struct NonReplayManager(VecSignalMaterializer<i32>);

        impl Signal for NonReplayManager {
            type Item = Update;

            fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Update> {
                ScalarSignalMaterializer::new(move |registrar| {
                    let source = registrar.materialize_signal_vec(self.0)?;
                    registrar.signal_entity(
                        source,
                        |In(diffs): In<Vec<VecDiff<i32>>>, mut sequence: Local<usize>| {
                            let value = match diffs.as_slice() {
                                [VecDiff::Replace { values }] if values.is_empty() => return None,
                                [VecDiff::Push { value }] => *value,
                                _ => panic!("expected one source batch: {diffs:?}"),
                            };
                            *sequence += 1;
                            Some(DynamicManagerUpdate::new(
                                Vec::<IndexedSlotCommand<ReplayableBroadcast<i32>>>::new(),
                                DynamicEmissions::One(Item::Manager {
                                    value,
                                    sequence: *sequence,
                                }),
                            ))
                        },
                        OutputDelivery::broadcast(),
                    )
                })
            }
        }

        #[test]
        fn ordinary_nonreplay_manager_constructor_orders_only_its_dynamic_relationship() {
            let mut app = App::new();
            app.add_plugins(JonmoPlugin::default().with_schedule::<SourceSchedule>());
            app.init_resource::<Ordinary>();
            let values = MutableVec::<i32>::builder().spawn(app.world_mut());
            // Broadcasting a scheduled non-replayable signal materializes an ordinary broadcast
            // manager without normalizing replayability, while moving only the broadcast wrapper
            // to `Last` keeps `NonReplayManager` and `values` staged in `SourceSchedule`.
            let manager = NonReplayManager(Box::new(values.signal_vec()).into_signal_vec_materializer())
                .schedule::<SourceSchedule>()
                .broadcast();
            let manager_claim = manager.clone().schedule::<Last>().register(app.world_mut());
            let source = values.signal_vec().register(app.world_mut());
            app.world_mut().run_schedule(SourceSchedule);
            app.update();
            let journal = Journal::default();
            let errors = Arc::new(Mutex::new(Vec::new()));
            let dynamic = ExternalDynamic {
                manager: manager.clone(),
                journal: journal.clone(),
                errors: errors.clone(),
                scheduled: false,
            }
            .register(app.world_mut());
            let ordinary = manager
                .clone()
                .map(|In(update): In<Update>, mut observed: ResMut<Ordinary>| {
                    observed.0.push(update);
                })
                .register(app.world_mut());
            for value in [1, 2, 3] {
                values.write(app.world_mut()).push(value);
                // Step the upstream producer per push so the non-replayable broadcast manager
                // queues all 3 updates before its single scalar drain below.
                app.world_mut().run_schedule(SourceSchedule);
            }
            #[cfg(feature = "imperative-signal-evaluation")]
            {
                let evaluated = run_signal_subgraph_once(app.world_mut(), manager_claim.entity())
                    .and_then(downcast_signal_value::<Update>)
                    .expect("ordinary manager produces the burst");
                assert_eq!(
                    decode_latest(evaluated),
                    markers(&[3], 3),
                    "scalar evaluation remains last-Some"
                );
            }
            assert!(journal.items().is_empty());
            for _ in 0..3 {
                app.update();
            }
            assert_eq!(journal.items(), markers(&[1, 2, 3], 1));
            let observed = &app.world().resource::<Ordinary>().0;
            assert_eq!(observed.len(), 1);
            assert_eq!(decode_latest(observed[0].clone()), markers(&[3], 3));
            #[cfg(feature = "imperative-signal-evaluation")]
            assert!(run_signal_subgraph_once(app.world_mut(), manager_claim.entity()).is_none());
            assert!(errors.lock().expect("error lock").is_empty());
            ordinary.release(app.world_mut());
            dynamic.release(app.world_mut());
            manager_claim.release(app.world_mut());
            source.release(app.world_mut());
        }

        #[test]
        fn retained_ordinary_terminal_discards_whole_queued_commands_before_reactivation() {
            let mut fixture = Fixture::new(Program::CaptureReady);
            let journal = Journal::default();
            let delivered = Journal::default();
            let errors = Arc::new(Mutex::new(Vec::new()));
            let terminal = ExternalDynamic {
                manager: fixture.manager.clone(),
                journal: journal.clone(),
                errors: errors.clone(),
                scheduled: false,
            }
            .map({
                let delivered = delivered.clone();
                move |In(items): In<Vec<Item>>| {
                    delivered.0.lock().expect("delivery lock").push(items);
                }
            })
            .broadcast();
            let claim = terminal.clone().register(fixture.app.world_mut());
            let identity = claim.entity();
            settle(&mut fixture);
            fixture.queue(&[1, 2]);
            assert_eq!(
                decode_latest(fixture.step_manager().expect("queue whole updates before bridge")),
                markers(&[2], 2)
            );
            let capture = fixture
                .app
                .world()
                .resource::<Production>()
                .capture
                .clone()
                .expect("command capture");
            assert!(
                capture.upgrade().is_some(),
                "earlier command stays owned by the undrained relationship"
            );
            assert!(journal.items().is_empty());
            assert!(delivered.items().is_empty());
            assert_eq!(fixture.app.world().resource::<Production>().selected_runs, 0);
            claim
                .try_release(fixture.app.world_mut())
                .expect("final ordinary terminal release");
            assert!(
                fixture.app.world().get_entity(identity.entity()).is_ok(),
                "ordinary terminal is structurally retained"
            );
            assert!(
                capture.upgrade().is_none(),
                "final release must immediately discard queued selected-signal captures"
            );
            settle(&mut fixture);
            assert!(
                journal.items().is_empty(),
                "inactive raw output cannot reduce old commands"
            );
            assert!(delivered.items().is_empty());
            assert_eq!(fixture.app.world().resource::<Production>().selected_runs, 0);

            let reactivated = terminal
                .clone()
                .try_register(fixture.app.world_mut())
                .expect("reactivate same ordinary terminal");
            assert_eq!(reactivated.entity(), identity);
            settle(&mut fixture);
            // The manager is independently retained and now Ready(2). A new relationship may
            // bootstrap that current value once, but must not reconstruct command 1 or its capture.
            assert_eq!(journal.items(), markers(&[2], 2));
            assert_eq!(delivered.items(), markers(&[2], 2));
            assert!(capture.upgrade().is_none());
            assert_eq!(fixture.app.world().resource::<Production>().selected_runs, 0);
            fixture.queue(&[3, 4]);
            settle(&mut fixture);
            let late = assert_late_manager_snapshot(&mut fixture, Item::Manager { value: 4, sequence: 4 });
            assert_eq!(journal.items(), markers(&[2, 3, 4], 2));
            assert_eq!(delivered.items(), markers(&[2, 3, 4], 2));
            assert!(capture.upgrade().is_none());
            assert!(errors.lock().expect("error lock").is_empty());
            fixture.finish([reactivated, late]);
        }

        #[test]
        fn active_direct_retainer_preserves_queued_commands_and_capture_control() {
            let mut fixture = Fixture::new(Program::CaptureReady);
            let journal = Journal::default();
            let errors = Arc::new(Mutex::new(Vec::new()));
            // Retain the intrinsically replayable output itself, not an ordinary descendant.
            let retained = ExternalDynamic {
                manager: fixture.manager.clone(),
                journal: journal.clone(),
                errors: errors.clone(),
                scheduled: false,
            }
            .schedule::<Last>()
            .broadcast();
            let claim = retained.clone().register(fixture.app.world_mut());
            let identity = claim.entity();
            settle(&mut fixture);
            fixture.queue(&[1, 2]);
            assert_eq!(
                decode_latest(fixture.step_manager().expect("queue whole updates before bridge")),
                markers(&[2], 2)
            );
            let capture = fixture
                .app
                .world()
                .resource::<Production>()
                .capture
                .clone()
                .expect("command capture");
            assert!(capture.upgrade().is_some());
            assert!(journal.items().is_empty());
            assert_eq!(fixture.app.world().resource::<Production>().selected_runs, 0);
            claim
                .try_release(fixture.app.world_mut())
                .expect("release direct claim with replay retainer still active");
            assert!(
                capture.upgrade().is_some(),
                "direct replay retention makes this a non-final deactivation"
            );
            settle(&mut fixture);
            let mut expected = vec![
                Item::Manager { value: 1, sequence: 1 },
                Item::Selected { index: 0, value: 10 },
                Item::Manager { value: 2, sequence: 2 },
            ];
            assert_eq!(
                journal.items(),
                expected,
                "active retainer must consume both whole updates normally"
            );
            assert!(
                capture.upgrade().is_some(),
                "the accepted selection owns the capture now"
            );
            let reactivated = retained
                .clone()
                .try_register(fixture.app.world_mut())
                .expect("register direct output again");
            assert_eq!(reactivated.entity(), identity);
            settle(&mut fixture);
            assert_eq!(
                journal.items(),
                expected,
                "direct re-registration cannot repeat consumed commands"
            );
            fixture.queue(&[3, 4]);
            settle(&mut fixture);
            expected.extend(markers(&[3, 4], 3));
            let late = assert_late_manager_snapshot(&mut fixture, Item::Manager { value: 4, sequence: 4 });
            assert_eq!(journal.items(), expected);
            assert!(errors.lock().expect("error lock").is_empty());
            reactivated.release(fixture.app.world_mut());
            drop(retained);
            settle(&mut fixture);
            assert!(
                capture.upgrade().is_none(),
                "dropping the last direct retainer releases the accepted selection"
            );
            fixture.finish([late]);
        }
    }

    #[derive(Resource, Default, Debug)]
    struct Collections {
        factories: Vec<i32>,
        vector: Vec<i32>,
        vector_batches: Vec<Vec<VecDiff<i32>>>,
        map: BTreeMap<i32, i32>,
        map_batches: Vec<Vec<MapDiff<i32, i32>>>,
    }

    #[test]
    fn filter_signal_preserves_multiple_future_source_batches() {
        let mut app = App::new();
        app.add_plugins(JonmoPlugin::default().with_schedule::<SourceSchedule>());
        app.init_resource::<Collections>();
        let values = MutableVec::<i32>::builder().spawn(app.world_mut());
        let branch = values
            .signal_vec()
            .map_in(core::convert::identity)
            .filter_signal(|In(value): In<i32>, mut seen: ResMut<Collections>| {
                seen.factories.push(value);
                signal::once(value % 2 == 0).cache()
            })
            .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, mut seen: ResMut<Collections>| {
                for diff in &diffs {
                    diff.clone().apply_to_vec(&mut seen.vector);
                }
                seen.vector_batches.push(diffs);
            })
            .register(app.world_mut());
        let source = values
            .signal_vec()
            .schedule::<SourceSchedule>()
            .register(app.world_mut());
        app.world_mut().run_schedule(SourceSchedule);
        app.update();
        assert!(app.world().resource::<Collections>().vector.is_empty());
        app.world_mut().resource_mut::<Collections>().vector_batches.clear();
        for value in [2, 3, 4] {
            values.write(app.world_mut()).push(value);
            app.world_mut().run_schedule(SourceSchedule);
        }
        assert!(app.world().resource::<Collections>().factories.is_empty());
        app.update();
        let seen = app.world().resource::<Collections>();
        assert_eq!(seen.factories, [2, 3, 4]);
        let actual = seen.vector.clone();
        let batches = seen.vector_batches.len();
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(app.world().resource::<Collections>().vector_batches.len(), batches);
        branch.release(app.world_mut());
        source.release(app.world_mut());
        assert_eq!(actual, [2, 4]);
    }

    #[test]
    fn map_value_signal_preserves_multiple_future_source_batches() {
        let mut app = App::new();
        app.add_plugins(JonmoPlugin::default().with_schedule::<SourceSchedule>());
        app.init_resource::<Collections>();
        let values = MutableBTreeMap::<i32, i32>::builder().spawn(app.world_mut());
        let branch = values
            .signal_map()
            .map_value(|In(value): In<i32>| value)
            .map_value_signal(|In(value): In<i32>, mut seen: ResMut<Collections>| {
                seen.factories.push(value);
                signal::once(value * 10).cache()
            })
            .for_each(|In(diffs): In<Vec<MapDiff<i32, i32>>>, mut seen: ResMut<Collections>| {
                for diff in &diffs {
                    match diff {
                        MapDiff::Replace { entries } => seen.map = entries.iter().copied().collect(),
                        MapDiff::Insert { key, value } | MapDiff::Update { key, value } => {
                            seen.map.insert(*key, *value);
                        }
                        MapDiff::Remove { key } => {
                            seen.map.remove(key);
                        }
                        MapDiff::Clear => seen.map.clear(),
                    }
                }
                seen.map_batches.push(diffs);
            })
            .register(app.world_mut());
        let source = values
            .signal_map()
            .schedule::<SourceSchedule>()
            .register(app.world_mut());
        app.world_mut().run_schedule(SourceSchedule);
        app.update();
        assert!(app.world().resource::<Collections>().map.is_empty());
        app.world_mut().resource_mut::<Collections>().map_batches.clear();
        for (key, value) in [(1, 10), (2, 20)] {
            values.write(app.world_mut()).insert(key, value);
            app.world_mut().run_schedule(SourceSchedule);
        }
        assert!(app.world().resource::<Collections>().factories.is_empty());
        app.update();
        let seen = app.world().resource::<Collections>();
        assert_eq!(seen.factories, [10, 20]);
        let actual = seen.map.clone();
        let batches = seen.map_batches.len();
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(app.world().resource::<Collections>().map_batches.len(), batches);
        branch.release(app.world_mut());
        source.release(app.world_mut());
        assert_eq!(actual, BTreeMap::from([(1, 100), (2, 200)]));
    }
}
